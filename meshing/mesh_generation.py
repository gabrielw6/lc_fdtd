"""Meshing module -- Gmsh mesh generation + quality query
(docs/meshing_module_plan.md Section 1, step 8; Section 4's uniform
mesh-size field; Section 6.2's invariant-property checks).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_QUALITY_TYPE = "minSICN"  # signed inverse condition number: 1=perfect regular tet, <=0=degenerate/inverted


class DegenerateMeshError(Exception):
    """Raised when mesh generation produces a zero-element or otherwise
    degenerate (zero/negative-quality) result."""


@dataclass(frozen=True)
class MeshQuality:
    n_elements: int
    min_element_quality: float
    max_aspect_ratio: float  # circumradius/inradius, 1.0 for a perfectly regular tet


def generate_mesh(characteristic_length: float) -> MeshQuality:
    """Applies `characteristic_length` as Gmsh's mesh-size field uniformly
    (Section 4 -- a first implementation; finer control near the sample
    interface is a documented later refinement, not required here) and
    generates a first-order tetrahedral mesh over whatever tagged geometry
    is already in the active Gmsh model (tagging.py). Raises
    `DegenerateMeshError` rather than silently returning a garbage mesh --
    zero elements, or any element with non-positive quality."""
    if characteristic_length <= 0:
        raise ValueError(f"characteristic_length must be > 0, got {characteristic_length!r}")

    import gmsh

    # Section 4 asks for h_char applied "uniformly, for a first
    # implementation" -- Gmsh's curvature- and boundary-extension-based
    # auto-sizing (both on by default) would otherwise silently override
    # MeshSizeMax near curved features (e.g. a spherical sample) *below*
    # the requested resolution, which is exactly the "finer control near
    # the sample interface" the doc defers as a later refinement, not
    # something that should happen implicitly on a "first implementation."
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", characteristic_length)
    gmsh.option.setNumber("Mesh.MeshSizeMin", characteristic_length / 5.0)
    # generate(3) on an already-meshed model is a no-op (confirmed
    # empirically) -- clear first so this function is idempotent/re-callable
    # with a new resolution on the same geometry, not just the first call.
    gmsh.model.mesh.clear()
    gmsh.model.mesh.generate(3)

    _elem_types, elem_tags_by_type, _elem_nodes = gmsh.model.mesh.getElements(3)
    all_tags = (
        np.concatenate([np.asarray(t) for t in elem_tags_by_type]) if elem_tags_by_type else np.array([])
    )
    if all_tags.size == 0:
        raise DegenerateMeshError("mesh generation produced zero 3D elements")

    tags_int = all_tags.astype(np.int64)
    quality = gmsh.model.mesh.getElementQualities(tags_int, _QUALITY_TYPE)
    min_quality = float(np.min(quality))
    if min_quality <= 0.0:
        raise DegenerateMeshError(
            f"mesh contains a degenerate/inverted element (min {_QUALITY_TYPE}={min_quality!r})"
        )

    inner = gmsh.model.mesh.getElementQualities(tags_int, "innerRadius")
    outer = gmsh.model.mesh.getElementQualities(tags_int, "outerRadius")
    max_aspect_ratio = float(np.max(outer / inner))

    return MeshQuality(
        n_elements=int(all_tags.size),
        min_element_quality=min_quality,
        max_aspect_ratio=max_aspect_ratio,
    )
