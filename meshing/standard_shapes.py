"""Meshing module -- CavityMode/SampleRegion -> OCC primitive dispatch
(docs/meshing_module_plan.md Section 1, step 3).

Reads Module 1/3 instances' *public dimension attributes* only -- never
calls their `.volume()`/`.contains()`. That distinction matters for Section
6.1's independence requirement: this module is allowed to consume Module
1/3's stated dimensions as input data, but must never rely on Module 1/3's
own computed geometric quantities to check itself.
"""
from __future__ import annotations

import numpy as np

from ..cavity import CavityMode, CoaxialCavity, CylindricalCavity, RectangularCavity
from ..sample import Cylinder, SampleRegion, Slab, Sphere, orthonormal_frame
from .geometry_spec import Matrix3, RigidTransform
from .transforms import apply_to_shape

DimTag = tuple[int, int]


def build_cavity_solid(cavity_mode: CavityMode) -> DimTag:
    """Returns the (dim, tag) of the OCC solid for the cavity's interior
    volume, built directly from `cavity_mode`'s own dimension attributes.
    Requires an active Gmsh model (caller's responsibility -- this function
    doesn't initialize/finalize Gmsh itself)."""
    import gmsh

    occ = gmsh.model.occ
    if isinstance(cavity_mode, RectangularCavity):
        tag = occ.addBox(0.0, 0.0, 0.0, cavity_mode.a, cavity_mode.b, cavity_mode.c)
    elif isinstance(cavity_mode, CoaxialCavity):
        # Section 6.1: CoaxialCavity is checked before CylindricalCavity --
        # both are unrelated classes here (no inheritance), but keeping the
        # more specific / less common case first avoids any future risk if
        # that ever changes.
        outer = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, cavity_mode.L, cavity_mode.b)
        inner = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, cavity_mode.L, cavity_mode.a)
        result, _ = occ.cut([(3, outer)], [(3, inner)])
        occ.synchronize()
        return result[0]
    elif isinstance(cavity_mode, CylindricalCavity):
        tag = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, cavity_mode.d, cavity_mode.a)
    else:
        raise ValueError(
            f"no OCC primitive builder registered for cavity type {type(cavity_mode).__name__!r}"
        )
    occ.synchronize()
    return (3, tag)


def build_sample_solid(region: SampleRegion) -> DimTag:
    """Returns the (dim, tag) of the OCC solid for `region`, positioned
    directly from its own center/axis/normal attributes -- already
    expressed in the cavity's frame (Module 3's own convention), so no
    extra transform is needed for the standard-shape sample path."""
    import gmsh

    occ = gmsh.model.occ
    if isinstance(region, Sphere):
        cx, cy, cz = (float(v) for v in region.center)
        tag = occ.addSphere(cx, cy, cz, region.radius)
    elif isinstance(region, Cylinder):
        axis = np.asarray(region.axis, dtype=float)
        base = np.asarray(region.center, dtype=float) - 0.5 * region.height * axis
        direction = region.height * axis
        tag = occ.addCylinder(*base.tolist(), *direction.tolist(), region.radius)
    elif isinstance(region, Slab):
        tag = _build_slab(region)
    else:
        raise ValueError(f"no OCC primitive builder registered for sample shape {type(region).__name__!r}")
    occ.synchronize()
    return (3, tag)


def _build_slab(region: Slab) -> int:
    """A box built axis-aligned and centered at the local origin, then
    rotated so its local (x, y, z) axes land on (e1, e2, normal) -- the same
    in-plane frame `orthonormal_frame` gives Module 3 itself, so the OCC
    solid's orientation matches what `Slab.contains()`/`quadrature_points()`
    actually describe -- and translated to `center`."""
    import gmsh

    occ = gmsh.model.occ
    ex, ey = region.extent
    tag = occ.addBox(-ex / 2.0, -ey / 2.0, -region.thickness / 2.0, ex, ey, region.thickness)
    occ.synchronize()

    e1, e2, n_hat = orthonormal_frame(region.normal)
    rotation: Matrix3 = _columns_to_matrix(e1, e2, n_hat)
    translation = tuple(float(v) for v in region.center)
    transform = RigidTransform(translation=translation, rotation=rotation)  # type: ignore[arg-type]
    apply_to_shape(transform, [(3, tag)])
    return tag


def _columns_to_matrix(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> Matrix3:
    R = np.stack([c0, c1, c2], axis=1)
    rows = [tuple(float(x) for x in row) for row in R]
    return (rows[0], rows[1], rows[2])  # type: ignore[return-value]
