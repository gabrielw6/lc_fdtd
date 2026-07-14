"""Meshing module -- export/round-trip via meshio; MeshResult/MeshStats
assembly (docs/meshing_module_plan.md Section 1, step 9; Section 2.3).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import meshio

from .mesh_generation import MeshQuality


@dataclass(frozen=True)
class MeshStats:
    n_elements: int
    n_vertices: int
    min_element_quality: float  # Gmsh's own quality metric (minSICN)
    max_aspect_ratio: float


@dataclass(frozen=True)
class MeshResult:
    mesh: meshio.Mesh
    sample_physical_tag: int
    background_physical_tag: int
    boundary_physical_tag: int
    cavity_volume: float  # from OCC mass properties (Section 2.3)
    sample_volume: float  # from OCC mass properties (Section 2.3)
    mesh_stats: MeshStats


def mesh_stats(quality: MeshQuality) -> MeshStats:
    """Assembles `MeshStats` from mesh_generation.py's `MeshQuality` plus a
    vertex count queried directly from the active Gmsh model."""
    import gmsh

    node_tags, _coords, _params = gmsh.model.mesh.getNodes()
    return MeshStats(
        n_elements=quality.n_elements,
        n_vertices=len(node_tags),
        min_element_quality=quality.min_element_quality,
        max_aspect_ratio=quality.max_aspect_ratio,
    )


def write_mesh(path: Path | str) -> None:
    """Writes the active Gmsh model's current mesh, with physical groups,
    to `path` (format inferred from extension, e.g. `.msh`) -- Gmsh's own
    writer, not meshio's (meshio doesn't generate meshes, only
    reads/writes/converts already-generated ones)."""
    import gmsh

    gmsh.write(str(path))


def read_mesh(path: Path | str) -> meshio.Mesh:
    """Reads a mesh file back via meshio -- the round-trip half of Section
    6.3's "round-trip a mesh through export and re-import via meshio"."""
    return meshio.read(str(path))


def assemble_mesh_result(
    mesh: meshio.Mesh,
    sample_physical_tag: int,
    background_physical_tag: int,
    boundary_physical_tag: int,
    cavity_volume: float,
    sample_volume: float,
    quality: MeshQuality,
) -> MeshResult:
    """Combines a read-back mesh with the tagging/generation info a caller
    (pipeline.py) already gathered from tagging.py/interference.py/
    mesh_generation.py into the public `MeshResult`. Only takes plain data
    (ints, floats, a meshio.Mesh, and mesh_generation's own `MeshQuality`)
    -- never tagging.py's or interference.py's own types -- so this module
    doesn't need to import them."""
    return MeshResult(
        mesh=mesh,
        sample_physical_tag=sample_physical_tag,
        background_physical_tag=background_physical_tag,
        boundary_physical_tag=boundary_physical_tag,
        cavity_volume=cavity_volume,
        sample_volume=sample_volume,
        mesh_stats=mesh_stats(quality),
    )
