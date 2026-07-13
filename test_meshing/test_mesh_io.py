"""Validation suite for mesh_io.py (docs/meshing_module_plan.md Section
6.3, "mesh_io.py")."""
import gmsh
import numpy as np
import pytest

from cavity_perturbation.meshing.mesh_generation import generate_mesh
from cavity_perturbation.meshing.mesh_io import assemble_mesh_result, mesh_stats, read_mesh, write_mesh
from cavity_perturbation.meshing.tagging import fragment_and_tag

A = B = C = 0.03


def _tetra_volume(coords: np.ndarray) -> float:
    a, b, c, d = coords
    return float(abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0)


def _build_and_mesh_boxy(h: float = 0.01):
    """A box-shaped sample (no curved surfaces) so the mesh's total volume
    matches the closed-form cavity volume *exactly*, isolating a round-trip
    bug from ordinary mesh-discretization error against a curved boundary."""
    occ = gmsh.model.occ
    cav = occ.addBox(0, 0, 0, A, B, C)
    sample = occ.addBox(0.01, 0.01, 0.01, 0.005, 0.005, 0.005)
    occ.synchronize()
    tagged = fragment_and_tag((3, cav), (3, sample))
    quality = generate_mesh(h)
    return tagged, quality


def test_round_trip_preserves_vertex_and_element_counts_and_volume(gmsh_model, tmp_path):
    _tagged, quality = _build_and_mesh_boxy()
    node_tags, _coords, _params = gmsh.model.mesh.getNodes()
    n_vertices_original = len(node_tags)

    path = tmp_path / "roundtrip.msh"
    write_mesh(path)
    mesh = read_mesh(path)

    n_vertices_read = len(mesh.points)
    tetra_blocks = [c for c in mesh.cells if c.type == "tetra"]
    n_tetra_read = sum(len(c.data) for c in tetra_blocks)
    volume_read = sum(_tetra_volume(mesh.points[cell]) for c in tetra_blocks for cell in c.data)

    assert n_vertices_read == n_vertices_original
    assert n_tetra_read == quality.n_elements
    assert volume_read == pytest.approx(A * B * C, rel=1e-9)


def test_mesh_stats_matches_generation_quality_plus_vertex_count(gmsh_model):
    _tagged, quality = _build_and_mesh_boxy()
    node_tags, _coords, _params = gmsh.model.mesh.getNodes()

    stats = mesh_stats(quality)
    assert stats.n_vertices == len(node_tags)
    assert stats.n_elements == quality.n_elements
    assert stats.min_element_quality == quality.min_element_quality
    assert stats.max_aspect_ratio == quality.max_aspect_ratio


def test_assemble_mesh_result(gmsh_model, tmp_path):
    tagged, quality = _build_and_mesh_boxy()
    path = tmp_path / "assembled.msh"
    write_mesh(path)
    mesh = read_mesh(path)

    cavity_volume = A * B * C
    sample_volume = 0.005**3

    result = assemble_mesh_result(
        mesh=mesh,
        sample_physical_tag=tagged.sample_physical_tag,
        background_physical_tag=tagged.background_physical_tag,
        boundary_physical_tag=tagged.boundary_physical_tag,
        cavity_volume=cavity_volume,
        sample_volume=sample_volume,
        quality=quality,
    )

    assert result.sample_physical_tag == tagged.sample_physical_tag
    assert result.background_physical_tag == tagged.background_physical_tag
    assert result.boundary_physical_tag == tagged.boundary_physical_tag
    assert result.cavity_volume == cavity_volume
    assert result.sample_volume == sample_volume
    assert result.mesh_stats.n_elements == quality.n_elements
    assert result.mesh is mesh
