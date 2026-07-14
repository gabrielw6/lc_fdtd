"""Validation suite for tagging.py (docs/meshing_module_plan.md Section
6.3, "tagging.py").

A box-shaped sample (not a real standard shape -- doesn't need to be for
this test) is used deliberately: no curved surfaces means the
piecewise-linear tet mesh reconstructs the true (closed-form) volume
*exactly*, isolating a tagging bug from ordinary mesh-discretization error
on a curved boundary.
"""
import gmsh
import numpy as np
import pytest

from meshing.tagging import fragment_and_tag

A, B, C = 0.03, 0.03, 0.03


def _tetra_volume(coords: np.ndarray) -> float:
    a, b, c, d = coords
    return float(abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0)


def _build_cavity_and_boxy_sample():
    occ = gmsh.model.occ
    cav = occ.addBox(0, 0, 0, A, B, C)
    sample = occ.addBox(0.01, 0.01, 0.01, 0.005, 0.005, 0.005)
    occ.synchronize()
    return (3, cav), (3, sample)


def test_volume_partition_matches_total_cavity_volume(gmsh_model):
    cav_dt, sample_dt = _build_cavity_and_boxy_sample()
    tagged = fragment_and_tag(cav_dt, sample_dt)

    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.01)
    gmsh.model.mesh.generate(3)

    total_mesh_volume = 0.0
    seen_entities = set()
    for dim, tag in tagged.sample_dim_tags + tagged.background_dim_tags:
        seen_entities.add((dim, tag))
        elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim, tag)
        for etype, ent_nodes in zip(elem_types, elem_node_tags):
            _name, _edim, _order, n_nodes, _local, _n_primary = gmsh.model.mesh.getElementProperties(etype)
            assert n_nodes == 4, f"expected first-order tetrahedra only, got element type {_name!r}"
            nodes = np.asarray(ent_nodes, dtype=int).reshape(-1, 4)
            for tet_nodes in nodes:
                coords = np.array([gmsh.model.mesh.getNode(int(n))[0] for n in tet_nodes])
                total_mesh_volume += _tetra_volume(coords)

    assert total_mesh_volume == pytest.approx(A * B * C, rel=1e-9)

    # Every dim=3 entity in the model is accounted for by exactly one of
    # the two physical groups -- a complete partition, not just "enough."
    all_3d_entities = set(gmsh.model.occ.getEntities(3))
    assert all_3d_entities == seen_entities


def test_boundary_area_matches_cavity_surface_area_and_excludes_interface(gmsh_model):
    cav_dt, sample_dt = _build_cavity_and_boxy_sample()
    tagged = fragment_and_tag(cav_dt, sample_dt)

    occ = gmsh.model.occ
    boundary_area = sum(occ.getMass(dim, tag) for dim, tag in tagged.boundary_dim_tags)
    expected_area = 2.0 * (A * B + B * C + C * A)
    assert boundary_area == pytest.approx(expected_area, rel=1e-9)

    # The sample's own 6 faces (the internal sample/background interface)
    # must NOT show up in the "boundary" (outer cavity wall) group.
    sample_own_faces = set(gmsh.model.getBoundary(list(tagged.sample_dim_tags), combined=False, oriented=False))
    assert sample_own_faces.isdisjoint(set(tagged.boundary_dim_tags))


def test_physical_group_tags_are_distinct(gmsh_model):
    cav_dt, sample_dt = _build_cavity_and_boxy_sample()
    tagged = fragment_and_tag(cav_dt, sample_dt)
    tags = {tagged.sample_physical_tag, tagged.background_physical_tag, tagged.boundary_physical_tag}
    assert len(tags) == 3
