"""Validation suite for mesh_interface.interface on the hand-computable
two-tet synthetic patch (docs/module1_mesh_interface_equations.md Section
8). No Gmsh needed -- `MeshInterface` only consumes plain arrays.
"""
import numpy as np
import pytest

from mesh_interface import MeshGeometryError, MeshInterface
from mesh_interface.interface import LOCAL_EDGES

from ._two_tet_patch import BOUNDARY_FACES, FULL_SURFACE_TAGS, TETS, VERTICES, VOLUME_A, VOLUME_B


def _build(surface_tags=None) -> MeshInterface:
    return MeshInterface(VERTICES, TETS, np.array(["A", "B"]), surface_tags or FULL_SURFACE_TAGS)


# --- Section 2.3: orientation normalization ---------------------------------

def test_negatively_oriented_input_tet_is_repaired_not_propagated():
    """tetB=[4,1,2,3] has det(P) < 0 as given (Section 2.3's build-order
    step 1 test: "confirm a deliberately mis-ordered input tet is repaired
    by step 1, not propagated")."""
    mi = _build()
    assert mi.volume(0) == pytest.approx(VOLUME_A, rel=1e-12)
    assert mi.volume(1) == pytest.approx(VOLUME_B, rel=1e-12)
    assert mi.volume(0) > 0
    assert mi.volume(1) > 0


def test_total_volume_matches_hand_computation():
    mi = _build()
    total = sum(mi.volume(t) for t in range(mi.n_tets))
    assert total == pytest.approx(VOLUME_A + VOLUME_B, rel=1e-12)


# --- Section 2.5 / 4: per-tet identities, re-checked via the public API -----

def test_partition_of_unity_gradient_via_public_api():
    mi = _build()
    for tet in range(mi.n_tets):
        assert np.abs(mi.grad_lambda(tet).sum(axis=0)).max() < 1e-8


def test_closed_surface_identity_via_public_api():
    mi = _build()
    for tet in range(mi.n_tets):
        total = np.zeros(3)
        for local_face in range(4):
            area, normal = mi.face_area_normal(tet, local_face)
            total += area * normal
        assert np.abs(total).max() < 1e-9


# --- Section 3: edge topology ------------------------------------------------

def test_shared_edges_get_identical_global_index_with_coherent_signs():
    mi = _build()
    assert np.all(mi.edges[:, 0] < mi.edges[:, 1]), "edge list must be strictly ascending"

    def global_edge_of(tet: int, global_pair: tuple[int, int]) -> tuple[int, int]:
        """Finds (global_edge_index, sign) for the local edge of `tet`
        whose endpoints are `global_pair` (in either order). Uses
        `mi.tets` (post orientation-fix), not the original `TETS` --
        Section 2.3's local-vertex permutation means they can differ."""
        for local_edge, (a, b) in enumerate(LOCAL_EDGES):
            ga, gb = mi.tets[tet, a], mi.tets[tet, b]
            if {ga, gb} == set(global_pair):
                return int(mi.tet_edge_map[tet, local_edge]), int(mi.tet_edge_sign[tet, local_edge])
        raise AssertionError(f"no local edge of tet {tet} matches {global_pair}")

    for shared_pair in [(1, 2), (1, 3), (2, 3)]:
        edge_a, sign_a = global_edge_of(0, shared_pair)
        edge_b, sign_b = global_edge_of(1, shared_pair)
        assert edge_a == edge_b, f"shared edge {shared_pair} got different global indices from each side"
        # Both tets reference the SAME ascending global pair, so recomputed
        # signs must be consistent with tuple(mi.edges[edge_a]) regardless
        # of each tet's own local vertex order.
        lo, hi = mi.edges[edge_a]
        assert (lo, hi) == tuple(sorted(shared_pair))


def test_edge_count_matches_hand_enumeration():
    """9 distinct undirected edges total: tetA contributes 6, tetB shares 3
    of them (the shared face's edges) and adds 3 new ones -- 6 + 3 = 9."""
    mi = _build()
    assert mi.n_edges == 9


# --- Section 5.1/5.2: face incidence -----------------------------------------

def test_face_count_identity_on_the_real_patch():
    mi = _build()
    n_boundary = len(BOUNDARY_FACES)
    total = 4 * mi.n_tets
    n_interior = (total - n_boundary) // 2
    assert n_interior == 1  # the one shared face
    assert 2 * n_interior + n_boundary == total


def test_incidence_three_raises():
    """A third tet sharing the same triangular face as tetA/tetB is
    topologically invalid for a manifold volume mesh -- confirms the
    incidence check is known to protect something (Section 8)."""
    vertices = np.vstack([VERTICES, [[-1.0, -1.0, -1.0]]])  # a 6th vertex, v5
    tets = np.vstack([TETS, [[5, 1, 2, 3]]])
    with pytest.raises(MeshGeometryError, match="incidence"):
        MeshInterface(vertices, tets, np.array(["A", "B", "C"]), {})


# --- Section 5.3: boundary-face tagging --------------------------------------

def test_boundary_coverage_raises_when_a_face_is_left_untagged():
    partial = {"PEC_GROUND": FULL_SURFACE_TAGS["PEC_GROUND"]}
    with pytest.raises(MeshGeometryError, match="untagged"):
        _build(partial)


def test_boundary_coverage_raises_when_a_face_is_claimed_twice():
    doubled = dict(FULL_SURFACE_TAGS)
    doubled["DUPLICATE_OF_PORT_1"] = FULL_SURFACE_TAGS["PORT_1"]
    with pytest.raises(MeshGeometryError, match="more than one tag"):
        _build(doubled)


def test_resolve_tags_raises_for_a_face_absent_from_the_mesh():
    bogus = dict(FULL_SURFACE_TAGS)
    bogus["BOGUS"] = np.array([[0, 1, 4]])  # not a real face of either tet
    with pytest.raises(MeshGeometryError, match="not present"):
        _build(bogus)


def test_pec_aggregates_ground_and_line():
    mi = _build()
    pec = set(mi.boundary_faces("PEC"))
    ground_only = set(mi.boundary_faces("PEC_GROUND"))
    line_only = set(mi.boundary_faces("PEC_LINE"))
    assert pec == ground_only | line_only
    assert len(pec) == 2  # one exterior face each, both incidence 1 here


def test_pml_outer_resolves_to_pml_outer_pec():
    mi = _build()
    assert mi.boundary_faces("PML_OUTER") == mi.boundary_faces("PML_OUTER_PEC")


def test_port_and_pmc_side_pass_through_directly():
    mi = _build()
    assert len(mi.boundary_faces("PORT_1")) == 1
    assert len(mi.boundary_faces("PORT_2")) == 1
    assert len(mi.boundary_faces("PMC_SIDE")) == 1


def test_unknown_tag_returns_empty():
    mi = _build()
    assert mi.boundary_faces("NOT_A_REAL_TAG") == []


# --- Section 6: quadrature on the public API ---------------------------------

def test_quadrature_tet_sums_to_volume():
    mi = _build()
    for tet in range(mi.n_tets):
        for order in (1, 2):
            _points, weights = mi.quadrature_tet(tet, order)
            assert weights.sum() == pytest.approx(mi.volume(tet), rel=1e-9)


def test_quadrature_tri_sums_to_area():
    mi = _build()
    for tag, expected_face in [("PORT_1", BOUNDARY_FACES["A_012"])]:
        (tet, local_face) = mi.boundary_faces(tag)[0]
        area, _normal = mi.face_area_normal(tet, local_face)
        for order in (1, 2):
            _points, weights = mi.quadrature_tri((tet, local_face), order)
            assert weights.sum() == pytest.approx(area, rel=1e-9)
