"""Validation suite for the port-aperture decoupling's Module 4 side
(docs/module4_ports_equations.md Section 3.7's "second mitigation" note):
`extract_cross_section` must pick up the smaller aperture Module 0 tags,
and the aperture's own side/top walls must come out PEC via the PORT_CAP
cap-face-sharing rule -- with *zero* changes to `cross_section.py` itself.
No port-mode solving here (that's numerically sensitive to mesh
resolution, orthogonal to what this file checks); a coarse/fast mesh
(sized for a low reference frequency) is fine for pure geometry/tagging
assertions.
"""
import numpy as np
import pytest

pytest.importorskip("gmsh")

from geometry_builder import GeometryBuilder, GeometryParams
from mesh_interface import MeshInterface
from ports.basis2d import TRI_LOCAL_EDGES
from ports.cross_section import extract_cross_section

_W_SUB = 0.010
_H_SUB = 0.002
_W_PORT = 0.008
_H_PORT = 0.006


@pytest.fixture(scope="module")
def port1():
    params = GeometryParams(
        w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=_H_SUB, W_sub=_W_SUB,
        eps_r_substrate=3.0, reference_frequency=6e9, target_elements_per_wavelength=6,
        W_port=_W_PORT, H_port=_H_PORT,
    )
    mesh_handle, _material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    return extract_cross_section(mesh, "PORT_1")


def _edge_to_verts(cs):
    edge_to_verts = {}
    for t in range(cs.n_triangles):
        for slot, (p, q) in enumerate(TRI_LOCAL_EDGES):
            edge_to_verts[int(cs.tri_edges[t, slot])] = (int(cs.triangles[t, p]), int(cs.triangles[t, q]))
    return edge_to_verts


def test_cross_section_extent_matches_the_aperture_not_the_full_domain(port1):
    y0_expected = (_W_SUB - _W_PORT) / 2.0
    y1_expected = (_W_SUB + _W_PORT) / 2.0
    assert float(port1.yz[:, 0].min()) == pytest.approx(y0_expected, abs=1e-9)
    assert float(port1.yz[:, 0].max()) == pytest.approx(y1_expected, abs=1e-9)
    assert float(port1.yz[:, 1].min()) == pytest.approx(0.0, abs=1e-9)
    assert float(port1.yz[:, 1].max()) == pytest.approx(_H_PORT, abs=1e-9)
    # strictly smaller than the full substrate+air cross-section it would
    # have been without W_port/H_port.
    assert _W_PORT < _W_SUB
    assert _H_PORT < _H_SUB + 3.0 * _H_SUB  # default h_air = 3*h_sub


def test_aperture_side_and_top_walls_are_pec_via_the_cap_sharing_rule(port1):
    """The whole point of the port-aperture decoupling (Module 0 Section
    1.4): the PORT_CAP-tagged region outside the aperture makes the
    aperture's own side (y=y0_port, y=y1_port) and top (z=H_port) walls
    PEC automatically, via Module 1's existing PEC-face-sharing rule --
    with no new Module 4 code. Ground (z=0) edges are PEC for an entirely
    separate reason (PEC_GROUND, present regardless of any aperture)."""
    edge_to_verts = _edge_to_verts(port1)
    y0, y1 = float(port1.yz[:, 0].min()), float(port1.yz[:, 0].max())
    tol = 1e-9

    side_pec = top_pec = ground_pec = 0
    for e in range(port1.n_edges):
        if not port1.pec_edges[e]:
            continue
        v0, v1 = edge_to_verts[e]
        y0e, z0e = port1.yz[v0]
        y1e, z1e = port1.yz[v1]
        if abs(y0e - y0) < tol and abs(y1e - y0) < tol:
            side_pec += 1
        elif abs(y0e - y1) < tol and abs(y1e - y1) < tol:
            side_pec += 1
        elif abs(z0e - _H_PORT) < tol and abs(z1e - _H_PORT) < tol:
            top_pec += 1
        elif abs(z0e - 0.0) < tol and abs(z1e - 0.0) < tol:
            ground_pec += 1

    assert side_pec > 0, "aperture side walls must be PEC (PORT_CAP folded into the PEC aggregate)"
    assert top_pec > 0, "aperture top wall must be PEC (PORT_CAP folded into the PEC aggregate)"
    assert ground_pec > 0, "ground-plane edges must still be PEC regardless of the aperture"


def test_total_pec_edge_count_matches_hand_verified_geometry(port1):
    """Hand-verified against this exact fixture (see the port-aperture
    review's own diagnostic run): 15 of 61 edges are PEC, split 6
    side-wall + 4 top-wall + 4 ground (+1 corner double-classified as
    both a side and ground edge in the naive per-edge scan, not asserted
    here -- the important structural facts are covered by the previous
    test)."""
    assert port1.n_edges == 61
    assert int(port1.pec_edges.sum()) == 15
