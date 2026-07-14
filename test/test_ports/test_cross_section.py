"""Validation suite for ports.cross_section (docs/module4_ports_equations.md
Section 2). Uses the real geometry_builder -> mesh_interface pipeline
(mirrors test_fem/test_assembly.py's `real_mesh_and_materials` fixture)
since Section 2.3's PEC-edge subtlety (ground plane vs. embedded trace) is
exactly the kind of thing that needs Module 0's real trace-footprint
construction to exercise meaningfully -- a hand-built toy tet mesh would
not reproduce the embedded-sheet topology Module 0 actually produces.
"""
import numpy as np
import pytest

from ports.cross_section import CrossSectionError, extract_cross_section

_PARAMS_KWARGS = dict(
    w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
    eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
    reference_frequency=25e9, target_elements_per_wavelength=8,
)


@pytest.fixture(scope="module")
def mesh():
    pytest.importorskip("gmsh")
    from geometry_builder import GeometryBuilder, GeometryParams
    from mesh_interface import MeshInterface

    params = GeometryParams(**_PARAMS_KWARGS)
    mesh_handle, _material_stub = GeometryBuilder().build(params)
    return MeshInterface.from_mesh_handle(mesh_handle)


@pytest.fixture(scope="module")
def port1(mesh):
    return extract_cross_section(mesh, "PORT_1")


def test_port_plane_is_x_equals_zero(port1):
    assert port1.x0 == pytest.approx(0.0, abs=1e-9)


def test_triangle_areas_all_positive(port1):
    assert np.all(port1.area > 0.0)


def test_total_area_matches_substrate_plus_air_cross_section(port1):
    expected = _PARAMS_KWARGS["W_sub"] * (_PARAMS_KWARGS["h_sub"] + _PARAMS_KWARGS["h_air"])
    assert float(port1.area.sum()) == pytest.approx(expected, rel=1e-9)


def test_only_substrate_and_air_tags_present(port1):
    """Section 2.2: ports sit in isotropic feed sections only -- LC and PML
    tags must never appear on a port cross-section triangle."""
    assert set(np.unique(port1.tri_tag)) == {"SUBSTRATE", "AIR"}


def test_ground_plane_pec_edges_span_full_width(port1):
    """Section 2.3: the outer ground-plane edge is PEC and spans the whole
    cross-section width (y in [0, W_sub]) at z=0."""
    pec_yz = port1.yz[port1.pec_vertices]
    at_ground = pec_yz[np.isclose(pec_yz[:, 1], 0.0, atol=1e-9)]
    assert len(at_ground) >= 2
    assert float(at_ground[:, 0].min()) == pytest.approx(0.0, abs=1e-9)
    assert float(at_ground[:, 0].max()) == pytest.approx(_PARAMS_KWARGS["W_sub"], rel=1e-6)


def test_trace_pec_edges_are_isolated_to_the_trace_footprint(port1):
    """Section 2.3's central subtlety: the embedded zero-thickness trace
    is ALSO PEC, at z=h_sub, but its y-extent is a proper subset of
    [0, W_sub] -- confirming "no node duplication needed for the
    zero-thickness trace" is actually working, not silently absent."""
    h_sub, w, W_sub = _PARAMS_KWARGS["h_sub"], _PARAMS_KWARGS["w"], _PARAMS_KWARGS["W_sub"]
    pec_yz = port1.yz[port1.pec_vertices]
    at_trace = pec_yz[np.isclose(pec_yz[:, 1], h_sub, atol=1e-9)]
    assert len(at_trace) >= 2
    y0_expected, y1_expected = (W_sub - w) / 2.0, (W_sub + w) / 2.0
    assert float(at_trace[:, 0].min()) == pytest.approx(y0_expected, rel=1e-6)
    assert float(at_trace[:, 0].max()) == pytest.approx(y1_expected, rel=1e-6)
    # Genuinely internal: does not touch the outer y=0/W_sub boundary.
    assert float(at_trace[:, 0].min()) > 1e-9
    assert float(at_trace[:, 0].max()) < W_sub - 1e-9


def test_pec_vertices_are_exactly_the_endpoints_of_pec_edges(port1):
    expected = np.zeros(port1.n_vertices, dtype=bool)
    for t in range(port1.n_triangles):
        for slot, (p, q) in enumerate([(0, 1), (0, 2), (1, 2)]):
            if port1.pec_edges[port1.tri_edges[t, slot]]:
                expected[port1.triangles[t, p]] = True
                expected[port1.triangles[t, q]] = True
    assert np.array_equal(expected, port1.pec_vertices)


def test_pmc_side_edges_are_not_pec(port1):
    """Section 2.3: the lateral y=0/y=W_sub edges are PMC_SIDE (natural),
    not PEC -- only the ground-plane (z=0) and trace edges should be PEC."""
    pec_yz_edges_midpoints = []
    for t in range(port1.n_triangles):
        for slot, (p, q) in enumerate([(0, 1), (0, 2), (1, 2)]):
            if port1.pec_edges[port1.tri_edges[t, slot]]:
                mid = 0.5 * (port1.yz[port1.triangles[t, p]] + port1.yz[port1.triangles[t, q]])
                pec_yz_edges_midpoints.append(mid)
    mids = np.array(pec_yz_edges_midpoints)
    # every PEC edge midpoint should be at z=0 (ground) or z=h_sub (trace),
    # never purely on the y=0/W_sub lateral walls at z>0.
    on_ground = np.isclose(mids[:, 1], 0.0, atol=1e-9)
    on_trace = np.isclose(mids[:, 1], _PARAMS_KWARGS["h_sub"], atol=1e-9)
    assert np.all(on_ground | on_trace)


def test_no_boundary_faces_raises(mesh):
    with pytest.raises(CrossSectionError):
        extract_cross_section(mesh, "NOT_A_REAL_TAG")
