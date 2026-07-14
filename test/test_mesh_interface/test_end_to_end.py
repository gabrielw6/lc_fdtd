"""End-to-end validation of MeshInterface against a real Module 0
(`geometry_builder`)-produced mesh (docs/module1_mesh_interface_equations.md
Section 8: "Total volume... the single strongest end-to-end geometry
check"). `GeometryBuilder.build()` manages its own Gmsh lifecycle, so this
needs Gmsh; `MeshInterface` itself does not.
"""
import numpy as np
import pytest

pytest.importorskip("gmsh")

from geometry_builder import GeometryBuilder, GeometryParams
from mesh_interface import MeshInterface

_PARAMS = GeometryParams(
    w=0.002,
    L=0.020,
    L_lc=0.008,
    W_lc=0.004,
    h_sub=0.002,
    W_sub=0.010,
    eps_r_substrate=3.0,
    h_air=0.006,
    h_pml=0.002,
    reference_frequency=25e9,
    target_elements_per_wavelength=8,
)


@pytest.fixture(scope="module")
def mesh_iface() -> MeshInterface:
    mesh_handle, _ = GeometryBuilder().build(_PARAMS)
    return MeshInterface.from_mesh_handle(mesh_handle)


def test_total_volume_matches_bounding_box(mesh_iface):
    total = sum(mesh_iface.volume(t) for t in range(mesh_iface.n_tets))
    expected = _PARAMS.L * _PARAMS.W_sub * (_PARAMS.h_sub + _PARAMS.h_air + _PARAMS.h_pml)
    assert total == pytest.approx(expected, rel=1e-9)


def test_pec_line_faces_are_doubled_because_they_are_interior(mesh_iface):
    """Section 4.3 of the Module 0 doc: PEC_LINE is an embedded, internal
    partition (incidence 2), unlike the other surface tags -- so
    `boundary_faces('PEC')` should carry one (tet, local_face) entry per
    side for every PEC_LINE triangle, on top of PEC_GROUND's single-sided
    entries."""
    ground = mesh_iface.boundary_faces("PEC_GROUND")
    line = mesh_iface.boundary_faces("PEC_LINE")
    pec = mesh_iface.boundary_faces("PEC")
    assert len(line) % 2 == 0, "PEC_LINE is interior -- every triangle should surface twice, once per side"
    assert len(pec) == len(ground) + len(line)


def test_every_documented_tag_resolves_to_faces(mesh_iface):
    for tag in ("PEC", "PORT_1", "PORT_2", "PML_OUTER", "PMC_SIDE"):
        assert len(mesh_iface.boundary_faces(tag)) > 0, f"{tag} resolved to zero faces"


def test_quadrature_tet_sums_to_volume_for_every_tet(mesh_iface):
    for order in (1, 2):
        errors = 0
        for tet in range(mesh_iface.n_tets):
            _points, weights, _bary = mesh_iface.quadrature_tet(tet, order)
            if not np.isclose(weights.sum(), mesh_iface.volume(tet), rtol=1e-9):
                errors += 1
        assert errors == 0


def test_quadrature_tri_sums_to_area_for_every_port_face(mesh_iface):
    for order in (1, 2):
        for face in mesh_iface.boundary_faces("PORT_1"):
            area, _normal = mesh_iface.face_area_normal(*face)
            _points, weights = mesh_iface.quadrature_tri(face, order)
            assert weights.sum() == pytest.approx(area, rel=1e-9)


def test_port_faces_lie_on_their_expected_plane(mesh_iface):
    """Cross-check against Module 0's own geometry: PORT_1 faces (and
    their quadrature points) must lie on x=0."""
    for face in mesh_iface.boundary_faces("PORT_1"):
        points, _weights = mesh_iface.quadrature_tri(face, 2)
        assert points[:, 0] == pytest.approx(0.0, abs=1e-9)


def test_from_mesh_handle_matches_direct_construction(mesh_iface):
    mesh_handle, _ = GeometryBuilder().build(_PARAMS)
    direct = MeshInterface(mesh_handle.vertices, mesh_handle.tets, mesh_handle.volume_tags, mesh_handle.surface_tags)
    assert direct.n_tets == mesh_iface.n_tets
    assert direct.n_edges == mesh_iface.n_edges


def test_tet_volume_tag_reproduces_input_array(mesh_iface):
    mesh_handle, _ = GeometryBuilder().build(_PARAMS)
    for tet in range(0, mesh_iface.n_tets, 137):  # a spot-check stride, not every tet
        assert mesh_iface.tet_volume_tag(tet) == str(mesh_handle.volume_tags[tet])


def test_pec_edge_dofs_matches_independent_enumeration(mesh_iface):
    """Module 1 doc Section 8's cross-check, on the real mesh: re-derive
    the PEC edge set from each PEC face's own global vertex triple, looked
    up in `mesh_iface.edges` by value, independent of `tet_edge_map`."""
    edge_index_by_pair = {tuple(e): i for i, e in enumerate(mesh_iface.edges.tolist())}

    expected: set[int] = set()
    for tet, local_face in mesh_iface.boundary_faces("PEC"):
        from mesh_interface.interface import LOCAL_FACES

        verts = [int(mesh_iface.tets[tet, v]) for v in LOCAL_FACES[local_face]]
        for i in range(3):
            for j in range(i + 1, 3):
                pair = tuple(sorted((verts[i], verts[j])))
                expected.add(edge_index_by_pair[pair])

    assert mesh_iface.pec_edge_dofs() == expected
    assert len(expected) > 0
