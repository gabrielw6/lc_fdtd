"""Validation suite for geometry_builder.builder (docs/module0_geometry_builder_equations.md
Section 6, "Consistency checks"). `GeometryBuilder.build()` manages its own
Gmsh init/finalize lifecycle (mirroring meshing.pipeline.build_mesh), so
these tests call it directly rather than using a `gmsh_model` fixture.
"""
import numpy as np
import pytest

pytest.importorskip("gmsh")

from geometry_builder.builder import GeometryBuilder
from geometry_builder.params import GeometryParams
from geometry_builder.tags import (
    AIR,
    LC,
    PEC_GROUND,
    PEC_LINE,
    PML_OUTER_PEC,
    PML_TOP,
    PMC_SIDE,
    PORT_1,
    PORT_2,
    SUBSTRATE,
    SURFACE_TAGS,
    VOLUME_TAGS,
)

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
def built():
    return GeometryBuilder().build(_PARAMS)


def _tet_volume(coords: np.ndarray) -> float:
    a, b, c, d = coords
    return float(abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0)


def test_every_volume_tag_present_and_nonempty(built):
    mesh_handle, _ = built
    present = set(np.unique(mesh_handle.volume_tags))
    assert present == set(VOLUME_TAGS)


def test_every_surface_tag_present_and_nonempty(built):
    mesh_handle, _ = built
    for name in SURFACE_TAGS:
        assert mesh_handle.surface_tags[name].shape[0] > 0, f"{name} has no triangles"


def test_lc_volume_matches_closed_form(built):
    """Section 6's bounding-volume check, re-derived independently from the
    mesh itself (not OCC's own mass query, which build() already checked
    internally) -- sums tet volumes by tag and compares to the closed form."""
    mesh_handle, _ = built
    mask = mesh_handle.volume_tags == LC
    tets = mesh_handle.tets[mask]
    volume = sum(_tet_volume(mesh_handle.vertices[t]) for t in tets)
    expected = _PARAMS.L_lc * _PARAMS.W_lc * _PARAMS.h_sub
    assert volume == pytest.approx(expected, rel=1e-6)


def test_substrate_volume_matches_closed_form(built):
    mesh_handle, _ = built
    mask = mesh_handle.volume_tags == SUBSTRATE
    tets = mesh_handle.tets[mask]
    volume = sum(_tet_volume(mesh_handle.vertices[t]) for t in tets)
    v_lc = _PARAMS.L_lc * _PARAMS.W_lc * _PARAMS.h_sub
    expected = _PARAMS.L * _PARAMS.W_sub * _PARAMS.h_sub - v_lc
    assert volume == pytest.approx(expected, rel=1e-6)


def test_total_volume_matches_bounding_box(built):
    """Module 1 doc Section 8's "single strongest end-to-end geometry
    check": sum of all tet volumes equals the analytic bounding volume of
    the whole model."""
    mesh_handle, _ = built
    volume = sum(_tet_volume(mesh_handle.vertices[t]) for t in mesh_handle.tets)
    expected = _PARAMS.L * _PARAMS.W_sub * (_PARAMS.h_sub + _PARAMS.h_air + _PARAMS.h_pml)
    assert volume == pytest.approx(expected, rel=1e-6)


def test_pec_ground_lies_on_z_equals_zero(built):
    mesh_handle, _ = built
    triangles = mesh_handle.surface_tags[PEC_GROUND]
    z = mesh_handle.vertices[triangles.ravel(), 2]
    assert z == pytest.approx(0.0, abs=1e-12)


def test_pec_line_lies_within_trace_footprint_on_the_interface_plane(built):
    mesh_handle, _ = built
    triangles = mesh_handle.surface_tags[PEC_LINE]
    pts = mesh_handle.vertices[triangles.ravel()]
    assert pts[:, 2] == pytest.approx(_PARAMS.h_sub, abs=1e-12)
    y0 = (_PARAMS.W_sub - _PARAMS.w) / 2.0
    y1 = (_PARAMS.W_sub + _PARAMS.w) / 2.0
    assert np.all(pts[:, 1] >= y0 - 1e-12)
    assert np.all(pts[:, 1] <= y1 + 1e-12)


def test_ports_lie_within_air_plus_substrate_height(built):
    """PORT_1/PORT_2 are restricted to z in [0, z_air_top] (substrate +
    air only) -- the PML's own end caps are PML_OUTER_PEC instead."""
    mesh_handle, _ = built
    z_air_top = _PARAMS.h_sub + _PARAMS.h_air
    for name, x_expected in ((PORT_1, 0.0), (PORT_2, _PARAMS.L)):
        triangles = mesh_handle.surface_tags[name]
        pts = mesh_handle.vertices[triangles.ravel()]
        assert pts[:, 0] == pytest.approx(x_expected, abs=1e-9)
        assert np.all(pts[:, 2] <= z_air_top + 1e-9)


def test_pml_outer_pec_lies_above_air_top(built):
    mesh_handle, _ = built
    z_air_top = _PARAMS.h_sub + _PARAMS.h_air
    triangles = mesh_handle.surface_tags[PML_OUTER_PEC]
    pts = mesh_handle.vertices[triangles.ravel()]
    assert np.all(pts[:, 2] >= z_air_top - 1e-9)


def test_pmc_side_lies_on_lateral_walls_within_substrate_and_air(built):
    mesh_handle, _ = built
    z_air_top = _PARAMS.h_sub + _PARAMS.h_air
    triangles = mesh_handle.surface_tags[PMC_SIDE]
    pts = mesh_handle.vertices[triangles.ravel()]
    on_wall = (np.abs(pts[:, 1] - 0.0) < 1e-9) | (np.abs(pts[:, 1] - _PARAMS.W_sub) < 1e-9)
    assert np.all(on_wall)
    assert np.all(pts[:, 2] <= z_air_top + 1e-9)


def test_material_spec_matches_geometry_params(built):
    _, material_spec = built
    assert material_spec.entries[AIR]["eps_r"] == 1.0
    assert material_spec.entries[SUBSTRATE]["eps_r"] == _PARAMS.eps_r_substrate
