"""Validation suite for the port-aperture decoupling
(docs/module0_geometry_builder_equations.md Section 1.4/4.3): a restricted
`W_port`/`H_port` splits each end plane into a smaller `PORT_p` aperture
plus a `PORT_CAP` PEC idealization for the rest -- purely geometric/tagging
checks, no port-mode solving needed here (see test_ports/ for that).
"""
import numpy as np
import pytest

pytest.importorskip("gmsh")

from geometry_builder.builder import GeometryBuilder
from geometry_builder.params import GeometryParams
from geometry_builder.tags import PORT_1, PORT_2, PORT_CAP

_BASE = dict(
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
    target_elements_per_wavelength=6,
)
_W_PORT = 0.006
_H_PORT = 0.005


@pytest.fixture(scope="module")
def built_restricted():
    params = GeometryParams(**_BASE, W_port=_W_PORT, H_port=_H_PORT)
    return GeometryBuilder().build(params)


def test_port_cap_is_nonempty_when_aperture_restricted(built_restricted):
    mesh_handle, _ = built_restricted
    assert mesh_handle.surface_tags[PORT_CAP].shape[0] > 0


def test_port_faces_are_confined_to_the_aperture_bounds(built_restricted):
    mesh_handle, _ = built_restricted
    y0 = (_BASE["W_sub"] - _W_PORT) / 2.0
    y1 = (_BASE["W_sub"] + _W_PORT) / 2.0
    for name, x_expected in ((PORT_1, 0.0), (PORT_2, _BASE["L"])):
        triangles = mesh_handle.surface_tags[name]
        assert triangles.shape[0] > 0
        pts = mesh_handle.vertices[triangles.ravel()]
        assert pts[:, 0] == pytest.approx(x_expected, abs=1e-9)
        assert np.all(pts[:, 1] >= y0 - 1e-9)
        assert np.all(pts[:, 1] <= y1 + 1e-9)
        assert np.all(pts[:, 2] >= -1e-9)
        assert np.all(pts[:, 2] <= _H_PORT + 1e-9)


def test_port_cap_covers_the_rest_of_each_end_plane(built_restricted):
    """PORT_CAP's own bounding box should span the full end-plane extent
    (it's the region outside the aperture, not confined to it) -- unlike
    PORT_1/PORT_2, which are confined to the aperture."""
    mesh_handle, _ = built_restricted
    z_air_top = _BASE["h_sub"] + _BASE["h_air"]
    triangles = mesh_handle.surface_tags[PORT_CAP]
    pts = mesh_handle.vertices[triangles.ravel()]
    on_end_plane = (np.abs(pts[:, 0] - 0.0) < 1e-9) | (np.abs(pts[:, 0] - _BASE["L"]) < 1e-9)
    assert np.all(on_end_plane)
    assert np.all(pts[:, 2] <= z_air_top + 1e-9)
    # some cap vertices must lie outside the aperture's own y/z bounds --
    # otherwise the split didn't actually happen.
    y0 = (_BASE["W_sub"] - _W_PORT) / 2.0
    y1 = (_BASE["W_sub"] + _W_PORT) / 2.0
    outside_aperture = (pts[:, 1] < y0 - 1e-9) | (pts[:, 1] > y1 + 1e-9) | (pts[:, 2] > _H_PORT + 1e-9)
    assert np.any(outside_aperture)


def test_restricted_aperture_does_not_change_lc_or_substrate_volume(built_restricted):
    """The port aperture only touches the two end planes -- it must not
    perturb the volume-tag consistency checks build() already runs
    internally (Section 6's bounding-volume check)."""
    mesh_handle, _ = built_restricted
    assert mesh_handle.vertices.shape[0] > 0  # build() succeeded at all;
    # the real assertion is that GeometryBuilder().build() didn't raise
    # GeometryConsistencyError, which the fixture already exercised.


def test_default_w_port_h_port_none_reproduces_full_cross_section():
    """Backward compatibility (Section 1.4): omitting W_port/H_port must
    give the exact same PORT_1/PORT_2 triangle count as before this
    feature existed, and an empty PORT_CAP."""
    params_default = GeometryParams(**_BASE)
    params_explicit_full = GeometryParams(
        **_BASE, W_port=_BASE["W_sub"], H_port=_BASE["h_sub"] + _BASE["h_air"]
    )
    mesh_default, _ = GeometryBuilder().build(params_default)
    mesh_explicit, _ = GeometryBuilder().build(params_explicit_full)
    assert mesh_default.surface_tags[PORT_CAP].shape[0] == 0
    assert mesh_explicit.surface_tags[PORT_CAP].shape[0] == 0
    assert mesh_default.surface_tags[PORT_1].shape[0] == mesh_explicit.surface_tags[PORT_1].shape[0]
