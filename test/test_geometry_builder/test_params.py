"""Validation suite for geometry_builder.params (docs/module0_geometry_builder_equations.md
Section 1.3, Section 6). No Gmsh needed -- pure arithmetic and validation."""
import pytest

from geometry_builder.params import GeometryParameterError, GeometryParams, derive

_BASE = dict(
    w=0.002,
    L=0.020,
    L_lc=0.008,
    W_lc=0.004,
    h_sub=0.002,
    W_sub=0.010,
    eps_r_substrate=3.0,
    reference_frequency=25e9,
)


def _params(**overrides) -> GeometryParams:
    return GeometryParams(**{**_BASE, **overrides})


def test_defaults_h_air_and_h_pml_are_multiples_of_h_sub():
    p = _params()
    assert p.h_air == pytest.approx(3.0 * p.h_sub)
    assert p.h_pml == pytest.approx(0.5 * p.h_sub)


def test_explicit_h_air_h_pml_are_not_overridden():
    p = _params(h_air=0.005, h_pml=0.001)
    assert p.h_air == pytest.approx(0.005)
    assert p.h_pml == pytest.approx(0.001)


def test_derived_quantities_hand_computed():
    p = _params()
    geom = derive(p)
    assert geom.x_c0 == pytest.approx((p.L - p.L_lc) / 2.0)
    assert geom.x_c1 == pytest.approx((p.L + p.L_lc) / 2.0)
    assert geom.y_lc0 == pytest.approx((p.W_sub - p.W_lc) / 2.0)
    assert geom.y_lc1 == pytest.approx((p.W_sub + p.W_lc) / 2.0)
    assert geom.y0_trace == pytest.approx((p.W_sub - p.w) / 2.0)
    assert geom.y1_trace == pytest.approx((p.W_sub + p.w) / 2.0)
    assert geom.z_gnd == pytest.approx(0.0)
    assert geom.z_iface == pytest.approx(p.h_sub)
    assert geom.z_air_top == pytest.approx(p.h_sub + p.h_air)
    assert geom.z_pml_top == pytest.approx(p.h_sub + p.h_air + p.h_pml)


def test_trace_width_equal_to_cavity_width_is_allowed():
    """Section 1.3: w <= W_lc, inclusive."""
    p = _params(w=0.004, W_lc=0.004)
    derive(p)  # must not raise


@pytest.mark.parametrize(
    "overrides",
    [
        {"w": 0.0},
        {"w": -0.001},
    ],
)
def test_rejects_nonpositive_w(overrides):
    with pytest.raises(GeometryParameterError):
        _params(**overrides)


@pytest.mark.parametrize("L_lc", [0.0, -0.001, 0.020, 0.021])
def test_rejects_L_lc_out_of_range(L_lc):
    with pytest.raises(GeometryParameterError):
        _params(L_lc=L_lc)


@pytest.mark.parametrize("W_lc", [0.0, -0.001, 0.010, 0.011])
def test_rejects_W_lc_out_of_range(W_lc):
    with pytest.raises(GeometryParameterError):
        _params(W_lc=W_lc)


def test_rejects_trace_wider_than_cavity():
    with pytest.raises(GeometryParameterError):
        _params(w=0.005, W_lc=0.004)


@pytest.mark.parametrize("field", ["h_sub"])
def test_rejects_nonpositive_h_sub(field):
    with pytest.raises(GeometryParameterError):
        _params(**{field: 0.0})


def test_rejects_nonpositive_h_air():
    with pytest.raises(GeometryParameterError):
        _params(h_air=0.0)


def test_rejects_nonpositive_h_pml():
    with pytest.raises(GeometryParameterError):
        _params(h_pml=-0.001)


def test_rejects_eps_r_substrate_below_one():
    with pytest.raises(GeometryParameterError):
        _params(eps_r_substrate=0.5)


@pytest.mark.parametrize("frequency", [0.0, -1e9])
def test_rejects_nonpositive_reference_frequency(frequency):
    with pytest.raises(GeometryParameterError):
        _params(reference_frequency=frequency)
