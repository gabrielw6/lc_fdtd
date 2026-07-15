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


# --- Section 1.4: port aperture ---


def test_w_port_h_port_default_to_full_cross_section():
    p = _params()
    assert p.W_port == pytest.approx(p.W_sub)
    assert p.H_port == pytest.approx(p.h_sub + p.h_air)


def test_explicit_w_port_h_port_are_not_overridden():
    p = _params(W_port=0.006, H_port=0.005)
    assert p.W_port == pytest.approx(0.006)
    assert p.H_port == pytest.approx(0.005)


@pytest.mark.parametrize("overrides", [{"W_port": 0.006}, {"H_port": 0.005}])
def test_rejects_w_port_h_port_given_separately(overrides):
    with pytest.raises(GeometryParameterError):
        _params(**overrides)


@pytest.mark.parametrize("W_port", [0.0009, 0.0, -0.001, 0.011])
def test_rejects_w_port_out_of_range(W_port):
    """W_port must satisfy w <= W_port <= W_sub (w=0.002, W_sub=0.010)."""
    with pytest.raises(GeometryParameterError):
        _params(W_port=W_port, H_port=0.005)


@pytest.mark.parametrize("H_port", [0.002, 0.001, 0.0, 0.009])
def test_rejects_h_port_out_of_range(H_port):
    """H_port must satisfy h_sub < H_port <= z_air_top (h_sub=0.002,
    z_air_top=h_sub+3*h_sub=0.008 by default)."""
    with pytest.raises(GeometryParameterError):
        _params(W_port=0.006, H_port=H_port)


def test_w_port_equal_to_w_sub_and_h_port_equal_to_z_air_top_are_allowed():
    p = _params(W_port=0.010, H_port=0.008)
    assert p.W_port == pytest.approx(0.010)
    assert p.H_port == pytest.approx(0.008)


def test_derived_port_aperture_bounds_hand_computed():
    p = _params(W_port=0.006, H_port=0.005)
    geom = derive(p)
    assert geom.y0_port == pytest.approx((p.W_sub - p.W_port) / 2.0)
    assert geom.y1_port == pytest.approx((p.W_sub + p.W_port) / 2.0)


def test_rejects_port_aperture_not_containing_trace():
    """w=0.002 centered on W_sub=0.010 -> y0_trace=0.004, y1_trace=0.006.
    A W_port narrower than w, offset so it's still centered, cannot
    contain the trace -- but centering makes w<=W_port sufficient here, so
    this exercises Section 1.3's hard w<=W_port bound instead (the
    explicit y0_port/y1_port containment check is the geometric-level
    consequence of that same bound, per derive()'s own defensive style)."""
    with pytest.raises(GeometryParameterError):
        _params(W_port=0.001, H_port=0.005)
