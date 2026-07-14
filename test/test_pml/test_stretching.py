"""Validation suite for pml.stretching (docs/module5_pml_equations.md
Section 2). Pure numpy, no mesh/gmsh needed.
"""
import inspect

import numpy as np
import pytest

from pml.stretching import kappa_profile, lambda_tensor, s_z, sigma_max_for_R0, sigma_profile

_ETA_0 = 376.730313668  # free-space impedance, sqrt(mu0/eps0) -- independent literal, not the module's own constant


# --- Section 2.4: sigma_max is frequency-independent ---


def test_sigma_max_for_R0_has_no_omega_parameter():
    """Structural check (Section 7 step 1): the derivation's whole point is
    that the k0/(omega*eps0)=eta0 cancellation makes sigma_max
    frequency-independent -- confirm that property wasn't accidentally
    broken by checking the function's own signature."""
    params = set(inspect.signature(sigma_max_for_R0).parameters)
    assert "omega" not in params


def test_sigma_max_reproduces_target_round_trip_reflection():
    """Section 2.4: e^(-2*eta0*sigma_max*d/(n+1)) == R0 by construction."""
    R0, n, d = 1e-6, 2, 0.002
    sigma_max = sigma_max_for_R0(R0, n, d)
    attenuation = np.exp(-2.0 * _ETA_0 * sigma_max * d / (n + 1))
    assert attenuation == pytest.approx(R0, rel=1e-10)


def test_sigma_max_positive_for_valid_R0():
    assert sigma_max_for_R0(1e-6, 2, 0.002) > 0
    assert sigma_max_for_R0(1e-5, 3, 0.001) > 0


# --- Section 2.3: grading profiles ---


def test_profiles_vanish_at_xi_zero_and_hit_max_at_thickness():
    d, sigma_max, kappa_max, n = 0.002, 27.5, 3.0, 2
    xi = np.array([0.0, d])
    sigma = sigma_profile(xi, d, sigma_max, n)
    kappa = kappa_profile(xi, d, kappa_max, n)
    assert sigma[0] == pytest.approx(0.0)
    assert sigma[1] == pytest.approx(sigma_max)
    assert kappa[0] == pytest.approx(1.0)
    assert kappa[1] == pytest.approx(kappa_max)


def test_profiles_monotone_increasing_in_xi():
    d = 0.002
    xi = np.linspace(0, d, 20)
    sigma = sigma_profile(xi, d, 27.5, 2)
    kappa = kappa_profile(xi, d, 3.0, 2)
    assert np.all(np.diff(sigma) >= 0)
    assert np.all(np.diff(kappa) >= 0)


# --- Section 2.2: s_z structure ---


def test_s_z_real_part_is_kappa_imag_part_is_minus_sigma_over_omega_eps0():
    from scipy import constants as c

    d, sigma_max, kappa_max, n, omega = 0.002, 27.5, 3.0, 2, 2 * np.pi * 25e9
    xi = np.array([0.0005, 0.0015])
    sz = s_z(xi, omega, d, sigma_max, kappa_max, n)
    kappa = kappa_profile(xi, d, kappa_max, n)
    sigma = sigma_profile(xi, d, sigma_max, n)
    assert sz.real == pytest.approx(kappa)
    assert sz.imag == pytest.approx(-sigma / (omega * c.epsilon_0))


def test_s_z_reduces_to_one_when_sigma_and_kappa_are_trivial():
    xi = np.array([0.0, 0.001, 0.002])
    sz = s_z(xi, omega=1e11, thickness=0.002, sigma_max=0.0, kappa_max=1.0, n=2)
    assert sz == pytest.approx(np.ones(3, dtype=complex))


# --- Section 4.1: Im(1/s_z) >= 0 (the passivity-exemption finding) ---


def test_inverse_s_z_has_nonnegative_imaginary_part():
    rng = np.random.default_rng(0)
    kappa = 1.0 + rng.uniform(0, 5, size=200)
    sigma = rng.uniform(0, 50, size=200)
    omega = 2 * np.pi * 25e9
    from scipy import constants as c

    sz = kappa - 1j * sigma / (omega * c.epsilon_0)
    assert np.all((1.0 / sz).imag >= -1e-15)


# --- Section 2.1: Lambda tensor ---


def test_lambda_reduction_to_identity():
    sz = np.array([1.0 + 0j])
    lam = lambda_tensor(sz)
    assert lam[0] == pytest.approx(np.eye(3))


def test_lambda_single_axis_form_diag_sz_sz_inv_sz():
    sz = np.array([2.0 - 1.5j])
    lam = lambda_tensor(sz)
    assert lam[0, 0, 0] == pytest.approx(sz[0])
    assert lam[0, 1, 1] == pytest.approx(sz[0])
    assert lam[0, 2, 2] == pytest.approx(1.0 / sz[0])
    off_diag = lam[0] - np.diag(np.diag(lam[0]))
    assert np.allclose(off_diag, 0)


def test_lambda_general_form_matches_boxed_formula():
    """Section 2.1's general `diag(sy*sz/sx, sz*sx/sy, sx*sy/sz)`, checked
    with non-trivial s_x, s_y (not this geometry's default, but the
    function's actual documented signature)."""
    sz = np.array([1.2 - 0.3j])
    sx, sy = 1.5 + 0.1j, 0.8 - 0.2j
    lam = lambda_tensor(sz, s_x=sx, s_y=sy)
    assert lam[0, 0, 0] == pytest.approx(sy * sz[0] / sx)
    assert lam[0, 1, 1] == pytest.approx(sz[0] * sx / sy)
    assert lam[0, 2, 2] == pytest.approx(sx * sy / sz[0])
