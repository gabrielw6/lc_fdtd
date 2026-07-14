"""Validation suite for mesh_sizing.py (docs/meshing_module_plan.md Section
6.3, "mesh_sizing.py"). No Gmsh needed -- pure arithmetic."""
import numpy as np
import pytest
from scipy import constants

from meshing.mesh_sizing import characteristic_length


def test_vacuum_reduces_to_c_over_f():
    f = 5e9
    h = characteristic_length(f, constants.epsilon_0, constants.mu_0, elements_per_wavelength=10)
    expected_lambda = constants.c / f
    assert h == pytest.approx(expected_lambda / 10, rel=1e-9)


def test_nonvacuum_background_hand_computed():
    f = 3e9
    eps_bg = 4.0 * constants.epsilon_0
    mu_bg = 1.0 * constants.mu_0
    h = characteristic_length(f, eps_bg, mu_bg, elements_per_wavelength=8)

    n = np.sqrt(eps_bg * mu_bg)
    expected_lambda = 1.0 / (f * n)
    assert h == pytest.approx(expected_lambda / 8, rel=1e-9)

    # Cross-check against the familiar refractive-index form for a
    # nonmagnetic medium: lambda = (c/f) / sqrt(eps_r*mu_r).
    expected_lambda_alt = (constants.c / f) / np.sqrt(4.0 * 1.0)
    assert expected_lambda == pytest.approx(expected_lambda_alt, rel=1e-9)


def test_default_elements_per_wavelength_is_ten():
    f = 1e9
    assert characteristic_length(f) == pytest.approx(characteristic_length(f, elements_per_wavelength=10))


def test_rejects_nonpositive_frequency():
    with pytest.raises(ValueError):
        characteristic_length(0.0)
    with pytest.raises(ValueError):
        characteristic_length(-1.0)


def test_rejects_nonpositive_elements_per_wavelength():
    with pytest.raises(ValueError):
        characteristic_length(1e9, elements_per_wavelength=0)
