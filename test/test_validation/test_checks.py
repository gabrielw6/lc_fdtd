"""docs/module8_validation_equations.md Section 2's shared check library,
tested against hand-built cases with a known answer (Section 6 build
step 2): a matrix that's exactly symmetric and one that isn't, an S set
that's exactly reciprocal and one that isn't, etc.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from material import ConstantMaterial
from pml import PMLMaterial
from validation.checks import (
    ValidationError,
    assert_passive,
    assert_reciprocal,
    assert_reduction,
    assert_symmetric,
    estimate_convergence_order,
    recommended_tolerance_from_pml,
)

# --- Section 2.1: assert_symmetric ---


def test_assert_symmetric_accepts_symmetric_dense_matrix():
    M = np.array([[1.0 + 1j, 2.0 - 1j], [2.0 - 1j, 3.0]])
    assert_symmetric(M)  # must not raise


def test_assert_symmetric_rejects_asymmetric_dense_matrix():
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValidationError):
        assert_symmetric(M)


def test_assert_symmetric_accepts_symmetric_sparse_matrix():
    M = sp.csr_matrix(np.array([[2.0, 1.0j], [1.0j, 5.0]]))
    assert_symmetric(M)  # must not raise


def test_assert_symmetric_rejects_asymmetric_sparse_matrix():
    M = sp.csr_matrix(np.array([[1.0, 5.0], [0.0, 1.0]]))
    with pytest.raises(ValidationError):
        assert_symmetric(M)


def test_assert_symmetric_tolerance_is_relative_to_matrix_scale():
    # Same absolute asymmetry (1e-8), very different scale -- should pass
    # for a large-scale matrix and fail for a tiny-scale one at the same tol.
    big = np.array([[1000.0, 1000.0 + 1e-8], [1000.0, 1000.0]])
    small = np.array([[1e-6, 1e-6 + 1e-8], [1e-6, 1e-6]])
    assert_symmetric(big, tol=1e-9)
    with pytest.raises(ValidationError):
        assert_symmetric(small, tol=1e-9)


# --- Section 2.2: assert_passive ---


def test_assert_passive_accepts_lossy_passive_tensor():
    eps = np.array([[[3.0 - 0.1j, 0, 0], [0, 3.0 - 0.1j, 0], [0, 0, 3.0 - 0.1j]]])
    assert_passive(eps)  # must not raise


def test_assert_passive_rejects_gain_tensor():
    eps = np.array([[[3.0 + 0.1j, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]])
    with pytest.raises(ValidationError):
        assert_passive(eps)


def test_assert_passive_exempts_pmlmaterial_by_default():
    """Section 2.2's required, always-active PMLMaterial skip-list entry
    (Module 5 Section 4)."""
    pml = PMLMaterial(ConstantMaterial(eps_r=1.0), omega=2 * np.pi * 25e9, z_air_top=0.008, thickness=0.002)
    deep_pml_eps = pml.epsilon(np.array([[0.0, 0.0, 0.0099]]))  # near the PEC backing -- Im(eps_zz) > 0
    assert deep_pml_eps[:, 2, 2].imag.max() > 0  # confirm the tensor really would fail the generic check
    assert_passive(deep_pml_eps, material=pml)  # must not raise -- exempted


def test_assert_passive_does_not_exempt_non_pml_material_by_default():
    eps = np.array([[[3.0 + 0.1j, 0, 0], [0, 3.0, 0], [0, 0, 3.0]]])
    with pytest.raises(ValidationError):
        assert_passive(eps, material=ConstantMaterial(eps_r=3.0))


def test_assert_passive_energy_dict_accepts_lossless_conserving_S():
    S = {("PORT_1", 1, "PORT_1"): 0.6 + 0j, ("PORT_2", 1, "PORT_1"): 0.8 + 0j}
    assert_passive(S, excitation_port="PORT_1", n_modes=1)  # 0.36+0.64=1.0, must not raise


def test_assert_passive_energy_dict_rejects_gain_S():
    S = {("PORT_1", 1, "PORT_1"): 0.9 + 0j, ("PORT_2", 1, "PORT_1"): 0.9 + 0j}  # 0.81+0.81=1.62 > 1
    with pytest.raises(ValidationError):
        assert_passive(S, excitation_port="PORT_1", n_modes=1)


def test_assert_passive_energy_dict_requires_excitation_and_n_modes():
    S = {("PORT_1", 1, "PORT_1"): 0.5 + 0j}
    with pytest.raises(ValueError):
        assert_passive(S)


# --- Section 2.3: assert_reciprocal ---


def test_assert_reciprocal_accepts_symmetric_S():
    S = np.array([[0.1 + 0.2j, 0.3 - 0.1j], [0.3 - 0.1j, 0.05j]])
    assert_reciprocal(S)  # must not raise


def test_assert_reciprocal_rejects_asymmetric_S():
    S = np.array([[0.1, 0.3], [0.5, 0.1]])
    with pytest.raises(ValidationError):
        assert_reciprocal(S)


def test_assert_reciprocal_handles_frequency_batched_S():
    S_ok = np.array([[[0.1, 0.2], [0.2, 0.1]], [[0.3, 0.4], [0.4, 0.3]]])
    assert_reciprocal(S_ok)  # (2 freqs, 2x2) -- must not raise

    S_bad = S_ok.copy()
    S_bad[1, 0, 1] = 0.99
    with pytest.raises(ValidationError):
        assert_reciprocal(S_bad)


# --- Section 2.4: estimate_convergence_order ---


def test_estimate_convergence_order_recovers_known_first_order():
    # A quantity converging as O(h): error halves each time h halves.
    errors = [0.08, 0.04, 0.02, 0.01]
    order = estimate_convergence_order(errors, refinement_ratio=2.0)
    assert order == pytest.approx(1.0, abs=1e-9)


def test_estimate_convergence_order_recovers_known_second_order():
    # O(h^2): error quarters each time h halves.
    errors = [0.16, 0.04, 0.01]
    order = estimate_convergence_order(errors, refinement_ratio=2.0)
    assert order == pytest.approx(2.0, abs=1e-9)


def test_estimate_convergence_order_requires_at_least_two_values():
    with pytest.raises(ValueError):
        estimate_convergence_order([0.1], refinement_ratio=2.0)


# --- Section 2.5: assert_reduction ---


def test_assert_reduction_accepts_matching_dense_results():
    a = np.array([1.0 + 1j, 2.0, 3.0 - 1j])
    b = a.copy()
    assert_reduction(a, b)  # must not raise


def test_assert_reduction_rejects_mismatched_results():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.5])
    with pytest.raises(ValidationError):
        assert_reduction(a, b, tol=1e-9)


def test_assert_reduction_rejects_shape_mismatch():
    with pytest.raises(ValidationError):
        assert_reduction(np.zeros(3), np.zeros(4))


def test_assert_reduction_accepts_matching_sparse_results():
    a = sp.csr_matrix(np.eye(3, dtype=complex) * 2.0)
    b = sp.csr_matrix(np.eye(3, dtype=complex) * 2.0)
    assert_reduction(a, b)  # must not raise


# --- Section 2.6: recommended_tolerance_from_pml ---


def test_recommended_tolerance_is_looser_than_R0():
    R0 = 1e-6
    tol = recommended_tolerance_from_pml(R0)
    assert tol > R0
    assert tol == pytest.approx(10 * R0)
