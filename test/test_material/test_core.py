"""Validation suite for material.core (docs/module2_material_equations.md
Section 7: "Symmetry & passivity: unit-tested against every MaterialModel
implementation, not just the LC path")."""
import numpy as np
import pytest

from material.core import (
    MaterialAssembly,
    MaterialModel,
    MaterialPassivityError,
    MaterialSymmetryError,
    MaterialTagError,
    assemble_symmetric_tensor,
    check_passive_generic,
    check_symmetric,
)


class _ToyMaterial(MaterialModel):
    """A MaterialModel whose returned tensor is caller-controlled, so the
    base class's automatic per-call checks (Section 1.2/1.3) can be
    exercised directly, independent of any real material implementation."""

    def __init__(self, eps_tensor: np.ndarray, mu_tensor: np.ndarray | None = None):
        self._eps = eps_tensor
        self._mu_tensor = mu_tensor if mu_tensor is not None else np.eye(3, dtype=complex)

    def _epsilon(self, points):
        return np.tile(self._eps, (points.shape[0], 1, 1))

    def _mu(self, points):
        return np.tile(self._mu_tensor, (points.shape[0], 1, 1))


def test_symmetric_tensor_passes_the_check():
    check_symmetric(np.array([[[1.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 3.0]]]))


def test_asymmetric_tensor_fails_the_check():
    with pytest.raises(MaterialSymmetryError):
        check_symmetric(np.array([[[1.0, 0.5, 0.0], [0.9, 2.0, 0.0], [0.0, 0.0, 3.0]]]))


def test_passive_tensor_passes_the_check():
    check_passive_generic(np.array([(2.0 - 0.1j) * np.eye(3)]))


def test_gain_tensor_fails_the_check():
    with pytest.raises(MaterialPassivityError):
        check_passive_generic(np.array([(2.0 + 0.1j) * np.eye(3)]))  # positive Im(eps) => gain


def test_material_model_epsilon_call_runs_symmetry_check_automatically():
    bad = np.array([[1.0, 0.5, 0.0], [0.9, 2.0, 0.0], [0.0, 0.0, 3.0]]) + 0j
    toy = _ToyMaterial(bad)
    with pytest.raises(MaterialSymmetryError):
        toy.epsilon(np.zeros((1, 3)))


def test_material_model_epsilon_call_runs_passivity_check_automatically():
    gain = (2.0 + 0.1j) * np.eye(3)
    toy = _ToyMaterial(gain)
    with pytest.raises(MaterialPassivityError):
        toy.epsilon(np.zeros((1, 3)))


def test_material_model_mu_call_runs_the_same_checks():
    gain_mu = (1.0 + 0.1j) * np.eye(3)
    toy = _ToyMaterial(np.eye(3, dtype=complex), gain_mu)
    with pytest.raises(MaterialPassivityError):
        toy.mu(np.zeros((1, 3)))


def test_valid_toy_material_passes_both_checks():
    toy = _ToyMaterial((3.0 - 0.2j) * np.eye(3))
    eps = toy.epsilon(np.zeros((2, 3)))
    assert eps.shape == (2, 3, 3)


def test_assemble_symmetric_tensor_matches_hand_layout():
    components = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    tensor = assemble_symmetric_tensor(components)
    expected = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]])
    assert np.allclose(tensor[0], expected)


def test_assemble_symmetric_tensor_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        assemble_symmetric_tensor(np.array([[1.0, 2.0, 3.0]]))


def test_material_assembly_dispatches_by_tag():
    toy_a = _ToyMaterial(2.0 * np.eye(3, dtype=complex))
    toy_b = _ToyMaterial(5.0 * np.eye(3, dtype=complex))
    assembly = MaterialAssembly({"A": toy_a, "B": toy_b})

    eps_a = assembly.epsilon("A", np.zeros((1, 3)))
    eps_b = assembly.epsilon("B", np.zeros((1, 3)))
    assert np.allclose(eps_a, 2.0 * np.eye(3))
    assert np.allclose(eps_b, 5.0 * np.eye(3))


def test_material_assembly_raises_for_unregistered_tag():
    assembly = MaterialAssembly({"A": _ToyMaterial(np.eye(3, dtype=complex))})
    with pytest.raises(MaterialTagError):
        assembly.epsilon("NOT_REGISTERED", np.zeros((1, 3)))
