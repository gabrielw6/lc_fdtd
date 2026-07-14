"""Validation suite for material.regions (docs/module2_material_equations.md
Section 3, Section 7's reduction targets)."""
import numpy as np
import pytest

from material.interpolation import CoverageError, SampledField
from material.regions import ConstantMaterial, ScalarFieldMaterial, TensorFieldMaterial

_SAMPLE_POINTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
)


def test_constant_material_is_isotropic_and_position_independent():
    cm = ConstantMaterial(eps_r=3.5, mu_r=1.0)
    points = np.array([[0.0, 0.0, 0.0], [10.0, -3.0, 2.0]])
    eps = cm.epsilon(points)
    assert eps.shape == (2, 3, 3)
    assert np.allclose(eps, 3.5 * np.eye(3))
    mu = cm.mu(points)
    assert np.allclose(mu, np.eye(3))


def test_constant_material_supports_complex_eps_r():
    cm = ConstantMaterial(eps_r=3.5 - 0.01j)
    eps = cm.epsilon(np.zeros((1, 3)))
    assert eps[0, 0, 0] == pytest.approx(3.5 - 0.01j)


def test_scalar_field_material_interpolates():
    values = np.array([1.0, 2.0, 1.0, 1.0, 3.0])
    field = SampledField.scattered(_SAMPLE_POINTS, values)
    sfm = ScalarFieldMaterial(field)
    eps = sfm.epsilon(_SAMPLE_POINTS[1:2])
    assert eps[0, 0, 0] == pytest.approx(2.0)
    assert np.allclose(eps[0], eps[0, 0, 0] * np.eye(3))


def test_scalar_field_reduces_to_constant_phase1():
    """Section 3.4: Phase 2 with a spatially constant scalar field must
    reproduce Phase 1 exactly."""
    field = SampledField.scattered(_SAMPLE_POINTS, np.full(5, 2.5))
    sfm = ScalarFieldMaterial(field)
    cm = ConstantMaterial(eps_r=2.5)
    query = np.array([[0.3, 0.3, 0.1]])
    assert np.allclose(sfm.epsilon(query), cm.epsilon(query), atol=1e-9)


def test_scalar_field_material_rejects_wrong_channel_count():
    field = SampledField.scattered(_SAMPLE_POINTS, np.tile([1.0, 2.0], (5, 1)))
    with pytest.raises(ValueError):
        ScalarFieldMaterial(field)


def test_scalar_field_material_enforces_coverage_guard():
    field = SampledField.scattered(_SAMPLE_POINTS, np.full(5, 2.5))
    with pytest.raises(CoverageError):
        ScalarFieldMaterial(field, region_bounds=(np.array([-1.0, -1.0, -1.0]), np.array([2.0, 2.0, 2.0])))


def test_scalar_field_from_file(tmp_path):
    path = tmp_path / "scalar.csv"
    path.write_text(
        "# units: m\n"
        "x, y, z, value\n"
        + "\n".join(f"{p[0]}, {p[1]}, {p[2]}, 2.5" for p in _SAMPLE_POINTS)
    )
    sfm = ScalarFieldMaterial.from_file(path)
    eps = sfm.epsilon(np.array([[0.2, 0.2, 0.2]]))
    assert eps[0, 0, 0] == pytest.approx(2.5)


def test_tensor_field_material_reassembles_components():
    components = np.tile([1.0, 0.5, 0.0, 2.0, 0.0, 3.0], (5, 1))
    field = SampledField.scattered(_SAMPLE_POINTS, components)
    tfm = TensorFieldMaterial(field)
    eps = tfm.epsilon(_SAMPLE_POINTS[:1])
    expected = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 3.0]])
    assert np.allclose(eps[0], expected)


def test_tensor_field_reduces_to_constant_isotropic_phase1():
    """Section 3.4: Phase 3 with a spatially constant, diagonal tensor
    with all three diagonal entries equal must reproduce Phase 1 exactly
    -- a second, independent route to the same reduction."""
    components = np.tile([2.5, 0.0, 0.0, 2.5, 0.0, 2.5], (5, 1))
    field = SampledField.scattered(_SAMPLE_POINTS, components)
    tfm = TensorFieldMaterial(field)
    cm = ConstantMaterial(eps_r=2.5)
    query = np.array([[0.2, 0.2, 0.2]])
    assert np.allclose(tfm.epsilon(query), cm.epsilon(query), atol=1e-9)


def test_tensor_field_material_rejects_wrong_channel_count():
    field = SampledField.scattered(_SAMPLE_POINTS, np.full(5, 2.5))
    with pytest.raises(ValueError):
        TensorFieldMaterial(field)
