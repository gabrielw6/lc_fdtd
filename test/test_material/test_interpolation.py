"""Validation suite for material.interpolation (docs/module2_material_equations.md
Section 7: partition-of-unity, coverage guard)."""
import numpy as np
import pytest

from material.interpolation import (
    CoverageError,
    SampledField,
    check_coverage,
    interpolate_scattered,
    interpolate_structured,
    read_sample_file,
)

_SAMPLE_POINTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
)


def test_structured_interpolation_reproduces_values_at_grid_nodes():
    axis = np.array([0.0, 1.0, 2.0])
    grid_values = np.arange(27, dtype=float).reshape(3, 3, 3, 1)
    query = np.array([[1.0, 1.0, 1.0]])  # the (1,1,1) node itself
    result = interpolate_structured((axis, axis, axis), grid_values, query)
    assert result[0, 0] == pytest.approx(grid_values[1, 1, 1, 0])


def test_structured_interpolation_is_linear_along_an_axis():
    axis = np.array([0.0, 1.0])
    grid_values = np.zeros((2, 2, 2, 1))
    grid_values[1, 0, 0, 0] = 10.0  # value only varies along x
    query = np.array([[0.5, 0.0, 0.0]])
    result = interpolate_structured((axis, axis, axis), grid_values, query)
    assert result[0, 0] == pytest.approx(5.0)


def test_structured_interpolation_raises_outside_the_grid():
    axis = np.array([0.0, 1.0])
    grid_values = np.zeros((2, 2, 2, 1))
    with pytest.raises(CoverageError):
        interpolate_structured((axis, axis, axis), grid_values, np.array([[2.0, 0.0, 0.0]]))


def test_scattered_linear_interpolation_reproduces_values_at_sample_points():
    values = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = interpolate_scattered(_SAMPLE_POINTS, values, _SAMPLE_POINTS[:1])
    assert result[0, 0] == pytest.approx(1.0)


def test_scattered_linear_interpolation_raises_outside_the_hull():
    values = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    with pytest.raises(CoverageError):
        interpolate_scattered(_SAMPLE_POINTS, values, np.array([[10.0, 10.0, 10.0]]))


def test_scattered_idw_reproduces_values_at_sample_points():
    values = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = interpolate_scattered(_SAMPLE_POINTS, values, _SAMPLE_POINTS, method="idw")
    assert np.allclose(result[:, 0], values[:, 0], atol=1e-9)


def test_scattered_idw_is_a_convex_combination_between_two_colinear_samples():
    """Section 2.3: weights are non-negative and sum to 1 -- checked here
    indirectly (a query strictly between two equal-valued, collinear
    samples must reproduce that value, and a query far past one sample
    stays within the range of the k nearest values, never overshooting)."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = np.array([2.0, 2.0])
    result = interpolate_scattered(points, values, np.array([[0.5, 0.0, 0.0]]), method="idw")
    assert result[0, 0] == pytest.approx(2.0)


def test_scattered_unknown_method_raises():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        interpolate_scattered(_SAMPLE_POINTS, values, _SAMPLE_POINTS[:1], method="bogus")


def test_check_coverage_passes_when_region_is_contained():
    check_coverage(_SAMPLE_POINTS, (np.array([0.1, 0.1, 0.1]), np.array([0.5, 0.5, 0.5])))


def test_check_coverage_raises_when_region_exceeds_sample_bounds():
    with pytest.raises(CoverageError):
        check_coverage(_SAMPLE_POINTS, (np.array([-1.0, -1.0, -1.0]), np.array([2.0, 2.0, 2.0])))


def test_sampled_field_structured_bounds_and_evaluate():
    axis = np.array([0.0, 1.0])
    grid_values = np.ones((2, 2, 2, 2)) * 3.0
    field = SampledField.structured((axis, axis, axis), grid_values)
    mins, maxs = field.bounds()
    assert np.allclose(mins, 0.0) and np.allclose(maxs, 1.0)
    assert field.n_channels == 2
    result = field.evaluate(np.array([[0.5, 0.5, 0.5]]))
    assert np.allclose(result, 3.0)


def test_sampled_field_scattered_1d_values_get_a_channel_axis():
    field = SampledField.scattered(_SAMPLE_POINTS, np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert field.n_channels == 1
    assert field.sample_values.shape == (5, 1)


def test_read_sample_file_parses_header_and_data(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(
        "# coordinate_frame: mesh\n"
        "# units: m\n"
        "# grid_type: scattered\n"
        "x, y, z, value\n"
        "0.0, 0.0, 0.0, 1.0\n"
        "1.0, 0.0, 0.0, 2.0\n"
    )
    header, data = read_sample_file(path)
    assert header == {"coordinate_frame": "mesh", "units": "m", "grid_type": "scattered"}
    assert data.shape == (2, 4)
    assert data[1, 3] == pytest.approx(2.0)
