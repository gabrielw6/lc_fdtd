"""Validation suite for transforms.py (docs/meshing_module_plan.md Section
6.3, "transforms.py"). Pure numpy -- no Gmsh needed for this file."""
import numpy as np
import pytest

from meshing import transforms
from meshing.geometry_spec import RigidTransform


def test_identity_leaves_points_unchanged():
    t = transforms.identity()
    p = np.array([1.0, 2.0, 3.0])
    assert transforms.apply_to_point(t, p) == pytest.approx(p)


def test_translation_only():
    t = transforms.translation_only(1.0, -2.0, 0.5)
    p = np.array([0.0, 0.0, 0.0])
    assert transforms.apply_to_point(t, p) == pytest.approx([1.0, -2.0, 0.5])


def test_from_axis_angle_90deg_about_z():
    t = transforms.from_axis_angle([0.0, 0.0, 1.0], np.pi / 2)
    p = np.array([1.0, 0.0, 0.0])
    assert transforms.apply_to_point(t, p) == pytest.approx([0.0, 1.0, 0.0], abs=1e-12)


def test_from_axis_angle_360deg_is_identity():
    t = transforms.from_axis_angle([1.0, 1.0, 1.0], 2.0 * np.pi)
    p = np.array([0.3, -0.7, 1.2])
    assert transforms.apply_to_point(t, p) == pytest.approx(p, abs=1e-10)


def test_from_axis_angle_rejects_zero_axis():
    with pytest.raises(ValueError):
        transforms.from_axis_angle([0.0, 0.0, 0.0], 1.0)


def test_rotation_matrix_is_orthogonal():
    t = transforms.from_axis_angle([1.0, 2.0, -1.0], 0.7)
    R = np.asarray(t.rotation)
    assert R @ R.T == pytest.approx(np.eye(3), abs=1e-10)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)


def test_compose_matches_sequential_application():
    """Section 6.3: 'confirm composing two transforms matches applying them
    in sequence.'"""
    rng = np.random.default_rng(0)
    first = transforms.from_axis_angle([0.0, 1.0, 0.0], 0.4)
    first = RigidTransform(translation=(1.0, 0.0, -0.5), rotation=first.rotation)
    second = transforms.from_axis_angle([1.0, 0.0, 1.0], -0.9)
    second = RigidTransform(translation=(-2.0, 3.0, 0.1), rotation=second.rotation)

    composed = transforms.compose(first, second)

    for _ in range(5):
        p = rng.uniform(-5, 5, size=3)
        sequential = transforms.apply_to_point(second, transforms.apply_to_point(first, p))
        assert transforms.apply_to_point(composed, p) == pytest.approx(sequential, abs=1e-10)


def test_compose_with_identity_is_noop():
    t = transforms.from_axis_angle([0.0, 0.0, 1.0], 1.1)
    t = RigidTransform(translation=(2.0, -1.0, 0.5), rotation=t.rotation)
    identity = transforms.identity()

    p = np.array([1.0, -2.0, 3.0])
    assert transforms.apply_to_point(transforms.compose(identity, t), p) == pytest.approx(
        transforms.apply_to_point(t, p)
    )
    assert transforms.apply_to_point(transforms.compose(t, identity), p) == pytest.approx(
        transforms.apply_to_point(t, p)
    )


def test_rigid_transform_is_hashable():
    t = transforms.from_axis_angle([1.0, 0.0, 0.0], 0.3)
    hash(t)  # must not raise -- Section 2.2's cache requirement
