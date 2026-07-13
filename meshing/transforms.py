"""Meshing module -- RigidTransform construction, composition, and
application (docs/meshing_module_plan.md Section 2.2, step 2).

Construction/composition helpers live here rather than on the
`RigidTransform` dataclass itself (geometry_spec.py), keeping that a plain,
hashable data container. Everything except `apply_to_shape` is pure numpy
-- geometry-agnostic rigid-transform math, testable without Gmsh installed
at all (the doc's own step-2 ordering: this module is built and tested
before Gmsh is ever touched). `apply_to_shape` imports `gmsh` lazily, inside
the function body, specifically to preserve that property for the rest of
this module.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .geometry_spec import Matrix3, RigidTransform

Array = np.ndarray

_IDENTITY_ROTATION: Matrix3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def identity() -> RigidTransform:
    return RigidTransform(translation=(0.0, 0.0, 0.0), rotation=_IDENTITY_ROTATION)


def translation_only(dx: float, dy: float, dz: float) -> RigidTransform:
    return RigidTransform(translation=(dx, dy, dz), rotation=_IDENTITY_ROTATION)


def from_axis_angle(axis: Sequence[float], angle: float) -> RigidTransform:
    """Rotation by `angle` radians about `axis` (through the origin, zero
    translation) -- Rodrigues' rotation formula."""
    a = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(a)
    if norm < 1e-300:
        raise ValueError("rotation axis must be nonzero")
    a = a / norm
    K = np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    rotation = _matrix_to_tuple(R)
    return RigidTransform(translation=(0.0, 0.0, 0.0), rotation=rotation)


def _matrix_to_tuple(R: Array) -> Matrix3:
    rows = [tuple(float(x) for x in row) for row in R]
    return (rows[0], rows[1], rows[2])  # type: ignore[return-value]


def _rotation_matrix(transform: RigidTransform) -> Array:
    return np.asarray(transform.rotation, dtype=float)


def _translation_vector(transform: RigidTransform) -> Array:
    return np.asarray(transform.translation, dtype=float)


def apply_to_point(transform: RigidTransform, point: Sequence[float]) -> Array:
    """p_out = R @ p_in + translation."""
    p = np.asarray(point, dtype=float)
    return _rotation_matrix(transform) @ p + _translation_vector(transform)


def compose(first: RigidTransform, second: RigidTransform) -> RigidTransform:
    """The single transform equivalent to applying `first` then `second`:
    `apply_to_point(compose(first, second), p) == apply_to_point(second,
    apply_to_point(first, p))` for any `p`."""
    R1, t1 = _rotation_matrix(first), _translation_vector(first)
    R2, t2 = _rotation_matrix(second), _translation_vector(second)
    R = R2 @ R1
    t = R2 @ t1 + t2
    return RigidTransform(translation=(float(t[0]), float(t[1]), float(t[2])), rotation=_matrix_to_tuple(R))


def apply_to_shape(transform: RigidTransform, dim_tags: Sequence[tuple[int, int]]) -> None:
    """Applies `transform` to OCC entities `dim_tags` in place, via Gmsh's
    `affineTransform` (a general 3x4 matrix), synchronizing afterward so the
    change is visible to subsequent OCC queries (mass properties, further
    booleans, etc.)."""
    import gmsh

    R = _rotation_matrix(transform)
    t = _translation_vector(transform)
    matrix = np.zeros((3, 4))
    matrix[:, :3] = R
    matrix[:, 3] = t
    gmsh.model.occ.affineTransform(list(dim_tags), matrix.flatten().tolist())
    gmsh.model.occ.synchronize()
