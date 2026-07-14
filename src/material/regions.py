"""material.regions -- Phases 1-3: materials supplied directly, not
director-derived (docs/module2_material_equations.md Section 3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import MaterialModel, assemble_symmetric_tensor
from .interpolation import SampledField, check_coverage, read_sample_file

RegionBounds = tuple[np.ndarray, np.ndarray]


class ConstantMaterial(MaterialModel):
    """Phase 1 (Section 3.1): eps_r(r) = eps_r,t * I, no interpolation."""

    def __init__(self, eps_r: complex, mu_r: complex = 1.0) -> None:
        self._eps_r = complex(eps_r)
        self._mu_r = complex(mu_r)

    def _epsilon(self, points: np.ndarray) -> np.ndarray:
        return np.tile(self._eps_r * np.eye(3, dtype=complex), (points.shape[0], 1, 1))

    def _mu(self, points: np.ndarray) -> np.ndarray:
        return np.tile(self._mu_r * np.eye(3, dtype=complex), (points.shape[0], 1, 1))


class ScalarFieldMaterial(MaterialModel):
    """Phase 2 (Section 3.2): eps_r(r) = eps_r,t(r) * I, a scalar field
    (C=1) evaluated via `material.interpolation`."""

    def __init__(self, field: SampledField, *, mu_r: complex = 1.0, region_bounds: RegionBounds | None = None) -> None:
        if field.n_channels != 1:
            raise ValueError(f"ScalarFieldMaterial needs a 1-channel field, got {field.n_channels} channels")
        if region_bounds is not None:
            check_coverage(_field_sample_points(field), region_bounds)
        self._field = field
        self._mu_r = complex(mu_r)

    @classmethod
    def from_file(
        cls, path: str | Path, *, mu_r: complex = 1.0, region_bounds: RegionBounds | None = None, method: str = "linear"
    ) -> "ScalarFieldMaterial":
        _header, data = read_sample_file(path)
        points, values = data[:, :3], data[:, 3]
        field = SampledField.scattered(points, values, method=method)
        return cls(field, mu_r=mu_r, region_bounds=region_bounds)

    def _epsilon(self, points: np.ndarray) -> np.ndarray:
        eps_scalar = self._field.evaluate(points)[:, 0]
        return eps_scalar[:, None, None] * np.eye(3, dtype=complex)[None, :, :]

    def _mu(self, points: np.ndarray) -> np.ndarray:
        return np.tile(self._mu_r * np.eye(3, dtype=complex), (points.shape[0], 1, 1))


class TensorFieldMaterial(MaterialModel):
    """Phase 3 (Section 3.3): a general symmetric tensor field, its six
    independent components (C=6, order xx,xy,xz,yy,yz,zz) supplied
    directly and evaluated via `material.interpolation`."""

    def __init__(self, field: SampledField, *, mu_r: complex = 1.0, region_bounds: RegionBounds | None = None) -> None:
        if field.n_channels != 6:
            raise ValueError(f"TensorFieldMaterial needs a 6-channel field, got {field.n_channels} channels")
        if region_bounds is not None:
            check_coverage(_field_sample_points(field), region_bounds)
        self._field = field
        self._mu_r = complex(mu_r)

    @classmethod
    def from_file(
        cls, path: str | Path, *, mu_r: complex = 1.0, region_bounds: RegionBounds | None = None, method: str = "linear"
    ) -> "TensorFieldMaterial":
        _header, data = read_sample_file(path)
        points, values = data[:, :3], data[:, 3:9]
        field = SampledField.scattered(points, values, method=method)
        return cls(field, mu_r=mu_r, region_bounds=region_bounds)

    def _epsilon(self, points: np.ndarray) -> np.ndarray:
        return assemble_symmetric_tensor(self._field.evaluate(points))

    def _mu(self, points: np.ndarray) -> np.ndarray:
        return np.tile(self._mu_r * np.eye(3, dtype=complex), (points.shape[0], 1, 1))


def _field_sample_points(field: SampledField) -> np.ndarray:
    """The points a coverage check should be measured against: the
    structured grid's corner lattice, or the scattered samples directly."""
    if field.kind == "scattered":
        return field.sample_points
    axes = field.grid_axes
    mins, maxs = field.bounds()
    return np.array([mins, maxs])
