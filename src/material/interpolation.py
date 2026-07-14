"""material.interpolation -- shared, low-level numerical primitive: a
multi-channel scalar-field interpolator over either a structured grid or
scattered samples, plus the coverage (extrapolation) guard
(docs/module2_material_equations.md Section 2).

Used with C=1 by `material.regions`' Phase 2 scalar path and C=6 by both
`material.regions`' Phase 3 tensor path and `material.tensor_interpolation`'s
Phase 4 path -- written and tested exactly once, generically over C, per
Section 2's explicit reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class CoverageError(RuntimeError):
    """Raised when a query would require extrapolation -- either the
    Section 2.4 load-time containment guard, or (defense in depth) the
    interpolator itself refusing an out-of-hull/out-of-grid point."""


def interpolate_structured(
    grid_axes: tuple[np.ndarray, np.ndarray, np.ndarray], grid_values: np.ndarray, query_points: np.ndarray
) -> np.ndarray:
    """Section 2.1: trilinear interpolation on a regular grid, exactly
    `scipy.interpolate.RegularGridInterpolator(method='linear')` (no custom
    trilinear code, per the doc's own implementation note). Real and
    imaginary parts are interpolated separately and recombined, so this
    works regardless of the installed scipy version's own complex-value
    support. Returns (M,C) complex."""
    from scipy.interpolate import RegularGridInterpolator

    query_points = np.atleast_2d(np.asarray(query_points, dtype=float))
    grid_values = np.asarray(grid_values)

    try:
        real_part = RegularGridInterpolator(grid_axes, grid_values.real, method="linear", bounds_error=True)(
            query_points
        )
    except ValueError as exc:
        raise CoverageError(f"query point outside the structured grid: {exc}") from exc

    if np.iscomplexobj(grid_values):
        imag_part = RegularGridInterpolator(grid_axes, grid_values.imag, method="linear", bounds_error=True)(
            query_points
        )
        return real_part + 1j * imag_part
    return real_part.astype(complex)


def interpolate_scattered(
    sample_points: np.ndarray, sample_values: np.ndarray, query_points: np.ndarray, *, method: str = "linear"
) -> np.ndarray:
    """Section 2.2. `method='linear'` (primary): piecewise-linear over the
    Delaunay triangulation of `sample_points`
    (`scipy.interpolate.LinearNDInterpolator`). `method='idw'` (documented
    fallback): inverse-distance weighting restricted to the k nearest
    samples, for point clouds too sparse/degenerate for a robust Delaunay
    triangulation. Both produce non-negative weights summing to 1 (Section
    2.3's convexity property). Returns (M,C) complex."""
    sample_points = np.asarray(sample_points, dtype=float)
    sample_values = np.asarray(sample_values)
    query_points = np.atleast_2d(np.asarray(query_points, dtype=float))

    if method == "linear":
        from scipy.interpolate import LinearNDInterpolator

        real_part = LinearNDInterpolator(sample_points, sample_values.real)(query_points)
        if np.any(np.isnan(real_part)):
            raise CoverageError("query point outside the convex hull of the scattered samples")
        if np.iscomplexobj(sample_values):
            imag_part = LinearNDInterpolator(sample_points, sample_values.imag)(query_points)
            return real_part + 1j * imag_part
        return real_part.astype(complex)
    if method == "idw":
        return _interpolate_idw(sample_points, sample_values, query_points)
    raise ValueError(f"unknown scattered interpolation method {method!r}, expected 'linear' or 'idw'")


def _interpolate_idw(
    sample_points: np.ndarray, sample_values: np.ndarray, query_points: np.ndarray, *, power: float = 2.0, k: int = 8
) -> np.ndarray:
    from scipy.spatial import cKDTree

    values = sample_values if sample_values.ndim > 1 else sample_values[:, None]
    tree = cKDTree(sample_points)
    k_eff = min(k, sample_points.shape[0])
    dist, idx = tree.query(query_points, k=k_eff)
    if k_eff == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    weights = np.zeros_like(dist)
    exact = dist[:, 0] <= 0.0
    nonexact = ~exact
    weights[nonexact] = 1.0 / dist[nonexact] ** power
    weights[exact, 0] = 1.0
    weights /= weights.sum(axis=1, keepdims=True)  # Section 2.3: sum_i w_i = 1, w_i >= 0

    gathered = values[idx]  # (M,k,C)
    return np.einsum("mk,mkc->mc", weights, gathered).astype(complex)


def check_coverage(sample_points: np.ndarray, region_bounds: tuple[np.ndarray, np.ndarray]) -> None:
    """Section 2.4's load-time guard: the mesh region this material
    applies to (its bounding box) must be contained within the sample
    bounding box, checked once, before any interpolator call -- never
    discovered lazily via a silently-wrong extrapolated tensor."""
    sample_points = np.asarray(sample_points, dtype=float)
    sample_min = sample_points.min(axis=0)
    sample_max = sample_points.max(axis=0)
    region_min = np.asarray(region_bounds[0], dtype=float)
    region_max = np.asarray(region_bounds[1], dtype=float)

    if np.any(region_min < sample_min) or np.any(region_max > sample_max):
        raise CoverageError(
            f"material region bounds (min={region_min.tolist()}, max={region_max.tolist()}) are not "
            f"contained within the sample bounds (min={sample_min.tolist()}, max={sample_max.tolist()}) "
            "-- would require extrapolation (a units mismatch, e.g. a file authored in mm queried "
            "against a mesh in m, typically shows up here as roughly a factor of 1000)"
        )


@dataclass(frozen=True)
class SampledField:
    """Wraps either a structured grid or a scattered sample set behind one
    `evaluate(query_points) -> (M,C)` interface (Section 2) -- the shared
    primitive `material.regions` and `material.tensor_interpolation` both
    build their `MaterialModel`s on top of."""

    kind: str  # "structured" | "scattered"
    grid_axes: tuple[np.ndarray, np.ndarray, np.ndarray] | None
    grid_values: np.ndarray | None
    sample_points: np.ndarray | None
    sample_values: np.ndarray | None
    method: str

    @classmethod
    def structured(cls, grid_axes: tuple[np.ndarray, np.ndarray, np.ndarray], grid_values: np.ndarray) -> "SampledField":
        axes = tuple(np.asarray(a, dtype=float) for a in grid_axes)
        values = np.asarray(grid_values)
        if values.ndim == 3:
            values = values[..., None]
        return cls("structured", axes, values, None, None, "linear")

    @classmethod
    def scattered(cls, sample_points: np.ndarray, sample_values: np.ndarray, *, method: str = "linear") -> "SampledField":
        points = np.asarray(sample_points, dtype=float)
        values = np.asarray(sample_values)
        if values.ndim == 1:
            values = values[:, None]
        return cls("scattered", None, None, points, values, method)

    @property
    def n_channels(self) -> int:
        return (self.grid_values if self.kind == "structured" else self.sample_values).shape[-1]

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "structured":
            mins = np.array([axis.min() for axis in self.grid_axes])
            maxs = np.array([axis.max() for axis in self.grid_axes])
        else:
            mins = self.sample_points.min(axis=0)
            maxs = self.sample_points.max(axis=0)
        return mins, maxs

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        if self.kind == "structured":
            return interpolate_structured(self.grid_axes, self.grid_values, query_points)
        return interpolate_scattered(self.sample_points, self.sample_values, query_points, method=self.method)


def read_sample_file(path: str | Path) -> tuple[dict[str, str], np.ndarray]:
    """Reads a Section 5.2-style sample file: `# key: value` header
    comment lines, an optional column-name row, then whitespace/comma-
    separated numeric data rows. Shared by the director-file parser
    (`material.tensor_interpolation`) and the plain scalar/tensor field
    file loaders (`material.regions`) -- both files use the same on-disk
    convention, differing only in how many value columns follow x,y,z.
    Returns `(header, data)` with `data` shape `(N, n_columns)`."""
    header: dict[str, str] = {}
    rows: list[list[float]] = []
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                key, sep, value = line[1:].partition(":")
                if sep:
                    header[key.strip().lower()] = value.strip()
                continue
            first_token = line.replace(",", " ").split()[0]
            try:
                float(first_token)
            except ValueError:
                continue  # the column-name row, e.g. "x, y, z, nx, ny, nz"
            rows.append([float(v) for v in line.replace(",", " ").split()])
    if not rows:
        raise ValueError(f"{path}: no data rows found")
    return header, np.array(rows, dtype=float)
