"""material.tensor_interpolation -- Phase 4, the LC director path
(docs/module2_material_equations.md Section 4). The only module that reads
a director file; owns only the conversion to eps_r(r) and its
interpolation -- the director *physics* that produced the file is out of
scope here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import MaterialModel, MaterialPassivityError, assemble_symmetric_tensor
from .interpolation import SampledField, check_coverage, read_sample_file

RegionBounds = tuple[np.ndarray, np.ndarray]

_NORM_TOLERANCE = 0.1  # Section 4.2: |n| outside [1-tol, 1+tol] is malformed, not round-off


class DirectorFieldError(RuntimeError):
    """Raised on a malformed director file or input (Section 4.2, 5.2)."""


def _build_lc_tensor_components(director_points: np.ndarray, director_vectors: np.ndarray, eps_perp: complex, eps_parallel: complex) -> np.ndarray:
    """Section 4.1: eps_r = eps_perp*I + (eps_parallel-eps_perp)*n n^T,
    computed once, here, at the input sample points -- never re-derived
    from n downstream (Section 4.8: the raw vector is discarded
    immediately after this call)."""
    norms = np.linalg.norm(director_vectors, axis=1)
    bad = (norms < 1.0 - _NORM_TOLERANCE) | (norms > 1.0 + _NORM_TOLERANCE)
    if np.any(bad):
        idx = int(np.argwhere(bad).ravel()[0])
        raise DirectorFieldError(
            f"{int(bad.sum())} director sample(s) have |n| outside [{1 - _NORM_TOLERANCE}, {1 + _NORM_TOLERANCE}] "
            f"(e.g. index {idx}: |n|={norms[idx]!r}) -- malformed input, not just round-off"
        )
    n_hat = director_vectors / norms[:, None]

    delta_eps = eps_parallel - eps_perp
    nnT = np.einsum("ni,nj->nij", n_hat, n_hat)  # (N,3,3)
    eps_tensor = eps_perp * np.eye(3, dtype=complex)[None, :, :] + delta_eps * nnT

    N = eps_tensor.shape[0]
    components = np.empty((N, 6), dtype=complex)
    components[:, 0] = eps_tensor[:, 0, 0]
    components[:, 1] = eps_tensor[:, 0, 1]
    components[:, 2] = eps_tensor[:, 0, 2]
    components[:, 3] = eps_tensor[:, 1, 1]
    components[:, 4] = eps_tensor[:, 1, 2]
    components[:, 5] = eps_tensor[:, 2, 2]
    return components


def _check_lc_passivity(eps_perp: complex, eps_parallel: complex) -> None:
    """Section 4.6: passivity of the whole interpolated tensor field
    reduces, provably, to two scalar checks made once here -- never a
    per-point eigendecomposition for this material (Section 1.3's generic
    check is overridden below for exactly this reason)."""
    eps_perp_loss = -eps_perp.imag  # eps_perp'' per Section 4.3's eps_perp = eps_perp' - j*eps_perp''
    eps_parallel_loss = -eps_parallel.imag
    tol = 1e-12
    if eps_perp_loss < -tol or eps_parallel_loss < -tol:
        raise MaterialPassivityError(
            f"non-passive LC material constants: eps_perp''={eps_perp_loss!r}, "
            f"eps_parallel''={eps_parallel_loss!r} must both be >= 0 (Section 4.6)"
        )


class DirectorFieldMaterial(MaterialModel):
    """Phase 4 -- `eps_r(r)` built from a director field and `eps_perp`/
    `eps_parallel` (Section 4). Reuses `material.interpolation`'s C=6
    primitive (Section 4.4) -- the identical one `TensorFieldMaterial`
    (Phase 3) uses, per Section 4.7's module-boundary contract."""

    def __init__(
        self,
        director_points: np.ndarray,
        director_vectors: np.ndarray,
        eps_perp: complex,
        eps_parallel: complex,
        *,
        mu_r: complex = 1.0,
        region_bounds: RegionBounds | None = None,
        interpolation_method: str = "linear",
    ) -> None:
        eps_perp = complex(eps_perp)
        eps_parallel = complex(eps_parallel)
        _check_lc_passivity(eps_perp, eps_parallel)  # Section 4.6, once, here

        director_points = np.asarray(director_points, dtype=float)
        director_vectors = np.asarray(director_vectors, dtype=float)
        components = _build_lc_tensor_components(director_points, director_vectors, eps_perp, eps_parallel)

        if region_bounds is not None:
            check_coverage(director_points, region_bounds)  # Section 4.2's containment check

        self._field = SampledField.scattered(director_points, components, method=interpolation_method)
        self._eps_perp = eps_perp
        self._eps_parallel = eps_parallel
        self._mu_r = complex(mu_r)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        eps_perp: complex,
        eps_parallel: complex,
        *,
        mu_r: complex = 1.0,
        region_bounds: RegionBounds | None = None,
        interpolation_method: str = "linear",
    ) -> "DirectorFieldMaterial":
        """Section 5.2's director-file format: `# coordinate_frame:`,
        `# units:`, `# grid_type:` header fields (mandatory, checked here),
        then `x,y,z,nx,ny,nz` rows."""
        header, data = read_sample_file(path)
        missing = {"coordinate_frame", "units", "grid_type"} - header.keys()
        if missing:
            raise DirectorFieldError(f"{path}: director file missing required header field(s): {sorted(missing)}")
        if header["grid_type"] not in ("structured", "scattered"):
            raise DirectorFieldError(f"{path}: grid_type must be 'structured' or 'scattered', got {header['grid_type']!r}")
        if data.shape[1] != 6:
            raise DirectorFieldError(f"{path}: expected 6 columns (x,y,z,nx,ny,nz), got {data.shape[1]}")

        points, vectors = data[:, :3], data[:, 3:6]
        return cls(
            points, vectors, eps_perp, eps_parallel,
            mu_r=mu_r, region_bounds=region_bounds, interpolation_method=interpolation_method,
        )

    def _epsilon(self, points: np.ndarray) -> np.ndarray:
        return assemble_symmetric_tensor(self._field.evaluate(points))

    def _mu(self, points: np.ndarray) -> np.ndarray:
        return np.tile(self._mu_r * np.eye(3, dtype=complex), (points.shape[0], 1, 1))

    def _check_epsilon_passive(self, eps: np.ndarray) -> None:
        """Section 4.6: already proven (and checked, once) at construction
        from eps_perp''/eps_parallel'' alone -- the expensive generic
        per-point eigendecomposition (Section 1.3) has no place here."""

    def _check_mu_passive(self, mu: np.ndarray) -> None:
        pass  # mu_r is a real, positive scalar multiple of I -- trivially passive
