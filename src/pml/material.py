"""pml.material -- PMLMaterial, the MaterialModel implementation of the
complex-coordinate-stretched PML tensor (docs/module5_pml_equations.md
Section 3).
"""
from __future__ import annotations

import numpy as np

from material import MaterialModel

from .stretching import lambda_tensor, s_z as _s_z_profile, sigma_max_for_R0


class PMLMaterial(MaterialModel):
    """Section 3.1: `eps_r^PML = Lambda @ eps_r,bg`, `mu_r^PML = Lambda @
    mu_r,bg` -- wraps a general background `MaterialModel` (dependency
    injection, Section 3.1) even though this geometry's background is
    always `AIR` (`ConstantMaterial(eps_r=1.0)`), so both reduce to
    `Lambda` itself and `eps_r^PML == mu_r^PML` exactly (Section 3.1's
    genuine, not approximate, simplification).

    Constructed fresh per frequency (Section 3.4): `omega` is baked in at
    construction; `epsilon`/`mu` still take only `points`, so Module 2's
    interface is untouched."""

    def __init__(
        self,
        background: MaterialModel,
        omega: float,
        z_air_top: float,
        thickness: float,
        R0: float = 1e-6,
        n: int = 2,
        kappa_max: float = 1.0,
    ) -> None:
        self._background = background
        self._omega = float(omega)
        self._z_air_top = float(z_air_top)
        self._thickness = float(thickness)
        self._n = int(n)
        self._kappa_max = float(kappa_max)
        self._sigma_max = sigma_max_for_R0(R0, n, thickness)

    def _lambda(self, points: np.ndarray) -> np.ndarray:
        xi = points[:, 2] - self._z_air_top  # Section 2.5: depth into the PML, from its own z-column
        s_z = _s_z_profile(xi, self._omega, self._thickness, self._sigma_max, self._kappa_max, self._n)
        return lambda_tensor(s_z)

    def _epsilon(self, points: np.ndarray) -> np.ndarray:
        return self._lambda(points) @ self._background.epsilon(points)

    def _mu(self, points: np.ndarray) -> np.ndarray:
        return self._lambda(points) @ self._background.mu(points)

    def _check_epsilon_passive(self, eps: np.ndarray) -> None:
        """Module 2 Section 1.3's documented exception (Module 5 Section
        4): `Lambda_zz=1/s_z` has `Im>=0` by construction (Section 4.1),
        which is what makes the tensor matched, not a sign error -- the
        generic per-tensor passivity check does not apply here. The real
        correctness criterion is the wave-attenuation/reflection argument
        (Section 2.4/5.1), not this per-call check."""

    def _check_mu_passive(self, mu: np.ndarray) -> None:
        """Same exemption as `_check_epsilon_passive` -- `mu_r^PML` is the
        identical `Lambda` tensor here (Section 3.1)."""
