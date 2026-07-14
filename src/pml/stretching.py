"""pml.stretching -- complex coordinate-stretching profile functions and
the Lambda tensor construction (docs/module5_pml_equations.md Section 2).

Top-only, single-axis PML (Module 0 Section 0.3): only `s_z` is ever
non-trivial in this geometry, so `lambda_tensor` defaults `s_x=s_y=1` --
the general 3-axis form (Section 2.1) is kept as the function's actual
signature so a future corner-graded PML needs no rewrite here, just a
different caller.
"""
from __future__ import annotations

import numpy as np
from scipy import constants as _c

_ETA_0 = float(np.sqrt(_c.mu_0 / _c.epsilon_0))  # ~376.730313668 ohm, free-space impedance


def sigma_max_for_R0(R0: float, n: int, thickness: float, eta0: float = _ETA_0) -> float:
    """Section 2.4's boxed result, derived from the round-trip amplitude
    attenuation e^(-2*eta0*sigma_max*d/(n+1)) == R0:

        sigma_max = -(n+1)*ln(R0) / (2*eta0*d)

    Takes no `omega` argument by construction -- Section 2.4's whole point
    is that the k0/(omega*eps0)=eta0 cancellation makes the required
    round-trip attenuation frequency-independent, so this is computed once
    per PML shell, not once per frequency point."""
    return -(n + 1) * np.log(R0) / (2.0 * eta0 * thickness)


def sigma_profile(xi: np.ndarray, thickness: float, sigma_max: float, n: int) -> np.ndarray:
    """Section 2.3: `sigma(xi) = sigma_max*(xi/d)^n`."""
    xi = np.asarray(xi, dtype=float)
    return sigma_max * (xi / thickness) ** n


def kappa_profile(xi: np.ndarray, thickness: float, kappa_max: float, n: int) -> np.ndarray:
    """Section 2.3: `kappa(xi) = 1 + (kappa_max-1)*(xi/d)^n`."""
    xi = np.asarray(xi, dtype=float)
    return 1.0 + (kappa_max - 1.0) * (xi / thickness) ** n


def s_z(
    xi: np.ndarray, omega: float, thickness: float, sigma_max: float, kappa_max: float, n: int
) -> np.ndarray:
    """Section 2.2: `s_z(omega,xi) = kappa(xi) - j*sigma(xi)/(omega*eps0)`."""
    sigma = sigma_profile(xi, thickness, sigma_max, n)
    kappa = kappa_profile(xi, thickness, kappa_max, n)
    return kappa - 1j * sigma / (omega * _c.epsilon_0)


def lambda_tensor(s_z: np.ndarray, s_x: complex = 1.0, s_y: complex = 1.0) -> np.ndarray:
    """Section 2.1's boxed general tensor
    `Lambda = diag(s_y*s_z/s_x, s_z*s_x/s_y, s_x*s_y/s_z)`, which for this
    geometry's `s_x=s_y=1` default collapses to `diag(s_z, s_z, 1/s_z)` --
    the two transverse axes scale *up* by `s_z`, the stretched axis itself
    scales by its *reciprocal* (the asymmetry that makes PML matched).
    `s_z`: `(M,)` complex array. Returns `(M,3,3)` complex."""
    s_z = np.asarray(s_z, dtype=complex)
    M = s_z.shape[0]
    lam = np.zeros((M, 3, 3), dtype=complex)
    lam[:, 0, 0] = s_y * s_z / s_x
    lam[:, 1, 1] = s_z * s_x / s_y
    lam[:, 2, 2] = s_x * s_y / s_z
    return lam
