"""validation.analytic_microstrip -- Hammerstad-Jensen microstrip formulas
(docs/module8_validation_equations.md Section 1), the actual reference
this whole project's Phase 1 gate has meant, unstated, since the first
design conversation. Quasi-static (no frequency dependence in eps_eff);
Section 1.4's dispersion caveat applies at the swept band's high end.

Verified against Steer's worked example (Section 1.5) before being trusted
to grade anything else -- see test/test_validation/test_analytic_microstrip.py.
"""
from __future__ import annotations

import numpy as np
from scipy import constants as _c


def eps_eff(eps_r: float, w: float, h: float) -> float:
    """Section 1.1's boxed effective permittivity."""
    u = w / h
    a = (
        1.0
        + (1.0 / 49.0) * np.log((u**4 + (u / 52.0) ** 2) / (u**4 + 0.432))
        + (1.0 / 18.7) * np.log(1.0 + (u / 18.1) ** 3)
    )
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3.0)) ** 0.053
    return (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 10.0 * h / w) ** (-a * b)


def z0(eps_r: float, w: float, h: float) -> float:
    """Section 1.2's boxed characteristic impedance -- a single continuous
    formula (via F1) valid for any w/h, no piecewise branching."""
    ee = eps_eff(eps_r, w, h)
    F1 = 6.0 + (2.0 * np.pi - 6.0) * np.exp(-((30.666 * h / w) ** 0.7528))
    z0_air = 60.0 * np.log(F1 * h / w + np.sqrt(1.0 + (2.0 * h / w) ** 2))
    return z0_air / np.sqrt(ee)


def beta(eps_r: float, w: float, h: float, omega: float) -> float:
    """Section 1.3: beta(omega) = k0*sqrt(eps_eff) = omega*sqrt(mu0*eps0*eps_eff)."""
    ee = eps_eff(eps_r, w, h)
    return omega * np.sqrt(_c.mu_0 * _c.epsilon_0 * ee)
