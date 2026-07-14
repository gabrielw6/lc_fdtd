"""Meshing module -- physical mesh-resolution sizing (docs/meshing_module_plan.md
Section 0.6/4). Independent of everything else in this package except
needing f, eps_bg, mu_bg as plain numbers -- no Gmsh dependency at all.
"""
from __future__ import annotations

from scipy import constants

_DEFAULT_ELEMENTS_PER_WAVELENGTH = 10  # conservative default for first-order Nedelec elements (Section 4)


def characteristic_length(
    frequency: float,
    background_eps: complex = constants.epsilon_0,
    background_mu: complex = constants.mu_0,
    elements_per_wavelength: int = _DEFAULT_ELEMENTS_PER_WAVELENGTH,
) -> float:
    """h_char = lambda / N_per_lambda (Section 4).

    `background_eps`/`background_mu` are **absolute** (SI) -- so
    `lambda = 1/(f*sqrt(eps_bg*mu_bg))`, no separate factor of c. That
    correctly reduces to the familiar `c/f` in vacuum (`eps_0*mu_0 = 1/c^2`)
    automatically; a literal `lambda=c/(f*sqrt(eps_bg*mu_bg))` would only be
    dimensionally correct for *relative* eps_bg/mu_bg and double-counts a
    factor of c^2 against absolute values.
    """
    if frequency <= 0:
        raise ValueError(f"frequency must be > 0, got {frequency!r}")
    if elements_per_wavelength <= 0:
        raise ValueError(f"elements_per_wavelength must be > 0, got {elements_per_wavelength!r}")

    refractive_index = (background_eps * background_mu) ** 0.5
    wavelength = 1.0 / (frequency * refractive_index)
    wavelength = wavelength.real if isinstance(wavelength, complex) else wavelength
    return float(wavelength) / elements_per_wavelength
