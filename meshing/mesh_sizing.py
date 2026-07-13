"""Meshing module -- physical mesh-resolution sizing (docs/meshing_module_plan.md
Section 0.6/4). Independent of everything else in this package except
needing f, eps_bg, mu_bg as plain numbers -- no Gmsh dependency at all.
"""
from __future__ import annotations

from scipy import constants

from ..cavity import CavityMode

_DEFAULT_ELEMENTS_PER_WAVELENGTH = 10  # conservative default for first-order Nedelec elements (Section 4)


def characteristic_length(
    frequency: float,
    background_eps: complex = constants.epsilon_0,
    background_mu: complex = constants.mu_0,
    elements_per_wavelength: int = _DEFAULT_ELEMENTS_PER_WAVELENGTH,
) -> float:
    """h_char = lambda / N_per_lambda (Section 4).

    `background_eps`/`background_mu` are **absolute** (SI), matching Module
    1's own `epsilon_bg`/`mu_bg` convention (CLAUDE.md's absolute-vs-
    relative rule) -- so `lambda = 1/(f*sqrt(eps_bg*mu_bg))`, no separate
    factor of c. That correctly reduces to the familiar `c/f` in vacuum
    (`eps_0*mu_0 = 1/c^2`) automatically. The doc's literal
    `lambda=c/(f*sqrt(eps_bg*mu_bg))` is only dimensionally correct for
    *relative* eps_bg/mu_bg; Section 0.6 sources these values directly from
    `CavityMode.epsilon_bg`/`.mu_bg`, which are absolute -- so a literal
    transcription would double-count a factor of c^2 relative to the
    vacuum case. Same absolute-vs-relative bug class as elsewhere in this
    project (see CLAUDE.md's Conventions section), just caught here before
    it was ever implemented literally.
    """
    if frequency <= 0:
        raise ValueError(f"frequency must be > 0, got {frequency!r}")
    if elements_per_wavelength <= 0:
        raise ValueError(f"elements_per_wavelength must be > 0, got {elements_per_wavelength!r}")

    refractive_index = (background_eps * background_mu) ** 0.5
    wavelength = 1.0 / (frequency * refractive_index)
    wavelength = wavelength.real if isinstance(wavelength, complex) else wavelength
    return float(wavelength) / elements_per_wavelength


def characteristic_length_from_cavity(
    cavity_mode: CavityMode, elements_per_wavelength: int = _DEFAULT_ELEMENTS_PER_WAVELENGTH
) -> float:
    """Section 0.6's standard-shape path: f, eps_bg, mu_bg read directly
    from the `CavityMode`'s own `f0`/`epsilon_bg`/`mu_bg` attributes."""
    return characteristic_length(
        cavity_mode.f0, cavity_mode.epsilon_bg, cavity_mode.mu_bg, elements_per_wavelength
    )
