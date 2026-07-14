"""geometry_builder.params -- GeometryParams: primary/secondary parameters,
validation, and derived quantities (docs/module0_geometry_builder_equations.md
Section 1).
"""
from __future__ import annotations

from dataclasses import dataclass


class GeometryParameterError(ValueError):
    """Raised when GeometryParams (or its derived quantities) fails
    validation (Section 1.3, Section 6) -- a bad parameter must never reach
    the CAD kernel."""


@dataclass(frozen=True)
class GeometryParams:
    """Section 1.1-1.2. `h_air`/`h_pml` default when omitted (`None`) --
    both are untuned placeholders per the doc's own admission, not values
    asserted correct in the abstract."""

    # --- primary, user-facing (Section 1.1) ---
    w: float
    L: float
    L_lc: float
    W_lc: float
    h_sub: float
    W_sub: float
    eps_r_substrate: float
    tan_delta_substrate: float = 0.0

    # --- secondary, defaulted (Section 1.2) ---
    h_air: float | None = None
    h_pml: float | None = None

    # --- mesh-resolution sizing for build() step 11 (not itself part of
    # the geometry) -- follows meshing.mesh_sizing's own convention: an
    # explicit reference_frequency, never derived or defaulted to zero ---
    reference_frequency: float = 0.0
    target_elements_per_wavelength: int = 10

    def __post_init__(self) -> None:
        if self.h_air is None:
            object.__setattr__(self, "h_air", 3.0 * self.h_sub)
        if self.h_pml is None:
            object.__setattr__(self, "h_pml", 0.5 * self.h_sub)
        _validate(self)


def _validate(p: GeometryParams) -> None:
    if p.w <= 0:
        raise GeometryParameterError(f"w must be > 0, got {p.w!r}")
    if not (0 < p.L_lc < p.L):
        raise GeometryParameterError(f"L_lc must satisfy 0 < L_lc < L, got L_lc={p.L_lc!r}, L={p.L!r}")
    if not (0 < p.W_lc < p.W_sub):
        raise GeometryParameterError(
            f"W_lc must satisfy 0 < W_lc < W_sub, got W_lc={p.W_lc!r}, W_sub={p.W_sub!r}"
        )
    if p.w > p.W_lc:
        raise GeometryParameterError(
            f"w must be <= W_lc (the trace must sit within the cavity's footprint), "
            f"got w={p.w!r}, W_lc={p.W_lc!r}"
        )
    if p.h_sub <= 0:
        raise GeometryParameterError(f"h_sub must be > 0, got {p.h_sub!r}")
    if p.h_air <= 0:
        raise GeometryParameterError(f"h_air must be > 0, got {p.h_air!r}")
    if p.h_pml <= 0:
        raise GeometryParameterError(f"h_pml must be > 0, got {p.h_pml!r}")
    if p.eps_r_substrate < 1:
        raise GeometryParameterError(f"eps_r_substrate must be >= 1, got {p.eps_r_substrate!r}")
    if p.reference_frequency <= 0:
        raise GeometryParameterError(f"reference_frequency must be > 0, got {p.reference_frequency!r}")


@dataclass(frozen=True)
class DerivedGeometry:
    """Derived quantities (Section 1.3), computed once from a validated
    `GeometryParams`."""

    x_c0: float
    x_c1: float
    y_lc0: float
    y_lc1: float
    y0_trace: float
    y1_trace: float
    z_gnd: float
    z_iface: float
    z_air_top: float
    z_pml_top: float


def derive(p: GeometryParams) -> DerivedGeometry:
    x_c0 = (p.L - p.L_lc) / 2.0
    x_c1 = (p.L + p.L_lc) / 2.0
    y_lc0 = (p.W_sub - p.W_lc) / 2.0
    y_lc1 = (p.W_sub + p.W_lc) / 2.0
    y0_trace = (p.W_sub - p.w) / 2.0
    y1_trace = (p.W_sub + p.w) / 2.0

    # Section 6, "Cavity margin": guaranteed by validation (0 < L_lc < L
    # together with centered placement), but asserted explicitly rather
    # than relying on the arithmetic to happen to work out -- this is what
    # keeps both port faces in the isotropic feed section the top-level
    # architecture doc's port design invariant needs.
    if not (x_c0 > 0 and x_c1 < p.L):
        raise GeometryParameterError(
            f"cavity does not leave a nonzero substrate margin at both ports: "
            f"x_c0={x_c0!r}, x_c1={x_c1!r}, L={p.L!r}"
        )
    # Section 6, "Trace footprint containment".
    if not (0 < y0_trace < y1_trace < p.W_sub):
        raise GeometryParameterError(
            f"trace footprint must lie strictly within the substrate width: "
            f"y0_trace={y0_trace!r}, y1_trace={y1_trace!r}, W_sub={p.W_sub!r}"
        )
    # Section 6, "Trace-within-cavity containment": both bounds asserted
    # explicitly, not just the width inequality that implies them -- a
    # centering bug could satisfy w<=W_lc while still placing the trace
    # off-center relative to the cavity.
    if not (y_lc0 <= y0_trace and y1_trace <= y_lc1):
        raise GeometryParameterError(
            f"trace footprint must lie within the LC cavity's width at every point along "
            f"the cavity's length: y0_trace={y0_trace!r}, y1_trace={y1_trace!r}, "
            f"y_lc0={y_lc0!r}, y_lc1={y_lc1!r}"
        )

    return DerivedGeometry(
        x_c0=x_c0,
        x_c1=x_c1,
        y_lc0=y_lc0,
        y_lc1=y_lc1,
        y0_trace=y0_trace,
        y1_trace=y1_trace,
        z_gnd=0.0,
        z_iface=p.h_sub,
        z_air_top=p.h_sub + p.h_air,
        z_pml_top=p.h_sub + p.h_air + p.h_pml,
    )
