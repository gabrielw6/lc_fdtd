"""ports.port_operator -- surface term -> B_p, g_p (Section 5), and
de-embedding (Section 6) (docs/module4_ports_equations.md).

Section 5.1 carries its own honesty flag (the overall sign came out
opposite an earlier top-level sketch); the structural check this doc
offers independent of that sign question is B_p's symmetry. `build_B`
(post-review) now assembles a form that is symmetric *by construction*,
substituting the modal-admittance identity `h_m=Y_m*(x_hat x e_m)`
directly into the formula rather than relying on it holding to
floating-point/discretization precision -- see `build_B`'s own docstring
for the derivation. The real acceptance criterion remains the end-to-end
reciprocity/passivity gate (Modules 6+7), not this in-isolation structural
property, but a symmetric-by-construction `B` removes an entire class of
spurious `SystemSymmetryError`s that had nothing to do with an actual
tensor-index bug.

**Injection/extraction normalization fix (added post-review, the
energy-conservation/passivity review).** `build_B`'s boxed formula (like
`build_g`'s) is derived assuming every mode is normalized so `N_m=1`
exactly (Section 5.1's "propagating the Section 4.3 fix" paragraph) --
but `ports.mode_solver._normalize` only enforces `P_m=1` (Section 4.2),
and `N_m` (the *unconjugated* self-overlap Section 4.3's `project` divides
by) is a *different* bilinear form from `P_m` (the conjugated Poynting
power), empirically `N_m=2*P_m=2` for a lossless mode, not 1 (see
`ports.mode_solver._self_overlap`'s own docstring). `project` (extraction)
already carries the explicit `1/N_m` division the doc calls for; `build_B`
(the term that routes the *solved* field's own reflected/scattered
component -- see the derivation in this function's own docstring below --
back into the system matrix, i.e. the same quantity `project` extracts)
did not, until now -- an injection/extraction normalization mismatch
(extraction correctly divides by ~2, the matching injection-side term did
not), not a PML- or discretization-driven effect, and the actual cause of
the passivity-gate deficit this review starts from. `build_g`'s own
`a_m^{+,inc}` term is *not* extracted via `project` (it is a directly
given amplitude, not measured from a solved field) and, confirmed by
re-deriving the surface term component by component, needs no such
correction -- see `build_g`'s own docstring.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy import constants as _c

from mesh_interface import MeshInterface

from .mode_solver import PortMode

_B_SYMMETRY_TOL = 1e-6


class PortOperatorError(RuntimeError):
    """Raised for a caller-visible misuse of `build_g`/`deembed` (an
    excitation or offset referencing a port/mode this call was not given)."""


def build_B(port_modes: dict[str, list[PortMode]], mesh: MeshInterface, omega: float) -> sp.csr_matrix:
    """Section 5.1's boxed `B_p`, summed over every port's retained modes
    and assembled into one global-DOF-sized (Section 9), low-rank/local
    sparse matrix -- each port only touches the rows/cols of its own edges.

    **Symmetric-by-construction form (added post-review).** Section 5.1's
    boxed formula is `B_ij = -j*omega*mu0 * sum_m Y_m * overlap_e[i] *
    overlap_h[j]` -- algebraically symmetric only via the modal-admittance
    identity `h_m = Y_m*(x_hat x e_m)`, which gives `overlap_h[j] =
    Y_m*overlap_e[j]` *analytically*. That identity holds to within the
    discrete field reconstruction's own error, not exactly -- for a
    marginal (near-degenerate, coarsely-resolved) mode the gap was
    confirmed to make `B` wildly asymmetric (up to ~130% relative),
    tripping `solve.system.factor`'s symmetry check on an otherwise-fine
    system. Substituting the identity directly into the formula --
    `overlap_h -> Y_m*overlap_e` -- gives the equal-when-the-identity-holds
    form, further divided by each mode's own `N_m` (`mode.self_overlap`,
    see the module docstring's injection/extraction normalization note)
    since the boxed formula assumes `N_m=1`-normalized modes but the modes
    actually available are only `P_m=1`-normalized:

        B_ij = -j*omega*mu0 * sum_m (Y_m**2) * overlap_e[i] * overlap_e[j] / N_m

    Every term in that sum is `(scalar) * outer(v, v)`, which is symmetric
    for *any* `v` and *any* mode quality -- not just analytically, in the
    literal matrix this function builds, regardless of how marginal the
    contributing mode is; dividing by the scalar `N_m` does not change
    that. `overlap_h` is still computed and stored on `PortMode` (Section
    5.2's cache) for any other consumer (e.g. Module 7 extraction) that
    needs the un-substituted quantity; this function no longer reads it.

    **Per-port axial sign -- checked, no correction needed here (per-port
    axial-orientation review).** PORT_1's outward normal is `-x_hat`
    (matching Section 5.1's derivation directly) but PORT_2's is `+x_hat`
    (`n_out=-s_p*x_hat` generally, `s_p=cs.axial_sign`). Substituting the
    axial-direction-consistent modal-admittance relation
    `h_m=Y_m*(s_p*x_hat x e_m)` (the form `ports.mode_solver._h_t_on_triangle`
    and `_mode_integrals` are threaded with) into the surface term's own
    `n_out=-s_p*x_hat` gives *two* factors of `s_p`, which cancel exactly
    (`s_p**2=1`) -- the same cancellation that already makes `mode.Y` itself
    port-orientation-invariant (see `_mode_integrals`'s `x_cross_e`
    comment). An earlier version of this function multiplied by an
    additional bare `cs.axial_sign` on the theory that one factor survived
    the substitution; that was checked against the two-port
    reciprocity/passivity gate (`test/test_extract/test_reciprocity_uniform_line.py`)
    on a box-mode-safe (single-mode) port aperture and empirically confirmed
    **wrong** -- it produced a non-reciprocal, non-passive (`|S22|>1`)
    result, while omitting it (the form actually assembled below) gives a
    clean symmetric, passive S-matrix. Reverted; kept as a documented
    negative result per this doc series' policy of recording what was tried
    and rejected, not just what shipped.
    """
    n = mesh.n_edges
    rows: list[int] = []
    cols: list[int] = []
    vals: list[complex] = []

    for modes in port_modes.values():
        if not modes:
            continue
        cs = modes[0].cross_section
        global_edges = cs.global_edge_ids

        block = np.zeros((cs.n_edges, cs.n_edges), dtype=complex)
        for mode in modes:
            block += (
                -1j * omega * _c.mu_0 * (mode.Y**2) * np.outer(mode.overlap_e, mode.overlap_e) / mode.self_overlap
            )

        ge = global_edges.tolist()
        for i, gi in enumerate(ge):
            for j, gj in enumerate(ge):
                rows.append(gi)
                cols.append(gj)
                vals.append(block[i, j])

    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=complex).tocsr()


def build_g(
    port_modes: dict[str, list[PortMode]],
    excitation: dict[tuple[str, int], complex],
    mesh: MeshInterface,
    omega: float,
) -> np.ndarray:
    """Section 5.1's boxed `g_p`, summed over the driven `(port, mode)`
    pairs in `excitation` (Section 5.3: `mode` is 1-indexed, matching "the
    dominant mode (n=1)" convention -- `modes[0]` is mode 1).

    **No `axial_sign` correction needed here (checked, per-port
    axial-orientation review) -- same reasoning as `build_B`'s own
    docstring**: substituting `h_m=Y_m*(s_p*x_hat x e_m)` into the surface
    term's `n_out=-s_p*x_hat` reduces to `Y_m*overlap_e` with the two
    factors of `s_p` cancelling exactly. Confirmed empirically against the
    two-port reciprocity/passivity gate.

    **No `1/N_m` correction needed here either (checked, the energy-
    conservation/passivity review -- contrast with `build_B`'s own fix).**
    The surface term splits into a piece proportional to the *known*
    incident amplitude `a_m^{+,inc}` (this function, `g_p`) and a piece
    proportional to the *unknown* solved-field coefficients (`build_B`).
    Only the second piece is ever measured back out via Section 4.3's
    `project`/`1/N_m` division -- `a_m^{+,inc}` is a directly given
    number in the same units `mode.e_t`/`h_t` already use (one copy of the
    `P_m=1`-normalized mode), not something extracted from a solved field,
    so it needs no additional `N_m` rescaling. Re-deriving Section 5.1's
    surface term component by component (tracking exactly where `a_m^-`
    -- expressed via `project`, hence `1/N_m` -- versus `a_m^+` -- given
    directly -- enters) confirms this split cleanly, and the passivity
    gate (`test/test_extract/test_reciprocity_uniform_line.py`) only
    needed `build_B`'s correction to reach `|S11|^2+|S21|^2~=1`."""
    n = mesh.n_edges
    g = np.zeros(n, dtype=complex)

    for (port_tag, mode_number), a_inc in excitation.items():
        if a_inc == 0:
            continue
        try:
            modes = port_modes[port_tag]
        except KeyError:
            raise PortOperatorError(f"excitation references port {port_tag!r}, not present in port_modes") from None
        if not (1 <= mode_number <= len(modes)):
            raise PortOperatorError(
                f"excitation references mode {mode_number} of port {port_tag!r}, which only has {len(modes)} retained"
            )
        mode = modes[mode_number - 1]
        cs = mode.cross_section
        contribution = -2j * omega * _c.mu_0 * mode.Y * a_inc * mode.overlap_e
        g[cs.global_edge_ids] += contribution

    return g


def deembed(S: np.ndarray, port_modes: dict[str, list[PortMode]], offsets: dict[str, float]) -> np.ndarray:
    """Section 6's per-`(port,mode)` phase/attenuation correction. `S`'s
    rows/columns are indexed by the flattened `(port, mode)` list obtained
    by iterating `port_modes` in its own (port, then mode) order -- the
    same order `S` must have been assembled in by its Module 7 caller."""
    phase_factors: list[complex] = []
    for port_tag, modes in port_modes.items():
        if not modes:
            continue
        try:
            d_p = offsets[port_tag]
        except KeyError:
            raise PortOperatorError(f"no de-embedding offset supplied for port {port_tag!r}") from None
        for mode in modes:
            phase_factors.append(complex(np.exp(mode.gamma * d_p)))

    phase = np.asarray(phase_factors, dtype=complex)
    if phase.shape[0] != S.shape[0] or phase.shape[0] != S.shape[1]:
        raise PortOperatorError(
            f"S has shape {S.shape} but {phase.shape[0]} (port,mode) pairs were found in port_modes -- "
            "S's indexing does not match the flattened port_modes order"
        )
    return S * phase[:, None] * phase[None, :]
