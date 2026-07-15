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
    form actually assembled below:

        B_ij = -j*omega*mu0 * sum_m (Y_m**2) * overlap_e[i] * overlap_e[j]

    Every term in that sum is `(scalar) * outer(v, v)`, which is symmetric
    for *any* `v` and *any* mode quality -- not just analytically, in the
    literal matrix this function builds, regardless of how marginal the
    contributing mode is. `overlap_h` is still computed and stored on
    `PortMode` (Section 5.2's cache) for any other consumer (e.g. Module 7
    extraction) that needs the un-substituted quantity; this function no
    longer reads it.
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
            block += -1j * omega * _c.mu_0 * (mode.Y**2) * np.outer(mode.overlap_e, mode.overlap_e)

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
    dominant mode (n=1)" convention -- `modes[0]` is mode 1)."""
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
