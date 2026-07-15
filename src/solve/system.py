"""solve.system -- PEC-elimination DOF restriction, complex-symmetric
factorization, and multi-RHS reuse (docs/module6_solve_sweep_equations.md
Sections 4-5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import SuperLU, splu

_SYMMETRY_TOL = 1e-6  # see factor()'s docstring: tightened back down (was
# 0.3) now that `ports.port_operator.build_B` assembles B_p symmetric *by
# construction* (Y_m**2 * outer(overlap_e, overlap_e), not
# Y_m * outer(overlap_e, overlap_h) -- the latter was only symmetric via a
# modal-admittance identity that holds analytically, not to floating-point
# precision, and was confirmed to produce up to ~130% relative asymmetry
# for a marginal mode). K_int/M_int (Module 3) are exactly symmetric to
# ~1e-9 and B_p is now exactly symmetric to machine precision, so 1e-6 is
# tight enough to catch a genuine transposed-tensor-index bug while still
# tolerating ordinary floating-point roundoff in the assembly/reduction.
_PIVOT_TOL = 1e-12


class SolveSingularityError(RuntimeError):
    """Raised when the factorization reports a near-zero pivot (Section
    5.4) -- signals a genuine resonance of the truncated domain or an
    under-absorbing PML, not something to accept silently."""


class SystemSymmetryError(RuntimeError):
    """Raised when the reduced system fails to be complex-symmetric
    (CLAUDE.md invariant 3) -- likely a transposed tensor index upstream
    (Module 2/3/4), not a caller mistake, but see `factor`'s `components`
    parameter for a diagnostic that names which contributing term (K/M vs
    B) actually broke symmetry rather than assuming which one."""


def build_restriction(pec_dofs: set[int], n_edges: int) -> sp.csr_matrix:
    """Section 4.1: the selection matrix `R` in `{0,1}^(Nf x Ne)` -- row
    `i` has a single 1 in the column of the `i`-th free DOF, in ascending
    order (a fixed, deterministic order, frequency- and bias-state-
    independent per Section 4.4, so this is built once and reused)."""
    free = [i for i in range(n_edges) if i not in pec_dofs]
    n_free = len(free)
    rows = np.arange(n_free)
    cols = np.array(free, dtype=np.int64)
    data = np.ones(n_free)
    return sp.csr_matrix((data, (rows, cols)), shape=(n_free, n_edges))


def reduce_system(A: sp.spmatrix, b: np.ndarray, R: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Section 4.2: `A_ff = R A R^T`, `b_f = R b`. Section 4.3: no RHS
    correction term is needed -- the essential BC enforces `a_c=0`
    exactly, and `g_p`'s PEC-adjacent entries are already structurally
    zero (Module 4 Section 3.5's test-space exclusion), so eliminating
    rows/columns is the complete step."""
    A_ff = (R @ A @ R.T).tocsr()
    b_f = R @ b
    return A_ff, b_f


def recover_solution(a_f: np.ndarray, R: sp.csr_matrix) -> np.ndarray:
    """Section 4.2: `a = R^T a_f` -- zero-pads the reduced solution back
    into the full DOF space at the constrained indices, exactly (Section
    4.3)."""
    return R.T @ a_f


@dataclass
class Factorization:
    """Wraps a `scipy.sparse.linalg.SuperLU` factorization (Section 5.2's
    "acceptable fallback": a generic asymmetric sparse LU gives the same
    correct answer as a dedicated complex-symmetric solver on a genuinely
    complex-symmetric matrix, since it does not assume symmetry rather
    than assuming the wrong kind -- just without the efficiency
    benefit)."""

    _lu: SuperLU

    def solve(self, b_f: np.ndarray) -> np.ndarray:
        return self._lu.solve(b_f)


def factor(A_ff: sp.spmatrix, *, components: dict[str, sp.spmatrix] | None = None) -> Factorization:
    """Section 5.1's complex-symmetric invariant, checked (not assumed);
    Section 5.2's factorization; Section 5.4's nonsingularity check.

    `components` (optional): the same-shape, already-reduced pieces that
    sum to `A_ff` (e.g. `{"K-k0^2*M": ..., "B (port operator)": ...}`) --
    used *only* to build a more useful diagnostic if the symmetry check
    below fails, by reporting each piece's own residual so the message
    names which term actually broke symmetry instead of asserting it must
    be a transposed tensor index. Omit it and `factor` still works exactly
    as before, just with a less specific error message on failure.

    `_SYMMETRY_TOL=1e-6` (tightened from an earlier 0.3): `K_int`/`M_int`
    (Module 3) are exactly complex-symmetric to ~1e-9, and `B_p`
    (`ports.port_operator.build_B`) is now assembled symmetric *by
    construction* (see that function's docstring) rather than relying on
    an identity that only holds analytically -- so ordinary floating-point
    roundoff is the only thing this tolerance needs to absorb, and a
    genuine transposed-tensor-index bug still produces a residual many
    orders of magnitude larger than this, not a borderline percentage."""
    A_csc = A_ff.tocsc()

    residual = float(np.abs(A_csc - A_csc.T).max()) if A_csc.nnz else 0.0
    scale = max(1.0, float(np.abs(A_csc).max()) if A_csc.nnz else 1.0)
    if residual > _SYMMETRY_TOL * scale:
        detail = ""
        if components:
            per_term = []
            for name, term in components.items():
                term_csc = term.tocsc()
                term_residual = float(np.abs(term_csc - term_csc.T).max()) if term_csc.nnz else 0.0
                per_term.append(f"{name}: max|term-term^T|={term_residual!r}")
            detail = " [" + "; ".join(per_term) + "]"
        raise SystemSymmetryError(
            f"A_ff is not complex-symmetric (max |A-A^T|={residual!r}, scale {scale!r}){detail} -- "
            "Section 5.1's invariant failed; check the per-term residuals above (when given) to "
            "localize which contributing term broke symmetry"
        )

    try:
        lu = splu(A_csc)
    except RuntimeError as exc:
        # SuperLU raises directly (rather than returning a factorization
        # with an inspectable zero pivot) for an exactly-singular matrix.
        raise SolveSingularityError(
            f"factorization failed: {exc} -- a genuine resonance of the truncated domain or an "
            "under-absorbing PML (Section 5.4), not a numerical fluke to ignore"
        ) from exc
    diag_u = np.abs(lu.U.diagonal())
    min_pivot = float(diag_u.min()) if diag_u.size else 0.0
    if diag_u.size == 0 or min_pivot < _PIVOT_TOL * scale:
        raise SolveSingularityError(
            f"near-zero pivot in factorization (min |diag(U)|={min_pivot!r}, scale {scale!r}) -- "
            "a genuine resonance of the truncated domain or an under-absorbing PML (Section "
            "5.4), not a numerical fluke to ignore"
        )
    return Factorization(lu)


def solve_with_factorization(fact: Factorization, b_f: np.ndarray) -> np.ndarray:
    """Section 5.3: reuse one factorization across every excitation's RHS
    at a given frequency."""
    return fact.solve(b_f)
