"""validation.checks -- shared generic correctness-check library
(docs/module8_validation_equations.md Section 2), consolidating patterns
already specified ad hoc across Modules 2-7 so they are implemented and
tested exactly once.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.sparse as sp

from extract import energy_balance
from pml import PMLMaterial

_DEFAULT_PASSIVITY_EXEMPTIONS: tuple[type, ...] = (PMLMaterial,)


class ValidationError(RuntimeError):
    """Raised by any `validation.checks` function when the checked
    quantity fails its stated invariant."""


def _frobenius_norm(M) -> float:
    if sp.issparse(M):
        M = M.tocoo()
        return float(np.sqrt(np.sum(np.abs(M.data) ** 2)))
    return float(np.linalg.norm(np.asarray(M), "fro"))


def assert_symmetric(M, tol: float = 1e-10) -> None:
    """Section 2.1: `||M-M^T||_F / ||M||_F < tol`. Machine-precision
    tolerance by default -- every use site (material tensors, Module 3's
    K/M, Module 4's B_p and S_tz=-T_zt^T) is an exact algebraic identity,
    not an approximate one, so a loose default would mask a real bug. A
    caller checking something with a known, documented imperfection
    (e.g. Module 4's B_p honesty flag) should pass an explicit, wider
    `tol`, not rely on a loosened default here."""
    diff = M - M.T
    denom = _frobenius_norm(M)
    numer = _frobenius_norm(diff)
    ratio = numer / denom if denom > 0 else numer
    if ratio >= tol:
        raise ValidationError(f"matrix not symmetric: ||M-M^T||_F/||M||_F={ratio!r} >= tol={tol!r}")


def assert_passive(
    tensor_or_S,
    tol: float = 1e-9,
    exceptions: Sequence[type] = (),
    *,
    material: object | None = None,
    excitation_port: str | None = None,
    n_modes: int | None = None,
) -> None:
    """Section 2.2, dispatched on `tensor_or_S`'s type:

    - `dict[(str,int,str), complex]` (Module 7's S-parameter convention):
      the *extended* energy-conservation sum (Module 7 Section 5) -- not
      the naive two-term dominant-only sum, which can legitimately read
      below 1 for a correct mode-converting structure. Requires
      `excitation_port` and `n_modes`.
    - `np.ndarray` (an evaluated `(...,3,3)` material tensor): eigenvalues
      of `-Im(tensor)` must be `>= -tol` (Module 2 Section 1.3's generic
      criterion). `PMLMaterial` is *always* exempted by default (Module 5
      Section 4's documented exception, wired in here so it is never an
      oversight to remember at each call site) -- pass `material=` (the
      `MaterialModel` instance the tensor came from) whenever checking a
      tensor that might have come from one; `exceptions` extends the
      (always-active) default exemption list with caller-specific types.
    """
    if isinstance(tensor_or_S, dict):
        if excitation_port is None or n_modes is None:
            raise ValueError("assert_passive on an S-parameter dict needs excitation_port and n_modes")
        balance = energy_balance(tensor_or_S, excitation_port, n_modes)
        if balance > 1.0 + tol:
            raise ValidationError(
                f"energy conservation violated: extended sum {balance!r} > 1+tol={1.0 + tol!r}"
            )
        return

    exempt_types = tuple(exceptions) + _DEFAULT_PASSIVITY_EXEMPTIONS
    if material is not None and isinstance(material, exempt_types):
        return

    tensor = np.asarray(tensor_or_S, dtype=complex)
    if tensor.ndim == 2:
        tensor = tensor[None, :, :]
    B = -tensor.imag
    min_eig = float(np.linalg.eigvalsh(B).min()) if B.size else 0.0
    if min_eig < -tol:
        raise ValidationError(
            f"non-passive tensor: min eigenvalue of -Im(tensor) is {min_eig!r} < -tol={-tol!r} "
            "(a positive imaginary eigenvalue is a gain medium)"
        )


def assert_reciprocal(S_dominant: np.ndarray, tol: float = 1e-9) -> None:
    """Section 2.3: `S_(p,1),(q,1) = S_(q,1),(p,1)`, scoped explicitly to
    `S^dominant` (a `(n_ports,n_ports)` or `(n_freq,n_ports,n_ports)`
    array) -- per Module 7 Section 3.3, full cross-mode reciprocity cannot
    be checked with the established single-mode excitation convention,
    and this function does not pretend otherwise by accepting anything
    beyond the dominant block."""
    S_dominant = np.asarray(S_dominant)
    residual = np.abs(S_dominant - np.swapaxes(S_dominant, -1, -2))
    scale = max(1.0, float(np.abs(S_dominant).max()) if S_dominant.size else 1.0)
    max_residual = float(residual.max()) if residual.size else 0.0
    if max_residual > tol * scale:
        raise ValidationError(
            f"S_dominant is not reciprocal (max |S-S^T|={max_residual!r} > tol*scale={tol * scale!r})"
        )


def estimate_convergence_order(errors: Sequence[float], refinement_ratio: float) -> float:
    """Section 2.4: `p_obs = ln(||e1||/||e2||) / ln(refinement_ratio)`,
    for successive errors (or successive differences, when the exact
    answer is unknown) at consecutive refinement levels. `errors` may
    have more than two entries (more than two refinement levels); the
    pairwise estimates from each consecutive pair are averaged into the
    single returned value, rather than requiring exactly two inputs."""
    errors = [float(e) for e in errors]
    if len(errors) < 2:
        raise ValueError("need at least two error/difference values to estimate a convergence order")
    orders = []
    for e1, e2 in zip(errors[:-1], errors[1:]):
        if e2 == 0:
            continue
        orders.append(np.log(abs(e1) / abs(e2)) / np.log(refinement_ratio))
    if not orders:
        raise ValueError("could not estimate a convergence order (every later error/difference is zero)")
    return float(np.mean(orders))


def assert_reduction(result_a, result_b, tol: float = 1e-9) -> None:
    """Section 2.5: a generic "configuration A matches configuration B"
    comparator, reused by every reduction check across this project
    (Phase 2->1, Phase 3->1, Phase 4->3, Module 5's sigma->0/kappa->1,
    Module 0's LC-to-substrate reduction) instead of a bespoke comparison
    at each call site. Accepts dense arrays or sparse matrices."""
    a = result_a.toarray() if sp.issparse(result_a) else np.asarray(result_a)
    b = result_b.toarray() if sp.issparse(result_b) else np.asarray(result_b)
    if a.shape != b.shape:
        raise ValidationError(f"shape mismatch in reduction check: {a.shape} vs {b.shape}")
    residual = np.abs(a - b)
    scale = max(1.0, float(np.abs(b).max()) if b.size else 1.0)
    max_residual = float(residual.max()) if residual.size else 0.0
    if max_residual > tol * scale:
        raise ValidationError(
            f"reduction check failed: max |a-b|={max_residual!r} > tol*scale={tol * scale!r}"
        )


def recommended_tolerance_from_pml(R0: float, factor: float = 10.0) -> float:
    """Section 2.6's concrete recommendation: reciprocity/energy-check
    tolerances should not be set tighter than the PML's own target
    reflection `R0` (Module 5 Section 2.4) -- a well-converged solve
    cannot do better than the residual reflection the truncation boundary
    itself permits. Returns `factor*R0` (default `10x` looser, within
    Section 2.6's suggested 5-10x range)."""
    return factor * R0
