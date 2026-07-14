"""validation.gates -- the phase-gated test architecture (docs/module8_validation_equations.md
Section 4) and the mesh h-refinement convergence-study methodology (Section 3).

This module owns no FEM-running orchestration (Section 0: "does not own any
module's own internals") -- `run_phase_gate` runs Section 2's checks
against already-computed results the caller supplies, and
`run_convergence_study` drives a caller-supplied `build_fn`/`quantity_fn`
pair rather than knowing anything about geometry/mesh/materials itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .checks import ValidationError, assert_passive, assert_reciprocal, assert_reduction, estimate_convergence_order


@dataclass
class GateReport:
    """Section 8's contract, plus `skipped` (a small, documented
    extension): a check whose required `results` keys are absent is
    reported as skipped, not silently treated as passed and not treated
    as a hard failure either -- a gate evaluated against partial results
    should say so, not claim more than it checked."""

    phase: int
    passed: bool
    failures: list[str]
    skipped: list[str] = field(default_factory=list)


def _run_reciprocity(results: dict[str, Any], failures: list[str], skipped: list[str]) -> None:
    if "S_dominant" not in results:
        skipped.append("reciprocity (missing 'S_dominant')")
        return
    try:
        assert_reciprocal(results["S_dominant"], results.get("reciprocity_tol", 1e-6))
    except ValidationError as exc:
        failures.append(f"reciprocity: {exc}")


def _run_analytic_beta(results: dict[str, Any], failures: list[str], skipped: list[str]) -> None:
    if not {"beta_fem", "beta_analytic"} <= results.keys():
        skipped.append("analytic beta comparison (missing 'beta_fem'/'beta_analytic')")
        return
    fem, analytic = results["beta_fem"], results["beta_analytic"]
    tol = results.get("beta_tol", 1e-2)  # relative; Section 1.4's dispersion caveat -- not machine precision
    rel = abs(fem - analytic) / max(1.0, abs(analytic))
    if rel > tol:
        failures.append(
            f"dominant-mode beta mismatch vs. Hammerstad-Jensen: relative error {rel!r} > tol {tol!r}"
        )


def _run_energy_conservation(results: dict[str, Any], failures: list[str], skipped: list[str]) -> None:
    if not {"S_energy", "excitation_port", "n_modes"} <= results.keys():
        skipped.append("extended energy conservation (missing 'S_energy'/'excitation_port'/'n_modes')")
        return
    try:
        assert_passive(
            results["S_energy"],
            results.get("energy_tol", 1e-2),
            excitation_port=results["excitation_port"],
            n_modes=results["n_modes"],
        )
    except ValidationError as exc:
        failures.append(f"energy conservation: {exc}")


def _run_h_convergence(results: dict[str, Any], failures: list[str], skipped: list[str]) -> None:
    if not {"convergence_errors", "convergence_ratio"} <= results.keys():
        skipped.append("h-convergence (missing 'convergence_errors'/'convergence_ratio')")
        return
    order = estimate_convergence_order(results["convergence_errors"], results["convergence_ratio"])
    expected = results.get("expected_order", 1.0)  # Section 3 step 4: lowest-order Whitney elements
    order_tol = results.get("order_tol", 0.5)
    if abs(order - expected) > order_tol:
        failures.append(
            f"h-convergence: observed order {order!r} not consistent with expected {expected!r} "
            f"(tol {order_tol!r})"
        )


def _run_reduction(results: dict[str, Any], failures: list[str], skipped: list[str]) -> None:
    if not {"reduced_result", "reference_result"} <= results.keys():
        skipped.append("reduction check (missing 'reduced_result'/'reference_result')")
        return
    try:
        assert_reduction(results["reduced_result"], results["reference_result"], results.get("reduction_tol", 1e-9))
    except ValidationError as exc:
        failures.append(f"reduction: {exc}")


def _check_phase1(results: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Section 4.1: analytic beta/Z0, reciprocity, |S11|^2+|S21|^2~1
    (via the extended sum, Section 2.2), h-convergence."""
    failures: list[str] = []
    skipped: list[str] = []
    _run_reciprocity(results, failures, skipped)
    _run_analytic_beta(results, failures, skipped)
    _run_energy_conservation(results, failures, skipped)
    _run_h_convergence(results, failures, skipped)
    return failures, skipped


def _check_phase2(results: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Section 4.1: reduction to Phase 1 (the primary structural check --
    layered-dielectric/quadrature-sensitivity comparisons are numeric
    studies this generic gate cannot fabricate without a caller-supplied
    reference, so they are represented the same way via
    'reduced_result'/'reference_result')."""
    failures: list[str] = []
    skipped: list[str] = []
    _run_reduction(results, failures, skipped)
    return failures, skipped


def _check_phase3(results: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Section 4.1: axis-aligned analytic check (represented as a
    reduction against the analytic reference) plus reciprocity/passivity
    as invariants for rotated axes."""
    failures: list[str] = []
    skipped: list[str] = []
    _run_reduction(results, failures, skipped)
    _run_reciprocity(results, failures, skipped)
    if "eps_tensor" in results:
        try:
            assert_passive(results["eps_tensor"], results.get("passivity_tol", 1e-9), material=results.get("material"))
        except ValidationError as exc:
            failures.append(f"passivity: {exc}")
    else:
        skipped.append("tensor passivity (missing 'eps_tensor')")
    return failures, skipped


def _check_phase4(results: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Section 4.1: module-boundary contract (uniform director = Phase 3
    tensor, bit-for-bit -- a reduction check), reciprocity/passivity, and
    the extended energy conservation with mode-conversion (Module 7
    Section 5). The symmetry-forbidden-conversion test (Module 7:
    in-plane director => exact zero conversion) is a specific numeric
    assertion this generic gate does not fabricate; run it directly via
    `extract`'s own S-parameter output in the calling test, same as
    Module 7's own test suite already does."""
    failures: list[str] = []
    skipped: list[str] = []
    _run_reduction(results, failures, skipped)
    _run_reciprocity(results, failures, skipped)
    _run_energy_conservation(results, failures, skipped)
    return failures, skipped


_PHASE_CHECKERS: dict[int, Callable[[dict[str, Any]], tuple[list[str], list[str]]]] = {
    1: _check_phase1,
    2: _check_phase2,
    3: _check_phase3,
    4: _check_phase4,
}


def run_phase_gate(phase: int, results: dict[str, Any]) -> GateReport:
    """Section 4.2's blocking mechanism, operationalized: run whichever of
    Section 2's checks apply to `phase` against already-computed
    `results`. A check whose required keys are missing from `results` is
    recorded in `GateReport.skipped`, not silently treated as passing."""
    try:
        checker = _PHASE_CHECKERS[phase]
    except KeyError:
        raise ValueError(f"unknown phase {phase!r}; expected one of {sorted(_PHASE_CHECKERS)}") from None

    failures, skipped = checker(results)
    return GateReport(phase=phase, passed=(len(failures) == 0), failures=failures, skipped=skipped)


@dataclass
class ConvergenceResult:
    levels: list[float]
    values: list[float]
    observed_order: float


def run_convergence_study(
    build_fn: Callable[[float], Any], refinement_levels: Sequence[float], quantity_fn: Callable[[Any], float]
) -> ConvergenceResult:
    """Section 3's methodology. `refinement_levels` are characteristic
    element sizes, strictly decreasing (finer as the sequence progresses,
    Section 3 step 1's "each roughly halving the characteristic element
    size") -- `build_fn(level)` produces a result object at that
    resolution, `quantity_fn(result)` extracts the scalar being tracked
    (e.g. dominant-mode beta or Z0 at a fixed frequency). The observed
    order is estimated from successive differences (Section 2.4), which
    also recovers the exact-reference case if a caller instead supplies
    `quantity_fn` returning `value - exact` at each level."""
    levels = list(refinement_levels)
    if len(levels) < 2:
        raise ValueError("need at least two refinement levels to estimate a convergence order")
    if any(levels[i] <= levels[i + 1] for i in range(len(levels) - 1)):
        raise ValueError("refinement_levels must be strictly decreasing (finer element size later in the list)")

    values = [float(quantity_fn(build_fn(level))) for level in levels]
    diffs = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    ratios = [levels[i] / levels[i + 1] for i in range(len(levels) - 1)]
    if any(abs(r - ratios[0]) > 1e-6 * ratios[0] for r in ratios):
        raise ValueError(
            "refinement_levels do not have a constant ratio between consecutive levels -- "
            "estimate_convergence_order assumes one fixed ratio (Section 2.4)"
        )

    order = estimate_convergence_order(diffs, ratios[0])
    return ConvergenceResult(levels=levels, values=values, observed_order=order)
