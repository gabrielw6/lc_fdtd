"""validation -- Module 8: analytic references, shared checks, phase-gated
acceptance plan (docs/module8_validation_equations.md).

Derives no new field physics (Section 0). Depends on `extract`
(`energy_balance`, for the extended passivity check) and `pml`
(`PMLMaterial`, for the always-active passivity exemption) -- the only two
runtime dependencies this otherwise-standalone validation layer needs.
Sits above all other modules; nothing else imports it.
"""
from .analytic_microstrip import beta, eps_eff, z0
from .checks import (
    ValidationError,
    assert_passive,
    assert_reciprocal,
    assert_reduction,
    assert_symmetric,
    estimate_convergence_order,
    recommended_tolerance_from_pml,
)
from .gates import ConvergenceResult, GateReport, run_convergence_study, run_phase_gate

__all__ = [
    "eps_eff",
    "z0",
    "beta",
    "ValidationError",
    "assert_symmetric",
    "assert_passive",
    "assert_reciprocal",
    "estimate_convergence_order",
    "assert_reduction",
    "recommended_tolerance_from_pml",
    "GateReport",
    "run_phase_gate",
    "ConvergenceResult",
    "run_convergence_study",
]
