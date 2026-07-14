"""docs/module8_validation_equations.md Section 3 (convergence-study
methodology) and Section 4 (phase-gated test architecture), tested on
cases with a known answer (Section 6 build step 3: "tested on a case with
a known analytic answer so the observed-order computation itself can be
checked").
"""
import pytest

from validation.gates import GateReport, run_convergence_study, run_phase_gate

# --- Section 3: run_convergence_study ---


def test_run_convergence_study_recovers_known_first_order():
    def build_fn(h):
        return h

    def quantity_fn(h):
        return 1.0 + 3.0 * h  # linear in h -> first-order convergence

    result = run_convergence_study(build_fn, [0.1, 0.05, 0.025, 0.0125], quantity_fn)
    assert result.observed_order == pytest.approx(1.0, abs=1e-9)
    assert result.values == [pytest.approx(1.0 + 3.0 * h) for h in [0.1, 0.05, 0.025, 0.0125]]


def test_run_convergence_study_recovers_known_second_order():
    def build_fn(h):
        return h

    def quantity_fn(h):
        return 1.0 + 3.0 * h**2  # quadratic in h -> second-order convergence

    result = run_convergence_study(build_fn, [0.1, 0.05, 0.025], quantity_fn)
    assert result.observed_order == pytest.approx(2.0, abs=1e-9)


def test_run_convergence_study_rejects_non_decreasing_levels():
    with pytest.raises(ValueError):
        run_convergence_study(lambda h: h, [0.05, 0.1], lambda h: h)


def test_run_convergence_study_rejects_inconsistent_ratio():
    with pytest.raises(ValueError):
        run_convergence_study(lambda h: h, [0.1, 0.05, 0.01], lambda h: h)  # ratios 2, 5


def test_run_convergence_study_rejects_too_few_levels():
    with pytest.raises(ValueError):
        run_convergence_study(lambda h: h, [0.1], lambda h: h)


# --- Section 4.2: the phase-gate mechanism ---


def test_run_phase_gate_passes_with_all_checks_satisfied():
    results = {
        "S_dominant": [[0.1, 0.2], [0.2, 0.1]],
        "beta_fem": 100.0,
        "beta_analytic": 100.5,
        "beta_tol": 0.01,
        "S_energy": {("PORT_1", 1, "PORT_1"): 0.6 + 0j, ("PORT_2", 1, "PORT_1"): 0.8 + 0j},
        "excitation_port": "PORT_1",
        "n_modes": 1,
        "convergence_errors": [0.08, 0.04, 0.02],
        "convergence_ratio": 2.0,
    }
    report = run_phase_gate(1, results)
    assert isinstance(report, GateReport)
    assert report.phase == 1
    assert report.passed
    assert report.failures == []
    assert report.skipped == []


def test_run_phase_gate_reports_skipped_checks_for_missing_keys():
    report = run_phase_gate(1, {})
    assert report.passed  # nothing failed -- nothing was checked either
    assert report.failures == []
    assert len(report.skipped) == 4


def test_run_phase_gate_fails_on_broken_reciprocity():
    results = {"S_dominant": [[0.1, 0.9], [0.2, 0.1]]}
    report = run_phase_gate(1, results)
    assert not report.passed
    assert len(report.failures) == 1
    assert "reciprocity" in report.failures[0]


def test_run_phase_gate_fails_on_broken_analytic_beta():
    results = {"beta_fem": 150.0, "beta_analytic": 100.0, "beta_tol": 0.01}
    report = run_phase_gate(1, results)
    assert not report.passed
    assert "beta" in report.failures[0]


def test_run_phase_gate_unknown_phase_raises():
    with pytest.raises(ValueError):
        run_phase_gate(99, {})


def test_run_phase_gate_phase2_reduction_check():
    ok_results = {"reduced_result": [1.0, 2.0, 3.0], "reference_result": [1.0, 2.0, 3.0]}
    assert run_phase_gate(2, ok_results).passed

    bad_results = {"reduced_result": [1.0, 2.0, 3.0], "reference_result": [1.0, 2.0, 9.0]}
    assert not run_phase_gate(2, bad_results).passed
