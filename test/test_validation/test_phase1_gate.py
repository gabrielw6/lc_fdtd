"""docs/module8_validation_equations.md Section 6 build step 5 / Section 7:
"Run Phase 1's full gate end-to-end" -- the first point in this whole
project where `run_phase_gate` is exercised against real Module 0-7
output, not a hand-built stand-in. Uniform, lossless, no-LC microstrip
(Phase 1 of the top-level plan).

Uses `n_modes=1` and a trivial PML, the same choices Module 6/7's own test
suites already established and documented the rationale for (a graded PML
profile needs a much finer mesh than this fast-test config affords; a
trivial PML is a PEC-lidded cavity, not a matched termination, so the
energy-conservation check specifically is not meaningful here and is
therefore left out of `results` -- Module 7's own test suite already
covers that check honestly with the infrastructure it has). Reciprocity
and the analytic-beta comparison, both real and meaningful with this
fixture, are what actually get exercised end-to-end here.
"""
import numpy as np
import pytest

from material import ConstantMaterial

_PARAMS_KWARGS = dict(
    w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
    eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
    reference_frequency=25e9, target_elements_per_wavelength=6,
)


@pytest.fixture(scope="module")
def phase1_gate_results():
    pytest.importorskip("gmsh")
    from extract import assemble_sweep_dataset, deembed, raw_s_parameters
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import MaterialAssembly, load_material_spec
    from mesh_interface import MeshInterface
    from solve import run_sweep
    from validation.analytic_microstrip import beta as beta_analytic

    params = GeometryParams(**_PARAMS_KWARGS)
    mesh_handle, material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    base_assembly = load_material_spec(geometry_stub=material_stub)
    tag_to_model = dict(base_assembly._tag_to_model)
    tag_to_model["LC"] = ConstantMaterial(eps_r=params.eps_r_substrate)  # Phase 1: no LC
    materials = MaterialAssembly(tag_to_model)

    pml_params = dict(
        background=ConstantMaterial(eps_r=1.0),
        z_air_top=params.h_sub + params.h_air,
        thickness=params.h_pml,
        R0=1.0, n=2, kappa_max=1.0,  # trivial -- see module docstring
    )
    omega = 2 * np.pi * params.reference_frequency
    sweep_results = run_sweep(mesh, materials, ["PORT_1", "PORT_2"], [omega], n_modes=1, pml_params=pml_params)

    raw_S = raw_s_parameters(sweep_results, ["PORT_1", "PORT_2"], 1)
    S = deembed(raw_S, sweep_results[0].port_modes, {"PORT_1": 0.0, "PORT_2": 0.0})
    dataset = assemble_sweep_dataset([omega], [S])

    beta_fem = float(sweep_results[0].port_modes["PORT_1"][0].gamma.imag)
    beta_hj = beta_analytic(params.eps_r_substrate, params.w, params.h_sub, omega)

    return {
        "S_dominant": dataset.S_dominant[0],
        "beta_fem": beta_fem,
        "beta_analytic": beta_hj,
        # Measured margins on this fixture (te=6, single point at 25 GHz,
        # not the low-frequency end Section 1.4 says agreement is
        # tightest at): beta relative error ~9%, reciprocity residual
        # ~0.2%. Tolerances below are set with headroom over those
        # measured values, not derived from them after the fact to force
        # a pass -- reciprocity in particular is genuinely tight here and
        # deserves a tolerance that would actually catch a regression.
        "beta_tol": 0.15,
        "reciprocity_tol": 0.01,
    }


def test_phase1_beta_fem_is_in_the_physical_index_range(phase1_gate_results):
    """A basic sanity check independent of the analytic formula's own
    accuracy: the FEM dominant mode's effective index must lie between
    the air and substrate indices."""
    omega = 2 * np.pi * _PARAMS_KWARGS["reference_frequency"]
    k0 = omega * np.sqrt(4e-7 * np.pi * 8.8541878128e-12)
    n_eff = phase1_gate_results["beta_fem"] / k0
    assert 1.0 <= n_eff <= np.sqrt(_PARAMS_KWARGS["eps_r_substrate"]) * 1.02


def test_phase1_gate_runs_end_to_end(phase1_gate_results):
    from validation.gates import run_phase_gate

    report = run_phase_gate(1, phase1_gate_results)
    # Energy conservation was deliberately not included (trivial-PML
    # fixture -- see module docstring), so it must show up as skipped,
    # and h-convergence wasn't run here either (a dedicated, slower test
    # is the right place for that, per Section 3's own multi-level cost).
    assert "extended energy conservation (missing 'S_energy'/'excitation_port'/'n_modes')" in report.skipped
    assert any("h-convergence" in s for s in report.skipped)
    if not report.passed:
        pytest.fail(
            f"Phase 1 gate failed: {report.failures}. Per Module 4 Section 3.6/5.1 and "
            "docs/CLAUDE.md's carry-forward notes, the first two places to check are the "
            "port eigenproblem block arrangement and the port operator overall sign."
        )
