"""Validation suite for solve.sweep's port mode tracking
(docs/module6_solve_sweep_equations.md Section 6, build step 5): "test it
standalone against a synthetic two-frequency case with a deliberately-
injected near-degenerate box mode, confirming tracking picks the correct
continuation where plain beta-sort would not" -- the direct regression test
for the failure Module 4's implementation found.

Construction: real solved `PortMode`s (so `mode_similarity`'s field overlap
is physically meaningful) from two frequencies of the real geometry, with
one candidate's `gamma` artificially inflated (via `dataclasses.replace`,
touching only the field `mode_similarity` never reads) to simulate a box
mode that would out-rank the true continuation on beta alone -- "deliberately
injected" exactly as the doc's build step 5 asks for, without needing a
hand-fabricated (and therefore risk-of-inconsistent) synthetic field.
"""
import dataclasses

import numpy as np
import pytest

from ports import mode_similarity
from solve.sweep import ModeTrackingError, TrackingState, track_modes

_PARAMS_KWARGS = dict(
    w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
    eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
    reference_frequency=25e9, target_elements_per_wavelength=8,
)


@pytest.fixture(scope="module")
def solver_and_omegas():
    pytest.importorskip("gmsh")
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import ConstantMaterial, MaterialAssembly, load_material_spec
    from mesh_interface import MeshInterface
    from ports import PortModeSolver

    params = GeometryParams(**_PARAMS_KWARGS)
    mesh_handle, material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    base_assembly = load_material_spec(geometry_stub=material_stub)
    tag_to_model = dict(base_assembly._tag_to_model)
    tag_to_model["LC"] = ConstantMaterial(eps_r=params.eps_r_substrate)
    materials = MaterialAssembly(tag_to_model)
    solver = PortModeSolver(mesh, materials)

    omega1 = 2 * np.pi * params.reference_frequency
    omega2 = 2 * np.pi * (params.reference_frequency * 1.02)  # a nearby step
    return solver, omega1, omega2


@pytest.fixture(scope="module")
def synthetic_case(solver_and_omegas):
    solver, omega1, omega2 = solver_and_omegas
    prev_mode = solver.solve("PORT_1", omega1, n_modes=1)[0]
    raw_candidates = solver.solve("PORT_1", omega2, n_modes=3)

    scores = [mode_similarity(c, prev_mode) for c in raw_candidates]
    true_idx = int(np.argmax(scores))
    impostor_idx = int(np.argmin(scores))
    assert true_idx != impostor_idx, "need a genuinely distinguishable pair to make this test meaningful"

    true_continuation = raw_candidates[true_idx]
    impostor = raw_candidates[impostor_idx]
    other = raw_candidates[[i for i in range(3) if i not in (true_idx, impostor_idx)][0]]

    # Inflate the impostor's beta far above every other candidate's --
    # `dataclasses.replace` only touches `gamma`; `e_t`/`h_t`/`e_edge_dofs`
    # (everything `mode_similarity` reads) stay exactly the impostor's own,
    # so the low-overlap field is preserved while the beta ranking is
    # deliberately corrupted.
    huge_beta = max(abs(m.gamma.imag) for m in raw_candidates) * 10.0
    corrupted_impostor = dataclasses.replace(impostor, gamma=1j * huge_beta)

    # Beta-descending order, with the corrupted impostor deliberately first
    # -- exactly what plain Section 3.7 beta-sort would hand to a naive
    # "take the top n_modes" selection.
    candidates = {"PORT_1": [corrupted_impostor, true_continuation, other]}
    state = TrackingState(modes={"PORT_1": [prev_mode]})
    return candidates, state, true_continuation, corrupted_impostor


def test_plain_beta_sort_would_pick_the_wrong_mode(synthetic_case):
    """Confirms the synthetic setup actually exercises the failure mode --
    otherwise this test suite would be vacuous."""
    candidates, _state, true_continuation, corrupted_impostor = synthetic_case
    beta_sorted_top1 = candidates["PORT_1"][0]  # already placed first by construction
    assert beta_sorted_top1 is corrupted_impostor
    assert beta_sorted_top1 is not true_continuation


def test_tracking_selects_the_true_continuation_not_the_higher_beta_impostor(synthetic_case):
    candidates, state, true_continuation, corrupted_impostor = synthetic_case
    selected, new_state = track_modes(candidates, state, is_first_step=False, n_modes=1)

    assert selected["PORT_1"][0] is true_continuation
    assert selected["PORT_1"][0] is not corrupted_impostor
    assert new_state.modes["PORT_1"][0] is true_continuation


def test_tracking_raises_when_no_candidate_clears_the_overlap_threshold(synthetic_case):
    """Section 6.2 step 5: a candidate pool with no genuine continuation at
    all (every candidate replaced by copies of the impostor's field) must
    raise, not silently accept a bad match."""
    _candidates, state, _true, corrupted_impostor = synthetic_case
    bad_pool = {"PORT_1": [corrupted_impostor, corrupted_impostor, corrupted_impostor]}
    with pytest.raises(ModeTrackingError):
        track_modes(bad_pool, state, is_first_step=False, n_modes=1)


def test_first_step_uses_plain_beta_sort(synthetic_case):
    candidates, _state, _true, corrupted_impostor = synthetic_case
    selected, new_state = track_modes(candidates, None, is_first_step=True, n_modes=1)
    assert selected["PORT_1"][0] is corrupted_impostor  # highest beta, no tracking applied
    assert new_state.modes["PORT_1"][0] is corrupted_impostor
