"""solve.sweep -- interior/PML caching split, port mode tracking, and the
frequency sweep orchestration (docs/module6_solve_sweep_equations.md
Sections 2-3, 6-7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy import constants as _c

from fem import assemble
from material import MaterialAssembly
from mesh_interface import MeshInterface
from pml import PMLMaterial
from ports import PortMode, PortModeSolver, build_B, build_g, mode_similarity

from .system import Factorization, build_restriction, factor, reduce_system, recover_solution, solve_with_factorization

_MIN_OVERLAP = 0.9  # Section 6.2 step 5
_PML_TAG = "PML_TOP"


class ModeTrackingError(RuntimeError):
    """Raised when a tracked mode's best-available overlap falls below the
    minimum-overlap threshold (Section 6.2 step 5) -- a frequency step too
    coarse, a mesh too coarse, or a genuine mode transition needing a
    human look, not something to accept silently."""


class SweepPreconditionError(RuntimeError):
    """Raised when Section 6.1's starting-frequency precondition fails:
    more candidates are propagating at the sweep's first frequency than
    `n_modes` -- plain beta-sort cannot be trusted even at the start."""


@dataclass
class TrackingState:
    """Section 6.4: the one stateful element in the sweep -- each port's
    currently-tracked modes, threaded through the loop and updated after
    every frequency step."""

    modes: dict[str, list[PortMode]]


@dataclass
class SweepResult:
    omega: float
    excitation: tuple[str, int]
    a: np.ndarray
    port_modes: dict[str, list[PortMode]]


def _is_propagating(gamma: complex) -> bool:
    """Same propagating-dominant classification `ports.mode_solver` uses
    internally: `|Re(gamma)| << |Im(gamma)|`, i.e. genuinely close to the
    lossless `gamma=j*beta` branch."""
    return gamma.imag != 0 and abs(gamma.real) < 1e-3 * abs(gamma.imag)


def check_starting_frequency_precondition(candidates: dict[str, list[PortMode]], n_modes: int) -> None:
    """Section 6.1's required check at omega_1: among the (already
    spurious-filtered, per `PortModeSolver.solve`) candidates, no more
    than `n_modes` may be propagating -- otherwise plain beta-sort is not
    trustworthy even at the start of the sweep."""
    for tag, modes in candidates.items():
        n_propagating = sum(1 for m in modes if _is_propagating(m.gamma))
        if n_propagating > n_modes:
            raise SweepPreconditionError(
                f"port {tag!r}: {n_propagating} candidate(s) are propagating at the starting "
                f"frequency, more than n_modes={n_modes} -- lower the starting frequency so box "
                "modes are still evanescent (Section 6.1), or fall back to manual selection at "
                "this one point"
            )


def track_modes(
    candidates: dict[str, list[PortMode]],
    state: TrackingState | None,
    is_first_step: bool,
    n_modes: int,
) -> tuple[dict[str, list[PortMode]], TrackingState]:
    """Section 6.2's per-step tracking procedure. `candidates` is the
    oversupplied, already spurious-filtered pool `PortModeSolver.solve`
    returns (ranked by descending beta internally). `n_modes` is not part
    of Section 10's minimal listed signature but is required to know how
    many of the oversupply to select at the first step (where there is no
    previous state to size against) -- a small, documented extension,
    matching this repo's precedent of extending a doc's minimal contract
    when the pure form is unworkable (e.g. Module 1's extensions for
    Module 3)."""
    if is_first_step or state is None:
        # Section 6.1: plain beta-sort under the starting-frequency
        # precondition (checked separately, once, by the caller). Modes
        # returned by PortModeSolver.solve are already beta-descending.
        selected = {tag: list(modes[:n_modes]) for tag, modes in candidates.items()}
        return selected, TrackingState(modes=selected)

    selected: dict[str, list[PortMode]] = {}
    for tag, prev_modes in state.modes.items():
        pool = list(candidates[tag])
        assigned: list[PortMode] = []
        for prev in prev_modes:
            if not pool:
                raise ModeTrackingError(
                    f"port {tag!r}: ran out of candidates while tracking {len(prev_modes)} "
                    "mode(s) -- request a larger oversupply"
                )
            scores = [mode_similarity(c, prev) for c in pool]
            best_idx = int(np.argmax(scores))
            best_score = scores[best_idx]
            if best_score < _MIN_OVERLAP:
                raise ModeTrackingError(
                    f"port {tag!r}: best available overlap {best_score!r} fell below the "
                    f"minimum threshold {_MIN_OVERLAP!r} (Section 6.2 step 5) -- frequency step "
                    "too coarse, mesh too coarse, or a genuine mode transition"
                )
            assigned.append(pool.pop(best_idx))
        selected[tag] = assigned
    return selected, TrackingState(modes=selected)


def _interior_pml_tets(mesh: MeshInterface) -> tuple[list[int], list[int]]:
    """Section 2's partition, computed once from Module 1's per-tet tags."""
    interior, pml = [], []
    for t in range(mesh.n_tets):
        (pml if mesh.tet_volume_tag(t) == _PML_TAG else interior).append(t)
    return interior, pml


def run_sweep(
    mesh: MeshInterface,
    interior_materials: MaterialAssembly,
    port_tags: list[str],
    frequencies: list[float],
    n_modes: int = 2,
    pml_params: dict | None = None,
) -> list["SweepResult"]:
    """Section 7's full per-frequency procedure, looped over `frequencies`.
    `pml_params`, if the mesh has a `PML_TOP` region, must supply
    `background` (a `MaterialModel`), `z_air_top`, `thickness`, and
    optionally `R0`, `n`, `kappa_max` (Module 5's `PMLMaterial` signature,
    minus `omega` which this function supplies fresh per frequency)."""
    interior_tets, pml_tets = _interior_pml_tets(mesh)

    # Section 2: the one-time interior cache.
    K_int, M_int = assemble(mesh, interior_materials, tet_subset=interior_tets)

    # Section 4.4: the restriction matrix, built once.
    pec_dofs = mesh.pec_edge_dofs()
    R = build_restriction(pec_dofs, mesh.n_edges)

    # Ports live in isotropic feed sections only (Module 4 Section 1) --
    # never in PML_TOP -- so the interior materials alone suffice.
    port_solver = PortModeSolver(mesh, interior_materials)

    oversupply = n_modes + 2  # Section 6.2 step 1
    tracking_state: TrackingState | None = None
    results: list[SweepResult] = []

    for step, omega in enumerate(frequencies):
        k0 = omega * np.sqrt(_c.mu_0 * _c.epsilon_0)

        # Section 3: per-frequency PML re-assembly.
        if pml_tets:
            if pml_params is None:
                raise ValueError("mesh has a PML_TOP region but no pml_params were supplied")
            pml_material = PMLMaterial(omega=omega, **pml_params)
            K_pml, M_pml = assemble(mesh, MaterialAssembly({_PML_TAG: pml_material}), tet_subset=pml_tets)
        else:
            K_pml = sp.csr_matrix(K_int.shape, dtype=complex)
            M_pml = sp.csr_matrix(M_int.shape, dtype=complex)

        # Section 6/7 step 2: solve, filter (internal to PortModeSolver),
        # and track each port's modes.
        candidates = {tag: port_solver.solve(tag, omega, n_modes=oversupply) for tag in port_tags}
        if step == 0:
            check_starting_frequency_precondition(candidates, n_modes)
        tracked, tracking_state = track_modes(candidates, tracking_state, is_first_step=(step == 0), n_modes=n_modes)

        # Section 7 step 3.
        B = build_B(tracked, mesh, omega)

        # Section 1 + Section 7 step 4: full system, reduced and factored once.
        A = (K_int + K_pml) - (k0**2) * (M_int + M_pml) + B
        A_ff, _ = reduce_system(A, np.zeros(mesh.n_edges, dtype=complex), R)
        fact: Factorization = factor(A_ff)

        # Section 7 step 5: one excitation per port's dominant tracked mode.
        for port_tag in port_tags:
            g = build_g(tracked, {(port_tag, 1): 1.0 + 0j}, mesh, omega)
            b_f = R @ g
            a_f = solve_with_factorization(fact, b_f)
            a = recover_solution(a_f, R)
            results.append(SweepResult(omega=omega, excitation=(port_tag, 1), a=a, port_modes=tracked))

    return results
