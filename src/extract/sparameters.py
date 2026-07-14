"""extract.sparameters -- modal amplitude extraction, raw generalized
S-parameters, de-embedding, extended energy conservation, and sweep
aggregation (docs/module7_extract_sparameters_equations.md).

Purely post-processing: reads `SweepResult.a` and `SweepResult.port_modes`
(Module 6), never solves anything new. Section 1's geometric fact (only a
port face's own three edges have nonzero tangential trace there) is what
makes `project_amplitude`'s edge-restricted sum exact, not an
approximation -- confirmed numerically in this package's own test suite,
not merely assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mesh_interface import MeshInterface
from ports import PortMode, extract_cross_section, field_from_edge_dofs, project
from solve import SweepResult

def port_face_edges(port_tag: str, mesh: MeshInterface) -> list[int]:
    """Section 9's contract: this port's own global edge indices, in the
    same ascending order `PortCrossSection.global_edge_ids` already uses
    (Module 4) -- so this is a thin, order-consistent wrapper, not a
    reimplementation of Module 4's own extraction."""
    return sorted(int(e) for e in extract_cross_section(mesh, port_tag).global_edge_ids)


def project_amplitude(a: np.ndarray, port_edges: list[int], mode: PortMode) -> complex:
    """Section 2's boxed `a_m^{(p),total}`. `port_edges` is checked (not
    just trusted) against `mode.cross_section`'s own edge set -- a
    mismatch here means the wrong port/mode pairing was passed, exactly
    the kind of caller mistake worth catching loudly rather than silently
    projecting onto the wrong cross-section."""
    cs = mode.cross_section
    if set(port_edges) != set(int(e) for e in cs.global_edge_ids):
        raise ValueError(
            "port_edges do not match this mode's own cross-section edges -- wrong port/mode/mesh pairing"
        )
    e_local = a[cs.global_edge_ids]
    E_t = field_from_edge_dofs(cs, e_local)
    return project(E_t, mode)


def raw_s_parameters(
    sweep_results: list[SweepResult], ports: list[str], n_modes: int
) -> dict[tuple[str, int, str], complex]:
    """Section 3.1's boxed `S_{(p,m),(q,1)}`, scoped to whichever
    frequency's `sweep_results` are passed (typically one frequency's
    worth -- one entry per excited port, Module 6 Section 7). Keyed
    `(port_p, mode_m, excited_port_q)`; the excitation's own mode index is
    always 1 by convention (Section 6) and so is not carried in the key.
    Reports every tracked mode up to `n_modes`, not just the dominant one
    (Section 3.2) -- `S^dominant`/`S^conversion` are views built later
    (`assemble_sweep_dataset`), not separate extraction passes."""
    S: dict[tuple[str, int, str], complex] = {}
    for result in sweep_results:
        q, excited_mode_number = result.excitation
        for p in ports:
            modes = result.port_modes[p]
            for mode_number, mode in enumerate(modes[:n_modes], start=1):
                port_edges = [int(e) for e in mode.cross_section.global_edge_ids]
                a_total = project_amplitude(result.a, port_edges, mode)
                incident = 1.0 if (p, mode_number) == (q, excited_mode_number) else 0.0
                S[(p, mode_number, q)] = a_total - incident
    return S


def deembed(
    S_raw: dict[tuple[str, int, str], complex],
    port_modes: dict[str, list[PortMode]],
    offsets: dict[str, float],
) -> dict[tuple[str, int, str], complex]:
    """Section 4's boxed formula, applying Module 4 Section 6's already-
    derived de-embedding, using each port's own tracked mode's `gamma` at
    this same frequency (already in `port_modes`, no re-solving).

    Scoped to one frequency at a time, matching `raw_s_parameters`' own
    scoping -- `S_raw`'s key has no frequency component, so `port_modes`
    here is Section 9's `port_modes_by_freq` interpreted as "the
    port_modes for the one frequency this S_raw came from" (the only
    self-consistent reading given `S_raw` itself is not frequency-keyed);
    a caller de-embedding a full sweep calls this once per frequency,
    alongside `raw_s_parameters`, before `assemble_sweep_dataset`."""
    S_deembedded: dict[tuple[str, int, str], complex] = {}
    for (p, m, q), value in S_raw.items():
        gamma_p = port_modes[p][m - 1].gamma
        gamma_q = port_modes[q][0].gamma  # the excited mode is always mode 1
        phase = np.exp(gamma_p * offsets[p]) * np.exp(gamma_q * offsets[q])
        S_deembedded[(p, m, q)] = value * phase
    return S_deembedded


def energy_balance(S_deembedded: dict[tuple[str, int, str], complex], excitation_port: str, n_modes: int) -> float:
    """Section 5's boxed extended sum, for one excitation and one
    frequency: `sum_p sum_{m=1}^{n_modes} |S_{(p,m),(q,1)}|^2`. Should
    approach 1 for a lossless, adequately-resolved configuration. Section
    5's own honesty note: a *small* deficit from 1 is expected with a
    finite `n_modes` (power scattered into uncaptured higher-order modes),
    not itself a bug; a *large* deficit's first diagnostic step is
    increasing `n_modes`, not assuming an assembly/extraction error."""
    return float(
        sum(
            abs(value) ** 2
            for (_p, m, q), value in S_deembedded.items()
            if q == excitation_port and m <= n_modes
        )
    )


@dataclass
class SParameterDataset:
    """Section 9's contract, plus `ports` (the index order `S_dominant`'s
    axes use) -- a small, additive field the doc's listed contract doesn't
    name but which is needed to interpret `S_dominant` at all; derived
    internally by `assemble_sweep_dataset`, never a caller-supplied
    parameter, so it does not change that function's documented
    signature."""

    frequencies: np.ndarray
    S_dominant: np.ndarray  # (n_freq, n_ports, n_ports) complex
    S_conversion: dict[tuple[str, int, str], np.ndarray]  # (port_p, mode_m>1, excited_port_q) -> (n_freq,)
    ports: list[str] = field(default_factory=list)


def assemble_sweep_dataset(
    frequencies: list[float], S_by_freq: list[dict[tuple[str, int, str], complex]]
) -> SParameterDataset:
    """Section 6: aggregate one dict per frequency (each already keyed
    `(port_p, mode_m, excited_port_q)`, e.g. from `deembed`) into the
    final dataset. `S_dominant`/`S_conversion` are documented *views* over
    the same underlying per-(p,m,q) values (Section 6), not separately
    computed or stored data. Port order for `S_dominant`'s axes is derived
    (sorted, deterministic) from the keys actually present, since Section
    9's signature does not pass `ports` in explicitly."""
    if len(frequencies) != len(S_by_freq):
        raise ValueError(f"{len(frequencies)} frequencies but {len(S_by_freq)} per-frequency S dicts")

    ports = sorted({p for S in S_by_freq for (p, _m, _q) in S} | {q for S in S_by_freq for (_p, _m, q) in S})
    port_index = {p: i for i, p in enumerate(ports)}
    n_freq, n_ports = len(frequencies), len(ports)

    S_dominant = np.zeros((n_freq, n_ports, n_ports), dtype=complex)
    S_conversion: dict[tuple[str, int, str], np.ndarray] = {}

    for f_idx, S in enumerate(S_by_freq):
        for (p, m, q), value in S.items():
            if m == 1:
                S_dominant[f_idx, port_index[p], port_index[q]] = value
            else:
                key = (p, m, q)
                if key not in S_conversion:
                    S_conversion[key] = np.zeros(n_freq, dtype=complex)
                S_conversion[key][f_idx] = value

    return SParameterDataset(
        frequencies=np.asarray(frequencies, dtype=float), S_dominant=S_dominant, S_conversion=S_conversion, ports=ports
    )
