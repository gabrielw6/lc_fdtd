# CLAUDE.md

Operational guidance for working in this repository. This file is **not** a mathematical
reference — the equation documents under `docs/` are the authoritative source of truth for all
physics and numerics. This file exists to keep code generation inside the scope, conventions,
and invariants those documents establish, and to record what must **not** be built yet.

Read this file first, then read the equation doc for whichever module you are about to touch,
**before** writing any code for it.

---

## 1. What this repository is

A full-wave FEM solver for **driven S-parameter** simulation of GHz-wave propagation through a
microstrip line whose PCB substrate contains a rectangular liquid-crystal (LC) cutout modeled
as a continuously-varying anisotropic dielectric. The scientific goal is to consume a
user-supplied director field `n(r)` and simulate wave propagation through the resulting
position-dependent permittivity tensor `ε(r)` — without the slab/cluster discretization that
tools like HFSS force on anisotropic volumes.

This is one of three separate thesis-chapter tools. The cavity-perturbation (Rayleigh–Ritz)
and MoM surface-impedance tools live in their own repositories and are **not** part of this
one. Do not import from, reference, or attempt to reconcile against them here.

## 2. Scope boundaries (hard)

**In scope**: driven S-parameter solve via FEM with Nédélec edge elements; anisotropic
position-dependent `ε(r)`; modal waveguide ports; PML; frequency sweep by per-point direct
solve; S-parameter extraction and de-embedding.

**Out of scope — do not implement**:
- **LC director physics** (Frank elastic energy, bias-field response, director relaxation).
  `n(r)` is a consumed input, full stop.
- **Eigenmode / resonant-Q analysis.** This tool solves the *driven* linear system
  `(K − k0² M + Σ Bₚ) a = Σ gₚ` at each frequency. There is no eigensolver for the 3D problem.
  (The 2D *port* eigenproblem in Module 4 is the only eigenproblem, and it is a means to the
  driven solve, not an end.)
- **MoM / surface-impedance / metal-coating** anything. Different tool.

## 3. Repository layout and document map

```
docs/
  architecture_fem_sparameter_modules.md   # top-level architecture, all modules, invariants
  module1_mesh_interface_equations.md      # Module 1 detailed spec
  module<N>_<name>_equations.md            # one per module, added as modules are specced
CLAUDE.md                                  # this file

src/lc_sparam/                             # implementation (mirrors the module map below)
  conventions.py
  mesh/interface.py
  material/{core,regions,tensor_interpolation}.py
  fem/{edge_elements,assembly}.py
  ports/{mode_solver,port_operator}.py
  pml/stretching.py
  solve/{system,sweep}.py
  extract/sparameters.py
  validation/{analytic_microstrip,checks}.py

tests/                                     # mirrors src; validation gates live here
```

**Authoritative source of truth**: the `docs/module*_equations.md` files. If code and a doc
disagree, the doc wins — fix the code, or if the doc is wrong, fix the doc *first* and note it,
then the code. Never let them silently drift.

## 4. Module map and status

| Module | Package | Depends on | Status |
|---|---|---|---|
| — Conventions | `conventions` | — | specced |
| 0 Geometry builder | `geometry.builder` | external mesh module | specced |
| 1 Mesh interface (geometry) | `mesh.interface` | Module 0's output | specced |
| 2 Material model | `material.*` | mesh | specced |
| 3 FEM assembly | `fem.*` | mesh, material | specced |
| 4 Ports | `ports.*` | mesh, material | implemented, verified (see note below) |
| 5 PML | `pml.*` | material | specced |
| 6 Solve / sweep | `solve.*` | fem, ports, pml | specced |
| 7 S-parameter extraction | `extract.*` | ports, solve | specced |
| 8 Validation | `validation.*` | all | specced |

Mesh **generation** (the meshing algorithm itself) is external (existing package). Module 0
builds and tags the one fixed topology (microstrip + centered LC cutout) from a handful of
dimension parameters and invokes that external mesher; Module 1 then consumes the tagged
mesh read-only. Module 0 also introduces three boundary tags beyond Module 1's original list —
`PML_OUTER_PEC`, `PMC_SIDE` (a *deliberate* natural/PMC truncation on the lateral faces,
explicitly tagged so Module 1's "every boundary face must resolve to a tag" coverage check
stays meaningful rather than flagging it as an omission), and `PORT_CAP` (folded into Module
1's combined `PEC` tag aggregate, not its own bucket — see below). See
`docs/module0_geometry_builder_equations.md`.

**Port-aperture decoupling** (added post-review, `module0_geometry_builder_equations.md` §1.4 /
`module4_ports_equations.md` §3.7): `GeometryParams.W_port`/`H_port` (both optional, default
`None` ⇒ full cross-section, backward compatible) let a `PORT_p` face be a sub-rectangle of the
domain's end-plane cross-section rather than the whole thing. The region outside the aperture
is tagged `PORT_CAP`, a PEC "cap" idealization folded into the `PEC` aggregate (`PEC_GROUND` ∪
`PEC_LINE` ∪ `PORT_CAP`) — this is what makes a restricted aperture's own side/top walls PEC in
Module 4's 2D port eigenproblem automatically, with **zero changes to `ports.cross_section`'s
extraction logic or the eigenproblem itself** (Module 4's `mode_solver.py` did later gain an
unrelated change, immediately below). `ports.sizing.check_port_sizing` (informational only,
never raises) flags an aperture too large (box-mode risk near Module 4 §3.7's known limitation)
or too small (fringing-field clipping risk); emitted once per port at the sweep's own `f_max`
(`solve.sweep.run_sweep`) and once per `PortModeSolver.solve` call. A correctly-sized aperture
still needs the mesh to actually resolve it — an under-resolved aperture can fail to find enough
well-conditioned modes at all (`PortModeError`), a distinct failure from mis-selecting a box
mode; sizing and mesh resolution are both necessary, neither alone sufficient.

**Single-mode-tolerant mode counting** (added post-review, alongside the port-aperture
decoupling above — `module4_ports_equations.md` §3.7's third mitigation): a correctly-sized
aperture is often genuinely single-mode by physics, but `PortModeSolver.solve`'s original
contract pinned "how many modes are required" and "how many to try to return" to the same
`n_modes` argument, so a single-mode port failed outright rather than succeeding with 1.
`solve(port_tag, omega, n_modes, n_desired=None)` now separates them: `n_modes` is the required
minimum (still raises `PortModeError` if not met); `n_desired` (default `n_modes`, so omitting
it preserves the old exact-match behavior) is `solve.sweep.run_sweep`'s tracking oversupply,
allowed to come back short without raising. `run_sweep` passes its own `n_modes` as the
required minimum and `n_modes+2` as `n_desired`. The CLI's `--n-modes` default changed to `1`
accordingly (physically correct for a plain isotropic line; raise it for anisotropic/LC cases
where a second, cross-polarization mode is genuinely expected).

**`B_p` symmetric-by-construction** (added post-review, same review as the two items above):
`ports.port_operator.build_B`'s original formula (`Y_m * outer(overlap_e, overlap_h)`) is
symmetric only via the modal-admittance identity `h_m = Y_m*(x_hat x e_m)`, which holds
analytically but not to the discrete field reconstruction's own precision — a marginal mode was
confirmed to produce up to ~130% relative asymmetry, tripping `solve.system.factor`'s symmetry
check on an otherwise-correct system with a misleading "transposed tensor index" diagnosis.
`build_B` now assembles `(Y_m**2) * outer(overlap_e, overlap_e)` instead — algebraically equal
when the identity holds, but symmetric *by construction* (a sum of `scalar * outer(v,v)` terms)
regardless of mode quality. `solve.system._SYMMETRY_TOL` was tightened back from `0.3` to
`1e-6` accordingly, and `factor()` gained an optional `components` parameter so a genuine
symmetry failure's error message can name which contributing term (`K-k0^2*M` vs `B`) actually
broke symmetry instead of always blaming a tensor-index bug.

**Per-port axial orientation** (separate later review): `ports`' formulas were derived assuming
`n_out=-x_hat` (true for PORT_1 only — PORT_2's outward normal is `+x_hat`, geometry-dependent,
not hardcoded). `PortCrossSection.axial_sign` (`s_p`, derived from the owning tet's centroid
relative to the port plane, never from a hardcoded `x0==0` test) threads through
`ports.mode_solver`'s axial/H-field/power quantities (`_h_t_on_triangle`, `_mode_integrals`,
`_mode_overlaps`, `_raw_overlap`) so `PortMode.h_t()` and the Poynting/`Y_m`-consistency checks
are physically correct for either port. Checked and confirmed **unneeded** in
`ports.port_operator.build_B`/`build_g` themselves: `mode.Y` and `overlap_e` are both provably
`s_p`-invariant (the two `s_p` factors any correct derivation picks up always cancel), and an
earlier attempt to add an explicit `s_p` to `build_B` was empirically falsified by
`test/test_extract/test_reciprocity_uniform_line.py` (non-reciprocal, non-passive `|S22|>1`) and
reverted — see `build_B`'s own docstring for the account of what was tried and rejected.

**Injection/extraction (`N_m`) normalization fix** (separate later review, energy conservation):
invariant 4 below already specified a "power-based reference-impedance definition `(Yₘ, Nₘ)`"
threading mode solver → port operator → extractor, but `build_B` was not actually reading `Nₘ` —
Section 5.1's boxed `B_p`/`g_p` are derived assuming `Nₘ=1`-normalized modes, while
`ports.mode_solver._normalize` only enforces `Pₘ=1` (a *different*, conjugated bilinear form;
`Nₘ=2·Pₘ=2` for a lossless mode, not 1 — see `_self_overlap`'s docstring). `extract.project`
(extraction) already carried the resulting explicit `1/Nₘ` division; `build_B` (which routes that
same *solved*-field quantity back into the system, i.e. the injection-side counterpart of the
identical normalization) did not, until this fix — the real cause of a passivity-gate deficit an
earlier docstring had misattributed to the (also-real, but energy-conserving) trivial-PML
reflection. `PortMode` now caches `self_overlap` (`Nₘ`, computed once per mode from already-
available `overlap_e`/`overlap_h`, no fresh quadrature) and `build_B` divides by it; `build_g`
needs no equivalent correction (its `a_m^{+,inc}` term is a directly-given amplitude, never routed
through `project`/`1/Nₘ` — see `build_g`'s own docstring for the re-derivation). The passivity gate
(`|S11|²+|S21|²~=1`, not merely `<=1`) is now a standing test
(`test/test_extract/test_reciprocity_uniform_line.py`), not just a reciprocity one.

`docs/module3_fem_assembly_equations.md` §1 added three small, purely geometric additions to
Module 1's contract — barycentric weights returned from `quadrature_tet`, a per-tet volume tag
accessor, and `pec_edge_dofs()` — already applied to `module1_mesh_interface_equations.md`. If
Module 1 is ever reimplemented from scratch, these three are not optional: Module 3 depends on
all of them.

`docs/module5_pml_equations.md` §4 required one small addition to Module 2's contract: `material.core`'s
generic passivity check (§1.3 there) is documented as **not applying to `PMLMaterial`** — its
normal-direction tensor component has a positive imaginary part by construction (this is what
makes it matched, not a sign error). Already applied to `module2_material_equations.md`. Do not
"fix" `PMLMaterial`'s sign to satisfy the generic check; validate PML correctness via the
reflection-coefficient test in Module 5 §5.1 instead.

**Carry-forward from Module 4's implementation, addressed in Module 6**: `ports.mode_solver`'s
mode selection (`module4_ports_equations.md` §3.7) had a known limitation — plain
$\beta$-sorting can occasionally select a spurious "box mode" of the PMC-walled port enclosure
instead of the physical quasi-TEM mode. `module6_solve_sweep_equations.md` §6 implements the
principled fix flagged there: frequency-to-frequency mode tracking via field-overlap
correlation, seeded by a starting-frequency precondition check (§6.1) where box modes are still
evanescent. This introduces the one genuinely stateful element in the whole sweep (`TrackingState`,
§6.4) — everything else per frequency is independent. Validate this specifically against the
synthetic near-degenerate test case Module 6 §8 step 5 calls for before trusting it on the real
geometry.

## 5. Global conventions (inherited by every module — never redefine locally)

From `docs/architecture_fem_sparameter_modules.md §0`:

- **Time convention `e^{+jωt}`.** Consequences that MUST hold in code:
  passive lossy dielectric `Im(ε_r) ≤ 0` (i.e. `ε_r = ε_r'(1 − j·tanδ)`); PML stretch
  `s = κ − j·σ/(ωε₀)` with `σ ≥ 0`. A sign flip here silently produces a gain medium.
- **Primary unknown is `E`**, discretized in **lowest-order Nédélec (edge) elements**.
- **Relative material tensors** `ε_r`, `μ_r` (3×3, dimensionless). `μ_r = I` physically, but
  assemblers MUST accept a tensor `μ_r` because PML supplies `μ_r ≠ I`.
- **`k0 = ω·sqrt(μ₀·ε₀)`**; the system parameter is `k0²`, operator `(K − k0² M)`. Do not
  reintroduce the `ω²M` shorthand in code — use `k0²` explicitly.
- Straight-sided tets + lowest order ⇒ **assemble in physical coordinates, no Piola map**
  (`module1 §1`). Reintroducing a covariant transform is a deferred item (§8), not a default.

## 6. Load-bearing invariants (violating any of these is a bug, even if tests pass)

These are the rules that keep the solver correct; several fail *silently* (wrong numbers, no
crash), so they are encoded as assertions, not left to reviewer vigilance.

1. **`material.core.MaterialModel` is the sole material interface.** `regions`,
   `tensor_interpolation`, and `pml.PMLMaterial` are all implementations of it. **The assembler
   must never branch on scalar-vs-tensor or isotropic-vs-LC-vs-PML.** It contracts
   `ε_r(quadrature_points) → (M,3,3)` and integrates. This is what makes the Phase 1→4 build a
   change of *input only* (§7).
2. **Interpolate the ε tensor, never the director `n`** (`module2`, once specced). Rationale:
   `n` and `−n` are the same state; interpolating `n` collapses across sign flips and breaks
   unit norm. Assert: feeding `n` and `−n` yields bit-identical `ε`; `tr(ε')` is invariant
   across the LC region; eigenvalues stay within `[ε⊥, ε∥]`.
3. **`ε_r` symmetric ⇒ `M`, `K`, `Bₚ` complex-symmetric ⇒ S-matrix symmetric.** One property,
   checked at every layer. Use a **complex-symmetric** factorization (LDLᵀ-type), never a
   Hermitian solver or Cholesky. Assert `‖M − Mᵀ‖ ≈ 0`; `Bₚ` is now assembled symmetric *by
   construction* (`(Yₘ)² · outer(overlap_e, overlap_e) / Nₘ`, not `Yₘ · outer(overlap_e,
   overlap_h)` — see the port-aperture-decoupling-review notes in §4 above; the `/Nₘ` is the
   injection/extraction normalization fix, also §4), so `solve.system.factor` checks this at a
   tight `1e-6` tolerance; a failure past that is a transposed tensor index and is the proximate
   cause of a non-reciprocal S. Dividing by the real-valued-for-lossless scalar `Nₘ` does not
   affect this symmetry property (still a sum of `scalar · outer(v,v)` terms).
4. **Ports live in isotropic feed sections** (geometry guarantees it: LC cutout length < line
   length). One **power-based** reference-impedance definition `(Yₘ, Nₘ)` threads the mode
   solver → port operator → extractor — `PortMode.self_overlap` is `Nₘ`, cached once per mode
   and read by both `build_B` (injection) and `extract.project`/`biorthogonality` (extraction);
   do not recompute either `Yₘ` or `Nₘ` independently downstream of `ports.mode_solver`.
5. **PML and LC are spatially disjoint.** `PMLMaterial` wraps an *isotropic* background and
   returns `Λ·ε_bg`, `Λ·μ_bg`. Never compose `Λ` with the LC tensor.
6. **Cache the frequency-independent interior once.** Interior `K`, `M` and the cached LC ε
   samples do not depend on ω. Only port blocks `Bₚ(ω)`, `gₚ(ω)` and PML blocks rebuild per
   frequency (both are geometrically localized / low-rank).
7. **Global edge orientation is fixed by ascending global vertex index** (`module1 §3`), with a
   per-tet sign `s_e`. This is the correctness linchpin for edge elements; a disagreement
   between two elements sharing an edge produces wrong fields with no error. Never bypass
   `mesh.interface.tet_edge_sign`.

The condensed list also lives in `architecture_fem_sparameter_modules.md §9`; if you change one
here, change it there too.

## 7. Build sequence — implement phase by phase, gate before advancing

The material tensor is the *only* thing that changes across phases. Do not start a phase until
the prior phase's gate passes (`validation.checks`).

- **Phase 1 — uniform isotropic microstrip.** Validates the entire non-material stack.
  Gates: β, Z₀ vs Hammerstad–Jensen/Wheeler; `S21 = S12`; `|S11|² + |S21|² = 1` (lossless);
  monotone h-convergence; `arg(S21)` linear in line length.
- **Phase 2 — position-dependent *scalar* ε(r).** Validates quadrature sampling.
  Gates: ε→const reproduces Phase 1; layered-dielectric vs quasi-static ε_eff; quadrature
  order N vs N+2 agree.
- **Phase 3 — general symmetric *tensor* ε(r), supplied directly.** Validates anisotropic M.
  Gates: axis-aligned uniaxial vs ordinary/extraordinary index; rotated axes gated on
  **invariants** (S symmetric + `|S| ≤ 1`); measurable second-port-mode excitation.
- **Phase 4 — LC region via director file.** Validates `tensor_interpolation` + boundary
  contract. Gates: uniform director reproduces Phase-3 rotated-uniaxial bit-for-bit;
  reciprocity + passivity; director-tilt `arg(S21)` monotone at the right order of magnitude.

**Reciprocity (S symmetric) and passivity (`|S| ≤ 1`) are the workhorse gates** — they hold
without any analytic reference and pinpoint the anisotropic-assembly and port-constant bugs.
Wire them into every phase from Phase 3 on.

## 8. Deferred — specified for later, do NOT implement now

Building these before they're needed adds dead code and bug surface. Each is gated on a
specific trigger:

- **Model-order reduction / adaptive sweep (AWE etc.)** — only if per-point sweep cost over
  many bias states becomes a bottleneck. Note: PML and port blocks are **rational, not affine**
  in ω, which is exactly where naive moment-matching struggles — factor this in before choosing
  an MOR method.
- **Higher-order / curved elements + covariant (Piola) transform** — only if lowest-order
  accuracy proves insufficient. Trigger recorded in `module1 §1`.
- **Broadband ε∥/ε⊥ dispersion** — currently assumed negligible over the fixed GHz window; add
  per-frequency material re-sampling only if the band widens enough to matter.
- **Analytic port/material fast paths** — default quadrature path first; optimize only if
  profiling shows a bottleneck, and gate any fast path on exact agreement with the default.

## 9. Coding conventions

- **Language**: Python; `numpy` for dense per-element math, `scipy.sparse` for global assembly,
  a complex-symmetric sparse direct solver for the system (see invariant 3).
- **Interfaces are ABCs**, matching the cavity-tool house style: `MaterialModel`,
  `MeshInterface`, `PortModeSolver` etc. define the contract; implementations follow.
- **Vectorize material evaluation over quadrature points** — `epsilon(points)` takes `(M,3)`
  and returns `(M,3,3)`; the assembler evaluates once per element at all its quadrature points.
- **Assertions encode invariants, not comments.** Every "must hold" in the module docs (volume
  consistency `Σw = V`, symmetry, orientation coherence, partition-of-unity
  `Σ∇λ_i = 0`) is a runtime check that raises near the bug, per the Module 2 "loud failure"
  philosophy. Prefer raising over silently returning a wrong number.
- **No frequency dependence outside `ports`, `pml`, and the sweep driver.** If you find ω
  leaking into `fem.assembly` or `material`, that's a design error.
- **Return real energies/powers as `float`**, asserting the imaginary part is negligible first
  (same correction as the cavity Module 2 `integrate_field_energy` return type).

## 10. Working style for the agent

1. Before implementing module N, open `docs/module<N>_*_equations.md` and implement to *its*
   equations and checks. If that doc doesn't exist yet, the module isn't ready — ask for it to
   be specced rather than improvising the physics.
2. Implement the validation checks for a module **alongside** the module, not after. A module
   with no passing gate is not done.
3. Respect the phase order (§7). Do not implement Phase 3 tensor handling while Phase 1 gates
   are red.
4. Do not implement anything in §8.
5. When a change touches an invariant (§6), update both this file and
   `architecture_fem_sparameter_modules.md §9` so they stay in sync.
