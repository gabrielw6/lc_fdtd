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
| 4 Ports | `ports.*` | mesh, material | specced |
| 5 PML | `pml.*` | material | not specced |
| 6 Solve / sweep | `solve.*` | fem, ports, pml | not specced |
| 7 S-parameter extraction | `extract.*` | ports, solve | not specced |
| 8 Validation | `validation.*` | all | not specced |

Mesh **generation** (the meshing algorithm itself) is external (existing package). Module 0
builds and tags the one fixed topology (microstrip + centered LC cutout) from a handful of
dimension parameters and invokes that external mesher; Module 1 then consumes the tagged
mesh read-only. Module 0 also introduces two boundary tags beyond Module 1's original list —
`PML_OUTER_PEC` and `PMC_SIDE` (the latter is a *deliberate* natural/PMC truncation on the
lateral faces, explicitly tagged so Module 1's "every boundary face must resolve to a tag"
coverage check stays meaningful rather than flagging it as an omission). See
`docs/module0_geometry_builder_equations.md`.

`docs/module3_fem_assembly_equations.md` §1 added three small, purely geometric additions to
Module 1's contract — barycentric weights returned from `quadrature_tet`, a per-tet volume tag
accessor, and `pec_edge_dofs()` — already applied to `module1_mesh_interface_equations.md`. If
Module 1 is ever reimplemented from scratch, these three are not optional: Module 3 depends on
all of them.

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
   Hermitian solver or Cholesky. Assert `‖M − Mᵀ‖ ≈ 0`; a failure is a transposed tensor index
   and is the proximate cause of a non-reciprocal S.
4. **Ports live in isotropic feed sections** (geometry guarantees it: LC cutout length < line
   length). One **power-based** reference-impedance definition `(Yₘ, Nₘ)` threads the mode
   solver → port operator → extractor. Do not re-derive `Yₘ` independently in the extractor.
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
