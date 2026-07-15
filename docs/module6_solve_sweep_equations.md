# Module 6 — `solve`: Global System Assembly & Frequency Sweep — Equations & Implementation Plan

Companion to the top-level architecture doc and Modules 0–5. Same conventions: no code,
precise equations, step-by-step build order, validation targets. This module is where several
previously-deferred pieces converge — the interior/PML caching split (Module 3 §5.2/§5.4), PEC
elimination (Module 1's `pec_edge_dofs()`), and the port mode-tracking algorithm `CLAUDE.md`
flagged as a **required** part of this module's design, not an optional refinement, following
Module 4's implementation findings.

Note on symbols: this document uses $\mathbf R$ for the DOF restriction/selection matrix (§4).
Module 4 already uses $P_m$ for modal power — the two do not overlap, but are worth
distinguishing explicitly since both are common symbols in adjacent literature.

---

## 0. What Module 6 owns vs. consumes

**Consumes**: Module 1 (`tet_volume_tag` for the interior/PML tet partition, §2;
`pec_edge_dofs()` for elimination, §4; `n_edges` for system sizing); Module 2 (an
already-built interior `MaterialAssembly` covering `SUBSTRATE`/`AIR`/`LC`, supplied by
whatever driver script orchestrates a run — Module 6 does not build this itself); Module 3
(`assemble(mesh, materials, tet_subset)`); Module 4 (`PortModeSolver.solve`, `build_B`,
`build_g`, `deembed`, and the corrected §4.3 projection, reused here for mode tracking);
Module 5 (`PMLMaterial`, constructed fresh per frequency).

**Owns**: the per-frequency global system assembly and complex-symmetric solve
(`solve.system`); the sweep orchestration, including the interior/PML caching split, the port
mode-tracking algorithm, and the excitation loop (`solve.sweep`).

**Does not own**: mesh or geometry generation; S-parameter extraction proper (Module 7) — this
module's output is, for each frequency and each driven excitation, a solved coefficient vector
$\mathbf a$, handed to Module 7 along with the tracked `PortMode` objects.

**Scoping note on caching granularity**: the interior $\mathbf K_{\text{int}},\mathbf
M_{\text{int}}$ cache (§2) is valid for one fixed LC bias state (one director field) across the
whole frequency sweep — not across bias states. A parameter study over multiple bias states
(the original motivating use case) invokes this module fresh, with a newly built interior
`MaterialAssembly`, once per bias state; the frequency sweep is the loop this module owns, the
bias-state loop is outside it.

---

## 1. The global system equation

At each frequency $\omega$, restating and now fully assembling the equation from the top-level
doc and Modules 3–5:

$$\Big(\mathbf K_{\text{int}} + \mathbf K_{\text{pml}}(\omega) - k_0(\omega)^2\big[\mathbf M_{\text{int}} + \mathbf M_{\text{pml}}(\omega)\big] + \sum_p \mathbf B_p(\omega)\Big)\,\mathbf a = \sum_p \mathbf g_p(\omega)$$

with $k_0(\omega)=\omega\sqrt{\mu_0\varepsilon_0}$ (`conventions.py`, unchanged). Every term's
origin: $\mathbf K_{\text{int}},\mathbf M_{\text{int}}$ from Module 3, cached once per bias
state (§2); $\mathbf K_{\text{pml}}(\omega),\mathbf M_{\text{pml}}(\omega)$ from Module 3
re-invoked on the `PML_TOP` subset with a fresh Module 5 `PMLMaterial(omega=\omega,\ldots)`
each frequency (§3); $\mathbf B_p(\omega)$ from Module 4's `build_B`, using that frequency's
*tracked* port modes (§6); $\mathbf g_p(\omega)$ from Module 4's `build_g`, one per driven
excitation (§7).

---

## 2. Interior/PML tet-subset partition and one-time caching

$$\text{interior\_tets} = \{t : \text{tet\_volume\_tag}(t) \ne \texttt{PML\_TOP}\}, \qquad \text{pml\_tets} = \{t : \text{tet\_volume\_tag}(t) = \texttt{PML\_TOP}\}$$

computed once from Module 1's per-tet tags. $\mathbf K_{\text{int}},\mathbf M_{\text{int}} =$
`assemble(mesh, interior_materials, tet_subset=interior_tets)`, called **exactly once** per
bias state, before the frequency loop begins. This is the direct application of Module 3 §5.2's
subset support to the split Module 3 §5.4 already resolved architecturally — nothing new is
required of Module 3 itself.

---

## 3. Per-frequency PML re-assembly

At each $\omega$: construct a **minimal** `MaterialAssembly` containing only the `PML_TOP`
entry — `{PML_TOP: PMLMaterial(background=AirMaterial, omega=ω, z_air_top, h_pml, R0, n,
kappa_max)}` — and call `assemble(mesh, pml_materials, tet_subset=pml_tets)`. This works without
merging into the interior registry because Module 3's `assemble` only ever looks up tags
actually present among the tets it's given, and `pml_tets` only ever carries the `PML_TOP` tag.
No changes to Module 3 or Module 5 are needed to support this — it's a direct consequence of
how both were already architected.

---

## 4. PEC essential boundary condition — DOF elimination

### 4.1 Restriction matrix

Let $C = \text{pec\_edge\_dofs}()$ (Module 1, a fixed set independent of frequency and bias
state — computed once, cached alongside $\mathbf K_{\text{int}},\mathbf M_{\text{int}}$), and
$F = \{0,\ldots,N_e-1\}\setminus C$ the free DOFs, $N_f=|F|$. Define the selection matrix
$\mathbf R \in \{0,1\}^{N_f\times N_e}$: row $i$ of $\mathbf R$ has a single $1$ in the column
corresponding to the $i$-th free DOF (in some fixed order), zero elsewhere.

### 4.2 Reduced system and solution recovery

$$\mathbf A_{ff} = \mathbf R\,\mathbf A\,\mathbf R^{T}, \qquad \mathbf b_f = \mathbf R\,\mathbf b, \qquad \mathbf a = \mathbf R^{T}\mathbf a_f$$

where $\mathbf A$ is the full assembled system (§1) and $\mathbf b=\sum_p\mathbf g_p(\omega)$.
$\mathbf R^{T}$ zero-pads the reduced solution back into the full DOF space at the constrained
indices — correct because the essential BC requires exactly $a_c=0$ for $c\in C$, not merely
small.

### 4.3 Why no RHS correction is needed

Since $a_c=0$ exactly is being *enforced*, not derived from a nonzero prescribed value,
eliminating rows/columns in $C$ from both $\mathbf A$ and $\mathbf b$ is the complete
elimination step — there is no nonzero-Dirichlet-data correction term to move to the RHS
(unlike inhomogeneous Dirichlet problems in general FEM). $\mathbf g_p$'s entries at PEC-adjacent
DOFs are already structurally zero regardless (Module 4 §3.5's port-side test-space exclusion
of PEC edges), so this is consistent, not merely convenient.

### 4.4 Caching

$\mathbf R$ is built once (frequency- and bias-state-independent, purely geometric via $C$) and
reused for every reduction in the sweep.

---

## 5. Complex-symmetric factorization and multi-RHS reuse

### 5.1 Why complex-symmetric, not Hermitian or generic-asymmetric

$\mathbf K,\mathbf M,\mathbf B_p$ are each complex-**symmetric** (proved, not assumed: Module 2
§1.2's material symmetry $\Rightarrow$ Module 3 §3.3's element-matrix symmetry $\Rightarrow$
global assembly preserves it; Module 4 §5.1 notes $\mathbf B_p$'s analogous structural
symmetry). $\mathbf A_{ff}$ therefore inherits complex symmetry
($\mathbf A_{ff}=\mathbf A_{ff}^{T}$, **not** $\mathbf A_{ff}=\mathbf A_{ff}^{H}$ — do not use a
Hermitian solver). The correct factorization is a complex-symmetric sparse direct solve (e.g. an
$LDL^{T}$-type factorization, not Cholesky, not a Hermitian $LDL^{H}$).

**Update (post-review): $\mathbf B_p$'s symmetry needed a construction fix, not just a proof.**
Module 4 §5.1's original formula, $B_{ij}=-j\omega\mu_0\sum_m Y_m\cdot\text{overlap}_e[i]\cdot
\text{overlap}_h[j]$, is symmetric only via the modal-admittance identity
$\mathbf h_m=Y_m\,\hat x\times\mathbf e_m$ (which gives $\text{overlap}_h[j]=Y_m\cdot
\text{overlap}_e[j]$ *analytically*) — that identity holds to within the discrete field
reconstruction's own error, not exactly, and a marginal (near-degenerate, coarsely-resolved)
mode was confirmed to make the assembled $\mathbf B_p$ asymmetric by up to ~130% relative,
tripping this section's factorization check on an otherwise-fine system. `ports.port_operator.build_B`
now substitutes the identity directly into the formula, assembling
$B_{ij}=-j\omega\mu_0\sum_m(Y_m)^2\cdot\text{overlap}_e[i]\cdot\text{overlap}_e[j]$ instead —
every term is `(scalar)*outer(v,v)`, symmetric in the literal matrix for any mode quality, not
just analytically. `solve.system.factor`'s symmetry tolerance was tightened back from an earlier
generous `0.3` to `1e-6` accordingly (see that function's own docstring), and it accepts an
optional `components` dict so a symmetry failure's error message can name which contributing
term (the $\mathbf K-k_0^2\mathbf M$ block vs. $\mathbf B_p$) actually broke symmetry, rather than
always blaming a transposed tensor index.

### 5.2 Solver library guidance

Preferred: a solver with an explicit complex-symmetric mode (e.g. Intel MKL PARDISO, or MUMPS'
symmetric-complex setting) — exploits the structure for roughly half the memory/factorization
cost of a generic solve. Acceptable fallback: a generic asymmetric sparse LU (e.g.
`scipy.sparse.linalg.splu`, backed by SuperLU) — gives the **same correct answer**, since it
doesn't assume symmetry rather than assuming the wrong kind, just without the efficiency
benefit. Do not use a solver that assumes Hermitian structure; it will silently produce wrong
results on a genuinely complex-symmetric (non-Hermitian) matrix.

### 5.3 Factor once per frequency, reuse across excitations

At each $\omega$, $\mathbf A_{ff}(\omega)$ is assembled and factored **once**; every driven
excitation at that frequency (§7) is a different RHS $\mathbf b_f$ solved against the same
factorization — the standard multiple-RHS reuse Module 3 §3.1 already anticipated for ports in
general, now stated concretely for the excitation loop.

### 5.4 Nonsingularity check

$\mathbf K$ alone has a large null space (Module 3 §3.3: gradient fields, $\nabla\times\nabla\phi=0$)
— expected and harmless, since $-k_0^2\mathbf M$ and $\sum_p\mathbf B_p$ regularize the assembled
system away from it. **Runtime check**: the factorization must report no exactly-zero or
suspiciously-tiny pivot at any swept frequency; a near-singular $\mathbf A_{ff}(\omega)$ signals
either a genuine resonance of the truncated domain or an under-absorbing PML (top-level doc
§6.1), not a physical result to accept silently.

---

## 6. Port mode tracking across the frequency sweep

This is the piece `CLAUDE.md` flags as required, following Module 4's finding that plain
$\beta$-sorting (Module 4 §3.7) can select a spurious PMC-box mode instead of the physical
quasi-TEM mode at some frequencies.

### 6.1 Starting-frequency precondition

Choose the sweep's first frequency $\omega_1$ low enough that the port cross-section's box
modes — which have their own cutoffs, unlike the quasi-TEM mode — are still evanescent.
**Required check at $\omega_1$**: among the candidates surviving Module 4 §3.6's physical-bounds
filter, confirm that no more than $n_{\text{modes}}$ have real, positive $\beta$ (propagating).
If more do, $\omega_1$ is not low enough for plain $\beta$-sorting to be trusted even at the
start — lower it, or fall back to a one-time manual/inspection-based selection at this single
point before tracking takes over for the rest of the sweep.

### 6.2 Per-step procedure ($k>1$)

1. Request a modest oversupply of candidates from Module 4's eigensolve (e.g.
   $n_{\text{modes}}+2$ to $+3$, not just $n_{\text{modes}}$) — tracking needs spare candidates
   to choose among, not just the exact count wanted. **Update, single-mode-tolerant mode
   counting (post-review)**: this oversupply is a *desired* count, not a required one. A
   correctly-sized port aperture (Module 0 §1.4/Module 4's port-aperture decoupling) is often
   genuinely single-mode — only the quasi-TEM mode propagates, and every higher candidate is
   evanescent/un-power-normalizable by physics, not by a solver defect — so requiring the full
   oversupply to exist would make a physically single-mode port fail to run at all.
   `ports.mode_solver.PortModeSolver.solve` takes `n_modes` (the true required minimum — this
   step's own $n_{\text{modes}}$, never relaxed) and `n_desired` (the oversupply above) as
   separate parameters: it raises only if fewer than `n_modes` valid candidates are found, and
   otherwise returns however many of the `n_desired` pool it actually has (silently fewer, when
   that's all there is) for the tracker below to select from.
2. Apply Module 4 §3.6's physical-bounds filter to discard numerically spurious solutions.
3. For each of the $n_{\text{modes}}$ previously-tracked modes (from step $k-1$), compute a
   similarity score against every surviving candidate at step $k$:

$$\text{overlap}(c, m) = \frac{\left|\int_S (\mathbf e_t^{(c)}\times\mathbf h_t^{(\text{prev},m)})\cdot\hat x\,dS\right|}{\sqrt{\int_S(\mathbf e_t^{(c)}\times\mathbf h_t^{(c)})\cdot\hat x\,dS}\,\sqrt{\int_S(\mathbf e_t^{(\text{prev},m)}\times\mathbf h_t^{(\text{prev},m)})\cdot\hat x\,dS}}$$

   using Module 4's corrected §4.3 overlap integral (self-normalized, per that fix). Note this
   is a *similarity* metric between modes of two **different** eigenproblems (different
   $\omega$) — it is not bounded by 1 via the same biorthogonality proof that holds for two
   modes of the *same* frequency's eigenproblem, but empirically stays close to 1 for a genuine
   continuation and drops sharply at a crossing/mismatch, which is what makes it a useful
   diagnostic. If Module 4's implementation already normalizes every cached mode so its own
   self-overlap is exactly 1 (one of the two equivalent fixes noted there), the denominator
   above is identically 1 and the score reduces to the bare numerator — confirm which convention
   is in effect before assuming the simplified form.
4. Assign each of the $n_{\text{modes}}$ previous modes to the candidate maximizing its overlap
   score. A simple greedy assignment (process modes in a fixed order, claim the best-scoring
   remaining candidate, remove it from the pool) is the default; if multiple modes could
   plausibly swap simultaneously at one frequency step, an optimal bipartite assignment (e.g.
   the Hungarian algorithm) is more robust and worth the modest extra cost — greedy is the
   simpler default, optimal assignment is the upgrade if greedy proves fragile in practice.
5. **Minimum-overlap threshold**: if the best available overlap for some mode $m$ falls below a
   threshold (e.g. $0.9$), do not silently accept the match — raise/warn. This signals either a
   frequency step too coarse for the mode shape to be tracked reliably, a mesh too coarse to
   resolve it consistently, or a genuine physical mode transition that needs a human look
   rather than an automated guess.

### 6.3 Ports are tracked independently

Module 0's geometry guarantees both port cross-sections are congruent (same materials, same
dimensions) — their 2D eigenproblems and box-mode structure are independent copies of the same
problem, so tracking runs separately per port with no cross-port coupling at this stage. This
also gives a **strong, cheap validation target** (§9): corresponding tracked modes at Port 1 and
Port 2 should have near-identical $\gamma_m,Y_m$ at every frequency, since the two cross-sections
are geometrically identical — a real consistency check specific to this geometry, not a general
port-solver requirement.

### 6.4 State carried across the sweep

Unlike every other per-frequency quantity in this pipeline, port mode tracking is **stateful**
across the sweep: a small `TrackingState` per port (the previous step's selected
$(\gamma_m,\mathbf e_m,\mathbf h_m)$ for each of the $n_{\text{modes}}$ tracked modes) must be
threaded through the sweep loop and updated after each frequency step. This is the one place in
the whole pipeline where frequency points are not independent of each other.

---

## 7. The sweep loop and excitation set

**Excitation set**: per Module 4 §5.3's established convention, only each port's *dominant*
tracked mode is driven as an excitation — for this two-port device, exactly two excitations,
$(\text{Port 1}, m{=}1)$ and $(\text{Port 2}, m{=}1)$, at every frequency. Higher tracked modes
are never directly excited; they contribute to $\mathbf B_p$'s loading and can still receive
scattered power, read out by Module 7.

**Full procedure, per frequency $\omega_k$ in the sweep**:

1. Assemble $\mathbf K_{\text{pml}}(\omega_k),\mathbf M_{\text{pml}}(\omega_k)$ (§3).
2. For each port: solve the oversupplied eigenproblem, filter (Module 4 §3.6), track against
   the previous step (§6.2) — or, at $k=1$, plain $\beta$-sort under the §6.1 precondition.
3. Build $\mathbf B_p(\omega_k)$ for each port from its tracked modes (Module 4 §5).
4. Assemble the full system $\mathbf A(\omega_k)$ (§1); reduce via $\mathbf R$ (§4); factor once
   (§5).
5. For each excitation $(q,1)$ in the excitation set: build $\mathbf g_{(q,1)}(\omega_k)$
   (Module 4 §5), solve $\mathbf A_{ff}\mathbf a_f=\mathbf R\mathbf g_{(q,1)}$ against the cached
   factorization, recover $\mathbf a=\mathbf R^{T}\mathbf a_f$, and hand $(\omega_k,(q,1),\mathbf
   a)$ to Module 7 alongside the tracked `PortMode` objects for that frequency.
6. Update each port's `TrackingState` (§6.4) with this step's selected modes before proceeding
   to $\omega_{k+1}$.

---

## 8. Step-by-step build order

1. Interior/PML tet-subset partition and the one-time $\mathbf K_{\text{int}},\mathbf
   M_{\text{int}}$ cache (§2) — build and test in isolation against a case with **no** PML tag
   present at all, confirming it matches a plain Module 3 call over the whole mesh.
2. The restriction matrix $\mathbf R$ and elimination (§4) — unit test on a small hand-built
   system with a known PEC-constrained DOF, confirming the reduced solve reproduces the
   full-system solve restricted to free DOFs exactly.
3. Complex-symmetric factorization wrapper (§5), tested first on the cached interior system
   alone (no ports, no PML) against a case where the answer is otherwise checkable.
4. Per-frequency PML re-assembly (§3), integrated into the loop — confirm $\sigma\to0$
   reproduces the no-PML case (Module 5's own reduction check, now exercised end-to-end).
5. Port mode tracking (§6), built and validated **before** wiring it into the full sweep —
   test it standalone against a synthetic two-frequency case with a deliberately-injected
   near-degenerate box mode, confirming tracking picks the correct continuation where plain
   $\beta$-sort would not.
6. The full sweep loop (§7), excitation set, and handoff to Module 7's expected input shape.
7. Run the full §9 validation suite — this is the point at which the top-level doc's Phase 1
   gate (reciprocity, $|S_{11}|^2+|S_{21}|^2=1$) finally becomes runnable end-to-end, closing
   out the honesty flags left open in Modules 3, 4, and 5.

---

## 9. Validation targets

- **Interior/PML split correctness**: a run with the PML region given trivial ($\sigma=0$)
  material reproduces a plain whole-mesh Module 3 assembly, to solver tolerance.
- **PEC elimination correctness**: the recovered full solution $\mathbf a=\mathbf R^T\mathbf a_f$
  satisfies the *original* unreduced system's rows at the constrained DOFs to machine-precision
  residual (i.e. $a_c=0$ exactly, and the full system's constrained rows are consistent with
  that), not just that the reduced solve itself converged.
- **Multi-RHS reuse correctness**: solving all of a frequency's excitations against one cached
  factorization gives results identical (not just close) to solving each independently with its
  own fresh factorization — a regression test on the §5.3 reuse architecture specifically.
- **Nonsingularity across the sweep** (§5.4): no near-zero pivot at any swept frequency for a
  well-posed (adequately-absorbing PML, non-resonant) configuration.
- **Mode-tracking correctness**: on the synthetic near-degenerate test case (build step 5),
  tracking selects the physically continuous mode where plain $\beta$-sort demonstrably would
  not — this is the direct regression test for the specific failure Module 4's implementation
  found.
- **Port congruence check** (§6.3): tracked $\gamma_m,Y_m$ at Port 1 and Port 2 agree closely at
  every frequency, given the geometrically congruent cross-sections — a strong, cheap,
  geometry-specific consistency check.
- **The big one — end-to-end Phase 1 gate**: for a uniform, lossless microstrip line (no LC,
  Phase 1 of the top-level plan), $S_{21}=S_{12}$ and $|S_{11}|^2+|S_{21}|^2=1$ across the swept
  band, and the dominant mode's extracted $\beta(\omega)$ matches Hammerstad–Jensen/Wheeler. If
  this fails, the first two places to check are Module 4 §3.6/§5.1's explicitly-flagged sign
  and arrangement questions — this module's assembly is the first point where those questions
  become empirically checkable rather than merely flagged.

---

## 10. Interface / class contract

```
# solve.system
def build_restriction(pec_dofs: set[int], n_edges: int) -> sparse matrix   # R, cached once
def reduce_system(A: sparse, b: array, R: sparse) -> (A_ff: sparse, b_f: array)
def recover_solution(a_f: array, R: sparse) -> array                        # a = R^T a_f
def factor(A_ff: sparse, *, components: dict[str, sparse] = None) -> Factorization   # complex-symmetric;
                                                                              # components: optional named
                                                                              # terms summing to A_ff, used
                                                                              # only to localize a symmetry
                                                                              # failure's message (post-review)
def solve_with_factorization(fact: Factorization, b_f: array) -> array

# solve.sweep
class TrackingState:
    modes: dict[str, list[PortMode]]   # per port tag, the currently-tracked modes

def track_modes(candidates: dict[str, list[PortMode]], state: Optional[TrackingState],
                 is_first_step: bool) -> (dict[str, list[PortMode]], TrackingState)

def run_sweep(mesh, interior_materials, port_tags: list[str], frequencies: list[float],
              n_modes: int = 2, pml_params: dict = ...) -> list[SweepResult]

class SweepResult:
    omega: float
    excitation: tuple[str, int]      # (port_tag, mode_index)
    a: array                         # full (unconstrained-space) solution vector
    port_modes: dict[str, list[PortMode]]   # this frequency's tracked modes, for Module 7
```

Module 7 is the sole downstream consumer of `SweepResult` — it needs nothing from this module
beyond the list of `(omega, excitation, a, port_modes)` tuples the sweep produces.
