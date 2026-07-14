# Module 8 — `validation`: Analytic References, Shared Checks, Phase-Gated Acceptance Plan

Companion to the top-level architecture doc and Modules 0–7. This module is different in kind
from the others: it derives no new field physics. What it owns is (a) the **actual analytic
reference formulas** that "compare against Hammerstad–Jensen" has meant, unstated, since the
very first design conversation of this project; (b) a **shared library** of the
symmetry/reciprocity/passivity/convergence checks that every other module has been performing
ad hoc, consolidated so they're implemented and tested exactly once; and (c) the **phase-gated
test architecture** that ties Modules 0–7's already-scattered validation targets into one
sequenced, enforceable plan.

---

## 0. What Module 8 owns vs. consumes

**Consumes**: nothing computationally — it reads the validation targets already specified in
every other module's own document and organizes them; it does not re-derive the physics behind
them.

**Owns**: `validation.analytic_microstrip` (§1) — the actual Hammerstad–Jensen equations, now
precisely stated and numerically verified against a worked example, not merely cited by name;
`validation.checks` (§2) — a shared library of generic correctness checks, replacing repeated
ad hoc implementations across Modules 2–7; the mesh-refinement convergence-study methodology
(§3); the phase-gated test architecture and known-limitations registry (§4); a consolidated
validation-target matrix (§5) mapping every check already specified in Modules 0–7 to its phase
and status.

**Does not own**: any module's own internals — this is purely a test/validation layer sitting
above all of them.

---

## 1. `validation.analytic_microstrip` — the actual reference formulas

### 1.1 Effective permittivity (Hammerstad–Jensen, full form)

For substrate height $h$, trace width $w$, relative permittivity $\varepsilon_r$, with
$u=w/h$:

$$\varepsilon_e = \frac{\varepsilon_r+1}{2} + \frac{\varepsilon_r-1}{2}\left(1+\frac{10h}{w}\right)^{-a\cdot b}$$

$$a(u) = 1 + \frac{1}{49}\ln\!\left[\frac{u^4+(u/52)^2}{u^4+0.432}\right] + \frac{1}{18.7}\ln\!\left[1+\left(\frac{u}{18.1}\right)^3\right]$$

$$b(\varepsilon_r) = 0.564\left[\frac{\varepsilon_r-0.9}{\varepsilon_r+3}\right]^{0.053}$$

**Accuracy**: better than $0.2\%$ for $0.01\le w/h\le100$ and $1\le\varepsilon_r\le128$ (Steer,
*Fundamentals of Microwave and RF Design*, citing Hammerstad & Jensen 1980) — this is a
quasi-static (frequency-independent) formula; §1.4 addresses what that means for validation at
higher swept frequencies.

**Simplified approximation** (optional, faster, up to $\sim1\%$ error, worst for narrow lines
and extreme $\varepsilon_r$) — useful for a quick sanity check but not the primary reference:

$$\varepsilon_e \approx \frac{\varepsilon_r+1}{2} + \frac{\varepsilon_r-1}{2}\cdot\frac{1}{\sqrt{1+12h/w}}$$

### 1.2 Characteristic impedance

$$Z_0 = \frac{Z_{0,\text{air}}}{\sqrt{\varepsilon_e}}, \qquad Z_{0,\text{air}} = 60\ln\!\left[\frac{F_1 h}{w} + \sqrt{1+\left(\frac{2h}{w}\right)^2}\right]$$

$$F_1 = 6 + (2\pi-6)\exp\!\left\{-\left(30.666\,\frac{h}{w}\right)^{0.7528}\right\}$$

**Accuracy**: better than $0.1\%$ for $w/h<1000$. This single continuous formula (via $F_1$)
replaces the older two-piece Wheeler-style formulas split at $w/h=1$ — no piecewise branching
needed in the implementation.

### 1.3 Propagation constant

$$\beta(\omega) = k_0(\omega)\sqrt{\varepsilon_e} = \omega\sqrt{\mu_0\varepsilon_0\varepsilon_e}$$

— the quantity Phase 1's gate compares the FEM solver's dominant-mode $\mathrm{Im}(\gamma)$
against.

### 1.4 Validity caveat — quasi-static vs. the full-wave solver

These formulas have **no frequency dependence in $\varepsilon_e$** — they are quasi-static
(DC/low-frequency) approximations. Real microstrip exhibits **dispersion**: $\varepsilon_e$
genuinely increases with frequency as fields concentrate more into the substrate. This means:

- Agreement should be tightest at the **low-frequency end** of the swept band (exactly where
  the top-level doc's Phase 1 gate already specifies checking).
- A small, **growing** deviation at higher swept frequencies is expected physics, not
  necessarily a solver bug — do not chase it as an error without first checking whether it's
  consistent with known dispersion behavior (fields shifting toward the substrate as frequency
  rises).
- If tighter validation across the full band is ever wanted, a dispersion-corrected model
  (Kirschning & Jansen, 1982) exists in the literature — cited here, not derived, since it's
  a more elaborate empirical fit than is safe to reproduce from memory without the same
  worked-example verification applied to §1.1–1.2 below.

### 1.5 Verification — a worked example, checked by hand

Since this reference is what everything else is graded against, it needs its own check before
being trusted. Steer's textbook gives a complete worked example: $w=600\,\mu\text{m}$,
$h=635\,\mu\text{m}$, $\varepsilon_r=4.1$ ($u=0.945$), at $f=5\,\text{GHz}$. Recomputing by hand
from §1.1–1.3 above reproduces every stated intermediate value exactly: $a=0.991$, $b=0.541$,
$\varepsilon_e=2.967$, $Z_{0,\text{air}}=129.7\,\Omega$, $Z_0=75.4\,\Omega$, and
$\beta=180.5\,\text{rad/m}$ at 5 GHz. **This exact test case should be the first unit test
written for `validation.analytic_microstrip`** — before it is ever used to grade the FEM
solver, confirm it reproduces this independently-published worked example precisely.

---

## 2. `validation.checks` — shared generic check library

Consolidating patterns already specified ad hoc across Modules 2–7, implemented and tested
**once** rather than reimplemented (and potentially subtly inconsistently) in each module's own
test suite.

### 2.1 `assert_symmetric(matrix, tol)`

$\|M-M^T\|_F / \|M\|_F < \text{tol}$. Used for: Module 2's material tensor check, Module 3's
element and global $\mathbf K,\mathbf M$, Module 4's $\mathbf B_p$ and the $S_{tz}=-T_{zt}^T$
identity. **Tolerance**: machine precision (e.g. $10^{-10}$ relative) — this is an exact
algebraic identity in every one of these cases, not an approximate one, so a loose tolerance
would mask a real bug.

### 2.2 `assert_passive(tensor_or_S, tol, exceptions=[])`

For a material tensor: eigenvalues of $-\mathrm{Im}(\varepsilon_r)$ (or $\mu_r$) are
$\ge-\text{tol}$ — **with the documented `PMLMaterial` exemption** (Module 2 §1.3, Module 5
§4) wired in as a required skip-list entry, not an oversight to remember separately each time
this check is invoked. For an S-parameter set: the **extended** energy-conservation sum
(Module 7 §5), $\sum_{p,m}|S_{(p,m),(q,1)}|^2 \le 1+\text{tol}$ — not the naive two-term
dominant-only sum, which (per Module 7 §5) can legitimately read below 1 for a correct
mode-converting structure.

### 2.3 `assert_reciprocal(S_dominant, tol)`

$S_{(p,1),(q,1)} = S_{(q,1),(p,1)}$. **Scoped explicitly to $S^{\text{dominant}}$ only** — per
Module 7 §3.3, full cross-mode reciprocity cannot be checked with the established single-mode
excitation convention, and this function should not silently pretend otherwise.

### 2.4 Convergence-order estimation

A generic helper used by: Module 1 §6.3's quadrature exactness checks, Module 3 §4's adaptive
quadrature-order doubling, and §3 below's mesh $h$-refinement study. Given results at two
resolutions (or orders) and the nominal refinement ratio, estimate the observed convergence
order via

$$p_{\text{obs}} = \frac{\ln(\|e_1\|/\|e_2\|)}{\ln(\text{refinement ratio})}$$

where $e_1,e_2$ are errors (or successive differences, when the exact answer is unknown) at
consecutive refinement levels — one implementation, reused everywhere a "confirm convergence
order" check is needed rather than three slightly different ad hoc versions.

### 2.5 Reduction-check helper

A generic "configuration A's result matches configuration B's to tolerance" comparator, used by
**every** reduction check across this whole project: Phase 2→1, Phase 3→1, Phase 4→3 (top-level
doc), Module 5's $\sigma\to0,\kappa\to1$ reduction, Module 0's LC-material-set-to-substrate
geometric reduction. One implementation, parametrized by which two result objects and which
tolerance, rather than a bespoke comparison written fresh at each of these call sites.

### 2.6 Tolerance guidance — tying validation tolerance to the PML's own floor

**A concrete recommendation, not just a placeholder number**: the reciprocity/energy-check
tolerance (§2.2, §2.3) should not be set tighter than the PML's own target reflection $R_0$
(Module 5 §2.4) — a well-converged solve cannot do better than the residual reflection the PML
itself permits, so demanding, say, $10^{-8}$ agreement when $R_0=10^{-6}$ would be demanding
precision the truncation boundary structurally cannot deliver, and any apparent "failure" at
that tightness is measuring the PML floor, not a bug. Set these tolerances a small factor
(e.g. $5$–$10\times$) looser than $R_0$, not tighter.

---

## 3. Mesh $h$-refinement convergence study — methodology

The phrase "converging under mesh refinement" has been repeated since Phase 1's first
statement without a concrete procedure. Here it is:

1. Generate the same geometry (Module 0) at **3–4 refinement levels**, each roughly halving
   the characteristic element size (a refinement ratio of $\sim2$ per level is standard and
   matches the doubling-based convergence checks already used elsewhere in this project for
   consistency).
2. At each level, extract the quantity being validated (e.g. dominant-mode $\beta$ or $Z_0$ at
   a fixed frequency).
3. Apply §2.4's convergence-order estimator to the sequence of results.
4. **Acceptance criterion**: the observed order should be consistent with the expected order for
   lowest-order Nédélec/Whitney elements (first-order accurate in the relevant field
   quantities) — a very low or non-monotonic observed order signals either an under-resolved
   mesh, a bug, or a quantity that isn't actually convergent (e.g. comparing against a
   quasi-static reference at a frequency where dispersion, §1.4, is already significant,
   which would show up as a convergence *plateau* at a nonzero offset from the reference, not
   divergence — worth distinguishing these two failure signatures).

---

## 4. Phase-gated test architecture

### 4.1 Consolidated phase definitions (recap, now in one place)

| Phase | What changes | Primary gate |
|---|---|---|
| 1 | Uniform isotropic microstrip | §1's analytic $\beta,Z_0$; reciprocity; $\|S_{11}\|^2+\|S_{21}\|^2=1$; $h$-convergence (§3) |
| 2 | Position-dependent scalar $\varepsilon_r$ | Reduction to Phase 1; layered-dielectric reference; quadrature-order sensitivity |
| 3 | General symmetric tensor, direct input | Axis-aligned analytic check; reciprocity/passivity as invariants for rotated axes |
| 4 | LC via director field | Module-boundary contract (uniform director = Phase 3 tensor, bit-for-bit); reciprocity/passivity; extended energy conservation with mode-conversion (Module 7 §5); symmetry-forbidden-conversion test (Module 7, in-plane director $\Rightarrow$ exact zero conversion) |

### 4.2 Blocking mechanism

Each phase's test suite is marked with an explicit phase dependency (e.g. a test-framework
marker or a simple dependency list in the test runner configuration); the runner refuses to
execute Phase $N{+}1$'s tests if any of Phase $N$'s gate tests are failing. This operationalizes
`CLAUDE.md` §7's instruction ("do not implement Phase 3 tensor handling while Phase 1 gates are
red") as an actual enforced mechanism rather than a discipline the implementer has to remember
unaided.

### 4.3 Known-limitations registry

A living record — **owned in structure here, maintained as a current instance in `CLAUDE.md`**
— of open honesty flags and their resolution status, so that "this was flagged as uncertain" and
"this has since been closed by test X" are both tracked in one place rather than scattered
across module docs that don't get revisited. Current entries at time of writing:

| Flag | Origin | Status |
|---|---|---|
| Port eigenproblem block arrangement | Module 4 §3.6 | Open — gated on Phase 1's real-$\gamma^2$ and analytic-$\beta$ checks, both passing per Module 4's implementation report |
| Port operator overall sign | Module 4 §5.1 | Open — gated on Phase 1's reciprocity/passivity gate (not yet run end-to-end at time of Module 6/7 specs) |
| Box-mode/quasi-TEM mode selection | Module 4 §3.7 | Addressed by Module 6 §6's frequency-tracking algorithm — validate against Module 6 §8 build step 5's synthetic near-degenerate test before considering closed |
| Projection self-normalization convention | Module 4 §4.3 | Resolved in implementation (self-normalizing fix applied); confirm which of the two equivalent conventions is in effect before relying on Module 6 §6.2's simplified tracking-overlap form |

### 4.4 Reporting

Once Phase 1's gate runs (the first point in this whole project where it becomes empirically
checkable, per Module 6/7), its pass/fail result is what actually closes out the two open Module
4 flags above — record the outcome here rather than leaving those flags open indefinitely once
they've been resolved by a passing gate.

---

## 5. Consolidated validation-target matrix

A compact index of every check already specified in Modules 0–7, for navigation rather than
re-derivation (each row's full description lives in its origin module's document):

| Origin | Check | Phase |
|---|---|---|
| Module 0 §6 | Bounding-volume, tag coverage, trace-containment | 1 |
| Module 1 §8 | Volume/gradient identities, edge orientation, quadrature exactness | 1 |
| Module 2 §7 | Symmetry/passivity, interpolation partition-of-unity, module-boundary contract | 3–4 |
| Module 3 §7 | Local/global basis checks, $\mathbf K,\mathbf M$ symmetry, reference-tet check | 1 |
| Module 4 §8 | Real $\gamma^2$, analytic $\beta$, biorthogonality, bounds filter | 1 |
| Module 5 §8 | Reduction, $\varepsilon_r=\mu_r$, reflection test (blocked on Modules 3/4/6) | 1 (PML present) |
| Module 6 §9 | PEC elimination, multi-RHS reuse, mode-tracking, port congruence, **Phase 1 gate** | 1 |
| Module 7 §8 | Tangential-trace geometric fact, extended energy conservation, de-embedding sanity, symmetry-forbidden-conversion | 1, 4 |

---

## 6. Step-by-step build order

1. `validation.analytic_microstrip` (§1), verified against the worked example (§1.5) **before**
   it is used to grade anything else.
2. `validation.checks` (§2), each function unit-tested against hand-built cases with a known
   answer (a matrix that's exactly symmetric and one that isn't; an $S$ set that's exactly
   reciprocal and one that isn't) — this library is trusted by every phase gate downstream, so
   it needs the same "verify before relying on it" treatment as §1.
3. The convergence-study procedure (§3), tested on a case with a known analytic answer so the
   observed-order computation itself can be checked against an expected value.
4. The phase-gating/blocking mechanism (§4.2) — wire it into the test runner configuration.
5. Run Phase 1's full gate end-to-end — this is the point that closes out the two remaining
   open Module 4 flags (§4.3), assuming it passes.
6. Proceed through Phases 2–4 only as each prior phase's gate goes green, per §4.2's
   enforcement.

---

## 7. Validating the validator

Since this module is the trust anchor for everything else, it gets the same scrutiny: §1.5's
worked-example check and §2's hand-built known-answer unit tests are not optional scaffolding,
they're what earns this module the right to grade the other seven.

---

## 8. Interface / class contract

```
# validation.analytic_microstrip
def eps_eff(eps_r: float, w: float, h: float) -> float
def z0(eps_r: float, w: float, h: float) -> float
def beta(eps_r: float, w: float, h: float, omega: float) -> float

# validation.checks
def assert_symmetric(M: sparse or array, tol: float = 1e-10) -> None
def assert_passive(tensor_or_S, tol: float, exceptions: list[type] = []) -> None
def assert_reciprocal(S_dominant: array, tol: float) -> None
def estimate_convergence_order(errors: list[float], refinement_ratio: float) -> float
def assert_reduction(result_a, result_b, tol: float) -> None

# validation.gates
def run_phase_gate(phase: int, results) -> GateReport
class GateReport:
    phase: int
    passed: bool
    failures: list[str]

def run_convergence_study(build_fn, refinement_levels: list[float], quantity_fn) -> ConvergenceResult
```

This is the last module in the architecture. With Modules 0–8 specced, the remaining work is
implementation against these documents, in the phase order Module 8 §4.2 enforces.
