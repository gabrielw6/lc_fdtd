# Module 5 — `pml`: Perfectly Matched Layer — Equations & Implementation Plan

Companion to the top-level architecture doc and Modules 0–4. Same conventions: no code,
precise equations with derivations, step-by-step build order, validation targets. One finding
here is important enough to flag up front rather than bury in §4: **the PML tensor's own
normal-direction component does not satisfy Module 2's generic passivity check**, and this is
expected, not a bug — §4 derives why and specifies the required exception to Module 2's
contract.

Because Module 0 fixed PML as **top-only, single-axis** (§0 point 3 there — no lateral PML,
sidestepping the LC-adjacency conflict entirely), this module is considerably simpler than a
general 3D corner-graded PML. That simplification is used throughout below rather than
re-deriving the general case and then restricting it.

---

## 0. What Module 5 owns vs. consumes

**Consumes**: Module 0's geometric parameters for the PML shell — $z_{\text{air,top}}$ (where
the PML begins) and $h_{\text{pml}}$ (its thickness) — and the `PML_TOP` volume tag; Module 2's
`MaterialModel` interface, which this module implements rather than extends; a background
`MaterialModel` to wrap (in this geometry, always `AIR`'s `ConstantMaterial(eps_r=1.0)`, since
Module 0 guarantees the PML shell sits entirely above the air region, never adjacent to
substrate).

**Owns**: the complex coordinate-stretching tensor construction (`pml.stretching`) and the
`PMLMaterial` class that packages it as a `MaterialModel` (Module 2 §1.1's contract).

**Does not own**: volume assembly (Module 3 — reused unchanged via its tet-subset support,
Module 3 §5.2); the frequency sweep that reconstructs `PMLMaterial` each iteration (Module 6);
ports (Module 4, unaffected by PML since ports sit in the substrate+air region, never touching
the PML shell, by the same geometric guarantee noted above).

---

## 1. Recap of design invariants already fixed elsewhere

- **Top-only truncation, single axis** (Module 0 §0.3): the PML shell stretches only $z$; the
  lateral faces are `PMC_SIDE` (natural boundary), not PML. This means $s_x=s_y=1$ everywhere
  in this application — only $s_z$ is ever non-trivial.
- **Spatial disjointness from LC** (top-level doc §5, Module 0 §0.3): guaranteed by geometry,
  not by anything this module does.
- **Background is always `AIR`** for this geometry (Module 0's stacking: substrate → air →
  PML), so $\varepsilon_{r,\text{bg}}=\mu_{r,\text{bg}}=\mathbf I$ here specifically — though
  §3.1 architects `PMLMaterial` to wrap a general background `MaterialModel`, not a hardcoded
  one, in case a future geometry variant places PML against a different background.
- **PEC-backed outer wall** (Module 0 §4.3, `PML_OUTER_PEC`): the shell's own exterior faces
  are PEC — already a boundary-condition detail Module 6 applies, not something this module
  computes.

---

## 2. Complex coordinate stretching

### 2.1 General tensor, then the single-axis specialization actually used

The general 3-axis complex-coordinate-stretching tensor (top-level doc §5.1):

$$\Lambda = \mathrm{diag}\!\left(\frac{s_ys_z}{s_x},\ \frac{s_zs_x}{s_y},\ \frac{s_xs_y}{s_z}\right)$$

With $s_x=s_y=1$ (§1), this collapses to:

$$\boxed{\;\Lambda = \mathrm{diag}(s_z,\ s_z,\ 1/s_z)\;}$$

— the two axes transverse to the stretched direction scale **up** by $s_z$, the stretched axis
itself scales by its **reciprocal**. This asymmetry (not $s_z$ in all three slots) is the
detail that makes PML matched rather than just lossy — and it's also the source of §4's
passivity finding, since inverting a complex number doesn't preserve the sign of its imaginary
part the way scaling by it does.

### 2.2 The stretch factor

$$s_z(\omega,\xi) = \kappa(\xi) - j\,\frac{\sigma(\xi)}{\omega\varepsilon_0}, \qquad \kappa(\xi)\ge1,\ \sigma(\xi)\ge0$$

where $\xi$ is depth into the PML (§2.5).

### 2.3 Grading profiles

Standard polynomial grading, order $n$ (typically 2–3), shared between $\sigma$ and $\kappa$
for simplicity (splitting into independent orders is a config option, not needed by default):

$$\sigma(\xi) = \sigma_{\max}\left(\frac{\xi}{d}\right)^{n}, \qquad \kappa(\xi) = 1 + (\kappa_{\max}-1)\left(\frac{\xi}{d}\right)^{n}, \qquad d = h_{\text{pml}}$$

$\kappa_{\max}>1$ helps absorb near-grazing/evanescent components that the imaginary
(pure-loss) term alone handles less efficiently — this is the reason $\kappa$ grading exists at
all, not merely an extra free parameter.

### 2.4 Deriving $\sigma_{\max}$ from a target reflection coefficient

This is worth deriving rather than just stating, since it's the actual justification for why
the formula has no $\omega$-dependence — a property that matters for the frequency-sweep
architecture (§6).

**One-way attenuation.** In the stretched coordinate $\tilde z=\int_0^\xi s_z(\xi')\,d\xi'$, a
plane wave $e^{-jk_0\tilde z}$ becomes

$$e^{-jk_0\tilde z} = \underbrace{e^{-jk_0\int_0^\xi\kappa\,d\xi'}}_{\text{phase only}}\cdot\underbrace{e^{-k_0\int_0^\xi \sigma/(\omega\varepsilon_0)\,d\xi'}}_{\text{real decay}}$$

using $-jk_0\cdot(-j)=-k_0$ on the imaginary part of $s_z$. The decay exponent simplifies using

$$\frac{k_0}{\omega\varepsilon_0} = \frac{\omega\sqrt{\mu_0\varepsilon_0}}{\omega\varepsilon_0} = \sqrt{\frac{\mu_0}{\varepsilon_0}} = \eta_0$$

— **the $\omega$'s cancel exactly**, so the one-way attenuation (in nepers) for a *given*
$\sigma(\xi)$ profile is $\eta_0\int_0^d\sigma(\xi)\,d\xi$, independent of frequency. This is
precisely why $\sigma_{\max}$ can be set once, not per frequency point.

**Round trip and target reflection.** A wave reaching the PEC backing reflects and traverses
the PML a second time, so the total (amplitude) attenuation is $e^{-2\eta_0\int_0^d\sigma\,d\xi}$.
Setting this equal to a target normal-incidence amplitude reflection coefficient $R_0$ (typical
values $10^{-5}$–$10^{-6}$) and using $\int_0^d\sigma_{\max}(\xi/d)^n\,d\xi = \sigma_{\max}d/(n+1)$:

$$e^{-2\eta_0\sigma_{\max}d/(n+1)} = R_0 \;\Longrightarrow\; \boxed{\;\sigma_{\max} = \frac{-(n+1)\ln R_0}{2\eta_0 d}\;}$$

matching the top-level doc's §5.2 formula exactly, with $\eta=\eta_0$ specifically because the
background here is air (§1) — not a general background impedance that would need computing
per-material.

**Two properties worth noting explicitly:**

- **Normal incidence is the conservative case.** PML's defining property is that it is
  reflectionless at *any* angle of incidence in the continuous (undiscretized) limit — but for
  a wave arriving obliquely, the physical path length through the shell is longer
  ($\propto 1/\cos\theta$), so it experiences *more* attenuation, not less. The formula above,
  derived at normal incidence, is therefore an upper bound on reflection, not a special case
  that might be beaten at other angles.
- **This is a target for the continuous PML, not a guarantee on the discretized one.** Actual
  numerical reflection will exceed $R_0$ if the mesh under-resolves the graded profile — §5.3
  addresses this.

### 2.5 The depth coordinate $\xi$

$$\xi = z - z_{\text{air,top}}, \qquad \xi \in [0, h_{\text{pml}}]$$

using Module 0's already-fixed $z$-levels — no new geometric parameters, just the two already
threaded through `GeometryParams`.

---

## 3. `PMLMaterial` — the `MaterialModel` implementation

### 3.1 General wrapping formula, specialized for this geometry

$$\varepsilon_r^{\text{PML}}(\mathbf r) = \Lambda(\mathbf r)\,\varepsilon_{r,\text{bg}}(\mathbf r), \qquad \mu_r^{\text{PML}}(\mathbf r) = \Lambda(\mathbf r)\,\mu_{r,\text{bg}}(\mathbf r)$$

architected as a matrix product against a general background `MaterialModel` (dependency
injection — `PMLMaterial(background, ...)`), so a future geometry variant with PML against a
different (still isotropic, per the design invariant) background needs no code change here.
**For this specific geometry**, background is always `AIR`
($\varepsilon_{r,\text{bg}}=\mu_{r,\text{bg}}=\mathbf I$), so both reduce to exactly $\Lambda$
itself:

$$\varepsilon_r^{\text{PML}}(\mathbf r) = \mu_r^{\text{PML}}(\mathbf r) = \Lambda(\mathbf r) = \mathrm{diag}(s_z,\ s_z,\ 1/s_z)$$

— $\varepsilon_r$ and $\mu_r$ are **identical** here, a genuine simplification (not an
approximation) worth exploiting: compute $\Lambda$ once per call and return it for both.

### 3.2 Vectorized evaluation

Given `points` of shape $(M,3)$ (global $(x,y,z)$ coordinates, per the whole codebase's
convention): extract $\xi = \text{points}[:,2] - z_{\text{air,top}}$ (the $z$-column — a simple
but easy-to-transpose detail worth stating explicitly), compute $\sigma(\xi),\kappa(\xi)$ (§2.3),
$s_z(\omega,\xi)$ (§2.2), and assemble the $(M,3,3)$ diagonal array — all vectorized, no
per-point Python loop.

### 3.3 What Module 3 actually needs: $\mu_r^{-1}$

Module 3 §3.1 integrates $\mu_r^{-1}$, not $\mu_r$. Since $\Lambda$ is diagonal, its inverse is
the diagonal of reciprocals — exact and cheap, no general $3\times3$ solve needed:

$$\Lambda^{-1} = \mathrm{diag}(1/s_z,\ 1/s_z,\ s_z)$$

`PMLMaterial.mu(points)` can either return $\Lambda$ directly (letting Module 3's generic
per-point inversion handle it, per that module's architecture) or expose the already-inverted
diagonal as an optimization — either is correct; the generic path costs nothing extra for a
diagonal matrix, so there's no strong reason to special-case it beyond documenting that the
inversion is exact when it happens.

### 3.4 Construction is per-frequency — no interface change

Per Module 3 §5.4's resolution: `PMLMaterial(background, omega, z_air_top, h_pml, R0, n,
kappa_max)` is **constructed fresh for each frequency point** by Module 6's sweep driver. Its
`epsilon(points)`/`mu(points)` methods still take no `omega` argument — the frequency is baked
in at construction, keeping Module 2's interface untouched. This module does not change that
resolution; it's restated here only to make explicit that `PMLMaterial` is the concrete class
that resolution was written for.

---

## 4. The passivity exception — why $\Lambda_{zz}$ fails Module 2's generic check, and why that's correct

### 4.1 The finding

Write $s_z = \kappa - jb$ with $b=\sigma/(\omega\varepsilon_0)\ge0$. Then

$$\frac{1}{s_z} = \frac{\kappa+jb}{\kappa^2+b^2} \;\Longrightarrow\; \mathrm{Im}\!\left(\frac{1}{s_z}\right) = \frac{b}{\kappa^2+b^2} \ge 0$$

Since $\Lambda_{zz}=1/s_z$, this means $\mathrm{Im}(\varepsilon_{r,zz}^{\text{PML}}) \ge 0$ —
**the opposite sign** from Module 2 §1.3's passivity requirement
($\mathrm{Im}(\varepsilon_r)\preceq0$ under the $e^{+j\omega t}$ convention). Module 2's generic
per-tensor check (an eigenvalue test on $-\mathrm{Im}(\varepsilon_r)$ being PSD) will — correctly,
given its own criterion — flag this component as failing.

### 4.2 Why this is not a gain medium

PML is not an ordinary passive dielectric; it's a complex-coordinate-stretching construct that
satisfies a modified form of Maxwell's equations chosen specifically to produce a decaying,
(ideally) reflectionless wave — it has no physical realization as an ordinary crystal or
material, and standard component-wise "$\mathrm{Im}(\varepsilon_r)\le0$" passivity criteria,
which are stated for ordinary anisotropic *passive* media, don't directly characterize it. The
correct demonstration that PML absorbs rather than amplifies energy is the **wave-attenuation
argument of §2.4** — the plane wave's amplitude genuinely decays through the shell (a real,
directly-computed exponential decay), which is the physically meaningful statement of
"absorbing," not the abstract sign of one diagonal tensor component in isolation.

### 4.3 Required action: exempt `PMLMaterial` from Module 2's generic passivity check

**This is a necessary small addition to Module 2's contract**, applied to that document now:
`material.core`'s generic passivity check (§1.3 there) must **not** be asserted against
`PMLMaterial` — either by having `MaterialAssembly` skip the check for this specific
`MaterialModel` subclass, or by `PMLMaterial` overriding a check-applicability flag. The
*correct* correctness criterion for PML is the reflection test (§5.1 below), not the generic
per-tensor check. Flagging this now prevents two bad outcomes: an implementer's test harness
wrongly rejecting a correct `PMLMaterial`, or — worse — someone "fixing" the sign of
$\Lambda_{zz}$ to satisfy the generic check, which would silently break the PML's matching
property (§2.1's asymmetric $1/s_z$ term is exactly what makes it matched; flipping its sign
to look passive would un-match it).

---

## 5. Reflection theory as the actual validation basis

### 5.1 Restating §2.4's result as the correctness criterion

The target $R_0$ (§2.4) is what a correctly implemented PML should approach, in $|S_{11}|$, for
a matched line terminated by the shell. This is the real test; §4's per-component passivity
check is simply the wrong tool for this material and is not used here.

### 5.2 Sources of residual reflection beyond the continuous-PML theory

- **Finite thickness / round trip**: already captured in the $R_0$ formula (§2.4) — this is
  not a source of *additional* error, just the theoretical floor itself.
- **Discretization within the PML**: a finite element mesh that under-resolves the graded
  $\sigma(\xi),\kappa(\xi)$ profile introduces reflection beyond $R_0$. **No new mechanism is
  needed to catch this** — Module 3 §4's adaptive quadrature-order procedure, applied uniformly
  to every element regardless of tag, will either successfully integrate a smooth-enough
  profile or hit its max-order raise if the mesh genuinely can't resolve it. A guideline worth
  stating for mesh generation (not a hard requirement enforced in code): at least 4–6 elements
  across the PML thickness for typical orders $n=2$–$3$.
  - **This distinguishes two different failure signatures** worth telling apart during
    validation: a $|S_{11}|$ floor that's too high **at all frequencies** more often points to
    under-resolved mesh inside the shell; a floor that rises specifically **at low frequency**
    (per the top-level doc §5.3) more often points to insufficient $\kappa_{\max}$ for
    near-grazing/evanescent absorption. Both ultimately show up as the same symptom
    (elevated $|S_{11}|$), so knowing which knob to check first is useful diagnostic guidance,
    not just a restatement of the test.
- **Evanescent components**: motivates $\kappa_{\max}>1$ (§2.3) — the imaginary
  (pure-loss) term alone is less effective on components that don't have a real propagation
  angle to begin with.

---

## 6. Integration with Modules 3 and 6

Restating Module 3 §5.2/§5.4's already-established resolution, concretely for this module: at
each frequency $\omega_k$ in the sweep, Module 6 (a) constructs a fresh
`PMLMaterial(background=AirMaterial, omega=ω_k, z_air_top, h_pml, R0, n, kappa_max)`, (b) wraps
it in a `MaterialAssembly` entry for the `PML_TOP` tag, and (c) calls Module 3's `assemble(...)`
restricted to the `PML_TOP` tet subset, producing $\mathbf K_{\text{pml}}(\omega_k),
\mathbf M_{\text{pml}}(\omega_k)$ to be added to the cached interior operators. This module
contributes nothing else to that call sequence — it is entirely encapsulated in what
`PMLMaterial` returns from `epsilon`/`mu`.

---

## 7. Step-by-step build order

1. `pml.stretching`: $\sigma(\xi),\kappa(\xi)$ profile functions (§2.3) and the $\sigma_{\max}$
   derivation (§2.4) as a standalone function of $(R_0,n,d,\eta_0)$ — unit-test that it takes
   no frequency argument (a structural check that the frequency-independence property of §2.4
   wasn't accidentally broken in the implementation).
2. $s_z(\omega,\xi)$ (§2.2) and $\Lambda$ construction (§2.1), vectorized over an array of
   $\xi$ values — unit test the reduction $\sigma\to0,\kappa\to1 \Rightarrow \Lambda\to\mathbf I$
   before anything else.
3. `PMLMaterial` (§3): wire it to a background `MaterialModel` (start with a hardcoded
   `ConstantMaterial(eps_r=1.0)` for air, matching this geometry, while keeping the constructor
   general per §3.1); implement `epsilon`/`mu` (§3.2), confirming both return identical values
   for this geometry (§3.1's simplification) as a cheap built-in check.
4. Apply the Module 2 contract exception (§4.3) — this is a small, required edit to
   `module2_material_equations.md`, applied alongside this document.
5. Reflection test harness (§5.1): requires Modules 3, 4, and 6 to exist first (it's an
   integration test spanning the whole assembled system), so this step is blocked until those
   are implemented — record it now so it isn't forgotten once they are.
6. Run the full §8 validation suite once the reflection test is unblocked.

---

## 8. Validation targets

- **Frequency-independence of $\sigma_{\max}$** (§2.4): the derived formula must not contain
  $\omega$ — a structural/code-review check as much as a numeric one.
- **Reduction check**: $\sigma\to0,\kappa\to1 \Rightarrow \Lambda\to\mathbf I$ exactly, and a
  full solve with the PML region given this trivial material must reproduce the same-mesh
  solve with no PML flagged at all, to solver tolerance (top-level doc §5.3).
- **$\varepsilon_r=\mu_r$ for this geometry** (§3.1): confirmed identical at every evaluated
  point, since both wrap the same $\mathbf I$ background.
- **Diagonal-inverse exactness** (§3.3): $\Lambda\cdot\Lambda^{-1}=\mathbf I$ to machine
  precision at sampled points.
- **Reflection test** (§5.1, blocked on Modules 3/4/6): $|S_{11}|$ approaches $R_0$ across the
  band for a matched line terminated by the PML; a low-frequency-specific rise diagnoses
  insufficient $\kappa_{\max}$; an across-the-band elevated floor diagnoses mesh under-resolution
  within the shell (§5.2) — check mesh density before touching $\kappa_{\max}$ or $R_0$ if the
  broad-band symptom appears.
- **Passivity-check exemption verified** (§4.3): confirm `MaterialAssembly` (or equivalent)
  actually skips the generic check for `PMLMaterial` — i.e., that the exemption is wired in,
  not merely documented.

---

## 9. Interface / class contract

```
# pml.stretching
def sigma_max_for_R0(R0: float, n: int, thickness: float, eta0: float = 376.730313668) -> float
def sigma_profile(xi: array, thickness: float, sigma_max: float, n: int) -> array
def kappa_profile(xi: array, thickness: float, kappa_max: float, n: int) -> array
def s_z(xi: array, omega: float, thickness, sigma_max, kappa_max, n) -> complex array
def lambda_tensor(s_z: complex array, s_x=1.0, s_y=1.0) -> (M,3,3) array   # general form; s_x=s_y=1 by default here

# pml.PMLMaterial
class PMLMaterial(MaterialModel):
    def __init__(self, background: MaterialModel, omega: float,
                 z_air_top: float, thickness: float,
                 R0: float = 1e-6, n: int = 2, kappa_max: float = 1.0): ...
    def epsilon(points: (M,3)) -> (M,3,3)   # = Lambda(points) @ background.epsilon(points)
    def mu(points: (M,3)) -> (M,3,3)        # = Lambda(points) @ background.mu(points)
```

Module 6 is the sole caller of the constructor (once per frequency, per §6); Module 3's
`assemble(...)` is the sole caller of `epsilon`/`mu` (via the `PML_TOP`-tagged
`MaterialAssembly` entry), through the exact same code path used for every other tag — PML
requires no special-casing anywhere outside this module and the one documented exception to
Module 2's passivity check.
