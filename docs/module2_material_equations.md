# Module 2 — `material`: Constitutive Tensor Field — Equations & Implementation Plan

Companion to `architecture_fem_sparameter_modules.md`, `module0_geometry_builder_equations.md`,
and `module1_mesh_interface_equations.md`. Same conventions: no code, precise equations with
provable invariants where they exist, step-by-step build order, validation targets.

Module 2 owns the constitutive tensor $\varepsilon_r(\mathbf r)$ (and trivially $\mu_r$) —
everything Module 3's mass matrix needs and nothing else. It does not own LC director physics
(external, per your instruction — the director field arrives pre-computed), FEM assembly, or
PML (Module 5, which *implements* this module's interface but is specified separately). Two
results below are worth flagging up front because they change what needs to be checked at
runtime versus proven once: interpolating the LC tensor by convex combination **provably**
keeps its eigenvalues bounded between $\varepsilon_\perp$ and $\varepsilon_\parallel$, and
**provably** reduces passivity to a single scalar check on the loss values — not a per-point
eigendecomposition. Both are derived in §4.5–4.6.

---

## 0. What Module 2 owns vs. consumes

**Consumes**: volume tags per tet from Modules 0/1 (`SUBSTRATE`, `AIR`, `LC`, and whatever
`PML_TOP` becomes once Module 5 wraps a background); quadrature points from Module 1's
contract, supplied by Module 3's assembler at call time; a user-authored material spec file
(§5); a pre-computed director field file for the `LC` tag (external — the director physics
that produced it is out of scope here, per your earlier instruction).

**Owns**:
- `material.core` — the interface every material implementation satisfies, plus the two
  runtime invariant checks (symmetry, passivity) that apply to *any* implementation.
- `material.interpolation` — a **shared, low-level numerical primitive** (structured-grid and
  scattered-data interpolation) used internally by both of the modules below. This is not a
  third user-facing module — it exists so the interpolation mathematics is implemented and
  tested exactly once, rather than twice with subtly different behavior.
- `material.regions` — Phases 1–3: constant, spatially-varying scalar, and spatially-varying
  general tensor material, all supplied directly (not derived from a director field).
- `material.tensor_interpolation` — Phase 4 only: the LC-specific path that converts a director
  field to $\varepsilon_r(\mathbf r)$ and interpolates it. Kept as its own module per your
  explicit instruction, even though it shares the low-level interpolation math with
  `material.regions` rather than duplicating it.
- `material.spec` — the loader that reads a user-facing spec file and builds the tag→material
  dispatch registry Module 3 actually queries.

**Does not own**: what happens with $\varepsilon_r(\mathbf r)$ once handed to Module 3;
anything frequency-dependent (material is cached once per the top-level architecture's
invariant #6); the director-field *physics* (Frank energy, bias response) — only its file
format and conversion to a tensor.

---

## 1. `material.core` — the interface and its two universal invariants

### 1.1 Contract

```
class MaterialModel(ABC):
    def epsilon(self, points: (M,3) array) -> (M,3,3) array   # relative permittivity tensor
    def mu(self,      points: (M,3) array) -> (M,3,3) array   # relative permeability tensor
```

Vectorized by contract: Module 3's assembler calls this once per element with all of that
element's quadrature points at once, never in a per-point Python loop. $\mu_r = \mathbf I$ for
every implementation in this document (no magnetic bias anywhere in Phases 1–4); Module 5's
`PMLMaterial` is the only future implementation for which `mu` is non-trivial.

### 1.2 Symmetry (required of every implementation)

$$\varepsilon_r(\mathbf r) = \varepsilon_r(\mathbf r)^{T} \quad\text{at every point.}$$

No magnetic bias $\Rightarrow$ reciprocal medium $\Rightarrow$ symmetric (not merely Hermitian)
tensor. This is the property Module 3 relies on to get a complex-symmetric $\mathbf M$, which is
what ultimately forces $S_{21}=S_{12}$. **Runtime check, every call**: assert
$\|\varepsilon_r - \varepsilon_r^{T}\| < \varepsilon_{\text{tol}}$ elementwise. Cheap, and it is
the single most diagnostic check in the whole material layer — a failure here is the proximate
cause of every downstream reciprocity failure in Module 3/8.

### 1.3 Passivity (required of every implementation, with one documented exception)

Under the $e^{+j\omega t}$ convention, write $\varepsilon_r = A - jB$ with $A,B$ real. Symmetry
of $\varepsilon_r$ forces $A=A^{T}$, $B=B^{T}$ separately (real and imaginary parts of a complex
*symmetric* matrix are each real symmetric). Passivity requires

$$B \succeq 0 \qquad\text{(negative-semidefinite imaginary part, i.e. } \mathrm{Im}(\varepsilon_r)\preceq 0\text{)}.$$

**Runtime check** (generic, for any implementation): all eigenvalues of $B$ are $\ge -\delta$
for a small numerical tolerance $\delta$. This is the expensive general-purpose check — an
eigendecomposition per call — and §4.6 shows it can be replaced by something far cheaper for
the specific uniaxial-LC case. Keep the general check in `material.core` as the defense-in-depth
default for any implementation that *isn't* the LC uniaxial construction (e.g. a hand-specified
Phase 3 tensor with no special structure); use the cheap scalar version (§4.6) wherever the
uniaxial structure actually applies.

**Documented exception: `PMLMaterial` (Module 5) is exempt from this check.** PML's
complex-coordinate-stretching tensor has a normal-direction component whose imaginary part is
*positive* by construction (Module 5 §4.1's derivation: $\mathrm{Im}(1/s_z)\ge0$), which is
what makes the tensor matched, not a sign error. `MaterialAssembly` (or `PMLMaterial` itself)
must skip this check for that specific `MaterialModel`; the correct correctness criterion for
PML is the wave-attenuation/reflection-coefficient argument in Module 5 §2.4/§5.1, not this
per-tensor test. Do not "fix" a correctly implemented `PMLMaterial` to satisfy this check —
doing so un-matches the layer.

### 1.4 `MaterialAssembly` — the tag-dispatch registry

The top-level architecture doc states that Module 3 queries "the material model" for
$\varepsilon_r$ at quadrature points, but doesn't specify how the assembler picks *which*
`MaterialModel` instance applies to a given tet. That dispatch lives here:

```
class MaterialAssembly:
    def __init__(self, tag_to_model: dict[str, MaterialModel]): ...
    def epsilon(self, tag: str, points) -> (M,3,3) array   # dispatches to tag_to_model[tag]
    def mu(self,      tag: str, points) -> (M,3,3) array
```

Module 3's assembler holds one `MaterialAssembly` instance, looks up each tet's volume tag
(from Module 1), and calls `epsilon(tag, quad_points)` — never touching an individual
`MaterialModel` directly. This is the concrete object `material.spec` (§5) builds.

---

## 2. `material.interpolation` — shared low-level primitive

A multi-channel scalar-field interpolator: given samples (structured grid or scattered points)
with $C$ value channels each, return interpolated values at arbitrary query points. Used with
$C=1$ by `material.regions`' Phase 2 scalar path, $C=6$ by both `material.regions`' Phase 3
tensor path and `material.tensor_interpolation`'s Phase 4 path (the six independent components
of a symmetric $3\times3$ tensor: $\varepsilon_{xx},\varepsilon_{xy},\varepsilon_{xz},
\varepsilon_{yy},\varepsilon_{yz},\varepsilon_{zz}$). Writing and testing this once, generically
over $C$, is what keeps Phases 2–4 from each re-deriving interpolation math.

### 2.1 Structured grid: trilinear interpolation

Given a regular grid with node values $f_{ijk}\in\mathbb C^{C}$ and a query point $\mathbf r$
falling in cell $[x_i,x_{i+1}]\times[y_j,y_{j+1}]\times[z_k,z_{k+1}]$, define local fractional
coordinates

$$t_x = \frac{x-x_i}{x_{i+1}-x_i}, \qquad t_y = \frac{y-y_j}{y_{j+1}-y_j}, \qquad t_z = \frac{z-z_k}{z_{k+1}-z_k}.$$

The trilinear estimate is the corner-weighted sum over the eight cell corners
$c=(c_x,c_y,c_z)\in\{0,1\}^3$:

$$f(\mathbf r) = \sum_{c\in\{0,1\}^3} w_c\, f_{i+c_x,\,j+c_y,\,k+c_z}, \qquad
w_c = \big[c_x t_x + (1-c_x)(1-t_x)\big]\big[c_y t_y + (1-c_y)(1-t_y)\big]\big[c_z t_z + (1-c_z)(1-t_z)\big].$$

**Implementation note**: this is exactly `scipy.interpolate.RegularGridInterpolator` with
`method='linear'`, applied once per channel (or vectorized over the channel axis in one call).
No custom trilinear code needs writing.

### 2.2 Scattered data: piecewise-linear over a Delaunay triangulation (primary), IDW (fallback)

**Primary scheme**: build the Delaunay triangulation of the input sample points; for a query
point falling inside simplex $(\mathbf r_0,\mathbf r_1,\mathbf r_2,\mathbf r_3)$, interpolate
using the barycentric coordinates of that simplex (Module 1 §2.1's $\lambda_i$, applied here to
the *sample* tetrahedralization rather than the FEM mesh):

$$f(\mathbf r) = \sum_{i=0}^{3} \lambda_i(\mathbf r)\, f(\mathbf r_i), \qquad \lambda_i \ge 0,\ \sum_i \lambda_i = 1.$$

**Implementation note**: `scipy.interpolate.LinearNDInterpolator`. Preferred over ad-hoc
inverse-distance weighting (IDW) because it is a proper local linear interpolant rather than a
global heuristic, but IDW is kept as a **documented fallback** for point clouds too sparse or
too degenerate (near-coplanar) for a robust Delaunay triangulation:

$$f(\mathbf r) = \frac{\sum_i w_i(\mathbf r)\, f(\mathbf r_i)}{\sum_i w_i(\mathbf r)}, \qquad w_i(\mathbf r) = \frac{1}{\|\mathbf r-\mathbf r_i\|^{p}} \ \ (p=2\text{ default}),$$

restricted to the $k$ nearest samples (found via a $k$-d tree). **Both schemes produce weights
that are non-negative and sum to 1** — this is not incidental; it is the exact property §2.3
depends on.

### 2.3 Why the convexity property matters (used in §4.5–4.6)

Both interpolation schemes above compute $f(\mathbf r) = \sum_i w_i f(\mathbf r_i)$ with
$w_i\ge0$ and $\sum_i w_i = 1$ — a **convex combination** of the sample values. This single fact
is what makes the eigenvalue-bound and trace-invariance proofs in §4.5 go through for *any*
choice of interpolation scheme satisfying it, and is the reason both schemes are described here
together rather than as competing alternatives: the choice between them is a numerical-accuracy
question, not a correctness one, precisely because both are convex.

### 2.4 Coverage guard (extrapolation check)

A query point outside the convex hull of the input samples forces extrapolation, which neither
scheme handles reliably (and which breaks the convexity property of §2.3 — extrapolation
weights are not guaranteed non-negative or summing to 1 in general implementations). **Required
check, at load time, once**: the mesh region this material applies to (its bounding box, from
Module 1) must be contained within the convex hull / bounding box of the input samples. Raise
before ever calling the interpolator on an out-of-coverage query point, rather than returning a
silently extrapolated (and potentially non-convex, hence unbounded) tensor.

---

## 3. `material.regions` — Phases 1–3 (materials supplied directly, not director-derived)

### 3.1 Phase 1 — constant, isotropic

$$\varepsilon_r(\mathbf r) = \varepsilon_{r,t}\,\mathbf I \qquad\text{(constant scalar per tag } t\text{)}.$$

No interpolation involved; every quadrature point in a `SUBSTRATE`-tagged tet gets the same
scalar.

### 3.2 Phase 2 — position-dependent, isotropic

$$\varepsilon_r(\mathbf r) = \varepsilon_{r,t}(\mathbf r)\,\mathbf I,$$

with $\varepsilon_{r,t}(\mathbf r)$ a scalar field supplied as samples (structured or scattered,
same file convention as §5.2 minus the director columns) and evaluated via `material.interpolation`
with $C=1$.

### 3.3 Phase 3 — position-dependent, general symmetric tensor (supplied directly)

$$\varepsilon_r(\mathbf r) = \begin{bmatrix}\varepsilon_{xx}&\varepsilon_{xy}&\varepsilon_{xz}\\ \varepsilon_{xy}&\varepsilon_{yy}&\varepsilon_{yz}\\ \varepsilon_{xz}&\varepsilon_{yz}&\varepsilon_{zz}\end{bmatrix}\!(\mathbf r),$$

the six components supplied as samples (constant or spatially varying) and evaluated via
`material.interpolation` with $C=6$, then reassembled into the $3\times3$ tensor. This is the
Phase 3 material path from the top-level architecture doc, and it is what
`material.tensor_interpolation`'s uniform-director test case must reproduce bit-for-bit
(§4.7) — the **module boundary contract** that ties Phase 3 and Phase 4 together.

### 3.4 Reduction checks

- Phase 2 with a spatially constant scalar field must reproduce Phase 1 exactly.
- Phase 3 with a spatially constant, diagonal tensor with all three diagonal entries equal must
  reproduce Phase 1 exactly (a second, independent route to the same reduction).

---

## 4. `material.tensor_interpolation` — Phase 4, the LC director path

The only module that reads the director file. Consumes the director field as an external,
already-computed input; owns only the conversion to $\varepsilon_r(\mathbf r)$ and its
interpolation.

### 4.1 Tensor construction

$$\varepsilon_r(\mathbf r) = \varepsilon_\perp\,\mathbf I + (\varepsilon_\parallel-\varepsilon_\perp)\,\mathbf n(\mathbf r)\,\mathbf n(\mathbf r)^{T}, \qquad |\mathbf n(\mathbf r)|=1.$$

Computed **once, at the input sample points**, immediately on file load — never re-derived from
$\mathbf n$ downstream. From this point on, only the tensor components are interpolated (§2.3's
convexity property is what makes this safe).

### 4.2 Input validation at parse time

- **Normalization guard**: if the file's stored $|\mathbf n|$ deviates from 1 by more than a
  small tolerance (e.g. outside $[0.9,1.1]$), raise — this catches malformed rows, not just
  round-off. Within tolerance, renormalize: $\mathbf n \leftarrow \mathbf n/|\mathbf n|$ before
  §4.1.
- **Coordinate-frame and units consistency**: the director file's stated coordinate frame and
  units must match the mesh's; assert the mesh's `LC`-tagged volume bounding box is contained
  within the director sample bounding box (§2.4), not merely overlapping — a units mismatch
  (e.g. file in mm, mesh in m) will generically show up as this containment check failing
  by roughly a factor of 1000, which is a useful diagnostic to report rather than just raise on.

### 4.3 Complex (lossy) LC parameters

$\varepsilon_\perp,\varepsilon_\parallel$ may be complex: $\varepsilon_\perp = \varepsilon_\perp'
- j\varepsilon_\perp''$, $\varepsilon_\parallel = \varepsilon_\parallel' - j\varepsilon_\parallel''$,
with $\varepsilon_\perp'',\varepsilon_\parallel''\ge0$ for passivity (§4.6). The director
$\mathbf n(\mathbf r)$ itself is always real — orientation is a real geometric quantity; only
the material constants carry loss.

### 4.4 Interpolation

The six components of $\varepsilon_r$ (§4.1) are interpolated via `material.interpolation` with
$C=6$ — the identical shared primitive `material.regions` uses for Phase 3, per §0. This module
supplies the *input* to that primitive (director → tensor conversion, §4.1) and consumes its
*output*; it does not reimplement trilinear or Delaunay interpolation.

### 4.5 Provable invariant: eigenvalues stay within $[\varepsilon_\perp,\varepsilon_\parallel]$

This is a real proof, not a heuristic expectation, and it is what justifies treating the
eigenvalue-bound check as a one-time regression test rather than a per-call runtime assertion.

Let $Q(\mathbf r) = \sum_i w_i(\mathbf r)\, \mathbf n_i\mathbf n_i^{T}$ be the interpolated
*orientation* part (the convex combination from §2.3, applied to the rank-1 projectors built
from each sample's director — equivalently, this falls out of interpolating the tensor
components directly, since $\varepsilon_r = \varepsilon_\perp\mathbf I + \Delta\varepsilon\, Q$
is linear in $Q$). For any unit vector $\mathbf v$:

$$\mathbf v^{T}Q(\mathbf r)\,\mathbf v = \sum_i w_i\,(\mathbf v\cdot\mathbf n_i)^2 \in [0,1],$$

since each term $(\mathbf v\cdot\mathbf n_i)^2\in[0,1]$ (Cauchy–Schwarz, unit vectors) and the
$w_i$ are non-negative and sum to 1. Since this holds for *every* unit $\mathbf v$, every
eigenvalue of $Q(\mathbf r)$ lies in $[0,1]$ — regardless of how many distinct director
orientations are being blended or how they're arranged. Because
$\varepsilon_r(\mathbf r) = \varepsilon_\perp\mathbf I + \Delta\varepsilon\,Q(\mathbf r)$ is an
affine function of $Q$, $\varepsilon_r$ and $Q$ share eigenvectors, and each eigenvalue of
$\varepsilon_r$ equals $\varepsilon_\perp + \lambda\,\Delta\varepsilon$ for the corresponding
$\lambda\in[0,1]$ — i.e. it lies exactly on the complex line segment joining $\varepsilon_\perp$
(at $\lambda=0$) and $\varepsilon_\parallel$ (at $\lambda=1$).

**Consequence for what to check**: the guarantee doesn't need re-proving per point at runtime —
it follows automatically from (a) $|\mathbf n_i|=1$ at every sample (§4.2's normalization guard)
and (b) the interpolation weights being non-negative and summing to 1 (§2.3, a property of the
*scheme*, verified once as a property of `material.interpolation` itself, not of each call).
**Runtime check**: assert weight partition-of-unity ($\sum_i w_i = 1$) as a property of the
interpolation primitive (tested once in `material.interpolation`'s own test suite). **Regression
test, not runtime check**: confirm eigenvalues of $\varepsilon_r(\mathbf r)$ lie in the expected
range at a sample of interpolated points, as a one-time validation of the whole pipeline (§7) —
expensive per-point eigendecomposition has no place in the production path once this is verified.

### 4.6 Provable simplification: passivity reduces to a scalar check

From §4.5, the real and imaginary parts decompose the same way: writing
$\varepsilon_\perp=\varepsilon_\perp'-j\varepsilon_\perp''$,
$\Delta\varepsilon=\Delta\varepsilon'-j\Delta\varepsilon''$,

$$\varepsilon_r(\mathbf r) = \underbrace{\big[\varepsilon_\perp'\mathbf I+\Delta\varepsilon'\,Q(\mathbf r)\big]}_{A(\mathbf r)} - j\underbrace{\big[\varepsilon_\perp''\mathbf I+\Delta\varepsilon''\,Q(\mathbf r)\big]}_{B(\mathbf r)}.$$

By the same eigenvalue argument applied to $B$: every eigenvalue of $B(\mathbf r)$ equals
$\varepsilon_\perp'' + \lambda\,\Delta\varepsilon''$ for some $\lambda\in[0,1]$, i.e. lies
between $\varepsilon_\perp''$ and $\varepsilon_\parallel''$. **If both endpoints are
non-negative, every eigenvalue of $B(\mathbf r)$ is non-negative, at every point, for any
director field whatsoever.** Passivity ($B\succeq0$ everywhere) is therefore guaranteed by
checking two scalars **once**, at material-spec load time:

$$\boxed{\;\varepsilon_\perp'' \ge 0 \quad\text{and}\quad \varepsilon_\parallel'' \ge 0\;}$$

This replaces what would otherwise be a per-quadrature-point $3\times3$ eigendecomposition
(Module 2.1's generic passivity check) with two scalar comparisons at load time. Use this cheap
version for the LC path specifically; keep the generic per-point check in `material.core` as
the fallback for any material implementation lacking this uniaxial structure (e.g. a
hand-authored Phase 3 tensor).

### 4.7 Module boundary contract

A **spatially uniform** director ($\mathbf n(\mathbf r)\equiv\mathbf n_0$ everywhere in the `LC`
volume) must produce, via this module, results bit-identical (to floating-point tolerance) to
`material.regions`' Phase 3 path given the direct tensor
$\varepsilon_\perp\mathbf I+\Delta\varepsilon\,\mathbf n_0\mathbf n_0^{T}$ as a spatially constant
input. This is what lets Phase 4 validation lean on Phase 3's (and, through it, Phase 1's)
analytic gates without deriving new anisotropic closed-form solutions (top-level doc §8).

### 4.8 $\mathbf n \to -\mathbf n$ invariance

$\mathbf n\mathbf n^{T} = (-\mathbf n)(-\mathbf n)^{T}$ identically — an exact algebraic identity
at every input sample point, guaranteed the instant §4.1 is applied and never revisited. Because
the raw director vector is discarded immediately after §4.1 (never re-read downstream), nothing
in the interpolation or assembly path can reintroduce sign sensitivity. **This is therefore a
parser-level unit test, not a runtime check**: feed a director file and a copy with every
$\mathbf n_i$ negated; confirm the two produce bit-identical $\varepsilon_r$ samples at the
input points.

---

## 5. `material.spec` — the loader

### 5.1 Spec file schema

Four explicit type values, one per build phase, so the spec vocabulary mirrors the phase plan
directly:

```yaml
materials:
  AIR:
    type: constant
    eps_r: 1.0

  SUBSTRATE:
    type: constant          # Phase 1
    eps_r: 3.5
    # type: scalar_field    # Phase 2 — replaces the above two lines
    # file: substrate_profile.csv
    # type: tensor_field    # Phase 3 — direct symmetric tensor, constant or field
    # eps_r_components: {xx: ..., xy: ..., ...}   # or: file: substrate_tensor.csv

  LC:
    type: director_field    # Phase 4 — the only type that reads a director file
    file: lc_director.csv
    eps_perp: 2.4
    eps_parallel: 3.0
    # eps_perp_im / eps_parallel_im, if lossy
```

`PML_TOP` never appears here — its material is always derived (Module 5), never spec-supplied.

### 5.2 Director file schema (formalized)

```
# lc_director.csv
# coordinate_frame: mesh
# units: mm
# grid_type: structured | scattered
x, y, z, nx, ny, nz
...
```

`coordinate_frame`, `units`, and `grid_type` are mandatory header fields (§4.2's consistency
check depends on all three being present and explicit, not inferred).

### 5.3 Build process

1. Parse the YAML; for each tag, validate the required fields for its declared `type`.
2. For `constant`/`scalar_field`/`tensor_field` tags: instantiate `material.regions`.
3. For `director_field` tags: instantiate `material.tensor_interpolation`, which internally
   runs §4.2's parse-time validation before anything else touches the file.
4. Assemble the per-tag instances into one `MaterialAssembly` (§1.4).
5. Return the `MaterialAssembly` to whatever built it (Module 3's assembler, or a driver script).

### 5.4 Merging with Module 0's auto-generated stub

Module 0 emits a partial spec (`AIR`, `SUBSTRATE` only) from the same geometry parameters used
to build the mesh. `material.spec`'s loader accepts this stub plus a user-supplied `LC` entry
(and any Phase 2/3 overrides for `SUBSTRATE`) and merges them into one spec before step 1 above
— so the common case (Phase 1 microstrip, no LC yet) requires no separate material file at all.

---

## 6. Step-by-step build order

1. `material.core`: the `MaterialModel` ABC and `MaterialAssembly` dispatcher (§1). No
   evaluators yet — just the contract and the two universal checks (symmetry, generic
   passivity), unit-tested against a hand-built toy tensor before anything else is built on it.
2. `material.interpolation`: structured (trilinear) and scattered (Delaunay-linear + IDW
   fallback) primitives (§2), generic over channel count $C$. Unit test: partition-of-unity
   ($\sum_i w_i=1$) for both schemes; reduction to nearest-sample value at the sample points
   themselves; the §2.4 coverage guard raises on an out-of-hull query.
3. `material.regions` Phase 1 (§3.1) — trivial, but wire it fully through `MaterialAssembly` to
   validate the dispatch mechanism end-to-end before anything more complex is built.
4. `material.regions` Phase 2 (§3.2), using step 2's $C=1$ path. Validate the reduction to
   Phase 1 (§3.4).
5. `material.regions` Phase 3 (§3.3), using step 2's $C=6$ path. Validate its own reduction to
   Phase 1 (§3.4) via a constant isotropic tensor input.
6. `material.tensor_interpolation` (§4): §4.1's tensor construction and §4.2's validation first
   (unit tested standalone against hand-picked director samples, including the §4.8 sign-flip
   test), *then* wire it to step 2's $C=6$ interpolation path.
7. Run the §4.7 module-boundary contract test (uniform director vs. step 5's direct-tensor
   path) — this is the test that validates step 6 against step 5 rather than against new
   physics.
8. `material.spec` (§5): YAML parsing and the merge with Module 0's stub, built last since it
   only wires together components already validated individually.
9. Run the full §7 validation suite.

---

## 7. Validation targets

- **Symmetry & passivity** (§1.2–1.3): unit-tested against every `MaterialModel`
  implementation, not just the LC path.
- **Interpolation partition-of-unity** (§2.3, §4.5): $\sum_i w_i(\mathbf r) = 1$ for both
  structured and scattered schemes, at several query points including cell/simplex boundaries.
- **Coverage guard** (§2.4): confirm it raises on a query point deliberately placed outside the
  sample convex hull.
- **Reductions** (§3.4): Phase 2 → Phase 1, Phase 3 → Phase 1, both to floating-point tolerance.
- **Module boundary contract** (§4.7): uniform director vs. direct Phase-3 tensor input,
  bit-identical to solver tolerance — the load-bearing test connecting Phases 3 and 4.
- **Sign-flip invariance** (§4.8): director file vs. its negation, bit-identical $\varepsilon_r$
  at input sample points.
- **Eigenvalue-bound spot check** (§4.5): at a handful of interpolated points (not every point,
  per the "regression test, not runtime check" distinction), confirm eigenvalues of
  $\varepsilon_r(\mathbf r)$ fall within $[\min(\varepsilon_\perp',\varepsilon_\parallel'),\max(\cdot)]$
  for the real part and the analogous bound for $-\mathrm{Im}$.
- **Passivity-reduction check** (§4.6): confirm the two-scalar check
  ($\varepsilon_\perp''\ge0,\varepsilon_\parallel''\ge0$) is actually enforced at spec-load time,
  and confirm a deliberately negative test value is rejected before any interpolation occurs.
- **Coordinate-frame mismatch detection** (§4.2): a deliberately mis-scaled test director file
  (wrong units) must fail the containment check, not silently extrapolate or misinterpret scale.

---

## 8. Interface / class contract summary

```
class MaterialModel(ABC):
    def epsilon(points: (M,3)) -> (M,3,3)
    def mu(points: (M,3)) -> (M,3,3)

class MaterialAssembly:
    def epsilon(tag: str, points: (M,3)) -> (M,3,3)
    def mu(tag: str, points: (M,3)) -> (M,3,3)

# material.interpolation (shared primitive, not directly user-facing)
def interpolate_structured(grid_points, grid_values: (...,C), query_points: (M,3)) -> (M,C)
def interpolate_scattered(sample_points, sample_values: (N,C), query_points: (M,3)) -> (M,C)

class ConstantMaterial(MaterialModel):        # Phase 1
class ScalarFieldMaterial(MaterialModel):     # Phase 2
class TensorFieldMaterial(MaterialModel):     # Phase 3
class DirectorFieldMaterial(MaterialModel):   # Phase 4 — material.tensor_interpolation

# material.spec
def load_material_spec(path, geometry_stub=None) -> MaterialAssembly
```

Module 3 (`fem.assembly`) is the sole downstream consumer, and it only ever sees
`MaterialAssembly.epsilon(tag, points)` — it is unaware of which of the four `MaterialModel`
subclasses actually answered the call, which is what makes the Phase 1→4 build sequence a
change of spec-file input rather than assembler code (top-level doc, invariant #1).
