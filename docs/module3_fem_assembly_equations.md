# Module 3 — `fem.edge_elements` + `fem.assembly`: Operators K, M — Equations & Implementation Plan

Companion to the top-level architecture doc and Modules 0–2. Same conventions: no code,
precise equations with proofs where they exist, step-by-step build order, validation targets.

Module 3 owns the Whitney/Nédélec edge basis and the assembly of the two frequency-independent
volume operators $\mathbf K,\mathbf M$. It consumes Module 1's per-tet geometry and edge
topology and Module 2's `MaterialAssembly`, and produces nothing frequency-, port-, or
PML-dependent — those are Modules 4–6's job. Three small gaps in Module 1's existing contract
surfaced while working this out; they're listed in §1 and applied as edits at the end of this
document.

---

## 0. What Module 3 owns vs. consumes

**Consumes**: per-tet gradients $\nabla\lambda_i$ and volume $V$ (Module 1 §2); the oriented
global edge list, `tet_edge_map`, `tet_edge_sign` (Module 1 §3); quadrature points and weights
(Module 1 §6); `MaterialAssembly.epsilon(tag, points)` / `.mu(tag, points)` (Module 2 §1.4).

**Owns**: the Whitney basis function and its curl (`fem.edge_elements`); the element matrices
$\mathbf K^{(e)},\mathbf M^{(e)}$ and their quadrature-order selection; global sparse assembly
into $\mathbf K,\mathbf M$ over the full edge-DOF space.

**Does not own**: essential (PEC) boundary condition *application* — Module 3 assembles the
full, unconstrained system; eliminating PEC-constrained DOFs is Module 6's job, using a DOF set
Module 1 computes geometrically (§1.3). Frequency dependence, ports, and PML are entirely
outside this module — see §5.4 for how PML's future frequency dependence is kept from leaking
into this module's interface.

---

## 1. Required additions to Module 1's contract

Working through the assembly equations below requires three pieces of information Module 1
already has internally but didn't expose in its published interface (§9 there). Each is a pure
geometric derivation from data Module 1 already holds — none of it is new physics, so it
belongs in Module 1, not here. Listed now, applied to that document at the end of this one.

1. **Barycentric weights at quadrature points.** `quadrature_tet(order)` currently returns
   `(points, weights)`. Module 3 needs the barycentric coordinates $\hat\lambda_{k,i}$
   ($i=0..3$) at each quadrature point $k$ to evaluate the basis functions there (§2.2) — and
   Module 1 already has them, since the physical point is *computed from* them
   ($\mathbf r_k=\sum_i\hat\lambda_{k,i}\mathbf r_i$, Module 1 §6.1). **Extend the return
   signature to `(points, weights, barycentric)`**, with `barycentric` shape $(M,4)$.
2. **Per-tet volume tag.** Module 1 consumes `volume_tags` as an input but never exposed a way
   to read it back per tet. Module 3's assembler needs to know which tag governs a given tet
   before it can query `MaterialAssembly`. **Add `tet_volume_tag(tet: int) -> str`** (or expose
   the raw `volume_tags` array directly — either satisfies the need).
3. **PEC edge DOF set.** Identifying which global edges lie on a `PEC`-tagged boundary face is
   a purely geometric derivation from data Module 1 already owns (`boundary_faces('PEC')`'s
   face vertex triples, intersected against the global `edges` list) — no material or frequency
   content. **Add `pec_edge_dofs() -> set[int]`**, computed once at load. Module 3 does not use
   this itself (it assembles the unconstrained system); it exists so Module 6 has it without
   re-deriving geometry.

---

## 2. `fem.edge_elements` — the Whitney/Nédélec basis

### 2.1 Local edge table and the clean orientation convention

Recall Module 1 §3.1's fixed local edge numbering: local edge $\ell=0..5$ has a **fixed** local
vertex pair $(a_\ell, b_\ell)$ with $a_\ell < b_\ell$ as *local* indices (e.g.
$\ell=0\Rightarrow(a_0,b_0)=(0,1)$). This pairing never changes per tet — it's a lookup table,
not a per-element computation.

The formula in §2.2 is always evaluated using this **fixed local pair**, never re-sorted by
global index. What Module 1's per-tet sign $s_\ell$ (`tet_edge_sign`) corrects for is exactly
this: the fixed local pair $(a_\ell,b_\ell)$ maps to *some* pair of global vertex indices
$(g_{a_\ell}, g_{b_\ell})$, which may or may not already be in ascending order. Concretely:

$$s_\ell = \begin{cases} +1 & \text{if } g_{a_\ell} < g_{b_\ell} \\ -1 & \text{if } g_{a_\ell} > g_{b_\ell}\end{cases}$$

**Why this makes the basis single-valued across elements**: the global DOF for edge $e$ is
defined once, globally, as $\int_e \mathbf E\cdot d\boldsymbol\ell$ integrated from the smaller
global vertex to the larger (Module 1's ascending convention). A tet's *unsigned* local formula
(§2.2, built from the fixed local pair) has tangential integral $+1$ along the path from
$a_\ell$ to $b_\ell$ *in local terms*. If $g_{a_\ell}<g_{b_\ell}$, that local path already runs
in the global-ascending direction, so the unsigned formula directly represents $+1$ per unit
global DOF ($s_\ell=+1$, no correction). If $g_{a_\ell}>g_{b_\ell}$, the local path runs
*backward* relative to the global convention, so the unsigned formula represents $-1$ per unit
global DOF, and must be negated ($s_\ell=-1$) before its coefficient can be identified with the
single global DOF value. Any two tets sharing a global edge apply this same rule independently
and always arrive at a mutually consistent signed basis function — this is the entire
correctness mechanism, and it is exactly what Module 1 already computes and stores.

### 2.2 Basis function

For local edge $\ell$ with fixed pair $(a_\ell,b_\ell)$, barycentric coordinates
$\lambda_{a_\ell}(\mathbf r), \lambda_{b_\ell}(\mathbf r)$ (affine, Module 1 §2.1), and
constant gradients $\mathbf g_{a_\ell}=\nabla\lambda_{a_\ell}$, $\mathbf g_{b_\ell}=\nabla\lambda_{b_\ell}$
(Module 1 §2.2):

$$\boxed{\;\mathbf W_\ell(\mathbf r) = s_\ell\big[\lambda_{a_\ell}(\mathbf r)\,\mathbf g_{b_\ell} - \lambda_{b_\ell}(\mathbf r)\,\mathbf g_{a_\ell}\big]\;}$$

Affine in $\mathbf r$ (product of an affine scalar and a constant vector, summed). At a
quadrature point $k$ with barycentric weights $\hat\lambda_{k,i}$ (§1, item 1), evaluate
directly — no need to reconstruct $\lambda_i(\mathbf r_k)$ from the affine coefficients:

$$\mathbf W_\ell(\mathbf r_k) = s_\ell\big[\hat\lambda_{k,a_\ell}\,\mathbf g_{b_\ell} - \hat\lambda_{k,b_\ell}\,\mathbf g_{a_\ell}\big]$$

### 2.3 Curl — constant per element

Using $\nabla\times(f\mathbf A) = (\nabla f)\times\mathbf A + f(\nabla\times\mathbf A)$ and the
fact that $\mathbf g_{a_\ell},\mathbf g_{b_\ell}$ are constant vectors (so their own curl is
zero):

$$\nabla\times(\lambda_{a_\ell}\mathbf g_{b_\ell}) = \mathbf g_{a_\ell}\times\mathbf g_{b_\ell}, \qquad
\nabla\times(\lambda_{b_\ell}\mathbf g_{a_\ell}) = \mathbf g_{b_\ell}\times\mathbf g_{a_\ell} = -\mathbf g_{a_\ell}\times\mathbf g_{b_\ell}$$

$$\boxed{\;\mathbf C_\ell \equiv \nabla\times\mathbf W_\ell = 2\,s_\ell\,(\mathbf g_{a_\ell}\times\mathbf g_{b_\ell})\;}$$

A **constant 3-vector per element**, per local edge — the standard, well-known property of
lowest-order Nédélec elements, and the reason the stiffness matrix (§3.1) collapses to
something much cheaper than the mass matrix.

### 2.4 Consistency checks

- **Tangential-trace normalization**: $\int_{e_\ell} \mathbf W_\ell\cdot d\boldsymbol\ell = 1$
  along edge $\ell$'s own path (from $a_\ell$ to $b_\ell$, unsigned formula) — a direct algebraic
  identity of the Whitney construction; verify once symbolically/numerically on a reference tet.
- **Curl reproduction**: numerically differentiate $\mathbf W_\ell$ (finite difference at a few
  points) and confirm it matches $\mathbf C_\ell$ from §2.3 to within discretization error — a
  cheap sanity test that the closed-form curl wasn't mis-derived or mis-transcribed into code.
- **DOF continuity across two elements** (the real test of §2.1's sign convention): build a
  two-tet patch sharing one face (hence three shared edges); evaluate the *signed* basis
  functions for the shared edges from both tets at a point on the shared face; confirm tangential
  components agree. A failure here means $s_\ell$ was computed or applied inconsistently, and it
  is the single most important test in this module — everything downstream (complex-symmetric
  $\mathbf M$, reciprocal $S$) depends on this holding.

---

## 3. Element matrices

### 3.1 Stiffness $\mathbf K^{(e)}$ — reduces to a single averaged tensor

$$K_{\ell m}^{(e)} = \int_{\text{tet}} (\nabla\times\mathbf W_\ell)\cdot\big(\mu_r^{-1}(\mathbf r)\,\nabla\times\mathbf W_m\big)\,dV$$

Since $\mathbf C_\ell = \nabla\times\mathbf W_\ell$ is **constant** (§2.3), it factors out of the
integral entirely:

$$K_{\ell m}^{(e)} = \mathbf C_\ell^{T}\left[\int_{\text{tet}} \mu_r^{-1}(\mathbf r)\,dV\right]\mathbf C_m
= \mathbf C_\ell^{T}\,\bar{\mathbf M}_\mu^{(e)}\,\mathbf C_m$$

$$\boxed{\;\bar{\mathbf M}_\mu^{(e)} = \sum_k w_k\,\mu_r^{-1}(\mathbf r_k)\;}$$

**This is the efficiency point worth designing around explicitly**: $\bar{\mathbf M}_\mu^{(e)}$
is a **single $3\times3$ tensor computed once per element**, not a per-$(\ell,m)$-pair
quadrature sum. All 21 unique entries of the symmetric $6\times6$ $\mathbf K^{(e)}$ then come
from cheap $3$-vector bilinear products against the same cached tensor — no repeated
quadrature. This holds regardless of whether $\mu_r$ varies within the element (today it never
does — $\mu_r=\mathbf I$ everywhere in Phases 1–4 — but the architecture is written generally
so Module 5's spatially-varying PML $\mu_r$ needs no change here). $\mu_r^{-1}(\mathbf r_k)$ is
a per-point $3\times3$ inverse of whatever `MaterialAssembly.mu(tag, points)` returns —
inexpensive regardless of tensor structure, and computed **before** summing (invert, then
integrate — the two only coincide when $\mu_r$ is element-constant, which is the common but not
general case, so the code path should not assume it).

### 3.2 Mass $\mathbf M^{(e)}$ — direct quadrature

$\mathbf W_\ell(\mathbf r)$ is affine, not constant, so no analogous simplification exists — this
is the integral Module 1's quadrature contract and Module 2's material evaluation exist to
serve:

$$\boxed{\;M_{\ell m}^{(e)} = \sum_k w_k\,\mathbf W_\ell(\mathbf r_k)^{T}\,\varepsilon_r(\mathbf r_k)\,\mathbf W_m(\mathbf r_k)\;}$$

with $\varepsilon_r(\mathbf r_k)$ from `MaterialAssembly.epsilon(tag, points)` where `tag` is
this element's volume tag (§1, item 2) and `points` is the *entire* array of this element's
quadrature points in one vectorized call.

### 3.3 Symmetry of $\mathbf K^{(e)},\mathbf M^{(e)}$ — proof, not assumption

For real vectors $\mathbf a,\mathbf b$ and any matrix $T$: $\mathbf a\cdot(T\mathbf b) =
\mathbf b\cdot(T^{T}\mathbf a)$. Applying this to $M_{m\ell}^{(e)} = \sum_k w_k\,\mathbf
W_m(\mathbf r_k)\cdot\big(\varepsilon_r(\mathbf r_k)\,\mathbf W_\ell(\mathbf r_k)\big) = \sum_k
w_k\,\mathbf W_\ell(\mathbf r_k)\cdot\big(\varepsilon_r(\mathbf r_k)^{T}\,\mathbf W_m(\mathbf
r_k)\big)$, and using Module 2 §1.2's guarantee $\varepsilon_r=\varepsilon_r^{T}$ at every
point:

$$M_{m\ell}^{(e)} = \sum_k w_k\,\mathbf W_\ell(\mathbf r_k)\cdot\big(\varepsilon_r(\mathbf r_k)\,\mathbf W_m(\mathbf r_k)\big) = M_{\ell m}^{(e)}$$

$\mathbf M^{(e)} = \mathbf M^{(e)T}$ **follows directly from Module 2's symmetry invariant** —
it is not a separate assumption Module 3 needs to independently guarantee. The identical
argument applied to $\bar{\mathbf M}_\mu^{(e)}$ (itself symmetric, since it's a volume integral
of pointwise-symmetric $\mu_r^{-1}$, and the inverse of a symmetric matrix is symmetric) gives
$\mathbf K^{(e)}=\mathbf K^{(e)T}$ the same way. **This is the mechanism by which Module 2's
symmetry check becomes Module 3's complex-symmetric global system** — worth stating explicitly
so a bug hunt for a non-reciprocal $S$ later knows exactly which proof to re-check first.

### 3.4 Passivity inheritance

Module 2 §1.3/§4.6 established $\mathrm{Im}(\varepsilon_r)\preceq0$, $\mathrm{Im}(\mu_r)\preceq0$
pointwise. Passivity of $\mathbf M^{(e)},\mathbf K^{(e)}$ (in the sense that the assembled
system doesn't manufacture gain) follows from the same pointwise property integrated with
non-negative quadrature weights ($w_k>0$, Module 1 §6.3) — a non-negative combination of
negative-semidefinite quadratic forms stays negative-semidefinite. No new proof needed; this is
noted so it's clear the element-level passivity is inherited, not independently asserted.

---

## 4. Quadrature order selection — adaptive, applied uniformly to $\mathbf K$ and $\mathbf M$

Rather than branching on "is this material constant or varying" (which would require a material
metadata flag Module 2 doesn't currently expose), apply the **same adaptive procedure
unconditionally** to every element — it converges immediately (at the lowest tried order) for
constant materials and does the right thing for spatially varying ones, at the cost of one
extra, cheap quadrature evaluation to *confirm* convergence in the constant case. This mirrors
the doubling-convergence pattern already used in the cavity-perturbation tool's field-energy
integration and in Module 1 §6.2/§6.3.

**Procedure, per element, for both $\bar{\mathbf M}_\mu^{(e)}$ and $\mathbf M^{(e)}$:**

1. Start at a low tabulated order $N$ (e.g. $N=2$).
2. Compute the quantity (the $3\times3$ tensor for $\bar{\mathbf M}_\mu^{(e)}$, or the full
   $6\times6$ matrix for $\mathbf M^{(e)}$) at order $N$ and again at order $N+2$.
3. Compare via relative Frobenius norm:
   $$\frac{\|X_{N+2} - X_N\|_F}{\|X_{N+2}\|_F} < \varepsilon_{\text{tol}} \quad (\text{suggested } 10^{-4})$$
4. If not converged, set $N\leftarrow N+2$ and repeat, up to Module 1's highest tabulated order
   (§6.2 there, up to $\sim6$). If still unconverged at the maximum order, **raise** — this
   signals either a material variation too sharp for the mesh resolution (needs mesh
   refinement, not more quadrature) or a genuine bug in the material evaluator.
5. Use $X_{N+2}$ (the finer estimate) as the element's contribution.

This cost is paid once per element, for the lifetime of the model — Module 3's output is
exactly the frequency-independent interior cache (top-level doc, invariant #6), so this
adaptive cost is never repeated across a frequency sweep.

---

## 5. Global assembly

### 5.1 Sparse accumulation

Standard COO-accumulate-then-convert pattern: for each element, compute the local
$6\times6$ $\mathbf K^{(e)},\mathbf M^{(e)}$ (§3), then scatter into global row/column index
triplets using `tet_edge_map[e]` (Module 1 §3.3) as the local-to-global DOF map — no further
sign bookkeeping needed at scatter time, since $s_\ell$ is already baked into $\mathbf W_\ell$
and $\mathbf C_\ell$ (§2.2–2.3). Accumulate triplets across all elements sharing an edge (the
standard "+=" assembly behavior), then convert to CSR for downstream use.

### 5.2 Tet-subset support (for Module 6's interior/PML split)

The assembly routine accepts an optional subset of tet indices (default: all tets). This is
what lets Module 6 call this same routine twice with different intent: once, at setup, over the
non-PML tets (`SUBSTRATE`, `AIR`, `LC`) — cached for the whole sweep — and once per frequency
over the `PML_TOP` tets only, using a freshly-built `MaterialAssembly` for that tag (§5.4).
Module 3 itself has no notion of "interior" vs "PML"; it only assembles whatever tet set and
`MaterialAssembly` it's handed.

### 5.3 PEC boundary condition — not applied here

Module 3 returns the full, unconstrained $\mathbf K,\mathbf M$ over all $N_e$ edge DOFs. It does
not eliminate PEC-constrained rows/columns — that's Module 6's job, using the `pec_edge_dofs()`
set Module 1 now exposes (§1, item 3). Keeping this split means Module 3 never needs to know
about boundary conditions at all, matching its scope as a pure "geometry + material → operator"
module.

### 5.4 Frequency independence — how PML's future $\omega$-dependence stays out of this module

Module 2's `MaterialModel.epsilon(points)`/`.mu(points)` interface takes **no frequency
argument**, and this module does not change that. Module 5's future `PMLMaterial` will still
satisfy this exact signature — its $\omega$-dependence (the complex stretch factors,
top-level doc §5.1) is resolved by **constructing a fresh `PMLMaterial` instance per frequency**
(e.g. `PMLMaterial(background, omega=ω_k, ...)`), not by threading $\omega$ through the
`epsilon`/`mu` call signature. Module 6's sweep driver is what re-instantiates it each iteration
and re-invokes Module 3's assembly routine (§5.2's subset support) on the `PML_TOP` tets against
that fresh instance. Module 3 itself is therefore completely oblivious to frequency in every
respect — it doesn't matter to this module whether a `MaterialModel` instance happens to have
had a frequency baked into its constructor.

---

## 6. Step-by-step build order

1. Apply the three Module 1 contract additions (§1) — this module cannot proceed without them.
2. `fem.edge_elements`: the basis function and curl (§2.1–2.3), unit-tested standalone on a
   single reference tet against the §2.4 checks (tangential-trace normalization, curl
   reproduction) before any element-matrix code touches them.
3. The two-tet-patch DOF-continuity test (§2.4) — build this test harness early; it is reused
   for every subsequent change to the basis or sign convention.
4. `fem.assembly`, element level: $\bar{\mathbf M}_\mu^{(e)}$ and $\mathbf K^{(e)}$ (§3.1) first
   (simpler — one tensor, no adaptive quadrature needed in Phases 1–3 since $\mu_r=\mathbf I$
   is trivially constant), then $\mathbf M^{(e)}$ (§3.2).
5. Wire in the adaptive quadrature procedure (§4), applied to both — test it explicitly
   confirms immediate convergence for `ConstantMaterial` tags and correctly escalates order for
   `ScalarFieldMaterial`/`TensorFieldMaterial`/`DirectorFieldMaterial` tags.
6. Global assembly (§5.1) with tet-subset support (§5.2) built in from the start, even though
   Phase 1–3 testing only exercises the "all tets" default — retrofitting subset support later
   would touch the same scatter loop twice.
7. Confirm §3.3's symmetry proof holds numerically on the assembled *global* $\mathbf K,\mathbf
   M$ (not just per-element) — this is the first point in the pipeline where a bug in `edges`
   dedup (Module 1) or in `tet_edge_sign` propagation could still surface despite every local
   check passing, since it's the only test that exercises inter-element accumulation at scale.
8. Run the full §7 validation suite.

---

## 7. Validation targets

- **Local checks** (§2.4): tangential-trace normalization, curl reproduction, two-tet DOF
  continuity — all on a hand-constructed reference tet / two-tet patch, run before any global
  assembly exists.
- **Global symmetry** (§3.3): $\|\mathbf K-\mathbf K^{T}\|,\|\mathbf M-\mathbf M^{T}\|\approx0$
  to machine precision, checked on the *assembled global* sparse matrices for a real multi-tet
  mesh — not just the per-element matrices.
- **Reference-tetrahedron check**: assemble $\mathbf K^{(e)},\mathbf M^{(e)}$ for the standard
  unit reference tetrahedron (vertices at the origin and the three unit axis points) with
  $\varepsilon_r=\mu_r=\mathbf I$, and compare against published closed-form Whitney
  edge-element matrices for this exact case (e.g. Jin, *The Finite Element Method in
  Electromagnetics*) — an external, independent numeric check that doesn't rely on this
  document's own derivation being self-consistent.
- **Adaptive quadrature convergence** (§4): confirm immediate (single-check) convergence for
  `ConstantMaterial`; confirm correct order escalation for a smoothly-varying test field;
  confirm the max-order raise fires on a deliberately pathological (high-spatial-frequency) test
  field that a coarse mesh cannot resolve.
- **$\bar{\mathbf M}_\mu^{(e)}$ symmetry**: spot-check on a tet with a deliberately
  non-trivial (but still symmetric, per Module 2's contract) $\mu_r$ — relevant once Module 5
  exists, but the test harness should be written now so Module 5 only needs to supply the
  input, not new test infrastructure.
- **Tet-subset consistency**: assembling over "all tets" must equal assembling over two
  disjoint subsets and adding the results — a cheap, strong regression test for §5.2's subset
  support, and exactly the operation Module 6 will rely on for the interior/PML split.

---

## 8. Interface / class contract

```
# fem.edge_elements
def whitney_basis(tet_grad_lambda: (4,3), tet_edge_sign: (6,), local_edge_table) -> callable
    # evaluates (6,3) array of W_ell at given barycentric weights (M,4) -> (6,M,3)
def whitney_curl(tet_grad_lambda: (4,3), tet_edge_sign: (6,), local_edge_table) -> (6,3) array
    # constant per tet; no quadrature point argument needed

# fem.assembly
def assemble(mesh: MeshInterface, materials: MaterialAssembly,
             tet_subset: Optional[array] = None) -> (K: sparse, M: sparse)
    # frequency-oblivious; K, M are complex-symmetric N_e x N_e sparse matrices
    # over the full (unconstrained) edge-DOF space
```

Module 6 is the sole downstream consumer of `assemble(...)`. It calls it once (interior tets,
cached for the sweep) and, once Module 5 exists, once per frequency (PML tets, fresh
`MaterialAssembly` each time) — both through the exact same function, differing only in which
tets and which material registry are passed in.
