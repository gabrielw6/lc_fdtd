# Module 1 — `mesh.interface`: Geometry Layer — Equations & Implementation Plan

Companion to `architecture_fem_sparameter_modules.md`. Same conventions as the
cavity-perturbation docs: no code, only scope, precise equations with checkable reduction
identities, step-by-step build order, and validation targets.

Module 1 owns the **geometry of the mesh**, not its generation. Mesh generation (Delaunay /
Gmsh / the existing mesh package) is external and consumed. What Module 1 turns "a list of
vertices and tetrahedra with physical tags" into is the set of per-element geometric
quantities every downstream operator integrates against: the constant barycentric gradients
$\nabla\lambda_i$, element volumes, a **globally consistent oriented edge list** (without
which edge-element assembly silently produces a wrong $\mathbf K,\mathbf M$), boundary faces
grouped by tag, and quadrature points/weights satisfying $\sum_q w_q = \mathrm{vol}$.

Keep the boundary sharp: if you find yourself building the Whitney basis
$\mathbf W_e = \lambda_a\nabla\lambda_b - \lambda_b\nabla\lambda_a$ here, that belongs to
Module 3 (`fem.edge_elements`). Module 1 stops at $\nabla\lambda_i$ and the oriented edge
list — the raw material Module 3 assembles the basis from.

---

## 0. What Module 1 owns vs. consumes

**Consumes** from the external mesh module (read-only inputs):

- `vertices` — $(N_v, 3)$ array of node coordinates $\mathbf r_v$.
- `tets` — $(N_t, 4)$ integer array; each row is four global vertex indices $[v_0,v_1,v_2,v_3]$.
- `volume_tags` — $(N_t,)$ region label per tet: one of `SUBSTRATE`, `AIR`, `LC`, `PML_*`.
- `surface_tags` — a map from a tagged boundary group name (`PEC_LINE`, `PEC_GROUND`,
  `PORT_1`, `PORT_2`, `PML_OUTER`) to a set of triangular faces, each face given as three
  global vertex indices.

**Owns** (everything below), and exposes to Modules 3–7:

- Per-tet geometry: signed volume $V$, four barycentric gradients $\nabla\lambda_i$ (constant
  vectors), face areas and outward normals.
- Global edge topology: the unique oriented edge list, the per-tet local→global edge map, and
  the per-tet edge sign $s_e\in\{+1,-1\}$.
- Boundary-face extraction and tag resolution.
- Quadrature rules on tetrahedra (volume integrals) and on triangles (port/PEC/PML surface
  integrals), parametrized by exactness order, with barycentric→physical point mapping.

**Does not own**: the Whitney/Nédélec basis and its curl (Module 3), the material tensor
(Module 2), anything frequency-dependent.

---

## 1. Design decision: work directly in physical coordinates (no Piola map)

For **lowest-order** Nédélec elements on **straight-sided (affine)** tetrahedra — which is the
entire scope here, since the geometry is rectangular and the LC region is a rectangular cutout
meshed with straight tets — the barycentric coordinates are affine functions of position, so
$\nabla\lambda_i$ is **constant within each element** and computable directly from the four
vertex coordinates. There is therefore **no need** for a reference-element-to-physical
covariant (Piola) transform: the basis is built and integrated in physical coordinates.

This is a deliberate non-implementation. The covariant transform
$\mathbf W^{\text{phys}} = J^{-T}\mathbf W^{\text{ref}}$ (and its curl counterpart with
$J/\det J$) is required only for curved elements or higher-order bases. Building it now would
be dead code and a source of index-transposition bugs. Record the assumption explicitly so a
future higher-order extension knows exactly what changes: *straight-sided + lowest-order ⇒
physical-coordinate assembly; anything else ⇒ reintroduce the Piola map here.*

---

## 2. Barycentric coordinates and their gradients

### 2.1 Definition

On a tet with vertices $\mathbf r_0,\mathbf r_1,\mathbf r_2,\mathbf r_3$, the barycentric
coordinate $\lambda_i(\mathbf r)$ is the unique affine function with

$$\lambda_i(\mathbf r_j) = \delta_{ij}, \qquad \sum_{i=0}^{3}\lambda_i(\mathbf r) = 1, \qquad \sum_{i=0}^{3}\lambda_i(\mathbf r)\,\mathbf r_i = \mathbf r.$$

Being affine, $\lambda_i(\mathbf r) = \alpha_i + \boldsymbol\beta_i\cdot\mathbf r$ with
$\boldsymbol\beta_i = \nabla\lambda_i$ a constant vector — the quantity Module 3 needs.

### 2.2 Computation via the geometry matrix

Assemble the $4\times4$ matrix

$$P = \begin{bmatrix} 1 & 1 & 1 & 1 \\ x_0 & x_1 & x_2 & x_3 \\ y_0 & y_1 & y_2 & y_3 \\ z_0 & z_1 & z_2 & z_3 \end{bmatrix}.$$

From the constraints of §2.1, $P\,[\lambda_0,\lambda_1,\lambda_2,\lambda_3]^{T} = [1,x,y,z]^{T}$,
hence $[\lambda_i] = P^{-1}[1,x,y,z]^{T}$. Reading off the affine coefficients:

$$\boxed{\;\alpha_i = (P^{-1})_{i,0}, \qquad \nabla\lambda_i = \big((P^{-1})_{i,1},\ (P^{-1})_{i,2},\ (P^{-1})_{i,3}\big)\;}$$

i.e. **$\nabla\lambda_i$ is the last three entries of row $i$ of $P^{-1}$.** One $4\times4$
inversion per tet yields all four gradients and all four constants at once.

### 2.3 Signed volume

$$\boxed{\;6V = \det P\;}$$

$V$ is signed: its sign encodes the orientation of the stored vertex ordering. Either take
$|V|$ for all magnitude uses (integration weights, face areas) **or** — cleaner — permute each
tet's vertices once at load so $\det P > 0$ for every element, giving a globally consistent
positive orientation. Do the permutation, and record that all downstream code may assume
$V>0$; this removes a class of sign bugs in the curl term of Module 3.

### 2.4 Geometric interpretation (used as a check, and for §4)

$\nabla\lambda_i$ points from the face opposite vertex $i$ toward vertex $i$, with magnitude
equal to the reciprocal of the perpendicular distance $h_i$ from vertex $i$ to that face.
Since $h_i = 3V/A_i$ (with $A_i$ the area of the opposite face):

$$\nabla\lambda_i = -\frac{A_i}{3V}\,\hat{\mathbf n}_i^{\text{out}},$$

where $\hat{\mathbf n}_i^{\text{out}}$ is the outward unit normal of the face opposite vertex
$i$. This is the bridge between the algebraic form (§2.2) and the face geometry (§4), and it
makes one of the §2.5 checks fall out for free.

### 2.5 Consistency checks (per tet, cheap, run on every element at load)

- **Partition of unity gradient**: $\sum_{i=0}^{3}\nabla\lambda_i = \mathbf 0$. Follows from
  $\sum_i\lambda_i \equiv 1$. A nonzero sum means $P^{-1}$ was computed or sliced wrong. This
  is the single cheapest catch-all for a broken $\nabla\lambda_i$.
- **Kronecker reproduction**: evaluating $\lambda_i(\mathbf r_j) = \alpha_i + \nabla\lambda_i\cdot\mathbf r_j$
  returns $\delta_{ij}$ (to machine precision) — validates the full affine coefficients, not
  just the gradients.
- **Degeneracy guard**: $|\det P| = 6|V| > \varepsilon_{\text{geo}}$ (a floor scaled by the
  mesh's characteristic edge length cubed). A near-zero volume is a sliver/degenerate tet the
  mesh module should not have produced; raise rather than divide by it in §2.2.

---

## 3. Global edge topology and orientation — the correctness linchpin

Lowest-order edge elements carry **one degree of freedom per edge**, equal to the line
integral of tangential $\mathbf E$ along that edge. The Whitney function for edge $(a,b)$ is
$\mathbf W_{(a,b)} = \lambda_a\nabla\lambda_b - \lambda_b\nabla\lambda_a$ (Module 3), and its
tangential integral along its own edge is $+1$ **for one orientation and $-1$ for the reverse**.
If two elements sharing an edge disagree on its orientation, one contributes $+\mathbf E_{\text{tan}}$
and the other $-\mathbf E_{\text{tan}}$ at the same DOF, breaking tangential continuity and
corrupting both $\mathbf K$ and $\mathbf M$ in a way that does **not** throw an error — it just
yields wrong fields. Module 1 removes this failure mode by fixing a single global convention.

### 3.1 Local edge numbering (fixed convention)

Each tet's six edges, in fixed local order:

$$e_0{=}(0,1),\ e_1{=}(0,2),\ e_2{=}(0,3),\ e_3{=}(1,2),\ e_4{=}(1,3),\ e_5{=}(2,3).$$

Each tet's four faces, named by the opposite local vertex:

$$f_0{=}(1,2,3),\ f_1{=}(0,2,3),\ f_2{=}(0,1,3),\ f_3{=}(0,1,2).$$

### 3.2 Global orientation rule

**Every edge is globally oriented from its lower global vertex index to its higher.**
Concretely, for a local edge with global endpoint indices $(g_a, g_b)$:

- Its canonical key is the sorted pair $(\min(g_a,g_b),\ \max(g_a,g_b))$ — used to identify
  the shared global edge across elements.
- Its per-tet sign is $s_e = +1$ if the local ordering already matches ascending global
  order, else $s_e = -1$.

Module 3 then builds the element basis as $s_e\big(\lambda_a\nabla\lambda_b - \lambda_b\nabla\lambda_a\big)$
using the **global-ascending** $(a,b)$, so both elements sharing an edge compute an identical
tangential trace and the DOF is single-valued. The sign $s_e$ is the only bookkeeping needed;
because the Whitney trace depends on orientation but not on which element you view it from,
ascending-global ordering is sufficient and no further reconciliation is required.

### 3.3 Outputs

- `edges` — unique global oriented edge list (ascending pairs); its length $N_e$ is the global
  edge-DOF count.
- `tet_edge_map` — $(N_t, 6)$ array: for each tet, the global edge index of each of its six
  local edges.
- `tet_edge_sign` — $(N_t, 6)$ array of $s_e\in\{\pm1\}$.

### 3.4 Consistency checks

- **Sign self-consistency**: for every shared edge, all incident tets reference the same global
  edge index; the product of stored sign and local-vs-global ordering is coherent (i.e.
  recomputing $s_e$ from `edges` reproduces `tet_edge_sign`).
- **Orientation of the canonical list**: every entry of `edges` is strictly ascending
  ($\text{edge}[0] < \text{edge}[1]$). A single non-ascending entry means the dedup/orientation
  pass has a bug that will surface as a localized wrong field, not a crash.

---

## 4. Face geometry — areas and outward normals

Needed for boundary integrals (PEC essential-BC identification, port and PML surface terms)
and for the §2.4 geometric interpretation.

For a triangular face with vertices $\mathbf p_0,\mathbf p_1,\mathbf p_2$:

$$\mathbf a = (\mathbf p_1-\mathbf p_0)\times(\mathbf p_2-\mathbf p_0), \qquad
A = \tfrac12\,\lvert\mathbf a\rvert, \qquad
\hat{\mathbf n} = \frac{\mathbf a}{\lvert\mathbf a\rvert}.$$

**Outward orientation convention**: for a face of a tet, "outward" is the normal pointing away
from the tet's opposite (fourth) vertex $\mathbf p_{\text{opp}}$. Fix the sign by requiring

$$\hat{\mathbf n}^{\text{out}}\cdot(\mathbf p_{\text{opp}} - \mathbf p_0) < 0,$$

flipping $\hat{\mathbf n}$ if the test fails. For a boundary face (belongs to exactly one tet,
§5) this is the physical outward normal used in the surface terms of Modules 4–5.

**Consistency check (per tet, and the tie-in to §2.5)**: the closed-surface identity

$$\sum_{k=0}^{3} A_k\,\hat{\mathbf n}_k^{\text{out}} = \mathbf 0$$

must hold to tolerance for the four faces of every tet (area-weighted outward normals of a
closed polyhedron sum to zero). Note this is *exactly* $\sum_i\nabla\lambda_i=\mathbf 0$
(§2.5) rescaled by §2.4 — so the two checks corroborate each other, and a failure of one but
not the other localizes the bug to either the gradient path or the face path.

---

## 5. Boundary-face extraction and tagging

### 5.1 Face incidence

Enumerate all $4N_t$ local faces, key each by its sorted triple of global vertices, and count
incidences:

- **interior face** ⇒ shared by exactly **2** tets;
- **boundary face** ⇒ belongs to exactly **1** tet.

Any face with incidence $\notin\{1,2\}$ signals a non-conforming or corrupt mesh — raise.

### 5.2 Count identity (mesh-integrity check)

$$4N_t = 2\,(\#\,\text{interior faces}) + 1\,(\#\,\text{boundary faces}).$$

This is a strong, cheap global check on the connectivity: it fails if any tet references a bad
vertex index or if the mesh has a crack (a face that should be shared appearing as two
distinct boundary faces). Prefer this over an Euler-characteristic test — it catches the
connectivity errors that actually occur and is unaffected by the topology of the LC cutout
region.

### 5.3 Tag resolution

Match each extracted boundary face against the `surface_tags` groups by its sorted vertex
triple, producing the tagged sets the solver consumes:

- `PEC` = `PEC_LINE` ∪ `PEC_GROUND` — where Module 6 imposes the essential BC
  $\hat{\mathbf n}\times\mathbf E = 0$ (these edge DOFs are constrained out, not assembled).
- `PORT_p` — one set per port; consumed by Module 4 (port face for the 2D mode solve and the
  surface term).
- `PML_OUTER` — PEC-backed outer wall; treated as `PEC` for BC purposes.

**Coverage check**: every boundary face resolves to exactly one tag; an untagged boundary face
means the model has an open surface the solver would treat as a perfect magnetic wall by
default — almost always unintended, so raise and report its location.

---

## 6. Quadrature contract

Module 1 supplies quadrature; Module 3 chooses the order based on the material variation
(constant vs. spatially varying $\varepsilon_r$). Both volume (tet) and surface (triangle)
rules are needed.

### 6.1 Barycentric-to-physical mapping

Rules are tabulated in barycentric coordinates $(\hat\lambda_0,\hat\lambda_1,\hat\lambda_2,\hat\lambda_3)$
with reference weights $\hat w_q$ normalized so $\sum_q\hat w_q = 1$. Map to physical space:

$$\mathbf r_q = \sum_{i} \hat\lambda_{q,i}\,\mathbf r_i, \qquad w_q = \hat w_q \cdot V.$$

**Required addition (driven by Module 3's needs)**: `quadrature_tet(order)` returns
`(points, weights, barycentric)`, exposing the barycentric weight array $\hat\lambda_{q,i}$
(shape $(M,4)$) alongside the physical points and scaled weights. These barycentric weights are
already computed internally as part of the mapping above — Module 3 needs them directly to
evaluate barycentric-coordinate-based basis functions (its $\lambda_i(\mathbf r_q)=\hat\lambda_{q,i}$)
without redundantly reconstructing them from the affine coefficients $\alpha_i,\nabla\lambda_i$.

The volume scaling by $V$ gives the required identity

$$\boxed{\;\sum_q w_q = V\;}$$

which is the **exact** contract Module 2's field-integration checks assume
(`region.quadrature_points` volume-consistency). Surface rules map identically with triangle
area $A$ in place of $V$, giving $\sum_q w_q = A$.

### 6.2 Exactness orders to expose

For **constant** $\varepsilon_r$ the mass integrand $\mathbf W_i\cdot\mathbf W_j$ is quadratic
in position (product of two linear Whitney functions), and the curl integrand
$(\nabla\times\mathbf W_i)\cdot(\nabla\times\mathbf W_j)$ is **constant**. So the minimum
workhorse rules are:

- **Order-1** (1 point at the centroid $\hat\lambda_i=\tfrac14$, $\hat w=1$): exact for the
  constant curl-curl integrand — sufficient for $\mathbf K$ with constant $\mu_r$.
- **Order-2** (4-point symmetric rule): exact for the quadratic mass integrand — sufficient for
  $\mathbf M$ with constant $\varepsilon_r$.

For **spatially varying** $\varepsilon_r$ (Phase 2 onward), the integrand is no longer
polynomial; expose higher symmetric rules (standard tabulated Keast rules, exactness orders
up to $\sim6$) so Module 3 can raise the order and run its $N$-vs-$N{+}2$ doubling check. The
contract is: `quadrature_tet(order)` returns points+weights exact to the requested polynomial
degree; the caller owns the order choice.

### 6.3 Consistency checks

- **Volume/area reproduction**: for every rule and every element, $\sum_q w_q$ equals $V$
  (resp. $A$) to tolerance — the same check Module 2 runs defensively, satisfied here by
  construction so it never fires downstream on a correct mesh.
- **Positivity**: all $\hat w_q > 0$ (use only positive-weight symmetric rules; negative
  weights degrade the conditioning of $\mathbf M$ and can break the passivity checks in
  Module 8).
- **Polynomial exactness spot-test**: integrating a monomial of the rule's claimed degree over
  a reference tet returns the analytic value — run once per tabulated rule at load, not per
  element.

---

## 7. Step-by-step build order

1. **Vertex/tet ingestion + orientation normalization** (§2.3): load `vertices`, `tets`;
   permute each tet's vertices so $\det P > 0$; store the permutation-consistent connectivity.
   Downstream assumes $V>0$.
2. **Per-tet geometry** (§2): for each tet, form $P$, invert once, extract $\{\nabla\lambda_i\}$
   and $V$; run the §2.5 checks. Cache $\{\nabla\lambda_i, V\}$ (these are reused every
   frequency and never change).
3. **Face geometry** (§4): compute the four face areas and outward normals per tet; run the
   §4 closed-surface check.
4. **Edge topology** (§3): enumerate local edges, dedup to the ascending-ordered global
   `edges`, build `tet_edge_map` and `tet_edge_sign`; run the §3.4 checks.
5. **Boundary extraction + tagging** (§5): face-incidence pass, count identity (§5.2), tag
   resolution and coverage check (§5.3).
6. **Quadrature tables** (§6): load positive-weight symmetric tet and triangle rules; run the
   per-rule exactness spot-test (§6.3). Wire the barycentric→physical mapping.
7. **Run the Section 8 validation suite** before any Module 3 code consumes this module.

Steps 2–3 produce the immutable per-element cache the sweep reuses; nothing in Module 1 is
frequency-dependent, so all of it runs exactly once per model.

---

## 8. Validation targets

- **Total volume**: $\sum_{\text{tets}} V$ equals the analytic volume of the rectangular model
  (bounding box minus nothing — the LC region is a tagged sub-volume, not a void) to
  quadrature tolerance. The single strongest end-to-end geometry check.
- **Per-tet identities** (§2.5, §4): $\sum_i\nabla\lambda_i=\mathbf 0$, Kronecker reproduction
  $\lambda_i(\mathbf r_j)=\delta_{ij}$, closed-surface $\sum_k A_k\hat{\mathbf n}_k^{\text{out}}=\mathbf 0$
  — assert on every element, not a sample.
- **Edge orientation** (§3.4): construct a two-tet patch sharing one face (hence three edges)
  and confirm the shared edges receive identical global indices and coherent signs from both
  tets; confirm a deliberately mis-ordered input tet is repaired by step 1, not propagated.
- **Face-count identity** (§5.2): holds exactly on the real mesh; and confirm it *fails* on a
  deliberately cracked test mesh (a duplicated vertex splitting one shared face into two
  boundary faces), so the check is known to protect something.
- **Boundary coverage** (§5.3): every boundary face tagged; confirm an untagged-face test mesh
  raises.
- **Quadrature** (§6.3): $\sum_q w_q = V$ on every tet for every exposed order; monomial
  exactness per rule; all weights positive.
- **New contract items** (§9): `tet_volume_tag` reproduces the input `volume_tags` array
  exactly for every tet; `pec_edge_dofs()` matches the edge set obtained by manually
  enumerating the three edges of every `PEC`-tagged face and intersecting against the global
  `edges` list — run as an independent cross-check rather than trusting the same code path
  twice.

---

## 9. Interface / class contract

```
class MeshInterface:
    # immutable per-element geometry (computed once at load)
    n_vertices: int
    n_tets: int
    n_edges: int                      # global edge-DOF count

    grad_lambda(tet: int) -> (4,3) array      # the four constant gradients
    volume(tet: int) -> float                 # V > 0 after orientation fix
    face_area_normal(tet, local_face) -> (float, (3,) array)   # A, outward n-hat

    # edge topology
    edges: (n_edges, 2) int array             # ascending global pairs
    tet_edge_map: (n_tets, 6) int array       # local edge -> global edge index
    tet_edge_sign: (n_tets, 6) int array      # s_e in {+1,-1}

    # volume tagging (required by Module 3, to dispatch material queries per tet)
    tet_volume_tag(tet: int) -> str           # or: volume_tags, (n_tets,) array

    # boundary
    boundary_faces(tag: str) -> list[(tet, local_face)]   # tag in {PEC, PORT_p, PML_OUTER, ...}

    # essential-BC support (required by Module 6, derived here since it's purely geometric)
    pec_edge_dofs() -> set[int]               # global edge indices lying on a PEC-tagged face

    # quadrature
    quadrature_tet(order: int) -> (points: (M,3), weights: (M,), barycentric: (M,4))
    quadrature_tri(face, order: int) -> (points: (M,3), weights: (M,))   # sum weights == A
```

Contract summary for consumers: `grad_lambda`, `volume`, and the edge maps are what Module 3
assembles $\mathbf K,\mathbf M$ from; `tet_volume_tag` is what Module 3 uses to dispatch each
element's quadrature-point material query to the right entry in Module 2's `MaterialAssembly`;
`boundary_faces('PORT_p')` + `quadrature_tri` are what Module 4 builds the port term from;
`pec_edge_dofs()` is the DOF set Module 6 eliminates when applying the essential BC (Module 3
itself never applies it — it assembles the full unconstrained system); and
`quadrature_tet(order)`'s `barycentric` output is what Module 3 evaluates its basis functions
against directly, without reconstructing $\lambda_i$ from the affine coefficients. Nothing here
depends on frequency or material — Module 1 is assembled once and read many times.
