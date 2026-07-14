# Module 4 — `ports`: Waveguide Mode Ports — Equations & Implementation Plan

Companion to the top-level architecture doc and Modules 0–3. Same conventions: no code,
precise equations, step-by-step build order, validation targets — with one exception, flagged
explicitly in §3.6: the exact matrix-block arrangement of the port-mode eigenproblem has
multiple equivalent conventions in the literature, and this document is honest about which
parts are derived with full confidence versus which single arrangement should be
cross-checked against a reference during implementation.

**Notation fixed for this entire document**: propagation direction is $\hat x$ (the global
length axis — this is where microstrip literature's generic "$\hat z$, axial direction" maps
onto *our* axes). The port cross-section is the global $(y,z)$ plane (width, height) at fixed
$x$. Transverse gradient $\nabla_t = (\partial_y,\partial_z)$. The field is split into a
transverse vector $\mathbf e_t=(e_y,e_z)$ and an axial scalar $e_x$ — this is a direct,
unrenamed use of the field's own $x$-component, not a new symbol. $\zeta$ denotes the
**local** axial coordinate measured from a port face into the domain ($\zeta=x$ for Port 1,
$\zeta=L-x$ for Port 2); modes vary as $e^{-\gamma\zeta}$.

---

## 0. What Module 4 owns vs. consumes

**Consumes**: Module 1's `boundary_faces('PORT_p')` and `boundary_faces('PEC')` (no new Module
1 contract additions needed — everything here is derived from what Module 1 already exposes,
including the `tet_volume_tag` accessor added for Module 3); Module 2's `MaterialAssembly`
(ports query it exactly as Module 3 does, just restricted to the 2D port face); Module 3's
Whitney basis definitions (reused conceptually — the port's own 2D edge basis is built the same
way, on triangles instead of tets).

**Owns**: the 2D mode eigenproblem (`ports.mode_solver`) and the surface-term-to-matrix-block
conversion (`ports.port_operator`), including the shared power normalization that threads
through both and into Module 7 (extraction).

**Does not own**: the 3D volume assembly (Module 3); PML (Module 5); the frequency sweep
orchestration or final linear solve (Module 6); S-parameter extraction proper (Module 7) —
though §4.3 flags a refinement to the top-level doc's projection formula that Module 7 must
inherit.

---

## 1. Design invariant recap

Every port face lies in an **isotropic feed section** — guaranteed by geometry (Module 0: the
LC cutout length is strictly less than the line length, and centered, so both port faces sit in
plain `SUBSTRATE`+`AIR` cross-sections). This is what makes the 2D mode problem below an
ordinary isotropic, inhomogeneous (dielectric step at the substrate/air interface) waveguide
eigenproblem — no anisotropy to contend with at the ports, ever.

A **single, unit-power normalization** is fixed once (§4.2) and reused without re-derivation in
the mode solver, the port operator, and Module 7's extraction — this is the thread that
prevents the classic bug class of inconsistent reference-impedance conventions across modules.

---

## 2. The 2D port cross-section: derived, not separately meshed

### 2.1 Extraction from the existing 3D mesh

`boundary_faces('PORT_p')` already returns the list of `(tet, local_face)` triangles making up
the port cross-section. **Module 4 does not generate its own 2D mesh** — it directly reuses
these triangles' vertices and connectivity (projected trivially, since they already lie in the
$x=\text{const}$ plane) as the 2D triangulation. This guarantees automatic conformity with the
3D mesh (no separate discretization to keep in sync) and means the 2D problem's DOF space is a
genuine subset of the 3D edge space (§2.5).

### 2.2 Material tagging

Each extracted 2D triangle inherits the `tet_volume_tag` (Module 1, added for Module 3) of its
adjacent 3D tet — `SUBSTRATE` below the interface, `AIR` above. `MaterialAssembly.epsilon(tag,
points)` is queried exactly as in Module 3, just evaluated at 2D quadrature points. No new
material code; this is the same interface, called from a different geometric context.

### 2.3 The trace as an internal PEC segment — the one non-obvious subtlety

The line conductor is zero-thickness (Module 0 §0, point 4) and sits *inside* the port
cross-section — air above it, substrate below — not on the cross-section's outer boundary.
Because Module 0 embeds the trace-footprint partition over the **full length** $[0,L]$ (Module
0 §3, step 4), its intersection with each port plane is already present as a conformal internal
edge chain at $z=h_{\text{sub}}$, $y\in[y_0^{\text{trace}},y_1^{\text{trace}}]$ in the extracted
2D mesh — this falls out of Module 0's construction for free, nothing new needed here.

A 2D mesh edge is **PEC** (tangential $\mathbf e_t=0$ enforced, DOF eliminated) iff it is also
an edge of some `PEC`-tagged 3D face (Module 1's already-combined `PEC = PEC_LINE ∪
PEC_GROUND` tag). This single check correctly captures both the ground-plane edge (outer
boundary of the cross-section) and the internal trace segment — **no node duplication is
needed for the zero-thickness trace**: a single shared set of edge DOFs along that internal
line, constrained to zero, correctly enforces vanishing tangential $E$ as seen from both the
air side and the substrate side simultaneously, which is exactly the physical zero-thickness-PEC-sheet
condition.

The cross-section's outer lateral edges ($y=0$, $y=W_{\text{sub}}$) are `PMC_SIDE` — natural
boundary, no term imposed, exactly as in the 3D problem. Module 4 therefore only needs to
identify the **PEC** edge set; everything else defaults to natural automatically.

### 2.4 2D barycentric geometry (triangle analogue of Module 1 §2)

For a triangle with vertices $\mathbf p_0,\mathbf p_1,\mathbf p_2$ in the $(y,z)$ plane, 2D
barycentric coordinates $\lambda_i$ satisfy the same defining relations as Module 1 §2.1, with
the $4\times4$ geometry matrix replaced by a $3\times3$ one:

$$P_{\triangle} = \begin{bmatrix}1&1&1\\ y_0&y_1&y_2\\ z_0&z_1&z_2\end{bmatrix}, \qquad 2A = \det P_\triangle, \qquad \nabla_t\lambda_i = \text{last two entries of row } i \text{ of } P_\triangle^{-1}.$$

Same orientation-normalization discipline as Module 1 §2.3 (permute vertices once so $A>0$).

### 2.5 2D Whitney edge basis and its (scalar) curl

Identical construction to Module 3 §2, restricted to a triangle's three edges. For local edge
$(a,b)$: $\mathbf N_{(a,b)}(\mathbf r) = \lambda_a\nabla_t\lambda_b - \lambda_b\nabla_t\lambda_a$.
Its curl is a **scalar** in 2D (the curl of a planar vector field is a scalar, not a vector):

$$\text{curl}_t\,\mathbf N_{(a,b)} = 2\,(\nabla_t\lambda_a \times \nabla_t\lambda_b)_{\text{2D}} = 2\left(\frac{\partial\lambda_a}{\partial y}\frac{\partial\lambda_b}{\partial z} - \frac{\partial\lambda_a}{\partial z}\frac{\partial\lambda_b}{\partial y}\right),$$

constant per triangle, same factor of 2 as the 3D case, derived by the identical vector-identity
argument. **Global edge orientation for this 2D sub-mesh reuses the exact same global ascending
vertex-index convention Module 1 already fixed for the 3D edges** — since every 2D edge here
*is* a 3D global edge (§2.1), its canonical ascending order is already determined; Module 4
computes its own per-triangle sign $s_\ell$ the same way Module 1 does for tets, from the same
global ordering, rather than inventing a separate 2D convention.

### 2.6 Nodal (Lagrange) basis for $e_x$

Standard linear (P1) nodal basis on the same triangulation, one DOF per vertex, using the same
barycentric coordinates as §2.4: $L_i(\mathbf r) = \lambda_i(\mathbf r)$. PEC vertices (those on
the ground-plane or trace edges) are Dirichlet-constrained ($e_x=0$), matching the PEC
tangential condition's implication for the axial field at a conductor.

---

## 3. The port-mode eigenproblem

### 3.1 Ansatz

$$\mathbf E(y,z,\zeta) = \big[\mathbf e_t(y,z) + \hat x\, e_x(y,z)\big]\,e^{-\gamma\zeta}$$

in an isotropic medium ($\varepsilon_r(y,z)$ scalar, $\mu_r=1$, per the design invariant §1).

### 3.2 Curl identity

Using $\nabla = \nabla_t + \hat x\,\partial_\zeta$ and $\partial_\zeta \to -\gamma$ on this
ansatz, the identical derivation used in the 3D case (Module 3 §2.3's vector-identity pattern,
here applied at the PDE level rather than to a single basis function) gives:

$$\nabla\times\mathbf E = \hat x\,(\text{curl}_t\,\mathbf e_t) - \hat x\times(\nabla_t e_x + \gamma\,\mathbf e_t)$$

### 3.3 Transverse PDE

Applying the same curl identity a second time to $\nabla\times\nabla\times\mathbf E$ (the
2D identities $\text{curl}_t(\hat x\times\mathbf D_t)=\nabla_t\cdot\mathbf D_t$ and
$\hat x\times(\hat x\times\mathbf D_t)=-\mathbf D_t$ for transverse $\mathbf D_t$), then
imposing $\nabla\times\nabla\times\mathbf E - k_0^2\varepsilon_r\mathbf E=0$ and separating the
transverse component:

$$\boxed{\;\nabla_t\times(\nabla_t\times\mathbf e_t) - k_0^2\varepsilon_r\,\mathbf e_t = \gamma^2\mathbf e_t + \gamma\,\nabla_t e_x\;}$$

### 3.4 Axial PDE

Separating the axial ($\hat x$) component of the same equation:

$$\boxed{\;\nabla_t\cdot(\nabla_t e_x + \gamma\,\mathbf e_t) + k_0^2\varepsilon_r\, e_x = 0\;}$$

These two PDEs are the confidently-derived core of this section — every vector identity used
to reach them is the direct 2D analogue of the identities already verified in Module 3.

### 3.5 Weak form — matrix block definitions

Discretize $\mathbf e_t$ in the 2D Whitney basis $\mathbf N_i$ (§2.5) and $e_x$ in the nodal
basis $L_i$ (§2.6). Test the transverse PDE with $\mathbf N_i$ (integrating the curl-curl term
by parts; the boundary term vanishes on `PEC` edges, where $\mathbf N_i$ is constrained to zero
by the test-space choice, and on `PMC_SIDE` automatically, per the natural-BC default). Test
the axial PDE with $L_i$ (integrating by parts; same vanishing-boundary-term argument). Define,
with $\tilde e_x \equiv \gamma e_x$ (the standard scaling that removes an explicit $\gamma$ from
the axial unknown, used below):

$$S_{tt,ij} = \int_S (\text{curl}_t\mathbf N_i)(\text{curl}_t\mathbf N_j)\,dS - k_0^2\int_S \varepsilon_r\,\mathbf N_i\cdot\mathbf N_j\,dS$$

$$S_{zz,ij} = \int_S \nabla_tL_i\cdot\nabla_tL_j\,dS - k_0^2\int_S \varepsilon_r\,L_iL_j\,dS$$

$$T_{tt,ij} = \int_S \mathbf N_i\cdot\mathbf N_j\,dS \qquad T_{zt,ij} = \int_S \nabla_tL_i\cdot\mathbf N_j\,dS \qquad S_{tz,ij} = -\int_S \mathbf N_i\cdot\nabla_tL_j\,dS$$

Note the exact algebraic relation $S_{tz} = -T_{zt}^{T}$ (both are the same integral up to
which index is on $\mathbf N$ vs. $\nabla_tL$ — a useful **runtime consistency check**,
independent of the eigenproblem arrangement question below.

### 3.6 The generalized eigenvalue problem — derived result, with an explicit honesty flag

Substituting $\tilde e_x=\gamma e_x$ into both weak forms and collecting terms gives a
well-defined generalized eigenvalue problem in $\gamma^2$:

$$\begin{bmatrix}S_{tt} & S_{tz}\\ 0 & S_{zz}\end{bmatrix}\begin{Bmatrix}\mathbf e_t\\ \tilde e_x\end{Bmatrix} = \gamma^2\begin{bmatrix}T_{tt} & 0\\ T_{zt} & 0\end{bmatrix}\begin{Bmatrix}\mathbf e_t\\ \tilde e_x\end{Bmatrix}$$

**This is derived correctly from §3.3–3.4** (every step verified), but the raw block
arrangement above is **not manifestly symmetric** — several equivalent conventions exist in the
literature (Jin, *The Finite Element Method in Electromagnetics*, Ch. 4; Lee, Sun & Cendes,
1991) for testing/scaling the axial equation that produce a symmetric-definite pencil instead,
and getting the sign/scaling choice exactly right without a reference to check against is easy
to get subtly wrong. **Implementation guidance**: cross-check the exact final block arrangement
against one such reference during implementation, rather than trusting this document's specific
arrangement as the single correct bookkeeping. Regardless of which equivalent arrangement is
used, the physics forces two things that serve as the real correctness criteria (§8): $\gamma^2$
must come out **real** for a lossless ($\varepsilon_r$ real) cross-section, and the dominant
mode's $\beta=\mathrm{Im}(\gamma)$ must match the analytic microstrip formulas (Hammerstad–
Jensen/Wheeler) at low frequency. Use a general (non-symmetric-pencil-capable) eigensolver (e.g.
QZ) regardless, since it handles both the symmetric and non-symmetric arrangements correctly —
there's no need to force symmetry algebraically if a general solver is used throughout.

### 3.7 Mode selection

Solve for the lowest few eigenvalues (2–3, per the top-level doc's cross-polarization
requirement) at each frequency, sorted by decreasing $\beta$ (the dominant, quasi-TEM-like mode
has the largest propagating $\beta$; higher-order modes have smaller $\beta$ or are evanescent).
Retain $(\gamma_m, \mathbf e_m, \tilde e_{x,m})$ for each retained mode $m$.

### 3.8 Recovering the transverse magnetic field

From $\mathbf H = (j/\omega\mu_0)\nabla\times\mathbf E$ and §3.2's curl identity, the transverse
part is:

$$\mathbf h_{t} = -\frac{j}{\omega\mu_0}\,\hat x\times(\nabla_t e_x + \gamma\,\mathbf e_t) = -\frac{j}{\omega\mu_0}\,\hat x\times\left(\frac{1}{\gamma}\nabla_t\tilde e_x + \gamma\,\mathbf e_t\right)$$

computed directly from the solved eigenvector, no additional linear solve required.

---

## 4. Modal admittance and normalization

### 4.1 Modal admittance $Y_m$

For any single guided eigenmode (this holds generally, not just for pure TE/TM — a consequence
of $\mathbf h_t$ being linearly tied to $\mathbf e_t$ through the same eigenmode structure),
$\mathbf h_t = Y_m\,\hat x\times\mathbf e_t$ for a mode-specific complex scalar $Y_m$. Extract it
robustly (rather than pointwise, which is noise-sensitive) via a least-squares-style projection:

$$Y_m = \frac{\int_S \mathbf h_t\cdot(\hat x\times\mathbf e_t)^{*}\,dS}{\int_S |\mathbf e_t|^2\,dS}$$

**Consistency check**: recompute the mode's power two independent ways and confirm agreement —
directly via the Poynting integral $P_m = \tfrac12\mathrm{Re}\int_S(\mathbf e_t\times\mathbf
h_t^{*})\cdot\hat x\,dS$, and via $P_m = \tfrac12\mathrm{Re}(Y_m)\int_S|\mathbf e_t|^2\,dS$ (which
follows algebraically from the $Y_m$ relation above). Disagreement flags an error in the
extracted $Y_m$ or in $\mathbf h_t$'s reconstruction (§3.8).

### 4.2 Power normalization

Scale the raw eigenvector $(\mathbf e_t,\mathbf h_t)\to(\alpha\mathbf e_t,\alpha\mathbf h_t)$ so
$P_m=1$:

$$|\alpha|^2 = \frac{1}{\tfrac12\mathrm{Re}(Y_m)\int_S|\mathbf e_t^{\text{raw}}|^2\,dS}$$

Fix $\alpha$'s phase by convention (e.g. real and positive at the point of maximum
$|\mathbf e_t|$) — the phase is physically arbitrary for a linear mode but must be applied
consistently so repeated solves at nearby frequencies don't flip sign discontinuously.

### 4.3 Modal projection — a refinement to the top-level doc's sketch

The top-level architecture doc's §4.2/§7 sketched modal amplitude extraction as an $E$-field
self-overlap, $V_m = (1/N_m)\int\mathbf E_t\cdot\mathbf e_m\,dS$. **This document refines that**:
the standard, generally-correct modal decomposition uses the **biorthogonality relation**
$\int_S(\mathbf e_m\times\mathbf h_n)\cdot\hat x\,dS = P_m\delta_{mn}$ (which holds generally,
not just when the $\mathbf e_m$ happen to be mutually $L^2$-orthogonal — a stronger condition
the simplified sketch implicitly assumed). With unit-power normalization ($P_m=1$), the correct
projection of an arbitrary tangential field $\mathbf E_t$ onto mode $m$ is:

$$\boxed{\;a_m = \int_S (\mathbf E_t\times\mathbf h_m)\cdot\hat x\,dS\;}$$

with **no denominator at all** once $P_m=1$ is baked into $\mathbf h_m$'s normalization — simpler
than the top-level sketch, not just more robust. **This refinement must propagate to Module 7**
(S-parameter extraction) when that module is specced; the projection there should use this
$H$-field-based overlap, not the $E$-field self-overlap sketched earlier.

---

## 5. Port operator: surface term → $\mathbf B_p$, $\mathbf g_p$

### 5.1 Derivation

On the port face, the retained surface term from Module 3 §3.1's weak form is $j\omega\mu_0
\int_{S_p}\mathbf W_i\cdot(\hat n^{\text{out}}\times\mathbf H_t)\,dS$, with $\hat
n^{\text{out}}=-\hat x$ (outward from the 3D domain, opposite the into-domain $\hat x$ used in
§3.1's ansatz). Expand the total tangential field as incident plus reflected modal
contributions, $\mathbf E_t = \sum_m(a_m^++a_m^-)\mathbf e_m$, $\mathbf H_t =
\sum_m(a_m^+-a_m^-)\mathbf h_m$ (the relative sign on $\mathbf H_t$ reflects the opposite
power-flow direction of the reflected wave). Using $\hat x\times\mathbf h_m = -Y_m\mathbf e_m$
(§4.1's relation) and the §4.3 projection to express $a_m^-$ in terms of the actual (unknown)
solved field, then separating the piece proportional to the unknown FEM coefficients $a_j$ from
the piece proportional to the known incident amplitude $a_m^{+,\text{inc}}$:

$$B_{p,ij} = -j\omega\mu_0\sum_m Y_m\left(\int_{S_p}\mathbf W_i\cdot\mathbf e_m\,dS\right)\left(\int_{S_p}(\mathbf W_j)_t\times\mathbf h_m\cdot\hat x\,dS\right)$$

$$g_{p,i} = -2j\omega\mu_0\sum_m Y_m\,a_m^{+,\text{inc}}\int_{S_p}\mathbf W_i\cdot\mathbf e_m\,dS$$

**Honesty flag, matching the top-level doc's own stance on this exact term**: the overall sign
here came out opposite the top-level doc's earlier compressed sketch of $g_{p,i}$. This is
exactly the kind of thing a sign convention (direction of $\hat x$, definition of $a_m^-$, phase
of the ansatz) can flip, and neither derivation should be trusted blindly — **the correctness
gate is reciprocity ($S_{21}=S_{12}$) and passivity ($|S|\le1$), asserted in Module 8, not the
sign tracked through any single derivation.** $B_{p,ij}$ is manifestly symmetric in $(i,j)$ up
to the ordering of the two integral factors — a useful structural check independent of the
overall-sign question.

### 5.2 Caching

The surface overlap integrals $\int_{S_p}\mathbf W_i\cdot\mathbf e_m\,dS$ and
$\int_{S_p}(\mathbf W_j)_t\times\mathbf h_m\cdot\hat x\,dS$ depend only on $(p,m,\omega)$ — not
on which excitation is being driven. Compute and cache them **once per port per mode per
frequency**; reuse for $\mathbf B_p$ (built once per frequency) and for every excitation's
$\mathbf g_p$ (one per (port, mode) driven in the sweep), and hand the same cached values to
Module 7's extraction (§4.3), which needs the identical overlaps.

### 5.3 Excitation convention

For a given frequency and a given (port, mode) excitation $(q,n)$: $a_n^{+,\text{inc},(q)}=1$,
all other $a_m^{+,\text{inc},(p)}=0$. In practice only the dominant mode ($n=1$) is driven per
port in the sweep (standard single-mode-per-port excitation convention); higher captured modes
are never directly excited but still contribute to $\mathbf B_p$'s loading and can receive
scattered power, extracted via Module 7.

---

## 6. De-embedding

Fields decay as $e^{-\gamma_m\zeta}$ from the port face into the domain. Shifting a reference
plane a distance $d_p$ further into the domain multiplies the port-face amplitude by
$e^{-\gamma_md_p}$; recovering the port-face-referenced amplitude from a plane shifted by $d_p$
therefore requires the inverse factor:

$$S^{\text{ref}}_{(p,m),(q,n)} = S_{(p,m),(q,n)}\,e^{+\gamma_md_p}\,e^{+\gamma_nd_q}$$

a per-mode analytic phase/attenuation correction, applied in Module 7.

---

## 7. Step-by-step build order

1. Extract each port's 2D triangulation from `boundary_faces('PORT_p')` (§2.1); tag triangles
   via `tet_volume_tag` (§2.2); identify the 2D PEC edge set via intersection with
   `boundary_faces('PEC')` (§2.3) — verify this correctly captures both the ground-plane edge
   and the internal trace segment on a hand-inspected small test case before proceeding.
2. 2D barycentric geometry and Whitney/nodal bases (§2.4–2.6), unit-tested the same way as
   Module 1/3's 3D analogues (partition-of-unity, Kronecker reproduction, curl reproduction).
3. Assemble $S_{tt}, S_{zz}, T_{tt}, T_{zt}, S_{tz}$ (§3.5) for a single test frequency on a
   simple two-material (substrate+air) cross-section; verify $S_{tz}=-T_{zt}^{T}$ numerically.
4. Solve the generalized eigenproblem (§3.6) via a general (QZ-capable) solver; verify real
   $\gamma^2$ for the lossless case **before** trusting any subsequent result — this is the
   first checkpoint that would catch a block-arrangement error from §3.6.
5. Reconstruct $\mathbf h_t$ (§3.8); extract $Y_m$ (§4.1) and run its Poynting-integral
   consistency check; apply the power normalization (§4.2).
6. Validate the dominant mode's $\beta(\omega)$ against Hammerstad–Jensen/Wheeler — this is the
   analytic gate that most directly tests whether §3.6's arrangement, despite the honesty flag,
   produced physically correct results.
7. Implement the §4.3 projection and verify biorthogonality ($\int(\mathbf e_m\times\mathbf
   h_n)\cdot\hat x\,dS\approx\delta_{mn}$ for the retained modes) numerically.
8. Implement `ports.port_operator` (§5): $\mathbf B_p$, then $\mathbf g_p$ for a single-mode
   excitation, with the caching strategy (§5.2) built in from the start.
9. Implement de-embedding (§6) — simple, low-risk relative to the rest of this module.
10. Run the full §8 validation suite.

---

## 8. Validation targets

- **Real eigenvalues (lossless case)**: $\gamma^2$ real to numerical tolerance for real
  $\varepsilon_r$ — the primary, non-negotiable check on §3.6's arrangement, run before anything
  downstream is trusted.
- **Analytic gate**: dominant mode $\beta(\omega)$ vs. Hammerstad–Jensen/Wheeler, converging
  under mesh refinement (ties into the top-level doc's Phase 1 gate).
- **Biorthogonality**: $\int(\mathbf e_m\times\mathbf h_n)\cdot\hat x\,dS \approx \delta_{mn}$
  for the retained modes, at several frequencies.
- **$Y_m$ consistency** (§4.1): the two independent power computations agree.
- **$S_{tz}=-T_{zt}^{T}$** (§3.5): exact algebraic identity, checked regardless of the §3.6
  arrangement question.
- **PEC edge correctness** (§2.3): confirm both the outer ground-plane edge and the internal
  trace segment are captured, on a hand-constructed small test mesh where the expected edge
  set is known in advance.
- **$\ge 2$ modes captured**, with the second mode's field pattern visually/numerically
  distinct from the dominant mode (not a numerical duplicate) — guards against a degenerate or
  under-resolved eigensolve silently returning the same mode twice.
- **$B_p$ symmetry** (§5.1): structural check independent of the overall sign question.
- **End-to-end sign resolution**: since §3.6 and §5.1 both carry explicit honesty flags, the
  real acceptance criterion for this module is the top-level doc's Phase 1 gate — reciprocity
  and $|S_{11}|^2+|S_{21}|^2=1$ on the full assembled system (Modules 3+4+6) for a uniform
  lossless line. If that gate fails, the sign questions flagged in §3.6/§5.1 are the first two
  places to check.

---

## 9. Interface / class contract

```
# ports.mode_solver
class PortModeSolver:
    def solve(port_tag: str, omega: float, n_modes: int = 2) -> list[PortMode]

class PortMode:
    gamma: complex             # propagation constant
    e_t, h_t: callable         # transverse field profiles, power-normalized (P_m = 1)
    Y: complex                 # modal admittance

# ports.port_operator
def build_B(port_modes: dict[str, list[PortMode]], mesh, omega) -> sparse matrix   # global-DOF sized, low-rank/local
def build_g(port_modes, excitation: dict[(str,int), complex], mesh, omega) -> vector
def deembed(S: array, port_modes, offsets: dict[str, float]) -> array
```

Module 6 calls `build_B` once per frequency (alongside the cached interior $\mathbf K,\mathbf
M$ from Module 3) and `build_g` once per excited (port, mode) pair per frequency. Module 7
consumes the same `PortMode` objects (their cached overlap integrals, §5.2) for extraction —
no independent re-derivation of the modal fields there.
