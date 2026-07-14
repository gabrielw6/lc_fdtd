# FEM Driven S-Parameter Solver — Module Architecture & Equations

Companion to the cavity-perturbation architecture docs. Same conventions: no code here,
only module scope, precise mathematics with checkable reduction identities, build order,
and validation targets. Where a module's math has an exact closed-form or invariant that a
correct implementation *must* reproduce, it is stated as a consistency check (mirroring the
Module 2 "Section 1.4" pattern), because those are the tests that catch bugs three modules
away from where they surface.

MoM / cavity-Q is out of scope for this document.

---

## 0. Global conventions (fixed once, inherited everywhere)

These are not negotiable per-module; they are the contract every module is written against.

- **Time convention**: $e^{+j\omega t}$. Consequences, used throughout:
  - A passive lossy dielectric has $\varepsilon_r = \varepsilon_r' - j\varepsilon_r''$ with
    $\varepsilon_r'' \ge 0$; equivalently $\varepsilon_r = \varepsilon_r'(1 - j\tan\delta)$.
  - The PML complex stretch is $s = \kappa - j\,\sigma/(\omega\varepsilon_0)$ with $\sigma \ge 0$
    (Module 5). A sign error here produces a *gain* medium — the single most common PML bug.
- **Primary unknown**: the electric field $\mathbf E(\mathbf r)$, expanded in lowest-order
  Nédélec (edge) basis functions $\mathbf W_j$: $\mathbf E \approx \sum_j a_j \mathbf W_j$.
  Edge elements enforce tangential continuity, leave normal components free (correct for
  dielectric interfaces), and place the spurious-mode null space at DC where it is harmless
  to a driven solve.
- **Relative material tensors**: $\varepsilon_r$, $\mu_r$ are dimensionless $3\times3$
  tensors. Here $\mu_r = \mathbf I$ everywhere (no magnetic bias), but the assembler must
  still accept a tensor $\mu_r$ because the PML region (Module 5) supplies an anisotropic
  $\mu_r \ne \mathbf I$.
- **Free-space wavenumber**: $k_0 = \omega\sqrt{\mu_0\varepsilon_0} = \omega/c_0$.
- **Dimensional convention for the system**: eigen/driven parameter is $k_0^2$, i.e. the
  master operator is $(\mathbf K - k_0^2\mathbf M)$. The shorthand $(\mathbf K - \omega^2\mathbf M)$
  used in the design notes corresponds to folding $\mu_0\varepsilon_0$ into $\mathbf M$; this
  document keeps $k_0^2$ explicit for dimensional correctness.

### 0.1 Package layout

```
lc_sparam/
  conventions.py            constants, k0(omega), time-convention helpers
  mesh/
    interface.py            MeshProvider adapter over the existing mesh module
  material/
    core.py                 MaterialModel (ABC)
    regions.py              RegionMaterial   (tag -> constant or scalar field)
    tensor_interpolation.py DirectorFieldMaterial (the only module touching n(r))
  fem/
    edge_elements.py        Nedelec basis, curl, quadrature rules
    assembly.py             StiffnessAssembler (K), MassAssembler (M)
  ports/
    mode_solver.py          PortModeSolver: 2D eigenproblem -> (beta_m, e_m, h_m)
    port_operator.py        PortOperator: B_p(omega) matrix block + g_p(omega) RHS
  pml/
    stretching.py           StretchProfile, PMLMaterial (wraps a background MaterialModel)
  solve/
    system.py               GlobalSystem: (K - k0^2 M + sum B_p) a = sum g_p
    sweep.py                FrequencySweep: cache interior blocks, rebuild port/PML per omega
  extract/
    sparameters.py          mode-overlap extraction -> S(omega), + de-embedding
  validation/
    analytic_microstrip.py  Hammerstad-Jensen / Wheeler references
    checks.py               reciprocity, passivity, convergence, reduction gates
```

### 0.2 Module dependency graph

```
mesh.interface ─┐
                ├─> fem.assembly ─┐
material.core ──┤                 ├─> solve.system ─> solve.sweep ─> extract.sparameters
   (regions,    ├─> ports.*  ─────┤
    tensor_int, ├─> pml.* ────────┘
    via core)   │
                └─ validation.* observes all of the above
```

Two rules the graph encodes:

1. **`material.core` is the only material interface anyone imports.** `regions`,
   `tensor_interpolation`, and `pml.PMLMaterial` are all *implementations* of it. The
   assembler, port solver, and PML never learn whether $\varepsilon_r$ came from a constant,
   a scalar field, or a director file. This is what makes the Phase 1→4 build sequence a
   change of *input only*.
2. **`ports` and `pml` depend on `material.core` but not on each other**, and neither
   depends on `solve`. They contribute blocks; `solve.system` sums them.

---

## 1. Module: `mesh.interface` — consumed contract

**Owns**: nothing physical. It is an adapter that presents the already-built mesh module
through the exact contract the FEM solver needs, so the solver has no dependency on that
module's internal representation.

**Consumes** from the existing mesh module and **must expose**:

- `tets` — tetrahedra with vertex coordinates and a consistent global edge numbering
  (required for Nédélec DOF assignment; the sign convention on shared edges must be global
  and deterministic).
- `edges` — global edge list with orientation; this is the DOF set.
- `boundary_faces(tag)` for tags: `PEC` (line conductor + ground plane), `PORT_p` (one per
  port face), `PML_*` (outer shell faces, PEC-backed), and `LC` (the cutout boundary — used
  only to tell the material module which region to query the director field in; **not** a
  material-conformance requirement, since $\varepsilon_r$ is continuous across it up to the
  $\varepsilon_\perp/\varepsilon_\parallel$ contrast).
- `quadrature_points(tet, order)` -> `(points, weights)` with
  $\sum_i w_i = \mathrm{vol(tet)}$.

**Consistency check (run once at load)**: $\sum_{\text{tets}} \mathrm{vol} =$ analytic bounding
volume of the rectangular model; and Euler characteristic of the edge/face/tet counts is
consistent (catches a broken edge map before it produces a silently wrong $\mathbf K$).

---

## 2. Module: `material` — the tensor field $\varepsilon_r(\mathbf r)$

This is the module the whole method exists to enable. It is split exactly as flagged in the
design notes: a core interface, a region-based implementation, and a separate
director-field interpolation implementation.

### 2.1 `material.core` — the interface

```
class MaterialModel(ABC):
    def epsilon(self, points: (M,3) array) -> (M,3,3) array   # relative eps tensor at each point
    def mu(self,      points: (M,3) array) -> (M,3,3) array   # relative mu  tensor (I here)
```

Vectorized over quadrature points by contract, because the assembler evaluates it once per
element at all quadrature points at once. Every downstream module sees only this.

**Consistency checks (unit tests against every implementation)**:

- Symmetry: `epsilon(p)` returns a symmetric tensor at every point,
  $\varepsilon_r = \varepsilon_r^{T}$ (no magnetic bias ⇒ reciprocal ⇒ symmetric, *not*
  merely Hermitian). A returned non-symmetric tensor is a bug, and it is the bug that later
  breaks S-matrix symmetry (Module 8), so catch it here.
- Passivity: $\mathrm{Im}(\varepsilon_r) \preceq 0$ (negative-semidefinite imaginary part)
  under the $e^{+j\omega t}$ convention. A positive imaginary eigenvalue is a gain medium.

### 2.2 `material.regions` — isotropic / scalar path (Phases 1–2)

For region tag $t$, returns $\varepsilon_r(\mathbf r) = \varepsilon_{r,t}(\mathbf r)\,\mathbf I$,
where $\varepsilon_{r,t}$ is either a constant (Phase 1) or a scalar function of position
(Phase 2). Air $\to \mathbf I$, substrate $\to \varepsilon_{r,\text{sub}}\mathbf I$,
conductors are handled by the PEC boundary condition (no volume material).

This path exercises the quadrature-point sampling machinery (Phase 2) **before** any
anisotropy exists, so that "does $\mathbf M$ integrate a spatially varying coefficient
correctly" is validated in isolation from "does $\mathbf M$ handle off-diagonal terms."

### 2.3 `material.tensor_interpolation` — the LC path (Phase 4)

The **only** module that reads the director file. The LC constitutive relation:

$$\varepsilon_r(\mathbf r) = \varepsilon_\perp\,\mathbf I + (\varepsilon_\parallel - \varepsilon_\perp)\,\mathbf n(\mathbf r)\,\mathbf n(\mathbf r)^{T},\qquad |\mathbf n| = 1.$$

Eigenvalues: $\varepsilon_\parallel$ along $\mathbf n$, $\varepsilon_\perp$ (doubly
degenerate) transverse. So $\varepsilon_r$ is a **uniaxial** tensor whose optic axis is the
director.

**The interpolation rule (this is the load-bearing decision)**: interpolate the six
independent components of the *tensor* $\varepsilon_r$, never the director vector
$\mathbf n$. Justifications, both of which are also consistency checks:

- **Headless-line invariance**: $\mathbf n$ and $-\mathbf n$ are the same physical state, so
  $(-\mathbf n)(-\mathbf n)^{T} = \mathbf n\mathbf n^{T}$. Interpolating $\mathbf n$
  component-wise collapses toward zero across any sign flip in the input field; interpolating
  $\varepsilon_r$ does not. **Check**: feeding the director file and its pointwise negation
  must produce *bit-identical* $\varepsilon_r$ samples.
- **Norm preservation**: linear interpolation of unit vectors does not preserve
  $|\mathbf n|=1$, which would corrupt the eigenvalue spread. Interpolating $\varepsilon_r$
  keeps eigenvalues in $[\varepsilon_\perp, \varepsilon_\parallel]$ automatically.
  **Check**: at every interpolated point, the eigenvalues of $\varepsilon_r$ lie within
  $[\min(\varepsilon_\perp',\varepsilon_\parallel'),\ \max(\varepsilon_\perp',\varepsilon_\parallel')]$
  to interpolation tolerance, and $\mathrm{tr}(\varepsilon_r') = 2\varepsilon_\perp' + \varepsilon_\parallel'$
  is invariant across the whole LC region (the trace is rotation-invariant, so it must not
  vary with director orientation — a strong, cheap global check).

Pipeline: director file (points + $\mathbf n$, or precomputed $\varepsilon_r$ grid) →
assemble $\varepsilon_r$ at input points → interpolate the 6 components to the assembler's
quadrature points → cache for the whole sweep (subject to negligible
$\varepsilon_\perp/\varepsilon_\parallel$ dispersion over the band).

**Module boundary contract (validated in Phase 4)**: a *spatially uniform* director must
reproduce, bit-for-bit within solver tolerance, the direct-tensor-input path of Module 3's
Phase-3 tests. This is what lets the general path be validated against Phase-1 analytics via
the $\Delta\varepsilon \to 0$ reduction, without ever deriving new closed-form anisotropic
microstrip formulas.

---

## 3. Module: `fem.assembly` — the operators $\mathbf K$, $\mathbf M$

**Owns**: assembly of the two volume operators from the vector-wave equation. Consumes edge
basis + curl from `fem.edge_elements`, material tensors from `material.core`, mesh/quadrature
from `mesh.interface`.

### 3.1 Strong and weak form

Source-free interior, driven through port boundaries. The vector wave equation for $\mathbf E$:

$$\nabla\times\!\big(\mu_r^{-1}\,\nabla\times\mathbf E\big) - k_0^2\,\varepsilon_r\,\mathbf E = 0 \quad\text{in } V.$$

Galerkin weak form with Nédélec test functions $\mathbf W_i$, after integrating the
curl-curl term by parts:

$$\underbrace{\int_V (\nabla\times\mathbf W_i)\cdot\big(\mu_r^{-1}\,\nabla\times\mathbf E\big)\,dV}_{\to\ \mathbf K}
\;-\; k_0^2 \underbrace{\int_V \mathbf W_i\cdot\big(\varepsilon_r\,\mathbf E\big)\,dV}_{\to\ \mathbf M}
\;-\; \oint_{\partial V} \mathbf W_i\cdot\big(\hat{\mathbf n}\times\mu_r^{-1}\nabla\times\mathbf E\big)\,dS \;=\;0.$$

The surface term vanishes on PEC (essential BC $\hat{\mathbf n}\times\mathbf E=0$) and on the
PEC-backed PML outer wall. It survives only on the port faces, where it becomes the port
operator (Module 4).

### 3.2 The matrices

With $\mathbf E = \sum_j a_j \mathbf W_j$:

$$\boxed{\;K_{ij} = \int_V (\nabla\times\mathbf W_i)\cdot\big(\mu_r^{-1}\,\nabla\times\mathbf W_j\big)\,dV\;}\qquad
\boxed{\;M_{ij} = \int_V \mathbf W_i\cdot\big(\varepsilon_r\,\mathbf W_j\big)\,dV\;}$$

Evaluated by quadrature: at each quadrature point $\mathbf r_q$ in element $e$,

$$M_{ij}^{(e)} = \sum_q w_q\; \mathbf W_i(\mathbf r_q)\cdot\big[\varepsilon_r(\mathbf r_q)\,\mathbf W_j(\mathbf r_q)\big],$$

where $\varepsilon_r(\mathbf r_q)$ is the **full tensor** from `material.core`. This is the
one and only generalization from the scalar cavity code: the scalar $\varepsilon$ becomes a
$3\times3$ contraction $\mathbf W_i\cdot\varepsilon_r\,\mathbf W_j$. Nothing else in the
assembler is aware of anisotropy.

### 3.3 Consistency checks

- **Symmetry (not Hermiticity)**: because $\varepsilon_r = \varepsilon_r^{T}$ (Module 2.1),
  $\mathbf W_i\cdot\varepsilon_r\mathbf W_j = \mathbf W_j\cdot\varepsilon_r\mathbf W_i$, so
  $\mathbf M = \mathbf M^{T}$ even when complex (lossy LC). Likewise $\mathbf K=\mathbf K^{T}$.
  The assembled global system is therefore **complex-symmetric**, which dictates the solver
  in Module 6 (a complex-symmetric factorization, e.g. $LDL^{T}$, not Cholesky and not a
  Hermitian solver). A test must assert $\|\mathbf M - \mathbf M^{T}\| \approx 0$ to machine
  precision — a failure here is a transposed tensor index in the contraction, and it is the
  proximate cause of a non-reciprocal S-matrix.
- **Quadrature order vs. material variation**: for a spatially varying $\varepsilon_r$, the
  integrand is no longer element-wise polynomial, so a quadrature order tuned for constant
  $\varepsilon_r$ under-integrates it (a variational crime that aliases). Rerunning at order
  $N$ and $N{+}2$ must change $\mathbf M$ by less than truncation tolerance; this is the
  Module 2 "doubling" idea applied to the mass matrix, and it must gate the LC assembly.
- **Null-space sanity**: $\mathbf K$ has a large null space (gradient fields,
  $\nabla\times\nabla\phi=0$); this is expected for edge elements and harmless for the driven
  solve because $-k_0^2\mathbf M$ and the port operator regularize the system away from
  $k_0=0$. Do **not** attempt to "fix" a singular $\mathbf K$; verify instead that
  $(\mathbf K - k_0^2\mathbf M + \sum_p\mathbf B_p)$ is nonsingular at each swept $\omega$.

---

## 4. Module: `ports` — modal ports and excitation

The design notes call this the fiddliest part; the architecture isolates it into two
sub-modules and threads a **single normalization choice** through both plus the extractor
(Module 7), because the recurring class of port bug is an inconsistency between how the mode
is normalized in the solver, in the operator, and in the S-extraction.

**Design invariant**: every port face lies in an **isotropic feed section** (plain
substrate, no LC — guaranteed by the geometry, since the LC cutout length is strictly less
than the line length). Consequences: the port cross-section problem is isotropic and
inhomogeneous (substrate + air), i.e. the standard, well-characterized quasi-TEM microstrip
mode solver, testable against analytics in isolation.

### 4.1 `ports.mode_solver` — 2D cross-section eigenproblem

On a port face, fields propagate as $e^{-\gamma z}$ with $\gamma = \alpha + j\beta$
(propagating $\Rightarrow \gamma = j\beta$). Split into transverse and longitudinal parts:

$$\mathbf E(x,y,z) = \big[\mathbf e_t(x,y) + \hat{\mathbf z}\,e_z(x,y)\big]\,e^{-\gamma z}.$$

Substituting into the 3D wave equation collapses it to a 2D generalized eigenvalue problem
on the port cross-section, discretized in a **mixed** basis: 2D Nédélec (edge) elements for
$\mathbf e_t$, nodal (Lagrange) elements for $e_z$. This pairing is what keeps the two field
components at their correct differential order and suppresses spurious transverse modes. In
block form, for unknowns $(\mathbf e_t, e_z)$ at fixed $\omega$:

$$
\begin{bmatrix} \mathbf S_{tt} & \mathbf 0 \\ \mathbf 0 & 0 \end{bmatrix}
\begin{Bmatrix} \mathbf e_t \\ e_z \end{Bmatrix}
= \beta^2
\begin{bmatrix} \mathbf T_{tt} & \mathbf T_{tz} \\ \mathbf T_{zt} & \mathbf T_{zz} \end{bmatrix}
\begin{Bmatrix} \mathbf e_t \\ e_z \end{Bmatrix},
$$

with, for isotropic feed material $\varepsilon_r,\mu_r$,

$$S_{tt,ij} = \int_{S_p}\!\Big[\tfrac{1}{\mu_r}(\nabla_t\times\mathbf N_i)(\nabla_t\times\mathbf N_j) - k_0^2\,\varepsilon_r\,\mathbf N_i\!\cdot\!\mathbf N_j\Big]dS,$$

and $\mathbf T$ blocks built from $\int \mathbf N_i\!\cdot\!\mathbf N_j$,
$\int \mathbf N_i\!\cdot\!\nabla_t L_j$, $\int \nabla_t L_i\!\cdot\!\nabla_t L_j$ against
$\tfrac1{\mu_r}$. (The standard $\tilde e_z = \gamma\,e_z$ scaling is applied so the problem
is linear in the eigenvalue $\beta^2$; that substitution is an implementation detail of this
module and does not leak out.) The eigenvalue is $\beta^2$ at the given $\omega$; the solver
runs **once per port per frequency** because $\beta$ and the modal profile depend on $\omega$.

**Outputs per mode $m$**: propagation constant $\gamma_m$, transverse electric profile
$\mathbf e_m(x,y)$, transverse magnetic profile $\mathbf h_m(x,y)$, and a scalar
normalization $N_m$ (§4.3).

**Consistency checks**:

- Orthogonality: $\int_{S_p} \mathbf e_m\cdot\mathbf e_n\,dS = 0$ for $m\ne n$ (distinct
  eigenvectors of the generalized problem).
- Mode count: capture $\ge 2$ modes per port. The dominant quasi-TEM mode is the design
  mode, but the LC's off-diagonal $\varepsilon_r$ inside the device converts power into a
  second port mode; truncating to one mode silently discards that power and biases $S$.
- Analytic gate (Phase 1): the dominant $\beta$ and derived $Z_0$ match Hammerstad–Jensen /
  Wheeler for the microstrip geometry within a few percent, converging under $h$-refinement.

### 4.2 `ports.port_operator` — surface term $\to$ matrix block $+$ RHS

On a port face the retained surface term of §3.1 is, using $\mu_r^{-1}\nabla\times\mathbf E = -j\omega\mu_0\mathbf H$,

$$-\oint_{S_p}\mathbf W_i\cdot\big(\hat{\mathbf n}\times\mu_r^{-1}\nabla\times\mathbf E\big)\,dS
= j\omega\mu_0\oint_{S_p}\mathbf W_i\cdot(\hat{\mathbf n}\times\mathbf H)\,dS.$$

Expand the tangential field in port modes with incident/reflected amplitudes
$a_m^{+}, a_m^{-}$. The transverse $\mathbf H$ satisfies
$\hat{\mathbf n}\times\mathbf h_m = Y_m\,\mathbf e_m$ with modal admittance $Y_m$ (mode- and
$\omega$-dependent), and the two propagation directions carry opposite $\mathbf H$ sign, so
$\hat{\mathbf n}\times\mathbf H_t = \sum_m Y_m(a_m^{+}-a_m^{-})\mathbf e_m$. Eliminating
$a_m^{-}$ via the modal projection of the FEM field on the face,
$a_m^{-} = \tfrac{1}{N_m}\int_{S_p}\mathbf E_t\cdot\mathbf e_m\,dS - a_m^{+}$, gives a block
that adds to the system matrix and a forcing vector:

$$\boxed{\;B^{(p)}_{ij} = j\omega\mu_0\sum_m \frac{Y_m}{N_m}\Big(\!\int_{S_p}\!\mathbf W_i\cdot\mathbf e_m\,dS\Big)\Big(\!\int_{S_p}\!\mathbf W_j\cdot\mathbf e_m\,dS\Big)\;}$$

$$\boxed{\;g^{(p)}_i = 2\,j\omega\mu_0\sum_m Y_m\,a_m^{+,\text{inc}}\int_{S_p}\!\mathbf W_i\cdot\mathbf e_m\,dS\;}$$

$B^{(p)}$ is symmetric (outer product of the same face-overlap vector), so it preserves the
complex-symmetry of the global system — a **required** structural check. It is also
**low rank and DOF-local**: nonzero only on edges touching the port face, one rank-1 update
per captured mode. That locality is what makes rebuilding it every frequency cheap (Module 6).

**The one place to be disciplined, not clever**: $Y_m$ and $N_m$ are *definition-dependent*
for a quasi-TEM mode (power vs. voltage-path vs. current-loop reference impedance). Fix a
single **power-based** definition and use the identical $(Y_m, N_m)$ in the mode solver, this
operator, and the extractor (Module 7). Do not re-derive $Y_m$ independently in the extractor.
The correctness of the constant placement above is asserted by the reciprocity/passivity
gates in Module 8, **not** trusted from this derivation — that is the intended safety net for
exactly this term.

### 4.3 Power normalization (the shared choice)

Choose $N_m$ so each mode carries unit time-averaged power through the face:

$$P_m = \tfrac12\,\mathrm{Re}\!\int_{S_p}(\mathbf e_m\times\mathbf h_m^{*})\cdot\hat{\mathbf n}\,dS = 1.$$

With this single choice, the reflected/transmitted modal amplitude $a_m^{-}$ *is* the
power-wave amplitude, and Module 7's S-parameters are direct amplitude ratios with no
extra $\sqrt{\mathrm{Re}\,Z}$ bookkeeping. This is the thread that ties the three port-related
modules together.

---

## 5. Module: `pml` — absorbing boundary as an anisotropic material

**Design invariant**: PML and LC are **spatially disjoint**. LC lives in the interior cutout;
PML is the outer shell truncating substrate and air. Neither region's tensor is ever composed
with the other's. Absorbing *inside* the anisotropic active material is avoided rather than
solved.

### 5.1 Complex coordinate stretching $\to$ material tensor

Under the $e^{+j\omega t}$ convention, the stretch factor along axis $q\in\{x,y,z\}$ is

$$s_q(\omega, \mathbf r) = \kappa_q(\mathbf r) - j\,\frac{\sigma_q(\mathbf r)}{\omega\varepsilon_0},\qquad \kappa_q\ge 1,\ \sigma_q\ge 0.$$

The complex-coordinate-stretched curl-curl operator is **identically** the ordinary operator
with modified material tensors $\varepsilon_r \to \Lambda\,\varepsilon_r$,
$\mu_r \to \Lambda\,\mu_r$, where

$$\Lambda = \mathrm{diag}\!\left(\frac{s_y s_z}{s_x},\ \frac{s_z s_x}{s_y},\ \frac{s_x s_y}{s_z}\right).$$

So `pml.PMLMaterial` is *itself* a `MaterialModel` (Module 2.1) that **wraps the isotropic
background** of the shell region and returns $\Lambda\,\varepsilon_{r,\text{bg}}$ and
$\Lambda\,\mu_{r,\text{bg}}$. The assembler is unchanged — it just receives a different tensor
in the shell. Because $\Lambda$ multiplies an *isotropic* background (never the LC tensor),
the result stays diagonal in the shell and there is no tensor-composition problem.

### 5.2 Loss profile

Polynomial grading over PML thickness $d$, standard form:

$$\sigma_q(\xi) = \sigma_{\max}\left(\frac{\xi}{d}\right)^{n},\qquad
\sigma_{\max} = -\frac{(n+1)\ln R_0}{2\,\eta\, d},$$

with $\xi$ the depth into the PML along its outward normal, $n\in\{2,3\}$, $\eta$ the wave
impedance of the background, and $R_0$ the target normal-incidence reflection (e.g.
$10^{-5}$). Faces stretch in one axis; edges and corners of the shell stretch in two/three
(the $\Lambda$ formula handles all cases with the same expression).

### 5.3 Consistency checks

- **Reduction**: as $\sigma_q\to0,\ \kappa_q\to1$, then $s_q\to1$ and $\Lambda\to\mathbf I$;
  `PMLMaterial` must return exactly the background material. **Check**: with $\sigma=0$ the
  full solve reproduces the same-mesh solve with no PML region flagged, to solver tolerance.
- **Frequency dependence is rational, not affine**: $s_q(\omega)$ makes both the PML
  contribution to $\mathbf K$ (via $\Lambda\mu_r^{-1}$) and to $\mathbf M$ (via
  $\Lambda\varepsilon_r$) rational in $\omega$. Harmless for the per-point direct solve, but
  it is a **flag for any future model-order reduction**: the PML and port blocks are *not*
  affine in $\omega$, which is the regime where naive moment-matching (AWE) breaks. Recorded
  here so that decision is informed later, not discovered.
- **Reflection test**: a single propagating mode launched at one port into a matched line
  terminated by PML gives $|S_{11}|$ at the target $R_0$ floor across the band; a rise of
  $|S_{11}|$ at the low-frequency end is the signature of insufficient $\kappa$ (needed for
  evanescent/low-$\beta$ absorption).

---

## 6. Module: `solve` — global system and frequency sweep

### 6.1 `solve.system` — the assembled linear system

At each frequency:

$$\Big(\underbrace{\mathbf K}_{\text{interior, }\omega\text{-indep}} + \mathbf K_{\text{pml}}(\omega) \;-\; k_0^2\big[\underbrace{\mathbf M}_{\text{interior, }\omega\text{-indep}} + \mathbf M_{\text{pml}}(\omega)\big] \;+\; \sum_p \mathbf B_p(\omega)\Big)\,\mathbf a \;=\; \sum_p \mathbf g_p(\omega).$$

The honest system, with the frequency dependence localized exactly where the notes placed it:
the novel anisotropic-$\mathbf M$ machinery (interior) is frequency-independent; all
$\omega$-dependence lives in the geometrically localized port and PML blocks.

Solver: **complex-symmetric** factorization (Module 3.3), one direct solve per RHS. Multiple
ports/modes are multiple RHS columns sharing one factorization.

### 6.2 `solve.sweep` — caching strategy

1. Assemble and cache the interior $\mathbf K,\mathbf M$ **once** (they do not depend on
   $\omega$; the cached LC tensor samples from Module 2.3 feed $\mathbf M$).
2. Per frequency $\omega_k$: rebuild only $\mathbf K_{\text{pml}},\mathbf M_{\text{pml}}$
   (small — shell DOFs), run each port's 2D eigenproblem (Module 4.1) to get
   $(\gamma_m,\mathbf e_m)$ at $\omega_k$, build $\mathbf B_p,\mathbf g_p$ (rank-few, port-local),
   assemble, factor, solve.
3. Starting strategy is per-point direct solve. MOR/adaptive sampling is deferred; §5.3's
   rationality note is the precondition to revisit before choosing an MOR method.

**Consistency check**: nonsingularity of the assembled operator at every $\omega_k$ (a near-zero
pivot flags a resonance of the truncated domain or an under-absorbing PML, not a physical result).

---

## 7. Module: `extract.sparameters`

On port $p$ the solved field gives the tangential trace $\mathbf E_t^{(p)} = \sum_j a_j\mathbf W_j|_{S_p}$.
Project onto each mode using the shared normalization $N_m$ (Module 4.3):

$$V_m^{(p)} = \frac{1}{N_m}\int_{S_p}\mathbf E_t^{(p)}\cdot\mathbf e_m\,dS,\qquad
a_m^{-,(p)} = V_m^{(p)} - a_m^{+,\text{inc},(p)}.$$

With unit-power modes, the generalized scattering parameter from mode $n$ at port $q$
(excited, $a_n^{+,(q)}=1$, all others zero) to mode $m$ at port $p$ is the direct amplitude
ratio:

$$S_{(p,m),(q,n)} = a_m^{-,(p)}\Big|_{\text{excite }(q,n)}.$$

**De-embedding** to reference planes shifts each port's plane by $d_p$ toward the device with
an analytic per-mode phase/attenuation factor:

$$S^{\text{ref}}_{(p,m),(q,n)} = S_{(p,m),(q,n)}\;e^{+\gamma_m d_p}\,e^{+\gamma_n d_q}.$$

**Consistency checks**:

- The transverse-$\mathbf E$ / transverse-$\mathbf h$ biorthogonality
  ($\int(\mathbf e_m\times\mathbf h_n^{*})\cdot\hat{\mathbf n}=P_m\delta_{mn}$) must hold to
  the same tolerance used in Module 4.3, or the projection mixes modes.
- Extraction reuses $\mathbf e_m, N_m$ **from the mode solver's cached output** — it must not
  recompute them with a different normalization (the single-choice rule of Module 4).

---

## 8. Module: `validation` — build sequence with gates

The build sequence is the Phase 1→4 plan; each phase's gate is a hard assertion, not a spot
check. Reciprocity and passivity are the workhorses because they hold *regardless of an
analytic reference* and pinpoint the anisotropic-assembly and port-constant bugs.

### Phase 1 — uniform isotropic microstrip (validates the entire non-material stack)
- **Analytic**: extracted $\beta(\omega), Z_0$ vs. Hammerstad–Jensen / Wheeler.
- **Reciprocity**: $S_{21}=S_{12}$ to solver tolerance.
- **Passivity / energy** (lossless): $|S_{11}|^2+|S_{21}|^2 = 1$.
- **Convergence**: $\beta, Z_0$ converge monotonically under $h$-refinement at the
  edge-element order; report the observed order.
- **Length scaling**: $\arg(S_{21})$ linear in line length at fixed $\omega$.

### Phase 2 — position-dependent scalar $\varepsilon_r(\mathbf r)$ (validates quadrature sampling)
- **Reduction anchor**: $\varepsilon_r(\mathbf r)\to$ const reproduces Phase 1 to tolerance.
- **Layered reference**: a stacked-dielectric profile vs. its quasi-static $\varepsilon_{\text{eff}}$.
- **Quadrature-order sensitivity** (Module 3.3): $N$ vs. $N{+}2$ agree within truncation tol.

### Phase 3 — general symmetric tensor $\varepsilon_r(\mathbf r)$, supplied directly (validates anisotropic $\mathbf M$)
- **Axis-aligned uniaxial** (diagonal tensor): ordinary/extraordinary effective indices vs.
  closed form.
- **Rotated axes** (off-diagonal terms present; no simple closed form): gate on **invariants**
  — $S$ symmetric (reciprocity) and $|S|\preceq1$ (passivity). A reciprocity break here is
  the canonical signature of a transposed tensor index or an inconsistent port constant.
- **Cross-polarization**: a rotated tensor measurably excites the second port mode
  (mechanism check for LC-induced mode conversion).

### Phase 4 — LC region via director file (validates `tensor_interpolation` + boundary contract)
- **Module boundary**: uniform director reproduces the Phase-3 rotated-uniaxial result
  bit-for-bit within tolerance (this is the $\Delta\varepsilon\to0$-style reduction that
  validates the general path against Phase-1 analytics without new anisotropic formulas).
- **Invariants**: reciprocity + passivity on the director-driven tensor.
- **Director-tilt monotonicity**: sweeping a synthetic director tilt, $\arg(S_{21})$ varies
  monotonically with magnitude consistent with a conformal-mapping "LC region as tunable
  effective $\varepsilon$" estimate — an order-of-magnitude gate that catches sign/unit errors
  the smaller unit tests cannot.

---

## 9. Summary of the load-bearing invariants

1. `material.core` is the sole material interface; scalar/tensor/LC/PML are all
   implementations of it. Phases 1→4 change input, not solver code.
2. Interpolate $\varepsilon_r$ (6 components), never $\mathbf n$ — kills the $\pm\mathbf n$
   ambiguity and preserves eigenvalues; checked by trace-invariance and $\mathbf n{\to}{-}\mathbf n$
   identity.
3. $\varepsilon_r$ symmetric ⇒ $\mathbf M,\mathbf K,\mathbf B_p$ complex-**symmetric** ⇒
   complex-symmetric solver ⇒ $S$ symmetric. One property, checked at every layer, catches the
   dominant bug class.
4. Ports live in isotropic feed sections (geometry guarantees it); one power-based
   $(Y_m,N_m)$ definition threads mode solver → port operator → extractor.
5. PML and LC spatially disjoint; PML is an isotropic-background material wrapper; its
   $\omega$-dependence is rational (flag for future MOR).
6. Interior $\mathbf K,\mathbf M$ cached once; only port + PML blocks rebuilt per frequency.
