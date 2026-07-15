# Module 4 — `ports`: Waveguide Mode Ports — Equations & Implementation Plan

Companion to the top-level architecture doc and Modules 0–3. Same conventions: no code,
precise equations, step-by-step build order, validation targets — with one exception, flagged
explicitly in §3.6: the exact matrix-block arrangement of the port-mode eigenproblem has
multiple equivalent conventions in the literature, and this document is honest about which
parts are derived with full confidence versus which single arrangement should be
cross-checked against a reference during implementation.

**Implementation status**: implemented in `src/ports/` (`cross_section.py`, `basis2d.py`,
`mode_solver.py`, `port_operator.py`), 31 tests passing alongside the rest of the repo. Three
things independent verification changed relative to this document's original text, applied
below at their respective sections: §3.6 gained a required spurious-mode filter (not merely a
caveat); §3.7 gained a documented, currently-unresolved mode-selection limitation with a
forward reference to Module 6; §4.3's projection formula had a genuine factor-of-2 gap, now
fixed by self-normalization rather than by resolving the exact convention analytically. A
`float()`-cast bug that silently discarded $\mathrm{Im}(\varepsilon_r)$ in $S_{tt}/S_{zz}$ for
lossy materials was caught and fixed before shipping — worth noting since it's exactly the kind
of silent-wrong-number bug this whole document series has tried to guard against with runtime
assertions rather than trust.

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

### 3.6 The generalized eigenvalue problem

**Correction (post-implementation review): the block below originally had the wrong sign on the
bottom-left block, caught by re-deriving §3.4's weak form directly rather than trusting the
original bookkeeping — exactly the cross-check this section's honesty flag asked for.**
Substituting $\tilde e_x=\gamma e_x$ into §3.3's transverse weak form reproduces
$S_{tt}\mathbf e_t + S_{tz}\tilde e_x = \gamma^2 T_{tt}\mathbf e_t$ as stated below (that row is
correct and was never in question). But testing §3.4's axial PDE with $L_i$ and integrating the
divergence term by parts gives
$-\big(S_{zz}e_x\big)_i - \gamma\big(T_{zt}\mathbf e_t\big)_i = 0$, i.e.
$S_{zz}e_x = -\gamma T_{zt}\mathbf e_t$; substituting $e_x=\tilde e_x/\gamma$ gives
$S_{zz}\tilde e_x = -\gamma^2 T_{zt}\mathbf e_t$ — a **minus** sign on $T_{zt}$, not the plus
sign originally boxed here. The corrected, well-defined generalized eigenvalue problem in
$\gamma^2$ is:

$$\begin{bmatrix}S_{tt} & S_{tz}\\ 0 & S_{zz}\end{bmatrix}\begin{Bmatrix}\mathbf e_t\\ \tilde e_x\end{Bmatrix} = \gamma^2\begin{bmatrix}T_{tt} & 0\\ -T_{zt} & 0\end{bmatrix}\begin{Bmatrix}\mathbf e_t\\ \tilde e_x\end{Bmatrix}$$

**Confirmed by an independent numerical experiment**: on a homogeneous PEC-walled rectangular
waveguide (closed-form TE/TM spectrum available), the original (+$T_{zt}$) arrangement produces
the correct TE spectrum (TE modes have $e_x\equiv0$, so they never exercise this block and can't
catch a sign error in it) but a broken TM spectrum — a spurious near-degenerate cluster instead
of the discrete TM11/TM21 modes, with the true TM21 missing entirely. The corrected
($-T_{zt}$) arrangement gives clean, mesh-convergent TM11/TM21 eigenvalues matching the
analytic values. This is a strictly stronger check than "$\gamma^2$ real for lossless" (§8) —
**both signs give real $\gamma^2$**, since realness follows from a more basic structural
property, not from this specific block's sign. The Hammerstad–Jensen $\beta(\omega)$ gate this
section always intended as the real acceptance test (below) is what actually distinguishes them,
and is the check that should have caught this before the honesty flag below was ever needed.

The remaining literature-cross-check caveat is now resolved by the above, not merely deferred: no
further sign/scaling ambiguity is open here. Regardless, the physics still forces two things that
serve as correctness criteria (§8): $\gamma^2$ must come out **real** for a lossless
($\varepsilon_r$ real) cross-section, and the dominant mode's $\beta=\mathrm{Im}(\gamma)$ must
match the analytic microstrip formulas (Hammerstad–Jensen/Wheeler) at low frequency — the second
of these is what actually caught the sign error above and should be run as a matter of course,
not treated as optional. Use a general (non-symmetric-pencil-capable) eigensolver (e.g. QZ)
regardless of arrangement, since it handles both the symmetric and non-symmetric cases
correctly — there's no need to force symmetry algebraically if a general solver is used
throughout.

**Confirmed by implementation: this discretization admits spurious eigenvalue solutions**, as
the honesty flag above anticipated. A **physical-bounds filter is required, not optional**:
discard any candidate whose $\beta$ falls outside $\big[k_0\sqrt{\varepsilon_{r,\min}},\,
k_0\sqrt{\varepsilon_{r,\max}}\big]$, where $\varepsilon_{r,\min},\varepsilon_{r,\max}$ are the
smallest and largest relative permittivities present in the port cross-section (here, $1$ for
air and $\varepsilon_{r,\text{sub}}$ for the substrate) — a genuine guided mode's effective index
must lie between the extremes of the index actually present, so anything outside that band is
numerical, not physical. Verified in implementation to remove the spurious solution cleanly
without discarding legitimate modes. This filter should run immediately after the eigensolve,
before mode selection (§3.7), since a spurious solution can otherwise be mistaken for a
legitimate higher-order mode rather than recognized as numerical noise.

### 3.7 Mode selection

Solve for the lowest few eigenvalues (2–3, per the top-level doc's cross-polarization
requirement) at each frequency, sorted by decreasing $\beta$ (the dominant, quasi-TEM-like mode
has the largest propagating $\beta$; higher-order modes have smaller $\beta$ or are evanescent).
Retain $(\gamma_m, \mathbf e_m, \tilde e_{x,m})$ for each retained mode $m$.

**Known limitation, confirmed by implementation, currently unresolved**: Module 0's
`PMC_SIDE` lateral truncation makes each port cross-section a finite, PMC-walled enclosure —
not a genuinely open one — and that enclosure supports its own "box modes" (ordinary waveguide
modes of the PMC-walled cross-section) with $\beta$ values that can sit close to, or briefly
cross, the real quasi-TEM microstrip mode's $\beta$ at some frequencies and mesh resolutions.
Plain $\beta$-sorting (as specified above) can therefore occasionally select a box mode instead
of the physical mode. A field-pattern discriminator (concentration of $|\mathbf e_t|$ near the
trace) was tried and did not generalize across mesh resolutions — it is not included here, since
a heuristic that helps at one resolution and hurts at another is worse than no heuristic, not
better. This is left as a documented limitation rather than a false fix.

**Update, post-review**: §3.6's block-arrangement sign bug (since fixed) was a plausible
contributor to this limitation, and fixing it does measurably help — a resolution that previously
selected a box mode as dominant now selects the correct quasi-TEM mode. It does **not** eliminate
the limitation, though: other resolutions still select a box mode, or a marginally-converged
near-degenerate candidate in a non-dominant slot. A second mitigation attempt (an admittance-phase
threshold, $\mathrm{Re}(Y_m)/|Y_m|$, to reject reactive-dominated non-dominant candidates before
normalization) hit the same "helps here, breaks there" pattern as the field-pattern discriminator
above and was likewise reverted rather than kept as a false fix. The dominant mode remains
reliable; non-dominant retained modes are not guaranteed clean.

**Second mitigation, added post-review: decouple the port aperture from the domain
cross-section (Module 0 §1.4).** Rather than trying to discriminate box modes from the
quasi-TEM mode after the fact, shrink the PMC-walled enclosure itself so its box-mode cutoff
rises above the band of interest — standard wave-port practice (size the aperture below
$\lambda/2$). Module 0 now accepts an optional `W_port, H_port` smaller than the full
cross-section; the region outside it becomes a PEC "cap" (folded into Module 1's `PEC` tag
aggregate), which makes the aperture's own side/top walls PEC in this module's eigenproblem for
free — **no change to `cross_section.py`, `mode_solver.py`, or the eigenproblem itself**;
`boundary_faces('PORT_p')` and `pec_edge_dofs()` already return the right thing once Module 0
tags the smaller aperture. `ports.sizing.check_port_sizing` (new, informational-only) flags an
aperture that's still too large (box-mode risk) or too small (fringing-field clipping risk) at
the sweep's own frequency band, without gating anything. This reduces but does **not**
eliminate this section's limitation on its own: a restricted aperture avoids the box-mode risk
by construction only when it's actually sized below $\lambda/2$, which the mesh must also be
fine enough to resolve — an aperture sized correctly on paper but under-resolved by the mesh
can fail to find enough well-conditioned modes at all (`PortModeError`, a distinct failure mode
from mis-selecting a box mode).

**Third mitigation, added post-review: stop requiring the oversupply itself to exist
(single-mode-tolerant mode counting).** A correctly-sized aperture (previous paragraph) is often
genuinely single-mode by physics — only the quasi-TEM mode propagates, and every higher
candidate is evanescent or fails power-normalization ($\tfrac12\mathrm{Re}(Y_m)\int|\mathbf
e_t|^2\,dS\le0$) not due to a solver defect but because there is no second physical mode to
find. `solve()`'s original contract conflated "how many modes are required" with "how many to
try to return," both pinned to the single `n_modes` argument, so a single-mode port failed
outright ("only 1 power-normalizable mode found, requested 4"). `solve(port_tag, omega,
n_modes, n_desired=None)` now separates them: `n_modes` is the true required minimum (raise
`PortModeError` only if fewer are found); `n_desired` (default `n_modes`, preserving the
original behavior when omitted) is how many to *try* to collect, for `solve.run_sweep`'s
tracking oversupply. `run_sweep` passes its own `n_modes` as the required minimum and
`n_modes+2` as `n_desired` — a physically single-mode port with `n_modes=1` now runs
successfully instead of needing an inflated `n_modes` just to avoid the old hard failure.

The frequency-to-frequency tracking approach below remains the principled fix for whatever
box-mode risk remains after sizing.

**Principled path forward, deferred to Module 6**: the standard resolution for this exact
problem (used in mode-tracking waveguide-port solvers generally) is **continuity across the
frequency sweep, not an absolute per-frequency criterion**. At the sweep's starting frequency
— chosen low enough that box modes (which have their own cutoffs) are still evanescent while
the quasi-TEM mode, having no cutoff, propagates — plain $\beta$-sorting should unambiguously
select the correct mode. At each subsequent frequency, instead of re-sorting by $\beta$ alone,
select the candidate whose field pattern has the highest overlap
($\int(\mathbf e_t^{(k)}\times\mathbf h_t^{(k-1)})\cdot\hat x\,dS$, i.e. the §4.3 projection
applied between consecutive frequency points) with the *previous* frequency's selected mode.
This discriminates robustly even through a near-degenerate crossing, since it tracks field-shape
continuity rather than an instantaneous $\beta$ ordering — but it inherently requires comparing
across frequency points, which only Module 6's sweep driver can do; `ports.mode_solver` in
isolation, solving one frequency at a time, cannot implement this itself. Record this now so
it isn't rediscovered as a surprise when Module 6 is specced.

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

**Note on a real inconsistency found here, resolved in §4.3**: this section's $P_m$ uses the
*conjugated* Poynting form ($\mathbf h_t^{*}$, with the customary $\tfrac12$), while §4.3's
biorthogonality relation, as originally stated, used the *unconjugated* overlap
$\int(\mathbf e_m\times\mathbf h_n)\cdot\hat x\,dS$ with no $\tfrac12$ — the two are genuinely
different bilinear forms (conjugated overlap gives physical time-averaged power; unconjugated
overlap is the reciprocity-based pairing that S-parameter theory actually needs) and are not
guaranteed to coincide numerically just because both are loosely called "the mode's power."
Implementation confirmed a real factor-of-2 gap between them. §4.3 now resolves this by
normalizing against whichever self-overlap the code actually computes, rather than assuming the
unconjugated overlap equals $P_m=1$ by construction.

### 4.2 Power normalization

Scale the raw eigenvector $(\mathbf e_t,\mathbf h_t)\to(\alpha\mathbf e_t,\alpha\mathbf h_t)$ so
$P_m=1$:

$$|\alpha|^2 = \frac{1}{\tfrac12\mathrm{Re}(Y_m)\int_S|\mathbf e_t^{\text{raw}}|^2\,dS}$$

Fix $\alpha$'s phase by convention (e.g. real and positive at the point of maximum
$|\mathbf e_t|$) — the phase is physically arbitrary for a linear mode but must be applied
consistently so repeated solves at nearby frequencies don't flip sign discontinuously.

### 4.3 Modal projection — a refinement to the top-level doc's sketch, corrected during implementation

The top-level architecture doc's §4.2/§7 sketched modal amplitude extraction as an $E$-field
self-overlap, $V_m = (1/N_m)\int\mathbf E_t\cdot\mathbf e_m\,dS$. **This document refines that**:
the standard, generally-correct modal decomposition uses the **biorthogonality relation**
$\int_S(\mathbf e_m\times\mathbf h_n)\cdot\hat x\,dS = N_m\delta_{mn}$ (which holds generally,
not just when the $\mathbf e_m$ happen to be mutually $L^2$-orthogonal — a stronger condition
the simplified sketch implicitly assumed) — using $N_m$ here deliberately, **not** $P_m$, per
the §4.1 note above: $N_m \equiv \int_S(\mathbf e_m\times\mathbf h_m)\cdot\hat x\,dS$ is whatever
the unconjugated self-overlap actually evaluates to for this mode's normalization convention,
not assumed equal to the conjugated $P_m=1$ target.

$$\boxed{\;a_m = \frac{1}{N_m}\int_S (\mathbf E_t\times\mathbf h_m)\cdot\hat x\,dS\;}$$

**Implementation guidance, corrected from this document's original claim**: the original text
asserted the denominator could be dropped entirely once $P_m=1$ (§4.2) was applied, reasoning
that unit-power normalization already bakes in $N_m=1$. **Verified false during
implementation** — a real factor-of-2 gap between $N_m$ and $P_m$ was found and confirmed both
analytically (the conjugate/unconjugate distinction, §4.1) and numerically. The robust fix,
applied in `port_operator.py`, is to **compute $N_m$ explicitly for each mode and divide by it**,
rather than trusting any single derivation of what $N_m$ "should" equal given the normalization
convention — this sidesteps needing to resolve the exact conjugation bookkeeping analytically,
since the code is self-consistent by construction regardless of which convention it happens to
be using internally. **This same self-normalization must be applied everywhere $N_m$ or $P_m$
appears in a projection** — including Module 7's extraction, once specced. This document's
first draft claimed the $1/N_m$ factor could be dropped entirely once $P_m=1$ was applied,
reasoning that unit-power normalization already bakes in $N_m=1$; that claim is **superseded**
by the finding above — keep the explicit division.

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

**Propagating the §4.3 fix**: the derivation above uses the §4.3 projection to express $a_m^-$,
so the same self-normalization applies here — either normalize each mode so $N_m=1$ exactly
(not merely $P_m=1$) before it's used in $\mathbf B_p,\mathbf g_p$, or insert the explicit
$1/N_m$ factor into both formulas above wherever an $\mathbf e_m$–$\mathbf h_m$ overlap appears.
This document's formulas above are written assuming one of those two equivalent fixes has been
applied — confirm which approach `port_operator.py` actually takes and that it's applied
consistently everywhere an overlap of this form is used, including Module 7 once it exists.

**Honesty flag, matching the top-level doc's own stance on this exact term**: the overall sign
here came out opposite the top-level doc's earlier compressed sketch of $g_{p,i}$. This is
exactly the kind of thing a sign convention (direction of $\hat x$, definition of $a_m^-$, phase
of the ansatz) can flip, and neither derivation should be trusted blindly — **the correctness
gate is reciprocity ($S_{21}=S_{12}$) and passivity ($|S|\le1$), asserted in Module 8, not the
sign tracked through any single derivation.** $B_{p,ij}$ is manifestly symmetric in $(i,j)$ up
to the ordering of the two integral factors — a useful structural check independent of the
overall-sign question.

**Update, post-review: assemble the symmetric-by-construction form, not the boxed formula
literally.** The boxed $B_{p,ij}$ above is symmetric only via $\mathbf h_m=Y_m\,\hat
x\times\mathbf e_m$ (§4.1), which gives $\int_{S_p}(\mathbf W_j)_t\times\mathbf h_m\cdot\hat
x\,dS = Y_m\int_{S_p}\mathbf W_j\cdot\mathbf e_m\,dS$ *analytically* — that identity holds to
within the discrete field reconstruction's own error, not exactly, and implementation confirmed
a marginal mode can make the literally-assembled $B_{p,ij}$ asymmetric by up to ~130% relative,
which is large enough to trip Module 6 §5's complex-symmetric factorization check on an
otherwise-fine system. Substituting the identity directly into the boxed formula gives an
equal-when-the-identity-holds form that is symmetric *unconditionally*:

$$B_{p,ij} = -j\omega\mu_0\sum_m (Y_m)^2\left(\int_{S_p}\mathbf W_i\cdot\mathbf e_m\,dS\right)\left(\int_{S_p}\mathbf W_j\cdot\mathbf e_m\,dS\right)$$

Every summand is `(scalar)*outer(v,v)` for `v = ` the $\mathbf e_m$-overlap vector — symmetric
in the literal matrix regardless of mode quality, not just in principle. `ports.port_operator.build_B`
assembles this form; the un-substituted $\int_{S_p}(\mathbf W_j)_t\times\mathbf h_m\cdot\hat
x\,dS$ overlap (`overlap_h`, §5.2) is still computed and cached in case another consumer (e.g.
Module 7) needs the un-substituted quantity, but `build_B` itself no longer reads it.

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
4. Solve the generalized eigenproblem (§3.6) via a general (QZ-capable) solver; **apply the
   physical-bounds filter immediately** (discard $\beta$ outside $[k_0\sqrt{\varepsilon_{r,\min}},
   k_0\sqrt{\varepsilon_{r,\max}}]$ — confirmed required, not optional); verify real $\gamma^2$
   for the lossless case **before** trusting any subsequent result — this is the first
   checkpoint that would catch a block-arrangement error from §3.6.
5. Reconstruct $\mathbf h_t$ (§3.8); extract $Y_m$ (§4.1) and run its Poynting-integral
   consistency check; apply the power normalization (§4.2).
6. Validate the dominant mode's $\beta(\omega)$ against Hammerstad–Jensen/Wheeler — this is the
   analytic gate that most directly tests whether §3.6's arrangement, despite the honesty flag,
   produced physically correct results.
7. Implement the §4.3 projection **with explicit self-normalization against each mode's
   computed $N_m$** (not assumed equal to 1) and verify biorthogonality ($\int(\mathbf
   e_m\times\mathbf h_n)\cdot\hat x\,dS\approx\delta_{mn}$ after normalization, for the retained
   modes) numerically.
7a. Robustness improvement, unrelated to the above but worth building in at the same time:
   `solve()` should skip individual eigensolve candidates that turn out to be un-normalizable
   (§4.2's normalization fails, e.g. $\mathrm{Re}(Y_m)\approx0$) rather than letting the whole
   call crash — a candidate that can't reach $P_m=1$ isn't a valid captured mode regardless of
   what caused the failure.
8. Implement `ports.port_operator` (§5): $\mathbf B_p$, then $\mathbf g_p$ for a single-mode
   excitation, with the caching strategy (§5.2) built in from the start.
9. Implement de-embedding (§6) — simple, low-risk relative to the rest of this module.
10. Run the full §8 validation suite.

---

## 8. Validation targets

- **Real eigenvalues (lossless case)**: $\gamma^2$ real to numerical tolerance for real
  $\varepsilon_r$ — the primary, non-negotiable check on §3.6's arrangement, run before anything
  downstream is trusted. **Confirmed in implementation to ~$10^{-18}$ relative precision.**
- **Physical-bounds filter** (§3.6): confirmed in implementation to remove the spurious
  eigensolve solution cleanly, without discarding any legitimate mode.
- **Analytic gate**: dominant mode $\beta(\omega)$ vs. Hammerstad–Jensen/Wheeler, converging
  under mesh refinement (ties into the top-level doc's Phase 1 gate).
- **Biorthogonality**: $\int(\mathbf e_m\times\mathbf h_n)\cdot\hat x\,dS \approx \delta_{mn}$
  for the retained modes, at several frequencies — **verified after applying the §4.3
  self-normalization fix**; checking this before the fix is exactly what would have exposed the
  factor-of-2 gap, worth keeping as a standing regression test for that reason.
- **$Y_m$ consistency** (§4.1): the two independent power computations agree.
- **$S_{tz}=-T_{zt}^{T}$** (§3.5): exact algebraic identity, checked regardless of the §3.6
  arrangement question.
- **PEC edge correctness** (§2.3): confirm both the outer ground-plane edge and the internal
  trace segment are captured, on a hand-constructed small test mesh where the expected edge
  set is known in advance. **Confirmed on the real microstrip geometry.**
- **$\ge 2$ modes captured**, with the second mode's field pattern visually/numerically
  distinct from the dominant mode (not a numerical duplicate) — guards against a degenerate or
  under-resolved eigensolve silently returning the same mode twice. This applies when `n_modes`
  (the *required* minimum, post-review's single-mode-tolerant counting) is $\ge2$; a port whose
  aperture is deliberately sized for single-mode operation (`n_modes=1`) is not expected to
  produce a second mode at all, and `solve`'s `n_desired` oversupply returning fewer than
  requested in that case is the correct, non-error outcome, not a test failure to chase.
- **Open validation item, not yet resolvable in isolation**: §3.7's box-mode mode-selection
  limitation has no test that can pass reliably within this module alone, since discriminating
  the quasi-TEM mode from a near-degenerate box mode by any *per-frequency* criterion was shown
  not to generalize across mesh resolutions. The only currently-known reliable resolution
  (frequency-to-frequency mode tracking, §3.7) requires Module 6 to exist. Track this as a known
  gap rather than closing it with an unconvincing single-frequency test.
- **$B_p$ symmetry** (§5.1): structural check independent of the overall sign question. Now
  exact (to machine precision, any mode quality), not merely approximate, since `build_B`
  assembles the symmetric-by-construction form (§5.1's post-review update) rather than the
  boxed formula literally.
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
    def solve(port_tag: str, omega: float, n_modes: int = 2, n_desired: int = None) -> list[PortMode]
        # n_modes: required minimum (raises PortModeError if fewer found)
        # n_desired: desired count to try to collect, default n_modes (post-review,
        #            single-mode-tolerant mode counting -- §3.7's third mitigation)

class PortMode:
    gamma: complex             # propagation constant
    e_t, h_t: callable         # transverse field profiles, power-normalized (P_m = 1)
    Y: complex                 # modal admittance

# ports.port_operator
def build_B(port_modes: dict[str, list[PortMode]], mesh, omega) -> sparse matrix   # global-DOF sized, low-rank/local
def build_g(port_modes, excitation: dict[(str,int), complex], mesh, omega) -> vector
def deembed(S: array, port_modes, offsets: dict[str, float]) -> array

# ports.sizing (added post-review, §3.7's box-mode mitigation) -- informational only, never raises
def check_port_sizing(W_port, H_port, h_sub, w, eps_r_max, f_max) -> list[str]   # warning messages
def check_port_sizing_for_cross_section(cs: PortCrossSection, materials, f_max) -> list[str]
```

Module 6 calls `build_B` once per frequency (alongside the cached interior $\mathbf K,\mathbf
M$ from Module 3) and `build_g` once per excited (port, mode) pair per frequency. Module 7
consumes the same `PortMode` objects (their cached overlap integrals, §5.2) for extraction —
no independent re-derivation of the modal fields there.
