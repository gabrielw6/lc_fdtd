# Module 0 — `geometry.builder`: Parametric Geometry & Tagging — Equations & Implementation Plan

Companion to `architecture_fem_sparameter_modules.md` and `module1_mesh_interface_equations.md`.
Same conventions: no code, precise geometry, step-by-step build order, validation targets.

Module 0 sits **upstream of Module 1**. Module 1 (`mesh.interface`) assumes a tagged tet mesh
already exists and only adapts it. Module 0 is what produces that tagged mesh for this specific
problem: it takes a small set of user-supplied dimensions, builds the one fixed topology
described in the second design prompt (microstrip line + centered rectangular LC cutout in the
substrate, ground plane below, air above), assigns the exact tag vocabulary Module 1 consumes,
invokes the external mesher, and hands the result onward. It does not do general CAD — the
topology never changes, only the numbers.

---

## 0. The fixed topology (recap, with the layout pinned down)

**Update**: the LC cavity has a transverse width $W_{\text{lc}}$ strictly smaller than the
substrate width $W_{\text{sub}}$ — it does not span the full cross-section. This is a real,
user-facing parameter (§1.1), not derived. The cavity is therefore a rectangular box embedded
within the substrate: bounded in length (as before), *and now also bounded in width*, but still
spanning the **full substrate height** $h_{\text{sub}}$ (only the width changed; depth was not
mentioned as a new knob, so it remains a full-depth trench, touching both the ground-plane
interface and the substrate/air interface).

Longitudinal side view (length × height, at the transverse centerline) is **unchanged** from
the previous version — the centerline still passes through the LC channel regardless of its
width, so this slice looks identical to before:

```
 z
 ^
 |   ┌──────────────────────────────────────────────┐   PML  (h_pml)
 |   ├──────────────────────────────────────────────┤   Air  (h_air)
 |   ══════════●══════════════════════════●══════════   <- trace (0 thickness)
 |   ┌───────┬────────────────┬───────────┐               Substrate (h_sub)
 |   │ SUBS  │   LC cavity    │   SUBS    │                 (centerline slice)
 |   └───────┴────────────────┴───────────┘
 |   ══════════════════════════════════════════════   <- ground plane (0 thickness)
 +─────────────────────────────────────────────────→ x
 x=0                x_c0    x_c1                 x=L
 PORT_1                                          PORT_2
```

What changes is the **transverse cross-section** — a slice at constant $x$ within the cavity's
length range, looking along the length axis (width × height):

```
 z
 ^
 |   ┌──────────────────────────────────────────────┐   PML  (h_pml)
 |   ├──────────────────────────────────────────────┤   Air  (h_air)
 |   ══════════════●●●●●●●●══════════════════════════   <- trace (width w, centered)
 |   ┌───────────┬────────────────┬─────────────────┐    Substrate (h_sub)
 |   │ SUBS wing │   LC cavity    │   SUBS wing      │      (only within cavity's x-range)
 |   └───────────┴────────────────┴─────────────────┘
 |   ══════════════════════════════════════════════   <- ground plane (0 thickness)
 +─────────────────────────────────────────────────→ y
 y=0        y_lc0         y_lc1               y=W_sub
```

Outside the cavity's length range ($x<x_{c0}$ or $x>x_{c1}$), this transverse slice is simply
uniform substrate across the full width — the "wings" only exist as a distinct region *within*
the cavity's length interval; elsewhere they merge into the ordinary full-width substrate block.

Five things about this layout are not fully fixed by your stated knob list and need an
explicit, justified choice before the tagging scheme can be written down:

1. **Cavity width is now a primary parameter** $W_{\text{lc}}$, with $0 < W_{\text{lc}} <
   W_{\text{sub}}$. Cavity **height still equals the full substrate height** $h_{\text{sub}}$
   (depth was not mentioned as changing, so the full-depth-trench reading from before stands
   for the vertical direction — only the width reading changed).
2. **Cavity placement**: centered both **transversely** ($y_{lc0}=(W_{\text{sub}}-W_{\text{lc}})/2$)
   and **longitudinally** ($x_{c0}=(L-L_{\text{lc}})/2$). Neither offset was requested, so
   both default to centered, matching the line's own centered placement.
3. **Domain truncation (open boundary handling) — this concern from the previous version is
   now resolved rather than merely worked around.** With $W_{\text{lc}} < W_{\text{sub}}$, the
   LC region has genuine substrate margin on both sides at every longitudinal position within
   the cavity range — it can never be adjacent to the lateral domain boundary regardless of
   what that boundary is. The earlier decision (PML truncates only the top; the lateral faces
   $y=0,W_{\text{sub}}$ are left as a natural/PMC boundary) still stands, but it is no longer
   load-bearing for correctness the way it was before: even if lateral PML were added later,
   the wing of `SUBSTRATE` between the LC box and the side wall keeps `PMLMaterial`'s isotropic
   background requirement satisfied automatically. This is a genuine robustness improvement
   from the correction, worth noting explicitly rather than treating as incidental.
4. **Trace width vs. cavity width**: the trace ($w$) should sit within the footprint of the LC
   cavity ($w \le W_{\text{lc}}$) for the LC to actually be under the line it's meant to tune —
   otherwise part of the trace's field sits over plain substrate with no tunability. This is
   now asserted as an explicit parameter check (§1.3), not merely a comment.
5. **Conductor and ground-plane thickness**: both modeled as **zero-thickness PEC sheets**
   (a boundary condition, not a meshed volume) — unchanged from before.

Two more quantities are needed but are **not user-facing knobs** — internal parameters with
defaults, exposed in the parameter object for later tuning but not part of the primary
interface:

- **Air height $h_{\text{air}}$** — how far above the trace the air region extends before PML
  begins. Needs to be large enough that the fringing/quasi-static field above the line has
  mostly decayed. No physics claim is hard-coded here; treat as a config default
  (e.g. a small multiple of $h_{\text{sub}}$) to be tuned once Phase 1 validation is running,
  not a value asserted as correct in the abstract.
- **PML thickness $h_{\text{pml}}$** — set jointly with the grading profile in Module 5
  (§5.2 of the top-level doc), which needs a thickness and a target reflection $R_0$. Not
  decided here; this module only reserves the volume slab for it.

---

## 1. Parameters

### 1.1 Primary — user-facing

| Symbol | Meaning |
|---|---|
| $w$ | Line (trace) width |
| $L$ | Total length (= substrate length = domain length in $x$) |
| $L_{\text{lc}}$ | LC cavity length, $0 < L_{\text{lc}} < L$ |
| $W_{\text{lc}}$ | LC cavity width (transverse), $0 < W_{\text{lc}} < W_{\text{sub}}$ |
| $h_{\text{sub}}$ | Substrate height |
| $W_{\text{sub}}$ | Substrate width |
| $\varepsilon_{r,\text{sub}}$ (and optionally $\tan\delta_{\text{sub}}$) | Substrate material parameters |

Cavity **height** is not a separate knob — it is fixed at the full substrate height
$h_{\text{sub}}$ (§0, point 1).

### 1.2 Secondary — internal, defaulted, exposed for later tuning

| Symbol | Meaning | Default rule (placeholder, tune empirically) |
|---|---|---|
| $h_{\text{air}}$ | Air region height | e.g. a small multiple of $h_{\text{sub}}$ |
| $h_{\text{pml}}$ | PML slab thickness | set with Module 5's grading formula |
| (cavity offset) | Longitudinal placement | fixed at centered; not exposed yet |

### 1.3 Derived quantities

$$x_{c0} = \frac{L - L_{\text{lc}}}{2}, \qquad x_{c1} = \frac{L + L_{\text{lc}}}{2}$$

$$y_{lc0} = \frac{W_{\text{sub}} - W_{\text{lc}}}{2}, \qquad y_{lc1} = \frac{W_{\text{sub}} + W_{\text{lc}}}{2}
\qquad \text{(transverse cavity bounds, centered)}$$

$$y_0^{\text{trace}} = \frac{W_{\text{sub}} - w}{2}, \qquad y_1^{\text{trace}} = \frac{W_{\text{sub}} + w}{2}$$

$$y_0^{\text{port}} = \frac{W_{\text{sub}} - W_{\text{port}}}{2}, \qquad y_1^{\text{port}} = \frac{W_{\text{sub}} + W_{\text{port}}}{2}
\qquad \text{(transverse port-aperture bounds, centered — §1.4)}$$

$$z_{\text{gnd}} = 0,\quad z_{\text{iface}} = h_{\text{sub}},\quad z_{\text{air,top}} = h_{\text{sub}}+h_{\text{air}},\quad z_{\text{pml,top}} = h_{\text{sub}}+h_{\text{air}}+h_{\text{pml}}$$

**Parameter validation (run before any CAD call)**: $w>0$; $0 < L_{\text{lc}} < L$ (strict, so a
nonzero substrate margin exists at both ports — see §6); $0 < W_{\text{lc}} < W_{\text{sub}}$
(strict, so both substrate wings have nonzero width); $w \le W_{\text{lc}}$ (the trace should
sit within the cavity's footprint — see §0 point 4; treat as a hard check, not a silent
allowance, since a violation means the LC isn't actually under the full line and the intended
tuning mechanism is only partially engaged); $h_{\text{sub}}, h_{\text{air}}, h_{\text{pml}} > 0$;
$w \le W_{\text{port}} \le W_{\text{sub}}$ and $h_{\text{sub}} < H_{\text{port}} \le
z_{\text{air,top}}$ (§1.4's aperture must contain the trace horizontally and extend above the
substrate/air interface, but never exceed the domain); $\varepsilon_{r,\text{sub}} \ge 1$. Fail
here, before touching the CAD kernel — a bad parameter should never reach the mesher.

### 1.4 Port aperture — decoupled from the domain cross-section

**Added post-implementation, per the Module 0/4 port-aperture-decoupling review.** Originally
the port face was welded to the *entire* domain cross-section at $x=0$/$x=L$ — Module 1's
`PORT_p` tag covered $y\in[0,W_{\text{sub}}]$, $z\in[0,z_{\text{air,top}}]$ in full, forcing
Module 4's 2D port eigenproblem onto an oversized, PMC-walled box that can support its own
near-degenerate "box modes" close to the quasi-TEM $\beta$ (Module 4 §3.7's documented
mode-selection limitation). Standard wave-port practice sizes the aperture below $\lambda/2$ to
push those box modes' cutoff above the band of interest.

$$W_{\text{port}},\ H_{\text{port}} \qquad \text{(port aperture width, height — both optional,
default None} \Rightarrow \text{full cross-section)}$$

Both `None` (the default) means $W_{\text{port}}=W_{\text{sub}}$, $H_{\text{port}}=z_{\text{air,top}}$
— bit-for-bit the pre-existing full-cross-section behavior, for backward compatibility. Must be
supplied together, never just one. §1.3 above already includes the resulting
$y_0^{\text{port}}, y_1^{\text{port}}$ derived bounds and §1.3's validation already includes the
aperture's hard bounds ($w\le W_{\text{port}}\le W_{\text{sub}}$,
$h_{\text{sub}}<H_{\text{port}}\le z_{\text{air,top}}$).

**Design**: the aperture is a rectangle centered on the trace horizontally, bottom on the
ground plane — $y\in[y_0^{\text{port}}, y_1^{\text{port}}]$, $z\in[0, H_{\text{port}}]$. The
end-plane region *outside* the aperture (when the aperture is strictly smaller than the full
cross-section) is tagged `PORT_CAP` — a PEC "cap" idealization (§4.3/4.4) — rather than left as
part of `PORT_p`. This is what raises the aperture's own box-mode cutoff, and it makes the
aperture's own side/top walls PEC in Module 4's 2D eigenproblem *for free*, via Module 1's
existing "a 2D edge is PEC iff it is an edge of some `PEC`-tagged 3D face" rule (Module 4 §2.3)
— no new Module 4 code needed. Fringing fields are assumed contained inside the aperture by the
lower-bound sizing rule (`ports.sizing.check_port_sizing`, informational only, never a hard
gate), so the cap sees negligible field — the standard accepted idealization for a sized wave
port.

**Implementation note, not assumed by the design above**: the 2D port-mode eigenproblem's
numerical conditioning depends on how many mesh triangles actually resolve the aperture — a
very small aperture at a coarse mesh density can leave Module 4 unable to find enough
well-conditioned modes (`PortModeError`, not a `ports.sizing` warning, since it's a distinct
resolution issue, not a box-mode-margin one). Sizing the aperture per `ports.sizing`'s rules of
thumb is necessary but not sufficient; the mesh must also resolve it.

---

## 2. Construction strategy: adjacent blocks, not boolean cut

**Decision, updated for the narrower cavity**: build the substrate/LC composite as **five
adjacent bricks**, not three. The pre- and post-cavity length intervals are still simple
full-width bricks; the cavity's own length interval is now split **transversely** into three
side-by-side bricks (left wing, LC, right wing), since the cavity no longer occupies the full
width there:

| Brick | $x$ range | $y$ range | $z$ range | Tag |
|---|---|---|---|---|
| Pre-cavity | $[0, x_{c0}]$ | $[0, W_{\text{sub}}]$ | $[0, h_{\text{sub}}]$ | `SUBSTRATE` |
| Left wing | $[x_{c0}, x_{c1}]$ | $[0, y_{lc0}]$ | $[0, h_{\text{sub}}]$ | `SUBSTRATE` |
| LC | $[x_{c0}, x_{c1}]$ | $[y_{lc0}, y_{lc1}]$ | $[0, h_{\text{sub}}]$ | `LC` |
| Right wing | $[x_{c0}, x_{c1}]$ | $[y_{lc1}, W_{\text{sub}}]$ | $[0, h_{\text{sub}}]$ | `SUBSTRATE` |
| Post-cavity | $[x_{c1}, L]$ | $[0, W_{\text{sub}}]$ | $[0, h_{\text{sub}}]$ | `SUBSTRATE` |

All five occupy the full substrate height $h_{\text{sub}}$ — only the pre/post bricks span the
full width, while the cavity's length interval is subdivided into the wing/LC/wing triple. In
plan view this is exactly "a rectangular cutout of the solid substrate" as originally
described: a single rectangular window ($L_{\text{lc}} \times W_{\text{lc}}$, centered) removed
from an otherwise uniform $L \times W_{\text{sub}}$ substrate slab. The five-brick decomposition
is simply the conformal tiling of that shape — a standard "domain with a rectangular window"
tiling — not a new geometric idea, just the natural generalization of the three-brick case once
the window has finite width.

Everything below this point (the fragment-based construction philosophy, the trace-embedding
strategy) still applies to this five-brick decomposition unchanged — the *reasoning* for why
one fragment operation across all pieces is preferred over a boolean cut is exactly the same,
now spanning five bricks instead of three.

**Why this matters**: a boolean subtraction produces a new solid whose boundary at the cut is
generated by the CAD kernel's tolerance-dependent intersection routine — a source of sliver
faces, near-duplicate vertices, and non-conformal meshing at exactly the interface that matters
most (the LC/substrate boundary, where the tensor discontinuity lives). Building three bricks
that already share coincident faces, then applying a single **fragment** operation (OpenCASCADE:
`BOPAlgo_Builder` / Gmsh `BooleanFragments`) across all of them, forces the mesher to place
matching nodes on that shared interface by construction — no cut, no tolerance risk, guaranteed
conformity. This is the same conformity requirement Module 1 states for the LC boundary
(architecture doc §2.3): the mesh must conform to *geometric* boundaries, and this is how that
conformity is obtained rather than hoped for.

The same reasoning applies to every subsequent interface: fragment the Air brick against the
composite's top face; fragment the PML brick against Air's top face; and fragment the trace
footprint as an **embedded partitioning rectangle** on the substrate/air interface plane (§3)
rather than tagging it via a post-mesh geometric query on triangle centroids, which risks
including or excluding boundary-straddling triangles at the trace's edge to within the mesher's
own tolerance.

---

## 3. Step-by-step build procedure

1. **Validate parameters** (§1.3).
2. **Compute derived quantities** (§1.3): $x_{c0},x_{c1}$, trace $y$-bounds, all $z$-levels.
3. **Build the five substrate/LC-composite bricks** (§2 table): pre-cavity and post-cavity
   (full width, $z\in[0,h_{\text{sub}}]$), and — within the cavity's length interval
   $x\in[x_{c0},x_{c1}]$ — the left wing ($y\in[0,y_{lc0}]$), the LC box
   ($y\in[y_{lc0},y_{lc1}]$), and the right wing ($y\in[y_{lc1},W_{\text{sub}}]$), all sharing
   the same $z\in[0,h_{\text{sub}}]$.
4. **Embed the trace-footprint partitioning rectangle** on the plane $z=h_{\text{sub}}$: a 2D
   rectangle $x\in[0,L],\ y\in[y_0^{\text{trace}},y_1^{\text{trace}}]$, added as an internal
   partitioning face so the mesher produces a distinct, exactly-bounded set of triangles
   covering the trace footprint, taggable directly (rather than a post-hoc query).
4a. **When the port aperture (§1.4) is strictly smaller than the full cross-section, embed one
   partitioning rectangle on each end plane** ($x=0$ and $x=L$), spanning
   $y\in[y_0^{\text{port}},y_1^{\text{port}}]$, $z\in[0,H_{\text{port}}]$ — the same
   embed-then-fragment pattern as step 4, just on a vertical plane instead of a horizontal one
   (built from raw points/lines/a curve loop rather than the CAD kernel's axis-aligned
   rectangle primitive, which only builds in a $z={\rm const}$ plane). Skipped entirely when the
   aperture equals the full cross-section (the default) — zero new geometry, bit-for-bit the
   pre-existing behavior.
5. **Fragment** the five bricks of step 3, together with every rectangle embedded in steps 4/4a,
   into one connected solid with coincident internal faces at $x=x_{c0}$, $x=x_{c1}$
   (longitudinal interfaces), $y=y_{lc0}$, $y=y_{lc1}$ **within the cavity's length interval
   only** (the new transverse interfaces introduced by the wings), and — when step 4a ran — the
   port-aperture outline on each end plane. A single fragment call across all pieces and all
   embedded rectangles handles every interface at once. The embedded end-plane rectangles'
   descendants (a fragment `out_map` entry, exactly like the trace rectangle's) give the
   aperture's own face(s) *exactly*, by construction — not a post-hoc centroid/bounding-box
   query with tolerance risk at the aperture boundary, consistent with this section's own
   embed-not-query philosophy.
6. **Build the Air brick**: $x\in[0,L]$, $y\in[0,W_{\text{sub}}]$, $z\in[h_{\text{sub}}, z_{\text{air,top}}]$.
   Fragment against the top face of step 5's composite (this is also where the trace-footprint
   partition from step 4 propagates upward if the trace needs representation on both sides of
   the interface — in the zero-thickness model it does not; the interface face itself carries
   the tag, shared by both volumes).
7. **Build the PML brick**: $x\in[0,L]$, $y\in[0,W_{\text{sub}}]$, $z\in[z_{\text{air,top}}, z_{\text{pml,top}}]$.
   Fragment against the top face of the Air brick.
8. **Assign volume tags** (§4.1).
9. **Assign surface tags** (§4.2), including the two tags introduced by this module beyond
   Module 1's original list (§4.3).
10. **Run the pre-mesh consistency checks** (§6) on the tagged solid model, before invoking the
    mesher — cheap, and catches a wrong fragment operation before spending time meshing.
11. **Invoke the external mesher** on the finalized, tagged solid model.
12. **Emit the auto-generated material-spec stub** for `SUBSTRATE` and `AIR` (§5) alongside the
    mesh, so a single set of user inputs produces both.
13. Hand the resulting mesh (vertices, tets, `volume_tags`, `surface_tags`) to
    `mesh.interface` (Module 1). Module 0's responsibility ends here.

---

## 4. Tag vocabulary

### 4.1 Volume tags

| Tag | Assigned to |
|---|---|
| `SUBSTRATE` | Pre-cavity, post-cavity, left wing, and right wing bricks (all four, same tag) |
| `LC` | The central LC brick only (now width-bounded, not full-width) |
| `AIR` | Air brick |
| `PML_TOP` | PML brick |

### 4.2 Surface tags — matching Module 1's original list

| Tag | Assigned to |
|---|---|
| `PEC_GROUND` | Bottom face, $z=0$, full $x,y$ extent |
| `PEC_LINE` | The embedded trace-footprint patch at $z=h_{\text{sub}}$ |
| `PORT_1` | Face at $x=0$, restricted to the port **aperture** $y\in[y_0^{\text{port}},y_1^{\text{port}}]$, $z\in[0,H_{\text{port}}]$ (§1.4) — equals the full substrate+air cross-section when the aperture is left at its default |
| `PORT_2` | Face at $x=L$, same restriction |

### 4.3 Surface tags — new, introduced by this module's truncation choices

These extend Module 1's tag set and should be added to its `boundary_faces(tag)` contract:

| Tag | Assigned to | Meaning |
|---|---|---|
| `PML_OUTER_PEC` | All exterior faces of the `PML_TOP` brick *except* its bottom (shared, interior, not a boundary) — top cap, two lateral sides, two end caps | The PEC backing that terminates the PML shell (top-level doc §5: "PEC-backed outer wall"). The PML block's own end-caps and sides are exterior too, since PML here is a thin shell only in $z$; they get the same PEC treatment as the top. |
| `PMC_SIDE` | Lateral faces $y=0$ and $y=W_{\text{sub}}$, for $z\in[0,z_{\text{air,top}}]$ (substrate + air portion only — the PML shell's own sides are already covered by `PML_OUTER_PEC`) | The natural (no boundary term) truncation from §0 point 3. **Explicitly tagged, not left untagged** — see §6. With the narrower cavity, these faces are now backed by `SUBSTRATE` (the wings) at every longitudinal position, including within the cavity's length range — never directly by `LC`. This wasn't guaranteed in the full-width-cavity version and is one of the robustness benefits of the correction (§0 point 3). |
| `PORT_CAP` | The end-plane region at $x=0$/$x=L$ **outside** the port aperture (§1.4) — empty (no physical group emitted) whenever the aperture equals the full cross-section | The PEC "cap" idealization step 4a's embedded rectangle makes taggable. Folded into Module 1's combined `PEC` set (`PEC_GROUND` ∪ `PEC_LINE` ∪ `PORT_CAP`), **not** kept as its own bucket — this is what makes the aperture's own side/top walls PEC in Module 4's 2D port eigenproblem automatically, via Module 1's existing "a 2D edge is PEC iff it borders a PEC-tagged 3D face" rule (Module 4 §2.3), with zero new Module 4 code. |

**Why `PMC_SIDE` (and, by the same argument, `PORT_CAP`) must be an explicit tag, not silence**:
Module 1's boundary coverage check (its §5.3) treats an *untagged* boundary face as a likely bug
— "the model has an open surface the solver would treat as a perfect magnetic wall by default —
almost always unintended." Here it *is* intended. Tagging it explicitly keeps that check
meaningful: every boundary face still resolves to exactly one tag, and the tag documents that
the boundary condition there is a deliberate modeling choice, not an omission. `PMC_SIDE` and
`PORT_CAP` are the two additions Module 1 needs to make to its tag vocabulary to accommodate
this module's output (`PORT_CAP` folded into the existing `PEC` aggregate, not a new bucket of
its own — see Module 1 §5.3).

---

## 5. Auto-generated material-spec stub

Since the user supplies $\varepsilon_{r,\text{sub}}$ (and optionally $\tan\delta$) as part of
the *same* parameter set used to build the geometry, Module 0 emits the corresponding entries
directly, so one set of inputs produces both the tagged mesh and its material assignment for
the non-LC tags:

```yaml
materials:
  AIR:
    type: constant
    eps_r: 1.0
  SUBSTRATE:
    type: constant
    eps_r: <eps_r_substrate>
    # tan_delta: <tan_delta_substrate>   # if supplied
```

`LC` is deliberately **not** emitted here — its entry (director-file reference,
$\varepsilon_\perp,\varepsilon_\parallel$) belongs to `material.spec` proper and is supplied
separately once the director-field module (Module 2.3) is wired in, per the earlier discussion
of the material specification format. `PML_TOP`'s material is never user- or spec-supplied at
all — it is derived automatically by `pml.PMLMaterial` wrapping whatever background occupies
that shell (here, always `AIR`, since the PML sits entirely above the air region).

---

## 6. Consistency checks (validation targets)

- **Parameter validation** (§1.3): reject invalid dimensions before any CAD call.
- **Cavity margin**: $x_{c0} > 0$ and $x_{c1} < L$ strictly (guaranteed by $L_{\text{lc}}<L$
  together with centered placement) — this is what keeps both port faces in the isotropic
  feed section required by the top-level architecture doc's port design invariant. Assert it
  explicitly rather than relying on the arithmetic to happen to work out.
- **Bounding-volume check** (pre-mesh, CAD-level), updated for the five-brick decomposition:

  $$V_{\text{LC}} = L_{\text{lc}}\cdot W_{\text{lc}}\cdot h_{\text{sub}}$$

  $$V_{\text{SUBSTRATE}} = L\cdot W_{\text{sub}}\cdot h_{\text{sub}} - V_{\text{LC}}$$

  i.e. the substrate-layer footprint ($L\times W_{\text{sub}}$) minus exactly the LC window —
  the four substrate bricks (pre, post, left wing, right wing) must sum to this value, not to
  some other partition. Together with $V_{\text{AIR}} = L\cdot W_{\text{sub}}\cdot h_{\text{air}}$
  and $V_{\text{PML}} = L\cdot W_{\text{sub}}\cdot h_{\text{pml}}$, the full check is:

  $$V_{\text{SUBSTRATE}} + V_{\text{LC}} + V_{\text{AIR}} + V_{\text{PML}} \stackrel{!}{=} L\cdot W_{\text{sub}}\cdot(h_{\text{sub}}+h_{\text{air}}+h_{\text{pml}})$$

  which holds identically given the two boxed equations above — the useful runtime check is
  therefore **not** this trivial identity itself, but confirming the *constructed* solid's
  volumes (summed per tag, post-fragment) match $V_{\text{SUBSTRATE}}$ and $V_{\text{LC}}$
  individually. This is what actually catches a wrong fragment operation (a gap or overlap
  between the wing bricks and the LC brick, which the trivial total-volume identity would not
  detect if the error is a compensating gap/overlap pair). This is the CAD-level analogue of
  Module 1's post-mesh "sum of tet volumes per tag" check (§8 there).
- **Tag coverage** (pre-mesh): every exterior face of the fragmented solid resolves to exactly
  one of `PEC_GROUND`, `PEC_LINE`, `PORT_1`, `PORT_2`, `PML_OUTER_PEC`, `PMC_SIDE`, `PORT_CAP`.
  No exterior face may be left untagged — this is what makes Module 1's downstream coverage
  check pass meaningfully rather than trivially. `PORT_CAP` is simply empty (no physical group)
  whenever the aperture is left at its default full-cross-section value.
- **Port-aperture containment** (§1.4, new): the embedded end-plane rectangles give the aperture
  face(s) *exactly* (step 4a/5), so this is checked by construction rather than as a separate
  post-mesh assertion — unlike the trace footprint (embedded on an *interior* plane, so its
  containment within the substrate/air interface footprint still needs the check below), the
  aperture rectangle is embedded directly on the *exterior* boundary it partitions, and its
  fragment descendants are used as the `PORT_p` face set with no further geometric query.
- **Trace footprint containment**: the embedded trace rectangle lies strictly within the
  substrate/air interface footprint, i.e. $0 < y_0^{\text{trace}} < y_1^{\text{trace}} <
  W_{\text{sub}}$ — a direct consequence of the $W_{\text{sub}}>w$ parameter check, re-asserted
  here at the geometry level since it is what guarantees `PEC_LINE` and `PMC_SIDE` never touch.
- **Trace-within-cavity containment** (new): $y_{lc0} \le y_0^{\text{trace}}$ and
  $y_1^{\text{trace}} \le y_{lc1}$ — the trace footprint lies within the LC footprint at every
  point along the cavity's length, the geometric-level consequence of the $w\le W_{\text{lc}}$
  parameter check (§1.3). Assert both bounds, not just the width inequality that implies them,
  since a centering bug could satisfy $w\le W_{\text{lc}}$ while still placing the trace
  off-center relative to the cavity.
- **Port-aperture/trace containment** (§1.4, new): $y_0^{\text{port}} \le y_0^{\text{trace}}$ and
  $y_1^{\text{trace}} \le y_1^{\text{port}}$ — same defensive-assertion style as the
  trace-within-cavity check just above (both bounds, not just $w\le W_{\text{port}}$, even
  though centering makes the width inequality sufficient here too).
- **Reduction tie-in** (solver-level, not this module's to run, but worth stating as the target
  this geometry construction is built to support): with the `LC` volume assigned the *same*
  material as `SUBSTRATE` (a trivial material-spec substitution, no geometry change), the
  built structure must reproduce the Phase-1 uniform-line result from the top-level validation
  plan. This is the geometry-side half of that reduction check — Module 0 guarantees the
  *shape* is right for it; Modules 2–8 exercise the material substitution and check the result.

---

## 7. Interface / module contract

```
GeometryParams:
    w, L, L_lc, W_lc, h_sub, W_sub     # primary, user-facing (W_lc < W_sub, w <= W_lc)
    eps_r_substrate, tan_delta_substrate = 0.0
    h_air = <default>, h_pml = <default>          # secondary, defaulted
    W_port = None, H_port = None       # port aperture (§1.4); None,None -> full cross-section

class GeometryBuilder:
    def build(params: GeometryParams) -> (mesh_handle, material_spec_stub)
        # 1-11: construct, tag, mesh (via external mesher)
        # 12: emit SUBSTRATE/AIR material_spec_stub
        # returns what Module 1 (mesh.interface) and material.spec consume directly
```

Everything downstream — Module 1's `MeshInterface`, Module 2's `material.spec` loader — is
unaware this module exists; it only ever sees a tagged mesh and a material-spec fragment that
happen to already be self-consistent, because Module 0 built them together from one parameter
set.
