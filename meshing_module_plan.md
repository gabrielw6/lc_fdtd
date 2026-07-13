# Meshing Module — Architecture Definition

Scope: a new sub-package, `meshing/`, that turns a cavity + sample geometry specification
into a tagged, conformal tetrahedral mesh ready for the FEM assembly step (Nédélec/scikit-fem,
per the earlier FEM discussion). No code — contracts, file breakdown, and an independent test
plan only.

This module has no dependency on `perturbation.py`, `inverse.py`, or any EM physics at all —
it only knows about geometry. It reads two existing modules' *public attributes* (Module 1's
`CavityMode` subclasses, Module 3's `SampleRegion` subclasses) for the standard-shape path,
but adds no new methods to either, and its own correctness is verified without reference to
either module's physics — see Section 6.

---

## 0. Design decisions up front

### 0.1 Standard shapes and custom (STEP) shapes are the same pipeline, different entry points

Every step downstream of "produce an OCC solid" — interference check, fragments, tagging,
mesh generation — is identical regardless of whether the solid came from a `CavityMode`'s
dimensions or an imported `.step` file. The module is structured around one shared pipeline
with two geometry *sources* feeding into it, not two parallel pipelines.

### 0.2 Materials never enter this module

Meshing depends only on geometry (shape, position) — not on $\epsilon_r,\mu_r$, loss tangent,
or `Material` at all. This mirrors the existing project-wide separation between geometry and
material (`SampleRegion` vs. `Material` in Module 3) and keeps this module cacheable
independently of anything a Module 5 fit varies.

### 0.3 Interference checking uses Gmsh's own geometry kernel, not Module 3's

The containment check (Section 3) is computed from OCC boolean operations and mass
properties — never by calling `SampleRegion.contains()` or `.volume()`. Using Module 3's own
methods to validate geometry that Module 3 also produced would not be an independent check;
it would just be testing that a copy of the same formula agrees with itself. The two must be
computed by genuinely different code paths for the check to mean anything (Section 6.1
elaborates why this matters for the independence requirement specifically).

### 0.4 Custom-sample position is an explicit rigid transform, not implicit

A `.step` sample file is authored in its own local frame. Module 3's standard `SampleRegion`
subclasses already encode position directly in the cavity's frame (`center`, `axis`/`normal`
attributes) — no separate position input needed there. A custom sample has no such built-in
frame relationship, so its position relative to the cavity must be a required, explicit
`RigidTransform` (translation + rotation), applied before anything else touches the shape.

### 0.5 Unit handling for imported STEP files is a named, explicit parameter

STEP files are very commonly authored in millimeters; this project's convention (`CLAUDE.md`)
is SI units throughout. Silently assuming a STEP file's units match the project's would be a
predictable source of a 1000× geometry error. Every STEP import call takes an explicit
`length_unit` (e.g. `'mm'`, `'m'`) and converts internally — never inferred, never defaulted
to something the caller might not expect.

**Implementation finding (verified empirically against Gmsh 4.15.2's OCC STEP translator, not
assumed from documentation): this 1000x error is not hypothetical, it is Gmsh's own default
behavior, and it happens regardless of what a file's header declares.** `Gmsh.option`'s
`Geometry.OCCTargetUnit` applies a *real* geometric rescale at import time (confirmed via
`getBoundingBox`, not a derived quantity like mass) — left at its own un-overridden default,
raw coordinate values are silently multiplied by 1000 on import, and this happens identically
whether the source file's own `SI_UNIT` entity declares `METRE` or `MILLIMETRE` (verified: two
otherwise-identical fixture files differing only in that declaration import to bit-identical
results). Setting `Geometry.OCCTargetUnit = "M"` makes Gmsh treat the file's raw numbers as an
exact, unscaled passthrough instead. `step_import.py` must always set this before calling
`importShapes`, then apply its own explicit scale from the caller's `length_unit` on top —
otherwise the caller's stated unit is silently compounded with Gmsh's own hidden default
rather than being the sole source of truth Section 0.5 asks for.

### 0.6 Mesh resolution is specified physically, not geometrically

A raw "characteristic length" number means nothing without reference to the wavelength at the
frequency being solved for. Resolution is specified as `target_elements_per_wavelength`
(default 10, conservative for first-order Nédélec elements — see Section 4), converted
internally to an absolute length using $\lambda=c/(f\sqrt{\epsilon_{bg}\mu_{bg}})$. For the
standard-shape path, $f,\epsilon_{bg},\mu_{bg}$ come directly from the `CavityMode`'s own
`f0`, `eps`, `mu` attributes; for a fully custom (STEP-imported) cavity, these have no source
object to read from and must be supplied explicitly as `reference_frequency`,
`background_eps`, `background_mu` (defaulting to vacuum).

---

## 1. Package layout

```
src/cavity_perturbation/meshing/
    __init__.py           # public re-exports only: build_mesh(), MeshResult, exceptions
    geometry_spec.py       # input dataclasses: StandardCavityInput, StepCavityInput,
                           # StandardSampleInput, StepSampleInput, RigidTransform
    standard_shapes.py     # CavityMode / SampleRegion -> OCC primitive, via dispatch table
    step_import.py         # STEP file import + unit conversion
    transforms.py          # RigidTransform application to OCC shapes; construction helpers
    interference.py        # containment check via OCC boolean intersection + mass properties
    tagging.py              # boolean fragments + physical group assignment
    mesh_sizing.py           # wavelength -> characteristic length conversion
    mesh_generation.py       # gmsh mesh generation call, mesh-quality query
    mesh_io.py                # export/round-trip via meshio; MeshResult, MeshStats
    cache.py                  # content-addressed mesh cache (Section 5)
    pipeline.py                # build_mesh(): orchestrates 0.1's shared pipeline
tests/test_meshing/
    test_standard_shapes.py
    test_step_import.py
    test_transforms.py
    test_interference.py
    test_tagging.py
    test_mesh_sizing.py
    test_mesh_generation.py
    test_mesh_io.py
    test_cache.py
    test_pipeline_integration.py
    fixtures/                 # small, hand-authored .step files for import tests
```

Each file above has exactly one responsibility, in line with the atomization requirement —
`pipeline.py` is the only file that calls into more than one of the others.

---

## 2. Data contracts

### 2.1 Geometry inputs

```
StandardCavityInput:  cavity_mode: CavityMode         # Module 1 instance, read-only
StepCavityInput:      path: Path, length_unit: str

StandardSampleInput:  region: SampleRegion            # Module 3 instance, read-only
StepSampleInput:      path: Path, length_unit: str, transform: RigidTransform
```
`CavityGeometryInput = StandardCavityInput | StepCavityInput` (and similarly for sample) —
dispatched by type, not by a boolean flag, at the single point in `pipeline.py` that needs to
know which source it's dealing with.

### 2.2 `RigidTransform`

```
translation: tuple[float, float, float]
rotation: tuple[tuple[float, float, float], ...]   # 3x3 orthogonal matrix, as nested tuples
```
Stored as tuples, not `numpy` arrays, specifically so the whole input spec is hashable for
the cache (Section 5) — convert to array only inside the functions that actually call Gmsh.
Construction helpers (`RigidTransform.identity()`, `.from_axis_angle(axis, angle)`,
`.translation_only(dx, dy, dz)`) live in `transforms.py`, not on the dataclass itself, keeping
the dataclass a plain, hashable data container.

### 2.3 Output

```
MeshResult:
    mesh: meshio.Mesh              # or mesh_path: Path, whichever the FEM assembly step prefers
    sample_physical_tag: int
    background_physical_tag: int
    boundary_physical_tag: int
    cavity_volume: float           # from OCC mass properties, independent of Module 1/3
    sample_volume: float           # from OCC mass properties, independent of Module 1/3
    mesh_stats: MeshStats
```
```
MeshStats:
    n_elements: int
    n_vertices: int
    min_element_quality: float     # Gmsh's own quality metric, e.g. gamma or SICN
    max_aspect_ratio: float
```
`cavity_volume`/`sample_volume` are included in the result deliberately — they're cheap
to compute (already needed for the interference check) and give any downstream caller (FEM
assembly, or a future GUI) an independent number to sanity-check against, without re-deriving it.

### 2.4 Exceptions

```
SampleExceedsCavityError(overlap_deficit: float)   # sample not fully contained (Section 3)
DegenerateMeshError(reason: str)                    # zero-volume / non-manifold result
StepUnitAmbiguityError                              # raised if length_unit is not supplied
```
Specific, structured exception types — not a generic `ValueError` — so a caller (including a
future GUI) can distinguish "your sample doesn't fit" from "the mesh generator produced
garbage" from "you forgot to specify units," and react to each differently.

---

## 3. Interference check

**Criterion**: let $V_{\text{cav}}$ be the cavity solid (interior region) and $V_s$ the
positioned sample solid. Compute, via OCC's own boolean intersection and mass-property query
(not Module 3's `contains()`/`volume()`):
$$\Delta = \mathrm{Volume}(V_s) - \mathrm{Volume}(V_{\text{cav}}\cap V_s)$$

$\Delta=0$ (to numerical tolerance $\tau$, suggested $\tau=10^{-6}\times\mathrm{Volume}(V_s)$,
scaled to the sample rather than a fixed absolute number since sample sizes span orders of
magnitude across this project's use cases) if and only if $V_s\subseteq V_{\text{cav}}$ —
this single volume comparison is set-theoretically sufficient on its own; no separate
boundary-crossing check is needed; $V_s\cap V_{\text{cav}}=V_s$ already implies full
containment.

A sample exactly flush against the cavity wall (zero-volume surface contact, no interior
overlap deficit) passes this check — that's a physically valid placement (a boundary-value
problem at a PEC wall is well-posed), not an error condition, so no special-casing for exact
boundary contact is needed; the volume criterion handles it correctly on its own.

Run this check **before** the boolean-fragments/tagging step (0.1's shared pipeline) and
raise `SampleExceedsCavityError(overlap_deficit=Δ)` immediately on failure — don't proceed to
a meshing step that would silently produce a geometrically invalid result.

**Correction (found via `test_pipeline_integration.py`, not any single module's own test in
isolation): the boolean intersection for this query must be computed on *copies* of
$V_{\text{cav}}$/$V_s$, never on the originals directly — even with `removeObject=`
`removeTool=False`.** Verified empirically that OCC's intersection, when $V_s$ is fully
contained (the common, passing case), can come back *aliasing $V_s$'s own entity tag* rather
than a distinct new solid (no new geometry is needed since the intersection equals $V_s$
exactly) — so discarding that "temporary" intersection result as cleanup silently deletes the
real sample solid out from under the rest of the pipeline. This is invisible to a test that
only exercises this section in isolation (nothing downstream ever touches the now-deleted
entity); it only surfaces once the tagging step (Section 1 step 6) tries to reuse the same
`sample_dim_tag` afterward and gets "Unknown OpenCASCADE entity." Work on `occ.copy(...)` of
both solids, intersect *those* (consumed by the boolean op itself, `removeObject=removeTool=`
`True` is fine there), and discard the copies and their result — never the caller's originals.

---

## 4. Mesh sizing

$$\lambda = \frac{1}{f\sqrt{\epsilon_{bg}\,\mu_{bg}}}, \qquad h_{\text{char}} = \frac{\lambda}{N_{\text{per}\lambda}}$$

**Correction (found implementing `mesh_sizing.py`): the original $\lambda=c/(f\sqrt{\epsilon_{bg}\mu_{bg}})$
is only dimensionally correct if $\epsilon_{bg},\mu_{bg}$ are *relative* (dimensionless).**
Section 0.6 sources these values directly from `CavityMode.epsilon_bg`/`.mu_bg`, which are
**absolute** (SI) per this project's own convention (CLAUDE.md's absolute-vs-relative rule,
the same one Module 4's $p_E,p_H$ and the Ritz plan's $K$/$M$ matrices both have to respect) —
with absolute $\epsilon_{bg},\mu_{bg}$, $1/\sqrt{\epsilon_{bg}\mu_{bg}}$ *is* the phase velocity
directly, so an extra factor of $c$ double-counts $c^2$ relative to the correct answer. The
corrected formula above (no separate $c$) still reduces to the familiar $\lambda=c/f$
automatically in vacuum, since $\epsilon_0\mu_0=1/c^2$ — verified directly in
`tests/test_mesh_sizing.py::test_vacuum_reduces_to_c_over_f`.

with $N_{\text{per}\lambda}$ (`target_elements_per_wavelength`) defaulting to 10 — a
conservative starting point for first-order (lowest-order) Nédélec elements, the element
type scikit-fem provides (per the earlier FEM discussion); this is a tunable convergence
parameter, not a fixed constant, and should be exposed the same way Module 2's quadrature
resolution and the Ritz plan's basis size are both exposed as adjustable, convergence-checked
parameters rather than hardcoded numbers.

Apply $h_{\text{char}}$ as Gmsh's mesh-size field uniformly for a first implementation;
finer control (smaller elements specifically near the sample interface, where the field
gradient and the accuracy requirement are both highest) is a reasonable later refinement, not
a first-pass requirement. **Implementation note**: Gmsh's curvature- and boundary-extension-
based auto-sizing are both *on* by default and will silently refine below `MeshSizeMax` near
any curved feature (e.g. a spherical sample) regardless of the requested resolution — exactly
the "finer control near the sample interface" this section defers as a later refinement, so a
literal "uniform first implementation" needs `Mesh.MeshSizeFromCurvature` and
`Mesh.MeshSizeExtendFromBoundary` both explicitly disabled, not just `MeshSizeMax` set.

---

## 5. Caching

Unlike Module 4's `PerturbationModel` cache (which had to key on `id(region)` and hold a
strong reference to avoid Python's `id()`-reuse hazard, since `SampleRegion` instances are
runtime objects), this module's inputs are **plain, hashable data** by construction (Section
2.1–2.2: STEP paths + `length_unit` + `RigidTransform`, or a `CavityMode`/`SampleRegion`'s own
dimensions) — so a content-addressed cache key (a hash of the full geometry spec plus
`target_elements_per_wavelength`) avoids the identity-cache hazard entirely, rather than
needing Module 4's workaround. For `StandardCavityInput`/`StandardSampleInput`, hash on the
concrete type and its plain-data field values (dimensions, mode indices, `eps`, `mu` for
cavities; center/axis/radius/etc. for samples) — not on `id()` of the object itself.

---

## 6. Verification plan — independent of every other module

### 6.1 What "independent" means here, precisely

Two distinct guarantees, both required:

1. **No shared ground truth with the modules being fed.** The interference check (Section 3)
   and every geometric assertion in this module's own tests must be computed from Gmsh/OCC's
   own mass-property and boolean-operation queries — never by calling back into
   `SampleRegion.volume()`, `.contains()`, or any Module 1/3 method. Where this module reads
   a `CavityMode`/`SampleRegion`'s *dimensions* (public attributes: `.a`, `.radius`,
   `.center`, etc.) for the standard-shape path, that is reading input data, not calling into
   Module 1/3's logic — the distinction matters: this module is allowed to consume Module
   1/3's stated dimensions, but must never rely on Module 1/3's own computed geometric
   quantities (their `volume()`, their `contains()`) to check itself.
2. **Runnable and meaningful in isolation.** `pytest tests/test_meshing/` must pass with no
   other module's tests run first, and — more importantly — without needing
   `perturbation.py`, `inverse.py`, or any EM-physics code to even be correct, or importable
   at all beyond the two read-only dataclasses. If this test suite's pass/fail status ever
   depends on whether Module 4's $\Delta$-conjugate fix is present, something has leaked
   across the boundary that shouldn't have.

### 6.2 Testing philosophy specific to this module

Unlike Modules 1–5 (closed-form equations, so exact-value regression tests are appropriate),
this module wraps a third-party unstructured mesh generator, which is not required to produce
bit-identical output between runs or Gmsh versions. Tests must check **invariant properties**,
not exact element counts or orderings:

- Volumes agree with the OCC mass-property computation to a stated tolerance.
- The mesh is manifold/watertight on the boundary (no gaps, no duplicate/overlapping faces).
- No degenerate elements (zero or negative volume/quality).
- Physical group tags partition the mesh completely (every element belongs to exactly one of
  sample/background; every boundary face belongs to the boundary group).
- Refining `target_elements_per_wavelength` upward changes `n_elements` in the expected
  direction and by roughly the expected scaling (more elements, denser mesh), without
  asserting an exact count.

### 6.3 Section-by-section test plan

- **`standard_shapes.py`**: for each of `RectangularCavity`, `CylindricalCavity`,
  `CoaxialCavity`, `Sphere`, `Cylinder`, `Slab` — build the OCC solid, compute its volume via
  OCC mass properties, and confirm it matches the *closed-form* geometric volume formula
  ($abc$, $\pi r^2 d$, $\pi(b^2-a^2)L$, $\frac43\pi r^3$, $\pi r^2h$,
  $\text{thickness}\times\text{extent}_0\times\text{extent}_1$) computed independently, right
  here in the test, from the same dimensions — not by calling Module 1/3's own `.volume()`.
- **`step_import.py`**: import a small hand-authored fixture `.step` file with a known,
  hand-computed volume (e.g. a unit cube authored directly in a text editor as STEP, not
  generated by this project's own code) in both `'mm'` and `'m'`, confirm the unit conversion
  produces the expected volume in each case; confirm `StepUnitAmbiguityError` fires when
  `length_unit` is omitted.
- **`transforms.py`**: apply a translation-only and a rotation-only `RigidTransform` to a
  simple shape (e.g. a box) and confirm the transformed shape's centroid and volume match a
  hand-computed expectation; confirm composing two transforms matches applying them in
  sequence.
- **`interference.py`**: three cases — sample fully inside (Δ≈0, passes), sample fully
  outside (Δ = full sample volume, fails), sample straddling the boundary (0<Δ<sample volume,
  fails) — plus the flush-against-the-wall edge case from Section 3 (Δ≈0, passes).
- **`tagging.py`**: after fragments + tagging, confirm every mesh element's tag is exactly
  one of {sample, background}, and every boundary-surface element is tagged boundary — a
  complete partition, checked by summing tagged-element volumes back up to the whole-cavity
  volume and comparing to the OCC-computed total.
- **`mesh_sizing.py`**: confirm the $\lambda\to h_{\text{char}}$ conversion is dimensionally
  correct (pass in known $f,\epsilon_{bg},\mu_{bg}$, check against a hand-computed $\lambda$)
  and that both the `CavityMode`-sourced path and the explicit-`reference_frequency` path
  give the same $h_{\text{char}}$ for matching inputs.
- **`mesh_generation.py`**: the invariant-property checks from 6.2 (manifold, no degenerate
  elements, quality metrics above a stated floor) — plus the resolution-scaling check.
- **`mesh_io.py`**: round-trip a mesh through export and re-import via `meshio` and confirm
  vertex count, element count, and total volume are preserved exactly.
- **`cache.py`**: confirm two calls with identical (by value, not by object identity) inputs
  hit the cache and skip re-meshing; confirm any single differing field (e.g. a different
  `target_elements_per_wavelength`) produces a cache miss and a fresh mesh.
- **`test_pipeline_integration.py`**: the only test file allowed to exercise more than one of
  the atomized pieces together — runs the full standard-shape path and the full STEP-import
  path end to end, including a deliberate interference failure, confirming the pipeline stops
  at the right stage (Section 3, before any meshing work) rather than meshing an invalid
  configuration and failing later, more confusingly, downstream.

---

## 7. Step-by-step implementation order

1. `geometry_spec.py` — plain dataclasses first, nothing else depends on Gmsh yet.
2. `transforms.py` — geometry-agnostic rigid-transform math, testable without Gmsh at all
   (pure `numpy`) before it's ever applied to an OCC shape.
3. `standard_shapes.py` — the dispatch table from `CavityMode`/`SampleRegion` types to OCC
   primitive constructors; run 6.3's volume cross-checks immediately.
4. `step_import.py` — STEP import + unit conversion, tested against hand-authored fixtures.
5. `interference.py` — depends on 3 and 4 both producing OCC solids; this is the first place
   the two geometry sources (standard, custom) genuinely meet.
6. `tagging.py` — boolean fragments + physical groups, run only after 5 passes.
7. `mesh_sizing.py` — independent of everything above except needing $f,\epsilon_{bg},\mu_{bg}$
   as plain numbers; can be built and tested in parallel with steps 3–6.
8. `mesh_generation.py` — consumes 6's tagged geometry and 7's characteristic length.
9. `mesh_io.py` — export/round-trip, `MeshResult`/`MeshStats` assembly.
10. `cache.py` — wraps the whole pipeline; build last, once the thing being cached is stable.
11. `pipeline.py` — orchestrates 1–10 into `build_mesh(cavity_input, sample_input, **sizing)`.
12. Run the full Section 6 test plan, including `test_pipeline_integration.py` last.
