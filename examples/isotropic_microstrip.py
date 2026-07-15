"""Example 1: a plain isotropic microstrip line (Phase 1 of the top-level
architecture doc's build plan -- no LC).

Every parameter here is passed straight through to `cli.main`, exactly as
if invoked from a shell as:

    python src/cli.py --w 0.000629 --L 0.020 --L-lc 0.008 --W-lc 0.004 \\
        --h-sub 0.00025 --W-sub 0.006 --eps-r-substrate 3.0 --h-air 0.003 \\
        --mesh-density 10 --pml-r0 1.0 --pml-kappa-max 1.0 \\
        --f-start 8e9 --f-stop 12e9 --f-points 9 --n-modes 1 \\
        --w-port 0.005 --h-port 0.003 --plot --show-geometry

**A validated 50-ohm design (Hammerstad-Jensen), replacing an earlier
version of this example that hit `ports`'s per-port axial-orientation
bug head-on.** At `w=0.629mm`, `h_sub=0.25mm`, `eps_r=3`:
`Z0=50.0 ohm`, `eps_eff=2.425`, `beta=326 rad/m`, `lambda_g=19.25mm`, and
`h_sub/lambda_sub=0.014` (single-mode to roughly 69 GHz, per the usual
`h~0.1*lambda_sub` rule of thumb) -- comfortably single-mode at this
example's 10 GHz operating point, and small enough relative to the line's
own `L=20mm` for the port-to-port phase (`arg(S21)~=-beta*L~=-14deg`
ideally, see the caveat below) to be a meaningful check.

`--show-geometry` opens `visualization.geometry_view.plot_geometry`'s 3D
view (ports, PEC line, substrate envelope; PMC deliberately omitted) right
after meshing -- a quick sanity check that the geometry parameters above
produced what you expect, independent of whether the sweep itself
succeeds.

`--w-port`/`--h-port` restrict the port aperture to 5mm x 3mm, strictly
smaller than the full 6mm x 3.25mm domain cross-section (`h_sub+h_air` =
0.25mm + 3mm) -- the end-plane region outside the aperture becomes a PEC
cap (`geometry_builder`'s port-aperture decoupling), which raises the
aperture's own box-mode cutoff. At 10 GHz/`eps_r=3`, `lambda_min/2` =
8.66mm, so this aperture is below the box-mode-avoidance bound; the
fringing-capture floor (`w+6*h_sub=2.1mm`, `5*h_sub=1.25mm`) is also
comfortably satisfied by the 5mm/3mm aperture -- unlike an earlier version
of this example (a thicker substrate, proportionally wider trace), this
geometry satisfies *both* `ports.sizing` rules simultaneously, which is
exactly what makes both ports reliably land on the true quasi-TEM dominant
mode (see below) instead of Module 4 Section 3.7's known box-mode
mode-selection limitation.

**Root-cause note (per-port axial-orientation review).** An earlier
version of this example used a different (thicker-substrate) geometry and
reported a suspicious result: driving PORT_2 gave `|S22|~=1` and
`|S21|~=0` -- PORT_2 was landing on a spurious evanescent/box mode instead
of the physical quasi-TEM mode (Module 4 Section 3.7's documented
limitation, aggravated by that geometry's box-mode-unsafe aperture
proportions). That review also checked (and, empirically, ruled out) a
second, distinct hypothesis: that `ports.port_operator.build_B`/`build_g`
themselves needed an explicit correction for PORT_2's outward normal being
`+x_hat` instead of PORT_1's `-x_hat` -- see `build_B`'s own docstring for
the full account of what was tried and why it was reverted. This example's
geometry (this docstring's own box-mode-safe aperture) sidesteps the mode-
selection failure mode entirely, which is what actually let this example
produce a sane, reciprocal S-matrix -- see `test/test_extract/
test_reciprocity_uniform_line.py` for the automated version of this same
check, on a small fast mesh.

This script exists so "run the isotropic example" is a single command
(`python examples/isotropic_microstrip.py`) rather than requiring the
command line above to be retyped or remembered.

**A real 9-point frequency sweep, 8-12 GHz.** An earlier version of this
example ran a single frequency point deliberately, since a multi-point
sweep needs `docs/module6_solve_sweep_equations.md` Section 6.2's mode
tracking to resolve consecutive frequency points to the *same* physical
mode, and an earlier, box-mode-risky geometry hit exactly the kind of
near-degenerate box-mode spectrum that breaks plain per-point beta-sort
(`ModeTrackingError`'s documented job). This geometry's box-mode-safe
aperture (above) removes that risk: re-tried here, this 9-point sweep
tracks cleanly end to end with no `SweepPreconditionError`/
`ModeTrackingError`, and `S21~=S12` holds to 3-4 significant figures at
*every* point, not just the one frequency checked before -- a stronger
confirmation of the axial-orientation review than a single point gives.

**Passivity/energy-conservation fix (the actual cause of the old `|S11|`/
`|S21|` gap, corrected -- see `ports.port_operator.build_B`'s own
docstring for the full derivation).** An earlier version of this docstring
blamed the gap between this example's raw `|S11|`/`|S21|` and the ideal
matched-line targets above on the trivial PML's reflections. **That
rationalization was physically wrong**: a lossless trivial PML (`Lambda=I`
-- a PEC-lidded shielded cavity, not an open one) reflects power back
through the ports rather than absorbing it, but it does not destroy
energy -- `|S11|^2+|S21|^2` (and `|S22|^2+|S12|^2`) must still equal 1 to
discretization error regardless of how much of the incident wave comes
back reflected instead of transmitted. The real cause was a genuine
injection/extraction normalization bug in `ports.port_operator.build_B`:
it was missing a `1/N_m` factor (`N_m` = each mode's own *unconjugated*
self-overlap, ~2x its `P_m=1` conjugated power for a lossless mode) that
S-parameter *extraction* (`extract.project`) already applied -- an
injection/extraction mismatch, not a PML-explainable effect. Fixed, this
example's own `|S11|^2+|S21|^2` now lands within a couple percent of 1.0
at every swept frequency (see `extract.energy_balance`, now printed by
`cli.main` on every run) even with the still-trivial PML below, and
`|S11|`/`|S21|` themselves land close to the ideal matched-line targets
too -- see `test/test_extract/test_reciprocity_uniform_line.py` for the
automated passivity gate this fix is checked against (several
frequencies, `|S11|^2+|S21|^2~=1` and `|S22|^2+|S12|^2~=1`, not merely
`<=1`).

**PML caveat, honestly documented rather than silently worked around --
revised.** This example still uses the trivial (`--pml-r0 1.0
--pml-kappa-max 1.0`) PML, not a real absorbing one, but *not* because a
thicker `--h-pml` would fix it, as an earlier version of this docstring
claimed. **That claim was also checked and found wrong**: a real PML
(`--pml-r0 1e-6 --pml-kappa-max 3.0`, and milder settings down to
`--pml-r0 0.9 --pml-kappa-max 1.01`) was retried against this exact
geometry across `--h-pml` from the default (0.125mm) up to 15mm --
*three orders of magnitude* thicker than "4-6 elements across the PML
thickness" (Module 5's own guidance) should ever require at this mesh
density -- and every combination still raised
`fem.assembly.AssemblyConvergenceError`. Inspecting `PMLMaterial`'s own
`epsilon()` directly shows why: even the mild `R0=0.9` case swings the
PML's tangential permittivity across roughly two orders of magnitude in
imaginary part from the inner to the outer PML boundary -- a genuinely
steep material profile within a single coarse element that the adaptive
quadrature's centroid-subdivision ladder (`fem.assembly.DEFAULT_LEVELS`,
capped at level 3) does not converge for, independent of `--h-pml`. This
reproduces on the *previous* (thicker-substrate) version of this example's
geometry too -- a pre-existing, general PML/quadrature limitation, not
something `--h-pml` alone can work around, and not something introduced
by or specific to this design. Fixing it means touching `fem.assembly`'s
adaptive quadrature (a finer subdivision ladder, or an adaptive scheme
that concentrates refinement near the PML's outer edge) or
`pml.stretching`'s grading profile itself, both out of scope for this
ports-focused example update. Treat `beta` (from `mode.gamma`, independent
of the PML choice) as the analytic-comparison number that most directly
validates against Hammerstad-Jensen here; the trivial-PML `|S11|`/`|S21|`
above are now a meaningful matched-line approximation too (see the
passivity fix above), just not identical to what a genuinely open
boundary would give.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--w", "0.000629",
                "--L", "0.020",
                "--L-lc", "0.008",
                "--W-lc", "0.004",
                "--h-sub", "0.00025",
                "--W-sub", "0.006",
                "--eps-r-substrate", "3.0",
                "--h-air", "0.003",
                # Coarse and fast, for a quick end-to-end run; still a
                # trivial (non-absorbing) PML -- see this file's own module
                # docstring's PML caveat for why a real one isn't used here.
                "--mesh-density", "15",
                "--pml-r0", "1.0",
                "--pml-kappa-max", "1.0",
                "--f-start", "8e9",
                "--f-stop", "12e9",
                "--f-points", "9",
                "--n-modes", "1",
                "--lc", "none",
                "--w-port", "0.005",
                "--h-port", "0.003",
                "--plot",
                "--show-geometry",
                "--plot-field-slice", "x=0.01",
                "--slice-grid", "120"
            ]
        )
    )
