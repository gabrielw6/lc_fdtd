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

**PML caveat, honestly documented rather than silently worked around.**
This example intentionally still uses the trivial (`--pml-r0 1.0
--pml-kappa-max 1.0`, i.e. `Lambda=I`, non-absorbing -- a PEC-lidded
shielded cavity, not an open one) PML, *not* a real absorbing one. A real
PML (e.g. `--pml-r0 1e-6 --pml-kappa-max 3.0`) was tried against this exact
geometry at every mesh density from 8 through 15 and consistently raised
`fem.assembly.AssemblyConvergenceError` -- the default `--h-pml` (half of
this design's already-thin 0.25mm substrate) is far too thin relative to
the wavelength-driven element size for the adaptive quadrature to resolve
the graded stretch profile within a single element (Module 5's own
guidance: "at least 4-6 elements across the PML thickness"), and this
reproduces on the *previous* (thicker-substrate) version of this example's
geometry too -- a pre-existing, general mesh/PML-resolution limitation,
not something introduced by or specific to this design. Fixing it means
touching `fem.assembly`'s adaptive quadrature or the geometry builder's
PML-region sizing, out of scope for this ports-focused example update.
Consequently: `|S11|`/`|S21|` reported by this example (see this file's
own module-level run output) do **not** match the ideal matched-line
targets (`|S11|~0`, `|S21|~1`) above -- the trivial PML's reflections
alone account for the gap. Reciprocity (`S21~=S12`, `S11~=S22`) is still
the meaningful, real check this example demonstrates; treat `beta` (from
`mode.gamma`, independent of the PML choice) as the analytic-comparison
number that actually validates against Hammerstad-Jensen here, not the
raw `|S|` magnitudes. For a genuinely matched-line result, a real PML
needs a separate, finer-mesh run once the resolution limitation above is
addressed -- raise `--mesh-density` *and* `--h-pml` together (not
`--mesh-density` alone, which this review confirmed does not by itself fix
it).
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
                "--mesh-density", "10",
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
