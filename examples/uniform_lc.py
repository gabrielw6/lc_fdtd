"""Example 2: the same microstrip line, now with the LC cavity filled by
a single, uniform director orientation (a Phase 3-equivalent case, per
`docs/module2_material_equations.md` Section 4.7's module-boundary
contract -- a spatially-uniform director field reduces to Phase 3's
rotated-uniaxial tensor exactly).

The director field here is *artificially generated*, not read from a
file: `--lc uniform` (see `cli.build_lc_material`) places the same
orientation vector at the 8 corners of the LC cavity's own bounding box
and lets Module 2's convex-combination interpolation reproduce that exact
direction at every interior point -- the simplest possible stand-in for a
real single-domain (all molecules aligned) LC bias state, useful for
exercising the pipeline without needing an actual director-field solver.

The chosen direction, (0, 1, 0.3) (not normalized on the command line --
the CLI normalizes it), has a nonzero y-component, so it is *not* confined
to the microstrip's vertical mirror-symmetry plane (the x-z plane through
the trace centerline) -- per `docs/module7_extract_sparameters_equations.md`
Section 8's exact selection-rule test, this should produce genuinely
nonzero mode conversion (the S_*_2_*_1 columns in the CSV output), unlike
an in-plane director, which the same test requires to give *exactly* zero
conversion. Try changing the direction to e.g. "1,0,0.3" (zero
y-component, in-plane) and compare the mode-conversion columns to see
that selection rule in action.

eps_perp/eps_parallel below are illustrative round numbers for a generic
nematic LC, not measured values for any specific real material.

A single frequency point, deliberately -- see examples/isotropic_microstrip.py's
docstring for why a multi-point sweep was tried and reverted on an earlier
version of this test geometry (a Module 4/6 box-mode/mode-tracking
limitation, not a plotting issue). `--plot` still renders the single point
as a marker.

`--show-mesh` opens `visualization.geometry_view.plot_mesh`'s 3D wireframe
of the tetrahedral mesh itself (the exact global edge list the edge-element
solver assembles on) right after meshing -- see
examples/isotropic_microstrip.py for the geometry-envelope view instead
(`--show-geometry`).

Geometry/port-aperture dimensions (w, h_sub, mesh-density, f-start,
w-port, h-port) match examples/isotropic_microstrip.py exactly -- see that
file's docstring for the box-mode/fringing-margin arithmetic behind those
numbers. The LC cavity sits well inside the isotropic feed section either
way (Module 4 Section 1's design invariant: port cross-sections are never
anisotropic), so the port-aperture sizing behaves identically here.

`--n-modes 2` here, unlike the isotropic example's `--n-modes 1`: this
case is the one where a second (cross-polarization) mode is physically
expected -- the off-axis director direction below couples power into it
(see the mode-conversion note below) -- so 2 is the *required* minimum,
not merely a desired oversupply (`ports.mode_solver.PortModeSolver.solve`'s
`n_modes`/`n_desired` split, port-aperture-decoupling review Part B: this
port must genuinely have 2 valid modes, or the sweep should fail loudly
rather than silently proceed with 1).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(
        main(
            [
                "--w", "0.00334",
                "--L", "0.020",
                "--L-lc", "0.008",
                "--W-lc", "0.004",
                "--h-sub", "0.0015",
                "--W-sub", "0.010",
                "--eps-r-substrate", "3.0",
                "--mesh-density", "10",
                "--pml-r0", "1.0",
                "--pml-kappa-max", "1.0",
                "--f-start", "10e9",
                "--f-points", "1",
                "--n-modes", "2",
                "--lc", "uniform",
                "--lc-direction", "0,1,0.3",
                "--eps-perp", "2.5",
                "--eps-parallel", "3.0",
                "--w-port", "0.008",
                "--h-port", "0.006",
                "--plot",
                "--show-mesh",
            ]
        )
    )
