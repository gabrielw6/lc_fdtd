"""Example 1: a plain isotropic microstrip line (Phase 1 of the top-level
architecture doc's build plan -- no LC).

Every parameter here is passed straight through to `cli.main`, exactly as
if invoked from a shell as:

    python src/cli.py --w 0.00334 --L 0.020 --L-lc 0.008 --W-lc 0.004 \\
        --h-sub 0.0015 --W-sub 0.010 --eps-r-substrate 3.0 \\
        --mesh-density 10 --pml-r0 1.0 --pml-kappa-max 1.0 \\
        --f-start 10e9 --f-points 1 --n-modes 1 \\
        --w-port 0.008 --h-port 0.006 --plot --show-geometry

`h_sub=1.5mm` at `eps_r=3` is electrically thin (single-mode-valid) up to
roughly 11.5 GHz (the usual h~0.1*lambda_sub rule of thumb) -- comfortably
above this example's 10 GHz operating point. An earlier version of this
example ran a thicker/wider substrate (h_sub=2mm, W_sub=10mm) at 25 GHz,
which was multimoded (valid only to ~9 GHz) and needed `--n-modes 2` just
to avoid `PortModeSolver.solve`'s old "found fewer modes than requested"
hard failure on a port that is physically single-mode. That failure mode
is gone now (`ports.mode_solver.PortModeSolver.solve`'s `n_modes`/
`n_desired` split, port-aperture-decoupling review Part B): `n_modes` is a
*required* minimum, and `solve.run_sweep` requests `n_modes+2` as a merely
*desired* oversupply for mode tracking, so `--n-modes 1` on a genuinely
single-mode port no longer raises.

`--show-geometry` opens `visualization.geometry_view.plot_geometry`'s 3D
view (ports, PEC line, substrate envelope; PMC deliberately omitted) right
after meshing -- a quick sanity check that the geometry parameters above
produced what you expect, independent of whether the sweep itself
succeeds.

`--w-port`/`--h-port` restrict the port aperture to 8mm x 6mm, strictly
smaller than the full 10mm x 7.5mm domain cross-section (h_sub+h_air =
1.5mm + 3*1.5mm) -- the end-plane region outside the aperture becomes a
PEC cap (`geometry_builder`'s port-aperture decoupling, Part A), which
raises the aperture's own box-mode cutoff. At 10 GHz/eps_r=3,
lambda_min/2 = 8.66mm, so this 8mm aperture is genuinely *below* the
box-mode-avoidance bound (unlike an earlier version of this example at 25
GHz, where 6mm was still above the tighter 3.46mm bound there) --
`ports.sizing.check_port_sizing` accordingly does **not** warn about
box-mode risk for this configuration. It does still warn about the
fringing-capture floor (`w+6*h_sub` = 12.34mm > `W_port` = 8mm, and
`5*h_sub` = 7.5mm > `H_port` = 6mm): satisfying that floor would need an
aperture *above* the 8.66mm box-mode ceiling, so this trace/substrate
proportion cannot satisfy both `ports.sizing` rules simultaneously at any
aperture size -- expected, not a bug; see
`ports.sizing.check_port_sizing`'s own docstring for what each rule means.

This script exists so "run the isotropic example" is a single command
(`python examples/isotropic_microstrip.py`) rather than requiring the
command line above to be retyped or remembered.

A single frequency point, deliberately -- `--plot` still renders it (as
one marker per S-parameter rather than a line, `plot_results`'s own
documented behavior for a single-point sweep). A multi-point sweep was
tried on an earlier version of this example and reverted:
`docs/module6_solve_sweep_equations.md` Section 6.2's mode tracking needs
consecutive frequency points to resolve to the *same* physical mode, and
on a coarser test geometry the port cross-section's box-mode spectrum was
dense enough that even a 0.1 GHz step landed on a different, essentially
orthogonal eigenvector each time -- a genuine eigenvalue reordering within
a near-degenerate cluster, not a bug in tracking (`ModeTrackingError` is
doing exactly its documented job: refusing to silently mismatch rather
than reporting a wrong answer). This is the same already-documented
Module 4 "box mode" limitation (`docs/CLAUDE.md`'s carry-forward notes).
This geometry's box-mode-safe aperture makes a multi-point sweep more
promising to retry than before, but it hasn't been re-tried here -- still
not something to chase further in a fast illustrative example.

Run parameters are otherwise chosen for speed (a coarse mesh, a
trivial/non-absorbing PML), not accuracy -- see the mesh-density and PML
notes below before using these settings for anything beyond "does the
pipeline run end to end and does the plot look sane." For a real result,
raise --mesh-density (Module 5's own guidance: at least 4-6 elements
across the PML thickness for a graded profile to be resolvable) and use a
real --pml-r0 (e.g. 1e-6).
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
                # Coarse and fast, for a quick end-to-end run -- see this
                # file's own module docstring for what to change for a
                # production-accuracy result.
                "--mesh-density", "10",
                "--pml-r0", "1.0",
                "--pml-kappa-max", "1.0",
                "--f-start", "10e9",
                "--f-points", "1",
                "--n-modes", "1",
                "--lc", "none",
                "--w-port", "0.008",
                "--h-port", "0.006",
                "--plot",
                "--show-geometry",
                "--plot-field-slice", "x=0.01",
                "--slice-grid", "120"
            ]
        )
    )
