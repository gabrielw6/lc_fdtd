"""Example 1: a plain isotropic microstrip line (Phase 1 of the top-level
architecture doc's build plan -- no LC).

Every parameter here is passed straight through to `cli.main`, exactly as
if invoked from a shell as:

    python src/cli.py --w 0.002 --L 0.020 --L-lc 0.008 --W-lc 0.004 \\
        --h-sub 0.002 --W-sub 0.010 --eps-r-substrate 3.0 \\
        --mesh-density 6 --pml-r0 1.0 --pml-kappa-max 1.0 \\
        --f-start 25e9 --f-points 1 --n-modes 2 --plot

This script exists so "run the isotropic example" is a single command
(`python examples/isotropic_microstrip.py`) rather than requiring the
command line above to be retyped or remembered.

A single frequency point, deliberately -- `--plot` still renders it (as
one marker per S-parameter rather than a line, `plot_results`'s own
documented behavior for a single-point sweep). A multi-point sweep was
tried here and reverted: `docs/module6_solve_sweep_equations.md` Section
6.2's mode tracking needs consecutive frequency points to resolve to the
*same* physical mode, and on this coarse test geometry the port
cross-section's box-mode spectrum near 25 GHz is dense enough that even a
0.1 GHz step (checked at both mesh-density 6 and 10) lands on a different,
essentially orthogonal eigenvector each time -- a genuine eigenvalue
reordering within a near-degenerate cluster, not a bug in tracking or in
this plotting feature (`ModeTrackingError` is doing exactly its documented
job: refusing to silently mismatch rather than reporting a wrong answer).
This is the same already-documented Module 4 "box mode" limitation
(`docs/CLAUDE.md`'s carry-forward notes), now surfaced at the sweep level
-- not something to chase further with example parameters. A real
multi-point sweep needs either a substantially finer mesh or a wider
substrate (W_sub) to separate the box-mode spectrum from the quasi-TEM
mode's beta, neither of which belongs in a fast illustrative example.

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
                "--w", "0.002",
                "--L", "0.020",
                "--L-lc", "0.008",
                "--W-lc", "0.004",
                "--h-sub", "0.002",
                "--W-sub", "0.010",
                "--eps-r-substrate", "3.0",
                # Coarse and fast, for a quick end-to-end run -- see this
                # file's own module docstring for what to change for a
                # production-accuracy result.
                "--mesh-density", "6",
                "--pml-r0", "1.0",
                "--pml-kappa-max", "1.0",
                "--f-start", "25e9",
                "--f-points", "1",
                "--n-modes", "2",
                "--lc", "none",
                "--plot",
            ]
        )
    )
