"""cli -- a small command-line driver over the full Module 0-8 pipeline:
geometry -> mesh -> materials -> frequency sweep -> S-parameter extraction.

Not itself a physics module (no `docs/module*_equations.md` companion) --
it introduces no new equations, just orchestrates the already-specified
modules end to end for interactive/scripted use. Every quantity it
computes is exactly what `solve.run_sweep` + `extract.*` already produce;
this file's only job is argument parsing, material-registry assembly, and
result formatting.

Usage:
    python src/cli.py --w 0.002 --L 0.020 --L-lc 0.008 --W-lc 0.004 \\
        --h-sub 0.002 --W-sub 0.010 --eps-r-substrate 3.0 \\
        --f-start 25e9 --f-stop 25e9 --f-points 1

Run with --help for the full option list.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from extract import assemble_sweep_dataset, deembed, raw_s_parameters  # noqa: E402
from geometry_builder import GeometryBuilder, GeometryParams  # noqa: E402
from geometry_builder.params import GeometryParameterError  # noqa: E402
from material import ConstantMaterial, DirectorFieldMaterial, MaterialAssembly, load_material_spec  # noqa: E402
from material.core import MaterialError  # noqa: E402
from material.interpolation import CoverageError  # noqa: E402
from material.tensor_interpolation import DirectorFieldError  # noqa: E402
from mesh_interface import MeshInterface  # noqa: E402
from mesh_interface.interface import MeshGeometryError  # noqa: E402
from ports import CrossSectionError, PortModeError  # noqa: E402
from solve import ModeTrackingError, SolveSingularityError, SweepPreconditionError, SystemSymmetryError  # noqa: E402
from solve import run_sweep  # noqa: E402

class CLIError(RuntimeError):
    """Raised for a bad CLI invocation (argument combination this parser's
    own type-checking can't express) -- caught in `main`, reported as a
    clean one-line error rather than a traceback."""


class PlottingUnavailableError(RuntimeError):
    """Raised when --plot/--plot-output is requested but matplotlib isn't
    installed -- caught in `main` and reported as a clean one-line error
    (with the install command), not a raw ImportError traceback."""


# Domain errors every pipeline stage can raise for an ill-posed (not
# buggy) input -- caught in `main` and reported as a clean one-line
# message. Anything *not* in this tuple is left to propagate as a full
# traceback, since it signals an actual bug rather than a bad input.
_KNOWN_ERRORS = (
    CLIError,
    PlottingUnavailableError,
    GeometryParameterError,
    MeshGeometryError,
    MaterialError,
    CoverageError,
    DirectorFieldError,
    CrossSectionError,
    PortModeError,
    SolveSingularityError,
    SystemSymmetryError,
    ModeTrackingError,
    SweepPreconditionError,
)


# ============================================================================
# Argument parsing
# ============================================================================


def _direction(text: str) -> np.ndarray:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'nx,ny,nz', got {text!r}")
    try:
        vec = np.array([float(p) for p in parts])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"could not parse direction {text!r}: {exc}") from exc
    if np.linalg.norm(vec) == 0:
        raise argparse.ArgumentTypeError("direction vector must be nonzero")
    return vec


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="Driven S-parameter FEM solve for a microstrip line with an optional LC-loaded cavity.",
    )

    geom = p.add_argument_group("geometry (docs/module0_geometry_builder_equations.md Section 1.1)")
    geom.add_argument("--w", type=float, required=True, help="trace width [m]")
    geom.add_argument("--L", type=float, required=True, help="total line length [m]")
    geom.add_argument("--L-lc", type=float, required=True, help="LC cavity length [m]")
    geom.add_argument("--W-lc", type=float, required=True, help="LC cavity width [m]")
    geom.add_argument("--h-sub", type=float, required=True, help="substrate height [m]")
    geom.add_argument("--W-sub", type=float, required=True, help="substrate width [m]")
    geom.add_argument("--eps-r-substrate", type=float, required=True, help="substrate relative permittivity")
    geom.add_argument("--tan-delta-substrate", type=float, default=0.0, help="substrate loss tangent (default 0.0)")
    geom.add_argument("--h-air", type=float, default=None, help="air region height [m] (default 3*h_sub)")
    geom.add_argument("--h-pml", type=float, default=None, help="PML shell thickness [m] (default 0.5*h_sub)")
    geom.add_argument(
        "--mesh-density", type=int, default=10, dest="mesh_density",
        help="target elements per wavelength at the reference frequency (default 10)",
    )

    freq = p.add_argument_group("frequency sweep")
    freq.add_argument("--f-start", type=float, required=True, help="sweep start frequency [Hz]")
    freq.add_argument("--f-stop", type=float, default=None, help="sweep stop frequency [Hz] (default: --f-start, single point)")
    freq.add_argument("--f-points", type=int, default=1, help="number of frequency points (default 1)")

    ports = p.add_argument_group("ports / sweep")
    ports.add_argument("--n-modes", type=int, default=2, help="tracked modes per port (default 2)")
    ports.add_argument("--offset-port1", type=float, default=0.0, help="de-embedding offset at PORT_1 [m] (default 0.0)")
    ports.add_argument("--offset-port2", type=float, default=0.0, help="de-embedding offset at PORT_2 [m] (default 0.0)")

    pml = p.add_argument_group("PML (docs/module5_pml_equations.md Section 2)")
    pml.add_argument("--pml-r0", type=float, default=1e-6, help="target reflection coefficient (default 1e-6)")
    pml.add_argument("--pml-n", type=int, default=2, help="grading polynomial order (default 2)")
    pml.add_argument("--pml-kappa-max", type=float, default=3.0, help="max real stretch (default 3.0)")

    lc = p.add_argument_group("LC material (docs/module2_material_equations.md Section 4; omit for an isotropic line)")
    lc.add_argument(
        "--lc", choices=["none", "uniform", "file"], default="none",
        help="LC region material: 'none' (isotropic, substrate-matched), 'uniform' (a single, "
        "artificially-generated director orientation), or 'file' (a real director field file)",
    )
    lc.add_argument("--lc-direction", type=_direction, default=None, metavar="NX,NY,NZ", help="director orientation for --lc uniform")
    lc.add_argument("--lc-director-file", type=Path, default=None, help="director field file path for --lc file")
    lc.add_argument("--eps-perp", type=float, default=None, help="LC ordinary relative permittivity (required if --lc != none)")
    lc.add_argument("--eps-parallel", type=float, default=None, help="LC extraordinary relative permittivity (required if --lc != none)")

    out = p.add_argument_group("output")
    out.add_argument("--output", type=Path, default=None, help="CSV output path (default: print to stdout)")
    out.add_argument("--quiet", action="store_true", help="suppress progress messages on stderr")
    out.add_argument(
        "--plot", action="store_true",
        help="show |S| vs. frequency plots (matplotlib) after the sweep completes",
    )
    out.add_argument("--plot-output", type=Path, default=None, help="save the plot to this path (in addition to, or instead of, --plot)")

    return p


# ============================================================================
# Material assembly
# ============================================================================


def _lc_region_bounds(params: GeometryParams) -> tuple[np.ndarray, np.ndarray]:
    """The LC cavity's own bounding box (docs/module0_geometry_builder_equations.md
    Section 1.3's derived quantities), independently recomputed here from
    `params` rather than reaching into `geometry_builder`'s internal
    `derive()` -- this module only needs the box, not the rest of
    `DerivedGeometry`."""
    x_c0 = (params.L - params.L_lc) / 2.0
    x_c1 = (params.L + params.L_lc) / 2.0
    y_lc0 = (params.W_sub - params.W_lc) / 2.0
    y_lc1 = (params.W_sub + params.W_lc) / 2.0
    return np.array([x_c0, y_lc0, 0.0]), np.array([x_c1, y_lc1, params.h_sub])


def build_lc_material(args: argparse.Namespace, params: GeometryParams) -> "ConstantMaterial | DirectorFieldMaterial":
    if args.lc == "none":
        return ConstantMaterial(eps_r=params.eps_r_substrate)

    if args.eps_perp is None or args.eps_parallel is None:
        raise CLIError("--eps-perp and --eps-parallel are required when --lc is not 'none'")

    lo, hi = _lc_region_bounds(params)

    if args.lc == "uniform":
        if args.lc_direction is None:
            raise CLIError("--lc-direction is required for --lc uniform")
        # The 8 corners of the LC cavity's bounding box, all carrying the
        # same orientation -- Section 4.1's convex-combination interpolation
        # (docs/module2_material_equations.md) then reproduces that exact
        # direction at every interior point, no file needed.
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        directions = np.tile(args.lc_direction / np.linalg.norm(args.lc_direction), (8, 1))
        return DirectorFieldMaterial(corners, directions, args.eps_perp, args.eps_parallel, region_bounds=(lo, hi))

    if args.lc == "file":
        if args.lc_director_file is None:
            raise CLIError("--lc-director-file is required for --lc file")
        return DirectorFieldMaterial.from_file(
            args.lc_director_file, args.eps_perp, args.eps_parallel, region_bounds=(lo, hi)
        )

    raise CLIError(f"unknown --lc value {args.lc!r}")  # pragma: no cover -- argparse choices already restrict this


# ============================================================================
# Result formatting
# ============================================================================


def _format_csv(frequencies: list[float], ports: list[str], S_by_freq: list[dict], n_modes: int) -> str:
    header = ["frequency_Hz"]
    for p in ports:
        for q in ports:
            header += [f"S_{p}_1_{q}_1_re", f"S_{p}_1_{q}_1_im"]
    for m in range(2, n_modes + 1):
        for p in ports:
            for q in ports:
                header += [f"S_{p}_{m}_{q}_1_re", f"S_{p}_{m}_{q}_1_im"]

    lines = [",".join(header)]
    for omega, S in zip(frequencies, S_by_freq):
        f_hz = float(omega) / (2.0 * np.pi)
        row = [repr(f_hz)]
        for m in range(1, n_modes + 1):
            for p in ports:
                for q in ports:
                    value = complex(S.get((p, m, q), 0j))
                    row += [repr(value.real), repr(value.imag)]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def plot_results(
    frequencies_hz: list[float],
    ports: list[str],
    S_by_freq: list[dict],
    n_modes: int,
    *,
    show: bool = True,
    output: "Path | None" = None,
):
    """`|S|` in dB vs. frequency: one subplot for the dominant (m=1) block
    (docs/module7_extract_sparameters_equations.md Section 3.2's
    `S^dominant`) and, whenever `n_modes>1`, a second subplot for every
    captured mode-conversion term (`S^conversion`) -- plotted separately
    because the two are typically orders of magnitude apart, and Section
    3.2's whole point is that the conversion rows matter on their own
    terms, not as an afterthought squeezed onto the same axes. A
    single-frequency sweep renders as isolated markers rather than a
    line, which is still a meaningful (if minimal) plot, not an error."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlottingUnavailableError(
            "matplotlib is required for --plot/--plot-output; install it with 'pip install matplotlib'"
        ) from exc

    freqs_ghz = [float(f) / 1e9 for f in frequencies_hz]
    marker = "o" if len(freqs_ghz) == 1 else None

    def _db(S: dict, key: tuple[str, int, str]) -> float:
        magnitude = abs(complex(S.get(key, 0j)))
        return 20.0 * np.log10(max(magnitude, 1e-12))

    n_rows = 2 if n_modes > 1 else 1
    fig, axes = plt.subplots(n_rows, 1, squeeze=False, figsize=(7.0, 4.5 * n_rows))

    ax_dom = axes[0, 0]
    for p in ports:
        for q in ports:
            values = [_db(S, (p, 1, q)) for S in S_by_freq]
            ax_dom.plot(freqs_ghz, values, marker=marker, label=f"S({p},1;{q},1)")
    ax_dom.set_xlabel("Frequency [GHz]")
    ax_dom.set_ylabel("|S| [dB]")
    ax_dom.set_title("Dominant-mode S-parameters")
    ax_dom.legend(fontsize=8)
    ax_dom.grid(True, alpha=0.3)

    if n_modes > 1:
        ax_conv = axes[1, 0]
        for m in range(2, n_modes + 1):
            for p in ports:
                for q in ports:
                    values = [_db(S, (p, m, q)) for S in S_by_freq]
                    ax_conv.plot(freqs_ghz, values, marker=marker, label=f"S({p},{m};{q},1)")
        ax_conv.set_xlabel("Frequency [GHz]")
        ax_conv.set_ylabel("|S| [dB]")
        ax_conv.set_title("Mode-conversion S-parameters")
        ax_conv.legend(fontsize=8)
        ax_conv.grid(True, alpha=0.3)

    fig.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================================
# Main
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ports = ["PORT_1", "PORT_2"]

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg, file=sys.stderr)

    try:
        f_stop = args.f_stop if args.f_stop is not None else args.f_start
        if args.f_points < 1:
            raise CLIError("--f-points must be >= 1")
        f_values = np.linspace(args.f_start, f_stop, args.f_points)
        reference_frequency = float(f_values[len(f_values) // 2])

        params = GeometryParams(
            w=args.w, L=args.L, L_lc=args.L_lc, W_lc=args.W_lc, h_sub=args.h_sub, W_sub=args.W_sub,
            eps_r_substrate=args.eps_r_substrate, tan_delta_substrate=args.tan_delta_substrate,
            h_air=args.h_air, h_pml=args.h_pml,
            reference_frequency=reference_frequency, target_elements_per_wavelength=args.mesh_density,
        )

        t0 = time.time()
        log("Building geometry and mesh...")
        mesh_handle, material_stub = GeometryBuilder().build(params)
        mesh = MeshInterface.from_mesh_handle(mesh_handle)
        log(f"  {mesh.n_tets} tets, {mesh.n_edges} edges ({time.time() - t0:.1f}s)")

        base_assembly = load_material_spec(geometry_stub=material_stub)
        tag_to_model = dict(base_assembly._tag_to_model)
        tag_to_model["LC"] = build_lc_material(args, params)
        materials = MaterialAssembly(tag_to_model)

        pml_params = dict(
            background=ConstantMaterial(eps_r=1.0),
            z_air_top=params.h_sub + params.h_air,
            thickness=params.h_pml,
            R0=args.pml_r0, n=args.pml_n, kappa_max=args.pml_kappa_max,
        )

        omegas = [2.0 * np.pi * float(f) for f in f_values]
        log(f"Running sweep over {len(omegas)} frequency point(s)...")
        t1 = time.time()
        sweep_results = run_sweep(mesh, materials, ports, omegas, n_modes=args.n_modes, pml_params=pml_params)
        log(f"  done ({time.time() - t1:.1f}s)")

        offsets = {"PORT_1": args.offset_port1, "PORT_2": args.offset_port2}
        S_by_freq = []
        for omega in omegas:
            freq_results = [r for r in sweep_results if r.omega == omega]
            raw_S = raw_s_parameters(freq_results, ports, args.n_modes)
            S_by_freq.append(deembed(raw_S, freq_results[0].port_modes, offsets))

        dataset = assemble_sweep_dataset(omegas, S_by_freq)  # noqa: F841 -- built for API completeness/future reuse
        csv_text = _format_csv(omegas, ports, S_by_freq, args.n_modes)

        if args.output is not None:
            args.output.write_text(csv_text)
            log(f"Wrote {args.output}")
        else:
            print(csv_text, end="")

        if args.plot or args.plot_output is not None:
            log("Plotting...")
            plot_results(
                f_values, ports, S_by_freq, args.n_modes,
                show=args.plot, output=args.plot_output,
            )
            if args.plot_output is not None:
                log(f"Wrote {args.plot_output}")

        return 0

    except _KNOWN_ERRORS as exc:
        print(f"Error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
