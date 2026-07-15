"""Validation suite for cli.py -- the argument parsing / material-assembly
/ output-formatting logic unit-tested directly, plus one real end-to-end
invocation mirroring examples/isotropic_microstrip.py.
"""
import matplotlib

matplotlib.use("Agg")  # never open a GUI window / call plt.show() for real during tests

import numpy as np
import pytest

from cli import CLIError, PlottingUnavailableError, _direction, _format_csv, build_arg_parser, build_lc_material, main, plot_results
from geometry_builder import GeometryParams
from material import ConstantMaterial, DirectorFieldMaterial

_REQUIRED_GEOM_ARGS = [
    "--w", "0.002", "--L", "0.020", "--L-lc", "0.008", "--W-lc", "0.004",
    "--h-sub", "0.002", "--W-sub", "0.010", "--eps-r-substrate", "3.0",
    "--f-start", "25e9",
]


# --- argument parsing ---


def test_parser_requires_geometry_args():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--f-start", "25e9"])  # missing --w etc.


def test_parser_accepts_minimal_valid_args():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS)
    assert args.w == pytest.approx(0.002)
    assert args.lc == "none"
    assert args.n_modes == 1  # default -- physically correct for a plain isotropic line
    assert args.f_stop is None


def test_direction_parses_valid_string():
    vec = _direction("0,1,0.3")
    assert vec == pytest.approx(np.array([0.0, 1.0, 0.3]))


def test_direction_rejects_wrong_component_count():
    with pytest.raises(Exception):  # argparse.ArgumentTypeError
        _direction("0,1")


def test_direction_rejects_zero_vector():
    with pytest.raises(Exception):
        _direction("0,0,0")


# --- material assembly ---


def _params(**overrides):
    kwargs = dict(
        w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
        eps_r_substrate=3.0, reference_frequency=25e9,
    )
    kwargs.update(overrides)
    return GeometryParams(**kwargs)


def test_build_lc_material_none_gives_constant_material_matching_substrate():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS)  # --lc defaults to "none"
    material = build_lc_material(args, _params())
    assert isinstance(material, ConstantMaterial)
    eps = material.epsilon(np.array([[0.01, 0.005, 0.001]]))
    assert eps[0, 0, 0] == pytest.approx(3.0)


def test_build_lc_material_uniform_gives_director_field_material():
    parser = build_arg_parser()
    args = parser.parse_args(
        _REQUIRED_GEOM_ARGS + ["--lc", "uniform", "--lc-direction", "0,1,0.3", "--eps-perp", "2.5", "--eps-parallel", "3.0"]
    )
    material = build_lc_material(args, _params())
    assert isinstance(material, DirectorFieldMaterial)
    # Sample at the LC region's own centroid -- should reproduce the
    # normalized direction's tensor (Section 4.1's eps = eps_perp*I +
    # (eps_par-eps_perp)*n n^T) since every corner carries the same n.
    eps = material.epsilon(np.array([[0.010, 0.005, 0.001]]))[0]
    n_hat = np.array([0.0, 1.0, 0.3]) / np.linalg.norm([0.0, 1.0, 0.3])
    expected = 2.5 * np.eye(3) + 0.5 * np.outer(n_hat, n_hat)
    assert eps.real == pytest.approx(expected, abs=1e-6)


def test_build_lc_material_uniform_requires_direction():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS + ["--lc", "uniform", "--eps-perp", "2.5", "--eps-parallel", "3.0"])
    with pytest.raises(CLIError):
        build_lc_material(args, _params())


def test_build_lc_material_requires_eps_perp_and_parallel():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS + ["--lc", "uniform", "--lc-direction", "0,1,0.3"])
    with pytest.raises(CLIError):
        build_lc_material(args, _params())


def test_build_lc_material_file_requires_path():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS + ["--lc", "file", "--eps-perp", "2.5", "--eps-parallel", "3.0"])
    with pytest.raises(CLIError):
        build_lc_material(args, _params())


# --- output formatting ---


def test_format_csv_structure_and_no_numpy_repr_leakage():
    omega = 2 * np.pi * 25e9
    S = {
        ("PORT_1", 1, "PORT_1"): np.complex128(-0.4 + 0.1j),
        ("PORT_1", 1, "PORT_2"): np.complex128(0.3 - 0.2j),
        ("PORT_2", 1, "PORT_1"): np.complex128(0.3 - 0.2j),
        ("PORT_2", 1, "PORT_2"): np.complex128(-0.35 + 0.05j),
        ("PORT_1", 2, "PORT_1"): np.complex128(-0.01j),
        ("PORT_1", 2, "PORT_2"): np.complex128(0.0),
        ("PORT_2", 2, "PORT_1"): np.complex128(0.0),
        ("PORT_2", 2, "PORT_2"): np.complex128(0.0),
    }
    csv_text = _format_csv([omega], ["PORT_1", "PORT_2"], [S], n_modes=2)
    lines = csv_text.strip().split("\n")
    assert len(lines) == 2  # header + one data row
    assert lines[0].split(",")[0] == "frequency_Hz"
    assert "np.float64" not in csv_text
    # every field beyond the header must parse as a plain float
    for field in lines[1].split(","):
        float(field)


def test_format_csv_missing_entries_default_to_zero():
    omega = 2 * np.pi * 25e9
    csv_text = _format_csv([omega], ["PORT_1", "PORT_2"], [{}], n_modes=1)
    values = [float(v) for v in csv_text.strip().split("\n")[1].split(",")[1:]]
    assert all(v == 0.0 for v in values)


# --- end-to-end (mirrors examples/isotropic_microstrip.py) ---


def test_main_runs_isotropic_case_end_to_end(capsys):
    pytest.importorskip("gmsh")
    exit_code = main(
        [
            "--w", "0.002", "--L", "0.020", "--L-lc", "0.008", "--W-lc", "0.004",
            "--h-sub", "0.002", "--W-sub", "0.010", "--eps-r-substrate", "3.0",
            "--mesh-density", "6", "--pml-r0", "1.0", "--pml-kappa-max", "1.0",
            "--f-start", "25e9", "--f-points", "1", "--n-modes", "2",
            "--lc", "none", "--quiet",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    lines = out.strip().split("\n")
    assert len(lines) == 2
    assert "np.float64" not in out


def test_main_reports_clean_error_on_bad_geometry(capsys):
    exit_code = main(
        [
            # w > W_lc violates Section 6's containment check -- an
            # ill-posed input, not a bug.
            "--w", "0.01", "--L", "0.020", "--L-lc", "0.008", "--W-lc", "0.004",
            "--h-sub", "0.002", "--W-sub", "0.010", "--eps-r-substrate", "3.0",
            "--f-start", "25e9", "--quiet",
        ]
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Error:" in err


# --- plotting ---


def test_parser_accepts_plot_flags():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS + ["--plot", "--plot-output", "out.png"])
    assert args.plot is True
    assert str(args.plot_output) == "out.png"


def test_parser_plot_defaults_are_off():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS)
    assert args.plot is False
    assert args.plot_output is None


def _synthetic_S_by_freq(n_points: int) -> list[dict]:
    return [
        {
            ("PORT_1", 1, "PORT_1"): -0.4 + 0.1j, ("PORT_1", 1, "PORT_2"): 0.3 - 0.2j,
            ("PORT_2", 1, "PORT_1"): 0.3 - 0.2j, ("PORT_2", 1, "PORT_2"): -0.35 + 0.05j,
            ("PORT_1", 2, "PORT_1"): -0.01j, ("PORT_1", 2, "PORT_2"): 0.0,
            ("PORT_2", 2, "PORT_1"): 0.0, ("PORT_2", 2, "PORT_2"): 0.0,
        }
        for _ in range(n_points)
    ]


def test_plot_results_multi_point_creates_two_subplots_for_n_modes_2():
    fig = plot_results([24e9, 25e9, 26e9], ["PORT_1", "PORT_2"], _synthetic_S_by_freq(3), n_modes=2, show=False)
    assert len(fig.axes) == 2  # dominant + conversion, per plot_results' own docstring


def test_plot_results_single_point_renders_without_error():
    fig = plot_results([25e9], ["PORT_1", "PORT_2"], _synthetic_S_by_freq(1), n_modes=2, show=False)
    assert len(fig.axes) == 2


def test_plot_results_n_modes_1_has_only_dominant_subplot():
    fig = plot_results([25e9], ["PORT_1", "PORT_2"], _synthetic_S_by_freq(1), n_modes=1, show=False)
    assert len(fig.axes) == 1


def test_plot_results_writes_output_file(tmp_path):
    out = tmp_path / "plot.png"
    plot_results([25e9], ["PORT_1", "PORT_2"], _synthetic_S_by_freq(1), n_modes=2, show=False, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_results_raises_clean_error_when_matplotlib_unavailable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", None)
    with pytest.raises(PlottingUnavailableError):
        plot_results([25e9], ["PORT_1", "PORT_2"], _synthetic_S_by_freq(1), n_modes=1, show=False)


def test_main_with_plot_output_writes_both_csv_and_plot(tmp_path, capsys):
    pytest.importorskip("gmsh")
    plot_path = tmp_path / "sweep.png"
    exit_code = main(
        [
            "--w", "0.002", "--L", "0.020", "--L-lc", "0.008", "--W-lc", "0.004",
            "--h-sub", "0.002", "--W-sub", "0.010", "--eps-r-substrate", "3.0",
            "--mesh-density", "6", "--pml-r0", "1.0", "--pml-kappa-max", "1.0",
            "--f-start", "25e9", "--f-points", "1", "--n-modes", "2",
            "--lc", "none", "--quiet", "--plot-output", str(plot_path),
        ]
    )
    assert exit_code == 0
    assert plot_path.exists()
    out = capsys.readouterr().out
    assert out.strip().split("\n")[0].startswith("frequency_Hz")


# --- geometry/mesh visualization ---


def test_parser_accepts_geometry_and_mesh_visualization_flags():
    parser = build_arg_parser()
    args = parser.parse_args(
        _REQUIRED_GEOM_ARGS
        + ["--show-geometry", "--geometry-output", "geo.png", "--show-mesh", "--mesh-output", "mesh.png", "--geometry-only"]
    )
    assert args.show_geometry is True
    assert str(args.geometry_output) == "geo.png"
    assert args.show_mesh is True
    assert str(args.mesh_output) == "mesh.png"
    assert args.geometry_only is True


def test_parser_geometry_and_mesh_visualization_defaults_are_off():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS)
    assert args.show_geometry is False
    assert args.geometry_output is None
    assert args.show_mesh is False
    assert args.mesh_output is None
    assert args.geometry_only is False


def test_main_geometry_only_writes_plots_and_skips_the_sweep(tmp_path, capsys):
    pytest.importorskip("gmsh")
    geometry_path = tmp_path / "geometry.png"
    mesh_path = tmp_path / "mesh.png"
    exit_code = main(
        [
            "--w", "0.002", "--L", "0.020", "--L-lc", "0.008", "--W-lc", "0.004",
            "--h-sub", "0.002", "--W-sub", "0.010", "--eps-r-substrate", "3.0",
            "--mesh-density", "6", "--f-start", "25e9", "--quiet",
            "--geometry-output", str(geometry_path), "--mesh-output", str(mesh_path),
            "--geometry-only",
        ]
    )
    assert exit_code == 0
    assert geometry_path.exists() and geometry_path.stat().st_size > 0
    assert mesh_path.exists() and mesh_path.stat().st_size > 0
    # --geometry-only must return before the sweep/CSV-formatting stage.
    assert "frequency_Hz" not in capsys.readouterr().out


def test_main_show_geometry_alongside_a_full_sweep_still_writes_csv(tmp_path, capsys):
    pytest.importorskip("gmsh")
    geometry_path = tmp_path / "geometry.png"
    exit_code = main(
        [
            "--w", "0.002", "--L", "0.020", "--L-lc", "0.008", "--W-lc", "0.004",
            "--h-sub", "0.002", "--W-sub", "0.010", "--eps-r-substrate", "3.0",
            "--mesh-density", "6", "--pml-r0", "1.0", "--pml-kappa-max", "1.0",
            "--f-start", "25e9", "--f-points", "1", "--n-modes", "2",
            "--lc", "none", "--quiet", "--geometry-output", str(geometry_path),
        ]
    )
    assert exit_code == 0
    assert geometry_path.exists() and geometry_path.stat().st_size > 0
    out = capsys.readouterr().out
    assert out.strip().split("\n")[0].startswith("frequency_Hz")


# --- port aperture (--w-port/--h-port) ---


def test_parser_accepts_w_port_h_port():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS + ["--w-port", "0.006", "--h-port", "0.005"])
    assert args.w_port == pytest.approx(0.006)
    assert args.h_port == pytest.approx(0.005)


def test_parser_w_port_h_port_default_to_none():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS)
    assert args.w_port is None
    assert args.h_port is None


def test_main_plumbs_w_port_h_port_into_geometry_params(monkeypatch):
    """Confirms the CLI arg -> GeometryParams wiring directly (no mesh
    build needed): patch the `GeometryBuilder` name `cli.main` actually
    resolves (bound into `cli`'s own namespace at import time, so patching
    the `geometry_builder` package's attribute would not be seen here) to
    capture the params it was called with, then let main() fail
    immediately afterward -- cheap and doesn't need Gmsh."""
    import cli as cli_module

    captured = {}

    class _FakeBuilder:
        def build(self, params):
            captured["params"] = params
            raise RuntimeError("stop here -- only the plumbing is under test")

    monkeypatch.setattr(cli_module, "GeometryBuilder", _FakeBuilder)
    with pytest.raises(RuntimeError, match="stop here"):
        main(_REQUIRED_GEOM_ARGS + ["--w-port", "0.006", "--h-port", "0.005", "--quiet"])
    assert captured["params"].W_port == pytest.approx(0.006)
    assert captured["params"].H_port == pytest.approx(0.005)


def test_main_reports_clean_error_when_w_port_given_without_h_port():
    exit_code = main(_REQUIRED_GEOM_ARGS + ["--w-port", "0.006", "--quiet"])
    assert exit_code == 1


def test_main_restricted_aperture_end_to_end(capsys):
    """A restricted-aperture end-to-end run at this file's original
    W_sub=10mm/h_sub=2mm/25 GHz geometry (independent of the current
    examples/isotropic_microstrip.py numbers, which since moved to a
    thinner K-band-valid substrate -- see test_main_k_band_single_mode_
    end_to_end below for a test that mirrors that example directly)."""
    pytest.importorskip("gmsh")
    exit_code = main(
        [
            "--w", "0.002", "--L", "0.020", "--L-lc", "0.008", "--W-lc", "0.004",
            "--h-sub", "0.002", "--W-sub", "0.010", "--eps-r-substrate", "3.0",
            "--mesh-density", "6", "--pml-r0", "1.0", "--pml-kappa-max", "1.0",
            "--f-start", "25e9", "--f-points", "1", "--n-modes", "1",
            "--lc", "none", "--w-port", "0.006", "--h-port", "0.006", "--quiet",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    lines = out.strip().split("\n")
    assert len(lines) == 2
    assert "np.float64" not in out


def test_main_k_band_single_mode_end_to_end(capsys):
    """Mirrors examples/isotropic_microstrip.py exactly: a thin,
    single-mode-valid substrate with a box-mode-safe port aperture
    (lambda_min/2=8.66mm at 10 GHz/eps_r=3, comfortably above the 8mm x
    6mm aperture) and the CLI's own default `--n-modes 1` -- the
    single-mode-tolerant mode-counting fix (`PortModeSolver.solve`'s
    n_modes/n_desired split) is what makes `--n-modes 1` succeed here
    instead of hard-failing on "only 1 mode found, requested 3"."""
    pytest.importorskip("gmsh")
    exit_code = main(
        [
            "--w", "0.00334", "--L", "0.020", "--L-lc", "0.008", "--W-lc", "0.004",
            "--h-sub", "0.0015", "--W-sub", "0.010", "--eps-r-substrate", "3.0",
            "--mesh-density", "10", "--pml-r0", "1.0", "--pml-kappa-max", "1.0",
            "--f-start", "10e9", "--f-points", "1",
            "--lc", "none", "--w-port", "0.008", "--h-port", "0.006", "--quiet",
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    lines = out.strip().split("\n")
    assert len(lines) == 2
    assert "np.float64" not in out


# --- field visualization (--plot-port-field / --plot-field-slice) ---

_K_BAND_ARGS = [
    "--w", "0.00334", "--L", "0.020", "--L-lc", "0.008", "--W-lc", "0.004",
    "--h-sub", "0.0015", "--W-sub", "0.010", "--eps-r-substrate", "3.0",
    "--mesh-density", "10", "--pml-r0", "1.0", "--pml-kappa-max", "1.0",
    "--f-start", "10e9", "--f-points", "1",
    "--lc", "none", "--w-port", "0.008", "--h-port", "0.006", "--quiet",
]


def test_parser_accepts_plot_port_field_bare_and_with_tag():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS + ["--plot-port-field"])
    assert args.plot_port_field == ["__ALL__"]

    args = parser.parse_args(_REQUIRED_GEOM_ARGS + ["--plot-port-field", "PORT_1", "--plot-port-field", "PORT_2"])
    assert args.plot_port_field == ["PORT_1", "PORT_2"]


def test_parser_plot_port_field_defaults_to_none():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS)
    assert args.plot_port_field is None
    assert args.port_field_style == "mag"
    assert args.port_field_component == "E"


def test_parser_accepts_plot_field_slice():
    parser = build_arg_parser()
    args = parser.parse_args(_REQUIRED_GEOM_ARGS + ["--plot-field-slice", "x=0.01"])
    assert args.plot_field_slice == ("x", pytest.approx(0.01))
    assert args.slice_grid == 120
    assert args.slice_field == "E"


def test_parser_rejects_malformed_plot_field_slice():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_REQUIRED_GEOM_ARGS + ["--plot-field-slice", "bogus"])
    with pytest.raises(SystemExit):
        parser.parse_args(_REQUIRED_GEOM_ARGS + ["--plot-field-slice", "q=0.01"])


def test_main_plot_port_field_unknown_port_reports_clean_error():
    exit_code = main(_K_BAND_ARGS + ["--plot-port-field", "PORT_99"])
    assert exit_code == 1


def test_main_slice_excitation_unknown_port_reports_clean_error():
    exit_code = main(_K_BAND_ARGS + ["--plot-field-slice", "x=0.01", "--slice-excitation", "PORT_99"])
    assert exit_code == 1


def test_main_plot_port_field_all_ports_end_to_end(tmp_path, capsys):
    pytest.importorskip("gmsh")
    out_path = tmp_path / "port_field.png"
    exit_code = main(
        _K_BAND_ARGS + ["--plot-port-field", "--port-field-style", "quiver", "--port-field-output", str(out_path)]
    )
    assert exit_code == 0
    assert out_path.exists() and out_path.stat().st_size > 0
    out = capsys.readouterr().out
    assert out.strip().split("\n")[0].startswith("frequency_Hz")


def test_main_plot_field_slice_end_to_end(tmp_path, capsys):
    pytest.importorskip("gmsh")
    out_path = tmp_path / "slice.png"
    exit_code = main(
        _K_BAND_ARGS
        + [
            "--plot-field-slice", "x=0.01", "--slice-grid", "24", "--slice-excitation", "PORT_1",
            "--field-slice-output", str(out_path),
        ]
    )
    assert exit_code == 0
    assert out_path.exists() and out_path.stat().st_size > 0
    out = capsys.readouterr().out
    assert out.strip().split("\n")[0].startswith("frequency_Hz")


def test_main_plot_port_field_and_field_slice_together(tmp_path):
    """Both new features combined in a single invocation, alongside the
    pre-existing --show-geometry/--plot -- a non-regression check that
    they don't interfere with each other or with the existing flags."""
    pytest.importorskip("gmsh")
    port_out = tmp_path / "port_field.png"
    slice_out = tmp_path / "slice.png"
    geom_out = tmp_path / "geometry.png"
    plot_out = tmp_path / "splot.png"
    exit_code = main(
        _K_BAND_ARGS
        + [
            "--geometry-output", str(geom_out),
            "--plot-output", str(plot_out),
            "--plot-port-field", "--port-field-output", str(port_out),
            "--plot-field-slice", "x=0.01", "--slice-grid", "20", "--field-slice-output", str(slice_out),
        ]
    )
    assert exit_code == 0
    for path in (port_out, slice_out, geom_out, plot_out):
        assert path.exists() and path.stat().st_size > 0
