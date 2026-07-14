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
    assert args.n_modes == 2  # default
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
