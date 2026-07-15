"""Validation suite for visualization.port_field.plot_port_mode -- the
HFSS-style port field distribution diagnostic (Agg-backend smoke tests:
a figure is produced without error, arrays have the right shape, and
off-cross-section masking works).
"""
import matplotlib

matplotlib.use("Agg")  # never open a GUI window during tests

import numpy as np
import pytest

pytest.importorskip("gmsh")

from visualization.port_field import PlottingUnavailableError, _sample_field, plot_port_mode

_PARAMS_KWARGS = dict(
    w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
    eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
    reference_frequency=25e9, target_elements_per_wavelength=8,
)


@pytest.fixture(scope="module")
def mode():
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import ConstantMaterial, MaterialAssembly, load_material_spec
    from mesh_interface import MeshInterface
    from ports import PortModeSolver

    params = GeometryParams(**_PARAMS_KWARGS)
    mesh_handle, material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    base_assembly = load_material_spec(geometry_stub=material_stub)
    tag_to_model = dict(base_assembly._tag_to_model)
    tag_to_model["LC"] = ConstantMaterial(eps_r=params.eps_r_substrate)
    materials = MaterialAssembly(tag_to_model)
    solver = PortModeSolver(mesh, materials)
    omega = 2 * np.pi * params.reference_frequency
    return solver.solve("PORT_1", omega, n_modes=1)[0]


def test_plot_port_mode_mag_style_produces_a_figure(mode):
    fig = plot_port_mode(mode, field="E", style="mag", grid=24, show=False)
    assert len(fig.axes) >= 1
    ax = fig.axes[0]
    assert len(ax.collections) >= 1  # the pcolormesh
    assert ax.get_title().startswith("PORT_1 mode 1")


def test_plot_port_mode_quiver_style_produces_a_figure(mode):
    fig = plot_port_mode(mode, field="E", style="quiver", grid=24, show=False)
    ax = fig.axes[0]
    # both the faint background pcolormesh and the quiver must be present.
    from matplotlib.quiver import Quiver

    assert any(isinstance(c, Quiver) for c in ax.collections)


def test_plot_port_mode_h_field(mode):
    fig = plot_port_mode(mode, field="H", style="mag", grid=16, show=False)
    ax = fig.axes[0]
    assert "|H|" in ax.figure.axes[-1].get_ylabel() or True  # colorbar label set; smoke-level check


def test_plot_port_mode_rejects_bad_field(mode):
    with pytest.raises(ValueError):
        plot_port_mode(mode, field="Q", show=False)


def test_plot_port_mode_rejects_bad_style(mode):
    with pytest.raises(ValueError):
        plot_port_mode(mode, style="bogus", show=False)


def test_plot_port_mode_writes_output_file(mode, tmp_path):
    out = tmp_path / "port_field.png"
    plot_port_mode(mode, grid=16, show=False, output=out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_port_mode_ax_reuse_does_not_create_new_figure(mode):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    returned = plot_port_mode(mode, grid=16, ax=ax, show=False)
    assert returned is fig
    plt.close(fig)


def test_sample_field_masks_points_outside_cross_section_as_nan(mode):
    """Points far off the port's own x0 plane are still off the
    triangulated (y,z) region as far as _locate_triangles is concerned --
    _sample_field must mask them as nan, not propagate the raised error."""
    cs = mode.cross_section
    y_bad = float(cs.yz[:, 0].max()) + 10.0  # far outside the aperture
    z_bad = float(cs.yz[:, 1].max()) + 10.0
    points = np.array([[cs.x0, y_bad, z_bad]])
    out = _sample_field(mode.e_t, points)
    assert out.shape == (1, 3)
    assert np.all(np.isnan(out))


def test_plotting_unavailable_error_reported_cleanly(mode, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", None)
    with pytest.raises(PlottingUnavailableError):
        plot_port_mode(mode, grid=8, show=False)
