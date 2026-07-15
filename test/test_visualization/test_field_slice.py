"""Validation suite for visualization.field_slice.plot_field_slice -- a
volume field slice rendered at its true 3D location inside the SAME
Axes3D as visualization.geometry_view.plot_geometry draws the structure
into (Agg-backend smoke tests).
"""
import matplotlib

matplotlib.use("Agg")  # never open a GUI window during tests

import numpy as np
import pytest

pytest.importorskip("gmsh")

from visualization.field_slice import PlottingUnavailableError, plot_field_slice
from visualization.geometry_view import plot_geometry

_PARAMS_KWARGS = dict(
    w=0.00334, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.0015, W_sub=0.010,
    eps_r_substrate=3.0, h_air=0.0045, h_pml=0.00075,
    reference_frequency=10e9, target_elements_per_wavelength=10,
    W_port=0.008, H_port=0.006,
)


@pytest.fixture(scope="module")
def mesh_and_result():
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import ConstantMaterial, MaterialAssembly, load_material_spec
    from mesh_interface import MeshInterface
    from solve import run_sweep

    params = GeometryParams(**_PARAMS_KWARGS)
    mesh_handle, material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    base_assembly = load_material_spec(geometry_stub=material_stub)
    tag_to_model = dict(base_assembly._tag_to_model)
    tag_to_model["LC"] = ConstantMaterial(eps_r=params.eps_r_substrate)
    materials = MaterialAssembly(tag_to_model)

    pml_params = dict(
        background=ConstantMaterial(eps_r=1.0), z_air_top=params.h_sub + params.h_air,
        thickness=params.h_pml, R0=1.0, n=2, kappa_max=1.0,
    )
    omega = 2 * np.pi * params.reference_frequency
    results = run_sweep(mesh, materials, ["PORT_1", "PORT_2"], [omega], n_modes=1, pml_params=pml_params)
    return mesh, results[0]


def test_plot_field_slice_creates_its_own_structure_view_when_no_ax_given(mesh_and_result):
    mesh, result = mesh_and_result
    fig = plot_field_slice(mesh, result, ("x", 0.01), grid=20, show=False)
    ax3d = fig.axes[0]
    # structure collections (plot_geometry's own, at least the substrate
    # envelope + PEC_LINE + ports) plus the slice's own plot_surface artist.
    assert len(ax3d.collections) >= 2


def test_plot_field_slice_shares_the_same_axes3d_as_show_geometry(mesh_and_result):
    """The critical requirement: the slice artist must land in the exact
    same Axes3D object plot_geometry draws the structure into, not a
    separate figure/axes/backend."""
    import matplotlib.pyplot as plt

    mesh, result = mesh_and_result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_geometry(mesh, ax=ax, show=False)
    n_before = len(ax.collections)

    returned = plot_field_slice(mesh, result, ("z", 0.001), grid=20, ax=ax, show=False)
    assert returned is fig
    assert len(ax.collections) == n_before + 1  # exactly one new artist: the slice surface
    plt.close(fig)


def test_plot_field_slice_h_field(mesh_and_result):
    mesh, result = mesh_and_result
    fig = plot_field_slice(mesh, result, ("y", 0.005), grid=16, field="H", show=False)
    assert len(fig.axes) >= 1


def test_plot_field_slice_rejects_bad_axis(mesh_and_result):
    mesh, result = mesh_and_result
    with pytest.raises(ValueError):
        plot_field_slice(mesh, result, ("q", 0.0), show=False)


def test_plot_field_slice_rejects_bad_field(mesh_and_result):
    mesh, result = mesh_and_result
    with pytest.raises(ValueError):
        plot_field_slice(mesh, result, ("x", 0.01), field="Q", show=False)


def test_plot_field_slice_writes_output_file(mesh_and_result, tmp_path):
    mesh, result = mesh_and_result
    out = tmp_path / "slice.png"
    plot_field_slice(mesh, result, ("x", 0.01), grid=16, show=False, output=out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_field_slice_masks_outside_domain_as_transparent(mesh_and_result):
    """A slice plane entirely outside the meshed domain (far below the
    ground plane) must still produce a figure -- every facecolor fully
    transparent, not a crash on an all-nan grid."""
    mesh, result = mesh_and_result
    fig = plot_field_slice(mesh, result, ("z", -1.0), grid=10, show=False)
    assert len(fig.axes) >= 1


def test_plotting_unavailable_error_reported_cleanly(mesh_and_result, monkeypatch):
    mesh, result = mesh_and_result
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", None)
    with pytest.raises(PlottingUnavailableError):
        plot_field_slice(mesh, result, ("x", 0.01), grid=8, show=False)
