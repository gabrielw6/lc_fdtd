"""Validation suite for PortModeSolver.solve's n_modes/n_desired split
(docs/module4_ports_equations.md Section 3.7's "third mitigation",
single-mode-tolerant mode counting, port-aperture-decoupling review Part
B). Uses a deliberately undersized/under-resolved port aperture (3mm x
3mm at mesh-density 6, 25 GHz) -- hand-verified to yield exactly one
power-normalizable mode regardless of how many are requested, the exact
scenario ("only 1 power-normalizable mode found, requested 4") that used
to hard-fail the whole port solve before this fix.
"""
import numpy as np
import pytest

pytest.importorskip("gmsh")

from ports.mode_solver import PortModeError, PortModeSolver

_PARAMS_KWARGS = dict(
    w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
    eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
    reference_frequency=25e9, target_elements_per_wavelength=6,
    W_port=0.003, H_port=0.003,
)


@pytest.fixture(scope="module")
def solver():
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import ConstantMaterial, MaterialAssembly, load_material_spec
    from mesh_interface import MeshInterface

    params = GeometryParams(**_PARAMS_KWARGS)
    mesh_handle, material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    base_assembly = load_material_spec(geometry_stub=material_stub)
    tag_to_model = dict(base_assembly._tag_to_model)
    tag_to_model["LC"] = ConstantMaterial(eps_r=params.eps_r_substrate)
    materials = MaterialAssembly(tag_to_model)
    return PortModeSolver(mesh, materials)


@pytest.fixture(scope="module")
def omega():
    return 2 * np.pi * _PARAMS_KWARGS["reference_frequency"]


def test_this_fixture_is_genuinely_single_mode_at_high_oversupply(solver, omega):
    """Confirms the fixture's own premise before testing the contract
    against it: however many are desired, only 1 is ever found."""
    modes = solver.solve("PORT_1", omega, n_modes=1, n_desired=8)
    assert len(modes) == 1


@pytest.mark.parametrize("n_desired", [1, 2, 4, None])
def test_solve_returns_fewer_than_desired_without_raising_when_required_is_met(solver, omega, n_desired):
    modes = solver.solve("PORT_1", omega, n_modes=1, n_desired=n_desired)
    assert len(modes) == 1


def test_solve_raises_when_required_minimum_exceeds_what_exists(solver, omega):
    with pytest.raises(PortModeError, match="required at least 2"):
        solver.solve("PORT_1", omega, n_modes=2, n_desired=4)


def test_solve_without_n_desired_preserves_original_exact_match_semantics(solver, omega):
    """Omitting n_desired must behave exactly as it did before this split
    existed: n_modes is both the required minimum and the desired count."""
    with pytest.raises(PortModeError):
        solver.solve("PORT_1", omega, n_modes=2)


def test_n_desired_less_than_n_modes_is_clamped_up_to_n_modes(solver, omega):
    """n_desired < n_modes would be a nonsensical request (desiring fewer
    than required) -- solve() clamps rather than silently under-collecting."""
    with pytest.raises(PortModeError, match="required at least 2"):
        solver.solve("PORT_1", omega, n_modes=2, n_desired=1)
