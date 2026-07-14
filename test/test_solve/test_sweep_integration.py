"""Integration suite for solve.sweep.run_sweep (docs/module6_solve_sweep_equations.md
Sections 3, 6-7, 9), on the real geometry_builder -> mesh_interface -> material
-> fem -> ports -> pml pipeline. The doc's "big one" (end-to-end Phase 1
reciprocity/passivity gate, Section 9) needs Module 7's S-parameter
extraction, which does not exist yet -- explicitly out of scope here, same
posture Module 5 took toward its own Section 5.1 reflection test.
"""
import numpy as np
import pytest

from material import ConstantMaterial

_PARAMS_KWARGS = dict(
    w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
    eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
    reference_frequency=25e9, target_elements_per_wavelength=6,
)


@pytest.fixture(scope="module")
def mesh_and_materials():
    pytest.importorskip("gmsh")
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import MaterialAssembly, load_material_spec
    from mesh_interface import MeshInterface

    params = GeometryParams(**_PARAMS_KWARGS)
    mesh_handle, material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    base_assembly = load_material_spec(geometry_stub=material_stub)
    tag_to_model = dict(base_assembly._tag_to_model)
    tag_to_model["LC"] = ConstantMaterial(eps_r=params.eps_r_substrate)
    materials = MaterialAssembly(tag_to_model)
    return mesh, materials, params


@pytest.fixture(scope="module")
def pml_params(mesh_and_materials):
    """Trivial (R0=1 => sigma_max=0, kappa_max=1 => Lambda=I exactly) PML
    parameters. This test suite validates the sweep's *mechanics* (system
    assembly, DOF elimination, port tracking, congruence) -- PML
    absorption quality is Module 5's own concern, already validated there.
    A realistic, steeply-graded PML profile needs several elements across
    the PML's own thickness to be resolvable by the adaptive quadrature
    (Module 5 build guidance); this coarse test mesh (chosen for CI speed,
    Section 8 build step elsewhere in this suite) does not have that, so a
    non-trivial profile would fail on mesh-resolution grounds unrelated to
    anything Module 6 is responsible for."""
    _mesh, _materials, params = mesh_and_materials
    return dict(
        background=ConstantMaterial(eps_r=1.0),
        z_air_top=params.h_sub + params.h_air,
        thickness=params.h_pml,
        R0=1.0,
        n=2,
        kappa_max=1.0,
    )


@pytest.fixture(scope="module")
def sweep_results(mesh_and_materials, pml_params):
    from solve import run_sweep

    mesh, materials, params = mesh_and_materials
    frequencies = [2 * np.pi * params.reference_frequency]
    return run_sweep(mesh, materials, ["PORT_1", "PORT_2"], frequencies, n_modes=1, pml_params=pml_params)


def test_sweep_produces_one_result_per_port_excitation(sweep_results):
    assert len(sweep_results) == 2
    excitations = {r.excitation for r in sweep_results}
    assert excitations == {("PORT_1", 1), ("PORT_2", 1)}


def test_solution_vector_has_full_unconstrained_edge_dimension(mesh_and_materials, sweep_results):
    mesh, _materials, _params = mesh_and_materials
    for r in sweep_results:
        assert r.a.shape == (mesh.n_edges,)


def test_pec_dofs_are_exactly_zero_in_the_recovered_solution(mesh_and_materials, sweep_results):
    mesh, _materials, _params = mesh_and_materials
    pec_dofs = mesh.pec_edge_dofs()
    for r in sweep_results:
        for dof in pec_dofs:
            assert r.a[dof] == 0.0


def test_solution_is_finite_and_nontrivial(sweep_results):
    for r in sweep_results:
        assert np.all(np.isfinite(r.a))
        assert np.linalg.norm(r.a) > 0.0


# --- Section 6.3: port congruence check (geometrically identical cross-sections) ---


def test_port_congruence_gamma_and_y_agree_closely(sweep_results):
    r = sweep_results[0]
    modes1 = r.port_modes["PORT_1"]
    modes2 = r.port_modes["PORT_2"]
    assert len(modes1) == len(modes2)
    for m1, m2 in zip(modes1, modes2):
        assert m1.gamma == pytest.approx(m2.gamma, rel=1e-3)
        assert m1.Y == pytest.approx(m2.Y, rel=1e-3)


# --- Section 3: per-frequency PML re-assembly, exercised through run_sweep ---


def test_trivial_pml_through_run_sweep_matches_air_on_pml_tets(mesh_and_materials):
    """Section 8 build step 4: 'confirm sigma->0 reproduces the no-PML
    case' -- exercised end-to-end through run_sweep's own PML re-assembly
    step (Section 3), not re-derived from scratch (Module 5's own test
    suite already covers the underlying PMLMaterial reduction at the
    element-matrix level)."""
    from fem import assemble
    from material import MaterialAssembly
    from pml import PMLMaterial
    from solve.sweep import _interior_pml_tets

    mesh, materials, params = mesh_and_materials
    _interior, pml_tets = _interior_pml_tets(mesh)
    omega = 2 * np.pi * params.reference_frequency

    trivial_pml = PMLMaterial(
        ConstantMaterial(eps_r=1.0), omega,
        z_air_top=params.h_sub + params.h_air, thickness=params.h_pml,
        R0=1.0, n=2, kappa_max=1.0,
    )
    K_pml, M_pml = assemble(mesh, MaterialAssembly({"PML_TOP": trivial_pml}), tet_subset=pml_tets)
    K_air, M_air = assemble(mesh, MaterialAssembly({"PML_TOP": ConstantMaterial(eps_r=1.0)}), tet_subset=pml_tets)

    assert abs(K_pml - K_air).max() < 1e-9
    assert abs(M_pml - M_air).max() < 1e-9


# --- Section 5.3: multi-RHS reuse, exercised at full-mesh scale ---


def test_both_excitations_satisfy_the_full_assembled_system(mesh_and_materials, sweep_results, pml_params):
    """Both port excitations at the sweep's one frequency are solved
    against the same cached factorization (Section 5.3) -- reconstruct
    the full system A(omega) independently (interior + PML + B_p, using
    the SAME tracked port_modes each SweepResult already carries, so this
    doesn't re-solve/re-track) and confirm each `a` satisfies `A a = g` on
    the free DOFs to solver tolerance. A stale-factorization or wrong-RHS
    bug would show up here as a residual far above solver tolerance."""
    from fem import assemble
    from ports import build_B, build_g
    from pml import PMLMaterial
    from material import MaterialAssembly
    from scipy import constants as sp_c
    from solve.sweep import _interior_pml_tets

    mesh, materials, params = mesh_and_materials
    interior, pml_tets = _interior_pml_tets(mesh)
    K_int, M_int = assemble(mesh, materials, tet_subset=interior)
    pec_dofs = mesh.pec_edge_dofs()
    free = np.array([i for i in range(mesh.n_edges) if i not in pec_dofs])

    for r in sweep_results:
        omega = r.omega
        k0 = omega * np.sqrt(sp_c.mu_0 * sp_c.epsilon_0)
        pml_material = PMLMaterial(omega=omega, **pml_params)
        K_pml, M_pml = assemble(mesh, MaterialAssembly({"PML_TOP": pml_material}), tet_subset=pml_tets)
        B = build_B(r.port_modes, mesh, omega)
        A = (K_int + K_pml) - (k0**2) * (M_int + M_pml) + B
        g = build_g(r.port_modes, {r.excitation: 1.0 + 0j}, mesh, omega)

        residual = (A @ r.a - g)[free]
        scale = max(1.0, float(np.abs(g).max()))
        assert float(np.abs(residual).max()) < 1e-6 * scale
