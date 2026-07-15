"""Validation suite for ports.port_operator (docs/module4_ports_equations.md
Sections 5-6).
"""
import numpy as np
import pytest

from mesh_interface import quadrature as mesh_quadrature
from ports.mode_solver import _e_t_on_triangle, _h_t_on_triangle
from ports.port_operator import PortOperatorError, build_B, build_g, deembed

_PARAMS_KWARGS = dict(
    w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
    eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
    reference_frequency=25e9, target_elements_per_wavelength=8,
)


@pytest.fixture(scope="module")
def mesh_and_solver():
    pytest.importorskip("gmsh")
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import ConstantMaterial, MaterialAssembly, load_material_spec
    from mesh_interface import MeshInterface
    from ports.mode_solver import PortModeSolver

    params = GeometryParams(**_PARAMS_KWARGS)
    mesh_handle, material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    base_assembly = load_material_spec(geometry_stub=material_stub)
    tag_to_model = dict(base_assembly._tag_to_model)
    tag_to_model["LC"] = ConstantMaterial(eps_r=params.eps_r_substrate)
    materials = MaterialAssembly(tag_to_model)
    return mesh, PortModeSolver(mesh, materials), 2 * np.pi * params.reference_frequency


@pytest.fixture(scope="module")
def port_modes(mesh_and_solver):
    _mesh, solver, omega = mesh_and_solver
    return {
        "PORT_1": solver.solve("PORT_1", omega, n_modes=2),
        "PORT_2": solver.solve("PORT_2", omega, n_modes=2),
    }


# --- Section 5.2: the closed-form cached overlaps must match brute-force quadrature ---


def test_overlap_h_closed_form_matches_direct_quadrature(port_modes):
    """`_mode_overlaps`'s `overlap_h` is derived algebraically (not by
    fresh quadrature) from S_tz/T_tt -- this is the riskiest piece of
    algebra in the module, so it gets an independent brute-force
    cross-check: integrate `(N_j x h_m).x_hat dS` directly, per edge, and
    compare."""
    mode = port_modes["PORT_1"][0]
    cs = mode.cross_section
    bary, w_hat = mesh_quadrature.tri_rule(2)

    brute = np.zeros(cs.n_edges, dtype=complex)
    for t in range(cs.n_triangles):
        weights = w_hat * float(cs.area[t])
        grad_t = cs.grad_t[t]
        sign = cs.tri_edge_sign[t]
        from ports.basis2d import whitney2d_basis

        N = whitney2d_basis(grad_t, sign)(bary)  # (3,M,2)
        h_field = _h_t_on_triangle(cs, t, bary, mode.e_edge_dofs, mode.ex_tilde_vertex_dofs, mode.gamma, mode.omega)
        for local_edge in range(3):
            cross = N[local_edge, :, 0] * h_field[:, 1] - N[local_edge, :, 1] * h_field[:, 0]
            brute[cs.tri_edges[t, local_edge]] += np.sum(weights * cross)

    assert brute == pytest.approx(mode.overlap_h, rel=1e-6, abs=1e-6 * np.abs(mode.overlap_h).max())


def test_overlap_e_matches_direct_quadrature(port_modes):
    mode = port_modes["PORT_1"][0]
    cs = mode.cross_section
    bary, w_hat = mesh_quadrature.tri_rule(2)

    brute = np.zeros(cs.n_edges, dtype=complex)
    for t in range(cs.n_triangles):
        weights = w_hat * float(cs.area[t])
        grad_t = cs.grad_t[t]
        sign = cs.tri_edge_sign[t]
        from ports.basis2d import whitney2d_basis

        N = whitney2d_basis(grad_t, sign)(bary)
        e_field = _e_t_on_triangle(cs, t, bary, mode.e_edge_dofs)
        for local_edge in range(3):
            dot = N[local_edge, :, 0] * e_field[:, 0] + N[local_edge, :, 1] * e_field[:, 1]
            brute[cs.tri_edges[t, local_edge]] += np.sum(weights * dot)

    assert brute == pytest.approx(mode.overlap_e, rel=1e-6, abs=1e-6 * np.abs(mode.overlap_e).max())


# --- Section 5: build_B / build_g ---


def test_build_B_is_supported_only_on_port_edges(mesh_and_solver, port_modes):
    mesh, _solver, omega = mesh_and_solver
    B = build_B(port_modes, mesh, omega)
    port1_edges = set(port_modes["PORT_1"][0].cross_section.global_edge_ids.tolist())
    port2_edges = set(port_modes["PORT_2"][0].cross_section.global_edge_ids.tolist())
    allowed = port1_edges | port2_edges
    rows, cols = B.nonzero()
    assert set(rows.tolist()) <= allowed
    assert set(cols.tolist()) <= allowed


def test_build_B_is_symmetric_to_near_machine_precision(mesh_and_solver, port_modes):
    """Post-review: `build_B` now assembles `(Y_m**2)*outer(overlap_e,
    overlap_e)` rather than `Y_m*outer(overlap_e, overlap_h)` (Section
    5.1's update) -- every summand is `scalar * outer(v, v)`, symmetric in
    the literal matrix regardless of mode quality, not merely "finite".
    Real relative asymmetry observed here is at the level of floating-
    point roundoff, several orders of magnitude tighter than the old
    formula's ~130% worst case (see test_build_B_is_symmetric_for_a_mode_
    with_an_inconsistent_overlap_h below for a direct demonstration that
    this holds even when the old formula would not have)."""
    mesh, _solver, omega = mesh_and_solver
    B = build_B(port_modes, mesh, omega).toarray()
    assert np.all(np.isfinite(B))
    residual = np.abs(B - B.T).max()
    scale = max(1.0, np.abs(B).max())
    assert residual <= 1e-9 * scale


def test_build_B_is_symmetric_for_a_mode_with_an_inconsistent_overlap_h(mesh_and_solver, port_modes):
    """The strongest possible demonstration that build_B no longer relies
    on overlap_h ~ Y*overlap_e holding even approximately: corrupt one
    mode's cached overlap_h with unrelated garbage (breaking the identity
    completely -- exactly what a marginal/poorly-resolved mode does, only
    more extreme) and confirm build_B is still exactly symmetric, because
    it never reads overlap_h at all."""
    import dataclasses

    mesh, _solver, omega = mesh_and_solver
    mode = port_modes["PORT_1"][0]
    corrupted = dataclasses.replace(mode, overlap_h=np.arange(mode.overlap_h.shape[0], dtype=complex) * (1 + 2j))
    corrupted_port_modes = {**port_modes, "PORT_1": [corrupted, *port_modes["PORT_1"][1:]]}

    B = build_B(corrupted_port_modes, mesh, omega).toarray()
    assert np.all(np.isfinite(B))
    residual = np.abs(B - B.T).max()
    scale = max(1.0, np.abs(B).max())
    assert residual <= 1e-9 * scale


def test_build_g_only_touches_excited_ports_edges(mesh_and_solver, port_modes):
    mesh, _solver, omega = mesh_and_solver
    g = build_g(port_modes, {("PORT_1", 1): 1.0 + 0j}, mesh, omega)
    port1_edges = set(port_modes["PORT_1"][0].cross_section.global_edge_ids.tolist())
    nonzero = set(np.flatnonzero(g).tolist())
    assert nonzero <= port1_edges
    assert len(nonzero) > 0


def test_build_g_zero_excitation_gives_zero_vector(mesh_and_solver, port_modes):
    mesh, _solver, omega = mesh_and_solver
    g = build_g(port_modes, {}, mesh, omega)
    assert np.all(g == 0)


def test_build_g_unknown_port_raises(mesh_and_solver, port_modes):
    mesh, _solver, omega = mesh_and_solver
    with pytest.raises(PortOperatorError):
        build_g(port_modes, {("PORT_99", 1): 1.0}, mesh, omega)


def test_build_g_mode_out_of_range_raises(mesh_and_solver, port_modes):
    mesh, _solver, omega = mesh_and_solver
    with pytest.raises(PortOperatorError):
        build_g(port_modes, {("PORT_1", 99): 1.0}, mesh, omega)


# --- Section 6: de-embedding ---


def test_deembed_zero_offset_is_identity(port_modes):
    n = sum(len(v) for v in port_modes.values())
    S = (np.eye(n) * 0.3 + 0.1j).astype(complex)
    offsets = {tag: 0.0 for tag in port_modes}
    S_ref = deembed(S, port_modes, offsets)
    assert S_ref == pytest.approx(S)


def test_deembed_missing_offset_raises(port_modes):
    n = sum(len(v) for v in port_modes.values())
    S = np.eye(n, dtype=complex)
    with pytest.raises(PortOperatorError):
        deembed(S, port_modes, {"PORT_1": 0.001})


def test_deembed_shape_mismatch_raises(port_modes):
    S = np.eye(2, dtype=complex)  # wrong size (4 (port,mode) pairs expected)
    offsets = {tag: 0.001 for tag in port_modes}
    with pytest.raises(PortOperatorError):
        deembed(S, port_modes, offsets)
