"""Validation suite for fem.assembly (docs/module3_fem_assembly_equations.md
Section 7): adaptive quadrature convergence, Mbar_mu symmetry, global
assembly symmetry, and tet-subset consistency.
"""
import numpy as np
import pytest

from fem import assemble, element_matrices
from fem.assembly import AssemblyConvergenceError, _composite_points_weights
from material import ConstantMaterial, MaterialAssembly
from material.core import MaterialModel
from mesh_interface import MeshInterface

_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_TETS = np.array([[0, 1, 2, 3]])
_SURFACE_TAGS = {"PEC_GROUND": np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])}


def _single_tet_mesh() -> MeshInterface:
    return MeshInterface(_VERTICES, _TETS, np.array(["A"]), _SURFACE_TAGS)


class _LinearScalarMaterial(MaterialModel):
    """eps_r(r) = 2 + 0.05*x -- a mild, genuinely smooth spatial variation
    (Section 7: "confirm correct order escalation for a smoothly-varying
    test field"). eps_r is degree-1 and W_l.W_m is degree-2, so the
    product is degree 3 -- not exactly integrated by the order-2 base
    rule at any fixed refinement, only convergent as refinement
    increases. Empirically (verified before picking this slope): level
    0-vs-1 does *not* converge (residual ~1.1e-4 > the 1e-4 tolerance),
    escalating to level 2 does (~6e-5) -- a genuine, non-immediate,
    non-ceiling-exhausting escalation."""

    def _epsilon(self, points):
        val = 2.0 + 0.05 * points[:, 0]
        return val[:, None, None] * np.eye(3, dtype=complex)[None, :, :]

    def _mu(self, points):
        return np.tile(np.eye(3, dtype=complex), (points.shape[0], 1, 1))


class _OscillatingMaterial(MaterialModel):
    """A spatial frequency far too high for any reachable refinement
    level to resolve (Section 7: "confirm the max-order raise fires on a
    deliberately pathological... field a coarse mesh cannot resolve")."""

    def _epsilon(self, points):
        val = 2.0 + np.sin(1.0e6 * points[:, 0])
        return val[:, None, None] * np.eye(3, dtype=complex)[None, :, :]

    def _mu(self, points):
        return np.tile(np.eye(3, dtype=complex), (points.shape[0], 1, 1))


class _NonTrivialMuMaterial(MaterialModel):
    """A symmetric, non-diagonal (but real, hence trivially passive) mu_r
    -- Section 7's Mbar_mu symmetry spot-check, written now so Module 5
    only needs to supply the input."""

    _MU = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.0], [0.0, 0.0, 1.5]], dtype=complex)

    def _epsilon(self, points):
        return np.tile(np.eye(3, dtype=complex), (points.shape[0], 1, 1))

    def _mu(self, points):
        return np.tile(self._MU, (points.shape[0], 1, 1))


# --- Section 4: adaptive quadrature convergence -----------------------------

def test_constant_material_converges_immediately_at_level_0_vs_1():
    """Section 4: 'converges immediately (at the lowest tried order) for
    constant materials' -- checked directly on the composite-quadrature
    primitive: level 0 and level 1 must already agree to floating-point
    precision for a spatially constant integrand (any partition of a
    domain, each piece exactly integrated by the same rule, sums to the
    same exact whole)."""
    coords = _VERTICES
    _points0, weights0 = _composite_points_weights(coords, 0)
    _points1, weights1 = _composite_points_weights(coords, 1)
    assert weights0.sum() == pytest.approx(weights1.sum(), rel=1e-12)


def test_constant_material_element_matrices_do_not_raise():
    mesh = _single_tet_mesh()
    materials = MaterialAssembly({"A": ConstantMaterial(eps_r=3.0)})
    element_matrices(mesh, 0, materials)  # must not raise


def test_smoothly_varying_material_escalates_then_converges():
    """Confirms genuine, non-immediate escalation, not just 'did not
    raise': level 0-vs-1 must NOT be within tolerance (ruling out trivial
    immediate convergence), but the procedure still succeeds overall
    (ruling out hitting the ceiling)."""
    from fem.assembly import _CONVERGENCE_TOL

    coords = _VERTICES
    field = _LinearScalarMaterial()
    grad = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sign = np.ones(6)

    from fem.assembly import _mass_matrix

    M0 = _mass_matrix(MaterialAssembly({"A": field}), "A", coords, grad, sign, 0)
    M1 = _mass_matrix(MaterialAssembly({"A": field}), "A", coords, grad, sign, 1)
    residual_0v1 = np.linalg.norm(M1 - M0) / np.linalg.norm(M1)
    assert residual_0v1 > _CONVERGENCE_TOL, "test material converges trivially at level 0 -- not exercising escalation"

    mesh = _single_tet_mesh()
    materials = MaterialAssembly({"A": field})
    K_e, M_e = element_matrices(mesh, 0, materials)  # must not raise
    assert M_e == pytest.approx(M_e.T, abs=1e-10)  # still symmetric once converged


def test_pathological_material_raises_convergence_error():
    mesh = _single_tet_mesh()
    materials = MaterialAssembly({"A": _OscillatingMaterial()})
    with pytest.raises(AssemblyConvergenceError):
        element_matrices(mesh, 0, materials)


# --- Section 7: Mbar_mu symmetry spot-check (non-trivial mu_r) -------------

def test_stiffness_matrix_symmetric_with_nontrivial_mu_r():
    mesh = _single_tet_mesh()
    materials = MaterialAssembly({"A": _NonTrivialMuMaterial()})
    K_e, _M_e = element_matrices(mesh, 0, materials)
    assert K_e == pytest.approx(K_e.T, abs=1e-10)


# --- Section 5.2/7: tet-subset consistency, global symmetry (real mesh) ----

@pytest.fixture(scope="module")
def real_mesh_and_materials():
    pytest.importorskip("gmsh")
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import load_material_spec

    params = GeometryParams(
        w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
        eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
        reference_frequency=25e9, target_elements_per_wavelength=8,
    )
    mesh_handle, material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)

    base_assembly = load_material_spec(geometry_stub=material_stub)
    tag_to_model = dict(base_assembly._tag_to_model)
    tag_to_model["LC"] = ConstantMaterial(eps_r=params.eps_r_substrate)  # Phase 1: no LC material yet
    materials = MaterialAssembly(tag_to_model)

    interior_tets = [t for t in range(mesh.n_tets) if mesh.tet_volume_tag(t) != "PML_TOP"]
    return mesh, materials, interior_tets[:600]  # bounded sample -- keeps the suite fast


def test_global_assembly_is_symmetric_on_a_real_mesh(real_mesh_and_materials):
    mesh, materials, tet_subset = real_mesh_and_materials
    K, M = assemble(mesh, materials, tet_subset=tet_subset)
    assert abs(K - K.T).max() < 1e-9
    assert abs(M - M.T).max() < 1e-9


def test_tet_subset_consistency_on_a_real_mesh(real_mesh_and_materials):
    mesh, materials, tet_subset = real_mesh_and_materials
    half = len(tet_subset) // 2
    subset_a, subset_b = tet_subset[:half], tet_subset[half:]

    K_whole, M_whole = assemble(mesh, materials, tet_subset=tet_subset)
    K_a, M_a = assemble(mesh, materials, tet_subset=subset_a)
    K_b, M_b = assemble(mesh, materials, tet_subset=subset_b)

    assert abs(K_whole - (K_a + K_b)).max() < 1e-9
    assert abs(M_whole - (M_a + M_b)).max() < 1e-9
