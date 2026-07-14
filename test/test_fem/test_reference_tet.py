"""Reference-tetrahedron validation for fem.assembly
(docs/module3_fem_assembly_equations.md Section 7): assembles K^(e), M^(e)
for the standard unit reference tet with eps_r=mu_r=I and compares against
a closed form derived independently here (hand computation via the exact
barycentric-monomial integral formula, not by re-running this module's own
code a second time) -- an external check in the sense the doc asks for,
even though it isn't literally transcribed from a textbook table (avoiding
an unverifiable transcription error from memory).
"""
import numpy as np
import pytest

from fem.assembly import element_matrices
from material import ConstantMaterial, MaterialAssembly
from mesh_interface import MeshInterface
from mesh_interface.interface import LOCAL_EDGES

_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_TETS = np.array([[0, 1, 2, 3]])
_SURFACE_TAGS = {
    "PEC_GROUND": np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]),
}


def _mesh() -> MeshInterface:
    return MeshInterface(_VERTICES, _TETS, np.array(["A"]), _SURFACE_TAGS)


def _hand_derived_K() -> np.ndarray:
    """K_lm = C_l . C_m * V (Section 3.1, mu_r=I constant => Mbar_mu=V*I).
    C_l = 2 (g_a x g_b) computed by hand from the known reference-tet
    gradients (g0=(-1,-1,-1), g1=(1,0,0), g2=(0,1,0), g3=(0,0,1))."""
    g = {0: (-1.0, -1.0, -1.0), 1: (1.0, 0.0, 0.0), 2: (0.0, 1.0, 0.0), 3: (0.0, 0.0, 1.0)}
    C = np.array([2.0 * np.cross(g[a], g[b]) for a, b in LOCAL_EDGES])
    V = 1.0 / 6.0
    return (C @ C.T) * V


def _hand_derived_M() -> np.ndarray:
    """M_lm = sum over the 4 expanded terms of
    integral(lambda_i lambda_j) * (dot of the two opposite gradients),
    using the exact simplex-monomial integral: integral(lambda_i lambda_j)
    = V/10 if i==j else V/20 (derived from the general barycentric-
    monomial formula already used and verified in Module 1/2's own
    quadrature self-tests)."""
    g = {0: np.array([-1.0, -1.0, -1.0]), 1: np.array([1.0, 0.0, 0.0]), 2: np.array([0.0, 1.0, 0.0]), 3: np.array([0.0, 0.0, 1.0])}
    V = 1.0 / 6.0

    def lam_int(i, j):
        return V / 10.0 if i == j else V / 20.0

    M = np.zeros((6, 6))
    for l, (a, b) in enumerate(LOCAL_EDGES):
        for m, (c, d) in enumerate(LOCAL_EDGES):
            M[l, m] = (
                lam_int(a, c) * g[b].dot(g[d])
                - lam_int(a, d) * g[b].dot(g[c])
                - lam_int(b, c) * g[a].dot(g[d])
                + lam_int(b, d) * g[a].dot(g[c])
            )
    return M


def test_reference_tet_stiffness_matrix_matches_hand_derivation():
    mesh = _mesh()
    materials = MaterialAssembly({"A": ConstantMaterial(eps_r=1.0, mu_r=1.0)})
    K_e, _M_e = element_matrices(mesh, 0, materials)
    assert K_e.real == pytest.approx(_hand_derived_K(), abs=1e-10)
    assert np.abs(K_e.imag).max() < 1e-12


def test_reference_tet_mass_matrix_matches_hand_derivation():
    mesh = _mesh()
    materials = MaterialAssembly({"A": ConstantMaterial(eps_r=1.0, mu_r=1.0)})
    _K_e, M_e = element_matrices(mesh, 0, materials)
    assert M_e.real == pytest.approx(_hand_derived_M(), abs=1e-10)
    assert np.abs(M_e.imag).max() < 1e-12


def test_reference_tet_matrices_are_symmetric():
    mesh = _mesh()
    materials = MaterialAssembly({"A": ConstantMaterial(eps_r=2.0 - 0.1j, mu_r=1.0)})
    K_e, M_e = element_matrices(mesh, 0, materials)
    assert K_e == pytest.approx(K_e.T, abs=1e-12)
    assert M_e == pytest.approx(M_e.T, abs=1e-12)


def test_reference_tet_mass_matrix_scales_with_eps_r():
    """M^(e) is linear in eps_r for a constant, isotropic material."""
    mesh = _mesh()
    base = MaterialAssembly({"A": ConstantMaterial(eps_r=1.0)})
    scaled = MaterialAssembly({"A": ConstantMaterial(eps_r=3.5)})
    _, M_base = element_matrices(mesh, 0, base)
    _, M_scaled = element_matrices(mesh, 0, scaled)
    assert M_scaled.real == pytest.approx(3.5 * M_base.real, rel=1e-9)
