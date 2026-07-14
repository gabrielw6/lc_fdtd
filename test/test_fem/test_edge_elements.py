"""Validation suite for fem.edge_elements (docs/module3_fem_assembly_equations.md
Section 2.4). No Gmsh needed -- pure numpy on hand-constructed tets.
"""
import numpy as np
import pytest

from fem.edge_elements import whitney_basis, whitney_curl
from mesh_interface.interface import LOCAL_EDGES

# Reference tet: vertices at the origin and the three unit axis points
# (docs/module3 Section 7's own reference case). lambda_0=1-x-y-z,
# lambda_1=x, lambda_2=y, lambda_3=z -- gradients read off directly.
_REF_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_REF_GRAD = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_UNSIGNED = np.ones(6)  # local order already ascending-global for [0,1,2,3]


def _lambda_affine(grad: np.ndarray, r0: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Independent (test-local) affine reconstruction lambda_i(r) =
    delta_{i,0} + grad_i . (r - r0), not reusing fem.assembly's private
    helper of the same idea."""
    rel = r - r0
    return np.array([1.0, 0.0, 0.0, 0.0]) + grad @ rel


def _whitney_at(grad: np.ndarray, sign: np.ndarray, r0: np.ndarray, r: np.ndarray, local_edge: int) -> np.ndarray:
    a, b = LOCAL_EDGES[local_edge]
    lam = _lambda_affine(grad, r0, r)
    return sign[local_edge] * (lam[a] * grad[b] - lam[b] * grad[a])


# --- Section 2.4: tangential-trace normalization ---------------------------

def test_tangential_trace_normalization_on_reference_tet():
    """integral over e_ell of W_ell . dl == 1, along e_ell's own path (a
    to b, unsigned formula) -- a direct algebraic identity of the Whitney
    construction, checked here by 1D Gauss-Legendre quadrature along each
    edge (exact, since the integrand is affine in the path parameter)."""
    basis = whitney_basis(_REF_GRAD, _UNSIGNED)
    t_nodes, t_weights = np.polynomial.legendre.leggauss(5)
    t = 0.5 * (t_nodes + 1.0)  # [-1,1] -> [0,1]
    w = 0.5 * t_weights

    for local_edge, (a, b) in enumerate(LOCAL_EDGES):
        bary = np.zeros((len(t), 4))
        bary[:, a] = 1.0 - t
        bary[:, b] = t
        W = basis(bary)[local_edge]  # (M,3)
        edge_vector = _REF_VERTICES[b] - _REF_VERTICES[a]
        integral = float(np.sum(w * (W @ edge_vector)))
        assert integral == pytest.approx(1.0, rel=1e-9), f"edge {local_edge}"


# --- Section 2.4: curl reproduction -----------------------------------------

def test_curl_matches_finite_difference():
    """Numerically differentiate W_ell (central difference -- exact here
    since W_ell is affine, so no truncation error) and confirm it matches
    the closed-form C_ell."""
    sign = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])  # arbitrary, nontrivial signs
    C = whitney_curl(_REF_GRAD, sign)
    r0 = np.array([0.2, 0.3, 0.1])  # an interior point, not a vertex
    h = 1e-4

    for local_edge in range(6):
        def F(r, local_edge=local_edge):
            return _whitney_at(_REF_GRAD, sign, _REF_VERTICES[0], r, local_edge)

        grads = np.empty((3, 3))  # grads[axis] = dF/d(axis)
        for axis in range(3):
            rp, rm = r0.copy(), r0.copy()
            rp[axis] += h
            rm[axis] -= h
            grads[axis] = (F(rp) - F(rm)) / (2 * h)
        curl_fd = np.array(
            [grads[1, 2] - grads[2, 1], grads[2, 0] - grads[0, 2], grads[0, 1] - grads[1, 0]]
        )
        assert curl_fd == pytest.approx(C[local_edge], rel=1e-6, abs=1e-8), f"edge {local_edge}"


# --- Section 2.4: DOF continuity across two elements (the real sign-convention test) ---

# Two-tet patch sharing face {1,2,3}: v0..v4, tetA=[0,1,2,3] (det P>0
# already), tetB=[4,1,2,3] (det P<0 as given -- exercises the orientation
# fix too).
_PATCH_VERTICES = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
)
_PATCH_TETS = np.array([[0, 1, 2, 3], [4, 1, 2, 3]])
_PATCH_SURFACE_TAGS = {
    "PEC_GROUND": np.array(
        [[0, 2, 3], [0, 1, 3], [0, 1, 2], [4, 2, 3], [4, 1, 3], [4, 1, 2]]
    )
}


def test_dof_continuity_across_shared_edges():
    """The single most important test in this module (Section 2.4): build
    a two-tet patch, evaluate the *signed* basis functions for each shared
    edge from both tets at that edge's own midpoint, and confirm the full
    vector (not just a projection) agrees -- since a point ON the edge
    itself has a well-defined tangential direction and W_ell there is
    parallel to it, agreement at that specific point is exactly the
    tangential-continuity claim."""
    from mesh_interface import MeshInterface

    mesh = MeshInterface(_PATCH_VERTICES, _PATCH_TETS, np.array(["A", "B"]), _PATCH_SURFACE_TAGS)

    grad_a, sign_a = mesh.grad_lambda(0), mesh.tet_edge_sign[0]
    grad_b, sign_b = mesh.grad_lambda(1), mesh.tet_edge_sign[1]
    r0_a, r0_b = mesh.vertices[mesh.tets[0, 0]], mesh.vertices[mesh.tets[1, 0]]

    for shared_pair in [(1, 2), (1, 3), (2, 3)]:
        midpoint = 0.5 * (_PATCH_VERTICES[shared_pair[0]] + _PATCH_VERTICES[shared_pair[1]])

        local_edge_a = _find_local_edge(mesh.tets[0], shared_pair)
        local_edge_b = _find_local_edge(mesh.tets[1], shared_pair)

        W_a = _whitney_at(grad_a, sign_a, r0_a, midpoint, local_edge_a)
        W_b = _whitney_at(grad_b, sign_b, r0_b, midpoint, local_edge_b)

        assert W_a == pytest.approx(W_b, abs=1e-9), f"shared edge {shared_pair}"


def _find_local_edge(tet_row: np.ndarray, global_pair: tuple[int, int]) -> int:
    target = set(global_pair)
    for local_edge, (a, b) in enumerate(LOCAL_EDGES):
        if {int(tet_row[a]), int(tet_row[b])} == target:
            return local_edge
    raise AssertionError(f"no local edge of {tet_row.tolist()} matches {global_pair}")
