"""Validation suite for ports.basis2d (docs/module4_ports_equations.md
Section 2.5). No mesh/gmsh needed -- pure numpy on a hand-constructed
reference triangle, mirroring test_fem/test_edge_elements.py's pattern for
the 3D Whitney basis.
"""
import numpy as np
import pytest

from ports.basis2d import TRI_LOCAL_EDGES, whitney2d_basis, whitney2d_curl

# Reference triangle: (0,0), (1,0), (0,1) in (y,z). lambda_0=1-y-z,
# lambda_1=y, lambda_2=z -- gradients read off directly.
_REF_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
_REF_GRAD = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
_UNSIGNED = np.ones(3)


def _lambda_affine(grad: np.ndarray, r0: np.ndarray, r: np.ndarray) -> np.ndarray:
    rel = r - r0
    return np.array([1.0, 0.0, 0.0]) + grad @ rel


def _whitney_at(grad: np.ndarray, sign: np.ndarray, r0: np.ndarray, r: np.ndarray, local_edge: int) -> np.ndarray:
    a, b = TRI_LOCAL_EDGES[local_edge]
    lam = _lambda_affine(grad, r0, r)
    return sign[local_edge] * (lam[a] * grad[b] - lam[b] * grad[a])


def test_tangential_trace_normalization_on_reference_triangle():
    """integral over e_ell of N_ell . dl == 1, along e_ell's own path (a to
    b) -- a direct algebraic identity of the Whitney construction, checked
    by 1D Gauss-Legendre quadrature (exact: the integrand is affine)."""
    basis = whitney2d_basis(_REF_GRAD, _UNSIGNED)
    t_nodes, t_weights = np.polynomial.legendre.leggauss(5)
    t = 0.5 * (t_nodes + 1.0)
    w = 0.5 * t_weights

    for local_edge, (a, b) in enumerate(TRI_LOCAL_EDGES):
        bary = np.zeros((len(t), 3))
        bary[:, a] = 1.0 - t
        bary[:, b] = t
        N = basis(bary)[local_edge]  # (M,2)
        edge_vector = _REF_VERTICES[b] - _REF_VERTICES[a]
        integral = float(np.sum(w * (N @ edge_vector)))
        assert integral == pytest.approx(1.0, rel=1e-9), f"edge {local_edge}"


def test_curl_matches_finite_difference():
    """Numerically differentiate N_ell (central difference, exact here
    since N_ell is affine) and confirm it matches the closed-form scalar
    curl_t."""
    sign = np.array([1.0, -1.0, 1.0])
    curl = whitney2d_curl(_REF_GRAD, sign)
    r0 = np.array([0.2, 0.3])
    h = 1e-4

    for local_edge in range(3):
        def F(r, local_edge=local_edge):
            return _whitney_at(_REF_GRAD, sign, _REF_VERTICES[0], r, local_edge)

        grads = np.empty((2, 2))  # grads[axis] = dF/d(axis)
        for axis in range(2):
            rp, rm = r0.copy(), r0.copy()
            rp[axis] += h
            rm[axis] -= h
            grads[axis] = (F(rp) - F(rm)) / (2 * h)
        # curl_t (F_y, F_z) = dFz/dy - dFy/dz = grads[0,1] - grads[1,0]
        curl_fd = grads[0, 1] - grads[1, 0]
        assert curl_fd == pytest.approx(curl[local_edge], rel=1e-6, abs=1e-8), f"edge {local_edge}"
