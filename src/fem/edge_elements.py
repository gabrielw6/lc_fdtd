"""fem.edge_elements -- the Whitney/Nedelec edge basis and its curl
(docs/module3_fem_assembly_equations.md Section 2).

Reuses `mesh_interface`'s own local edge table (Module 1 Section 3.1) as
the single source of truth for the fixed local (a_l, b_l) pairs, rather
than a second, potentially-drifting copy.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from mesh_interface.interface import LOCAL_EDGES

LocalEdgeTable = Sequence[tuple[int, int]]


def whitney_basis(
    tet_grad_lambda: np.ndarray, tet_edge_sign: np.ndarray, local_edge_table: LocalEdgeTable = LOCAL_EDGES
) -> Callable[[np.ndarray], np.ndarray]:
    """Section 2.2: returns a function evaluating all six signed Whitney
    basis functions at a set of quadrature points, given directly as
    barycentric weights (`mesh.quadrature_tet`'s third return value) --
    `W_ell(r_k) = s_ell [ lambda_{a_ell}(r_k) g_{b_ell} - lambda_{b_ell}(r_k) g_{a_ell} ]`.

    `evaluate(barycentric: (M,4)) -> (6,M,3)`.
    """
    grad = np.asarray(tet_grad_lambda, dtype=float)  # (4,3)
    sign = np.asarray(tet_edge_sign, dtype=float)  # (6,)

    def evaluate(barycentric: np.ndarray) -> np.ndarray:
        bary = np.asarray(barycentric, dtype=float)  # (M,4)
        M = bary.shape[0]
        W = np.empty((len(local_edge_table), M, 3))
        for local_edge, (a, b) in enumerate(local_edge_table):
            W[local_edge] = sign[local_edge] * (
                bary[:, a, None] * grad[b][None, :] - bary[:, b, None] * grad[a][None, :]
            )
        return W

    return evaluate


def whitney_curl(
    tet_grad_lambda: np.ndarray, tet_edge_sign: np.ndarray, local_edge_table: LocalEdgeTable = LOCAL_EDGES
) -> np.ndarray:
    """Section 2.3: `C_ell = curl(W_ell) = 2 s_ell (g_{a_ell} x g_{b_ell})`
    -- a constant 3-vector per local edge, no quadrature-point argument
    needed. Returns `(6,3)`."""
    grad = np.asarray(tet_grad_lambda, dtype=float)
    sign = np.asarray(tet_edge_sign, dtype=float)

    C = np.empty((len(local_edge_table), 3))
    for local_edge, (a, b) in enumerate(local_edge_table):
        C[local_edge] = 2.0 * sign[local_edge] * np.cross(grad[a], grad[b])
    return C
