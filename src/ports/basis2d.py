"""ports.basis2d -- the 2D Whitney edge basis and its scalar curl, restricted
to a triangle (docs/module4_ports_equations.md Section 2.5).

Identical construction to `fem.edge_elements`'s 3D Whitney basis, just with
2D barycentric gradients and a scalar (not vector) curl -- the direct
triangle analogue, not a reinvention.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

TRI_LOCAL_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
LocalEdgeTable = Sequence[tuple[int, int]]


def whitney2d_basis(
    grad_t: np.ndarray, sign: np.ndarray, local_edge_table: LocalEdgeTable = TRI_LOCAL_EDGES
) -> Callable[[np.ndarray], np.ndarray]:
    """Section 2.5: `N_(a,b)(r) = lambda_a grad_t(lambda_b) - lambda_b
    grad_t(lambda_a)`. `evaluate(barycentric: (M,3)) -> (3,M,2)`."""
    grad = np.asarray(grad_t, dtype=float)  # (3,2)
    s = np.asarray(sign, dtype=float)  # (3,)

    def evaluate(barycentric: np.ndarray) -> np.ndarray:
        bary = np.asarray(barycentric, dtype=float)  # (M,3)
        M = bary.shape[0]
        N = np.empty((len(local_edge_table), M, 2))
        for local_edge, (a, b) in enumerate(local_edge_table):
            N[local_edge] = s[local_edge] * (
                bary[:, a, None] * grad[b][None, :] - bary[:, b, None] * grad[a][None, :]
            )
        return N

    return evaluate


def whitney2d_curl(
    grad_t: np.ndarray, sign: np.ndarray, local_edge_table: LocalEdgeTable = TRI_LOCAL_EDGES
) -> np.ndarray:
    """Section 2.5: `curl_t N_(a,b) = 2(grad_t(lambda_a) x grad_t(lambda_b))`,
    a *scalar* (2D cross product) constant per triangle. Returns `(3,)`."""
    grad = np.asarray(grad_t, dtype=float)
    s = np.asarray(sign, dtype=float)

    curl = np.empty(len(local_edge_table))
    for local_edge, (a, b) in enumerate(local_edge_table):
        ga, gb = grad[a], grad[b]
        cross2d = ga[0] * gb[1] - ga[1] * gb[0]
        curl[local_edge] = 2.0 * s[local_edge] * cross2d
    return curl
