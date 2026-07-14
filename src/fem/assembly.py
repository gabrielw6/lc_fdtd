"""fem.assembly -- element matrices K^(e), M^(e) and global sparse
assembly (docs/module3_fem_assembly_equations.md Sections 3-5).

Consumes `mesh_interface.MeshInterface` and `material.MaterialAssembly`
directly -- these two types *are* the interface abstraction layer for
Modules 1 and 2 respectively (the architecture doc's own dependency
graph has `mesh.interface` and `material.core` feeding `fem.assembly`
directly), so depending on them here is the intended design, not a
layering violation. `fem` never reaches into `geometry_builder`, `meshing`,
or `material.regions`/`tensor_interpolation` directly.

**Adaptive quadrature, adapted (Section 4).** The doc's escalation ladder
is polynomial-exactness order (2, 4, 6, ...). `mesh_interface.quadrature`
only tabulates orders 1-2 -- a deliberate, documented scope limit there
(only coefficients verifiable with full confidence against the exact
barycentric-monomial integral formula). Order 1 is *not* exact for M's
quadratic Whitney-product integrand even when eps_r is constant, so
order-1-vs-order-2 can never show convergence regardless of the material
-- there is no usable two-point polynomial-order ladder available yet.
Rather than transcribe unverified higher-order coefficients, escalation
here is **spatial refinement** of the already-verified order-2 rule:
level L subdivides the tet into 4^L sub-tets (by connecting each face to
the centroid -- an exact, gap/overlap-free decomposition) and sums the
order-2 rule over each. This is a genuinely convergent quadrature sequence
by construction, and for a spatially constant integrand levels 0 and 1
agree to floating-point precision, exactly reproducing the doc's
"immediate convergence for constant materials" behavior.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp

from material import MaterialAssembly
from mesh_interface import MeshInterface
from mesh_interface import quadrature as mesh_quadrature
from mesh_interface.interface import LOCAL_EDGES, LOCAL_FACES

from .edge_elements import whitney_curl

_CONVERGENCE_TOL = 1e-4
_BASE_ORDER = 2  # Section 6.2: the minimum order exact for the quadratic mass integrand

DEFAULT_LEVELS: tuple[int, ...] = (0, 1, 2, 3)


class AssemblyConvergenceError(RuntimeError):
    """Raised when the adaptive quadrature procedure (Section 4) fails to
    converge by the finest available refinement level -- signals either
    material variation too sharp for the mesh resolution, or a material-
    evaluator bug. Never silently used as the answer."""


def _adaptive_quadrature(
    compute: Callable[[int], np.ndarray], levels: Sequence[int] = DEFAULT_LEVELS, tol: float = _CONVERGENCE_TOL
) -> np.ndarray:
    """Section 4's procedure, generic over what `compute(level)` returns
    (a (3,3) tensor for Mbar_mu, or a (6,6) matrix for M): start at
    `levels[0]`, escalate through the rest, comparing consecutive
    estimates by relative Frobenius norm; use the finer estimate once
    converged; raise if the ladder is exhausted first."""
    if len(levels) < 2:
        raise ValueError("need at least two refinement levels to run the adaptive procedure")

    prev = compute(levels[0])
    for level in levels[1:]:
        curr = compute(level)
        denom = np.linalg.norm(curr)
        residual = np.linalg.norm(curr - prev) / denom if denom > 0 else np.linalg.norm(curr - prev)
        if residual < tol:
            return curr
        prev = curr

    raise AssemblyConvergenceError(
        f"adaptive quadrature did not converge by refinement level {levels[-1]} (the finest tried) -- "
        "material variation may be too sharp for the mesh resolution (needs refinement), or the "
        "material evaluator has a bug"
    )


def _centroid_subdivide(coords: np.ndarray, level: int) -> list[np.ndarray]:
    """`coords`: (4,3) physical vertices. Recursively splits into 4^level
    sub-tets by connecting each of the 4 faces to the parent's centroid --
    a standard, exact (volume-preserving, gap/overlap-free) pyramidal
    decomposition, not a new quadrature coefficient table."""
    if level <= 0:
        return [coords]
    centroid = coords.mean(axis=0, keepdims=True)
    pieces: list[np.ndarray] = []
    for face in LOCAL_FACES:
        sub_coords = np.concatenate([coords[list(face)], centroid], axis=0)
        pieces.extend(_centroid_subdivide(sub_coords, level - 1))
    return pieces


def _tet_volume(coords: np.ndarray) -> float:
    return float(abs(np.linalg.det(coords[1:] - coords[0])) / 6.0)


def _composite_points_weights(coords: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Physical quadrature points/weights for the composite rule at
    `level`: the order-2 base rule mapped onto each of the 4^level
    centroid sub-tets."""
    bary_ref, w_hat = mesh_quadrature.tet_rule(_BASE_ORDER)
    all_points = []
    all_weights = []
    for sub_coords in _centroid_subdivide(coords, level):
        all_points.append(bary_ref @ sub_coords)
        all_weights.append(w_hat * _tet_volume(sub_coords))
    return np.concatenate(all_points, axis=0), np.concatenate(all_weights, axis=0)


def _whitney_at_physical_points(
    grad: np.ndarray, sign: np.ndarray, vertex0: np.ndarray, points: np.ndarray,
    local_edge_table: Sequence[tuple[int, int]] = LOCAL_EDGES,
) -> np.ndarray:
    """W_ell(r) for arbitrary physical points r -- needed because the
    composite sub-tet points above don't carry a barycentric coordinate of
    their own against the *parent* tet's rule. Uses the affine
    reconstruction lambda_i(r) = delta_{i,0} + grad_i . (r - vertex0),
    valid for any physical point, not just the parent's own quadrature
    points -- vertex 0 is where lambda_0=1, lambda_{i!=0}=0 by definition
    (Kronecker), giving the needed anchor for free."""
    rel = points - vertex0[None, :]
    lam = np.zeros((points.shape[0], 4))
    lam[:, 0] = 1.0
    lam += rel @ grad.T

    M = points.shape[0]
    W = np.empty((len(local_edge_table), M, 3))
    for local_edge, (a, b) in enumerate(local_edge_table):
        W[local_edge] = sign[local_edge] * (
            lam[:, a, None] * grad[b][None, :] - lam[:, b, None] * grad[a][None, :]
        )
    return W


def _mu_bar(materials: MaterialAssembly, tag: str, coords: np.ndarray, level: int) -> np.ndarray:
    """Section 3.1's boxed Mbar_mu^(e) = sum_k w_k mu_r^-1(r_k) -- invert
    mu_r pointwise, *then* integrate."""
    points, weights = _composite_points_weights(coords, level)
    mu = materials.mu(tag, points)  # (M,3,3)
    mu_inv = np.linalg.inv(mu)
    return np.einsum("k,kij->ij", weights, mu_inv)


def _mass_matrix(
    materials: MaterialAssembly, tag: str, coords: np.ndarray, grad: np.ndarray, sign: np.ndarray, level: int
) -> np.ndarray:
    """Section 3.2's boxed M_lm^(e) = sum_k w_k W_l(r_k) . (eps_r(r_k) W_m(r_k))."""
    points, weights = _composite_points_weights(coords, level)
    eps = materials.epsilon(tag, points)  # (M,3,3)
    W = _whitney_at_physical_points(grad, sign, coords[0], points)  # (6,M,3)
    eps_W = np.einsum("kij,mkj->mki", eps, W)
    return np.einsum("k,lki,mki->lm", weights, W, eps_W)


def _stiffness_matrix(C: np.ndarray, mu_bar: np.ndarray) -> np.ndarray:
    """Section 3.1: K_lm^(e) = C_l . (Mbar_mu C_m) -- cheap 3-vector
    bilinear products against the single cached tensor, no quadrature."""
    return np.einsum("li,ij,mj->lm", C, mu_bar, C)


def element_matrices(
    mesh: MeshInterface, tet: int, materials: MaterialAssembly, levels: Sequence[int] = DEFAULT_LEVELS
) -> tuple[np.ndarray, np.ndarray]:
    """Assembles `(K^(e), M^(e))` for one tet -- both `(6,6)` complex."""
    grad = mesh.grad_lambda(tet)
    sign = mesh.tet_edge_sign[tet]
    tag = mesh.tet_volume_tag(tet)
    coords = mesh.vertices[mesh.tets[tet]]

    C = whitney_curl(grad, sign)
    mu_bar = _adaptive_quadrature(lambda level: _mu_bar(materials, tag, coords, level), levels)
    K_e = _stiffness_matrix(C, mu_bar)

    M_e = _adaptive_quadrature(lambda level: _mass_matrix(materials, tag, coords, grad, sign, level), levels)

    return K_e, M_e


def assemble(
    mesh: MeshInterface,
    materials: MaterialAssembly,
    tet_subset: Sequence[int] | None = None,
    levels: Sequence[int] = DEFAULT_LEVELS,
) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """Section 5: global sparse assembly. Frequency-oblivious; `K`, `M`
    are complex-symmetric `(n_edges, n_edges)` sparse matrices over the
    full, *unconstrained* edge-DOF space -- no PEC elimination here
    (Section 5.3; that's Module 6's job via `mesh.pec_edge_dofs()`).

    `tet_subset` (Section 5.2): defaults to every tet; passing a subset is
    what lets a caller assemble the interior once (cached for a sweep) and
    the PML tets separately, per frequency, against a fresh
    `MaterialAssembly` -- `fem.assembly` itself has no notion of
    "interior" vs "PML", it only assembles whatever it's handed.
    """
    tets = range(mesh.n_tets) if tet_subset is None else tet_subset

    rows: list[int] = []
    cols: list[int] = []
    K_vals: list[complex] = []
    M_vals: list[complex] = []

    for tet in tets:
        edge_map = mesh.tet_edge_map[tet]  # (6,) local edge -> global edge index
        K_e, M_e = element_matrices(mesh, tet, materials, levels)
        for local_row in range(6):
            global_row = int(edge_map[local_row])
            for local_col in range(6):
                global_col = int(edge_map[local_col])
                rows.append(global_row)
                cols.append(global_col)
                K_vals.append(K_e[local_row, local_col])
                M_vals.append(M_e[local_row, local_col])

    n = mesh.n_edges
    K = sp.coo_matrix((K_vals, (rows, cols)), shape=(n, n), dtype=complex).tocsr()
    M = sp.coo_matrix((M_vals, (rows, cols)), shape=(n, n), dtype=complex).tocsr()
    return K, M
