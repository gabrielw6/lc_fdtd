"""fem.field_eval -- reconstructs the driven E-field (and, from it, H) at
arbitrary physical points from a full-length edge-DOF vector (e.g.
`SweepResult.a`).

Not itself a `docs/module*_equations.md` module -- a post-processing
utility for visualization (`viz.field_slice`), consuming `fem.assembly`'s
own basis exactly: `fem.edge_elements.whitney_basis` is that module's
"single source of truth" for the Whitney basis + the global-edge sign
convention (its own docstring's words) -- `mesh.tet_edge_sign`/
`mesh.tet_edge_map`, read the same way `fem.assembly.assemble` reads them
when building `K`/`M`. Reusing that function directly (rather than
re-deriving the basis formula here) is what guarantees a reconstructed
|E| agrees with what the solver actually assembled; re-deriving it
independently would risk a silent sign mismatch.

Point location mirrors `ports.mode_solver._locate_triangles`'s pattern
(barycentric containment test with a small negative tolerance), just in
3D against tets instead of in 2D against triangles: the same affine
reconstruction `fem.assembly._whitney_at_physical_points` already uses
internally for non-quadrature physical points (`lambda_i(r) = delta_{i,0}
+ grad_i . (r - vertex_0)`) gives each tet's barycentric coordinates at an
arbitrary point directly from `mesh.grad_lambda(tet)`, with no need for a
fresh matrix inversion per tet.
"""
from __future__ import annotations

import numpy as np

from mesh_interface import MeshInterface

from .edge_elements import whitney_basis, whitney_curl

_CONTAINMENT_TOL = 1e-7


def _tet_barycentric(mesh: MeshInterface, tet: int, points: np.ndarray) -> np.ndarray:
    """`lambda_i(r)` for every point in `points` (M,3), against tet `tet`
    -- the same affine formula `fem.assembly._whitney_at_physical_points`
    uses, re-derived here from `mesh.grad_lambda`/`mesh.vertices` (both
    public) rather than reaching into that private helper."""
    grad = mesh.grad_lambda(tet)  # (4,3)
    vertex0 = mesh.vertices[mesh.tets[tet, 0]]
    rel = points - vertex0[None, :]
    lam = np.zeros((points.shape[0], 4))
    lam[:, 0] = 1.0
    lam += rel @ grad.T
    return lam


def evaluate_edge_field(mesh: MeshInterface, a: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """Reconstructs `E(r) = sum_e a[e]*N_e(r)` at each of `points_xyz`
    (M,3) from a full-length (`mesh.n_edges`) driven DOF vector `a` (e.g.
    `SweepResult.a`). Brute-force tet location (acceptable for these mesh
    sizes, per an optional-but-unimplemented AABB prefilter): for each
    point, scan tets until one contains it (barycentric coordinates all
    `>= -tol`), then evaluate that tet's 6 Whitney edge functions
    (`fem.edge_elements.whitney_basis` -- the exact basis+sign convention
    `fem.assembly.assemble` used to build `K`/`M`) and dot with `a` at
    that tet's 6 global edge indices (`mesh.tet_edge_map`). A point inside
    no tet (outside the meshed domain) gets a row of `nan+nan*j`, marking
    it for the caller to mask rather than silently plotting zero."""
    points = np.atleast_2d(np.asarray(points_xyz, dtype=float))
    a = np.asarray(a)
    M = points.shape[0]
    out = np.full((M, 3), np.nan + 0j, dtype=complex)

    remaining = np.arange(M)
    for tet in range(mesh.n_tets):
        if remaining.size == 0:
            break
        lam = _tet_barycentric(mesh, tet, points[remaining])
        inside = np.all(lam >= -_CONTAINMENT_TOL, axis=1)
        if not np.any(inside):
            continue

        hit = remaining[inside]
        grad = mesh.grad_lambda(tet)
        sign = mesh.tet_edge_sign[tet]
        edge_map = mesh.tet_edge_map[tet]  # (6,) local edge -> global edge index

        W = whitney_basis(grad, sign)(lam[inside])  # (6, Mhit, 3)
        coeffs = a[edge_map].astype(complex)  # (6,)
        out[hit] = np.einsum("e,eMd->Md", coeffs, W)

        remaining = remaining[~inside]

    return out


def evaluate_curl_field(mesh: MeshInterface, a: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    """`curl(E)(r) = sum_e a[e]*C_e` -- constant per tet
    (`fem.edge_elements.whitney_curl`, the same function `fem.assembly`
    uses for `K^(e)`), so this only needs point *location*, not a fresh
    per-point basis evaluation. Same nan-for-outside-domain contract as
    `evaluate_edge_field`. `H = (1j/(omega*mu0))*curl(E)` (Module 4
    Section 3.8's identity, restated here for a full 3D driven field
    rather than a 2D port mode) is the caller's job -- this function
    returns `curl(E)` only, unscaled."""
    points = np.atleast_2d(np.asarray(points_xyz, dtype=float))
    a = np.asarray(a)
    M = points.shape[0]
    out = np.full((M, 3), np.nan + 0j, dtype=complex)

    remaining = np.arange(M)
    for tet in range(mesh.n_tets):
        if remaining.size == 0:
            break
        lam = _tet_barycentric(mesh, tet, points[remaining])
        inside = np.all(lam >= -_CONTAINMENT_TOL, axis=1)
        if not np.any(inside):
            continue

        hit = remaining[inside]
        grad = mesh.grad_lambda(tet)
        sign = mesh.tet_edge_sign[tet]
        edge_map = mesh.tet_edge_map[tet]

        C = whitney_curl(grad, sign)  # (6,3), constant over the tet
        coeffs = a[edge_map].astype(complex)  # (6,)
        out[hit] = coeffs @ C  # (Mhit,3) broadcast: same curl vector for every hit point

        remaining = remaining[~inside]

    return out
