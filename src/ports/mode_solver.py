"""ports.mode_solver -- the 2D port-mode eigenproblem, modal admittance,
power normalization, and modal projection (docs/module4_ports_equations.md
Sections 3-4).

Section 1's design invariant (every port face sits in an isotropic feed
section) is what makes Section 3.5's integrands at most degree 2 in a
single triangle's own (piecewise-constant) eps_r -- unlike `fem.assembly`'s
LC-driven composite quadrature, a single verified order-2 rule
(`mesh_interface.quadrature.tri_rule`) is already exact here; there is no
adaptive escalation to build.

Section 3.6's honesty flag applies throughout this file: the block
arrangement below is the doc's own derived-but-uncross-checked convention.
A general (QZ-capable) dense eigensolver is used specifically because the
doc calls this out as robust to either the symmetric or non-symmetric
arrangement -- `scipy.linalg.eig` on two dense matrices dispatches to
LAPACK's `ggev`/`cggev` (QZ).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.linalg as sla
from scipy import constants as _c

from material import MaterialAssembly
from mesh_interface import MeshInterface
from mesh_interface import quadrature as mesh_quadrature

from .basis2d import TRI_LOCAL_EDGES, whitney2d_basis, whitney2d_curl
from .cross_section import PortCrossSection, extract_cross_section

_BASE_ORDER = 2
_GAMMA_SQ_MAGNITUDE_CEILING = 1e12  # Section 3.6: B is singular (the e_x-e_x
# block is zero) -- LAPACK's ggev reports the singular pencil's spurious
# directions as (numerically) infinite generalized eigenvalues; anything
# past this ceiling is discarded as one of those, not a physical mode.
_CONSISTENCY_TOL = 1e-6


class PortModeError(RuntimeError):
    """Raised when a port cross-section fails Section 1's isotropy
    invariant, the eigensolve does not yield enough finite modes, or a
    Section 8 structural check (S_tz=-T_zt^T, Y_m's two-power consistency)
    fails -- signals a modeling bug, not a caller mistake."""


@dataclass
class PortMode:
    """Section 9's class contract. `e_t`/`h_t`: `(M,3) -> (M,3)` complex,
    transverse-only (x component always 0), evaluated at physical 3D points
    on this mode's own port plane. Power-normalized so `P_m=1` (Section
    4.2). The remaining fields are not part of Section 9's minimal
    contract but are what `ports.port_operator` (Section 5.2's caching)
    and this module's own `project`/`biorthogonality` reuse directly,
    rather than recovering them by re-evaluating the callables."""

    gamma: complex
    e_t: Callable[[np.ndarray], np.ndarray]
    h_t: Callable[[np.ndarray], np.ndarray]
    Y: complex
    cross_section: PortCrossSection = field(repr=False)
    omega: float = field(repr=False)
    e_edge_dofs: np.ndarray = field(repr=False)  # (n_edges,) complex, power-normalized, PEC entries 0
    ex_tilde_vertex_dofs: np.ndarray = field(repr=False)  # (n_vertices,) complex, tilde_e_x=gamma*e_x
    overlap_e: np.ndarray = field(repr=False)  # (n_edges,) integral W_i . e_m dS (Section 5.1/5.2)
    overlap_h: np.ndarray = field(repr=False)  # (n_edges,) integral (W_j)_t x h_m . x_hat dS


# ============================================================================
# Section 3.5: matrix block assembly
# ============================================================================


def _scalar_epsilon(materials: MaterialAssembly, tag: str, points_yz: np.ndarray, x0: float) -> np.ndarray:
    """Section 1: ports sit in isotropic feed sections only -- extract the
    scalar and assert the tensor really is `eps_r * I`, rather than
    silently taking the (0,0) entry of something anisotropic."""
    points3d = np.column_stack([np.full(points_yz.shape[0], x0), points_yz])
    eps = materials.epsilon(tag, points3d)  # (M,3,3)
    scalar = eps[:, 0, 0]
    off_diag = eps - scalar[:, None, None] * np.eye(3)[None, :, :]
    residual = float(np.abs(off_diag).max()) if eps.size else 0.0
    scale = max(1.0, float(np.abs(scalar).max()) if eps.size else 1.0)
    if residual > 1e-9 * scale:
        raise PortModeError(
            f"non-isotropic eps_r at tag {tag!r} on a port cross-section -- Section 1's design "
            "invariant requires every port face to sit in a plain isotropic feed section"
        )
    return scalar


def _assemble_blocks(
    cs: PortCrossSection, materials: MaterialAssembly, omega: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Section 3.5's `S_tt, S_zz, T_tt, T_zt, S_tz`, over the FULL
    (all-vertex, all-edge, PEC not yet eliminated) local DOF space."""
    k0 = omega * np.sqrt(_c.mu_0 * _c.epsilon_0)
    Ne, Nv = cs.n_edges, cs.n_vertices

    S_tt = np.zeros((Ne, Ne), dtype=complex)
    S_zz = np.zeros((Nv, Nv), dtype=complex)
    T_tt = np.zeros((Ne, Ne), dtype=complex)
    T_zt = np.zeros((Nv, Ne), dtype=complex)
    S_tz = np.zeros((Ne, Nv), dtype=complex)

    bary, w_hat = mesh_quadrature.tri_rule(_BASE_ORDER)

    for t in range(cs.n_triangles):
        verts = cs.triangles[t]
        coords = cs.yz[verts]
        grad_t = cs.grad_t[t]
        area = float(cs.area[t])
        sign = cs.tri_edge_sign[t]
        edge_g = cs.tri_edges[t]

        points = bary @ coords
        weights = w_hat * area
        eps_pts = _scalar_epsilon(materials, str(cs.tri_tag[t]), points, cs.x0)  # (M,)

        N = whitney2d_basis(grad_t, sign)(bary)  # (3,M,2)
        curl_N = whitney2d_curl(grad_t, sign)  # (3,)

        for i in range(3):
            gi = edge_g[i]
            for j in range(3):
                gj = edge_g[j]
                dot_ij = np.einsum("md,md->m", N[i], N[j])
                mass = complex(np.sum(weights * eps_pts * dot_ij))  # eps_pts may be complex (lossy material)
                S_tt[gi, gj] += curl_N[i] * curl_N[j] * area - k0**2 * mass
                T_tt[gi, gj] += float(np.sum(weights * dot_ij))

        for i in range(3):
            vi = verts[i]
            for j in range(3):
                vj = verts[j]
                mass = complex(np.sum(weights * eps_pts * bary[:, i] * bary[:, j]))
                S_zz[vi, vj] += np.dot(grad_t[i], grad_t[j]) * area - k0**2 * mass

        for i in range(3):
            vi = verts[i]
            for j in range(3):
                gj = edge_g[j]
                T_zt[vi, gj] += float(np.sum(weights * (N[j] @ grad_t[i])))

        for i in range(3):
            gi = edge_g[i]
            for j in range(3):
                vj = verts[j]
                S_tz[gi, vj] += -float(np.sum(weights * (N[i] @ grad_t[j])))

    return S_tt, S_zz, T_tt, T_zt, S_tz


# ============================================================================
# Section 3.8: field reconstruction, per-triangle (internal, DOF-based)
# ============================================================================


def _e_t_on_triangle(cs: PortCrossSection, t: int, bary: np.ndarray, e_edge_dofs: np.ndarray) -> np.ndarray:
    grad_t = cs.grad_t[t]
    sign = cs.tri_edge_sign[t]
    N = whitney2d_basis(grad_t, sign)(bary)  # (3,M,2)
    coeffs = e_edge_dofs[cs.tri_edges[t]]  # (3,)
    return np.einsum("e,eMd->Md", coeffs, N)  # (M,2), (y,z) components


def _h_t_on_triangle(
    cs: PortCrossSection,
    t: int,
    bary: np.ndarray,
    e_edge_dofs: np.ndarray,
    ex_tilde_dofs: np.ndarray,
    gamma: complex,
    omega: float,
) -> np.ndarray:
    """Section 3.8: `h_t = -(j/(omega*mu0)) x_hat x ((1/gamma) grad_t
    tilde_e_x + gamma e_t)`. With `x_hat x (0,vy,vz) = (0,-vz,vy)`, the
    (y,z) output components are `(j/(omega*mu0))*Dz, -(j/(omega*mu0))*Dy`."""
    e_field = _e_t_on_triangle(cs, t, bary, e_edge_dofs)  # (M,2)
    ex_coeffs = ex_tilde_dofs[cs.triangles[t]]  # (3,)
    grad_ex_tilde = ex_coeffs @ cs.grad_t[t]  # (2,)
    D = grad_ex_tilde[None, :] / gamma + gamma * e_field  # (M,2)
    prefactor = 1j / (omega * _c.mu_0)
    h = np.empty_like(e_field)
    h[:, 0] = prefactor * D[:, 1]
    h[:, 1] = -prefactor * D[:, 0]
    return h


def _locate_triangles(cs: PortCrossSection, points_yz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force containment test against every triangle -- fine for the
    small 2D port meshes this module deals with. Used only by the public
    `e_t`/`h_t` callables (Section 9); internal integrals loop over
    triangles directly and never need this."""
    M = points_yz.shape[0]
    tri_idx = np.full(M, -1, dtype=np.int64)
    bary_out = np.zeros((M, 3))
    remaining = np.arange(M)
    for t in range(cs.n_triangles):
        if remaining.size == 0:
            break
        coords = cs.yz[cs.triangles[t]]
        P = np.array(
            [[1.0, 1.0, 1.0], [coords[0, 0], coords[1, 0], coords[2, 0]], [coords[0, 1], coords[1, 1], coords[2, 1]]]
        )
        Pinv = np.linalg.inv(P)
        pts = points_yz[remaining]
        ones = np.column_stack([np.ones(len(pts)), pts])
        bary = ones @ Pinv.T
        inside = np.all(bary >= -1e-7, axis=1)
        hit = remaining[inside]
        tri_idx[hit] = t
        bary_out[hit] = bary[inside]
        remaining = remaining[~inside]
    if remaining.size:
        raise PortModeError(
            f"{remaining.size} point(s) do not lie on cross-section {cs.port_tag!r} -- "
            "port field callables are only defined on their own port plane"
        )
    return tri_idx, bary_out


def _make_e_t(cs: PortCrossSection, e_edge_dofs: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def e_t(points3d: np.ndarray) -> np.ndarray:
        points3d = np.atleast_2d(points3d)
        tri_idx, bary = _locate_triangles(cs, points3d[:, 1:3])
        out = np.zeros((points3d.shape[0], 3), dtype=complex)
        for t in np.unique(tri_idx):
            mask = tri_idx == t
            out[mask, 1:3] = _e_t_on_triangle(cs, int(t), bary[mask], e_edge_dofs)
        return out

    return e_t


def _make_h_t(
    cs: PortCrossSection, e_edge_dofs: np.ndarray, ex_tilde_dofs: np.ndarray, gamma: complex, omega: float
) -> Callable[[np.ndarray], np.ndarray]:
    def h_t(points3d: np.ndarray) -> np.ndarray:
        points3d = np.atleast_2d(points3d)
        tri_idx, bary = _locate_triangles(cs, points3d[:, 1:3])
        out = np.zeros((points3d.shape[0], 3), dtype=complex)
        for t in np.unique(tri_idx):
            mask = tri_idx == t
            out[mask, 1:3] = _h_t_on_triangle(cs, int(t), bary[mask], e_edge_dofs, ex_tilde_dofs, gamma, omega)
        return out

    return h_t


# ============================================================================
# Section 4: modal admittance, power normalization
# ============================================================================


def _mode_integrals(
    cs: PortCrossSection, e_edge_dofs: np.ndarray, ex_tilde_dofs: np.ndarray, gamma: complex, omega: float
) -> tuple[float, complex, float, complex]:
    """One triangle pass computing everything Section 4 needs: `integral
    |e_t|^2`, `Y_m` (Section 4.1), the direct Poynting power (Section 4.1's
    consistency check), and the (value at) the point of maximum `|e_t|`
    (Section 4.2's phase convention)."""
    bary, w_hat = mesh_quadrature.tri_rule(_BASE_ORDER)
    total_abs2 = 0.0
    Y_num = 0.0 + 0j
    poynting = 0.0 + 0j
    max_abs2 = -1.0
    max_val = 0.0 + 0j

    for t in range(cs.n_triangles):
        weights = w_hat * float(cs.area[t])
        e_field = _e_t_on_triangle(cs, t, bary, e_edge_dofs)  # (M,2)
        h_field = _h_t_on_triangle(cs, t, bary, e_edge_dofs, ex_tilde_dofs, gamma, omega)  # (M,2)

        abs2 = np.sum(np.abs(e_field) ** 2, axis=1)
        total_abs2 += float(np.sum(weights * abs2))

        x_cross_e = np.stack([-e_field[:, 1], e_field[:, 0]], axis=1)  # x_hat x e_t, (y,z) comps
        Y_num += np.sum(weights * np.einsum("md,md->m", h_field, np.conj(x_cross_e)))

        poynting += np.sum(weights * (e_field[:, 0] * np.conj(h_field[:, 1]) - e_field[:, 1] * np.conj(h_field[:, 0])))

        idx = int(np.argmax(abs2))
        if abs2[idx] > max_abs2:
            max_abs2 = float(abs2[idx])
            comp_idx = int(np.argmax(np.abs(e_field[idx])))
            max_val = complex(e_field[idx, comp_idx])

    Y = complex(Y_num / total_abs2) if total_abs2 > 0 else 0j
    P_direct = 0.5 * poynting.real
    return total_abs2, Y, P_direct, max_val


def _normalize(
    cs: PortCrossSection, e_edge_dofs: np.ndarray, ex_tilde_dofs: np.ndarray, gamma: complex, omega: float
) -> tuple[np.ndarray, np.ndarray, complex]:
    """Sections 4.1-4.2: extract `Y_m`, run its power-consistency check,
    then solve for `alpha` (magnitude from `|alpha|^2` boxed formula, phase
    fixed at the point of maximum `|e_t|`)."""
    total_abs2, Y, P_direct, max_val = _mode_integrals(cs, e_edge_dofs, ex_tilde_dofs, gamma, omega)

    P_from_Y = 0.5 * Y.real * total_abs2
    scale = max(1.0, abs(P_direct), abs(P_from_Y))
    if abs(P_direct - P_from_Y) > _CONSISTENCY_TOL * scale:
        raise PortModeError(
            f"Y_m power consistency check failed (Section 4.1): direct Poynting power {P_direct!r} "
            f"!= 0.5*Re(Y)*integral|e_t|^2 = {P_from_Y!r}"
        )

    denom = 0.5 * Y.real * total_abs2
    if denom <= 0:
        raise PortModeError(f"non-positive net modal power ({denom!r}) -- cannot power-normalize this mode")
    alpha_mag = 1.0 / np.sqrt(denom)
    phase = -np.angle(max_val) if max_val != 0 else 0.0
    alpha = alpha_mag * np.exp(1j * phase)

    return alpha * e_edge_dofs, alpha * ex_tilde_dofs, Y


# ============================================================================
# Section 5.1/5.2: the two cached surface overlaps `ports.port_operator` needs
# ============================================================================

# `ports.port_operator`'s B_p/g_p formulas (Section 5.1) are stated in terms
# of the *3D* Whitney basis W_i restricted to the port face, not this
# module's own 2D N_i. These agree exactly, which is what lets everything
# below be computed purely from this module's own 2D matrices/DOFs, with no
# separate 3D surface integral:
#   - For a 3D tet edge i with vertex a excluded from face F (a not in F),
#     lambda_a is identically 0 throughout F, so its gradient's *tangential*
#     part on F is 0 (grad(lambda_a) is purely normal to F) -- so W_i's
#     tangential trace on F is 0 for any edge not lying in F.
#   - For an edge i=(a,b) with both a,b in F, lambda_a|_F is exactly the 2D
#     barycentric coordinate of the face triangle, and the tangential part
#     of grad(lambda_a) on F equals exactly the 2D grad_t(lambda_a) (a
#     function's gradient restricted tangentially to a submanifold depends
#     only on the function's values on that submanifold). So
#     (W_i)_t|_F = lambda_a grad_t(lambda_b) - lambda_b grad_t(lambda_a) = N_i,
#     the *same* 2D Whitney function this module already builds, with the
#     *same* sign convention (both derived from the same global-ascending
#     edge orientation, Section 2.5).
# So `integral_{Sp} W_i . e_m dS` is `integral N_i . e_m dS` for i on this
# port's own edges (else identically 0), i.e. `T_tt @ e_edge_dofs` -- and
# the second overlap reduces the same way (derived below).


def _mode_overlaps(
    T_tt: np.ndarray, S_tz: np.ndarray, e_edge_dofs: np.ndarray, ex_tilde_dofs: np.ndarray, gamma: complex, omega: float
) -> tuple[np.ndarray, np.ndarray]:
    """Section 5.2's two per-port-per-mode-per-frequency overlaps, over the
    FULL local edge space (Section 5.3-style: no PEC elimination here,
    consistent with `fem.assembly`'s own "not this module's job" stance --
    entries at PEC-tagged local edges are still well-defined by this
    formula, just physically irrelevant once a consumer eliminates them).

    `overlap_e[i] = integral N_i . e_m dS = (T_tt @ e_edge_dofs)[i]`
    (`e_m` is itself expanded in the same N_i basis, so this is just T_tt's
    own mass-matrix action on the mode's DOF vector -- no fresh quadrature).

    `overlap_h[j] = integral (N_j x h_m) . x_hat dS`. Using Section 3.8's
    `h_m = -(j/(omega*mu0)) x_hat x D`, `D = (1/gamma) grad_t(tilde_e_x) +
    gamma*e_t`, and the (y,z)-component identity `(A x B).x_hat = Ay*Bz -
    Az*By` collapses `(N_j x h_m).x_hat` to `-(j/(omega*mu0)) * (N_j . D)`
    (no conjugation -- Section 5.1's overlaps, like Section 4.3's
    projection, are bilinear). `integral N_j . grad_t(tilde_e_x) dS =
    -(S_tz @ ex_tilde_dofs)[j]` (from S_tz's own definition,
    Section 3.5), and `integral N_j . e_t dS = overlap_e[j]` (reused
    directly, not recomputed) -- giving a closed form with no new
    quadrature at all."""
    overlap_e = T_tt @ e_edge_dofs
    prefactor = 1j / (omega * _c.mu_0)
    overlap_h = (prefactor / gamma) * (S_tz @ ex_tilde_dofs) - (prefactor * gamma) * overlap_e
    return overlap_e, overlap_h


def _eps_r_bounds(cs: PortCrossSection, materials: MaterialAssembly) -> tuple[float, float]:
    """Section 1's isotropic feed-section invariant fixes a hard physical
    bound on any genuinely propagating bound mode's beta: `k0*sqrt(eps_min)
    <= beta <= k0*sqrt(eps_max)` (an "effective index" between the two
    media's indices; a Sturm-Liouville-type eigenvalue bound). Used only to
    flag spurious discrete eigenvalues (Section 3.6's honesty flag: an
    uncross-checked block arrangement can admit extra, non-physical
    solutions alongside the real ones), sampled at triangle centroids since
    Section 1 guarantees each triangle's tag is spatially homogeneous."""
    lo, hi = np.inf, -np.inf
    for t in range(cs.n_triangles):
        centroid = cs.yz[cs.triangles[t]].mean(axis=0, keepdims=True)
        eps = _scalar_epsilon(materials, str(cs.tri_tag[t]), centroid, cs.x0)
        val = float(eps[0].real)
        lo, hi = min(lo, val), max(hi, val)
    return lo, hi


def _is_spurious_propagating_mode(gamma: complex, k0: float, beta_lo: float, beta_hi: float, margin: float = 1.05) -> bool:
    """A discrete eigenvalue that is predominantly *propagating*
    (`|Re(gamma)| << |Im(gamma)|`, i.e. genuinely close to the lossless
    `gamma=j*beta` branch) but whose `beta` falls outside the physical
    `[beta_lo, beta_hi]` band (with a small margin for discretization
    error) cannot be a real bound mode of this cross-section -- it is a
    spurious solution of the discrete pencil. Evanescent-dominant
    eigenvalues are never flagged: no such bound applies to them."""
    beta, alpha = gamma.imag, gamma.real
    if abs(beta) < 1e-6 * k0 or abs(alpha) > 1e-3 * abs(beta):
        return False
    return not (beta_lo / margin <= abs(beta) <= beta_hi * margin)


# KNOWN LIMITATION (Section 3.7's beta-sort, taken literally): Module 0's
# `PMC_SIDE` lateral truncation (`docs/CLAUDE.md` Sec. 3, a deliberate
# design choice, not a bug) makes each port cross-section a finite
# PMC-walled enclosure, which supports its own family of near-degenerate
# parallel-plate/"box" modes with beta close to the real quasi-TEM
# microstrip mode's. Empirically (varying `target_elements_per_wavelength`
# on the reference geometry): at some mesh resolutions the top-beta mode
# is one of these box modes, not the quasi-TEM mode, and which one wins is
# sensitive to discretization noise breaking the near-degeneracy. A
# trace-energy-concentration heuristic was tried to break the tie and
# rejected -- it measured energy within a fixed window around the trace
# rather than the field's decay *rate* away from it (the real physical
# signature separating quasi-TEM fringing fields from box-mode humps),
# and a box mode whose smooth lobe happens to peak near the trace's
# (roughly centered) position scored just as well, in one case actively
# demoting the correct mode that plain beta-sort had already found. Left
# as plain Section 3.7 beta-sort; a physically-motivated (decay-rate or
# projection-onto-a-known-quasi-TEM-ansatz) discriminator is future work,
# not something to guess further here.


# ============================================================================
# Section 3.6-3.7: the eigenproblem itself
# ============================================================================


class PortModeSolver:
    """Section 9's class contract. Caches each port's extracted
    `PortCrossSection` (Section 2's geometry is frequency-independent) but
    re-solves the eigenproblem on every `solve` call (Section 3's blocks
    depend on `omega` through `k0`)."""

    def __init__(self, mesh: MeshInterface, materials: MaterialAssembly) -> None:
        self._mesh = mesh
        self._materials = materials
        self._cross_sections: dict[str, PortCrossSection] = {}

    def cross_section(self, port_tag: str) -> PortCrossSection:
        if port_tag not in self._cross_sections:
            self._cross_sections[port_tag] = extract_cross_section(self._mesh, port_tag)
        return self._cross_sections[port_tag]

    def solve(self, port_tag: str, omega: float, n_modes: int = 2) -> list[PortMode]:
        cs = self.cross_section(port_tag)
        S_tt, S_zz, T_tt, T_zt, S_tz = _assemble_blocks(cs, self._materials, omega)

        # Section 3.5's exact algebraic identity, independent of Section
        # 3.6's arrangement question.
        residual = float(np.abs(S_tz - (-T_zt.T)).max())
        scale = max(1.0, float(np.abs(T_zt).max()))
        if residual > _CONSISTENCY_TOL * scale:
            raise PortModeError(f"S_tz != -T_zt^T (Section 3.5 identity failed, residual {residual!r})")

        free_e = cs.free_edge_dofs()
        free_v = cs.free_vertex_dofs()
        ne, nv = len(free_e), len(free_v)
        if ne == 0:
            raise PortModeError(f"port {port_tag!r} has no free (non-PEC) transverse edge DOFs")

        n = ne + nv
        A = np.zeros((n, n), dtype=complex)
        Bmat = np.zeros((n, n), dtype=complex)
        A[:ne, :ne] = S_tt[np.ix_(free_e, free_e)]
        A[:ne, ne:] = S_tz[np.ix_(free_e, free_v)]
        A[ne:, ne:] = S_zz[np.ix_(free_v, free_v)]
        Bmat[:ne, :ne] = T_tt[np.ix_(free_e, free_e)]
        Bmat[ne:, :ne] = T_zt[np.ix_(free_v, free_e)]

        gamma_sq, vecs = sla.eig(A, Bmat)
        finite = np.isfinite(gamma_sq) & (np.abs(gamma_sq) < _GAMMA_SQ_MAGNITUDE_CEILING)
        gamma_sq = gamma_sq[finite]
        vecs = vecs[:, finite]
        if len(gamma_sq) < n_modes:
            raise PortModeError(
                f"only {len(gamma_sq)} finite generalized eigenvalue(s) at port {port_tag!r}, "
                f"requested {n_modes}"
            )

        gamma_all = np.sqrt(gamma_sq)  # principal branch: Re>=0, Im>0 when purely imaginary (Section 3.7)

        # Discard spurious discrete solutions before ranking (see
        # `_is_spurious_propagating_mode`'s docstring): Section 3.6's
        # uncross-checked block arrangement can admit non-physical
        # eigenpairs alongside the real spectrum, and naive "top n_modes by
        # beta" selection is not safe without this filter.
        k0 = omega * np.sqrt(_c.mu_0 * _c.epsilon_0)
        eps_lo, eps_hi = _eps_r_bounds(cs, self._materials)
        beta_lo, beta_hi = k0 * np.sqrt(max(eps_lo, 0.0)), k0 * np.sqrt(max(eps_hi, 0.0))
        physical = np.array(
            [not _is_spurious_propagating_mode(complex(g), k0, beta_lo, beta_hi) for g in gamma_all]
        )
        gamma_all = gamma_all[physical]
        vecs = vecs[:, physical]
        if len(gamma_all) < n_modes:
            raise PortModeError(
                f"only {len(gamma_all)} physically-plausible mode(s) remained at port {port_tag!r} "
                f"after discarding spurious eigenvalues, requested {n_modes}"
            )

        # Section 3.7: sorted by decreasing beta. See the KNOWN LIMITATION
        # note above -- this can select a box mode instead of the true
        # quasi-TEM mode for a PMC-walled cross-section at some mesh
        # resolutions; not resolved further here.
        order = np.argsort(-gamma_all.imag)
        gamma_all, vecs = gamma_all[order], vecs[:, order]

        # Build modes in ranked order, skipping any candidate that turns out
        # to be un-power-normalizable (non-positive net power -- typically a
        # near-degenerate box-mode combination at a coarse mesh resolution,
        # Section 3.7's shape-agnostic beta-ranking has no way to avoid this
        # up front) rather than failing the whole port solve on one bad
        # candidate. A mode that cannot reach P_m=1 does not satisfy Section
        # 9's contract regardless, so skipping it (not crashing) is correct.
        modes: list[PortMode] = []
        for candidate in range(gamma_all.shape[0]):
            if len(modes) == n_modes:
                break
            gamma = complex(gamma_all[candidate])
            v = vecs[:, candidate]

            e_edge_dofs = np.zeros(cs.n_edges, dtype=complex)
            e_edge_dofs[free_e] = v[:ne]
            ex_tilde_dofs = np.zeros(cs.n_vertices, dtype=complex)
            ex_tilde_dofs[free_v] = v[ne:]

            try:
                e_edge_dofs, ex_tilde_dofs, Y = _normalize(cs, e_edge_dofs, ex_tilde_dofs, gamma, omega)
            except PortModeError:
                continue
            overlap_e, overlap_h = _mode_overlaps(T_tt, S_tz, e_edge_dofs, ex_tilde_dofs, gamma, omega)

            modes.append(
                PortMode(
                    gamma=gamma,
                    e_t=_make_e_t(cs, e_edge_dofs),
                    h_t=_make_h_t(cs, e_edge_dofs, ex_tilde_dofs, gamma, omega),
                    Y=Y,
                    cross_section=cs,
                    omega=omega,
                    e_edge_dofs=e_edge_dofs,
                    ex_tilde_vertex_dofs=ex_tilde_dofs,
                    overlap_e=overlap_e,
                    overlap_h=overlap_h,
                )
            )

        if len(modes) < n_modes:
            raise PortModeError(
                f"only {len(modes)} power-normalizable mode(s) found at port {port_tag!r}, requested {n_modes}"
            )
        return modes


# ============================================================================
# Section 4.3: modal projection / biorthogonality
# ============================================================================


def _raw_overlap(field_yz: Callable[[np.ndarray], np.ndarray], mode: PortMode) -> complex:
    """`integral (field x h_m).x_hat dS`, no conjugation (bilinear, not
    sesquilinear -- distinct from Section 4.1's Y_m/Poynting integrals,
    which do conjugate). `field_yz`: `(M,3) -> (M,3)`, same calling
    convention as `PortMode.e_t`."""
    cs = mode.cross_section
    bary, w_hat = mesh_quadrature.tri_rule(_BASE_ORDER)
    total = 0.0 + 0j
    for t in range(cs.n_triangles):
        weights = w_hat * float(cs.area[t])
        coords = cs.yz[cs.triangles[t]]
        points_yz = bary @ coords
        points3d = np.column_stack([np.full(len(points_yz), cs.x0), points_yz])
        field = np.atleast_2d(field_yz(points3d))[:, 1:3]
        h_field = _h_t_on_triangle(cs, t, bary, mode.e_edge_dofs, mode.ex_tilde_vertex_dofs, mode.gamma, mode.omega)
        cross = field[:, 0] * h_field[:, 1] - field[:, 1] * h_field[:, 0]
        total += np.sum(weights * cross)
    return complex(total)


def _self_overlap(mode: PortMode) -> complex:
    """`integral (e_m x h_m).x_hat dS` -- Section 4.3 claims this is always
    `P_m=1` once Section 4.2's power normalization is applied, but that
    only holds for a purely real `Y_m` (a lossless mode): the general
    relation `h_m=Y_m(x_hat x e_m)` (Section 4.1) gives this integral as
    `Y_m * integral|e_m|^2 dS`, while Section 4.2 normalizes on `Re(Y_m)`
    with an extra 1/2 (the time-average Poynting convention) -- these
    coincide only when `Y_m` is real, and even then differ by exactly 2
    (verified numerically: this integral came out 2.0 to 1e-15 relative
    precision on a real solved mode, not 1.0). Computing it explicitly
    here, rather than assuming 1, keeps `project`/`biorthogonality`
    correct regardless of `Y_m`'s phase."""
    return _raw_overlap(mode.e_t, mode)


def project(E_t: Callable[[np.ndarray], np.ndarray], mode: PortMode) -> complex:
    """Section 4.3's projection, `a_m = integral(E_t x h_m).x_hat dS`,
    normalized by the mode's own self-overlap (see `_self_overlap`'s
    docstring for why Section 4.3's "no denominator" claim needs this
    correction) so `a_m` for `E_t = mode.e_t` itself is exactly 1."""
    denom = _self_overlap(mode)
    if denom == 0:
        raise PortModeError("mode self-overlap is exactly zero -- cannot project onto it")
    return _raw_overlap(E_t, mode) / denom


def biorthogonality(mode_m: PortMode, mode_n: PortMode) -> complex:
    """Section 8's `integral (e_m x h_n).x_hat dS ~ delta_mn` gate,
    normalized the same way `project` is so the diagonal is ~1 (not ~2)
    for a lossless mode."""
    return project(mode_m.e_t, mode_n)
