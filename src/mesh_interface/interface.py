"""mesh_interface.interface -- MeshInterface: per-element geometry, global
edge topology, boundary-face tagging, and quadrature
(docs/module1_mesh_interface_equations.md Sections 2-7).

Consumes four plain arrays (vertices, tets, volume_tags, surface_tags) --
"a list of vertices and tetrahedra with physical tags" (doc Section 0) --
and owns everything downstream FEM modules need: grad_lambda_i, signed
volume, face areas/normals, the global oriented edge list, boundary-face
tag resolution, and quadrature. Nothing here is frequency- or material-
dependent; it is built once (Section 7) and read many times.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import quadrature

DimTag = tuple[int, int]

# Section 3.1: fixed local edge/face numbering.
LOCAL_EDGES: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
LOCAL_FACES: tuple[tuple[int, int, int], ...] = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))

# For each local face, the local edge indices whose both endpoints lie on
# that face -- computed from LOCAL_EDGES/LOCAL_FACES above rather than
# hand-transcribed, so it can never drift out of sync with them. Used by
# `pec_edge_dofs()`.
LOCAL_FACE_EDGES: tuple[tuple[int, ...], ...] = tuple(
    tuple(local_edge for local_edge, (a, b) in enumerate(LOCAL_EDGES) if a in face and b in face)
    for face in LOCAL_FACES
)

# Section 5.3: tag resolution -- Module 0's concrete surface-tag names
# aggregated under Module 1's generic vocabulary.
_PEC_TAGS: tuple[str, ...] = ("PEC_GROUND", "PEC_LINE")
_PML_OUTER_TAGS: tuple[str, ...] = ("PML_OUTER_PEC",)


class MeshGeometryError(RuntimeError):
    """Raised when a load-time consistency check (Sections 2.5, 3.4, 4,
    5.2, 5.3) fails -- signals a corrupt/non-conforming mesh, not a bad
    caller argument."""


class MeshInterface:
    """Section 9's class contract. Construct via `MeshInterface(vertices,
    tets, volume_tags, surface_tags)` directly, or `from_mesh_handle(...)`
    for a `geometry_builder.MeshHandle` (the only place this module
    touches `geometry_builder` at all)."""

    def __init__(
        self,
        vertices: np.ndarray,
        tets: np.ndarray,
        volume_tags: np.ndarray,
        surface_tags: dict[str, np.ndarray],
    ) -> None:
        self.vertices = np.asarray(vertices, dtype=float)
        self.volume_tags = np.asarray(volume_tags)

        # Step 1 (Section 2.3): orientation normalization -- downstream
        # code may assume V > 0 unconditionally after this.
        self.tets = _orient_tets(self.vertices, np.asarray(tets, dtype=np.int64))

        # Step 2 (Section 2): per-tet geometry.
        self._grad_lambda, self._volume, alpha = _tet_geometry(self.vertices, self.tets)
        _check_tet_geometry(self.vertices, self.tets, self._grad_lambda, self._volume, alpha)

        # Step 3 (Section 4): face areas + outward normals.
        self._face_area, self._face_normal = _face_geometry(self.vertices, self.tets)
        _check_face_geometry(self._grad_lambda, self._volume, self._face_area, self._face_normal)

        # Step 4 (Section 3): global edge topology.
        self.edges, self.tet_edge_map, self.tet_edge_sign = _edge_topology(self.tets)
        _check_edge_topology(self.tets, self.edges, self.tet_edge_map, self.tet_edge_sign)

        # Step 5 (Section 5): boundary extraction + tag resolution.
        face_lookup, boundary_keys = _face_incidence(self.tets)
        self._tagged_faces = _resolve_tags(face_lookup, surface_tags)
        _check_boundary_coverage(boundary_keys, surface_tags)

    @classmethod
    def from_mesh_handle(cls, handle: Any) -> "MeshInterface":
        """Convenience constructor for `geometry_builder.MeshHandle` (duck-
        typed on `.vertices`/`.tets`/`.volume_tags`/`.surface_tags` rather
        than importing the type, so this module has no hard dependency on
        `geometry_builder` -- Module 0's own doc: "Module 1 is unaware
        Module 0 exists; it only ever sees a tagged mesh.")."""
        return cls(handle.vertices, handle.tets, handle.volume_tags, handle.surface_tags)

    # --- Section 9: sizes ---

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_tets(self) -> int:
        return self.tets.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edges.shape[0]

    # --- Section 9: per-element geometry ---

    def grad_lambda(self, tet: int) -> np.ndarray:
        """The four constant barycentric gradients of tet `tet`, (4,3)."""
        return self._grad_lambda[tet]

    def volume(self, tet: int) -> float:
        """V > 0 (Section 2.3's orientation fix)."""
        return float(self._volume[tet])

    def face_area_normal(self, tet: int, local_face: int) -> tuple[float, np.ndarray]:
        """(area, outward unit normal) of `tet`'s face opposite local
        vertex `local_face` (Section 3.1's naming)."""
        return float(self._face_area[tet, local_face]), self._face_normal[tet, local_face]

    def tet_volume_tag(self, tet: int) -> str:
        """Module 3 §1 item 2: the region tag governing `tet` -- what
        Module 3's assembler dispatches its `MaterialAssembly` query on."""
        return str(self.volume_tags[tet])

    # --- Section 9: boundary ---

    def boundary_faces(self, tag: str) -> list[DimTag]:
        """`(tet, local_face)` pairs for `tag` (Section 5.3):
        `'PEC'` = `PEC_LINE` union `PEC_GROUND`; `'PML_OUTER'` =
        `PML_OUTER_PEC`; anything else (`'PORT_1'`, `'PMC_SIDE'`, ...) is
        looked up by that exact name in the input `surface_tags`. Faces
        that are topologically interior (e.g. the embedded `PEC_LINE`
        sheet, incidence 2) return one entry per incident tet."""
        if tag == "PEC":
            names: tuple[str, ...] = _PEC_TAGS
        elif tag == "PML_OUTER":
            names = _PML_OUTER_TAGS
        else:
            names = (tag,)
        out: list[DimTag] = []
        for name in names:
            out.extend(self._tagged_faces.get(name, []))
        return out

    def pec_edge_dofs(self) -> set[int]:
        """Module 3 §1 item 3: global edge indices lying on a `PEC`-tagged
        face -- the DOF set Module 6 eliminates when applying the
        essential BC. Purely geometric (derived once, here, from data this
        class already owns); Module 3 itself never applies it."""
        dofs: set[int] = set()
        for tet, local_face in self.boundary_faces("PEC"):
            for local_edge in LOCAL_FACE_EDGES[local_face]:
                dofs.add(int(self.tet_edge_map[tet, local_edge]))
        return dofs

    # --- Section 9: quadrature ---

    def quadrature_tet(self, tet: int, order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Section 6.1's mapping applied to `tet`: `sum(weights) ==
        volume(tet)`. Also returns the `(M,4)` barycentric weights used to
        build the physical points -- Module 3 §1 item 1: it needs them
        directly to evaluate its basis functions at the same quadrature
        points, without reconstructing lambda_i from the affine
        coefficients. (Section 9 lists `quadrature_tet` without a `tet`
        argument, but Section 6.1's own mapping -- r_q = sum_i lambda_i
        r_i, scaled by this element's V -- is inherently per-element; the
        argument is added here so the documented `sum(weights)==V`
        contract is actually achievable.)"""
        bary, w_hat = quadrature.tet_rule(order)
        coords = self.vertices[self.tets[tet]]
        points = bary @ coords
        weights = w_hat * self._volume[tet]
        return points, weights, bary

    def quadrature_tri(self, face: DimTag, order: int) -> tuple[np.ndarray, np.ndarray]:
        """Section 6.1's mapping for a triangular face `(tet,
        local_face)`: `sum(weights) == area`."""
        tet, local_face = face
        local_vertices = LOCAL_FACES[local_face]
        coords = self.vertices[self.tets[tet][list(local_vertices)]]
        bary, w_hat = quadrature.tri_rule(order)
        points = bary @ coords
        weights = w_hat * self._face_area[tet, local_face]
        return points, weights


# ============================================================================
# Section 2: per-tet geometry
# ============================================================================


def _build_P(coords: np.ndarray) -> np.ndarray:
    """Section 2.2's geometry matrix for every tet at once. `coords`:
    (Nt,4,3), vertex j's (x,y,z) in coords[:,j,:]. Returns P: (Nt,4,4) with
    row 0 = ones, rows 1-3 = x/y/z of vertices 0..3 (columns) -- matching
    P[lambda_0..3]^T = [1,x,y,z]^T exactly."""
    Nt = coords.shape[0]
    P = np.empty((Nt, 4, 4))
    P[:, 0, :] = 1.0
    P[:, 1, :] = coords[:, :, 0]
    P[:, 2, :] = coords[:, :, 1]
    P[:, 3, :] = coords[:, :, 2]
    return P


def _orient_tets(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Section 2.3: permute each tet's local vertex order (swap the last
    two local indices, which negates det P) so det P > 0 for every
    element."""
    coords = vertices[tets]
    det = np.linalg.det(_build_P(coords))
    negative = det < 0
    fixed = tets.copy()
    fixed[negative, 2], fixed[negative, 3] = tets[negative, 3], tets[negative, 2]
    return fixed


def _tet_geometry(vertices: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Section 2.2: one 4x4 inversion per tet yields alpha_i and
    grad_lambda_i at once; Section 2.3: 6V = det(P)."""
    coords = vertices[tets]
    P = _build_P(coords)
    det = np.linalg.det(P)
    volume = det / 6.0
    Q = np.linalg.inv(P)
    alpha = Q[:, :, 0]
    grad_lambda = Q[:, :, 1:4]
    return grad_lambda, volume, alpha


def _characteristic_length(vertices: np.ndarray) -> float:
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    return float(max(extents.max(), 1.0))


def _check_tet_geometry(
    vertices: np.ndarray, tets: np.ndarray, grad_lambda: np.ndarray, volume: np.ndarray, alpha: np.ndarray
) -> None:
    # Degeneracy guard (Section 2.5): a floor scaled by the model's own
    # characteristic length cubed -- raise rather than let downstream code
    # divide by a near-zero volume.
    char_length = _characteristic_length(vertices)
    floor = 1e-12 * char_length**3
    if np.any(volume <= floor):
        bad = int(np.argwhere(volume <= floor).ravel()[0])
        raise MeshGeometryError(
            f"tet {bad} has near-zero/negative volume ({volume[bad]!r} <= floor {floor!r}) -- "
            "a sliver/degenerate tet the mesh module should not have produced"
        )

    # Partition of unity gradient: sum_i grad_lambda_i == 0 (Section 2.5).
    residual = float(np.abs(grad_lambda.sum(axis=1)).max())
    tol = 1e-8 * max(1.0, float(np.abs(grad_lambda).max()))
    if residual > tol:
        raise MeshGeometryError(
            f"sum_i grad_lambda_i != 0 (max residual {residual!r}, tol {tol!r}) -- "
            "P^-1 computed or sliced wrong"
        )

    # Kronecker reproduction: lambda_i(r_j) == delta_ij (Section 2.5).
    coords = vertices[tets]
    lam = alpha[:, :, None] + np.einsum("tik,tjk->tij", grad_lambda, coords)
    residual_kron = float(np.abs(lam - np.eye(4)[None, :, :]).max())
    if residual_kron > 1e-6:
        raise MeshGeometryError(f"Kronecker reproduction lambda_i(r_j)==delta_ij failed (max residual {residual_kron!r})")


# ============================================================================
# Section 4: face geometry
# ============================================================================


def _face_geometry(vertices: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = vertices[tets]
    Nt = tets.shape[0]
    areas = np.empty((Nt, 4))
    normals = np.empty((Nt, 4, 3))
    for local_face, (i0, i1, i2) in enumerate(LOCAL_FACES):
        p0, p1, p2 = coords[:, i0, :], coords[:, i1, :], coords[:, i2, :]
        cross = np.cross(p1 - p0, p2 - p0)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        n_hat = cross / (2.0 * area[:, None])

        # LOCAL_FACES is indexed by its opposite local vertex, so
        # local_face IS that opposite vertex's index (Section 4).
        p_opp = coords[:, local_face, :]
        needs_flip = np.einsum("tk,tk->t", n_hat, p_opp - p0) >= 0.0
        n_hat[needs_flip] *= -1.0

        areas[:, local_face] = area
        normals[:, local_face, :] = n_hat
    return areas, normals


def _check_face_geometry(
    grad_lambda: np.ndarray, volume: np.ndarray, face_area: np.ndarray, face_normal: np.ndarray
) -> None:
    # Closed-surface identity: sum_k A_k n_hat_k == 0 (Section 4).
    weighted_sum = (face_area[:, :, None] * face_normal).sum(axis=1)
    scale = max(1.0, float(np.abs(face_area).max()))
    residual = float(np.abs(weighted_sum).max())
    if residual > 1e-9 * scale:
        raise MeshGeometryError(
            f"closed-surface identity sum_k A_k n_hat_k != 0 (max residual {residual!r}, scale {scale!r})"
        )

    # Section 2.4's tie-in: grad_lambda_i == -(A_i/3V) n_hat_i. A failure
    # here but not the closed-surface check (or vice versa) localizes the
    # bug to the gradient path or the face path specifically.
    predicted = -(face_area[:, :, None] / (3.0 * volume[:, None, None])) * face_normal
    scale2 = max(1.0, float(np.abs(grad_lambda).max()))
    residual2 = float(np.abs(predicted - grad_lambda).max())
    if residual2 > 1e-6 * scale2:
        raise MeshGeometryError(
            f"grad_lambda_i != -(A_i/3V) n_hat_i (Section 2.4 identity failed, "
            f"max residual {residual2!r}, scale {scale2!r})"
        )


# ============================================================================
# Section 3: global edge topology
# ============================================================================


def _edge_topology(tets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    endpoints = tets[:, np.array(LOCAL_EDGES)]  # (Nt,6,2): global vertex pair per local edge
    lo = endpoints.min(axis=2)
    hi = endpoints.max(axis=2)
    sign = np.where(endpoints[:, :, 0] < endpoints[:, :, 1], 1, -1).astype(np.int8)

    keys = np.stack([lo, hi], axis=-1).reshape(-1, 2)
    edges, inverse = np.unique(keys, axis=0, return_inverse=True)
    tet_edge_map = inverse.reshape(tets.shape[0], 6).astype(np.int64)
    return edges, tet_edge_map, sign


def _check_edge_topology(
    tets: np.ndarray, edges: np.ndarray, tet_edge_map: np.ndarray, tet_edge_sign: np.ndarray
) -> None:
    if not np.all(edges[:, 0] < edges[:, 1]):
        raise MeshGeometryError("global edge list contains a non-ascending entry")

    # Sign self-consistency (Section 3.4): recomputing s_e from `edges` +
    # `tets` must reproduce `tet_edge_sign`.
    endpoints = tets[:, np.array(LOCAL_EDGES)]
    canonical_lo = edges[tet_edge_map][:, :, 0]
    recomputed_sign = np.where(endpoints[:, :, 0] == canonical_lo, 1, -1).astype(np.int8)
    if not np.array_equal(recomputed_sign, tet_edge_sign):
        raise MeshGeometryError("tet_edge_sign is not self-consistent with edges/tet_edge_map")


# ============================================================================
# Section 5: boundary-face extraction and tagging
# ============================================================================


def _face_incidence(tets: np.ndarray) -> tuple[dict[tuple[int, int, int], list[DimTag]], set[tuple[int, int, int]]]:
    Nt = tets.shape[0]
    local_face_vertices = tets[:, np.array(LOCAL_FACES)]  # (Nt,4,3)
    flat = local_face_vertices.reshape(-1, 3)
    keys = np.sort(flat, axis=1)
    unique_keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)

    bad = (counts != 1) & (counts != 2)
    if np.any(bad):
        raise MeshGeometryError(
            f"{int(bad.sum())} face(s) with incidence not in {{1,2}} -- non-conforming or corrupt mesh"
        )

    # Section 5.2's count identity, as an explicit, independently-derived
    # check (not merely implied by how counts was built above).
    n_boundary = int(np.sum(counts == 1))
    n_interior = int(np.sum(counts == 2))
    total = 4 * Nt
    if 2 * n_interior + n_boundary != total:
        raise MeshGeometryError(
            f"face count identity failed: 4*Nt={total} != 2*interior({n_interior})+boundary({n_boundary})"
        )

    tet_idx = np.repeat(np.arange(Nt), 4)
    local_face_idx = np.tile(np.arange(4), Nt)
    face_lookup: dict[tuple[int, int, int], list[DimTag]] = {}
    for key_idx, t, lf in zip(inverse.tolist(), tet_idx.tolist(), local_face_idx.tolist()):
        key = tuple(int(v) for v in unique_keys[key_idx])
        face_lookup.setdefault(key, []).append((t, lf))

    boundary_keys = {key for key, occurrences in face_lookup.items() if len(occurrences) == 1}
    return face_lookup, boundary_keys


def _resolve_tags(
    face_lookup: dict[tuple[int, int, int], list[DimTag]], surface_tags: dict[str, np.ndarray]
) -> dict[str, list[DimTag]]:
    tagged: dict[str, list[DimTag]] = {}
    for name, triangles in surface_tags.items():
        entries: list[DimTag] = []
        for tri in np.asarray(triangles, dtype=np.int64):
            key = tuple(int(v) for v in np.sort(tri))
            matches = face_lookup.get(key)
            if matches is None:
                raise MeshGeometryError(f"surface_tags[{name!r}] references face {key} not present in the tet mesh")
            entries.extend(matches)
        tagged[name] = entries
    return tagged


def _check_boundary_coverage(boundary_keys: set[tuple[int, int, int]], surface_tags: dict[str, np.ndarray]) -> None:
    """Section 5.3's coverage check: every topologically-exterior face
    (incidence 1) resolves to exactly one tag. A tag mapped to an
    interior face (the embedded `PEC_LINE` sheet, incidence 2) is simply
    never a member of `boundary_keys`, so it is excluded from this check
    by construction -- no special-casing by name needed (Section 0's
    generic contract is preserved even though the concrete tag names come
    from Module 0)."""
    claims: dict[tuple[int, int, int], list[str]] = {}
    for name, triangles in surface_tags.items():
        for tri in np.asarray(triangles, dtype=np.int64):
            key = tuple(int(v) for v in np.sort(tri))
            if key in boundary_keys:
                claims.setdefault(key, []).append(name)

    missing = boundary_keys - claims.keys()
    if missing:
        raise MeshGeometryError(
            f"{len(missing)} boundary face(s) left untagged -- the solver would treat them as a "
            "perfect magnetic wall by default, almost always unintended"
        )
    multiply_tagged = {key: names for key, names in claims.items() if len(names) > 1}
    if multiply_tagged:
        example = next(iter(multiply_tagged.items()))
        raise MeshGeometryError(
            f"{len(multiply_tagged)} boundary face(s) claimed by more than one tag, e.g. {example}"
        )
