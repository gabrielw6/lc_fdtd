"""ports.cross_section -- derives a port's 2D triangulation directly from
`MeshInterface.boundary_faces('PORT_p')` (docs/module4_ports_equations.md
Section 2). No separate 2D mesh is generated (Section 2.1): each 2D vertex
and edge here *is* a 3D global vertex/edge (Section 2.5), so this module's
only job is bookkeeping -- projecting to (y, z), fixing local orientation,
and building the compact local numbering the 2D eigenproblem needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mesh_interface import MeshInterface
from mesh_interface import quadrature as mesh_quadrature
from mesh_interface.interface import LOCAL_FACES

from .basis2d import TRI_LOCAL_EDGES


class CrossSectionError(RuntimeError):
    """Raised when a port's tagged faces do not form a valid, planar 2D
    cross-section (Section 2.1/2.4) -- a corrupt tag or non-conforming mesh,
    not a caller mistake."""


@dataclass
class PortCrossSection:
    """Section 2's derived 2D triangulation. `global_vertex_ids` and
    `global_edge_ids` are the bridge back to the 3D mesh (Section 2.5) --
    everything else is local/compact numbering private to this port's own
    2D eigenproblem."""

    port_tag: str
    x0: float  # the port plane's fixed x coordinate (Section 2.1)
    axial_sign: float  # +1 if the domain interior sits at x>x0 (outward normal -x_hat,
    # e.g. PORT_1 at x=0), -1 if the interior sits at x<x0 (outward normal +x_hat, e.g.
    # PORT_2 at x=L) -- see docs/module4_ports_equations.md Section 3.1's notation
    # ("hat x = the global length axis") vs. Section 5.1's n_out=-x_hat, which the
    # document derives for PORT_1 only. This is the s_p correction factor threading
    # into every axial (H-field/power/operator) quantity in mode_solver.py and
    # port_operator.py -- the 2D transverse eigenproblem itself (S_tt, S_zz, T_tt,
    # T_zt, S_tz, gamma^2, e_t) never uses an axial direction and so never reads this.
    yz: np.ndarray  # (Nv,2) local-vertex (y,z) coordinates
    global_vertex_ids: np.ndarray  # (Nv,) int -> 3D global vertex index
    triangles: np.ndarray  # (Nt,3) local vertex indices, oriented (area>0)
    tri_edges: np.ndarray  # (Nt,3) local edge index (TRI_LOCAL_EDGES slot order)
    tri_edge_sign: np.ndarray  # (Nt,3) float, +-1
    tri_tag: np.ndarray  # (Nt,) str, tet_volume_tag of the owning 3D tet (Section 2.2)
    grad_t: np.ndarray  # (Nt,3,2) constant per-triangle barycentric gradients (Section 2.4)
    area: np.ndarray  # (Nt,) triangle areas
    global_edge_ids: np.ndarray  # (Ne,) int -> 3D global edge index, ascending
    pec_edges: np.ndarray  # (Ne,) bool (Section 2.3)
    pec_vertices: np.ndarray  # (Nv,) bool (Section 2.6)

    @property
    def n_vertices(self) -> int:
        return self.yz.shape[0]

    @property
    def n_edges(self) -> int:
        return self.global_edge_ids.shape[0]

    @property
    def n_triangles(self) -> int:
        return self.triangles.shape[0]

    def free_edge_dofs(self) -> np.ndarray:
        """Non-PEC edge indices -- the transverse test/trial space (Section
        3.5: "the boundary term vanishes on PEC edges, where N_i is
        constrained to zero by the test-space choice")."""
        return np.flatnonzero(~self.pec_edges)

    def free_vertex_dofs(self) -> np.ndarray:
        """Non-PEC vertex indices -- the axial (nodal) test/trial space
        (Section 2.6's Dirichlet elimination)."""
        return np.flatnonzero(~self.pec_vertices)


def _triangle_geometry(yz: np.ndarray, triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Section 2.4's `P_triangle` inversion, vectorized over all triangles
    at once -- the direct 2D analogue of `mesh_interface._tet_geometry`."""
    coords = yz[triangles]  # (Nt,3,2)
    Nt = triangles.shape[0]
    P = np.empty((Nt, 3, 3))
    P[:, 0, :] = 1.0
    P[:, 1, :] = coords[:, :, 0]
    P[:, 2, :] = coords[:, :, 1]
    det = np.linalg.det(P)
    area = det / 2.0
    Q = np.linalg.inv(P)
    grad_t = Q[:, :, 1:3]
    return grad_t, area


def triangle_quadrature(coords_yz: np.ndarray, area: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Physical (y,z) quadrature points/weights for one already-oriented
    triangle -- Module 1's `quadrature_tri` pattern (`sum(weights)==area`),
    reused here since Module 4 has its own 2D triangulation rather than a
    `(tet, local_face)` handle to hand to `MeshInterface.quadrature_tri`."""
    bary, w_hat = mesh_quadrature.tri_rule(order)
    points = bary @ coords_yz
    weights = w_hat * area
    return points, weights


def extract_cross_section(mesh: MeshInterface, port_tag: str) -> PortCrossSection:
    """Section 2.1's extraction + Section 2.2's tagging + Section 2.3's PEC
    edge/vertex identification, in one pass over `boundary_faces(port_tag)`."""
    faces = mesh.boundary_faces(port_tag)
    if not faces:
        raise CrossSectionError(f"no boundary faces tagged {port_tag!r}")

    pec_edges_3d = mesh.pec_edge_dofs()
    edge_lookup = {(int(lo), int(hi)): idx for idx, (lo, hi) in enumerate(mesh.edges)}

    vid_to_local: dict[int, int] = {}
    global_vertex_ids: list[int] = []
    yz_list: list[np.ndarray] = []
    triangles: list[list[int]] = []
    tri_tag: list[str] = []
    tri_edge_3d_ids: list[list[int]] = []
    tri_edge_sign: list[list[float]] = []
    x0_samples: list[float] = []
    edge3d_set: set[int] = set()

    for tet, local_face in faces:
        tri_local_tet = LOCAL_FACES[local_face]  # 3 tet-local vertex indices
        gverts = [int(mesh.tets[tet, v]) for v in tri_local_tet]
        coords3d = mesh.vertices[gverts]  # (3,3)
        x0_samples.extend(coords3d[:, 0].tolist())
        yz = coords3d[:, 1:3].copy()

        # Section 2.4's orientation-normalization discipline: permute once so area>0.
        P = np.array([[1.0, 1.0, 1.0], [yz[0, 0], yz[1, 0], yz[2, 0]], [yz[0, 1], yz[1, 1], yz[2, 1]]])
        if np.linalg.det(P) < 0.0:
            gverts[1], gverts[2] = gverts[2], gverts[1]
            yz[[1, 2]] = yz[[2, 1]]

        local_idx = []
        for gv, p in zip(gverts, yz):
            if gv not in vid_to_local:
                vid_to_local[gv] = len(global_vertex_ids)
                global_vertex_ids.append(gv)
                yz_list.append(p)
            local_idx.append(vid_to_local[gv])
        triangles.append(local_idx)
        tri_tag.append(mesh.tet_volume_tag(tet))

        edge_ids_this_tri = []
        signs_this_tri = []
        for p, q in TRI_LOCAL_EDGES:
            ga, gb = gverts[p], gverts[q]
            lo, hi = (ga, gb) if ga < gb else (gb, ga)
            eid3d = edge_lookup[(lo, hi)]
            edge3d_set.add(eid3d)
            edge_ids_this_tri.append(eid3d)
            signs_this_tri.append(1.0 if ga < gb else -1.0)
        tri_edge_3d_ids.append(edge_ids_this_tri)
        tri_edge_sign.append(signs_this_tri)

    x0_arr = np.asarray(x0_samples)
    x0 = float(x0_arr[0])
    scale = max(1.0, abs(x0))
    if not np.allclose(x0_arr, x0, atol=1e-9 * scale):
        raise CrossSectionError(
            f"port {port_tag!r} faces are not coplanar in x (max deviation "
            f"{float(np.abs(x0_arr - x0).max())!r}) -- not a valid port cross-section"
        )

    # axial_sign (s_p): determined from the mesh geometry, not from a
    # hardcoded x0==0 test -- the owning tet's centroid x tells which side
    # of the port plane the domain interior sits on.
    tet0, _local_face0 = faces[0]
    centroid_x0 = float(mesh.vertices[mesh.tets[tet0]][:, 0].mean())
    axial_sign = 1.0 if centroid_x0 > x0 else -1.0

    global_edge_ids = np.array(sorted(edge3d_set), dtype=np.int64)
    eid3d_to_local = {int(e): i for i, e in enumerate(global_edge_ids)}
    tri_edges_local = np.array(
        [[eid3d_to_local[e] for e in row] for row in tri_edge_3d_ids], dtype=np.int64
    )

    yz_arr = np.array(yz_list)
    global_vertex_ids_arr = np.array(global_vertex_ids, dtype=np.int64)
    triangles_arr = np.array(triangles, dtype=np.int64)
    tri_edge_sign_arr = np.array(tri_edge_sign)
    tri_tag_arr = np.array(tri_tag)

    grad_t, area = _triangle_geometry(yz_arr, triangles_arr)
    char_length = max(1.0, float(yz_arr.max(axis=0).max() - yz_arr.min(axis=0).min()))
    floor = 1e-12 * char_length**2
    if np.any(area <= floor):
        bad = int(np.argwhere(area <= floor).ravel()[0])
        raise CrossSectionError(f"triangle {bad} of port {port_tag!r} has near-zero/negative area ({area[bad]!r})")

    pec_edges_mask = np.array([eid in pec_edges_3d for eid in global_edge_ids])
    pec_vertices_mask = np.zeros(len(global_vertex_ids), dtype=bool)
    tri_pec_edges = pec_edges_mask[tri_edges_local]  # (Nt,3)
    for tri_local, pec_row in zip(triangles_arr, tri_pec_edges):
        for (p, q), is_pec in zip(TRI_LOCAL_EDGES, pec_row):
            if is_pec:
                pec_vertices_mask[tri_local[p]] = True
                pec_vertices_mask[tri_local[q]] = True

    return PortCrossSection(
        port_tag=port_tag,
        x0=x0,
        axial_sign=axial_sign,
        yz=yz_arr,
        global_vertex_ids=global_vertex_ids_arr,
        triangles=triangles_arr,
        tri_edges=tri_edges_local,
        tri_edge_sign=tri_edge_sign_arr,
        tri_tag=tri_tag_arr,
        grad_t=grad_t,
        area=area,
        global_edge_ids=global_edge_ids,
        pec_edges=pec_edges_mask,
        pec_vertices=pec_vertices_mask,
    )
