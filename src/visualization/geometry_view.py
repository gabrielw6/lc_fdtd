"""visualization.geometry_view -- 3D sanity-check plots of the tagged
geometry and of the tetrahedral mesh itself, built only from a
`MeshInterface`.

Not a physics module (no `docs/module*_equations.md` companion) -- like
`cli.py`, it introduces no new equations. It exists so a user can look at
what Module 0/1 actually built (or the mesh the solver will actually see)
without waiting through a full frequency sweep. Reads only through
`MeshInterface`'s public contract (`vertices`, `tets`, `volume_tags`,
`edges`, `boundary_faces`) -- never `geometry_builder` internals -- so it
works the same way regardless of how the mesh was produced.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from mesh_interface.interface import LOCAL_FACES, MeshInterface

if TYPE_CHECKING:
    from matplotlib.figure import Figure


class PlottingUnavailableError(RuntimeError):
    """Raised when a plotting function is called but matplotlib isn't
    installed -- reported as a clean one-line error (with the install
    command), not a raw ImportError traceback."""


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    except ImportError as exc:
        raise PlottingUnavailableError(
            "matplotlib is required for geometry/mesh visualization; install it with 'pip install matplotlib'"
        ) from exc
    return plt, Poly3DCollection, Line3DCollection


# Surface tags shown by `plot_geometry`, in draw order, with (facecolor,
# alpha). PMC_SIDE (the lateral truncation boundary) and PML_OUTER_PEC are
# deliberately excluded -- per the caller's request, this view is a sanity
# check of the physical circuit (ports, trace, substrate), not the
# simulation truncation apparatus.
_GEOMETRY_TAGS: tuple[tuple[str, str, float], ...] = (
    ("PORT_1", "tab:red", 0.45),
    ("PORT_2", "tab:orange", 0.45),
    ("PEC_LINE", "tab:blue", 0.9),
)


def _triangles_for_tag(mesh: MeshInterface, tag: str) -> np.ndarray:
    """`(M,3,3)` triangle vertex coordinates for every face tagged `tag`."""
    faces = mesh.boundary_faces(tag)
    if not faces:
        return np.empty((0, 3, 3))
    tris = np.empty((len(faces), 3, 3))
    for i, (tet, local_face) in enumerate(faces):
        verts = mesh.tets[tet][list(LOCAL_FACES[local_face])]
        tris[i] = mesh.vertices[verts]
    return tris


def _volume_envelope_triangles(mesh: MeshInterface, tags: tuple[str, ...]) -> np.ndarray:
    """A volume group's own boundary: every tet face incident to a
    `tags`-tagged tet on one side and either nothing or a tet tagged
    outside `tags` on the other. Generalizes what was originally
    `_substrate_envelope_triangles` (a single-tag `("SUBSTRATE",)` call) so
    the same envelope-tracing logic can draw the LC cavity's own outline
    too (`("LC",)`) -- independent of which faces happen to carry a
    surface tag, since most of a volume's own interface with its neighbors
    (everywhere but the PEC_LINE patch) is untagged.

    Mirrors the face-key grouping `mesh_interface.interface._face_incidence`
    uses internally, but recomputed here rather than reaching into that
    private helper -- this module only touches `MeshInterface`'s public
    surface (Section 10 of CLAUDE.md: interfaces, not internals).
    """
    tets = mesh.tets
    in_group = np.isin(mesh.volume_tags, tags)

    local_face_vertices = tets[:, np.array(LOCAL_FACES)]  # (Nt,4,3)
    keys = np.sort(local_face_vertices.reshape(-1, 3), axis=1)
    tet_idx = np.repeat(np.arange(tets.shape[0]), 4)
    local_face_idx = np.tile(np.arange(4), tets.shape[0])

    face_lookup: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for key, t, lf in zip(map(tuple, keys.tolist()), tet_idx.tolist(), local_face_idx.tolist()):
        face_lookup.setdefault(key, []).append((t, lf))

    out = []
    for occurrences in face_lookup.values():
        group_sides = [(t, lf) for t, lf in occurrences if in_group[t]]
        if not group_sides or len(group_sides) == len(occurrences) == 2:
            continue  # not touching the group, or interior to the group's own block
        t, lf = group_sides[0]
        verts = tets[t][list(LOCAL_FACES[lf])]
        out.append(mesh.vertices[verts])
    return np.array(out) if out else np.empty((0, 3, 3))


def _set_equal_3d_bounds(ax, vertices: np.ndarray) -> None:
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    ax.set_xlim(float(mins[0]), float(maxs[0]))
    ax.set_ylim(float(mins[1]), float(maxs[1]))
    ax.set_zlim(float(mins[2]), float(maxs[2]))
    ax.set_box_aspect(np.maximum(maxs - mins, 1e-12))


def plot_geometry(
    mesh: MeshInterface,
    *,
    show_substrate: bool = True,
    show: bool = True,
    output: "Path | None" = None,
    elev: float = 25.0,
    azim: float = -60.0,
    ax=None,
    alpha_scale: float = 1.0,
) -> "Figure":
    """3D view of the tagged geometry: PORT_1/PORT_2 (translucent), the
    PEC_LINE trace (opaque), and -- unless `show_substrate=False` -- the
    substrate slab's own envelope (translucent gray) *and* the LC cavity's
    own envelope (translucent purple, its own legend entry), drawn
    separately since `MeshInterface.volume_tags` always distinguishes
    `SUBSTRATE` from `LC` (the LC region is `"LC"`-tagged even for `--lc
    none`, where its material happens to equal the substrate's -- see
    `cli.build_lc_material`). Drawing only the `SUBSTRATE` envelope (as
    this function originally did) rendered that shared boundary as the
    substrate's own outer surface, i.e. a rectangular notch cut out of the
    slab, regardless of whether the LC region was electrically
    substrate-matched or a real anisotropic cavity -- a rendering artifact
    only, not a meshing/material bug (the region is correctly meshed and
    assigned `eps_r=eps_r_substrate` for `--lc none`). PMC_SIDE and the PML
    shell are never drawn (see `_GEOMETRY_TAGS`).

    `ax`/`alpha_scale` (added for `viz.field_slice`'s reuse -- CLI's own
    `--show-geometry` never passes either, so its behavior is bit-for-bit
    unchanged): pass an existing `Axes3D` to draw the structure into it
    instead of creating a new figure -- the caller then owns show/save/
    close, so this function returns immediately after drawing rather than
    doing any of those. `alpha_scale` uniformly scales every tag's alpha
    (e.g. `0.4` for a faint structure a field slice should read through);
    `1.0` (default) reproduces the plain `--show-geometry` alphas exactly."""
    from matplotlib.patches import Patch

    plt, Poly3DCollection, _ = _require_matplotlib()
    owns_figure = ax is None
    if owns_figure:
        fig = plt.figure(figsize=(9.0, 6.0))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    handles = []
    if show_substrate:
        tris = _volume_envelope_triangles(mesh, ("SUBSTRATE",))
        if len(tris):
            ax.add_collection3d(
                Poly3DCollection(tris, facecolor="lightgray", edgecolor="none", alpha=0.25 * alpha_scale)
            )
        lc_tris = _volume_envelope_triangles(mesh, ("LC",))
        if len(lc_tris):
            ax.add_collection3d(
                Poly3DCollection(lc_tris, facecolor="mediumpurple", edgecolor="none", alpha=0.35 * alpha_scale)
            )
            handles.append(Patch(facecolor="mediumpurple", alpha=0.35, label="LC"))
    for tag, color, alpha in _GEOMETRY_TAGS:
        tris = _triangles_for_tag(mesh, tag)
        if len(tris) == 0:
            continue
        ax.add_collection3d(
            Poly3DCollection(tris, facecolor=color, edgecolor="k", linewidths=0.1, alpha=alpha * alpha_scale)
        )
        handles.append(Patch(facecolor=color, alpha=alpha, label=tag))

    _set_equal_3d_bounds(ax, mesh.vertices)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Geometry (ports, PEC line, substrate envelope)")
    ax.view_init(elev=elev, azim=azim)
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=8)

    if not owns_figure:
        return fig

    if output is not None:
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_mesh(
    mesh: MeshInterface,
    *,
    show: bool = True,
    output: "Path | None" = None,
    elev: float = 25.0,
    azim: float = -60.0,
) -> "Figure":
    """3D wireframe of the tetrahedral mesh: every entry of `mesh.edges` --
    the exact global edge list the edge-element solver assembles on
    (`mesh_interface.interface` Section 3) -- drawn as one line segment."""
    plt, _, Line3DCollection = _require_matplotlib()
    fig = plt.figure(figsize=(9.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    segments = mesh.vertices[mesh.edges]  # (n_edges, 2, 3)
    ax.add_collection3d(Line3DCollection(segments, colors="steelblue", linewidths=0.2, alpha=0.5))

    _set_equal_3d_bounds(ax, mesh.vertices)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"Tetrahedral mesh ({mesh.n_tets} tets, {mesh.n_edges} edges)")
    ax.view_init(elev=elev, azim=azim)

    if output is not None:
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
