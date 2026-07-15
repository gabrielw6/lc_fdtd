"""visualization.port_field -- the HFSS-style "port field distribution"
diagnostic: samples a `PortMode`'s transverse field on its own (y,z)
cross-section and renders it as a 2D magnitude map or quiver plot. Makes
a clean quasi-TEM pattern vs. a wall-attached box-mode pattern (Module 4
Section 3.7's known mode-selection limitation) visually obvious at a
glance, without needing to inspect raw field arrays.

Point evaluation reuses `PortMode.e_t`/`h_t` directly (the same callables
`ports.mode_solver` itself and `extract.sparameters` use) -- no new field
reconstruction here, just sampling + rendering.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ports.basis2d import TRI_LOCAL_EDGES
from ports.mode_solver import PortMode, PortModeError

from .geometry_view import PlottingUnavailableError

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlottingUnavailableError(
            "matplotlib is required for port-field visualization; install it with 'pip install matplotlib'"
        ) from exc
    return plt


def _sample_field(field_fn, points3d: np.ndarray) -> np.ndarray:
    """Evaluates `field_fn` (`mode.e_t`/`mode.h_t`) at every point,
    masking any point off the cross-section as nan rather than letting
    `PortModeError` abort the whole grid. Tries the fast whole-batch call
    first (the common case: a grid spanning the cross-section's own
    bounding box lies entirely inside it, since that box *is* the
    triangulated region for a rectangular port aperture) and only falls
    back to a slower per-point loop if that raises."""
    try:
        return np.asarray(field_fn(points3d))
    except PortModeError:
        pass

    out = np.full((points3d.shape[0], 3), np.nan, dtype=complex)
    for i in range(points3d.shape[0]):
        try:
            out[i] = field_fn(points3d[i : i + 1])[0]
        except PortModeError:
            continue
    return out


def _pec_edge_segments(mode: PortMode) -> np.ndarray:
    """`(Ne_pec, 2, 2)` (y,z) endpoint pairs for the cross-section's own
    PEC edges (ground plane + embedded trace, Module 4 Section 2.3) --
    overlaid on the field plot so the reader can see the conductor
    geometry the field pattern lives against."""
    cs = mode.cross_section
    edge_to_verts: dict[int, tuple[int, int]] = {}
    for t in range(cs.n_triangles):
        for slot, (p, q) in enumerate(TRI_LOCAL_EDGES):
            edge_to_verts[int(cs.tri_edges[t, slot])] = (int(cs.triangles[t, p]), int(cs.triangles[t, q]))

    segments = []
    for e_local, is_pec in enumerate(cs.pec_edges):
        if not is_pec:
            continue
        v0, v1 = edge_to_verts[e_local]
        segments.append([cs.yz[v0], cs.yz[v1]])
    return np.array(segments) if segments else np.empty((0, 2, 2))


def plot_port_mode(
    mode: PortMode,
    *,
    field: str = "E",
    style: str = "mag",
    grid: int = 120,
    mode_index: int = 1,
    ax=None,
    show: bool = True,
    output: "Path | None" = None,
    cmap: str = "magma",
) -> "Figure":
    """Samples `mode.e_t` (`field='E'`) or `mode.h_t` (`field='H'`) on a
    `grid x grid` lattice spanning the cross-section's own (y,z) bounding
    box, at the fixed `x=cross_section.x0` plane, and renders it:

    `style='mag'`: `pcolormesh` of `sqrt(sum(|component|^2))`, a
    perceptually-uniform sequential colormap (default `'magma'`) --
    `|E|`/`|H|` is phase-independent, the natural default for "where is
    the field concentrated" (the diagnostic this function exists for).
    `style='quiver'`: `quiver` of `Re(field_y), Re(field_z)` over a faint
    magnitude background (same colormap, low alpha) -- shows *direction*,
    at the cost of picking a phase reference (`Re(.)`).

    Points off the cross-section are masked (`_sample_field`). The
    cross-section's own PEC edges (ground + embedded trace) are drawn as
    black line segments for context. Title reports the port tag (from
    `mode.cross_section.port_tag`), `mode_index` (caller-supplied --
    `PortMode` itself doesn't carry its own rank), `beta=Im(gamma)`, and
    `Y`. Colorbar labeled `"|E| (a.u.)"`/`"|H| (a.u.)"` -- scaling is
    excitation/normalization dependent (Module 4 Section 4.2's `P_m=1`
    convention), not an absolute physical unit.
    """
    if field not in ("E", "H"):
        raise ValueError(f"field must be 'E' or 'H', got {field!r}")
    if style not in ("mag", "quiver"):
        raise ValueError(f"style must be 'mag' or 'quiver', got {style!r}")

    plt = _require_matplotlib()
    cs = mode.cross_section
    y = np.linspace(float(cs.yz[:, 0].min()), float(cs.yz[:, 0].max()), grid)
    z = np.linspace(float(cs.yz[:, 1].min()), float(cs.yz[:, 1].max()), grid)
    Y, Z = np.meshgrid(y, z, indexing="xy")
    points3d = np.column_stack([np.full(Y.size, cs.x0), Y.ravel(), Z.ravel()])

    field_fn = mode.e_t if field == "E" else mode.h_t
    values = _sample_field(field_fn, points3d)  # (grid*grid, 3) complex, transverse-only
    mag = np.sqrt(np.sum(np.abs(values) ** 2, axis=1)).reshape(grid, grid)

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
    else:
        fig = ax.figure

    if style == "mag":
        mesh_plot = ax.pcolormesh(Y, Z, np.ma.masked_invalid(mag), cmap=cmap, shading="auto")
    else:
        mesh_plot = ax.pcolormesh(Y, Z, np.ma.masked_invalid(mag), cmap=cmap, shading="auto", alpha=0.35)
        Ey = values[:, 1].real.reshape(grid, grid)
        Ez = values[:, 2].real.reshape(grid, grid)
        stride = max(1, grid // 20)
        ax.quiver(
            Y[::stride, ::stride], Z[::stride, ::stride], Ey[::stride, ::stride], Ez[::stride, ::stride], color="k"
        )

    segments = _pec_edge_segments(mode)
    for seg in segments:
        ax.plot(seg[:, 0], seg[:, 1], color="white", linewidth=1.5, solid_capstyle="round")
        ax.plot(seg[:, 0], seg[:, 1], color="black", linewidth=0.6, solid_capstyle="round")

    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")
    beta = mode.gamma.imag
    ax.set_title(f"{cs.port_tag} mode {mode_index}: beta={beta:.4g} rad/m, Y={mode.Y!r}")
    cbar = fig.colorbar(mesh_plot, ax=ax, shrink=0.85)
    cbar.set_label(f"|{field}| (a.u.)")

    if not owns_figure:
        return fig

    if output is not None:
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
