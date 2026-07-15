"""visualization.field_slice -- a 3D, axis-aligned volume-field slice
rendered at its true location inside the SAME `Axes3D` that
`visualization.geometry_view.plot_geometry` draws the structure into
(reused directly, via its `ax`/`alpha_scale` parameters -- no second 3D
plotting backend introduced here).

Field values come from `fem.field_eval.evaluate_edge_field`, applied
directly to a driven-solve `SweepResult.a` -- the same reconstruction
`fem.assembly` implicitly defines via its own Whitney basis/sign
convention (see that module's docstring for where this is sourced from).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import constants as _c

from fem.field_eval import evaluate_curl_field, evaluate_edge_field
from mesh_interface import MeshInterface

from .geometry_view import PlottingUnavailableError, plot_geometry

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from solve.sweep import SweepResult

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def _require_matplotlib():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlottingUnavailableError(
            "matplotlib is required for field-slice visualization; install it with 'pip install matplotlib'"
        ) from exc
    return plt, mpl


def _plane_grid(mesh: MeshInterface, axis: str, value: float, grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`X, Y, Z`: `(grid,grid)` physical coordinate arrays for
    `ax.plot_surface` on the `axis=value` plane; `points`: the same
    `(grid*grid, 3)` points flattened, for `evaluate_edge_field`."""
    axis_idx = _AXIS_INDEX[axis]
    other_idx = [i for i in range(3) if i != axis_idx]

    lo = mesh.vertices[:, other_idx].min(axis=0)
    hi = mesh.vertices[:, other_idx].max(axis=0)
    u = np.linspace(float(lo[0]), float(hi[0]), grid)
    v = np.linspace(float(lo[1]), float(hi[1]), grid)
    U, V = np.meshgrid(u, v, indexing="xy")

    coords: list[np.ndarray] = [np.empty(0)] * 3
    coords[axis_idx] = np.full_like(U, value)
    coords[other_idx[0]] = U
    coords[other_idx[1]] = V
    X, Y, Z = coords

    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return X, Y, Z, points


def plot_field_slice(
    mesh: MeshInterface,
    result: "SweepResult",
    plane: tuple[str, float],
    *,
    grid: int = 120,
    field: str = "E",
    ax=None,
    show: bool = True,
    output: "Path | None" = None,
    cmap: str = "magma",
    structure_alpha_scale: float = 0.35,
) -> "Figure":
    """An axis-aligned magnitude slice of the driven field
    (`result.a`, e.g. `sweep_results[i].a`) at `plane=(axis, value)`
    (`axis` one of `'x'/'y'/'z'`), rendered as a colored surface at its
    true 3D location, overlaid on the structure (`plot_geometry`, drawn
    translucent via `structure_alpha_scale` so the slice reads through
    it) inside the same `Axes3D`.

    `field='E'` (default, phase-independent `|E|`) or `field='H'`
    (`H = (1j/(omega*mu0))*curl(E)`, Module 4 Section 3.8's identity
    applied to the full driven field via `fem.field_eval.evaluate_curl_field`,
    using `result.omega`). Points outside the meshed domain
    (`evaluate_edge_field`'s nan contract) are rendered fully transparent,
    not masked out of the grid geometry -- the plane's *coordinates* are
    always finite, only the *color* at an outside point is invisible.

    If `ax` is given, draws into it and returns without creating a new
    figure or calling show/save (mirrors `plot_geometry`'s own `ax`
    contract) -- this is how the CLI composes `--show-geometry` and
    `--plot-field-slice` into one figure. If `ax` is None, builds its own
    structure+slice figure internally (`plot_geometry(ax=..., show=False)`),
    so a slice always appears in-context even without `--show-geometry`.
    """
    axis, value = plane
    if axis not in _AXIS_INDEX:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    if field not in ("E", "H"):
        raise ValueError(f"field must be 'E' or 'H', got {field!r}")

    plt, mpl = _require_matplotlib()
    X, Y, Z, points = _plane_grid(mesh, axis, value, grid)

    if field == "E":
        values = evaluate_edge_field(mesh, result.a, points)
    else:
        curl = evaluate_curl_field(mesh, result.a, points)
        values = (1j / (result.omega * _c.mu_0)) * curl
    mag = np.sqrt(np.sum(np.abs(values) ** 2, axis=1)).reshape(X.shape)

    owns_figure = ax is None
    if owns_figure:
        fig = plt.figure(figsize=(9.0, 7.0))
        ax = fig.add_subplot(111, projection="3d")
        plot_geometry(mesh, ax=ax, show=False, alpha_scale=structure_alpha_scale)
    else:
        fig = ax.figure

    finite = np.isfinite(mag)
    if np.any(finite):
        vmin, vmax = float(np.min(mag[finite])), float(np.max(mag[finite]))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-30
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    colors = cmap_obj(norm(np.where(finite, mag, vmin)))  # (grid,grid,4)
    colors[~finite] = (0.0, 0.0, 0.0, 0.0)  # fully transparent outside the meshed domain

    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    mappable.set_array(mag[finite] if np.any(finite) else np.zeros(1))
    label = "|E| (a.u.)" if field == "E" else "|H| (a.u.)"
    fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label=label)

    ax.set_title(f"Field slice ({field}): {axis}={value:.4g}")

    if not owns_figure:
        return fig

    if output is not None:
        fig.savefig(output, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
