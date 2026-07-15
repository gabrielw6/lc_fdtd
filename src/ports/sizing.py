"""ports.sizing -- heuristic port-aperture sizing checks. Not part of any
`docs/module4_ports_equations.md` equation and never raises: these are
rules of thumb (the HFSS CPW-port tutorial's own guidance), not physical
invariants. Violating one doesn't make a result wrong by itself -- it
raises the chance of hitting Section 3.7's known box-mode mode-selection
limitation (aperture too large) or of clipping the trace's fringing field
before it decays (aperture too small).

Deliberately has no dependency on `geometry_builder`: every quantity is
inferred from an already-extracted `PortCrossSection` plus a
`MaterialAssembly`, so `ports` keeps its own stated invariant ("never
imports fem, geometry_builder, or meshing", `ports/__init__.py`) intact.
"""
from __future__ import annotations

import numpy as np
from scipy import constants

from material import MaterialAssembly

from .basis2d import TRI_LOCAL_EDGES
from .cross_section import PortCrossSection

# Heuristic thresholds (HFSS CPW port tutorial rule of thumb) -- tunable,
# not physically derived quantities.
_BOX_MODE_FRACTION = 0.5  # aperture dimension should stay below lambda_min/2
_FRINGE_WIDTH_MARGIN = 6.0  # W_port should be >= w + this many * h_sub
_FRINGE_HEIGHT_MARGIN = 4.0  # H_port should be >= h_sub + this many * h_sub


def check_port_sizing(
    W_port: float, H_port: float, h_sub: float, w: float, eps_r_max: float, f_max: float
) -> list[str]:
    """Human-readable warnings for a port aperture `W_port x H_port` that
    is either large enough to plausibly admit a spurious box mode at or
    below `f_max` (upper bound), or small enough to risk clipping the
    trace's fringing field (lower bound). Never raises -- both rules are
    rules of thumb, not hard invariants."""
    out: list[str] = []

    lambda_min = constants.c / (f_max * eps_r_max**0.5)
    box_mode_limit = _BOX_MODE_FRACTION * lambda_min
    largest = max(W_port, H_port)
    if largest >= box_mode_limit:
        f_c = constants.c / (2.0 * largest * eps_r_max**0.5)
        overshoot = f_max / f_c
        out.append(
            f"port aperture ({W_port:.4g} x {H_port:.4g} m) is not below lambda_min/2 "
            f"({box_mode_limit:.4g} m at f_max={f_max:.4g} Hz, eps_r_max={eps_r_max:.4g}): "
            f"the lowest box mode of this PEC-walled aperture cuts on near {f_c:.4g} Hz, "
            f"{overshoot:.2g}x below f_max -- risk of Module 4's known box-mode mode-selection "
            "limitation (HFSS CPW port tutorial rule of thumb: keep the port smaller than lambda/2)"
        )

    width_floor = w + _FRINGE_WIDTH_MARGIN * h_sub
    if W_port < width_floor:
        out.append(
            f"port width W_port={W_port:.4g} m is less than trace width + "
            f"{_FRINGE_WIDTH_MARGIN:.0f}*h_sub ({width_floor:.4g} m): the aperture may clip the "
            "trace's transverse fringing field (HFSS CPW port tutorial rule of thumb)"
        )
    height_floor = h_sub + _FRINGE_HEIGHT_MARGIN * h_sub
    if H_port < height_floor:
        out.append(
            f"port height H_port={H_port:.4g} m is less than {1 + _FRINGE_HEIGHT_MARGIN:.0f}*h_sub "
            f"({height_floor:.4g} m): the aperture may clip the trace's vertical fringing field "
            "into the air region (HFSS CPW port tutorial rule of thumb)"
        )
    return out


def infer_h_sub_and_trace_width(cs: PortCrossSection) -> tuple[float, float]:
    """Best-effort geometric inference of `h_sub` and the trace width `w`
    directly from an already-extracted cross-section's own tagging (used
    only for `check_port_sizing`'s fringing-margin warning).

    `h_sub`: the top `z` reached by any `SUBSTRATE`-tagged triangle.
    `w`: the y-extent of the PEC edge set lying entirely at `z=h_sub` --
    the embedded trace footprint (Section 2.3), distinct from the `z=0`
    ground edges and from any vertical PEC aperture-wall edges (a
    restricted aperture's side walls run at constant y, varying z, so they
    never have both endpoints at the same z=h_sub)."""
    sub_tris = cs.triangles[cs.tri_tag == "SUBSTRATE"]
    if sub_tris.size == 0:
        return 0.0, 0.0
    h_sub = float(cs.yz[sub_tris.ravel(), 1].max())

    edge_to_verts: dict[int, tuple[int, int]] = {}
    for t in range(cs.n_triangles):
        for slot, (p, q) in enumerate(TRI_LOCAL_EDGES):
            edge_to_verts[int(cs.tri_edges[t, slot])] = (int(cs.triangles[t, p]), int(cs.triangles[t, q]))

    tol = 1e-9 * max(1.0, h_sub)
    trace_ys: list[float] = []
    for e_local, is_pec in enumerate(cs.pec_edges):
        if not is_pec:
            continue
        v0, v1 = edge_to_verts[e_local]
        z0, z1 = cs.yz[v0, 1], cs.yz[v1, 1]
        if abs(z0 - h_sub) < tol and abs(z1 - h_sub) < tol:
            trace_ys.extend([float(cs.yz[v0, 0]), float(cs.yz[v1, 0])])

    w = float(max(trace_ys) - min(trace_ys)) if trace_ys else 0.0
    return h_sub, w


def _max_eps_r(cs: PortCrossSection, materials: MaterialAssembly) -> float:
    """Max real relative permittivity present on the cross-section --
    deliberately not `mode_solver._eps_r_bounds`'s stricter isotropy-
    asserting version, since this is only a wavelength heuristic, not a
    spurious-mode filter."""
    hi = 1.0
    for t in range(cs.n_triangles):
        centroid = cs.yz[cs.triangles[t]].mean(axis=0, keepdims=True)
        point3d = np.column_stack([[cs.x0], centroid])
        eps = materials.epsilon(str(cs.tri_tag[t]), point3d)
        hi = max(hi, float(np.real(eps[0, 0, 0])))
    return hi


def check_port_sizing_for_cross_section(
    cs: PortCrossSection, materials: MaterialAssembly, f_max: float
) -> list[str]:
    """Ergonomic wrapper for the two real call sites (`PortModeSolver.solve`,
    `solve.sweep.run_sweep`): derives every `check_port_sizing` argument
    directly from an already-extracted cross-section, so neither call site
    needs to separately track `h_sub`/`w`/`eps_r_max` itself."""
    W_port = float(cs.yz[:, 0].max() - cs.yz[:, 0].min())
    H_port = float(cs.yz[:, 1].max() - cs.yz[:, 1].min())
    h_sub, w = infer_h_sub_and_trace_width(cs)
    eps_r_max = _max_eps_r(cs, materials)
    return check_port_sizing(W_port, H_port, h_sub, w, eps_r_max, f_max)
