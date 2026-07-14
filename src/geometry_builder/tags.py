"""geometry_builder.tags -- tag vocabulary (docs/module0_geometry_builder_equations.md
Section 4) and the mesh_handle / material_spec_stub output contracts (Section
7, consumed by Module 1 per docs/module1_mesh_interface_equations.md Section 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- Volume tags (Section 4.1) ---
SUBSTRATE = "SUBSTRATE"
LC = "LC"
AIR = "AIR"
PML_TOP = "PML_TOP"

# --- Surface tags -- Module 1's original list (Section 4.2) ---
PEC_GROUND = "PEC_GROUND"
PEC_LINE = "PEC_LINE"
PORT_1 = "PORT_1"
PORT_2 = "PORT_2"

# --- Surface tags -- new, introduced by this module (Section 4.3): these
# extend Module 1's `boundary_faces(tag)` vocabulary beyond PEC/PORT_p/
# PML_OUTER -- PML_OUTER_PEC is this module's concrete PML_OUTER, and
# PMC_SIDE is the explicitly-tagged natural (no boundary term) truncation. ---
PML_OUTER_PEC = "PML_OUTER_PEC"
PMC_SIDE = "PMC_SIDE"

VOLUME_TAGS: tuple[str, ...] = (SUBSTRATE, LC, AIR, PML_TOP)
SURFACE_TAGS: tuple[str, ...] = (PEC_GROUND, PEC_LINE, PORT_1, PORT_2, PML_OUTER_PEC, PMC_SIDE)


@dataclass(frozen=True)
class MeshHandle:
    """The raw tagged mesh Module 1 (`mesh.interface`) consumes (Module 1
    doc Section 0): vertex coordinates, tet connectivity, a per-tet volume
    tag, and named boundary-face groups. Module 0's responsibility ends
    here (Section 7) -- everything downstream only ever sees this."""

    vertices: np.ndarray  # (N_v, 3) float
    tets: np.ndarray  # (N_t, 4) int, global vertex indices
    volume_tags: np.ndarray  # (N_t,) str, one of VOLUME_TAGS
    surface_tags: dict[str, np.ndarray]  # name -> (M, 3) int, global vertex indices


@dataclass(frozen=True)
class MaterialSpecStub:
    """Auto-generated material-spec fragment (Section 5) -- SUBSTRATE/AIR
    only; `LC` is deliberately withheld (its entry belongs to
    `material.spec` proper, supplied once the director-field module is
    wired in)."""

    entries: dict[str, dict[str, float | str]]
