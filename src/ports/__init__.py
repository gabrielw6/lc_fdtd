"""ports -- Module 4: waveguide mode ports (docs/module4_ports_equations.md).

Depends on `mesh_interface.MeshInterface` (Section 0: `boundary_faces`,
`tet_volume_tag`, `pec_edge_dofs` -- no new Module 1 contract additions
needed) and `material.MaterialAssembly` (queried exactly as Module 3 does,
restricted to 2D port-face points). Never imports `fem`, `geometry_builder`,
or `meshing`.
"""
from .cross_section import CrossSectionError, PortCrossSection, extract_cross_section
from .mode_solver import (
    PortMode,
    PortModeError,
    PortModeSolver,
    biorthogonality,
    field_from_edge_dofs,
    mode_similarity,
    project,
)
from .port_operator import PortOperatorError, build_B, build_g, deembed

__all__ = [
    "PortModeSolver",
    "PortMode",
    "PortModeError",
    "project",
    "field_from_edge_dofs",
    "biorthogonality",
    "mode_similarity",
    "build_B",
    "build_g",
    "deembed",
    "PortOperatorError",
    "PortCrossSection",
    "extract_cross_section",
    "CrossSectionError",
]
