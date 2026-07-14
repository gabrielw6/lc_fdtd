"""extract -- Module 7: modal amplitude extraction, S-parameters,
de-embedding (docs/module7_extract_sparameters_equations.md).

Depends on `ports` (`PortMode`, `extract_cross_section`,
`field_from_edge_dofs`, `project`) and `solve` (`SweepResult`) -- the
architecture's own dependency graph. Purely post-processing: never
imports `fem`, `pml`, `geometry_builder`, or `meshing`, and never solves
anything -- it only reads out what Module 6 already solved.
"""
from .sparameters import (
    SParameterDataset,
    assemble_sweep_dataset,
    deembed,
    energy_balance,
    port_face_edges,
    project_amplitude,
    raw_s_parameters,
)

__all__ = [
    "port_face_edges",
    "project_amplitude",
    "raw_s_parameters",
    "deembed",
    "energy_balance",
    "assemble_sweep_dataset",
    "SParameterDataset",
]
