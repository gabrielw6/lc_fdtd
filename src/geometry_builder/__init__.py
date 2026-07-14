"""geometry_builder -- Module 0: parametric geometry & tagging for the
fixed microstrip + LC-cutout topology (docs/module0_geometry_builder_equations.md).

Sits upstream of Module 1 (`mesh.interface`, not yet built): it takes a
small set of user-supplied dimensions, builds the one fixed topology, tags
it with the exact vocabulary Module 1 consumes, and hands the result
onward. It does not do general CAD -- the topology never changes, only the
numbers.

Depends on `meshing`'s generic primitives only (box/fragment/physical-group
construction, mesh generation, mesh I/O, mesh sizing); `meshing` has no
dependency in the other direction and no knowledge of this module.
"""
from .builder import GeometryBuilder, GeometryConsistencyError
from .material_spec import build_material_spec
from .params import DerivedGeometry, GeometryParameterError, GeometryParams, derive
from .tags import (
    AIR,
    LC,
    PEC_GROUND,
    PEC_LINE,
    PML_OUTER_PEC,
    PML_TOP,
    PMC_SIDE,
    PORT_1,
    PORT_2,
    SUBSTRATE,
    SURFACE_TAGS,
    VOLUME_TAGS,
    MaterialSpecStub,
    MeshHandle,
)

__all__ = [
    "GeometryBuilder",
    "GeometryConsistencyError",
    "GeometryParams",
    "GeometryParameterError",
    "DerivedGeometry",
    "derive",
    "build_material_spec",
    "MeshHandle",
    "MaterialSpecStub",
    "SUBSTRATE",
    "LC",
    "AIR",
    "PML_TOP",
    "PEC_GROUND",
    "PEC_LINE",
    "PORT_1",
    "PORT_2",
    "PML_OUTER_PEC",
    "PMC_SIDE",
    "SURFACE_TAGS",
    "VOLUME_TAGS",
]
