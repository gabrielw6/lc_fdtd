"""meshing -- turns an outer (background) volume + inner (sample) volume
geometry specification into a tagged, conformal tetrahedral mesh
(docs/meshing_module_plan.md). Geometry only -- no EM-physics dependency.

Requires the "mesh" extra (gmsh, meshio): pip install -e ".[mesh]".
"""
from .cache import MeshCache
from .geometry_spec import (
    Box,
    CavityGeometryInput,
    CoaxialDomain,
    Cylinder,
    CylindricalDomain,
    RigidTransform,
    SampleGeometryInput,
    Slab,
    Sphere,
    StepCavityInput,
    StepSampleInput,
)
from .interference import SampleExceedsCavityError
from .mesh_generation import DegenerateMeshError
from .mesh_io import MeshResult, MeshStats
from .pipeline import build_mesh
from .step_import import StepUnitAmbiguityError

__all__ = [
    "build_mesh",
    "MeshResult",
    "MeshStats",
    "MeshCache",
    "CavityGeometryInput",
    "SampleGeometryInput",
    "Box",
    "CylindricalDomain",
    "CoaxialDomain",
    "Sphere",
    "Cylinder",
    "Slab",
    "StepCavityInput",
    "StepSampleInput",
    "RigidTransform",
    "SampleExceedsCavityError",
    "DegenerateMeshError",
    "StepUnitAmbiguityError",
]
