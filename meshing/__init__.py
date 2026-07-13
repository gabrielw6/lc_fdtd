"""cavity_perturbation.meshing -- turns a cavity + sample geometry
specification into a tagged, conformal tetrahedral mesh
(docs/meshing_module_plan.md). No dependency on perturbation.py,
inverse.py, or any EM physics -- geometry only.

Requires the "mesh" extra (gmsh, meshio): pip install -e ".[mesh]".
"""
from .cache import MeshCache
from .geometry_spec import (
    CavityGeometryInput,
    RigidTransform,
    SampleGeometryInput,
    StandardCavityInput,
    StandardSampleInput,
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
    "StandardCavityInput",
    "StandardSampleInput",
    "StepCavityInput",
    "StepSampleInput",
    "RigidTransform",
    "SampleExceedsCavityError",
    "DegenerateMeshError",
    "StepUnitAmbiguityError",
]
