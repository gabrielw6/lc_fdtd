"""material -- Module 2: the constitutive tensor field eps_r(r) (and
trivially mu_r) (docs/module2_material_equations.md).

Depends only on `numpy`/`scipy`/`PyYAML`. `material.spec.load_material_spec`
accepts a `geometry_builder.MaterialSpecStub` duck-typed on `.entries` (or a
plain dict) -- this package has no import dependency on `geometry_builder`,
`mesh_interface`, or `meshing` at all; it consumes plain (M,3) point arrays
and tag strings, matching Module 3's stated contract, not anything specific
to how the mesh or geometry were built.
"""
from .core import (
    MaterialAssembly,
    MaterialError,
    MaterialModel,
    MaterialPassivityError,
    MaterialSymmetryError,
    MaterialTagError,
)
from .interpolation import CoverageError, SampledField, interpolate_scattered, interpolate_structured
from .regions import ConstantMaterial, ScalarFieldMaterial, TensorFieldMaterial
from .spec import MaterialSpecError, load_material_spec
from .tensor_interpolation import DirectorFieldError, DirectorFieldMaterial

__all__ = [
    "MaterialModel",
    "MaterialAssembly",
    "MaterialError",
    "MaterialSymmetryError",
    "MaterialPassivityError",
    "MaterialTagError",
    "SampledField",
    "CoverageError",
    "interpolate_structured",
    "interpolate_scattered",
    "ConstantMaterial",
    "ScalarFieldMaterial",
    "TensorFieldMaterial",
    "DirectorFieldMaterial",
    "DirectorFieldError",
    "load_material_spec",
    "MaterialSpecError",
]
