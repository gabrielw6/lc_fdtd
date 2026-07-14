"""pml -- Module 5: the PML complex coordinate-stretching material
(docs/module5_pml_equations.md).

`PMLMaterial` implements `material.core.MaterialModel` (Module 2's
interface) rather than extending it -- `fem.assembly` consumes it through
the exact same `MaterialAssembly` tag-dispatch path as every other
material, no special-casing anywhere downstream. Depends only on
`material.MaterialModel` (dependency injection for the background) and
numpy/scipy; never imports `mesh_interface`, `geometry_builder`, `fem`, or
`ports`.
"""
from .material import PMLMaterial
from .stretching import kappa_profile, lambda_tensor, s_z, sigma_max_for_R0, sigma_profile

__all__ = [
    "PMLMaterial",
    "sigma_max_for_R0",
    "sigma_profile",
    "kappa_profile",
    "s_z",
    "lambda_tensor",
]
