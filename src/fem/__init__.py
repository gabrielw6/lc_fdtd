"""fem -- Module 3: the Whitney/Nedelec edge basis and the frequency-
independent volume operators K, M (docs/module3_fem_assembly_equations.md).

Depends on `mesh_interface.MeshInterface` and `material.MaterialAssembly`
directly -- the architecture's own dependency graph has both feeding
`fem.assembly`, so this is the intended "interface abstraction layer"
boundary, not a violation of it. `fem` never imports `geometry_builder`,
`meshing`, or any concrete `MaterialModel` subclass.
"""
from .assembly import DEFAULT_LEVELS, AssemblyConvergenceError, assemble, element_matrices
from .edge_elements import whitney_basis, whitney_curl
from .field_eval import evaluate_curl_field, evaluate_edge_field

__all__ = [
    "assemble",
    "element_matrices",
    "AssemblyConvergenceError",
    "DEFAULT_LEVELS",
    "whitney_basis",
    "whitney_curl",
    "evaluate_edge_field",
    "evaluate_curl_field",
]
