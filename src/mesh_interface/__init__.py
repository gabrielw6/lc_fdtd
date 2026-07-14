"""mesh_interface -- Module 1: `mesh.interface`, the geometry layer between
Module 0's tagged mesh and every FEM operator downstream
(docs/module1_mesh_interface_equations.md).

Depends only on `numpy`; the `MeshInterface.from_mesh_handle` convenience
constructor duck-types `geometry_builder.MeshHandle` rather than importing
it, so this module has no hard dependency on `geometry_builder` at all --
it consumes "a list of vertices and tetrahedra with physical tags," not
anything specific to how Module 0 produced them.
"""
from .interface import MeshGeometryError, MeshInterface
from .quadrature import QuadratureError

__all__ = ["MeshInterface", "MeshGeometryError", "QuadratureError"]
