"""Regression test for the `_characteristic_length` degeneracy-floor fix
(`mesh_interface.interface`, post-review): a `max(extent, 1.0)` clamp
silently turned the intended *relative* `1e-12 * extent**3` degeneracy
floor into an *absolute* `1e-12 m^3` for every sub-metre geometry this
repo builds, which false-positively rejected well-shaped fine tets at
higher mesh densities as "slivers."

This exact geometry+mesh-density combination (`examples/isotropic_microstrip
.py`'s own parameters at `--mesh-density 16`) previously raised
`mesh_interface.interface.MeshGeometryError: tet N has near-zero/negative
volume ... <= floor 1e-12` on construction, even though every tet in this
mesh is well-shaped -- confirmed by `src/meshing/mesh_generation.py`'s own
scale-invariant `minSICN` quality gate, which the mesh already passed (it
raises its own `DegenerateMeshError` on any non-positive-quality tet, and
did not). Only the mis-scaled *absolute* floor in `mesh_interface` was
ever at fault; the mesh itself was always valid.
"""
import pytest

pytest.importorskip("gmsh")

from geometry_builder import GeometryBuilder, GeometryParams
from mesh_interface import MeshInterface

_PARAMS = GeometryParams(
    w=0.000629,
    L=0.020,
    L_lc=0.008,
    W_lc=0.004,
    h_sub=0.00025,
    W_sub=0.006,
    eps_r_substrate=3.0,
    h_air=0.003,
    W_port=0.005,
    H_port=0.003,
    reference_frequency=10e9,
    target_elements_per_wavelength=16,
)


def test_fine_mesh_density_builds_without_a_false_positive_degeneracy_rejection():
    mesh_handle, _material_stub = GeometryBuilder().build(_PARAMS)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    assert mesh.n_tets > 0
    assert all(mesh.volume(t) > 0.0 for t in range(mesh.n_tets))
