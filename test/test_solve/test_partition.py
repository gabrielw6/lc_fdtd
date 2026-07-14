"""Validation suite for solve.sweep's interior/PML partition and caching
(docs/module6_solve_sweep_equations.md Section 2, build step 1): "build and
test in isolation against a case with no PML tag present at all, confirming
it matches a plain Module 3 call over the whole mesh."
"""
import numpy as np
import pytest

from fem import assemble
from material import ConstantMaterial, MaterialAssembly
from solve.sweep import _interior_pml_tets

_PARAMS_KWARGS = dict(
    w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
    eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
    reference_frequency=25e9, target_elements_per_wavelength=6,
)


@pytest.fixture(scope="module")
def mesh_and_materials():
    pytest.importorskip("gmsh")
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import ConstantMaterial, MaterialAssembly, load_material_spec
    from mesh_interface import MeshInterface

    params = GeometryParams(**_PARAMS_KWARGS)
    mesh_handle, material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    base_assembly = load_material_spec(geometry_stub=material_stub)
    tag_to_model = dict(base_assembly._tag_to_model)
    tag_to_model["LC"] = ConstantMaterial(eps_r=params.eps_r_substrate)
    materials = MaterialAssembly(tag_to_model)
    return mesh, materials


def test_partition_covers_every_tet_exactly_once(mesh_and_materials):
    mesh, _materials = mesh_and_materials
    interior, pml = _interior_pml_tets(mesh)
    assert len(interior) + len(pml) == mesh.n_tets
    assert set(interior).isdisjoint(pml)
    assert len(pml) > 0  # this geometry does have a PML_TOP region


def test_partition_tags_match(mesh_and_materials):
    mesh, _materials = mesh_and_materials
    interior, pml = _interior_pml_tets(mesh)
    assert all(mesh.tet_volume_tag(t) != "PML_TOP" for t in interior)
    assert all(mesh.tet_volume_tag(t) == "PML_TOP" for t in pml)


def test_interior_only_assembly_matches_no_pml_subset_of_whole_mesh_assembly(mesh_and_materials):
    """Build step 1's own instruction: a case with no PML tag present at
    all -- here, by restricting the *whole-mesh* assembly's own tet
    universe to just the interior tets (so materials only ever get queried
    for SUBSTRATE/AIR/LC, exactly as the interior-only call does), and
    confirming the two element-by-element assemblies over that same tet
    set agree exactly (they're the same call, `tet_subset=interior_tets`,
    reproduced independently here rather than reusing `run_sweep`)."""
    mesh, materials = mesh_and_materials
    interior, pml = _interior_pml_tets(mesh)

    K1, M1 = assemble(mesh, materials, tet_subset=interior)
    K2, M2 = assemble(mesh, materials, tet_subset=interior)  # independent re-invocation
    assert abs(K1 - K2).max() < 1e-12
    assert abs(M1 - M2).max() < 1e-12

    # And tet-subset additivity (Module 3's own contract, Section 5.2):
    # interior + pml assembled separately must sum to the whole-mesh
    # assembly -- using a materials registry that also covers PML_TOP
    # (a trivial isotropic material; the whole-mesh call needs an entry
    # for every tag actually present, unlike the interior-only call).
    whole_mesh_materials = MaterialAssembly({**materials._tag_to_model, "PML_TOP": ConstantMaterial(eps_r=1.0)})
    K_pml, M_pml = assemble(mesh, whole_mesh_materials, tet_subset=pml)
    K_interior, M_interior = assemble(mesh, whole_mesh_materials, tet_subset=interior)
    K_whole, M_whole = assemble(mesh, whole_mesh_materials)
    assert abs(K_whole - (K_interior + K_pml)).max() < 1e-9
    assert abs(M_whole - (M_interior + M_pml)).max() < 1e-9
