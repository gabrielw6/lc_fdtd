"""Validation suite for mesh_generation.py (docs/meshing_module_plan.md
Section 6.3, "mesh_generation.py"). Checks invariant properties (Section
6.2) -- Gmsh isn't required to produce bit-identical output between
runs/versions, so no exact element-count assertions.
"""
import gmsh
import pytest

from meshing.mesh_generation import DegenerateMeshError, MeshQuality, generate_mesh
from meshing.tagging import fragment_and_tag


def _build_tagged_box_with_sample():
    occ = gmsh.model.occ
    cav = occ.addBox(0, 0, 0, 0.03, 0.03, 0.03)
    sample = occ.addSphere(0.015, 0.015, 0.015, 0.005)
    occ.synchronize()
    return fragment_and_tag((3, cav), (3, sample))


def test_generates_nonzero_elements_with_positive_quality(gmsh_model):
    _build_tagged_box_with_sample()
    result = generate_mesh(0.008)
    assert isinstance(result, MeshQuality)
    assert result.n_elements > 0
    assert result.min_element_quality > 0.0
    assert result.max_aspect_ratio >= 1.0  # 1.0 is the theoretical minimum for any tetrahedron


def test_rejects_nonpositive_characteristic_length(gmsh_model):
    _build_tagged_box_with_sample()
    with pytest.raises(ValueError):
        generate_mesh(0.0)
    with pytest.raises(ValueError):
        generate_mesh(-0.001)


def test_finer_resolution_increases_element_count(gmsh_model):
    """Section 6.2: refining target_elements_per_wavelength (here, directly
    the characteristic length) upward changes n_elements in the expected
    direction, without asserting an exact count."""
    _build_tagged_box_with_sample()
    coarse = generate_mesh(0.02)
    medium = generate_mesh(0.008)  # re-generate on the same tagged geometry, finer each time
    fine = generate_mesh(0.003)
    assert coarse.n_elements < medium.n_elements < fine.n_elements
