"""Validation suite for standard_shapes.py (docs/meshing_module_plan.md
Section 6.3, "standard_shapes.py"). Volumes are computed independently,
right here, from each shape's raw dimensions -- never by calling
CavityMode/SampleRegion's own .volume(), per Section 6.1's independence
requirement.
"""
import gmsh
import numpy as np
import pytest

from cavity_perturbation.cavity import CoaxialCavity, CylindricalCavity, RectangularCavity
from cavity_perturbation.meshing import standard_shapes
from cavity_perturbation.sample import Cylinder, Slab, Sphere


def _occ_volume(dim_tag: tuple[int, int]) -> float:
    return gmsh.model.occ.getMass(*dim_tag)  # type: ignore[no-any-return]


# --- Cavities ----------------------------------------------------------------

def test_rectangular_cavity_volume(gmsh_model):
    a, b, c = 0.03, 0.04, 0.05
    cav = RectangularCavity(a, b, c)
    dim_tag = standard_shapes.build_cavity_solid(cav)
    assert _occ_volume(dim_tag) == pytest.approx(a * b * c, rel=1e-9)


def test_cylindrical_cavity_volume(gmsh_model):
    radius, length = 0.02, 0.05
    cav = CylindricalCavity(radius, length)
    dim_tag = standard_shapes.build_cavity_solid(cav)
    assert _occ_volume(dim_tag) == pytest.approx(np.pi * radius**2 * length, rel=1e-6)


def test_coaxial_cavity_volume(gmsh_model):
    r_inner, r_outer, length = 0.01, 0.023, 0.2
    cav = CoaxialCavity(r_inner, r_outer, length)
    dim_tag = standard_shapes.build_cavity_solid(cav)
    expected = np.pi * (r_outer**2 - r_inner**2) * length
    assert _occ_volume(dim_tag) == pytest.approx(expected, rel=1e-6)


def test_unsupported_cavity_type_raises(gmsh_model):
    class FakeCavity:
        pass

    with pytest.raises(ValueError):
        standard_shapes.build_cavity_solid(FakeCavity())  # type: ignore[arg-type]


# --- Samples -------------------------------------------------------------

def test_sphere_sample_volume(gmsh_model):
    region = Sphere(center=[0.01, 0.02, 0.03], radius=0.005)
    dim_tag = standard_shapes.build_sample_solid(region)
    assert _occ_volume(dim_tag) == pytest.approx(4.0 / 3.0 * np.pi * 0.005**3, rel=1e-6)


def test_cylinder_sample_volume(gmsh_model):
    region = Cylinder(center=[0.01, 0.02, 0.03], axis=[0.0, 1.0, 0.0], radius=0.001, height=0.01)
    dim_tag = standard_shapes.build_sample_solid(region)
    assert _occ_volume(dim_tag) == pytest.approx(np.pi * 0.001**2 * 0.01, rel=1e-6)


def test_slab_sample_volume(gmsh_model):
    region = Slab(center=[0.01, 0.02, 0.03], normal=[0.0, 0.0, 1.0], thickness=0.001, extent=(0.01, 0.02))
    dim_tag = standard_shapes.build_sample_solid(region)
    assert _occ_volume(dim_tag) == pytest.approx(0.001 * 0.01 * 0.02, rel=1e-6)


def test_slab_sample_volume_arbitrary_normal(gmsh_model):
    region = Slab(center=[0.0, 0.0, 0.0], normal=[1.0, 1.0, 1.0], thickness=0.001, extent=(0.01, 0.02))
    dim_tag = standard_shapes.build_sample_solid(region)
    assert _occ_volume(dim_tag) == pytest.approx(0.001 * 0.01 * 0.02, rel=1e-6)


def test_unsupported_sample_type_raises(gmsh_model):
    class FakeRegion:
        pass

    with pytest.raises(ValueError):
        standard_shapes.build_sample_solid(FakeRegion())  # type: ignore[arg-type]


# --- Position/orientation cross-checks (volume alone can't catch these) ----

def test_cylinder_sample_position_centroid(gmsh_model):
    """A base-point-vs-center convention bug would still pass the volume
    check above but move the centroid -- check it explicitly."""
    center = np.array([0.01, 0.02, 0.03])
    region = Cylinder(center=center, axis=[0.0, 0.0, 1.0], radius=0.001, height=0.01)
    dim_tag = standard_shapes.build_sample_solid(region)
    com = gmsh.model.occ.getCenterOfMass(*dim_tag)
    assert np.array(com) == pytest.approx(center, abs=1e-9)


def test_slab_sample_position_and_orientation(gmsh_model):
    """A wrong rotation could still preserve volume -- check the bounding
    box extent along each axis lands where `normal` says it should."""
    center = np.array([0.0, 0.0, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    region = Slab(center=center, normal=normal, thickness=0.002, extent=(0.01, 0.03))
    dim_tag = standard_shapes.build_sample_solid(region)
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(*dim_tag)
    # rel=1e-3, not tighter: OCC's bounding box on a rotated B-rep carries a
    # little of its own tessellation/tolerance slop -- still easily tight
    # enough to catch a genuine wrong-axis orientation bug (which would be
    # off by 5x-15x, not 1e-4).
    assert (ymax - ymin) == pytest.approx(0.002, rel=1e-3)  # thickness along normal (y)
    extents_xz = sorted([xmax - xmin, zmax - zmin])
    assert extents_xz == pytest.approx([0.01, 0.03], rel=1e-3)  # extent[0]/[1] on the other two axes
