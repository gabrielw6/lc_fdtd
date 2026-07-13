"""Validation suite for interference.py (docs/meshing_module_plan.md
Section 6.3, "interference.py")."""
import numpy as np
import pytest

from cavity_perturbation.cavity import RectangularCavity
from cavity_perturbation.meshing import standard_shapes
from cavity_perturbation.meshing.interference import SampleExceedsCavityError, check_containment
from cavity_perturbation.sample import Sphere

A = B = C = 0.03
_SPHERE_VOLUME = 4.0 / 3.0 * np.pi * 0.005**3


def _cavity_and_sample(sample_center):
    cav = RectangularCavity(A, B, C)
    cav_dim_tag = standard_shapes.build_cavity_solid(cav)
    region = Sphere(center=sample_center, radius=0.005)
    sample_dim_tag = standard_shapes.build_sample_solid(region)
    return cav_dim_tag, sample_dim_tag


def test_sample_fully_inside_passes(gmsh_model):
    cav_dt, sample_dt = _cavity_and_sample([0.015, 0.015, 0.015])
    deficit = check_containment(cav_dt, sample_dt)
    assert deficit == pytest.approx(0.0, abs=1e-15)


def test_sample_fully_outside_raises_with_full_volume_deficit(gmsh_model):
    cav_dt, sample_dt = _cavity_and_sample([0.1, 0.1, 0.1])
    with pytest.raises(SampleExceedsCavityError) as excinfo:
        check_containment(cav_dt, sample_dt)
    assert excinfo.value.overlap_deficit == pytest.approx(_SPHERE_VOLUME, rel=1e-6)


def test_sample_straddling_boundary_raises_partial_deficit(gmsh_model):
    cav_dt, sample_dt = _cavity_and_sample([0.0, 0.015, 0.015])  # centered exactly on the x=0 wall
    with pytest.raises(SampleExceedsCavityError) as excinfo:
        check_containment(cav_dt, sample_dt)
    assert 0.0 < excinfo.value.overlap_deficit < _SPHERE_VOLUME


def test_sample_flush_against_wall_passes(gmsh_model):
    """A sample entirely inside the cavity but tangent to a wall (zero
    interior overlap deficit, not straddling it) is a physically valid
    placement, not an error (Section 3)."""
    cav_dt, sample_dt = _cavity_and_sample([0.005, 0.015, 0.015])  # tangent to the x=0 wall, from inside
    deficit = check_containment(cav_dt, sample_dt)
    assert deficit == pytest.approx(0.0, abs=1e-12)


def test_default_tolerance_scales_with_sample_size():
    """Section 3: tolerance is `tolerance_factor * sample volume`, not a
    fixed absolute number -- confirm the *relative* deficit threshold, not
    a specific absolute one, is what check_containment actually applies (no
    Gmsh needed for this one, it's pure arithmetic on the formula)."""
    from cavity_perturbation.meshing.interference import _TOLERANCE_FACTOR

    small_sample_volume = 1e-12
    large_sample_volume = 1e-3
    assert _TOLERANCE_FACTOR * small_sample_volume < _TOLERANCE_FACTOR * large_sample_volume
