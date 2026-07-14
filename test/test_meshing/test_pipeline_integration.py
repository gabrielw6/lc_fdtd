"""Integration test for pipeline.py / build_mesh() (docs/meshing_module_plan.md
Section 6.3, "test_pipeline_integration.py"). The only test file in this
suite allowed to exercise more than one atomized piece together.
"""
import math
from pathlib import Path

import pytest

from meshing import tagging as tagging_module
from meshing.cache import MeshCache
from meshing.geometry_spec import Box, RigidTransform, Sphere, StepCavityInput, StepSampleInput
from meshing.interference import SampleExceedsCavityError
from meshing.pipeline import build_mesh

FIXTURE = Path(__file__).parent / "fixtures" / "unit_cube.step"
_IDENTITY_ROTATION = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def test_standard_shape_path_end_to_end():
    cav = Box(0.03, 0.03, 0.03)
    region = Sphere(center=[0.015, 0.015, 0.015], radius=0.003)

    # target_elements_per_wavelength=20, not the module's smaller default:
    # with curvature-adaptive sizing intentionally off (mesh_generation.py,
    # matching Section 4's "uniformly, for a first implementation"), a
    # uniform element size *larger* than the sample itself produces a
    # degenerate triangulation on its curved surface -- this needs h_char
    # comfortably smaller than the 3mm sample radius.
    result = build_mesh(cav, region, reference_frequency=5e9, target_elements_per_wavelength=20)

    assert result.mesh_stats.n_elements > 0
    assert result.mesh_stats.min_element_quality > 0.0
    assert result.cavity_volume == pytest.approx(0.03**3, rel=1e-9)
    assert result.sample_volume == pytest.approx(4.0 / 3.0 * math.pi * 0.003**3, rel=1e-6)
    assert len({result.sample_physical_tag, result.background_physical_tag, result.boundary_physical_tag}) == 3


def test_step_import_path_end_to_end():
    """The unit_cube.step fixture (raw 1x1x1 units) used as BOTH the outer
    volume (length_unit='m' -> a literal 1m cube) and, scaled down and
    repositioned, the sample (length_unit='mm' -> a 1mm cube, translated to
    sit inside it) -- exercising the STEP path for both geometry sources at
    once."""
    cavity_input = StepCavityInput(path=FIXTURE, length_unit="m")
    sample_input = StepSampleInput(
        path=FIXTURE,
        length_unit="mm",
        transform=RigidTransform(translation=(0.4, 0.4, 0.4), rotation=_IDENTITY_ROTATION),
    )

    result = build_mesh(
        cavity_input,
        sample_input,
        reference_frequency=1e9,
        target_elements_per_wavelength=6,
    )

    assert result.mesh_stats.n_elements > 0
    assert result.cavity_volume == pytest.approx(1.0, rel=1e-6)
    assert result.sample_volume == pytest.approx(1e-9, rel=1e-6)


def test_interference_failure_stops_before_any_meshing_work(monkeypatch):
    """Sample fully outside the outer volume -- confirm build_mesh raises
    SampleExceedsCavityError AND never even calls tagging.fragment_and_tag
    (Section 3: stops before fragment/tag/mesh, not a slower, more
    confusing failure downstream)."""
    called = {"fragment_and_tag": False}
    original = tagging_module.fragment_and_tag

    def spy(*args, **kwargs):
        called["fragment_and_tag"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(tagging_module, "fragment_and_tag", spy)

    cav = Box(0.03, 0.03, 0.03)
    region = Sphere(center=[10.0, 10.0, 10.0], radius=0.003)  # nowhere near the outer volume
    with pytest.raises(SampleExceedsCavityError):
        build_mesh(cav, region, reference_frequency=5e9)

    assert called["fragment_and_tag"] is False


def test_cache_skips_recompute_across_build_mesh_calls():
    cache = MeshCache()
    cav = Box(0.03, 0.03, 0.03)
    region = Sphere(center=[0.015, 0.015, 0.015], radius=0.003)

    result1 = build_mesh(
        cav, region, reference_frequency=5e9, target_elements_per_wavelength=20, cache=cache
    )

    # A fresh shape instance with the SAME values -- should still hit the
    # cache (Section 5), not remesh.
    cav2 = Box(0.03, 0.03, 0.03)
    region2 = Sphere(center=[0.015, 0.015, 0.015], radius=0.003)
    result2 = build_mesh(
        cav2, region2, reference_frequency=5e9, target_elements_per_wavelength=20, cache=cache
    )

    assert result1 is result2
    assert len(cache) == 1
