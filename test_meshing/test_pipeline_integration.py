"""Integration test for pipeline.py / build_mesh() (docs/meshing_module_plan.md
Section 6.3, "test_pipeline_integration.py"). The only test file in this
suite allowed to exercise more than one atomized piece together.
"""
import math
from pathlib import Path

import pytest

from cavity_perturbation.cavity import ModeIndex, RectangularCavity
from cavity_perturbation.meshing import tagging as tagging_module
from cavity_perturbation.meshing.cache import MeshCache
from cavity_perturbation.meshing.geometry_spec import (
    RigidTransform,
    StandardCavityInput,
    StandardSampleInput,
    StepCavityInput,
    StepSampleInput,
)
from cavity_perturbation.meshing.interference import SampleExceedsCavityError
from cavity_perturbation.meshing.pipeline import build_mesh
from cavity_perturbation.sample import Sphere

FIXTURE = Path(__file__).parent / "fixtures" / "unit_cube.step"
_IDENTITY_ROTATION = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def test_standard_shape_path_end_to_end():
    cav = RectangularCavity(0.03, 0.03, 0.03, ModeIndex("TE", (0, 1, 1)))
    region = Sphere(center=[0.015, 0.015, 0.015], radius=0.003)

    # target_elements_per_wavelength=20, not the module's smaller default:
    # with curvature-adaptive sizing intentionally off (mesh_generation.py,
    # matching Section 4's "uniformly, for a first implementation"), a
    # uniform element size *larger* than the sample itself produces a
    # degenerate triangulation on its curved surface -- this needs h_char
    # comfortably smaller than the 3mm sample radius.
    result = build_mesh(
        StandardCavityInput(cavity_mode=cav),
        StandardSampleInput(region=region),
        target_elements_per_wavelength=20,
    )

    assert result.mesh_stats.n_elements > 0
    assert result.mesh_stats.min_element_quality > 0.0
    assert result.cavity_volume == pytest.approx(0.03**3, rel=1e-9)
    assert result.sample_volume == pytest.approx(4.0 / 3.0 * math.pi * 0.003**3, rel=1e-6)
    assert len({result.sample_physical_tag, result.background_physical_tag, result.boundary_physical_tag}) == 3


def test_step_import_path_end_to_end():
    """The unit_cube.step fixture (raw 1x1x1 units) used as BOTH the cavity
    (length_unit='m' -> a literal 1m cube) and, scaled down and
    repositioned, the sample (length_unit='mm' -> a 1mm cube, translated to
    sit inside the cavity) -- exercising the STEP path for both geometry
    sources at once."""
    cavity_input = StepCavityInput(path=FIXTURE, length_unit="m")
    sample_input = StepSampleInput(
        path=FIXTURE,
        length_unit="mm",
        transform=RigidTransform(translation=(0.4, 0.4, 0.4), rotation=_IDENTITY_ROTATION),
    )

    result = build_mesh(
        cavity_input,
        sample_input,
        target_elements_per_wavelength=6,
        reference_frequency=1e9,
    )

    assert result.mesh_stats.n_elements > 0
    assert result.cavity_volume == pytest.approx(1.0, rel=1e-6)
    assert result.sample_volume == pytest.approx(1e-9, rel=1e-6)


def test_step_cavity_without_reference_frequency_raises():
    cavity_input = StepCavityInput(path=FIXTURE, length_unit="m")
    sample_input = StepSampleInput(
        path=FIXTURE,
        length_unit="mm",
        transform=RigidTransform(translation=(0.4, 0.4, 0.4), rotation=_IDENTITY_ROTATION),
    )
    with pytest.raises(ValueError):
        build_mesh(cavity_input, sample_input, target_elements_per_wavelength=6)


def test_interference_failure_stops_before_any_meshing_work(monkeypatch):
    """Sample fully outside the cavity -- confirm build_mesh raises
    SampleExceedsCavityError AND never even calls tagging.fragment_and_tag
    (Section 3: stops before fragment/tag/mesh, not a slower, more
    confusing failure downstream)."""
    called = {"fragment_and_tag": False}
    original = tagging_module.fragment_and_tag

    def spy(*args, **kwargs):
        called["fragment_and_tag"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(tagging_module, "fragment_and_tag", spy)

    cav = RectangularCavity(0.03, 0.03, 0.03, ModeIndex("TE", (0, 1, 1)))
    region = Sphere(center=[10.0, 10.0, 10.0], radius=0.003)  # nowhere near the cavity
    with pytest.raises(SampleExceedsCavityError):
        build_mesh(StandardCavityInput(cavity_mode=cav), StandardSampleInput(region=region))

    assert called["fragment_and_tag"] is False


def test_cache_skips_recompute_across_build_mesh_calls():
    cache = MeshCache()
    cav = RectangularCavity(0.03, 0.03, 0.03, ModeIndex("TE", (0, 1, 1)))
    region = Sphere(center=[0.015, 0.015, 0.015], radius=0.003)

    result1 = build_mesh(
        StandardCavityInput(cavity_mode=cav),
        StandardSampleInput(region=region),
        target_elements_per_wavelength=20,
        cache=cache,
    )

    # A fresh CavityMode/SampleRegion instance with the SAME values --
    # should still hit the cache (Section 5), not remesh.
    cav2 = RectangularCavity(0.03, 0.03, 0.03, ModeIndex("TE", (0, 1, 1)))
    region2 = Sphere(center=[0.015, 0.015, 0.015], radius=0.003)
    result2 = build_mesh(
        StandardCavityInput(cavity_mode=cav2),
        StandardSampleInput(region=region2),
        target_elements_per_wavelength=20,
        cache=cache,
    )

    assert result1 is result2
    assert len(cache) == 1
