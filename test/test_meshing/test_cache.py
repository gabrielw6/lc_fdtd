"""Validation suite for cache.py (docs/meshing_module_plan.md Section 6.3,
"cache.py"). No Gmsh needed -- pure key/lookup logic, not meshing."""
from pathlib import Path

from meshing.cache import MeshCache, geometry_cache_key
from meshing.geometry_spec import Box, Sphere, StepCavityInput


def _standard_inputs(radius: float = 0.005) -> tuple[Box, Sphere]:
    cav = Box(0.03, 0.03, 0.03)
    region = Sphere(center=[0.015, 0.015, 0.015], radius=radius)
    return cav, region


def test_identical_by_value_inputs_produce_the_same_key_across_different_instances():
    """Two SEPARATE shape instances with the same dimensions must hash
    identically -- proving the key is value-based, not id()-based
    (Section 5)."""
    cavity_input1, sample_input1 = _standard_inputs()
    cavity_input2, sample_input2 = _standard_inputs()
    assert cavity_input1 is not cavity_input2
    assert sample_input1 is not sample_input2

    key1 = geometry_cache_key(cavity_input1, sample_input1, target_elements_per_wavelength=10)
    key2 = geometry_cache_key(cavity_input2, sample_input2, target_elements_per_wavelength=10)
    assert key1 == key2


def test_differing_sample_dimension_produces_a_different_key():
    cavity_input1, sample_input1 = _standard_inputs(radius=0.005)
    cavity_input2, sample_input2 = _standard_inputs(radius=0.006)
    key1 = geometry_cache_key(cavity_input1, sample_input1, target_elements_per_wavelength=10)
    key2 = geometry_cache_key(cavity_input2, sample_input2, target_elements_per_wavelength=10)
    assert key1 != key2


def test_differing_resolution_produces_a_different_key():
    cavity_input, sample_input = _standard_inputs()
    key1 = geometry_cache_key(cavity_input, sample_input, target_elements_per_wavelength=10)
    key2 = geometry_cache_key(cavity_input, sample_input, target_elements_per_wavelength=20)
    assert key1 != key2


def test_step_inputs_are_directly_hashable_as_cache_keys():
    _, sample_input = _standard_inputs()
    cavity_input1 = StepCavityInput(path=Path("foo.step"), length_unit="mm")
    cavity_input2 = StepCavityInput(path=Path("foo.step"), length_unit="mm")
    key1 = geometry_cache_key(cavity_input1, sample_input, 10)
    key2 = geometry_cache_key(cavity_input2, sample_input, 10)
    assert key1 == key2

    cavity_input3 = StepCavityInput(path=Path("foo.step"), length_unit="m")  # different unit
    key3 = geometry_cache_key(cavity_input3, sample_input, 10)
    assert key3 != key1


def test_get_or_compute_hits_cache_on_repeat_call():
    cache = MeshCache()
    cavity_input, sample_input = _standard_inputs()
    calls = {"n": 0}

    def compute() -> object:
        calls["n"] += 1
        return object()

    result1 = cache.get_or_compute(cavity_input, sample_input, 10, compute)
    result2 = cache.get_or_compute(cavity_input, sample_input, 10, compute)
    assert result1 is result2
    assert calls["n"] == 1
    assert len(cache) == 1


def test_get_or_compute_hits_cache_for_new_instance_with_same_values():
    """Confirms the cache lookup, not just the key equality, works
    end-to-end across distinct instances -- skipping re-meshing exactly
    when Section 5 says it should."""
    cache = MeshCache()
    cavity_input1, sample_input1 = _standard_inputs()
    cavity_input2, sample_input2 = _standard_inputs()
    calls = {"n": 0}

    def compute() -> object:
        calls["n"] += 1
        return object()

    r1 = cache.get_or_compute(cavity_input1, sample_input1, 10, compute)
    r2 = cache.get_or_compute(cavity_input2, sample_input2, 10, compute)
    assert r1 is r2
    assert calls["n"] == 1


def test_get_or_compute_misses_cache_for_differing_resolution():
    cache = MeshCache()
    cavity_input, sample_input = _standard_inputs()
    calls = {"n": 0}

    def compute() -> object:
        calls["n"] += 1
        return object()

    cache.get_or_compute(cavity_input, sample_input, 10, compute)
    cache.get_or_compute(cavity_input, sample_input, 20, compute)
    assert calls["n"] == 2
    assert len(cache) == 2
