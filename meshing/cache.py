"""Meshing module -- content-addressed mesh cache (docs/meshing_module_plan.md
Section 5).

Unlike Module 4's `PerturbationModel` cache (which had to key on
`id(region)` and hold a strong reference to avoid Python's `id()`-reuse
hazard, since `SampleRegion` instances are runtime objects), this module's
inputs are plain, hashable data by construction: the STEP path (Section
2.1-2.2) is already plain data (paths + `length_unit` + `RigidTransform`),
and for the standard-shape path this module hashes on the wrapped
`CavityMode`/`SampleRegion`'s own dimension *values* directly -- never on
`id()` of the wrapper or the wrapped object -- so a plain dict is safe to
use directly, with no strong-reference workaround needed.
"""
from __future__ import annotations

from typing import Any, Callable, Hashable

from ..cavity import CavityMode, CoaxialCavity, CylindricalCavity, RectangularCavity
from ..sample import Cylinder, SampleRegion, Slab, Sphere
from .geometry_spec import CavityGeometryInput, SampleGeometryInput, StandardCavityInput, StandardSampleInput


def _cavity_key(cavity_mode: CavityMode) -> Hashable:
    if isinstance(cavity_mode, RectangularCavity):
        dims: tuple[float, ...] = (cavity_mode.a, cavity_mode.b, cavity_mode.c)
    elif isinstance(cavity_mode, CoaxialCavity):
        dims = (cavity_mode.a, cavity_mode.b, cavity_mode.L)
    elif isinstance(cavity_mode, CylindricalCavity):
        dims = (cavity_mode.a, cavity_mode.d)
    else:
        raise ValueError(f"no cache key extractor registered for cavity type {type(cavity_mode).__name__!r}")
    return (
        type(cavity_mode).__name__,
        dims,
        cavity_mode.mode.kind,
        cavity_mode.mode.indices,
        cavity_mode.epsilon_bg,
        cavity_mode.mu_bg,
    )


def _sample_key(region: SampleRegion) -> Hashable:
    if isinstance(region, Sphere):
        return (type(region).__name__, tuple(region.center), region.radius)
    if isinstance(region, Cylinder):
        return (type(region).__name__, tuple(region.center), tuple(region.axis), region.radius, region.height)
    if isinstance(region, Slab):
        return (type(region).__name__, tuple(region.center), tuple(region.normal), region.thickness, region.extent)
    raise ValueError(f"no cache key extractor registered for sample shape {type(region).__name__!r}")


def geometry_cache_key(
    cavity_input: CavityGeometryInput, sample_input: SampleGeometryInput, target_elements_per_wavelength: int
) -> Hashable:
    """A single hashable key capturing the full geometry spec plus the
    resolution parameter -- plain-value hashing throughout, no id()-based
    component anywhere."""
    cavity_key: Hashable
    if isinstance(cavity_input, StandardCavityInput):
        cavity_key = ("standard", _cavity_key(cavity_input.cavity_mode))
    else:
        cavity_key = ("step", cavity_input)

    sample_key: Hashable
    if isinstance(sample_input, StandardSampleInput):
        sample_key = ("standard", _sample_key(sample_input.region))
    else:
        sample_key = ("step", sample_input)

    return (cavity_key, sample_key, target_elements_per_wavelength)


class MeshCache:
    """In-memory content-addressed cache from a geometry spec (Section 5)
    to whatever the pipeline computed for it. Deliberately generic in what
    it stores (not typed to mesh_io.py's `MeshResult`) so this module has
    no dependency on the rest of the meshing pipeline's own types."""

    def __init__(self) -> None:
        self._store: dict[Hashable, Any] = {}

    def get_or_compute(
        self,
        cavity_input: CavityGeometryInput,
        sample_input: SampleGeometryInput,
        target_elements_per_wavelength: int,
        compute: Callable[[], Any],
    ) -> Any:
        key = geometry_cache_key(cavity_input, sample_input, target_elements_per_wavelength)
        if key not in self._store:
            self._store[key] = compute()
        return self._store[key]

    def __len__(self) -> int:
        return len(self._store)
