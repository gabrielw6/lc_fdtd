"""Meshing module -- content-addressed mesh cache (docs/meshing_module_plan.md
Section 5).

Inputs are plain, hashable data by construction: standard shapes are frozen
dataclasses of plain dimension values, and STEP inputs are already plain
data (paths + `length_unit` + `RigidTransform`) -- so a plain dict is safe
to use directly, keyed on value, never on `id()`.
"""
from __future__ import annotations

from typing import Any, Callable, Hashable

from .geometry_spec import (
    Box,
    CavityGeometryInput,
    CoaxialDomain,
    Cylinder,
    CylindricalDomain,
    SampleGeometryInput,
    Slab,
    Sphere,
    StepCavityInput,
    StepSampleInput,
)


def _shape_key(shape: Any) -> Hashable:
    if isinstance(shape, Box):
        return (type(shape).__name__, (shape.a, shape.b, shape.c))
    if isinstance(shape, CylindricalDomain):
        return (type(shape).__name__, (shape.radius, shape.length))
    if isinstance(shape, CoaxialDomain):
        return (type(shape).__name__, (shape.inner_radius, shape.outer_radius, shape.length))
    if isinstance(shape, Sphere):
        return (type(shape).__name__, tuple(shape.center), shape.radius)
    if isinstance(shape, Cylinder):
        return (type(shape).__name__, tuple(shape.center), tuple(shape.axis), shape.radius, shape.height)
    if isinstance(shape, Slab):
        return (type(shape).__name__, tuple(shape.center), tuple(shape.normal), shape.thickness, shape.extent)
    raise ValueError(f"no cache key extractor registered for shape {type(shape).__name__!r}")


def geometry_cache_key(
    cavity_input: CavityGeometryInput, sample_input: SampleGeometryInput, target_elements_per_wavelength: int
) -> Hashable:
    """A single hashable key capturing the full geometry spec plus the
    resolution parameter -- plain-value hashing throughout, no id()-based
    component anywhere."""
    cavity_key: Hashable = (
        ("step", cavity_input) if isinstance(cavity_input, StepCavityInput) else ("standard", _shape_key(cavity_input))
    )
    sample_key: Hashable = (
        ("step", sample_input) if isinstance(sample_input, StepSampleInput) else ("standard", _shape_key(sample_input))
    )
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
