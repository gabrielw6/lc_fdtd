"""Meshing module -- geometry input contracts (docs/meshing_module_plan.md
Section 2.1-2.2).

Plain, hashable dataclasses describing where an outer (background) volume
and an inner (sample) volume's geometry comes from -- a standard primitive
shape, or an imported STEP file -- and how a custom (STEP) sample is
positioned relative to the outer volume. No Gmsh/OCC dependency at all --
this module knows nothing about meshing, only about geometry *inputs*.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

Matrix3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True)
class RigidTransform:
    """Rotation then translation: p_out = R @ p_in + translation. Stored as
    plain tuples (not numpy arrays) so the whole geometry spec stays
    hashable for the content-addressed cache (Section 5) -- convert to an
    array only inside the functions in transforms.py that actually apply
    this to an OCC shape. Construction/composition helpers
    (`.identity()`, `.from_axis_angle(...)`, `.translation_only(...)`,
    `compose(...)`) live in transforms.py, not here, keeping this a plain
    data container (Section 2.2)."""

    translation: tuple[float, float, float]
    rotation: Matrix3


@dataclass(frozen=True)
class Box:
    """Rectangular outer volume, axis-aligned with one corner at the origin."""

    a: float
    b: float
    c: float


@dataclass(frozen=True)
class CylindricalDomain:
    """Cylindrical outer volume, axis along z, base centered at the origin."""

    radius: float
    length: float


@dataclass(frozen=True)
class CoaxialDomain:
    """Coaxial-shell outer volume (annulus swept along z), base at the origin."""

    inner_radius: float
    outer_radius: float
    length: float


@dataclass(frozen=True)
class Sphere:
    """A spherical inner (sample) region, positioned by `center`."""

    center: tuple[float, float, float]
    radius: float


@dataclass(frozen=True)
class Cylinder:
    """A cylindrical inner (sample) region, positioned by `center` and `axis`."""

    center: tuple[float, float, float]
    axis: tuple[float, float, float]
    radius: float
    height: float


@dataclass(frozen=True)
class Slab:
    """A rectangular inner (sample) region: a box of side `extent[0]` x
    `extent[1]` in the plane perpendicular to `normal`, and `thickness`
    along `normal`, centered at `center`."""

    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    thickness: float
    extent: tuple[float, float]


@dataclass(frozen=True)
class StepCavityInput:
    """An imported outer-volume solid. `length_unit` is required (Section
    0.5) -- never inferred from the STEP file's own header metadata."""

    path: Path
    length_unit: str


@dataclass(frozen=True)
class StepSampleInput:
    """An imported sample solid, authored in its own local frame -- unlike
    a standard shape, it has no built-in relationship to the outer
    volume's frame, so `transform` is required, not optional (Section 0.4)."""

    path: Path
    length_unit: str
    transform: RigidTransform


CavityGeometryInput = Box | CylindricalDomain | CoaxialDomain | StepCavityInput
SampleGeometryInput = Sphere | Cylinder | Slab | StepSampleInput
