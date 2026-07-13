"""Meshing module -- geometry input contracts (docs/meshing_module_plan.md
Section 2.1-2.2).

Plain, hashable dataclasses describing where a cavity/sample's geometry
comes from (a Module 1 `CavityMode` / Module 3 `SampleRegion` instance, or
an imported STEP file) and how a custom (STEP) sample is positioned
relative to the cavity. No Gmsh/OCC dependency at all -- this module knows
nothing about meshing, only about geometry *inputs*.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..cavity import CavityMode
from ..sample import SampleRegion

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
class StandardCavityInput:
    """A Module 1 `CavityMode` instance, read-only -- position/orientation
    is whatever that instance's own frame already is."""

    cavity_mode: CavityMode


@dataclass(frozen=True)
class StepCavityInput:
    """An imported cavity solid. `length_unit` is required (Section 0.5) --
    never inferred from the STEP file's own header metadata."""

    path: Path
    length_unit: str


@dataclass(frozen=True)
class StandardSampleInput:
    """A Module 3 `SampleRegion` instance, read-only -- position/orientation
    already encoded in its `center`/`axis`/`normal` attributes, in the
    cavity's own frame."""

    region: SampleRegion


@dataclass(frozen=True)
class StepSampleInput:
    """An imported sample solid, authored in its own local frame -- unlike
    a `SampleRegion`, it has no built-in relationship to the cavity's frame,
    so `transform` is required, not optional (Section 0.4)."""

    path: Path
    length_unit: str
    transform: RigidTransform


CavityGeometryInput = StandardCavityInput | StepCavityInput
SampleGeometryInput = StandardSampleInput | StepSampleInput
