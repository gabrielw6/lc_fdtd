"""Meshing module -- STEP file import + explicit unit conversion
(docs/meshing_module_plan.md Section 0.5, step 4).

**Empirical finding (verified against Gmsh 4.15.2's OCC STEP translator,
not assumed from documentation):** Gmsh's OCC importer does NOT read or
honor a STEP file's own declared unit metadata (its
GLOBAL_UNIT_ASSIGNED_CONTEXT / SI_UNIT entities) when deciding how to scale
imported coordinates -- verified by importing the identical raw coordinate
values under two fixture files that declare METRE vs. MILLIMETRE and
getting bit-identical results either way. Instead, Gmsh's
`Geometry.OCCTargetUnit` option applies a *real* geometric rescale at
import time (confirmed via `getBoundingBox`, not just a derived quantity
like mass): with it left at Gmsh's own default, a file's raw coordinate
values are silently multiplied by 1000 relative to leaving them alone.
That default is *exactly* the "predictable 1000x geometry error" Section
0.5 warns about, just one layer deeper than "the caller forgot to specify
units" -- it's the underlying library's own surprising default. Setting
`Geometry.OCCTargetUnit = "M"` makes Gmsh treat the file's raw numbers as
a literal, unscaled passthrough (1 file unit = 1 metre internally); this
module always does that first, then applies its own explicit scale from
`length_unit` on top via `occ.dilate` -- so the *caller's* stated unit is
the only thing that determines the imported geometry's real-world size,
never Gmsh's own (undocumented, file-independent) default.
"""
from __future__ import annotations

from pathlib import Path

DimTag = tuple[int, int]

_METERS_PER_UNIT = {
    "m": 1.0,
    "mm": 1e-3,
    "cm": 1e-2,
    "in": 0.0254,
}


class StepUnitAmbiguityError(Exception):
    """Raised when a STEP import is requested without an explicit
    `length_unit` (Section 0.5) -- never inferred, never defaulted."""


def import_step(path: Path | str, length_unit: str | None) -> list[DimTag]:
    """Imports every solid in the STEP file at `path`, scaling raw
    coordinates by `length_unit`'s meters-per-unit factor so the result is
    in the project's SI convention (CLAUDE.md) regardless of what the file
    itself claims or what Gmsh would otherwise default to. Requires an
    active Gmsh model (caller's responsibility -- this function doesn't
    initialize/finalize Gmsh itself)."""
    if length_unit is None:
        raise StepUnitAmbiguityError(
            f"length_unit is required to import a STEP file ({path}) -- never inferred "
            "from the file's own header, and never defaulted (docs/meshing_module_plan.md "
            "Section 0.5)"
        )
    if length_unit not in _METERS_PER_UNIT:
        raise ValueError(f"unknown length_unit {length_unit!r}, expected one of {sorted(_METERS_PER_UNIT)}")

    import gmsh

    occ = gmsh.model.occ
    # Force a raw, unscaled passthrough -- see this module's docstring for
    # why Gmsh's own default here is itself an unwanted 1000x rescale.
    gmsh.option.setString("Geometry.OCCTargetUnit", "M")
    dim_tags = occ.importShapes(str(path))
    occ.synchronize()

    scale = _METERS_PER_UNIT[length_unit]
    if scale != 1.0 and dim_tags:
        occ.dilate(dim_tags, 0.0, 0.0, 0.0, scale, scale, scale)
        occ.synchronize()
    return dim_tags
