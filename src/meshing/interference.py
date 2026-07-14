"""Meshing module -- containment check via OCC boolean intersection + mass
properties (docs/meshing_module_plan.md Section 3).

Computed entirely from Gmsh's own OCC boolean-intersection and mass-
property query, independent of however the sample/cavity geometry was
specified (Section 0.3/6.1) -- so this check means something even when the
geometry came from the same primitive-shape dimensions being tested.
"""
from __future__ import annotations

DimTag = tuple[int, int]

_TOLERANCE_FACTOR = 1e-6  # relative to sample volume (Section 3) -- sample sizes span orders of magnitude here


class SampleExceedsCavityError(Exception):
    def __init__(self, overlap_deficit: float) -> None:
        self.overlap_deficit = overlap_deficit
        super().__init__(f"sample is not fully contained in the cavity: overlap deficit = {overlap_deficit!r} m^3")


def check_containment(
    cavity_dim_tag: DimTag, sample_dim_tag: DimTag, tolerance_factor: float = _TOLERANCE_FACTOR
) -> float:
    """Delta = Volume(sample) - Volume(cavity intersect sample). Delta=0 (to
    `tolerance_factor` x sample volume) if and only if sample is a subset
    of the cavity -- this single volume comparison is set-theoretically
    sufficient on its own, no separate boundary-crossing check needed; a
    sample exactly flush against the cavity wall (zero-volume surface
    contact) passes, correctly, with no special-casing (Section 3).

    Raises `SampleExceedsCavityError` if the sample isn't fully contained.
    Non-destructive: the cavity and sample solids passed in are left fully
    intact so the pipeline can still fragment and tag them afterward.

    Implementation note: this intersects *copies* of the cavity/sample, not
    the originals directly (even with `removeObject=removeTool=False`).
    When the sample is fully contained, OCC's intersection can be
    geometrically identical to the sample and come back *aliasing the
    sample's own tag* rather than a distinct new entity (verified
    empirically) -- discarding that "temporary" intersection result would
    then delete the real sample out from under the rest of the pipeline.
    Working on throwaway copies (consumed by the boolean op itself, then
    fully discarded) avoids ever touching the real geometry for what's
    meant to be a pure query.
    """
    import gmsh

    occ = gmsh.model.occ
    sample_volume = occ.getMass(*sample_dim_tag)

    cavity_copy = occ.copy([cavity_dim_tag])
    sample_copy = occ.copy([sample_dim_tag])
    occ.synchronize()

    intersection, _ = occ.intersect(cavity_copy, sample_copy, removeObject=True, removeTool=True)
    occ.synchronize()
    if intersection:
        intersection_volume = sum(occ.getMass(*dim_tag) for dim_tag in intersection)
        occ.remove(intersection, recursive=True)
        occ.synchronize()
    else:
        intersection_volume = 0.0

    overlap_deficit = sample_volume - intersection_volume
    if overlap_deficit > tolerance_factor * sample_volume:
        raise SampleExceedsCavityError(overlap_deficit)
    return overlap_deficit
