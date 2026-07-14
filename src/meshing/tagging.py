"""Meshing module -- boolean fragments + physical group tagging
(docs/meshing_module_plan.md Section 1, step 6). Run only after the
interference check (interference.py) passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

DimTag = tuple[int, int]


@dataclass(frozen=True)
class TaggedGeometry:
    sample_dim_tags: tuple[DimTag, ...]
    background_dim_tags: tuple[DimTag, ...]
    boundary_dim_tags: tuple[DimTag, ...]
    sample_physical_tag: int
    background_physical_tag: int
    boundary_physical_tag: int


def fragment_and_tag(cavity_dim_tag: DimTag, sample_dim_tag: DimTag) -> TaggedGeometry:
    """Boolean-fragments the cavity solid against the (already-contained --
    interference.py's job, not this function's) sample solid, splitting the
    cavity into (sample-shaped, background) pieces, and assigns three
    physical groups: sample (dim 3), background (dim 3), and boundary (dim
    2, the outer surface of the whole assembly -- `getBoundary(...,
    combined=True)` excludes the internal sample/background interface,
    leaving only the true exterior cavity walls)."""
    import gmsh

    occ = gmsh.model.occ
    out, out_map = occ.fragment([cavity_dim_tag], [sample_dim_tag])
    occ.synchronize()

    # out_map[0]: sub-shapes the cavity solid fragmented into (background +
    # the sample-shaped piece it shares with the tool). out_map[1]: sub-
    # shapes the sample solid fragmented into (just the sample volume
    # itself, since it's fully contained). The tag shared by both *is* the
    # sample; everything else from the cavity side is background.
    sample_tags = set(out_map[1])
    background_tags = set(out_map[0]) - sample_tags
    all_tags = sample_tags | background_tags

    boundary = gmsh.model.getBoundary(list(all_tags), combined=True, oriented=False)

    sample_pg = gmsh.model.addPhysicalGroup(3, [tag for _, tag in sample_tags], name="sample")
    background_pg = gmsh.model.addPhysicalGroup(3, [tag for _, tag in background_tags], name="background")
    boundary_pg = gmsh.model.addPhysicalGroup(2, [tag for _, tag in boundary], name="boundary")

    return TaggedGeometry(
        sample_dim_tags=tuple(sample_tags),
        background_dim_tags=tuple(background_tags),
        boundary_dim_tags=tuple(boundary),
        sample_physical_tag=sample_pg,
        background_physical_tag=background_pg,
        boundary_physical_tag=boundary_pg,
    )


def fragment(
    objects: Sequence[DimTag], tools: Sequence[DimTag] = ()
) -> tuple[list[DimTag], list[list[DimTag]]]:
    """Thin wrapper over OCC's boolean fragment across an arbitrary number
    of entities of any dimension -- the generic building block a caller
    with a fixed multi-piece topology (more than the cavity/sample pair
    `fragment_and_tag` assumes) uses directly. Returns the raw `(out,
    out_map)` Gmsh gives back: `out_map` has one entry per input (`objects`
    first, then `tools`, in input order), each the list of output entities
    that input became -- the caller's own bookkeeping for which piece is
    which."""
    import gmsh

    occ = gmsh.model.occ
    out, out_map = occ.fragment(list(objects), list(tools))
    occ.synchronize()
    return out, out_map


def add_physical_group(dim: int, tags: Sequence[int], name: str) -> int:
    """Thin wrapper over `gmsh.model.addPhysicalGroup` -- the generic
    building block for assigning an arbitrary caller-chosen name to an
    arbitrary set of entities, as opposed to `fragment_and_tag`'s fixed
    sample/background/boundary vocabulary."""
    import gmsh

    return gmsh.model.addPhysicalGroup(dim, list(tags), name=name)
