"""geometry_builder.material_spec -- auto-generated material-spec stub
(docs/module0_geometry_builder_equations.md Section 5).
"""
from __future__ import annotations

from .params import GeometryParams
from .tags import AIR, SUBSTRATE, MaterialSpecStub


def build_material_spec(params: GeometryParams) -> MaterialSpecStub:
    """One set of user inputs (the same `GeometryParams` used to build the
    geometry) produces both the tagged mesh and this material-spec
    fragment for the non-LC tags."""
    substrate_entry: dict[str, float | str] = {"type": "constant", "eps_r": params.eps_r_substrate}
    if params.tan_delta_substrate:
        substrate_entry["tan_delta"] = params.tan_delta_substrate

    return MaterialSpecStub(
        entries={
            AIR: {"type": "constant", "eps_r": 1.0},
            SUBSTRATE: substrate_entry,
        }
    )
