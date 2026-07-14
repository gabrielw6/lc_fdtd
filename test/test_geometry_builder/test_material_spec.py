"""Validation suite for geometry_builder.material_spec (docs/module0_geometry_builder_equations.md
Section 5). No Gmsh needed -- pure data assembly."""
from geometry_builder.material_spec import build_material_spec
from geometry_builder.params import GeometryParams
from geometry_builder.tags import AIR, LC, SUBSTRATE

_BASE = dict(
    w=0.002,
    L=0.020,
    L_lc=0.008,
    W_lc=0.004,
    h_sub=0.002,
    W_sub=0.010,
    eps_r_substrate=3.0,
    reference_frequency=25e9,
)


def test_air_and_substrate_present_lc_absent():
    params = GeometryParams(**_BASE)
    spec = build_material_spec(params)

    assert set(spec.entries) == {AIR, SUBSTRATE}
    assert LC not in spec.entries
    assert spec.entries[AIR] == {"type": "constant", "eps_r": 1.0}
    assert spec.entries[SUBSTRATE] == {"type": "constant", "eps_r": 3.0}


def test_tan_delta_included_when_nonzero():
    params = GeometryParams(**_BASE, tan_delta_substrate=0.002)
    spec = build_material_spec(params)
    assert spec.entries[SUBSTRATE]["tan_delta"] == 0.002


def test_tan_delta_omitted_when_zero():
    params = GeometryParams(**_BASE, tan_delta_substrate=0.0)
    spec = build_material_spec(params)
    assert "tan_delta" not in spec.entries[SUBSTRATE]
