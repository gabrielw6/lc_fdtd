"""Validation suite for ports.sizing -- informational port-aperture sizing
warnings (never raises), added alongside the Module 0/4 port-aperture
decoupling. `check_port_sizing` itself is a pure function (no Gmsh
needed); `check_port_sizing_for_cross_section`/`infer_h_sub_and_trace_width`
are exercised against a real extracted cross-section (Gmsh-gated) since
they read a `PortCrossSection`'s own tagging.
"""
import numpy as np
import pytest

from ports.sizing import check_port_sizing, check_port_sizing_for_cross_section, infer_h_sub_and_trace_width

# --- check_port_sizing: pure function, no Gmsh --------------------------------

_W_SUB = 0.010
_H_SUB = 0.002
_W = 0.002
_EPS_R = 3.0


def test_silent_when_aperture_well_sized():
    """A small, well-margined aperture at a low enough frequency should
    trigger neither rule."""
    msgs = check_port_sizing(W_port=0.010, H_port=0.010, h_sub=0.0005, w=0.0005, eps_r_max=1.0, f_max=1e9)
    assert msgs == []


def test_box_mode_warning_fires_when_aperture_at_or_above_lambda_min_over_2():
    # lambda_min/2 at 25 GHz, eps_r=3 is ~3.46mm; 8mm is well above it.
    msgs = check_port_sizing(W_port=0.008, H_port=0.006, h_sub=_H_SUB, w=_W, eps_r_max=_EPS_R, f_max=25e9)
    assert any("lambda_min/2" in m for m in msgs)


def test_box_mode_warning_silent_when_aperture_safely_below_bound():
    msgs = check_port_sizing(W_port=0.001, H_port=0.001, h_sub=_H_SUB, w=_W, eps_r_max=_EPS_R, f_max=25e9)
    assert not any("lambda_min/2" in m for m in msgs)


def test_fringe_width_warning_fires_when_w_port_too_narrow():
    # width_floor = w + 6*h_sub = 0.002 + 0.012 = 0.014
    msgs = check_port_sizing(W_port=0.008, H_port=0.010, h_sub=_H_SUB, w=_W, eps_r_max=1.0, f_max=1e9)
    assert any("clip the trace's transverse fringing field" in m for m in msgs)


def test_fringe_width_warning_silent_when_w_port_wide_enough():
    msgs = check_port_sizing(W_port=0.020, H_port=0.020, h_sub=_H_SUB, w=_W, eps_r_max=1.0, f_max=1e9)
    assert not any("transverse fringing field" in m for m in msgs)


def test_fringe_height_warning_fires_when_h_port_too_short():
    # height_floor = h_sub + 4*h_sub = 5*0.002 = 0.010
    msgs = check_port_sizing(W_port=0.020, H_port=0.006, h_sub=_H_SUB, w=_W, eps_r_max=1.0, f_max=1e9)
    assert any("vertical fringing field" in m for m in msgs)


def test_fringe_height_warning_silent_when_h_port_tall_enough():
    msgs = check_port_sizing(W_port=0.020, H_port=0.020, h_sub=_H_SUB, w=_W, eps_r_max=1.0, f_max=1e9)
    assert not any("vertical fringing field" in m for m in msgs)


def test_never_raises_on_degenerate_inputs():
    # w=0, h_sub=0 -- shouldn't happen in practice, but the function must
    # not raise regardless (it's informational only).
    msgs = check_port_sizing(W_port=0.001, H_port=0.001, h_sub=0.0, w=0.0, eps_r_max=1.0, f_max=1e9)
    assert isinstance(msgs, list)


# --- check_port_sizing_for_cross_section / infer_h_sub_and_trace_width --------


@pytest.fixture(scope="module")
def port1_cross_section():
    pytest.importorskip("gmsh")
    from geometry_builder import GeometryBuilder, GeometryParams
    from mesh_interface import MeshInterface
    from ports.cross_section import extract_cross_section

    params = GeometryParams(
        w=_W, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=_H_SUB, W_sub=_W_SUB,
        eps_r_substrate=_EPS_R, reference_frequency=6e9, target_elements_per_wavelength=6,
        W_port=0.008, H_port=0.006,
    )
    mesh_handle, _material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    return extract_cross_section(mesh, "PORT_1")


def test_infer_h_sub_and_trace_width_matches_geometry(port1_cross_section):
    h_sub, w = infer_h_sub_and_trace_width(port1_cross_section)
    assert h_sub == pytest.approx(_H_SUB, abs=1e-9)
    assert w == pytest.approx(_W, abs=1e-9)


def test_check_port_sizing_for_cross_section_matches_manual_computation(port1_cross_section):
    from material import ConstantMaterial, MaterialAssembly

    materials = MaterialAssembly({"SUBSTRATE": ConstantMaterial(eps_r=_EPS_R), "AIR": ConstantMaterial(eps_r=1.0)})
    msgs = check_port_sizing_for_cross_section(port1_cross_section, materials, f_max=25e9)
    manual = check_port_sizing(W_port=0.008, H_port=0.006, h_sub=_H_SUB, w=_W, eps_r_max=_EPS_R, f_max=25e9)
    assert msgs == manual
    assert len(msgs) > 0  # this aperture/frequency combination is known to warn (see test_sizing above)
