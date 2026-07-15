"""Two-port reciprocity/passivity gate on a uniform (LC-cavity-present-but-
material-matched, i.e. electromagnetically uniform) line -- the top-level
architecture doc's Phase 1 acceptance test (`docs/CLAUDE.md` Section 7).

This is also the empirical check for the per-port axial-orientation review
of `ports.cross_section.PortCrossSection.axial_sign` / `ports.mode_solver`
/ `ports.port_operator` (PORT_2's outward normal is `+x_hat`, not `-x_hat`
like PORT_1). The review's conclusion, confirmed by this exact test: the
per-port axial sign must be threaded through `ports.mode_solver`'s
axial/H-field/power quantities (`_h_t_on_triangle`, `_mode_integrals`,
`_mode_overlaps`, `_raw_overlap`) so `PortMode.h_t()` and the Poynting
power/`Y_m`-consistency checks are physically correct for either port --
but `ports.port_operator.build_B`/`build_g` (which only ever consume
`mode.Y` and `overlap_e`, both already provably axial-sign-invariant) need
**no** additional correction; an earlier attempt to add one to `build_B`
was empirically falsified by this very test (it produced a non-reciprocal,
non-passive `|S22|>1` result) and was reverted -- see `build_B`'s own
docstring for the full account.

The geometry here is deliberately the Part-3-validated single-mode 50-ohm
microstrip design (`examples/isotropic_microstrip.py`'s own parameters,
minus the trivial/non-absorbing PML swapped in for CI speed/reliability --
a real graded PML at this substrate thickness needs a much finer mesh than
this fast unit test affords, a separate mesh-resolution concern unrelated
to anything this file is testing) -- its box-mode-safe port aperture (per
`ports.sizing`) is what makes both ports reliably land on the true
quasi-TEM dominant mode at a single frequency point (no mode-tracking
possible with one point) rather than Module 4 Section 3.7's documented
box-mode limitation, which would otherwise swamp the reciprocity signal
this test is actually after.
"""
import numpy as np
import pytest

from material import ConstantMaterial

_PARAMS_KWARGS = dict(
    w=0.000629, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.00025, W_sub=0.006,
    eps_r_substrate=3.0, h_air=0.003, W_port=0.005, H_port=0.003,
    reference_frequency=10e9, target_elements_per_wavelength=10,
)
_HAMMERSTAD_JENSEN_BETA = 326.0  # rad/m, this file's own module docstring / examples/isotropic_microstrip.py


@pytest.fixture(scope="module")
def mesh_and_materials():
    pytest.importorskip("gmsh")
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import MaterialAssembly, load_material_spec
    from mesh_interface import MeshInterface

    params = GeometryParams(**_PARAMS_KWARGS)
    mesh_handle, material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)
    base_assembly = load_material_spec(geometry_stub=material_stub)
    tag_to_model = dict(base_assembly._tag_to_model)
    tag_to_model["LC"] = ConstantMaterial(eps_r=params.eps_r_substrate)  # uniform line: LC slot matches substrate
    materials = MaterialAssembly(tag_to_model)
    return mesh, materials, params


@pytest.fixture(scope="module")
def sweep_results(mesh_and_materials):
    from solve import run_sweep

    mesh, materials, params = mesh_and_materials
    pml_params = dict(
        background=ConstantMaterial(eps_r=1.0),
        z_air_top=params.h_sub + params.h_air,
        thickness=params.h_pml,
        R0=1.0, n=2, kappa_max=1.0,  # trivial (non-absorbing) PML -- see module docstring
    )
    frequencies = [2 * np.pi * params.reference_frequency]
    return run_sweep(mesh, materials, ["PORT_1", "PORT_2"], frequencies, n_modes=1, pml_params=pml_params)


@pytest.fixture(scope="module")
def S(sweep_results):
    from extract import deembed, raw_s_parameters

    raw_S = raw_s_parameters(sweep_results, ["PORT_1", "PORT_2"], 1)
    port_modes = sweep_results[0].port_modes
    return deembed(raw_S, port_modes, {"PORT_1": 0.0, "PORT_2": 0.0})


def test_axial_sign_is_opposite_at_the_two_ports(sweep_results):
    """The root-cause fact this whole gate is checking around: PORT_1's
    outward normal is -x_hat (interior at x>0), PORT_2's is +x_hat
    (interior at x<L) -- `axial_sign` must disagree between them, not be
    hardcoded to the same value."""
    port_modes = sweep_results[0].port_modes
    s1 = port_modes["PORT_1"][0].cross_section.axial_sign
    s2 = port_modes["PORT_2"][0].cross_section.axial_sign
    assert s1 == pytest.approx(1.0)
    assert s2 == pytest.approx(-1.0)


def test_both_ports_found_the_dominant_propagating_mode(sweep_results):
    """Section 6.1's starting-frequency precondition already checks this
    internally (would have raised `SweepPreconditionError` otherwise), but
    it's worth asserting directly here too: this test's whole point is a
    clean axial-sign signal, which a port that landed on a spurious
    evanescent/box mode (Module 4 Section 3.7) instead of the true
    quasi-TEM mode would otherwise swamp with unrelated noise."""
    port_modes = sweep_results[0].port_modes
    for tag in ("PORT_1", "PORT_2"):
        gamma = port_modes[tag][0].gamma
        assert abs(gamma.real) < 1e-3 * abs(gamma.imag)
        assert gamma.imag == pytest.approx(_HAMMERSTAD_JENSEN_BETA, rel=0.1)


def test_port_congruence_gamma_and_y_agree(sweep_results):
    """A uniform line's two (geometrically congruent) port cross-sections
    must extract the same gamma/Y regardless of which end of the line they
    sit at -- both are provably axial-sign-invariant (see this module's own
    docstring)."""
    port_modes = sweep_results[0].port_modes
    m1, m2 = port_modes["PORT_1"][0], port_modes["PORT_2"][0]
    assert m1.gamma == pytest.approx(m2.gamma, rel=1e-6)
    assert m1.Y == pytest.approx(m2.Y, rel=1e-6)


def test_reciprocity_s21_equals_s12(S):
    """S_(PORT_2,1),(PORT_1,1) == S_(PORT_1,1),(PORT_2,1) -- a uniform
    line's S-matrix must be symmetric."""
    s21 = S[("PORT_2", 1, "PORT_1")]
    s12 = S[("PORT_1", 1, "PORT_2")]
    assert s21 == pytest.approx(s12, rel=1e-2, abs=1e-3)


def test_reciprocity_s11_equals_s22(S):
    """A geometrically symmetric uniform line's two self-reflections must
    also agree (not just the dominant reciprocity pairing above)."""
    s11 = S[("PORT_1", 1, "PORT_1")]
    s22 = S[("PORT_2", 1, "PORT_2")]
    assert s11 == pytest.approx(s22, rel=5e-2, abs=1e-2)


def test_passivity_upper_bound(S):
    """|S11|^2 + |S21|^2 <= 1 (energy cannot be manufactured) -- a hard
    physical bound that holds regardless of how well the (here: trivial,
    non-absorbing) PML absorbs, since the whole system remains lossless and
    linear either way. A small numerical headroom (not exactly 1.0)
    accounts for finite-element discretization error, not a loosened
    physical requirement. Before the per-port axial-orientation review
    (`build_B` mistakenly carrying an extra `axial_sign` factor), this same
    gate on this same geometry gave `|S22|~2.4` -- more than double the
    physical bound -- which is exactly what this assertion is guarding
    against regressing to."""
    s11 = S[("PORT_1", 1, "PORT_1")]
    s21 = S[("PORT_2", 1, "PORT_1")]
    total = abs(s11) ** 2 + abs(s21) ** 2
    assert total <= 1.0 + 1e-2
