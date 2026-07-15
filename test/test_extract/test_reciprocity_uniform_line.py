"""Two-port reciprocity/passivity gate on a uniform (LC-cavity-present-but-
material-matched, i.e. electromagnetically uniform) line -- the top-level
architecture doc's Phase 1 acceptance test (`docs/CLAUDE.md` Section 7):
`S21==S12`, `S11==S22`, and `|S11|^2+|S21|^2==1` (energy conservation),
all `== 1` to discretization error, not merely bounded.

This is also the empirical check for two separate reviews of
`ports.mode_solver`/`ports.port_operator`:

1. **Per-port axial-orientation review** (`ports.cross_section
   .PortCrossSection.axial_sign`): PORT_2's outward normal is `+x_hat`, not
   `-x_hat` like PORT_1. Conclusion, confirmed here: the per-port axial
   sign must be threaded through `ports.mode_solver`'s axial/H-field/power
   quantities (`_h_t_on_triangle`, `_mode_integrals`, `_mode_overlaps`,
   `_raw_overlap`) so `PortMode.h_t()` and the Poynting power/`Y_m`-
   consistency checks are physically correct for either port -- but
   `ports.port_operator.build_B`/`build_g` need **no** additional
   axial-sign correction (an earlier attempt to add one to `build_B` was
   empirically falsified by this same test file, reverted; see `build_B`'s
   own docstring).
2. **Injection/extraction (energy-conservation) normalization review**:
   `build_B` was missing an explicit `1/N_m` factor (`N_m` = `PortMode
   .self_overlap`, the *unconjugated* self-overlap Section 4.3's `project`
   already divides by, ~2x the conjugated `P_m=1` power for a lossless
   mode -- see `ports.mode_solver._self_overlap`'s own docstring) that
   `project` (S-parameter *extraction*) already applied but `build_B`
   (which routes that same *solved*-field quantity back into the system
   as the port boundary condition, i.e. the injection side of the exact
   same normalization) did not -- an injection/extraction mismatch, not a
   PML- or discretization-driven effect. Before this fix, this exact test
   file's passivity gate (below) failed with `|S11|^2+|S21|^2` around 0.3,
   nowhere near a PML-reflection-explainable deficit for a lossless
   structure; see `build_B`'s own docstring for the full derivation.

The geometry here is deliberately the Part-3-validated single-mode 50-ohm
microstrip design (`examples/isotropic_microstrip.py`'s own parameters,
minus the trivial/non-absorbing PML swapped in for CI speed/reliability --
a real graded PML at this substrate thickness needs a much finer mesh than
this fast unit test affords, a separate mesh-resolution concern unrelated
to anything this file is testing). Its box-mode-safe port aperture (per
`ports.sizing`) is what makes both ports reliably land on the true
quasi-TEM dominant mode rather than Module 4 Section 3.7's documented
box-mode limitation, which would otherwise swamp the signal this test is
actually after -- and, being lossless with a trivial (non-absorbing but
still lossless) PML, energy conservation must hold exactly regardless of
how much of the incident wave the PML reflects back through the ports:
a lossless PEC/PMC/trivial-PML/real-eps structure conserves energy
whether or not the boundary happens to be a good absorber.
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
_SWEEP_FREQUENCIES_HZ = [8e9, 9e9, 10e9, 11e9, 12e9]  # several frequencies, per the doc's own passivity gate


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
    frequencies = [2 * np.pi * f for f in _SWEEP_FREQUENCIES_HZ]
    return run_sweep(mesh, materials, ["PORT_1", "PORT_2"], frequencies, n_modes=1, pml_params=pml_params)


@pytest.fixture(scope="module")
def S_by_freq(sweep_results):
    """One deembedded S-dict per swept frequency, in `_SWEEP_FREQUENCIES_HZ`
    order -- mirrors `cli.main`'s own per-frequency `raw_s_parameters`/
    `deembed` loop."""
    from extract import deembed, raw_s_parameters

    by_omega: dict[float, list] = {}
    for r in sweep_results:
        by_omega.setdefault(r.omega, []).append(r)

    out = []
    for omega in sorted(by_omega):
        freq_results = by_omega[omega]
        raw_S = raw_s_parameters(freq_results, ["PORT_1", "PORT_2"], 1)
        out.append(deembed(raw_S, freq_results[0].port_modes, {"PORT_1": 0.0, "PORT_2": 0.0}))
    return out


@pytest.fixture(scope="module")
def S(S_by_freq):
    """The single-frequency tests below (axial sign, mode congruence,
    reciprocity) only need one representative point -- the middle of the
    sweep."""
    return S_by_freq[len(S_by_freq) // 2]


def test_axial_sign_is_opposite_at_the_two_ports(sweep_results):
    """The root-cause fact the axial-orientation review is checking around:
    PORT_1's outward normal is -x_hat (interior at x>0), PORT_2's is
    +x_hat (interior at x<L) -- `axial_sign` must disagree between them,
    not be hardcoded to the same value."""
    port_modes = sweep_results[0].port_modes
    s1 = port_modes["PORT_1"][0].cross_section.axial_sign
    s2 = port_modes["PORT_2"][0].cross_section.axial_sign
    assert s1 == pytest.approx(1.0)
    assert s2 == pytest.approx(-1.0)


def test_both_ports_found_the_dominant_propagating_mode(sweep_results):
    """Section 6.1's starting-frequency precondition already checks this
    internally (would have raised `SweepPreconditionError` otherwise), but
    it's worth asserting directly here too: this test's whole point is a
    clean axial-sign/passivity signal, which a port that landed on a
    spurious evanescent/box mode (Module 4 Section 3.7) instead of the
    true quasi-TEM mode would otherwise swamp with unrelated noise.
    `_HAMMERSTAD_JENSEN_BETA` is the 10 GHz value -- `sweep_results[0]` is
    now the sweep's first (8 GHz) point, so scale the target by frequency
    (beta is linear in omega for a fixed eps_eff) rather than comparing
    across frequencies directly."""
    first_step_port_modes = sweep_results[0].port_modes
    expected_beta = _HAMMERSTAD_JENSEN_BETA * (_SWEEP_FREQUENCIES_HZ[0] / 10e9)
    for tag in ("PORT_1", "PORT_2"):
        gamma = first_step_port_modes[tag][0].gamma
        assert abs(gamma.real) < 1e-3 * abs(gamma.imag)
        assert gamma.imag == pytest.approx(expected_beta, rel=0.15)


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
    also agree (not just the dominant reciprocity pairing above). Both are
    individually small on this well-matched line (~0.1), so the tolerance
    is absolute, not relative -- a relative comparison is noisy this close
    to zero."""
    s11 = S[("PORT_1", 1, "PORT_1")]
    s22 = S[("PORT_2", 1, "PORT_2")]
    assert s11 == pytest.approx(s22, abs=2e-2)


# --- Passivity / energy conservation -- the gate the doc has always
# specified alongside reciprocity, previously missing from this suite. ---


def test_passivity_energy_conservation_at_several_frequencies(S_by_freq):
    """`|S11|^2+|S21|^2 ~= 1` and `|S22|^2+|S12|^2 ~= 1` at several
    frequencies, to discretization error -- not merely `<= 1`. A lossless
    PEC/PMC/trivial-PML/real-eps structure conserves energy exactly
    regardless of how much of the incident wave a non-absorbing boundary
    reflects back out through the ports; a deficit here signals a real
    normalization bug (as the injection/extraction `1/N_m` mismatch this
    test file's own module docstring describes was), not something a PML
    choice can explain away."""
    for S in S_by_freq:
        s11, s21 = S[("PORT_1", 1, "PORT_1")], S[("PORT_2", 1, "PORT_1")]
        s22, s12 = S[("PORT_2", 1, "PORT_2")], S[("PORT_1", 1, "PORT_2")]
        assert abs(s11) ** 2 + abs(s21) ** 2 == pytest.approx(1.0, abs=0.05)
        assert abs(s22) ** 2 + abs(s12) ** 2 == pytest.approx(1.0, abs=0.05)


def test_passivity_upper_bound_at_several_frequencies(S_by_freq):
    """`|S_ij| <= 1` at every frequency swept -- energy cannot be
    manufactured, a hard physical bound independent of the (here: trivial,
    non-absorbing) PML's absorption quality."""
    for S in S_by_freq:
        for value in S.values():
            assert abs(value) <= 1.0 + 1e-2
