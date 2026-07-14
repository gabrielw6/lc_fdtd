"""Validation suite for pml.PMLMaterial (docs/module5_pml_equations.md
Sections 3-4).
"""
import numpy as np
import pytest

from material import ConstantMaterial
from material.core import MaterialPassivityError, check_passive_generic
from pml import PMLMaterial

_OMEGA = 2 * np.pi * 25e9
_Z_AIR_TOP = 0.008
_THICKNESS = 0.002


def _pml(R0=1e-6, n=2, kappa_max=3.0):
    return PMLMaterial(ConstantMaterial(eps_r=1.0), _OMEGA, _Z_AIR_TOP, _THICKNESS, R0=R0, n=n, kappa_max=kappa_max)


def _points(xis):
    return np.array([[0.0, 0.0, _Z_AIR_TOP + xi] for xi in xis])


# --- Section 3.1: eps_r == mu_r for this geometry ---


def test_epsilon_equals_mu_for_air_background():
    pml = _pml()
    pts = _points([0.0, 0.0005, 0.0015, 0.002])
    assert pml.epsilon(pts) == pytest.approx(pml.mu(pts))


def test_epsilon_reduces_to_identity_at_xi_zero():
    """xi=0 => sigma=0, kappa=1 => Lambda=I."""
    pml = _pml()
    eps = pml.epsilon(_points([0.0]))
    assert eps[0] == pytest.approx(np.eye(3))


def test_epsilon_is_diagonal_at_every_depth():
    pml = _pml()
    eps = pml.epsilon(_points([0.0, 0.001, 0.002]))
    for e in eps:
        off_diag = e - np.diag(np.diag(e))
        assert np.allclose(off_diag, 0)


# --- Section 3.3: diagonal-inverse exactness ---


def test_lambda_inverse_is_exact():
    pml = _pml()
    pts = _points([0.0003, 0.001, 0.0019])
    eps = pml.epsilon(pts)  # == Lambda here (AIR background)
    for e in eps:
        assert e @ np.linalg.inv(e) == pytest.approx(np.eye(3), abs=1e-10)


# --- Section 4.3: the passivity-check exemption is actually wired in ---


def test_generic_passivity_check_would_reject_a_deep_pml_point():
    """Confirms the underlying tensor really *would* fail Module 2's
    generic check (Section 4.1's finding) -- otherwise the exemption test
    below would be vacuous."""
    pml = _pml()
    eps_deep = pml.epsilon(_points([0.0019]))[0]  # near the PEC backing: large Im(1/s_z)
    with pytest.raises(MaterialPassivityError):
        check_passive_generic(eps_deep[None, :, :])


def test_epsilon_call_does_not_raise_despite_failing_generic_check():
    """The actual exemption (Section 4.3): calling .epsilon()/.mu() through
    PMLMaterial's own template-method hooks must not raise, even though
    the raw tensor fails the generic check (previous test)."""
    pml = _pml()
    pts = _points([0.0, 0.0005, 0.001, 0.0015, 0.002])
    pml.epsilon(pts)  # must not raise
    pml.mu(pts)  # must not raise


def test_zz_component_imaginary_part_is_nonnegative_by_construction():
    """Section 4.1's derivation, confirmed on the actual PMLMaterial
    output (not just the standalone stretching functions)."""
    pml = _pml()
    eps = pml.epsilon(_points([0.0005, 0.001, 0.0015, 0.002]))
    assert np.all(eps[:, 2, 2].imag >= -1e-15)


# --- Section 8: trivial-PML reduction, checked at the fem.assembly level ---


def test_trivial_pml_reproduces_air_element_matrices_on_real_mesh():
    """Section 8: 'a full solve with the PML region given this trivial
    material must reproduce the same-mesh solve with no PML flagged at
    all' -- checked here at the element-matrix level (Modules 6/7 don't
    exist yet to run the full solve), on the PML_TOP tets of a real
    geometry_builder mesh, via fem.assembly.element_matrices."""
    pytest.importorskip("gmsh")
    from fem import element_matrices
    from geometry_builder import GeometryBuilder, GeometryParams
    from material import MaterialAssembly
    from mesh_interface import MeshInterface

    params = GeometryParams(
        w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
        eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
        reference_frequency=25e9, target_elements_per_wavelength=6,
    )
    mesh_handle, _material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)

    pml_tets = [t for t in range(mesh.n_tets) if mesh.tet_volume_tag(t) == "PML_TOP"]
    assert len(pml_tets) > 0
    tet = pml_tets[0]

    # R0=1 => ln(R0)=0 => sigma_max=0 => s_z=kappa=1 everywhere (kappa_max
    # irrelevant when sigma_max=0, since kappa(xi) still grades from 1 to
    # kappa_max -- force kappa_max=1 too for a truly trivial Lambda=I PML).
    trivial_pml = PMLMaterial(
        ConstantMaterial(eps_r=1.0), 2 * np.pi * params.reference_frequency,
        z_air_top=params.h_sub + params.h_air, thickness=params.h_pml,
        R0=1.0, n=2, kappa_max=1.0,
    )
    air_material = ConstantMaterial(eps_r=1.0)

    K_pml, M_pml = element_matrices(mesh, tet, MaterialAssembly({"PML_TOP": trivial_pml}))
    K_air, M_air = element_matrices(mesh, tet, MaterialAssembly({"PML_TOP": air_material}))

    assert K_pml == pytest.approx(K_air, abs=1e-9)
    assert M_pml == pytest.approx(M_air, abs=1e-9)
