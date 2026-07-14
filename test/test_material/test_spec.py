"""Validation suite for material.spec (docs/module2_material_equations.md
Section 5)."""
import numpy as np
import pytest

from material.core import MaterialTagError
from material.spec import MaterialSpecError, load_material_spec

_GEOMETRY_STUB = {
    "AIR": {"type": "constant", "eps_r": 1.0},
    "SUBSTRATE": {"type": "constant", "eps_r": 3.5, "tan_delta": 0.001},
}


class _StubObject:
    """Duck-typed stand-in for `geometry_builder.MaterialSpecStub`."""

    def __init__(self, entries: dict):
        self.entries = entries


def test_constant_type_from_geometry_stub_alone(tmp_path):
    assembly = load_material_spec(geometry_stub=_GEOMETRY_STUB)
    eps_air = assembly.epsilon("AIR", np.zeros((1, 3)))
    assert np.allclose(eps_air, np.eye(3))
    eps_sub = assembly.epsilon("SUBSTRATE", np.zeros((1, 3)))
    assert eps_sub[0, 0, 0] == pytest.approx(3.5 * (1 - 1j * 0.001))


def test_geometry_stub_accepts_duck_typed_object_with_entries_attribute():
    assembly = load_material_spec(geometry_stub=_StubObject(_GEOMETRY_STUB))
    eps_air = assembly.epsilon("AIR", np.zeros((1, 3)))
    assert np.allclose(eps_air, np.eye(3))


def test_user_yaml_overrides_geometry_stub_entry(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("materials:\n  SUBSTRATE:\n    type: constant\n    eps_r: 9.0\n")
    assembly = load_material_spec(spec_path, geometry_stub=_GEOMETRY_STUB)
    eps_sub = assembly.epsilon("SUBSTRATE", np.zeros((1, 3)))
    assert eps_sub[0, 0, 0] == pytest.approx(9.0)  # user value wins, not the stub's 3.5


def test_scalar_field_type_from_yaml_with_relative_file(tmp_path):
    points = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
    (tmp_path / "profile.csv").write_text(
        "x, y, z, value\n" + "\n".join(f"{x}, {y}, {z}, 2.0" for x, y, z in points)
    )
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("materials:\n  SUBSTRATE:\n    type: scalar_field\n    file: profile.csv\n")
    assembly = load_material_spec(spec_path)
    eps = assembly.epsilon("SUBSTRATE", np.array([[0.2, 0.2, 0.2]]))
    assert eps[0, 0, 0] == pytest.approx(2.0)


def test_tensor_field_type_from_eps_r_components(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "materials:\n"
        "  SUBSTRATE:\n"
        "    type: tensor_field\n"
        "    eps_r_components: {xx: 2.5, xy: 0.0, xz: 0.0, yy: 2.5, yz: 0.0, zz: 3.0}\n"
    )
    assembly = load_material_spec(spec_path)
    eps = assembly.epsilon("SUBSTRATE", np.array([[123.0, -45.0, 6.0]]))  # far outside any real mesh
    expected = np.diag([2.5, 2.5, 3.0])
    assert np.allclose(eps[0], expected)


def test_director_field_type_from_yaml(tmp_path):
    director_path = tmp_path / "lc.csv"
    pts = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    lines = ["# coordinate_frame: mesh", "# units: m", "# grid_type: scattered", "x,y,z,nx,ny,nz"]
    lines += [f"{x},{y},{z},0,0,1" for x, y, z in pts]
    director_path.write_text("\n".join(lines))

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "materials:\n"
        "  LC:\n"
        "    type: director_field\n"
        "    file: lc.csv\n"
        "    eps_perp: 2.0\n"
        "    eps_parallel: 3.0\n"
        "    eps_perp_im: 0.05\n"
    )
    assembly = load_material_spec(spec_path)
    eps = assembly.epsilon("LC", np.array([[0.25, 0.25, 0.1]]))
    expected = 2.0 * np.eye(3) + 1.0 * np.outer([0, 0, 1], [0, 0, 1])
    assert np.allclose(eps[0].real, expected, atol=1e-9)
    assert eps[0, 0, 0].imag == pytest.approx(-0.05, abs=1e-9)  # eps_perp_im -> -j*0.05


def test_unknown_type_raises_material_spec_error(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("materials:\n  X:\n    type: not_a_real_type\n")
    with pytest.raises(MaterialSpecError):
        load_material_spec(spec_path)


def test_missing_required_field_raises_material_spec_error(tmp_path):
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text("materials:\n  X:\n    type: constant\n")  # no eps_r
    with pytest.raises(MaterialSpecError):
        load_material_spec(spec_path)


def test_no_materials_at_all_raises():
    with pytest.raises(MaterialSpecError):
        load_material_spec()


def test_pml_top_never_needs_a_spec_entry():
    """Section 5.1: PML_TOP never appears in a spec file -- its material
    is always derived (Module 5), so querying it here is simply an
    unregistered tag, same as any other typo."""
    assembly = load_material_spec(geometry_stub=_GEOMETRY_STUB)
    with pytest.raises(MaterialTagError):
        assembly.epsilon("PML_TOP", np.zeros((1, 3)))
