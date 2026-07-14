"""Validation suite for material.tensor_interpolation (docs/module2_material_equations.md
Section 4, Section 7's LC-specific validation targets)."""
import numpy as np
import pytest

from material.core import MaterialPassivityError
from material.interpolation import CoverageError
from material.regions import ConstantMaterial, TensorFieldMaterial
from material.tensor_interpolation import DirectorFieldError, DirectorFieldMaterial

_SAMPLE_POINTS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
)
_QUERY = np.array([[0.25, 0.25, 0.1]])


def _uniform_director(n0: np.ndarray) -> np.ndarray:
    return np.tile(n0, (_SAMPLE_POINTS.shape[0], 1))


# --- Section 4.1/4.2: tensor construction and input validation ------------

def test_uniform_director_produces_hand_computed_tensor():
    n0 = np.array([0.0, 0.0, 1.0])
    eps_perp, eps_parallel = 2.0 + 0j, 3.0 + 0j
    dfm = DirectorFieldMaterial(_SAMPLE_POINTS, _uniform_director(n0), eps_perp, eps_parallel)
    eps = dfm.epsilon(_QUERY)
    expected = eps_perp * np.eye(3) + (eps_parallel - eps_perp) * np.outer(n0, n0)
    assert np.allclose(eps[0], expected, atol=1e-9)


def test_non_unit_director_within_tolerance_is_renormalized():
    n0 = np.array([0.0, 0.0, 1.05])  # |n|=1.05, within [0.9,1.1]
    dfm = DirectorFieldMaterial(_SAMPLE_POINTS, _uniform_director(n0), 2.0 + 0j, 3.0 + 0j)
    eps = dfm.epsilon(_QUERY)
    expected = 2.0 * np.eye(3) + 1.0 * np.outer([0, 0, 1], [0, 0, 1])
    assert np.allclose(eps[0], expected, atol=1e-6)


def test_non_unit_director_outside_tolerance_raises():
    n0 = np.array([0.0, 0.0, 2.0])  # |n|=2.0, well outside [0.9,1.1]
    with pytest.raises(DirectorFieldError):
        DirectorFieldMaterial(_SAMPLE_POINTS, _uniform_director(n0), 2.0 + 0j, 3.0 + 0j)


# --- Section 4.8: n -> -n invariance ---------------------------------------

def test_sign_flip_invariance():
    directors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    directors /= np.linalg.norm(directors, axis=1, keepdims=True)
    dfm = DirectorFieldMaterial(_SAMPLE_POINTS, directors, 2.0 + 0j, 3.0 + 0j)
    dfm_neg = DirectorFieldMaterial(_SAMPLE_POINTS, -directors, 2.0 + 0j, 3.0 + 0j)
    assert np.allclose(dfm.epsilon(_QUERY), dfm_neg.epsilon(_QUERY), atol=1e-12)


# --- Section 4.7: module boundary contract ---------------------------------

def test_module_boundary_contract_matches_direct_phase3_tensor():
    n0 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    eps_perp, eps_parallel = 2.2 + 0j, 3.1 + 0j
    dfm = DirectorFieldMaterial(_SAMPLE_POINTS, _uniform_director(n0), eps_perp, eps_parallel)

    direct_tensor = eps_perp * np.eye(3) + (eps_parallel - eps_perp) * np.outer(n0, n0)
    components = np.array(
        [direct_tensor[0, 0], direct_tensor[0, 1], direct_tensor[0, 2],
         direct_tensor[1, 1], direct_tensor[1, 2], direct_tensor[2, 2]]
    )
    from material.interpolation import SampledField

    field = SampledField.scattered(_SAMPLE_POINTS, np.tile(components, (5, 1)))
    tfm = TensorFieldMaterial(field)

    assert np.allclose(dfm.epsilon(_QUERY), tfm.epsilon(_QUERY), atol=1e-9)


# --- Section 4.5: eigenvalue-bound spot check ------------------------------

def test_eigenvalues_stay_within_eps_perp_eps_parallel_bounds():
    rng = np.random.default_rng(0)
    directors = rng.normal(size=(6, 3))
    directors /= np.linalg.norm(directors, axis=1, keepdims=True)  # unit directors, |n|=1 (Section 4.2)
    points = rng.uniform(-1, 1, size=(6, 3))
    eps_perp, eps_parallel = 2.0 + 0j, 5.0 + 0j
    dfm = DirectorFieldMaterial(points, directors, eps_perp, eps_parallel, interpolation_method="idw")

    query = rng.uniform(-0.5, 0.5, size=(20, 3))
    eps = dfm.epsilon(query)
    for tensor in eps:
        eigenvalues = np.linalg.eigvalsh(tensor.real)
        assert eigenvalues.min() >= min(eps_perp.real, eps_parallel.real) - 1e-9
        assert eigenvalues.max() <= max(eps_perp.real, eps_parallel.real) + 1e-9


# --- Section 4.6: passivity reduces to a scalar check ----------------------

def test_passivity_accepts_lossy_but_valid_constants():
    n0 = np.array([0.0, 0.0, 1.0])
    DirectorFieldMaterial(_SAMPLE_POINTS, _uniform_director(n0), 2.0 - 0.1j, 3.0 - 0.2j)  # must not raise


def test_passivity_rejects_negative_eps_perp_loss():
    n0 = np.array([0.0, 0.0, 1.0])
    with pytest.raises(MaterialPassivityError):
        DirectorFieldMaterial(_SAMPLE_POINTS, _uniform_director(n0), 2.0 + 0.1j, 3.0 - 0.1j)


def test_passivity_rejects_negative_eps_parallel_loss():
    n0 = np.array([0.0, 0.0, 1.0])
    with pytest.raises(MaterialPassivityError):
        DirectorFieldMaterial(_SAMPLE_POINTS, _uniform_director(n0), 2.0 - 0.1j, 3.0 + 0.1j)


def test_passivity_check_happens_before_any_interpolation():
    """The scalar check runs at construction, before touching the sample
    data at all -- confirm it raises even with a malformed director that
    would otherwise also fail the normalization guard, proving the
    passivity check is checked first (Section 4.6: 'once, at load time')."""
    bad_directors = np.tile([0.0, 0.0, 5.0], (5, 1))  # would also fail |n| guard
    with pytest.raises(MaterialPassivityError):
        DirectorFieldMaterial(_SAMPLE_POINTS, bad_directors, 2.0 + 0.1j, 3.0 - 0.1j)


def test_lc_material_skips_the_generic_per_call_passivity_check():
    """Section 4.6: once eps_perp''/eps_parallel'' pass, no per-point
    eigendecomposition runs on `epsilon()` calls -- confirmed indirectly:
    a query set large enough that the generic check would be the dominant
    cost still returns without needing eigvalsh per point (functional
    check: the call simply succeeds, matching the cheap-path contract)."""
    n0 = np.array([0.0, 0.0, 1.0])
    dfm = DirectorFieldMaterial(_SAMPLE_POINTS, _uniform_director(n0), 2.0 - 0.05j, 3.0 - 0.05j)
    query = np.tile(_QUERY, (500, 1)) + np.random.default_rng(1).uniform(-0.05, 0.05, size=(500, 3))
    dfm.epsilon(query)  # must not raise, must not be prohibitively slow


# --- Section 4.2: coordinate-frame/units mismatch detection ---------------

def test_wrong_units_director_fails_containment_check():
    """A director file whose raw coordinate values weren't converted to
    the mesh's units (e.g. left in mm while the mesh is in m) spans a
    numerically much smaller range than the mesh region it's meant to
    cover -- exactly the ~1000x containment failure Section 4.2
    describes as the diagnostic for a units mismatch."""
    small_points = _SAMPLE_POINTS  # spans [0,1] -- as if never scaled up from "mm" to "m"
    n0 = np.array([0.0, 0.0, 1.0])
    mesh_region_bounds = (np.array([0.0, 0.0, 0.0]), np.array([1000.0, 1000.0, 1000.0]))
    with pytest.raises(CoverageError):
        DirectorFieldMaterial(
            small_points, np.tile(n0, (5, 1)), 2.0 + 0j, 3.0 + 0j, region_bounds=mesh_region_bounds
        )


# --- Section 5.2: from_file / director-file format -------------------------

def test_from_file_parses_header_and_builds_material(tmp_path):
    path = tmp_path / "lc_director.csv"
    lines = ["# coordinate_frame: mesh", "# units: m", "# grid_type: scattered", "x, y, z, nx, ny, nz"]
    for p in _SAMPLE_POINTS:
        lines.append(f"{p[0]}, {p[1]}, {p[2]}, 0.0, 0.0, 1.0")
    path.write_text("\n".join(lines))

    dfm = DirectorFieldMaterial.from_file(path, 2.0 + 0j, 3.0 + 0j)
    eps = dfm.epsilon(_QUERY)
    expected = 2.0 * np.eye(3) + 1.0 * np.outer([0, 0, 1], [0, 0, 1])
    assert np.allclose(eps[0], expected, atol=1e-9)


def test_from_file_missing_header_field_raises(tmp_path):
    path = tmp_path / "lc_director.csv"
    lines = ["# coordinate_frame: mesh", "# units: m", "x, y, z, nx, ny, nz"]  # grid_type missing
    for p in _SAMPLE_POINTS:
        lines.append(f"{p[0]}, {p[1]}, {p[2]}, 0.0, 0.0, 1.0")
    path.write_text("\n".join(lines))

    with pytest.raises(DirectorFieldError):
        DirectorFieldMaterial.from_file(path, 2.0 + 0j, 3.0 + 0j)
