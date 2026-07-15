"""Validation suite for fem.field_eval -- reconstructs the driven E-field
(and curl(E)) at arbitrary physical points from a full-length edge-DOF
vector, reusing `fem.edge_elements.whitney_basis`/`whitney_curl` directly
(the same basis + global-edge sign convention `fem.assembly.assemble`
uses to build K/M -- see field_eval.py's own module docstring for why
that reuse, not a re-derivation, is what this module depends on for
correctness).
"""
import numpy as np
import pytest

from fem.edge_elements import whitney_basis
from fem.field_eval import evaluate_curl_field, evaluate_edge_field
from mesh_interface import MeshInterface
from mesh_interface.interface import LOCAL_EDGES

# A single reference tet (docs/module3 Section 7's own reference case,
# reused from test_edge_elements.py's _REF_VERTICES/_REF_GRAD): vertices
# at the origin and the three unit axis points.
_SINGLE_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_SINGLE_TETS = np.array([[0, 1, 2, 3]])
_SINGLE_SURFACE_TAGS = {"PEC_GROUND": np.array([[0, 2, 3], [0, 1, 3], [0, 1, 2], [1, 2, 3]])}


def _single_tet_mesh() -> MeshInterface:
    return MeshInterface(_SINGLE_VERTICES, _SINGLE_TETS, np.array(["A"]), _SINGLE_SURFACE_TAGS)


# --- Whitney DOF property, through evaluate_edge_field (not raw whitney_basis) ---


def test_dof_property_line_integral_is_one_on_its_own_edge_and_zero_on_others():
    """The exact test pattern edge_elements.py's own
    test_tangential_trace_normalization_on_reference_tet uses (1D
    Gauss-Legendre along each edge's own path), but exercised through the
    full evaluate_edge_field reconstruction (point location + DOF lookup
    + basis evaluation), not the raw whitney_basis callable -- this is
    what actually proves evaluate_edge_field's point-location and
    edge-map bookkeeping agree with the basis's own convention, not just
    the basis formula in isolation."""
    mesh = _single_tet_mesh()
    t_nodes, t_weights = np.polynomial.legendre.leggauss(5)
    t = 0.5 * (t_nodes + 1.0)
    w = 0.5 * t_weights

    edge_map = mesh.tet_edge_map[0]  # local edge index -> global edge id

    for driven_local_edge in range(6):
        a = np.zeros(mesh.n_edges, dtype=complex)
        a[edge_map[driven_local_edge]] = 1.0 + 0j

        for local_edge, (la, lb) in enumerate(LOCAL_EDGES):
            p_a, p_b = _SINGLE_VERTICES[la], _SINGLE_VERTICES[lb]
            points = p_a[None, :] * (1 - t)[:, None] + p_b[None, :] * t[:, None]
            field = evaluate_edge_field(mesh, a, points)
            assert np.all(np.isfinite(field)), "all sample points lie inside the single tet"

            edge_vector = p_b - p_a
            integral = complex(np.sum(w * (field @ edge_vector)))
            expected = 1.0 if local_edge == driven_local_edge else 0.0
            assert integral == pytest.approx(expected, abs=1e-9), (
                f"driven edge {driven_local_edge}, sampled edge {local_edge}"
            )


def test_reconstruction_matches_whitney_basis_directly():
    """A second, independent check: evaluate_edge_field at the tet's own
    centroid must equal the direct whitney_basis evaluation dotted with
    the DOF vector -- catches a sign/index mismatch that a DOF-property
    test alone (checking only 0/1 outcomes) could miss."""
    mesh = _single_tet_mesh()
    rng = np.random.default_rng(0)
    a = np.zeros(mesh.n_edges, dtype=complex)
    a[mesh.tet_edge_map[0]] = rng.normal(size=6) + 1j * rng.normal(size=6)

    centroid = _SINGLE_VERTICES.mean(axis=0, keepdims=True)
    got = evaluate_edge_field(mesh, a, centroid)[0]

    bary = np.array([[0.25, 0.25, 0.25, 0.25]])
    grad, sign = mesh.grad_lambda(0), mesh.tet_edge_sign[0]
    W = whitney_basis(grad, sign)(bary)[:, 0, :]  # (6,3)
    expected = a[mesh.tet_edge_map[0]] @ W

    assert got == pytest.approx(expected, abs=1e-12)


# --- outside-domain masking ---


def test_points_outside_the_mesh_are_nan():
    mesh = _single_tet_mesh()
    a = np.ones(mesh.n_edges, dtype=complex)
    far_away = np.array([[10.0, 10.0, 10.0], [-5.0, -5.0, -5.0]])
    field = evaluate_edge_field(mesh, a, far_away)
    assert field.shape == (2, 3)
    assert np.all(np.isnan(field))


def test_mixed_inside_and_outside_points():
    mesh = _single_tet_mesh()
    a = np.ones(mesh.n_edges, dtype=complex)
    points = np.array([[0.2, 0.2, 0.2], [10.0, 10.0, 10.0]])  # first inside, second far outside
    field = evaluate_edge_field(mesh, a, points)
    assert np.all(np.isfinite(field[0]))
    assert np.all(np.isnan(field[1]))


def test_evaluate_curl_field_outside_mesh_is_nan_and_constant_inside():
    mesh = _single_tet_mesh()
    a = np.ones(mesh.n_edges, dtype=complex)
    points = np.array([[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [10.0, 10.0, 10.0]])
    curl = evaluate_curl_field(mesh, a, points)
    assert np.all(np.isnan(curl[2]))
    # curl is constant per tet -- the two interior points must agree exactly.
    assert curl[0] == pytest.approx(curl[1], abs=1e-12)
    assert np.all(np.isfinite(curl[0]))


# --- point-location smoke test on a real (gmsh-built) mesh ---


def test_point_location_smoke_test_on_real_mesh():
    pytest.importorskip("gmsh")
    from geometry_builder import GeometryBuilder, GeometryParams

    params = GeometryParams(
        w=0.002, L=0.020, L_lc=0.008, W_lc=0.004, h_sub=0.002, W_sub=0.010,
        eps_r_substrate=3.0, h_air=0.006, h_pml=0.002,
        reference_frequency=25e9, target_elements_per_wavelength=6,
    )
    mesh_handle, _material_stub = GeometryBuilder().build(params)
    mesh = MeshInterface.from_mesh_handle(mesh_handle)

    rng = np.random.default_rng(1)
    a = rng.normal(size=mesh.n_edges) + 1j * rng.normal(size=mesh.n_edges)

    # A point deep inside the substrate (must be found, finite field) and
    # one far outside the whole domain (must be nan).
    interior = np.array([[0.010, 0.005, 0.001]])
    outside = np.array([[-1.0, -1.0, -1.0]])
    points = np.concatenate([interior, outside], axis=0)

    field = evaluate_edge_field(mesh, a, points)
    assert np.all(np.isfinite(field[0]))
    assert np.all(np.isnan(field[1]))
