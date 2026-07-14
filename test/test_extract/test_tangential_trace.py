"""docs/module7_extract_sparameters_equations.md Section 1's geometric
fact, confirmed numerically (build step 1): only a boundary face's own
three edges have nonzero tangential trace there -- an edge connecting a
face vertex to the opposite (non-face) vertex has a purely-normal Whitney
function on that face, hence zero tangential component. This is the fact
`project_amplitude`'s edge-restricted sum relies on being exact.
"""
import numpy as np

from fem.edge_elements import whitney_basis
from mesh_interface import MeshInterface

# Single reference tet: vertices at the origin and the three unit axis
# points. Face opposite vertex 0 (LOCAL_FACES[0] = (1,2,3)) is tagged as a
# port face; its plane is x+y+z=1, normal (1,1,1)/sqrt(3).
_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_TETS = np.array([[0, 1, 2, 3]])
_SURFACE_TAGS = {
    "PORT_1": np.array([[1, 2, 3]]),
    "PEC_GROUND": np.array([[0, 2, 3], [0, 1, 3], [0, 1, 2]]),
}

_FACE_LOCAL_EDGES = {3, 4, 5}  # LOCAL_EDGES indices for (1,2), (1,3), (2,3)
_NON_FACE_LOCAL_EDGES = {0, 1, 2}  # (0,1), (0,2), (0,3) -- touch the opposite vertex


def _sample_face_barycentric(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    face_bary = rng.dirichlet([1.0, 1.0, 1.0], size=n)  # (n,3), each row sums to 1
    bary = np.zeros((n, 4))
    bary[:, 0] = 0.0
    bary[:, 1:] = face_bary
    return bary


def test_non_face_edges_have_zero_tangential_trace_on_the_boundary_face():
    mesh = MeshInterface(_VERTICES, _TETS, np.array(["A"]), _SURFACE_TAGS)
    grad = mesh.grad_lambda(0)
    sign = mesh.tet_edge_sign[0]
    basis = whitney_basis(grad, sign)

    bary = _sample_face_barycentric(30)
    W = basis(bary)  # (6, 30, 3)
    n_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)

    for local_edge in _NON_FACE_LOCAL_EDGES:
        v = W[local_edge]
        normal_component = (v @ n_hat)[:, None] * n_hat[None, :]
        tangential = v - normal_component
        assert np.abs(tangential).max() < 1e-10, f"local edge {local_edge}"


def test_face_edges_have_nonzero_tangential_trace_on_the_boundary_face():
    """Confirms the test setup isn't vacuous -- the face's own edges
    really do have a nonzero tangential component there."""
    mesh = MeshInterface(_VERTICES, _TETS, np.array(["A"]), _SURFACE_TAGS)
    grad = mesh.grad_lambda(0)
    sign = mesh.tet_edge_sign[0]
    basis = whitney_basis(grad, sign)

    bary = _sample_face_barycentric(30)
    W = basis(bary)
    n_hat = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)

    for local_edge in _FACE_LOCAL_EDGES:
        v = W[local_edge]
        normal_component = (v @ n_hat)[:, None] * n_hat[None, :]
        tangential = v - normal_component
        assert np.abs(tangential).max() > 1e-6, f"local edge {local_edge}"
