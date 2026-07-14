"""Validation suite for solve.system (docs/module6_solve_sweep_equations.md
Sections 4-5). Hand-built small systems -- no mesh/gmsh needed, per build
step 2's own instruction ("unit test on a small hand-built system").
"""
import numpy as np
import pytest
import scipy.sparse as sp

from solve.system import (
    SolveSingularityError,
    SystemSymmetryError,
    build_restriction,
    factor,
    recover_solution,
    reduce_system,
    solve_with_factorization,
)

# A hand-built 4x4 complex-symmetric system with edge 1 constrained (PEC).
# Symmetric by construction (A = A^T); invertible (checked below).
_A = np.array(
    [
        [4.0 + 1j, 1.0 - 2j, 0.5j, 0.0],
        [1.0 - 2j, 5.0 - 1j, 0.0, 2.0 + 1j],
        [0.5j, 0.0, 3.0 + 0.5j, 1.0j],
        [0.0, 2.0 + 1j, 1.0j, 6.0 - 2j],
    ],
    dtype=complex,
)
_B = np.array([1.0 + 0j, 0.0, 2.0 - 1j, 0.5j])  # b[1]=0 -- PEC-adjacent entry structurally zero
_PEC_DOFS = {1}


def test_a_matrix_is_genuinely_symmetric_and_invertible():
    assert _A == pytest.approx(_A.T)
    assert abs(np.linalg.det(_A)) > 1e-6


# --- Section 4.1: restriction matrix ---


def test_build_restriction_shape_and_structure():
    R = build_restriction(_PEC_DOFS, n_edges=4)
    assert R.shape == (3, 4)
    dense = R.toarray()
    # Row i selects free DOF i, in ascending order: free = [0, 2, 3].
    assert np.array_equal(dense, np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))


# --- Section 4.2/4.3: reduced solve reproduces the full-system solve on free DOFs ---


def test_reduced_solve_matches_full_system_restricted_to_free_dofs():
    R = build_restriction(_PEC_DOFS, n_edges=4)
    A_sp = sp.csr_matrix(_A)

    A_ff, b_f = reduce_system(A_sp, _B, R)
    fact = factor(A_ff)
    a_f = solve_with_factorization(fact, b_f)
    a = recover_solution(a_f, R)

    # a_c = 0 exactly at the constrained DOF (Section 4.3).
    assert a[1] == 0.0

    # Direct dense solve of the free-DOF-only 3x3 subsystem must match.
    free = [0, 2, 3]
    a_direct = np.linalg.solve(_A[np.ix_(free, free)], _B[free])
    assert a[free] == pytest.approx(a_direct, abs=1e-10)

    # And the full system's free rows are satisfied by the recovered a.
    residual = _A[free, :] @ a - _B[free]
    assert np.abs(residual).max() < 1e-9


def test_reduce_system_produces_symmetric_reduced_matrix():
    R = build_restriction(_PEC_DOFS, n_edges=4)
    A_ff, _b_f = reduce_system(sp.csr_matrix(_A), _B, R)
    dense = A_ff.toarray()
    assert dense == pytest.approx(dense.T)


# --- Section 5: factorization ---


def test_factor_rejects_non_symmetric_matrix():
    # Relative asymmetry >> _SYMMETRY_TOL (0.3, deliberately loose --
    # see factor()'s docstring for why): off-diagonal entries 10 and 0
    # against a scale of 10 give a residual/scale ratio of 1.0, an order
    # of magnitude past the tolerance, representative of what a genuinely
    # transposed tensor index looks like (not a borderline percentage).
    non_symmetric = sp.csr_matrix(np.array([[1.0, 10.0], [0.0, 1.0]], dtype=complex))
    with pytest.raises(SystemSymmetryError):
        factor(non_symmetric)


def test_factor_rejects_singular_matrix():
    singular = sp.csr_matrix(np.array([[1.0 + 0j, 1.0 + 0j], [1.0 + 0j, 1.0 + 0j]], dtype=complex))
    with pytest.raises(SolveSingularityError):
        factor(singular)


def test_factor_accepts_well_posed_symmetric_matrix():
    R = build_restriction(_PEC_DOFS, n_edges=4)
    A_ff, _ = reduce_system(sp.csr_matrix(_A), _B, R)
    factor(A_ff)  # must not raise


# --- Section 5.3: multi-RHS reuse gives identical results to independent solves ---


def test_multi_rhs_reuse_matches_independent_solves():
    R = build_restriction(_PEC_DOFS, n_edges=4)
    A_sp = sp.csr_matrix(_A)
    A_ff, _ = reduce_system(A_sp, np.zeros(4, dtype=complex), R)
    fact = factor(A_ff)

    rng = np.random.default_rng(0)
    rhs_list = [rng.normal(size=4) + 1j * rng.normal(size=4) for _ in range(3)]
    # Zero out the PEC-adjacent entry in each RHS, matching Section 4.3's
    # structural-zero assumption.
    for b in rhs_list:
        b[1] = 0.0

    reused_results = []
    for b in rhs_list:
        _, b_f = reduce_system(A_sp, b, R)
        a_f = solve_with_factorization(fact, b_f)
        reused_results.append(recover_solution(a_f, R))

    for b, reused in zip(rhs_list, reused_results):
        _, b_f = reduce_system(A_sp, b, R)
        fresh_fact = factor(A_ff)  # independent fresh factorization each time
        a_f_fresh = solve_with_factorization(fresh_fact, b_f)
        fresh = recover_solution(a_f_fresh, R)
        assert reused == pytest.approx(fresh, abs=1e-10)

    # Cross-check: results must differ across different RHS (not a no-op reuse bug).
    assert not np.allclose(reused_results[0], reused_results[1])
