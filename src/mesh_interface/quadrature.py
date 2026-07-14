"""mesh_interface.quadrature -- tabulated positive-weight symmetric
quadrature rules for tetrahedra and triangles, in barycentric coordinates
(docs/module1_mesh_interface_equations.md Section 6).

Each rule is self-verified against the exact barycentric-monomial integral
formula at import time (Section 6.3's "polynomial exactness spot-test...
run once per tabulated rule at load") -- a wrong coefficient fails loudly
here rather than propagating a silently wrong K/M downstream. Only orders
that can be stated with full confidence are tabulated (Section 6.2's
explicit "minimum workhorse rules": order 1 for the constant curl-curl
integrand, order 2 for the quadratic mass integrand); the higher Keast
rules Section 6.2 mentions for spatially varying eps_r are a documented
extension point -- add a table entry and the self-test verifies it, rather
than transcribing coefficients without a way to check them.
"""
from __future__ import annotations

from math import factorial

import numpy as np


class QuadratureError(RuntimeError):
    """Raised if a tabulated rule fails its own exactness self-test at
    import time -- indicates a transcription error in the table, not a
    caller mistake."""


def _tet_order2_points(a: float, b: float) -> np.ndarray:
    return np.array(
        [
            [a, b, b, b],
            [b, a, b, b],
            [b, b, a, b],
            [b, b, b, a],
        ]
    )


def _tri_order2_points(a: float, b: float) -> np.ndarray:
    return np.array(
        [
            [a, b, b],
            [b, a, b],
            [b, b, a],
        ]
    )


# --- Tetrahedron rules (barycentric (l0,l1,l2,l3), reference weights sum to 1) ---
_TET_RULES: dict[int, tuple[np.ndarray, np.ndarray]] = {
    1: (np.array([[0.25, 0.25, 0.25, 0.25]]), np.array([1.0])),
    2: (
        _tet_order2_points(0.5854101966249685, 0.13819660112501052),
        np.full(4, 0.25),
    ),
}

# --- Triangle rules (barycentric (l0,l1,l2), reference weights sum to 1) ---
_TRI_RULES: dict[int, tuple[np.ndarray, np.ndarray]] = {
    1: (np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]]), np.array([1.0])),
    2: (_tri_order2_points(2.0 / 3.0, 1.0 / 6.0), np.full(3, 1.0 / 3.0)),
}


def available_tet_orders() -> tuple[int, ...]:
    """The polynomial-exactness orders currently tabulated for a tet, in
    ascending order -- Module 3's adaptive quadrature procedure escalates
    over exactly this list; it never assumes a specific ceiling."""
    return tuple(sorted(_TET_RULES))


def tet_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference barycentric points (M,4) and weights (M,) for a tet,
    exact to polynomial degree `order`, with sum(weights) == 1."""
    try:
        return _TET_RULES[order]
    except KeyError:
        raise ValueError(
            f"no tetrahedron quadrature rule tabulated for order {order!r}; "
            f"available orders: {sorted(_TET_RULES)}"
        ) from None


def tri_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference barycentric points (M,3) and weights (M,) for a triangle,
    exact to polynomial degree `order`, with sum(weights) == 1."""
    try:
        return _TRI_RULES[order]
    except KeyError:
        raise ValueError(
            f"no triangle quadrature rule tabulated for order {order!r}; "
            f"available orders: {sorted(_TRI_RULES)}"
        ) from None


def _simplex_monomial_integral(exponents: tuple[int, ...]) -> float:
    """Exact integral of a barycentric monomial prod(lambda_i^a_i) over a
    d-simplex, normalized to unit measure (i.e. integrating the constant 1
    gives 1 -- matching the reference rules' sum(weights)==1 convention;
    physical points/weights are obtained by mapping + scaling by the real
    V or A outside this module, Section 6.1): the standard barycentric-
    monomial formula, generalized to d dimensions,
    integral = prod(a_i!) / (sum(a_i)+d)! * d! .
    Cross-checked at d=2,3 against the elementary "average of lambda_i over
    a simplex is 1/(d+1)" identity before being trusted here."""
    d = len(exponents) - 1
    return float(
        np.prod([factorial(a) for a in exponents]) * factorial(d) / factorial(sum(exponents) + d)
    )


def _monomials_up_to_degree(n_vars: int, degree: int):
    for total in range(degree + 1):
        yield from _monomials_of_degree(n_vars, total)


def _monomials_of_degree(n_vars: int, degree: int):
    if n_vars == 1:
        yield (degree,)
        return
    for first in range(degree + 1):
        for rest in _monomials_of_degree(n_vars - 1, degree - first):
            yield (first,) + rest


def _verify_rule(name: str, order: int, points: np.ndarray, weights: np.ndarray) -> None:
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise QuadratureError(f"{name} order {order}: weights do not sum to 1 (got {weights.sum()!r})")
    if np.any(weights <= 0):
        raise QuadratureError(f"{name} order {order}: rule has a non-positive weight (Section 6.3)")
    if not np.allclose(points.sum(axis=1), 1.0, atol=1e-12):
        raise QuadratureError(f"{name} order {order}: a point's barycentric coordinates do not sum to 1")

    n_bary = points.shape[1]
    for exponents in _monomials_up_to_degree(n_bary, order):
        numeric = float(np.sum(weights * np.prod(points ** np.array(exponents), axis=1)))
        exact = _simplex_monomial_integral(exponents)
        if not np.isclose(numeric, exact, rtol=1e-9, atol=1e-12):
            raise QuadratureError(
                f"{name} order {order}: monomial {exponents} integrates to {numeric!r}, expected {exact!r}"
            )


def _verify_all_rules() -> None:
    for order, (points, weights) in _TET_RULES.items():
        _verify_rule("tetrahedron", order, points, weights)
    for order, (points, weights) in _TRI_RULES.items():
        _verify_rule("triangle", order, points, weights)


_verify_all_rules()
