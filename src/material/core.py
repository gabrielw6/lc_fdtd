"""material.core -- the MaterialModel interface, its two universal
invariant checks, and the tag-dispatch registry
(docs/module2_material_equations.md Section 1).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

_SYMMETRY_TOL = 1e-9
_PASSIVITY_TOL = 1e-9


class MaterialError(RuntimeError):
    """Base class for every error this package raises."""


class MaterialSymmetryError(MaterialError):
    """Raised when a tensor fails eps == eps^T (Section 1.2) -- the
    proximate cause of every downstream reciprocity failure in Module 3/8."""


class MaterialPassivityError(MaterialError):
    """Raised when a tensor (or, for the LC path, a material constant)
    fails the passivity requirement (Section 1.3/4.6)."""


class MaterialTagError(MaterialError):
    """Raised when `MaterialAssembly` is queried for a tag it has no
    model registered for."""


def check_symmetric(tensor: np.ndarray, tol: float = _SYMMETRY_TOL) -> None:
    """Section 1.2: eps_r(r) == eps_r(r)^T at every point (not merely
    Hermitian -- no magnetic bias means a reciprocal, symmetric medium)."""
    residual = float(np.abs(tensor - np.swapaxes(tensor, -1, -2)).max())
    if residual > tol:
        raise MaterialSymmetryError(f"tensor is not symmetric (max |eps - eps^T| = {residual!r} > tol {tol!r})")


def check_passive_generic(tensor: np.ndarray, tol: float = _PASSIVITY_TOL) -> None:
    """Section 1.3's generic, expensive per-call check: with
    eps_r = A - jB, passivity requires B = -Im(eps_r) to be positive
    semidefinite at every point -- a full eigendecomposition. Kept as the
    defense-in-depth default for any implementation without special
    structure; the LC path (Section 4.6) replaces this with two scalar
    comparisons made once, at load time."""
    B = -np.imag(tensor)
    eigenvalues = np.linalg.eigvalsh(B)
    min_eig = float(eigenvalues.min())
    if min_eig < -tol:
        raise MaterialPassivityError(
            f"non-passive material: min eigenvalue of -Im(eps_r) is {min_eig!r} < -{tol!r} "
            "(a positive imaginary eigenvalue is a gain medium)"
        )


def assemble_symmetric_tensor(components: np.ndarray) -> np.ndarray:
    """Reassembles (M,6) components [xx,xy,xz,yy,yz,zz] (Section 2's
    channel order) into (M,3,3) symmetric tensors -- shared by
    `material.regions`' Phase 3 and `material.tensor_interpolation`'s
    Phase 4 (Section 4.4)."""
    components = np.asarray(components, dtype=complex)
    if components.shape[-1] != 6:
        raise ValueError(f"expected 6 tensor components (xx,xy,xz,yy,yz,zz), got shape {components.shape!r}")
    M = components.shape[0]
    eps = np.empty((M, 3, 3), dtype=complex)
    xx, xy, xz, yy, yz, zz = (components[:, i] for i in range(6))
    eps[:, 0, 0], eps[:, 0, 1], eps[:, 0, 2] = xx, xy, xz
    eps[:, 1, 0], eps[:, 1, 1], eps[:, 1, 2] = xy, yy, yz
    eps[:, 2, 0], eps[:, 2, 1], eps[:, 2, 2] = xz, yz, zz
    return eps


class MaterialModel(ABC):
    """Section 1.1's contract. `epsilon`/`mu` are template methods: they
    call the subclass's `_epsilon`/`_mu` and then run the two universal
    invariant checks (Sections 1.2-1.3) on every call, so no subclass can
    forget them. A subclass with a provably stronger structure (the LC
    path, Section 4.6) overrides `_check_epsilon_passive` to replace the
    generic per-call eigendecomposition with its own cheaper, already-
    load-time-verified guarantee."""

    def epsilon(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        eps = np.asarray(self._epsilon(points), dtype=complex)
        check_symmetric(eps)
        self._check_epsilon_passive(eps)
        return eps

    def mu(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        mu = np.asarray(self._mu(points), dtype=complex)
        check_symmetric(mu)
        self._check_mu_passive(mu)
        return mu

    @abstractmethod
    def _epsilon(self, points: np.ndarray) -> np.ndarray:
        """Returns (M,3,3) relative permittivity tensors, unchecked."""

    @abstractmethod
    def _mu(self, points: np.ndarray) -> np.ndarray:
        """Returns (M,3,3) relative permeability tensors, unchecked."""

    def _check_epsilon_passive(self, eps: np.ndarray) -> None:
        check_passive_generic(eps)

    def _check_mu_passive(self, mu: np.ndarray) -> None:
        check_passive_generic(mu)


class MaterialAssembly:
    """Section 1.4: the tag-dispatch registry Module 3's assembler
    actually queries -- it is unaware which `MaterialModel` subclass
    answered a given tag's call."""

    def __init__(self, tag_to_model: dict[str, MaterialModel]) -> None:
        self._tag_to_model = dict(tag_to_model)

    def epsilon(self, tag: str, points: np.ndarray) -> np.ndarray:
        return self._model_for(tag).epsilon(points)

    def mu(self, tag: str, points: np.ndarray) -> np.ndarray:
        return self._model_for(tag).mu(points)

    def _model_for(self, tag: str) -> MaterialModel:
        try:
            return self._tag_to_model[tag]
        except KeyError:
            raise MaterialTagError(
                f"no material registered for tag {tag!r}; registered tags: {sorted(self._tag_to_model)}"
            ) from None
