"""Meshing module -- standard-shape geometry -> OCC primitive dispatch
(docs/meshing_module_plan.md Section 1, step 3).

Builds an OCC solid directly from a shape dataclass's own dimension
fields -- plain geometric data, nothing else.
"""
from __future__ import annotations

import numpy as np

from .geometry_spec import Box, CoaxialDomain, Cylinder, CylindricalDomain, Matrix3, RigidTransform, Slab, Sphere
from .transforms import apply_to_shape, orthonormal_frame

DimTag = tuple[int, int]

CavityShape = Box | CylindricalDomain | CoaxialDomain
SampleShape = Sphere | Cylinder | Slab


def build_box(corner: tuple[float, float, float], dims: tuple[float, float, float]) -> DimTag:
    """Returns the (dim, tag) of an axis-aligned OCC box of size `dims`
    with its lower corner at `corner`. Generic building block -- a caller
    needing more than one box at controlled positions (e.g. a multi-brick
    parametric layout) uses this directly instead of the single-origin-
    cornered `Box` dispatch in `build_cavity_solid`. Requires an active
    Gmsh model (caller's responsibility)."""
    import gmsh

    occ = gmsh.model.occ
    cx, cy, cz = (float(v) for v in corner)
    dx, dy, dz = (float(v) for v in dims)
    tag = occ.addBox(cx, cy, cz, dx, dy, dz)
    occ.synchronize()
    return (3, tag)


def build_cavity_solid(shape: CavityShape) -> DimTag:
    """Returns the (dim, tag) of the OCC solid for the outer volume, built
    directly from `shape`'s own dimension fields. Requires an active Gmsh
    model (caller's responsibility -- this function doesn't initialize/
    finalize Gmsh itself)."""
    import gmsh

    occ = gmsh.model.occ
    if isinstance(shape, Box):
        return build_box((0.0, 0.0, 0.0), (shape.a, shape.b, shape.c))
    elif isinstance(shape, CoaxialDomain):
        # CoaxialDomain is checked before CylindricalDomain -- both are
        # unrelated classes here (no inheritance), but keeping the more
        # specific / less common case first avoids any future risk if that
        # ever changes.
        outer = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, shape.length, shape.outer_radius)
        inner = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, shape.length, shape.inner_radius)
        result, _ = occ.cut([(3, outer)], [(3, inner)])
        occ.synchronize()
        return result[0]
    elif isinstance(shape, CylindricalDomain):
        tag = occ.addCylinder(0.0, 0.0, 0.0, 0.0, 0.0, shape.length, shape.radius)
    else:
        raise ValueError(f"no OCC primitive builder registered for cavity shape {type(shape).__name__!r}")
    occ.synchronize()
    return (3, tag)


def build_sample_solid(shape: SampleShape) -> DimTag:
    """Returns the (dim, tag) of the OCC solid for `shape`, positioned
    directly from its own center/axis/normal fields."""
    import gmsh

    occ = gmsh.model.occ
    if isinstance(shape, Sphere):
        cx, cy, cz = (float(v) for v in shape.center)
        tag = occ.addSphere(cx, cy, cz, shape.radius)
    elif isinstance(shape, Cylinder):
        axis = np.asarray(shape.axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        base = np.asarray(shape.center, dtype=float) - 0.5 * shape.height * axis
        direction = shape.height * axis
        tag = occ.addCylinder(*base.tolist(), *direction.tolist(), shape.radius)
    elif isinstance(shape, Slab):
        tag = _build_slab(shape)
    else:
        raise ValueError(f"no OCC primitive builder registered for sample shape {type(shape).__name__!r}")
    occ.synchronize()
    return (3, tag)


def _build_slab(shape: Slab) -> int:
    """A box built axis-aligned and centered at the local origin, then
    rotated so its local (x, y, z) axes land on (e1, e2, normal), and
    translated to `center`."""
    import gmsh

    occ = gmsh.model.occ
    ex, ey = shape.extent
    tag = occ.addBox(-ex / 2.0, -ey / 2.0, -shape.thickness / 2.0, ex, ey, shape.thickness)
    occ.synchronize()

    e1, e2, n_hat = orthonormal_frame(shape.normal)
    rotation: Matrix3 = _columns_to_matrix(e1, e2, n_hat)
    translation = tuple(float(v) for v in shape.center)
    transform = RigidTransform(translation=translation, rotation=rotation)  # type: ignore[arg-type]
    apply_to_shape(transform, [(3, tag)])
    return tag


def _columns_to_matrix(c0: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> Matrix3:
    R = np.stack([c0, c1, c2], axis=1)
    rows = [tuple(float(x) for x in row) for row in R]
    return (rows[0], rows[1], rows[2])  # type: ignore[return-value]
