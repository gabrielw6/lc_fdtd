"""material.spec -- the loader that reads a user-facing spec file and
builds the tag->material dispatch registry Module 3 queries
(docs/module2_material_equations.md Section 5).

Built last (Section 6, step 8): it only wires together components already
validated individually. Requires PyYAML (a spec file is a small, static
config; no reason to hand-roll a parser for it).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .core import MaterialAssembly, MaterialModel
from .regions import ConstantMaterial, ScalarFieldMaterial, TensorFieldMaterial
from .tensor_interpolation import DirectorFieldMaterial

RegionBounds = tuple[np.ndarray, np.ndarray]

_TENSOR_COMPONENT_ORDER = ("xx", "xy", "xz", "yy", "yz", "zz")
_CONSTANT_FIELD_HALF_EXTENT = 1.0e3  # meters: generously larger than any physically reasonable model


class MaterialSpecError(RuntimeError):
    """Raised on a malformed material spec (missing/unknown type, missing
    required field for a type) -- caught here, at load time, rather than
    surfacing as an unrelated AttributeError deep in Module 3."""


def load_material_spec(
    path: str | Path | None = None,
    geometry_stub: Any = None,
    region_bounds: dict[str, RegionBounds] | None = None,
) -> MaterialAssembly:
    """Section 5.3's build process + Section 5.4's merge. `geometry_stub`
    is duck-typed on `.entries` (a `geometry_builder.MaterialSpecStub`) or
    accepted directly as a plain `{tag: {type, eps_r, ...}}` dict -- this
    module never imports `geometry_builder`. `region_bounds` is an
    optional `{tag: (mins, maxs)}` map threaded into each interpolated
    tag's Section 2.4 coverage check.
    """
    if path is not None:
        path = Path(path)
        with open(path) as f:
            user_spec = yaml.safe_load(f) or {}
        materials: dict[str, dict] = dict(user_spec.get("materials", {}))
        base_dir = path.parent
    else:
        materials = {}
        base_dir = Path(".")

    if geometry_stub is not None:
        stub_entries = getattr(geometry_stub, "entries", geometry_stub)
        merged = dict(stub_entries)
        merged.update(materials)  # user-supplied entries take precedence over Module 0's stub
        materials = merged

    if not materials:
        raise MaterialSpecError("no materials supplied -- need a spec file, a geometry_stub, or both")

    region_bounds = region_bounds or {}
    tag_to_model: dict[str, MaterialModel] = {}
    for tag, entry in materials.items():
        tag_to_model[tag] = _build_model(tag, dict(entry), base_dir, region_bounds.get(tag))

    return MaterialAssembly(tag_to_model)


def _build_model(tag: str, entry: dict, base_dir: Path, bounds: RegionBounds | None) -> MaterialModel:
    kind = entry.get("type")
    if kind == "constant":
        return _build_constant(tag, entry)
    if kind == "scalar_field":
        return _build_scalar_field(tag, entry, base_dir, bounds)
    if kind == "tensor_field":
        return _build_tensor_field(tag, entry, base_dir, bounds)
    if kind == "director_field":
        return _build_director_field(tag, entry, base_dir, bounds)
    raise MaterialSpecError(
        f"tag {tag!r}: unknown material type {kind!r} (expected 'constant', 'scalar_field', "
        "'tensor_field', or 'director_field')"
    )


def _build_constant(tag: str, entry: dict) -> ConstantMaterial:
    if "eps_r" not in entry:
        raise MaterialSpecError(f"tag {tag!r}: type 'constant' requires 'eps_r'")
    eps_r = complex(entry["eps_r"])
    tan_delta = entry.get("tan_delta", 0.0)
    if tan_delta:
        eps_r = eps_r * (1.0 - 1j * tan_delta)
    return ConstantMaterial(eps_r, complex(entry.get("mu_r", 1.0)))


def _build_scalar_field(tag: str, entry: dict, base_dir: Path, bounds: RegionBounds | None) -> ScalarFieldMaterial:
    if "file" not in entry:
        raise MaterialSpecError(f"tag {tag!r}: type 'scalar_field' requires 'file'")
    return ScalarFieldMaterial.from_file(
        base_dir / entry["file"], mu_r=complex(entry.get("mu_r", 1.0)), region_bounds=bounds
    )


def _build_tensor_field(tag: str, entry: dict, base_dir: Path, bounds: RegionBounds | None) -> TensorFieldMaterial:
    mu_r = complex(entry.get("mu_r", 1.0))
    if "file" in entry:
        return TensorFieldMaterial.from_file(base_dir / entry["file"], mu_r=mu_r, region_bounds=bounds)
    if "eps_r_components" in entry:
        comps = entry["eps_r_components"]
        missing = [k for k in _TENSOR_COMPONENT_ORDER if k not in comps]
        if missing:
            raise MaterialSpecError(f"tag {tag!r}: eps_r_components missing key(s) {missing}")
        values = np.array([complex(comps[k]) for k in _TENSOR_COMPONENT_ORDER])
        field = _constant_tensor_field(values)
        return TensorFieldMaterial(field, mu_r=mu_r, region_bounds=bounds)
    raise MaterialSpecError(f"tag {tag!r}: type 'tensor_field' requires 'file' or 'eps_r_components'")


def _build_director_field(tag: str, entry: dict, base_dir: Path, bounds: RegionBounds | None) -> DirectorFieldMaterial:
    if "file" not in entry:
        raise MaterialSpecError(f"tag {tag!r}: type 'director_field' requires 'file'")
    if "eps_perp" not in entry or "eps_parallel" not in entry:
        raise MaterialSpecError(f"tag {tag!r}: type 'director_field' requires 'eps_perp' and 'eps_parallel'")
    eps_perp = complex(entry["eps_perp"]) - 1j * entry.get("eps_perp_im", 0.0)
    eps_parallel = complex(entry["eps_parallel"]) - 1j * entry.get("eps_parallel_im", 0.0)
    return DirectorFieldMaterial.from_file(
        base_dir / entry["file"], eps_perp, eps_parallel, mu_r=complex(entry.get("mu_r", 1.0)), region_bounds=bounds
    )


def _constant_tensor_field(components: np.ndarray):
    """A spatially-constant tensor, given directly (not from a file), is
    represented as a trivial structured grid spanning a generously large
    box -- interpolation then returns the same value everywhere within it,
    without needing a degenerate single-point Delaunay triangulation."""
    from .interpolation import SampledField

    axis = np.array([-_CONSTANT_FIELD_HALF_EXTENT, _CONSTANT_FIELD_HALF_EXTENT])
    grid_values = np.broadcast_to(components, (2, 2, 2, 6)).copy()
    return SampledField.structured((axis, axis, axis), grid_values)
