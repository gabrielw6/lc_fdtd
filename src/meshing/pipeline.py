"""Meshing module -- `build_mesh()` orchestration (docs/meshing_module_plan.md
Section 1, step 11; Section 0.1's shared pipeline). The only file in this
package allowed to call into more than one of the others.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from scipy import constants

from . import mesh_generation, mesh_io, mesh_sizing, standard_shapes, step_import, tagging
from .cache import MeshCache
from .geometry_spec import CavityGeometryInput, SampleGeometryInput, StepCavityInput, StepSampleInput
from .interference import check_containment
from .transforms import apply_to_shape

DimTag = tuple[int, int]

_DEFAULT_ELEMENTS_PER_WAVELENGTH = 10


def _resolve_cavity_solid(cavity_input: CavityGeometryInput) -> DimTag:
    if isinstance(cavity_input, StepCavityInput):
        dim_tags = step_import.import_step(cavity_input.path, cavity_input.length_unit)
        if len(dim_tags) != 1:
            raise ValueError(f"expected exactly one solid in {cavity_input.path}, got {len(dim_tags)}")
        return dim_tags[0]
    return standard_shapes.build_cavity_solid(cavity_input)


def _resolve_sample_solid(sample_input: SampleGeometryInput) -> DimTag:
    if isinstance(sample_input, StepSampleInput):
        dim_tags = step_import.import_step(sample_input.path, sample_input.length_unit)
        if len(dim_tags) != 1:
            raise ValueError(f"expected exactly one solid in {sample_input.path}, got {len(dim_tags)}")
        # Section 0.4: a STEP sample's position relative to the outer volume
        # is an explicit, required rigid transform -- applied before
        # anything else touches the shape.
        apply_to_shape(sample_input.transform, dim_tags)
        return dim_tags[0]
    return standard_shapes.build_sample_solid(sample_input)


def build_mesh(
    cavity_input: CavityGeometryInput,
    sample_input: SampleGeometryInput,
    reference_frequency: float,
    target_elements_per_wavelength: int = _DEFAULT_ELEMENTS_PER_WAVELENGTH,
    background_eps: complex | None = None,
    background_mu: complex | None = None,
    cache: MeshCache | None = None,
) -> mesh_io.MeshResult:
    """Orchestrates the full pipeline (Section 0.1: one shared pipeline, two
    geometry sources) -- resolve geometry, size the mesh, check containment
    (raising *before* any meshing work if it fails, Section 3), fragment +
    tag, generate, export/round-trip, assemble the result. Optionally
    cached (Section 5) by the full geometry spec + resolution.

    `reference_frequency` drives mesh sizing (Section 0.6) and is always
    required; `background_eps`/`background_mu` default to vacuum.
    """

    def compute() -> mesh_io.MeshResult:
        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("fem_mesh")
        try:
            cavity_dim_tag = _resolve_cavity_solid(cavity_input)
            sample_dim_tag = _resolve_sample_solid(sample_input)

            occ = gmsh.model.occ
            cavity_volume = occ.getMass(*cavity_dim_tag)
            sample_volume = occ.getMass(*sample_dim_tag)

            # Section 3: containment check BEFORE any meshing work -- don't
            # proceed to fragment/tag/mesh a geometrically invalid config.
            check_containment(cavity_dim_tag, sample_dim_tag)

            tagged = tagging.fragment_and_tag(cavity_dim_tag, sample_dim_tag)

            eps = background_eps if background_eps is not None else constants.epsilon_0
            mu = background_mu if background_mu is not None else constants.mu_0
            h_char = mesh_sizing.characteristic_length(
                reference_frequency, eps, mu, target_elements_per_wavelength
            )
            quality = mesh_generation.generate_mesh(h_char)

            with tempfile.TemporaryDirectory() as tmp_dir:
                mesh_path = Path(tmp_dir) / "mesh.msh"
                mesh_io.write_mesh(mesh_path)
                mesh = mesh_io.read_mesh(mesh_path)

            return mesh_io.assemble_mesh_result(
                mesh=mesh,
                sample_physical_tag=tagged.sample_physical_tag,
                background_physical_tag=tagged.background_physical_tag,
                boundary_physical_tag=tagged.boundary_physical_tag,
                cavity_volume=cavity_volume,
                sample_volume=sample_volume,
                quality=quality,
            )
        finally:
            gmsh.finalize()

    if cache is None:
        return compute()
    return cache.get_or_compute(cavity_input, sample_input, target_elements_per_wavelength, compute)
