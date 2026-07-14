"""geometry_builder.builder -- GeometryBuilder.build(): the step-by-step
construction procedure (docs/module0_geometry_builder_equations.md Sections
2, 3, 6).

Depends only on `meshing`'s generic primitives (box/fragment/physical-group
construction, mesh generation, mesh I/O, mesh sizing) -- never the reverse,
and never on anything downstream (Module 1). This is the "interface
abstraction layer" boundary: `meshing` knows nothing about microstrips,
LC cutouts, ports, or PML; this module knows nothing about `meshing`'s own
cavity+sample pipeline (`pipeline.py`, `cache.py`, `geometry_spec.py`).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy import constants

from meshing import mesh_generation, mesh_io, mesh_sizing
from meshing import standard_shapes
from meshing import tagging as meshing_tagging

from .material_spec import build_material_spec
from .params import DerivedGeometry, GeometryParams, derive
from .tags import (
    AIR,
    LC,
    PEC_GROUND,
    PEC_LINE,
    PML_OUTER_PEC,
    PML_TOP,
    PMC_SIDE,
    PORT_1,
    PORT_2,
    SUBSTRATE,
    SURFACE_TAGS,
    MaterialSpecStub,
    MeshHandle,
)

DimTag = tuple[int, int]

_VOLUME_TOLERANCE_FACTOR = 1e-6  # relative, mirrors meshing.interference's convention


class GeometryConsistencyError(RuntimeError):
    """Raised when a post-construction consistency check (Section 6) fails
    -- signals a bad fragment/tagging operation, not a bad user parameter
    (those are caught earlier, at `GeometryParams` construction, as
    `GeometryParameterError`)."""


class GeometryBuilder:
    """Builds the one fixed topology docs/module0_geometry_builder_equations.md
    describes: a microstrip line with a centered rectangular LC cutout in
    the substrate, ground plane below, air + PML above. Module 0 sits
    upstream of Module 1 (`mesh.interface`) -- it produces a tagged mesh,
    Module 1 only adapts one (Section 7)."""

    def build(self, params: GeometryParams) -> tuple[MeshHandle, MaterialSpecStub]:
        geom = derive(params)  # Section 1.3 + Section 6's pre-CAD checks

        import gmsh

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("geometry_builder")
        try:
            volume_entities, rect_descendants = self._build_and_fragment(params, geom)
            volume_groups = self._tag_volumes(volume_entities)
            self._check_volumes(volume_groups, params, geom)

            surface_groups = self._tag_surfaces(params, geom, volume_entities, rect_descendants)
            self._check_surface_coverage(volume_entities, surface_groups)

            h_char = mesh_sizing.characteristic_length(
                params.reference_frequency,
                params.eps_r_substrate * constants.epsilon_0,
                constants.mu_0,
                params.target_elements_per_wavelength,
            )
            mesh_generation.generate_mesh(h_char)

            mesh_handle = self._read_back()
        finally:
            gmsh.finalize()

        return mesh_handle, build_material_spec(params)

    # --- Section 3, steps 3-8: geometry construction, fragment, tagging ---

    def _build_and_fragment(
        self, params: GeometryParams, geom: DerivedGeometry
    ) -> tuple[list[list[DimTag]], list[DimTag]]:
        h_sub = params.h_sub

        # Section 2's five-brick decomposition: pre-cavity, left wing, LC,
        # right wing, post-cavity, all sharing z in [0, h_sub].
        pre = standard_shapes.build_box((0.0, 0.0, 0.0), (geom.x_c0, params.W_sub, h_sub))
        left_wing = standard_shapes.build_box((geom.x_c0, 0.0, 0.0), (params.L_lc, geom.y_lc0, h_sub))
        lc = standard_shapes.build_box(
            (geom.x_c0, geom.y_lc0, 0.0), (params.L_lc, params.W_lc, h_sub)
        )
        right_wing = standard_shapes.build_box(
            (geom.x_c0, geom.y_lc1, 0.0), (params.L_lc, params.W_sub - geom.y_lc1, h_sub)
        )
        post = standard_shapes.build_box((geom.x_c1, 0.0, 0.0), (params.L - geom.x_c1, params.W_sub, h_sub))

        # Section 3, steps 6-7: Air and PML slabs stacked above the substrate.
        air = standard_shapes.build_box((0.0, 0.0, h_sub), (params.L, params.W_sub, params.h_air))
        pml = standard_shapes.build_box(
            (0.0, 0.0, geom.z_air_top), (params.L, params.W_sub, params.h_pml)
        )

        # Section 3, step 4: the trace-footprint partitioning rectangle,
        # embedded on the substrate/air interface plane z=h_sub.
        import gmsh

        occ = gmsh.model.occ
        rect_tag = occ.addRectangle(0.0, geom.y0_trace, h_sub, params.L, params.w)
        occ.synchronize()

        # Section 3, step 5 (composite) + step 6's propagation note: one
        # fragment call across every brick and the embedded rectangle --
        # already-conformal interfaces (the five bricks against each other)
        # are a no-op; the Air/PML interfaces become conformal here, and
        # the rectangle splits the substrate/air interface into the trace
        # patch plus its surrounding remainder.
        objects = [pre, left_wing, lc, right_wing, post, air, pml]
        _out, out_map = meshing_tagging.fragment(objects, [(2, rect_tag)])

        volume_entities = out_map[:7]
        rect_descendants = out_map[7]
        return volume_entities, rect_descendants

    def _tag_volumes(self, volume_entities: list[list[DimTag]]) -> dict[str, list[DimTag]]:
        pre, left_wing, lc, right_wing, post, air, pml = volume_entities
        groups = {
            SUBSTRATE: [*pre, *left_wing, *right_wing, *post],
            LC: list(lc),
            AIR: list(air),
            PML_TOP: list(pml),
        }
        for name, tags in groups.items():
            meshing_tagging.add_physical_group(3, [t for _, t in tags], name)
        return groups

    def _tag_surfaces(
        self,
        params: GeometryParams,
        geom: DerivedGeometry,
        volume_entities: list[list[DimTag]],
        rect_descendants: list[DimTag],
    ) -> dict[str, list[DimTag]]:
        import gmsh

        occ = gmsh.model.occ
        all_volumes = [t for entities in volume_entities for t in entities]
        exterior = gmsh.model.getBoundary(all_volumes, combined=True, oriented=False)

        # Coordinate tolerance scaled to the model's own size (Section 0.5's
        # unit-handling discipline applied here too: never a bare literal).
        eps = 1e-9 * max(geom.z_pml_top, params.L, params.W_sub)

        buckets: dict[str, list[DimTag]] = {name: [] for name in SURFACE_TAGS}
        for dim, tag in exterior:
            cx, cy, cz = occ.getCenterOfMass(dim, tag)
            if abs(cz - geom.z_gnd) < eps:
                buckets[PEC_GROUND].append((dim, tag))
            elif cz <= geom.z_air_top + eps and abs(cx - 0.0) < eps:
                buckets[PORT_1].append((dim, tag))
            elif cz <= geom.z_air_top + eps and abs(cx - params.L) < eps:
                buckets[PORT_2].append((dim, tag))
            elif cz <= geom.z_air_top + eps and (abs(cy - 0.0) < eps or abs(cy - params.W_sub) < eps):
                buckets[PMC_SIDE].append((dim, tag))
            else:
                # Section 4.3: everything else exterior at this point is a
                # face of the PML_TOP brick other than its (interior,
                # shared) bottom -- top cap, two lateral sides, two end caps.
                buckets[PML_OUTER_PEC].append((dim, tag))

        buckets[PEC_LINE] = list(rect_descendants)

        for name in SURFACE_TAGS:
            if buckets[name]:
                meshing_tagging.add_physical_group(2, [t for _, t in buckets[name]], name)
        return buckets

    # --- Section 6: pre-mesh consistency checks ---

    def _check_volumes(
        self, groups: dict[str, list[DimTag]], params: GeometryParams, geom: DerivedGeometry
    ) -> None:
        import gmsh

        occ = gmsh.model.occ
        v_lc_expected = params.L_lc * params.W_lc * params.h_sub
        v_substrate_expected = params.L * params.W_sub * params.h_sub - v_lc_expected

        v_lc_actual = sum(occ.getMass(dim, tag) for dim, tag in groups[LC])
        v_substrate_actual = sum(occ.getMass(dim, tag) for dim, tag in groups[SUBSTRATE])

        if abs(v_lc_actual - v_lc_expected) > _VOLUME_TOLERANCE_FACTOR * v_lc_expected:
            raise GeometryConsistencyError(
                f"LC volume mismatch after fragment: expected {v_lc_expected!r}, got {v_lc_actual!r} "
                "-- likely a gap/overlap between the LC brick and its wings"
            )
        if abs(v_substrate_actual - v_substrate_expected) > _VOLUME_TOLERANCE_FACTOR * v_substrate_expected:
            raise GeometryConsistencyError(
                f"SUBSTRATE volume mismatch after fragment: expected {v_substrate_expected!r}, "
                f"got {v_substrate_actual!r} -- likely a gap/overlap in the five-brick decomposition"
            )

    def _check_surface_coverage(
        self, volume_entities: list[list[DimTag]], surface_groups: dict[str, list[DimTag]]
    ) -> None:
        import gmsh

        all_volumes = [t for entities in volume_entities for t in entities]
        exterior = set(gmsh.model.getBoundary(all_volumes, combined=True, oriented=False))

        # PEC_LINE is deliberately excluded: it is an internal partitioning
        # face (Section 4.3's list is the exterior-wall vocabulary), not a
        # member of `exterior`.
        tagged: set[DimTag] = set()
        for name in (PEC_GROUND, PORT_1, PORT_2, PMC_SIDE, PML_OUTER_PEC):
            tagged.update(surface_groups[name])

        missing = exterior - tagged
        if missing:
            raise GeometryConsistencyError(
                f"{len(missing)} exterior face(s) left untagged (would be a de-facto PMC wall "
                f"by default, almost certainly unintended): {sorted(missing)}"
            )

    # --- Section 3, step 13: hand the mesh onward ---

    def _read_back(self) -> MeshHandle:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "mesh.msh"
            mesh_io.write_mesh(path)
            mesh = mesh_io.read_mesh(path)

        name_by_physical_id: dict[tuple[int, int], str] = {}
        for name, (tag_id, dim) in mesh.field_data.items():
            name_by_physical_id[(int(tag_id), int(dim))] = name

        tets_blocks: list[np.ndarray] = []
        volume_tag_blocks: list[np.ndarray] = []
        surface_chunks: dict[str, list[np.ndarray]] = {name: [] for name in SURFACE_TAGS}

        physical_by_block = mesh.cell_data.get("gmsh:physical", [])
        for block, phys in zip(mesh.cells, physical_by_block):
            phys = np.asarray(phys)
            if block.type == "tetra":
                for pid in np.unique(phys):
                    name = name_by_physical_id.get((int(pid), 3))
                    if name is None:
                        continue
                    conn = block.data[phys == pid]
                    tets_blocks.append(conn)
                    volume_tag_blocks.append(np.full(len(conn), name))
            elif block.type == "triangle":
                for pid in np.unique(phys):
                    name = name_by_physical_id.get((int(pid), 2))
                    if name is None or name not in surface_chunks:
                        continue
                    surface_chunks[name].append(block.data[phys == pid])

        tets = np.concatenate(tets_blocks, axis=0) if tets_blocks else np.empty((0, 4), dtype=int)
        volume_tags = (
            np.concatenate(volume_tag_blocks, axis=0) if volume_tag_blocks else np.empty((0,), dtype=object)
        )
        surface_tags = {
            name: (np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3), dtype=int))
            for name, chunks in surface_chunks.items()
        }

        return MeshHandle(vertices=mesh.points, tets=tets, volume_tags=volume_tags, surface_tags=surface_tags)
