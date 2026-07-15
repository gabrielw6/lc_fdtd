"""Validation suite for visualization.geometry_view: plot_geometry and
plot_mesh, exercised on the hand-computable two-tet synthetic mesh in
_synthetic_mesh.py (real SUBSTRATE/AIR/PORT_1/PORT_2/PEC_LINE/PMC_SIDE
tags, no Gmsh needed).
"""
import matplotlib

matplotlib.use("Agg")  # never open a GUI window during tests

import numpy as np
import pytest

from mesh_interface import MeshInterface
from visualization.geometry_view import (
    PlottingUnavailableError,
    _triangles_for_tag,
    _volume_envelope_triangles,
    plot_geometry,
    plot_mesh,
)

from ._synthetic_mesh import SURFACE_TAGS, TETS, VERTICES, VOLUME_TAGS


def _mesh() -> MeshInterface:
    return MeshInterface(VERTICES, TETS, VOLUME_TAGS, SURFACE_TAGS)


# --- triangle extraction (no matplotlib needed) ------------------------------


def test_triangles_for_tag_returns_one_triangle_per_boundary_face():
    mesh = _mesh()
    tris = _triangles_for_tag(mesh, "PORT_1")
    assert tris.shape == (1, 3, 3)
    expected = VERTICES[[0, 1, 2]]  # BOUNDARY_FACES["A_012"]
    assert set(map(tuple, tris[0])) == set(map(tuple, expected))


def test_triangles_for_tag_empty_for_unknown_tag():
    mesh = _mesh()
    tris = _triangles_for_tag(mesh, "PORT_3")
    assert tris.shape == (0, 3, 3)


def test_substrate_envelope_includes_boundary_and_substrate_air_interface():
    """tetA=SUBSTRATE contributes its 3 exterior boundary faces plus the
    interior SUBSTRATE/AIR interface face {1,2,3} (untagged, picked up from
    the SUBSTRATE side) -- tetB=AIR contributes nothing."""
    mesh = _mesh()
    tris = _volume_envelope_triangles(mesh, ("SUBSTRATE",))
    assert tris.shape == (4, 3, 3)


def test_substrate_envelope_empty_when_no_substrate_tets():
    mesh = MeshInterface(VERTICES, TETS, np.array(["AIR", "AIR"]), SURFACE_TAGS)
    tris = _volume_envelope_triangles(mesh, ("SUBSTRATE",))
    assert len(tris) == 0


def test_lc_envelope_includes_boundary_and_lc_air_interface():
    """The same envelope logic, generalized to the LC volume group
    (post-review geometry-viewer fix): tetA relabeled LC behaves exactly
    like tetA=SUBSTRATE did above -- 3 exterior faces plus the interior
    LC/AIR interface face, tetB=AIR contributing nothing."""
    mesh = MeshInterface(VERTICES, TETS, np.array(["LC", "AIR"]), SURFACE_TAGS)
    tris = _volume_envelope_triangles(mesh, ("LC",))
    assert tris.shape == (4, 3, 3)


def test_combined_group_has_no_notch_at_the_shared_internal_face():
    """`_volume_envelope_triangles` accepts a multi-tag group (its actual
    documented contract, generalized from the single-tag SUBSTRATE-only
    original): querying ("SUBSTRATE", "LC") together must NOT pick up
    their shared internal face {1,2,3} -- unlike querying ("SUBSTRATE",)
    alone, which renders that same face as an outer boundary, i.e. a
    notch. `plot_geometry` itself calls this with single-tag groups
    (drawing SUBSTRATE and LC as two separately-colored envelopes, the
    geometry-viewer fix's actual chosen approach -- see its own
    docstring), not the combined form exercised here, but the combined
    form is part of this helper's own generalized contract and worth
    covering directly."""
    mesh = MeshInterface(VERTICES, TETS, np.array(["SUBSTRATE", "LC"]), SURFACE_TAGS)
    combined = _volume_envelope_triangles(mesh, ("SUBSTRATE", "LC"))
    substrate_only = _volume_envelope_triangles(mesh, ("SUBSTRATE",))

    shared_face = frozenset(map(tuple, VERTICES[[1, 2, 3]]))

    def _has_shared_face(tris: np.ndarray) -> bool:
        return any(frozenset(map(tuple, tri)) == shared_face for tri in tris)

    assert not _has_shared_face(combined)
    assert _has_shared_face(substrate_only)
    # tetA+tetB form one contiguous 6-exterior-face block once combined
    # (each tet's own 3 non-shared faces); SUBSTRATE alone (4: tetA's 3
    # exterior + the shared face it still claims) is a strict subset of
    # the combined group's face count, not merely "fewer notches."
    assert len(combined) == 6
    assert len(substrate_only) == 4


# --- plot_geometry ------------------------------------------------------------


def test_plot_geometry_draws_ports_and_line_and_substrate_not_pmc():
    mesh = _mesh()
    fig = plot_geometry(mesh, show=False)
    ax = fig.axes[0]
    # substrate envelope (1) + PORT_1 + PORT_2 + PEC_LINE = 4 collections;
    # PMC_SIDE and PML_OUTER_PEC must never contribute one.
    assert len(ax.collections) == 4
    legend_labels = {t.get_text() for t in ax.get_legend().get_texts()}
    assert legend_labels == {"PORT_1", "PORT_2", "PEC_LINE"}
    assert "PMC_SIDE" not in legend_labels
    assert "PML_OUTER_PEC" not in legend_labels


def test_plot_geometry_show_substrate_false_omits_envelope_collection():
    mesh = _mesh()
    fig = plot_geometry(mesh, show=False, show_substrate=False)
    ax = fig.axes[0]
    assert len(ax.collections) == 3  # PORT_1 + PORT_2 + PEC_LINE only


def test_plot_geometry_writes_output_file(tmp_path):
    mesh = _mesh()
    out = tmp_path / "geometry.png"
    plot_geometry(mesh, show=False, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


# --- plot_mesh ------------------------------------------------------------


def test_plot_mesh_draws_every_global_edge_once():
    mesh = _mesh()
    fig = plot_mesh(mesh, show=False)
    ax = fig.axes[0]
    assert len(ax.collections) == 1
    # Line3DCollection only populates the 2D-projected get_segments() at
    # draw time; the pre-projection 3D segments it was constructed from are
    # what's actually asserted here.
    segments = ax.collections[0]._segments3d
    assert len(segments) == mesh.n_edges == 9


def test_plot_mesh_writes_output_file(tmp_path):
    mesh = _mesh()
    out = tmp_path / "mesh.png"
    plot_mesh(mesh, show=False, output=out)
    assert out.exists()
    assert out.stat().st_size > 0


# --- matplotlib-unavailable path --------------------------------------------


def test_plot_geometry_raises_clean_error_when_matplotlib_unavailable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", None)
    with pytest.raises(PlottingUnavailableError):
        plot_geometry(_mesh(), show=False)


def test_plot_mesh_raises_clean_error_when_matplotlib_unavailable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "matplotlib.pyplot", None)
    with pytest.raises(PlottingUnavailableError):
        plot_mesh(_mesh(), show=False)
