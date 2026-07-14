"""Hand-computable two-tet synthetic mesh for test_visualization, tagged
with `geometry_builder.tags`' actual vocabulary (SUBSTRATE/AIR volumes;
PORT_1/PORT_2/PEC_LINE/PEC_GROUND/PML_OUTER_PEC/PMC_SIDE surfaces) rather
than the placeholder "A"/"B" tags `test_mesh_interface`'s own fixture uses
-- this module exercises `visualization.geometry_view`'s tag-name
dispatch, so the tags need to be the real ones.

    v0=(0,0,0) v1=(1,0,0) v2=(0,1,0) v3=(0,0,1) v4=(1,1,1)
    tetA = [0,1,2,3]  SUBSTRATE (V=1/6)
    tetB = [4,1,2,3]  AIR       (V=1/3)

tetA and tetB share face {1,2,3} (interior, incidence 2) -- the
SUBSTRATE/AIR interface `_substrate_envelope_triangles` must pick up from
the SUBSTRATE side even though it carries no surface tag, exactly like the
real substrate/air interface away from the PEC_LINE patch. The other six
faces are each incident to exactly one tet (boundary, incidence 1).
"""
import numpy as np

VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],  # 0
        [1.0, 0.0, 0.0],  # 1
        [0.0, 1.0, 0.0],  # 2
        [0.0, 0.0, 1.0],  # 3
        [1.0, 1.0, 1.0],  # 4
    ]
)
TETS = np.array([[0, 1, 2, 3], [4, 1, 2, 3]])
VOLUME_TAGS = np.array(["SUBSTRATE", "AIR"])

BOUNDARY_FACES = {
    "A_023": (0, 2, 3),
    "A_013": (0, 1, 3),
    "A_012": (0, 1, 2),
    "B_423": (4, 2, 3),
    "B_413": (4, 1, 3),
    "B_412": (4, 1, 2),
}

SURFACE_TAGS = {
    "PEC_GROUND": np.array([BOUNDARY_FACES["A_023"]]),
    "PEC_LINE": np.array([BOUNDARY_FACES["A_013"]]),
    "PORT_1": np.array([BOUNDARY_FACES["A_012"]]),
    "PORT_2": np.array([BOUNDARY_FACES["B_423"]]),
    "PML_OUTER_PEC": np.array([BOUNDARY_FACES["B_413"]]),
    "PMC_SIDE": np.array([BOUNDARY_FACES["B_412"]]),
}
