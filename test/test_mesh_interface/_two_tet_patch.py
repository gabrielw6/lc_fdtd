"""Shared hand-computable two-tet synthetic mesh used across
test_mesh_interface's unit tests -- no Gmsh needed, `MeshInterface` only
consumes plain arrays.

    v0=(0,0,0) v1=(1,0,0) v2=(0,1,0) v3=(0,0,1) v4=(1,1,1)
    tetA = [0,1,2,3]  (det P = +1  -> V = 1/6, already positively oriented)
    tetB = [4,1,2,3]  (det P = -2  -> V = 1/3, negatively oriented as given)

tetA and tetB share face {1,2,3} (interior, incidence 2); the other six
faces are each incident to exactly one tet (boundary, incidence 1):
    {0,2,3}, {0,1,3}, {0,1,2}  (from tetA)
    {4,2,3}, {4,1,3}, {4,1,2}  (from tetB)
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
VOLUME_A = 1.0 / 6.0
VOLUME_B = 1.0 / 3.0

BOUNDARY_FACES = {
    "A_023": (0, 2, 3),
    "A_013": (0, 1, 3),
    "A_012": (0, 1, 2),
    "B_423": (4, 2, 3),
    "B_413": (4, 1, 3),
    "B_412": (4, 1, 2),
}

FULL_SURFACE_TAGS = {
    "PEC_GROUND": np.array([BOUNDARY_FACES["A_023"]]),
    "PEC_LINE": np.array([BOUNDARY_FACES["A_013"]]),
    "PORT_1": np.array([BOUNDARY_FACES["A_012"]]),
    "PORT_2": np.array([BOUNDARY_FACES["B_423"]]),
    "PML_OUTER_PEC": np.array([BOUNDARY_FACES["B_413"]]),
    "PMC_SIDE": np.array([BOUNDARY_FACES["B_412"]]),
}
