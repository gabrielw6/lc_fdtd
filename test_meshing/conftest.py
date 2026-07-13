"""Shared fixtures for the meshing test suite. Gmsh is a stateful, global
(single active model at a time) C++ library wrapped via SWIG -- every test
that touches it gets its own initialize()/finalize() cycle and a freshly
named model, so tests never see another test's leftover entities."""
import itertools

import pytest

gmsh = pytest.importorskip("gmsh")

_model_counter = itertools.count()


@pytest.fixture
def gmsh_model():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # quiet Gmsh's own console logging in test output
    gmsh.model.add(f"test_{next(_model_counter)}")
    try:
        yield gmsh.model
    finally:
        gmsh.finalize()
