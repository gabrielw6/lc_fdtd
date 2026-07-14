"""visualization -- 3D geometry/mesh plotting for sanity-checking the FEM
input, independent of running a full sweep. See `geometry_view` for the two
entry points."""
from .geometry_view import PlottingUnavailableError, plot_geometry, plot_mesh

__all__ = ["PlottingUnavailableError", "plot_geometry", "plot_mesh"]
