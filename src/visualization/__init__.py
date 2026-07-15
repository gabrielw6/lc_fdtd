"""visualization -- 3D geometry/mesh plotting for sanity-checking the FEM
input, independent of running a full sweep. See `geometry_view` for the
structure/mesh views, `port_field` for the port-mode field diagnostic,
and `field_slice` for a volume field slice overlaid on the structure."""
from .field_slice import plot_field_slice
from .geometry_view import PlottingUnavailableError, plot_geometry, plot_mesh
from .port_field import plot_port_mode

__all__ = [
    "PlottingUnavailableError",
    "plot_geometry",
    "plot_mesh",
    "plot_port_mode",
    "plot_field_slice",
]
