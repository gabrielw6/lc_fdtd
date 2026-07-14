"""solve -- Module 6: global system assembly and the frequency sweep
(docs/module6_solve_sweep_equations.md).

Depends on `fem.assemble`, `ports` (`PortModeSolver`, `build_B`, `build_g`,
`mode_similarity`), `pml.PMLMaterial`, `material.MaterialAssembly`, and
`mesh_interface.MeshInterface` -- the architecture's own dependency graph
has all five feeding this module. Never imports `geometry_builder` or
`meshing`; Module 7 is the sole downstream consumer of `SweepResult`.
"""
from .sweep import (
    ModeTrackingError,
    SweepPreconditionError,
    SweepResult,
    TrackingState,
    check_starting_frequency_precondition,
    run_sweep,
    track_modes,
)
from .system import (
    Factorization,
    SolveSingularityError,
    SystemSymmetryError,
    build_restriction,
    factor,
    recover_solution,
    reduce_system,
    solve_with_factorization,
)

__all__ = [
    "build_restriction",
    "reduce_system",
    "recover_solution",
    "factor",
    "solve_with_factorization",
    "Factorization",
    "SolveSingularityError",
    "SystemSymmetryError",
    "TrackingState",
    "track_modes",
    "check_starting_frequency_precondition",
    "run_sweep",
    "SweepResult",
    "ModeTrackingError",
    "SweepPreconditionError",
]
