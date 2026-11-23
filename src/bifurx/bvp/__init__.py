"""bifurx.bvp: Boundary Value Problem (BVP) collocation solvers for periodic orbits."""

from bifurx.bvp.collocation import (
    CollocationProblem,
    PeriodicOrbitResult,
    collocation_matrices,
    continue_periodic_orbit,
    initialize_from_hopf,
    periodic_orbit_arclength_weights,
    solve_periodic_orbit,
)
from bifurx.bvp.condensation import (
    CondensedCollocationFactor,
    factor_condensed_collocation,
    make_collocation_bordered_solver,
    solve_condensed_bordered,
    solve_condensed_collocation,
)
from bifurx.bvp.floquet import compute_floquet_multipliers, evaluate_floquet_diagnostics
from bifurx.bvp.mesh import redistribute_mesh

__all__ = [
    "CollocationProblem",
    "CondensedCollocationFactor",
    "PeriodicOrbitResult",
    "collocation_matrices",
    "compute_floquet_multipliers",
    "continue_periodic_orbit",
    "evaluate_floquet_diagnostics",
    "factor_condensed_collocation",
    "initialize_from_hopf",
    "make_collocation_bordered_solver",
    "periodic_orbit_arclength_weights",
    "redistribute_mesh",
    "solve_condensed_bordered",
    "solve_condensed_collocation",
    "solve_periodic_orbit",
]
