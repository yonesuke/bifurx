"""bifurx.bvp: Boundary Value Problem (BVP) collocation solvers for periodic orbits."""

from bifurx.bvp.collocation import (
    CollocationProblem,
    PeriodicOrbitResult,
    branch_from_hopf,
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
from bifurx.bvp.floquet import (
    compute_floquet_exponents,
    compute_floquet_multipliers,
    evaluate_floquet_diagnostics,
    evaluate_floquet_diagnostics_from_factors,
    monodromy_factors_condensation,
    periodic_real_schur,
    periodic_schur_eigenvalues,
)
from bifurx.bvp.mesh import redistribute_mesh
from bifurx.bvp.periodic_bifurcations import (
    branch_from_period_doubling,
    refine_periodic_bifurcation,
)
from bifurx.bvp.periodic_codim2 import (
    FixedPeriodContinuationResult,
    PeriodicCodim2Result,
    continue_fixed_period_2p,
    continue_periodic_bifurcation_2p,
)

__all__ = [
    "CollocationProblem",
    "CondensedCollocationFactor",
    "FixedPeriodContinuationResult",
    "PeriodicCodim2Result",
    "PeriodicOrbitResult",
    "branch_from_hopf",
    "branch_from_period_doubling",
    "collocation_matrices",
    "compute_floquet_exponents",
    "compute_floquet_multipliers",
    "continue_fixed_period_2p",
    "continue_periodic_bifurcation_2p",
    "continue_periodic_orbit",
    "evaluate_floquet_diagnostics",
    "evaluate_floquet_diagnostics_from_factors",
    "factor_condensed_collocation",
    "initialize_from_hopf",
    "make_collocation_bordered_solver",
    "monodromy_factors_condensation",
    "periodic_orbit_arclength_weights",
    "periodic_real_schur",
    "periodic_schur_eigenvalues",
    "redistribute_mesh",
    "refine_periodic_bifurcation",
    "solve_condensed_bordered",
    "solve_condensed_collocation",
    "solve_periodic_orbit",
]
