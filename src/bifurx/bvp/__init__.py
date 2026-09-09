"""bifurx.bvp: Boundary Value Problem (BVP) collocation solvers for periodic orbits."""

from bifurx.bvp.collocation import (
    CollocationProblem,
    PeriodicOrbitResult,
    collocation_matrices,
    eval_lagrange_basis,
    gauss_legendre_nodes,
    initialize_from_hopf,
    solve_periodic_orbit,
)
from bifurx.bvp.floquet import (
    compute_floquet_multipliers,
    compute_monodromy_condensation,
    compute_monodromy_variational,
)
from bifurx.bvp.mesh import equidistribute_mesh, redistribute_mesh

__all__ = [
    "CollocationProblem",
    "PeriodicOrbitResult",
    "collocation_matrices",
    "compute_floquet_multipliers",
    "compute_monodromy_condensation",
    "compute_monodromy_variational",
    "equidistribute_mesh",
    "eval_lagrange_basis",
    "gauss_legendre_nodes",
    "initialize_from_hopf",
    "redistribute_mesh",
    "solve_periodic_orbit",
]
