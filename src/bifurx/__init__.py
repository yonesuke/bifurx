"""bifurx: Numerical continuation and bifurcation analysis in JAX.

A modern reimagining of AUTO-07P.
"""

from bifurx.bifurcations.switch import compute_bifurcation_tangents, switch_branch
from bifurx.bifurcations.test_funcs import (
    bialternate_matrix,
    refine_bifurcation_point,
    refine_fold_moore_spence,
    test_func_branch_point,
    test_func_fold,
    test_func_hopf,
)
from bifurx.bvp.collocation import (
    CollocationProblem,
    PeriodicOrbitResult,
    PeriodicOrbitResultJax,
    collocation_matrices,
    gauss_legendre_nodes,
    initialize_from_hopf,
    solve_periodic_orbit,
    solve_periodic_orbit_jax,
)
from bifurx.bvp.floquet import (
    compute_floquet_multipliers,
    compute_monodromy_condensation,
    compute_monodromy_variational,
)
from bifurx.bvp.mesh import equidistribute_mesh, redistribute_mesh
from bifurx.codim2.moore_spence import (
    Codim2Result,
    continuation_codim2,
)
from bifurx.continuation.arclength import (
    BifurcationPoint,
    ContinuationResult,
    ContinuationScanResult,
    compute_initial_tangent,
    continuation,
    continuation_scan,
)
from bifurx.plot.diagram import plot_diagram
from bifurx.prc.adjoint import (
    IPRCContinuationResult,
    IPRCResult,
    compute_iprc,
    compute_iprc_from_collocation,
    continuation_iprc,
    solve_coupled_orbit_prc,
)
from bifurx.problem import BifurcationProblem
from bifurx.solvers.bordered import (
    BorderedSolution,
    solve_bordered_block,
    solve_bordered_direct,
    solve_bordered_system,
)
from bifurx.solvers.newton import (
    NewtonResult,
    NewtonResultJax,
    bordered_newton_solve,
    bordered_newton_solve_jax,
)

__version__ = "0.1.0"

__all__ = [
    "BifurcationPoint",
    "BifurcationProblem",
    "BorderedSolution",
    "Codim2Result",
    "CollocationProblem",
    "ContinuationResult",
    "ContinuationScanResult",
    "IPRCContinuationResult",
    "IPRCResult",
    "NewtonResult",
    "NewtonResultJax",
    "PeriodicOrbitResult",
    "PeriodicOrbitResultJax",
    "bialternate_matrix",
    "bordered_newton_solve",
    "bordered_newton_solve_jax",
    "collocation_matrices",
    "compute_bifurcation_tangents",
    "compute_floquet_multipliers",
    "compute_initial_tangent",
    "compute_iprc",
    "compute_iprc_from_collocation",
    "compute_monodromy_condensation",
    "compute_monodromy_variational",
    "continuation",
    "continuation_codim2",
    "continuation_iprc",
    "continuation_scan",
    "equidistribute_mesh",
    "gauss_legendre_nodes",
    "initialize_from_hopf",
    "plot_diagram",
    "redistribute_mesh",
    "refine_bifurcation_point",
    "refine_fold_moore_spence",
    "solve_bordered_block",
    "solve_bordered_direct",
    "solve_bordered_system",
    "solve_coupled_orbit_prc",
    "solve_periodic_orbit",
    "solve_periodic_orbit_jax",
    "switch_branch",
    "test_func_branch_point",
    "test_func_fold",
    "test_func_hopf",
]
