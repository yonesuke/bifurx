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
from bifurx.continuation.arclength import (
    BifurcationPoint,
    ContinuationResult,
    compute_initial_tangent,
    continuation,
)
from bifurx.plot.diagram import plot_diagram
from bifurx.problem import BifurcationProblem
from bifurx.solvers.bordered import (
    BorderedSolution,
    solve_bordered_block,
    solve_bordered_direct,
    solve_bordered_system,
)
from bifurx.solvers.newton import NewtonResult, bordered_newton_solve

__version__ = "0.1.0"

__all__ = [
    "BifurcationPoint",
    "BifurcationProblem",
    "BorderedSolution",
    "ContinuationResult",
    "NewtonResult",
    "bialternate_matrix",
    "bordered_newton_solve",
    "compute_bifurcation_tangents",
    "compute_initial_tangent",
    "continuation",
    "plot_diagram",
    "refine_bifurcation_point",
    "refine_fold_moore_spence",
    "solve_bordered_block",
    "solve_bordered_direct",
    "solve_bordered_system",
    "switch_branch",
    "test_func_branch_point",
    "test_func_fold",
    "test_func_hopf",
]
