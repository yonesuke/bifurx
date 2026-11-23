"""bifurx: Numerical continuation and bifurcation analysis in JAX.

A modern reimagining of AUTO-07P.
"""

import logging

import jax

_logger = logging.getLogger("bifurx")

# Enable 64-bit precision in JAX by default for numerical continuation.
if not getattr(jax.config, "jax_enable_x64", False):
    jax.config.update("jax_enable_x64", True)
    _logger.info("bifurx enabled jax_enable_x64=True for double precision continuation.")

from bifurx.bifurcations import (
    compute_first_lyapunov_coefficient,
    evaluate_point_diagnostics,
    extract_leading_hopf_pair,
    refine_bifurcation_point,
    switch_branch,
    test_func_hopf,
)
from bifurx.bvp.collocation import (
    CollocationProblem,
    PeriodicOrbitResult,
    branch_from_hopf,
    collocation_matrices,
    continue_periodic_orbit,
    initialize_from_hopf,
    solve_periodic_orbit,
)
from bifurx.bvp.floquet import compute_floquet_multipliers
from bifurx.bvp.mesh import redistribute_mesh
from bifurx.codim2 import (
    Codim2Point,
    Codim2Result,
    continuation_codim2,
)
from bifurx.continuation import (
    BifurcationPoint,
    ContinuationResult,
    ContinuationScanResult,
    NewtonResult,
    bordered_newton_solve,
    compute_initial_tangent,
    continuation,
    continuation_scan,
)
from bifurx.homotopy import (
    BorderedSolution,
    TrackerParameters,
    WeightedNormOptions,
    solve_bordered_block,
    solve_bordered_direct,
    solve_bordered_system,
)
from bifurx.options import (
    CollocationOptions,
    ContinuationOptions,
    NewtonOptions,
)
from bifurx.prc import (
    IPRCContinuationResult,
    IPRCResult,
    compute_iprc,
    continuation_iprc,
)
from bifurx.problem import BifurcationProblem

__version__ = "0.1.0"

__all__ = [
    "BifurcationPoint",
    "BifurcationProblem",
    "BorderedSolution",
    "Codim2Point",
    "Codim2Result",
    "CollocationOptions",
    "CollocationProblem",
    "ContinuationOptions",
    "ContinuationResult",
    "ContinuationScanResult",
    "IPRCContinuationResult",
    "IPRCResult",
    "NewtonOptions",
    "NewtonResult",
    "PeriodicOrbitResult",
    "TrackerParameters",
    "WeightedNormOptions",
    "bordered_newton_solve",
    "branch_from_hopf",
    "collocation_matrices",
    "compute_first_lyapunov_coefficient",
    "compute_floquet_multipliers",
    "compute_initial_tangent",
    "compute_iprc",
    "continue_periodic_orbit",
    "continuation",
    "continuation_codim2",
    "continuation_iprc",
    "continuation_scan",
    "evaluate_point_diagnostics",
    "extract_leading_hopf_pair",
    "initialize_from_hopf",
    "redistribute_mesh",
    "refine_bifurcation_point",
    "solve_bordered_block",
    "solve_bordered_direct",
    "solve_bordered_system",
    "solve_periodic_orbit",
    "switch_branch",
    "test_func_hopf",
]
