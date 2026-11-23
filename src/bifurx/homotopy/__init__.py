"""bifurx.homotopy: Standalone pure-JAX homotopy & arclength path-tracking core."""

from bifurx.homotopy.bordered import (
    BlockBorderedFactorization,
    BorderedFactorization,
    BorderedOperatorSolver,
    BorderedSolution,
    build_bordered_matrix,
    factor_bordered_block,
    factor_bordered_matrix,
    make_gmres_bordered_solver,
    solve_bordered_block,
    solve_bordered_direct,
    solve_bordered_system,
    solve_factored_bordered,
    solve_factored_bordered_block,
)
from bifurx.homotopy.corrector import CorrectorResult, correct_arclength_step
from bifurx.homotopy.norm import (
    WeightedNormOptions,
    init_weighted_norm,
    update_weighted_norm,
    weighted_distance,
    weighted_inf_norm,
)
from bifurx.homotopy.predictor import (
    compute_arclength_taylor3,
    estimate_local_error,
    estimate_pade_trust_region,
    pade21_predict,
    tangent_predict,
)
from bifurx.homotopy.stepsize import (
    TrackerParameters,
    kantorovich_h,
    update_step_size,
)
from bifurx.homotopy.tracker import (
    TrackScanResult,
    compute_tangent,
    track_arclength_scan,
)

__all__ = [
    "BlockBorderedFactorization",
    "BorderedFactorization",
    "BorderedOperatorSolver",
    "BorderedSolution",
    "CorrectorResult",
    "TrackScanResult",
    "TrackerParameters",
    "WeightedNormOptions",
    "build_bordered_matrix",
    "compute_arclength_taylor3",
    "compute_tangent",
    "correct_arclength_step",
    "estimate_local_error",
    "estimate_pade_trust_region",
    "factor_bordered_block",
    "factor_bordered_matrix",
    "init_weighted_norm",
    "kantorovich_h",
    "make_gmres_bordered_solver",
    "pade21_predict",
    "solve_bordered_block",
    "solve_bordered_direct",
    "solve_bordered_system",
    "solve_factored_bordered",
    "solve_factored_bordered_block",
    "tangent_predict",
    "track_arclength_scan",
    "update_step_size",
    "update_weighted_norm",
    "weighted_distance",
    "weighted_inf_norm",
]
