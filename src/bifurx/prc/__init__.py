"""bifurx.prc: Modern Adjoint BVP Collocation for infinitesimal Phase Response Curves (iPRC)."""

from bifurx.prc.adjoint import (
    IPRCContinuationResult,
    IPRCResult,
    compute_iprc,
    compute_iprc_from_collocation,
    continuation_iprc,
    find_zero_crossings,
    solve_coupled_orbit_prc,
)

__all__ = [
    "IPRCContinuationResult",
    "IPRCResult",
    "compute_iprc",
    "compute_iprc_from_collocation",
    "continuation_iprc",
    "find_zero_crossings",
    "solve_coupled_orbit_prc",
]
