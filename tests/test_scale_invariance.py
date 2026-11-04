from __future__ import annotations

import jax.numpy as jnp

from bifurx.bifurcations.test_funcs import test_func_hopf as hopf_test_func
from bifurx.continuation.arclength import BifurcationPoint, continuation
from bifurx.options import ContinuationOptions, NewtonOptions
from bifurx.problem import BifurcationProblem
from bifurx.solvers.newton import bordered_newton_solve


def f_large_scale_pitchfork(u: jnp.ndarray, p: float | jnp.ndarray) -> jnp.ndarray:
    """Pitchfork shifted to large state norm u ~ 1e4."""
    shift = 1e4
    v = u[0] - shift
    return jnp.array([p * v - v**3])


def f_small_param_fold(u: jnp.ndarray, p: float | jnp.ndarray) -> jnp.ndarray:
    """Fold bifurcation located in a small-scale parameter range p in [0, 1e-4]."""
    p_star = 5e-5
    return jnp.array([(p - p_star) - u[0] ** 2])


def f_small_param_pitchfork(u: jnp.ndarray, p: float | jnp.ndarray) -> jnp.ndarray:
    """Pitchfork bifurcation located in a small-scale parameter range p in [0, 1e-4]."""
    p_star = 5e-5
    return jnp.array([(p - p_star) * 1e4 * u[0] - u[0] ** 3])


def test_bordered_newton_large_scale_convergence():
    """Verify bordered Newton achieves scale-invariant convergence for large state norm u ~ 1e4."""
    prob = BifurcationProblem(f_large_scale_pitchfork, u0=[10000.0], p0=0.0)

    u_prev = jnp.array([10000.0])
    p_prev = jnp.array(0.0)
    tau_prev = jnp.array([0.0, 1.0])
    ds = 0.05
    u_init = u_prev + ds * tau_prev[:1]
    p_init = p_prev + ds * tau_prev[-1]

    # Solver should converge cleanly using relative norm tolerance
    res = bordered_newton_solve(
        prob,
        u_init=u_init,
        p_init=p_init,
        u_prev=u_prev,
        p_prev=p_prev,
        tau_prev=tau_prev,
        ds=ds,
    )
    assert bool(res.converged)
    assert abs(float(res.u[0]) - 10000.0) < 1e-6
    assert abs(float(res.p) - 0.05) < 1e-6


def test_continuation_large_scale_system():
    """Verify continuation and bifurcation detection on a system with u ~ 1e4."""
    prob = BifurcationProblem(f_large_scale_pitchfork, u0=[10000.0], p0=-0.5)

    opts = ContinuationOptions(
        ds=0.05,
        max_steps=25,
        newton=NewtonOptions(tol=1e-8, rel_tol=1e-8),
    )
    res = continuation(problem=prob, options=opts)

    assert res.num_points >= 15
    # Verify branch point detected at p = 0, u = 10000
    bps = res.get_bifurcations("BP")
    assert len(bps) >= 1
    bp = bps[0]
    assert abs(bp.p - 0.0) < 1e-3
    assert abs(float(bp.u[0]) - 10000.0) < 1e-3


def test_continuation_small_scale_parameter_range():
    """Verify continuation and bifurcation detection for tiny parameter scales p in [0, 1e-4]."""
    # 1. Fold bifurcation at p = 5e-5, u = 0.0
    prob_fold = BifurcationProblem(f_small_param_fold, u0=[2e-5], p0=5e-5 + 4e-10)
    opts_fold = ContinuationOptions(
        ds=1e-5,
        max_steps=10,
        direction=-1.0,
        bif_p_tol=1e-4,
    )
    res_fold = continuation(problem=prob_fold, options=opts_fold)

    assert res_fold.num_points >= 5
    lps = res_fold.get_bifurcations("LP")
    assert len(lps) == 1
    lp = lps[0]
    assert abs(lp.p - 5e-5) < 1e-6
    assert abs(float(lp.u[0])) < 1e-4

    # 2. Pitchfork branch point at p = 5e-5, u = 0.0
    prob_bp = BifurcationProblem(
        f_small_param_pitchfork,
        u0=[0.0],
        p0=0.0,
        param_bounds=(0.0, 1e-4),
    )
    opts_bp = ContinuationOptions(
        ds=1e-5,
        max_steps=15,
        bif_p_tol=1e-4,
    )
    res_bp = continuation(problem=prob_bp, options=opts_bp)
    assert res_bp.num_points >= 10
    bps = res_bp.get_bifurcations("BP")
    assert len(bps) == 1
    bp = bps[0]
    assert abs(bp.p - 5e-5) < 1e-6
    assert abs(float(bp.u[0])) < 1e-6


def test_deduplication_large_state_norm():
    """Verify de-duplication correctly merges close bifurcation points at large state norm u ~ 1e5."""
    opts = ContinuationOptions(bif_p_tol=1e-4, bif_u_tol=1e-3)

    # Two points close in relative norm (diff = 0.5 on ||u|| ~ 1e5 is relative diff 5e-6)
    bp1 = BifurcationPoint(
        bif_type="LP",
        u=jnp.array([100000.0, 50000.0]),
        p=2.0,
        step=5,
        tangent=jnp.array([0.0, 0.0, 1.0]),
    )
    bp2 = BifurcationPoint(
        bif_type="LP",
        u=jnp.array([100000.5, 50000.2]),
        p=2.0001,
        step=6,
        tangent=jnp.array([0.0, 0.0, 1.0]),
    )

    # Check the relative scale-invariant formula
    p_diff = abs(bp2.p - bp1.p)
    u_diff = float(jnp.linalg.norm(bp2.u - bp1.u))

    p_thresh = opts.bif_p_tol * (1.0 + abs(bp1.p))
    u_thresh = opts.bif_u_tol * (1.0 + float(jnp.linalg.norm(bp1.u)))

    # Without scale-invariance, absolute check u_diff < 1e-3 would fail (u_diff ≈ 0.54)
    assert u_diff > 1e-3, "Absolute check would fail to deduplicate"
    # With scale-invariance, relative check succeeds
    assert p_diff < p_thresh
    assert u_diff < u_thresh


def test_hopf_test_function_scale_invariance():
    """Verify Hopf imaginary component test function handles large and small eigenvalues gracefully."""
    # 1. Large complex eigenvalues: lambda = 0.0 +/- 1e5 * i
    Ju_large = jnp.array([[0.0, 1e5], [-1e5, 0.0]])
    hb_val_large = hopf_test_func(Ju_large)
    assert jnp.isfinite(hb_val_large)
    assert abs(float(hb_val_large)) < 1e-6

    # 2. Large real eigenvalues with small numerical imaginary noise: lambda = 1e5 +/- 0.1 * i
    # Since 0.1 < 1e-4 * (1.0 + 1e5) = 10.0001, this noise is correctly filtered out
    eigs_noisy = jnp.array([1e5 + 0.1j, 1e5 - 0.1j])
    hb_val_noisy = hopf_test_func(Ju_large, eigs=eigs_noisy)
    assert jnp.isnan(hb_val_noisy)

    # 3. Standard complex eigenvalues: lambda = -1.5 +/- 3.0 * i
    Ju_std = jnp.array([[-1.5, 3.0], [-3.0, -1.5]])
    hb_val_std = hopf_test_func(Ju_std)
    assert jnp.isfinite(hb_val_std)
    assert abs(float(hb_val_std) - (-1.5)) < 1e-6

    # 4. Purely real eigenvalues
    Ju_real = jnp.array([[2.0, 0.0], [0.0, -3.0]])
    hb_val_real = hopf_test_func(Ju_real)
    assert jnp.isnan(hb_val_real)
