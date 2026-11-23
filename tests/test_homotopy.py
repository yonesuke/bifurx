from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bifurx.homotopy import (
    BlockBorderedFactorization,
    BorderedFactorization,
    BorderedOperatorSolver,
    BorderedSolution,
    CorrectorResult,
    TrackerParameters,
    TrackScanResult,
    WeightedNormOptions,
    build_bordered_matrix,
    compute_arclength_taylor3,
    compute_tangent,
    correct_arclength_step,
    estimate_local_error,
    estimate_pade_trust_region,
    factor_bordered_block,
    factor_bordered_matrix,
    init_weighted_norm,
    kantorovich_h,
    make_gmres_bordered_solver,
    pade21_predict,
    solve_bordered_block,
    solve_bordered_direct,
    solve_bordered_system,
    solve_factored_bordered,
    solve_factored_bordered_block,
    tangent_predict,
    track_arclength_scan,
    update_step_size,
    update_weighted_norm,
    weighted_distance,
    weighted_inf_norm,
)


def _fold_fn(u: jax.Array, p: jax.Array) -> jax.Array:
    return jnp.array([p - u[0] ** 2])


def _fold_ju(u: jax.Array, _p: jax.Array) -> jax.Array:
    return jnp.array([[-2.0 * u[0]]])


def _fold_jp(_u: jax.Array, _p: jax.Array) -> jax.Array:
    return jnp.array([1.0])


def _ab_fn(u: jax.Array, p1: jax.Array) -> jax.Array:
    u1, u2 = u[0], u[1]
    e = jnp.exp(u2)
    return jnp.array([-u1 + p1 * (1.0 - u1) * e, -u2 + p1 * 14.0 * (1.0 - u1) * e - 2.0 * u2])


def _ab_ju(u: jax.Array, p1: jax.Array) -> jax.Array:
    return jax.jacobian(lambda u_v: _ab_fn(u_v, p1))(u)


def _ab_jp(u: jax.Array, p1: jax.Array) -> jax.Array:
    return jax.jacobian(lambda p_v: _ab_fn(u, p_v))(p1)


def test_weighted_norm_origin_mixed_scale_and_pytree() -> None:
    """Verify WeightedNorm handles origin initialization, mixed scales, and PyTree round-trip."""
    opts = WeightedNormOptions()
    leaves, treedef = jax.tree_util.tree_flatten(opts)
    assert jax.tree_util.tree_unflatten(treedef, leaves) == opts
    assert opts.replace(scale_min=1e-3).scale_min == 1e-3

    # Origin z=0 should initialize cleanly to scale_abs_min and not blow up on update
    w0 = init_weighted_norm(jnp.zeros(3, dtype=jnp.float64), opts)
    assert jnp.allclose(w0, opts.scale_abs_min)
    w1 = update_weighted_norm(w0, jnp.array([0.0, 0.0, 1e-5]), opts)
    assert bool(jnp.all(jnp.isfinite(w1)))
    assert float(w1[2]) >= opts.scale_abs_min

    # Mixed scale: 1e4 and 1e-4
    z_mixed = jnp.array([1e4, 1e-4])
    w_mixed = init_weighted_norm(z_mixed, opts)
    assert float(w_mixed[0]) == pytest.approx(1e4)
    assert float(w_mixed[1]) == pytest.approx(1.0)
    assert float(weighted_inf_norm(z_mixed, w_mixed)) == pytest.approx(1.0)
    assert float(weighted_distance(z_mixed, jnp.zeros(2), w_mixed)) == pytest.approx(1.0)


def test_bordered_factorization_reuse_and_direct_equivalence() -> None:
    """Verify BorderedFactorization solves multiple RHSs matching direct bordered solves."""
    A = jnp.array([[2.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    b = jnp.array([1.0, 2.0], dtype=jnp.float64)
    c = jnp.array([3.0, 1.0], dtype=jnp.float64)
    d = 4.0

    fac = factor_bordered_matrix(A, b, c, d)
    assert isinstance(fac, BorderedFactorization)
    assert bool(fac.use_lu)

    f1 = jnp.array([5.0, 6.0], dtype=jnp.float64)
    f2 = jnp.array([-1.0, 2.5], dtype=jnp.float64)
    sol_fac_1 = solve_factored_bordered(fac, f1, 7.0)
    sol_fac_2 = solve_factored_bordered(fac, f2, -3.0)
    sol_dir_1 = solve_bordered_direct(A, b, c, d, f1, 7.0)
    sol_dir_2 = solve_bordered_direct(A, b, c, d, f2, -3.0)
    sol_blk_1 = solve_bordered_block(A, b, c, d, f1, 7.0)

    assert isinstance(sol_fac_1, BorderedSolution)
    assert jnp.allclose(sol_fac_1.z, sol_dir_1.z, atol=1e-12)
    assert jnp.allclose(sol_fac_2.z, sol_dir_2.z, atol=1e-12)
    assert jnp.allclose(sol_blk_1.z, sol_dir_1.z, atol=1e-12)
    assert build_bordered_matrix(A, b, c, d).shape == (3, 3)

    # Singular A, invertible bordered matrix M
    A_sing = jnp.array([[1.0, 2.0], [2.0, 4.0]], dtype=jnp.float64)
    sol_auto = solve_bordered_system(
        A_sing, jnp.array([1.0, 0.0]), jnp.array([1.0, 0.0]), 0.0, jnp.array([3.0, 4.0]), 1.0
    )
    assert jnp.allclose(sol_auto.z, jnp.array([1.0, 0.5, 1.0]), atol=1e-10)


def test_pade_taylor_coefficients_and_trust_region() -> None:
    """Verify exact Taylor coefficients, Padé (2,1) accuracy, and straight-line trust region."""
    # 1. Exact Taylor coefficients at fold of p - u^2 = 0: u(s) = s, p(s) = s^2
    u0 = jnp.array([0.0])
    p0 = jnp.array(0.0)
    tau = jnp.array([1.0, 0.0])
    Ju = _fold_ju(u0, p0)
    Jp = _fold_jp(u0, p0)
    z2, z3, tr = compute_arclength_taylor3(_fold_fn, u0, p0, tau, Ju, Jp)
    assert jnp.allclose(z2, jnp.array([0.0, 1.0]), atol=1e-12)
    assert jnp.allclose(z3, jnp.array([0.0, 0.0]), atol=1e-12)
    assert bool(jnp.isinf(tr))

    # 2. Straight-line branch trust region stays inf
    tr_line = estimate_pade_trust_region(jnp.array([0.0, 1.0]), jnp.zeros(2), jnp.zeros(2))
    assert bool(jnp.isinf(tr_line))

    # 3. Rational/nonlinear curve: Padé (2,1) beats first-order tangent prediction
    def exp_fn(u: jax.Array, p: jax.Array) -> jax.Array:
        return jnp.array([u[0] - jnp.exp(p)])

    u_e = jnp.array([1.0])
    p_e = jnp.array(0.0)
    ju_e = jnp.array([[1.0]])
    jp_e = jnp.array([-1.0])
    tau_e = compute_tangent(lambda _u, _p: ju_e, lambda _u, _p: jp_e, u_e, p_e)
    z2_e, z3_e, tr_e = compute_arclength_taylor3(exp_fn, u_e, p_e, tau_e, ju_e, jp_e)
    assert bool(jnp.isfinite(tr_e)) and float(tr_e) > 0.0

    ds = jnp.array(0.15)
    u_tan, p_tan = tangent_predict(u_e, p_e, tau_e, ds)
    u_pad, p_pad = pade21_predict(u_e, p_e, tau_e, z2_e, z3_e, tr_e, ds)
    res_tan = float(jnp.linalg.norm(exp_fn(u_tan, p_tan)))
    res_pad = float(jnp.linalg.norm(exp_fn(u_pad, p_pad)))
    assert res_pad < 1e-2 * res_tan

    err = estimate_local_error(u_pad, p_pad, u_tan, p_tan, ds, 4, jnp.ones(2))
    assert float(err) > 0.0


def test_pade_trust_region_scale_invariance_and_froissart_doublets() -> None:
    """Verify scale invariance, arclength equivariance, cubic inflections, and doublet immunity."""
    # 1. Exact diagonal scaling invariance: (D z_k, D w) -> identical trust region
    z1 = jnp.array([0.8, -0.3, 1.2, -0.5], dtype=jnp.float64)
    z2 = jnp.array([0.4, 0.25, -0.6, 0.1], dtype=jnp.float64)
    z3 = jnp.array([0.2, -0.5, 0.3, 0.05], dtype=jnp.float64)
    w = jnp.array([1.5, 0.5, 2.0, 0.8], dtype=jnp.float64)
    d = jnp.array([1e4, 1e-4, 1e2, 1e-2], dtype=jnp.float64)

    tr = estimate_pade_trust_region(z1, z2, z3, weights=w)
    tr_scaled = estimate_pade_trust_region(d * z1, d * z2, d * z3, weights=d * w)
    assert bool(jnp.isfinite(tr)) and float(tr) > 0.0
    assert float(tr_scaled) == pytest.approx(float(tr), rel=1e-12)

    # 2. Arclength rescaling equivariance: s -> beta * s scales trust region by 1 / beta
    for beta in (1e-4, 7.5, 1e4):
        tr_beta = estimate_pade_trust_region(beta * z1, (beta**2) * z2, (beta**3) * z3, weights=w)
        assert float(tr_beta) == pytest.approx((1.0 / beta) * float(tr), rel=1e-12)

    # 3. High-dimensional Froissart doublet immunity (unit and weighted scales)
    n_dim = 50
    z1_hd = jnp.ones(n_dim, dtype=jnp.float64)
    z2_hd = jnp.zeros(n_dim, dtype=jnp.float64).at[:5].set(1e-9).at[5:10].set(1.0)
    z3_hd = jnp.zeros(n_dim, dtype=jnp.float64).at[:5].set(1.0).at[5:10].set(0.5)
    tr_hd_unit = estimate_pade_trust_region(z1_hd, z2_hd, z3_hd)
    assert float(tr_hd_unit) == pytest.approx(1.0, rel=1e-12)

    w_hd = jnp.geomspace(1e-3, 1e4, n_dim, dtype=jnp.float64)
    tr_hd_weighted = estimate_pade_trust_region(
        w_hd * z1_hd, w_hd * z2_hd, w_hd * z3_hd, weights=w_hd
    )
    assert float(tr_hd_weighted) == pytest.approx(1.0, rel=1e-12)

    # 4. Global pure-cubic inflection returns inf
    tr_cubic = estimate_pade_trust_region(
        jnp.array([1.0, 0.0], dtype=jnp.float64),
        jnp.array([0.0, 0.0], dtype=jnp.float64),
        jnp.array([1.0, 0.0], dtype=jnp.float64),
    )
    assert bool(jnp.isinf(tr_cubic))

    # 5. compute_arclength_taylor3 with diagonal rescaling of exp_fn
    def exp_fn(u: jax.Array, p: jax.Array) -> jax.Array:
        return jnp.array([u[0] - jnp.exp(p)])

    u_e = jnp.array([1.0], dtype=jnp.float64)
    p_e = jnp.array(0.0, dtype=jnp.float64)
    w_e = jnp.array([1.5, 0.7], dtype=jnp.float64)
    w_arc_e = 1.0 / (w_e**2)
    ju_e = jax.jacobian(lambda u_v: exp_fn(u_v, p_e))(u_e)
    jp_e = jax.jacobian(lambda p_v: exp_fn(u_e, p_v))(p_e)
    tau_e = compute_tangent(
        lambda _u, _p: ju_e, lambda _u, _p: jp_e, u_e, p_e, arclength_weights=w_arc_e
    )
    z2_e, z3_e, tr_e = compute_arclength_taylor3(
        exp_fn,
        u_e,
        p_e,
        tau_e,
        ju_e,
        jp_e,
        border_c=w_arc_e[:1] * tau_e[:1],
        border_d=w_arc_e[-1] * tau_e[-1],
        weights=w_e,
    )

    d_u = jnp.array([1e3], dtype=jnp.float64)
    d_p = jnp.array(1e-2, dtype=jnp.float64)
    d_vec = jnp.concatenate([d_u, jnp.atleast_1d(d_p)])

    def exp_scaled(u_tilde: jax.Array, p_tilde: jax.Array) -> jax.Array:
        return exp_fn(u_tilde / d_u, p_tilde / d_p)

    u_t = d_u * u_e
    p_t = d_p * p_e
    w_t = d_vec * w_e
    w_arc_t = 1.0 / (w_t**2)
    ju_t = jax.jacobian(lambda u_v: exp_scaled(u_v, p_t))(u_t)
    jp_t = jax.jacobian(lambda p_v: exp_scaled(u_t, p_v))(p_t)
    tau_t = d_vec * tau_e
    z2_t, z3_t, tr_t = compute_arclength_taylor3(
        exp_scaled,
        u_t,
        p_t,
        tau_t,
        ju_t,
        jp_t,
        border_c=w_arc_t[:1] * tau_t[:1],
        border_d=w_arc_t[-1] * tau_t[-1],
        weights=w_t,
    )
    assert bool(jnp.isfinite(tr_e)) and float(tr_e) > 0.0
    assert jnp.allclose(z2_t, d_vec * z2_e, rtol=1e-12, atol=0.0)
    assert jnp.allclose(z3_t, d_vec * z3_e, rtol=1e-12, atol=0.0)
    assert float(tr_t) == pytest.approx(float(tr_e), rel=1e-12)


def test_corrector_contraction_early_abort_and_stepsize_control() -> None:
    """Verify contraction-rate early abort, machine-precision safety, and Timme stepsize law."""
    params = TrackerParameters()
    leaves, treedef = jax.tree_util.tree_flatten(params)
    assert jax.tree_util.tree_unflatten(treedef, leaves) == params
    assert params.replace(a=0.25).a == 0.25
    assert float(kantorovich_h(0.125)) == pytest.approx(0.1951941016011238, rel=1e-6)

    # 1. Good initial guess converges cleanly and estimates omega
    u_prev = jnp.array([1.0])
    p_prev = jnp.array(1.0)
    tau_prev = compute_tangent(_fold_ju, _fold_jp, u_prev, p_prev, direction=-1.0)
    ds = jnp.array(0.05)
    u_pred, p_pred = tangent_predict(u_prev, p_prev, tau_prev, ds)
    res_ok = correct_arclength_step(
        _fold_fn,
        _fold_ju,
        _fold_jp,
        u_pred,
        p_pred,
        u_prev,
        p_prev,
        tau_prev,
        ds,
        enforce_contraction=True,
    )
    assert isinstance(res_ok, CorrectorResult)
    assert bool(res_ok.converged)
    assert not bool(res_ok.terminated_early)

    # 2. Wildly bad guess on steep exponential triggers early contraction abort on iteration 0
    u_ab = jnp.array([0.0, 0.0])
    p_ab = jnp.array(0.0)
    tau_ab = compute_tangent(_ab_ju, _ab_jp, u_ab, p_ab)
    res_bad = correct_arclength_step(
        _ab_fn,
        _ab_ju,
        _ab_jp,
        u_ab + 5.0 * tau_ab[:2],
        p_ab + 5.0 * tau_ab[2],
        u_ab,
        p_ab,
        tau_ab,
        jnp.array(5.0),
        omega_init=10.0,
        first_step=False,
        enforce_contraction=True,
    )
    assert not bool(res_bad.converged)
    assert bool(res_bad.terminated_early)
    assert int(res_bad.iterations) == 1

    # 3. Rejected step shrinks step size analytically
    ds_next = update_step_size(
        ds=jnp.array(0.1),
        converged=jnp.bool_(False),
        omega=jnp.array(2.0),
        omega_prev=jnp.array(1.0),
        local_error=jnp.array(0.01),
        trust_region=jnp.array(1.0),
        theta=jnp.array(0.4),
        iterations=jnp.int32(2),
        last_step_failed=jnp.bool_(False),
        predictor_order=4,
        ds_min=jnp.array(1e-6),
        ds_max=jnp.array(0.5),
        params=params,
    )
    assert 0.01 < float(ds_next) < 0.1


def test_track_arclength_scan_jit_vmap_and_pade_accuracy() -> None:
    """Verify standalone `track_arclength_scan` works under JIT/vmap for tangent and Padé (2,1)."""
    u0 = jnp.array([0.0, 0.0])
    p0 = jnp.array(0.0)
    tau0 = compute_tangent(_ab_ju, _ab_jp, u0, p0)

    res_tan = jax.jit(
        lambda: track_arclength_scan(
            _ab_fn,
            _ab_ju,
            _ab_jp,
            u0,
            p0,
            tau0,
            ds=0.03,
            ds_max=0.08,
            n_steps=140,
            adaptive_step=True,
            predictor="tangent",
        )
    )()
    res_pad = jax.jit(
        lambda: track_arclength_scan(
            _ab_fn,
            _ab_ju,
            _ab_jp,
            u0,
            p0,
            tau0,
            ds=0.03,
            ds_max=0.08,
            n_steps=140,
            adaptive_step=True,
            predictor="pade21",
        )
    )()
    assert isinstance(res_tan, TrackScanResult)
    assert bool(jnp.all(res_tan.converged))
    assert bool(jnp.all(res_pad.converged))
    # Padé (2,1) should require fewer Newton iterations than 1st-order tangent on smooth S-curve
    assert int(jnp.sum(res_pad.iterations)) <= int(jnp.sum(res_tan.iterations))

    # vmap with Padé (2,1) over a batch of fold starting points
    p_batch = jnp.linspace(0.5, 2.0, 8)
    u_batch = jnp.sqrt(p_batch)[:, None]

    def _run_one(u_start: jax.Array, p_start: jax.Array) -> TrackScanResult:
        tau_s = compute_tangent(_fold_ju, _fold_jp, u_start, p_start, direction=-1.0)
        return track_arclength_scan(
            _fold_fn,
            _fold_ju,
            _fold_jp,
            u_start,
            p_start,
            tau_s,
            ds=0.04,
            n_steps=20,
            adaptive_step=True,
            predictor="pade21",
        )

    batch_res = jax.jit(jax.vmap(_run_one))(u_batch, p_batch)
    assert batch_res.u.shape == (8, 20, 1)
    assert bool(jnp.all(batch_res.converged))
    assert jnp.allclose(batch_res.u[:, :, 0] ** 2, batch_res.p, atol=1e-7)


def test_weighted_arclength_high_dim_metric() -> None:
    """Verify arclength_weights prevents high state dimension from crushing dp/ds (#15)."""
    dim = 100

    def high_dim_fn(u: jax.Array, p: jax.Array) -> jax.Array:
        return u**2 - p

    def high_dim_ju(u: jax.Array, _p: jax.Array) -> jax.Array:
        return jnp.diag(2.0 * u)

    def high_dim_jp(_u: jax.Array, _p: jax.Array) -> jax.Array:
        return -jnp.ones(dim, dtype=jnp.float64)

    u0 = jnp.ones(dim, dtype=jnp.float64)
    p0 = jnp.array(1.0, dtype=jnp.float64)

    tau_unweighted = compute_tangent(high_dim_ju, high_dim_jp, u0, p0)
    # In unweighted Euclidean norm, dp/ds shrinks as O(1/sqrt(dim))
    assert abs(float(tau_unweighted[-1])) < 0.25

    w_arc = jnp.concatenate([jnp.full(dim, 1.0 / dim), jnp.array([1.0])])
    tau_weighted = compute_tangent(
        high_dim_ju, high_dim_jp, u0, p0, arclength_weights=w_arc, param_mode="arclength"
    )
    # In mesh/dim-invariant weighted norm, sum((1/dim) * (0.5)^2) + 1 = 1.25 -> dp/ds = 1/sqrt(1.25)
    assert float(tau_weighted[-1]) == pytest.approx(1.0 / float(jnp.sqrt(1.25)), rel=1e-6)

    res = jax.jit(
        lambda: track_arclength_scan(
            high_dim_fn,
            high_dim_ju,
            high_dim_jp,
            u0,
            p0,
            tau_weighted,
            ds=0.1,
            n_steps=8,
            adaptive_step=True,
            arclength_weights=w_arc,
        )
    )()
    assert bool(jnp.all(res.converged))
    assert float(res.p[-1]) > 1.5
    assert float(jnp.max(res.residual_norms)) < 1e-8
    assert jnp.allclose(res.u**2, res.p[:, None], atol=1e-8)


def test_natural_mode_exact_target_and_chart_accept_hooks() -> None:
    """Verify param_mode='natural' hits p_target exactly and supports chart_fn/accept_fn + pade21."""

    def charted_fold_fn(u: jax.Array, p: jax.Array, ref: jax.Array) -> jax.Array:
        # Auxiliary shift in second coordinate anchored against moving chart reference
        return jnp.array([u[0] ** 2 - p, u[1] - (ref[1] + 0.01 * (u[0] - ref[0]))])

    ju_fn = jax.jacobian(charted_fold_fn, argnums=0)
    jp_fn = jax.jacobian(charted_fold_fn, argnums=1)

    u0 = jnp.array([1.0, 0.0])
    p0 = jnp.array(1.0)
    tau0 = compute_tangent(
        lambda u_v, p_v: ju_fn(u_v, p_v, u0),
        lambda u_v, p_v: jp_fn(u_v, p_v, u0),
        u0,
        p0,
        direction=1.0,
        param_mode="natural",
    )
    assert float(tau0[-1]) == 1.0

    # Reject steps that drop u[0] below threshold via accept_fn (always true here)
    res_nat = jax.jit(
        lambda: track_arclength_scan(
            charted_fold_fn,
            ju_fn,
            jp_fn,
            u0,
            p0,
            tau0,
            ds=0.13,
            ds_max=0.2,
            n_steps=10,
            p_min=0.5,
            p_max=1.35,
            adaptive_step=True,
            predictor="pade21",
            param_mode="natural",
            p_target=1.35,
            chart_fn=lambda u_c, _p_c: u_c,
            accept_fn=lambda u_c, _p_c: u_c[0] > 0.5,
        )
    )()
    p_kept = res_nat.p[res_nat.converged]
    assert int(p_kept.shape[0]) >= 3
    assert float(p_kept[-1]) == 1.35
    assert jnp.allclose(res_nat.u[res_nat.converged, 0] ** 2, p_kept, atol=1e-8)

    # accept_fn returning False forces rejection and shrinks ds until termination
    res_rej = track_arclength_scan(
        charted_fold_fn,
        ju_fn,
        jp_fn,
        u0,
        p0,
        tau0,
        ds=0.1,
        ds_min=0.05,
        n_steps=5,
        adaptive_step=True,
        param_mode="natural",
        p_target=1.5,
        chart_fn=lambda u_c, _p_c: u_c,
        accept_fn=lambda u_c, _p_c: u_c[0] > 10.0,
    )
    assert not bool(jnp.any(res_rej.converged))


def test_corrector_kantorovich_omega_and_singular_rejection() -> None:
    """Verify Kantorovich final check, stiff sub-1e-7 omega estimation, and singular rejection."""
    # 1. Kantorovich + final simplified-Newton accuracy check converges in 1 step in WeightedNorm
    u_prev = jnp.array([1.0], dtype=jnp.float64)
    p_prev = jnp.array(1.0, dtype=jnp.float64)
    tau = compute_tangent(_fold_ju, _fold_jp, u_prev, p_prev, direction=1.0)
    ds = jnp.array(0.05, dtype=jnp.float64)
    u_tan, p_tan = tangent_predict(u_prev, p_prev, tau, ds)
    res_exact = correct_arclength_step(
        _fold_fn,
        _fold_ju,
        _fold_jp,
        u_tan,
        p_tan,
        u_prev,
        p_prev,
        tau,
        ds,
        tol=1e-14,
        step_tol=1e-14,
    )
    u_kant = res_exact.u + jnp.array([5e-5], dtype=jnp.float64)
    p_kant = res_exact.p
    w_kant = init_weighted_norm(jnp.concatenate([u_kant, jnp.atleast_1d(p_kant)]))
    tol_kant = 1e-8
    res_kant = correct_arclength_step(
        _fold_fn,
        _fold_ju,
        _fold_jp,
        u_kant,
        p_kant,
        u_prev,
        p_prev,
        tau,
        ds,
        weights=w_kant,
        omega_init=2.0,
        first_step=False,
        tol=tol_kant,
        step_tol=tol_kant,
        enforce_contraction=True,
    )
    assert bool(res_kant.converged)
    assert not bool(res_kant.terminated_early)
    assert int(res_kant.iterations) == 1
    assert float(res_kant.accuracy) <= tol_kant
    assert float(res_kant.omega) != pytest.approx(0.5, abs=1e-3)

    # 2. Stiff quadratic with prev_sw < 1e-7 still estimates omega above roundoff
    def stiff_fn(u: jax.Array, p: jax.Array) -> jax.Array:
        return jnp.array([p - 1e10 * (u[0] - 1.0) ** 2])

    def stiff_ju(u: jax.Array, _p: jax.Array) -> jax.Array:
        return jnp.array([[-2e10 * (u[0] - 1.0)]])

    def stiff_jp(_u: jax.Array, _p: jax.Array) -> jax.Array:
        return jnp.array([1.0])

    u_sp = jnp.array([1.0 + 1e-8], dtype=jnp.float64)
    p_sp = jnp.array(1e-6, dtype=jnp.float64)
    tau_s = jnp.array([0.0, 1.0], dtype=jnp.float64)
    res_stiff = correct_arclength_step(
        stiff_fn,
        stiff_ju,
        stiff_jp,
        jnp.array([1.0 + 6e-8], dtype=jnp.float64),
        p_sp,
        u_sp,
        p_sp,
        tau_s,
        jnp.array(0.0, dtype=jnp.float64),
        weights=jnp.ones(2, dtype=jnp.float64),
        omega_init=0.0,
        first_step=True,
        tol=1e-10,
        step_tol=1e-10,
        enforce_contraction=False,
        track_contraction=True,
    )
    assert bool(res_stiff.converged)
    assert float(res_stiff.norm_dz0) < 1e-7
    assert float(res_stiff.omega) > 1e6

    # 3. Singular inconsistent system is rejected while consistent rank-deficient system passes
    res_inconsistent = correct_arclength_step(
        lambda _u, _p: jnp.array([1.0]),
        lambda _u, _p: jnp.array([[0.0]]),
        lambda _u, _p: jnp.array([0.0]),
        jnp.array([0.0], dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
        jnp.array([0.0], dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
        jnp.array([0.0, 1.0], dtype=jnp.float64),
        jnp.array(0.0, dtype=jnp.float64),
    )
    assert not bool(res_inconsistent.converged)
    assert bool(res_inconsistent.terminated_early)

    u_rd0 = jnp.array([0.5, 0.5], dtype=jnp.float64)
    p_rd0 = jnp.array(1.0, dtype=jnp.float64)
    tau_rd = jnp.array([0.5, 0.5, 1.0], dtype=jnp.float64)
    tau_rd = tau_rd / jnp.linalg.norm(tau_rd)
    res_consistent = correct_arclength_step(
        lambda u, p: jnp.array([u[0] + u[1] - p, 2.0 * (u[0] + u[1] - p)]),
        lambda _u, _p: jnp.array([[1.0, 1.0], [2.0, 2.0]]),
        lambda _u, _p: jnp.array([-1.0, -2.0]),
        u_rd0 + jnp.array([0.1, 0.0], dtype=jnp.float64),
        p_rd0,
        u_rd0,
        p_rd0,
        tau_rd,
        jnp.array(0.05, dtype=jnp.float64),
    )
    assert bool(res_consistent.converged)
    assert float(res_consistent.residual_norm) < 1e-8


def test_bordered_operator_solver_gmres_and_lu_reuse() -> None:
    """Verify matrix-free GMRES BorderedOperatorSolver and LU factorization reuse in corrector."""
    # 1. BlockBorderedFactorization matches dense direct solve
    u_ab0 = jnp.array([0.0, 0.0], dtype=jnp.float64)
    p_ab0 = jnp.array(0.0, dtype=jnp.float64)
    tau_ab = compute_tangent(_ab_ju, _ab_jp, u_ab0, p_ab0)
    fac_blk = factor_bordered_block(
        _ab_ju(u_ab0, p_ab0), _ab_jp(u_ab0, p_ab0), tau_ab[:2], tau_ab[2]
    )
    assert isinstance(fac_blk, BlockBorderedFactorization)
    sol_blk = solve_factored_bordered_block(fac_blk, jnp.array([0.2, -0.3]), 0.5)
    sol_dir = solve_bordered_direct(
        _ab_ju(u_ab0, p_ab0),
        _ab_jp(u_ab0, p_ab0),
        tau_ab[:2],
        tau_ab[2],
        jnp.array([0.2, -0.3]),
        0.5,
    )
    assert jnp.allclose(sol_blk.z, sol_dir.z, atol=1e-12)

    # 2. Matrix-free GMRES BorderedOperatorSolver for tangent, pade21 taylor3, and scan (jit + vmap)
    gmres_solver = make_gmres_bordered_solver(_ab_fn, tol=1e-11)
    tau_gmres = compute_tangent(u0=u_ab0, p0=p_ab0, bordered_solver=gmres_solver)
    assert jnp.allclose(tau_gmres, tau_ab, atol=1e-8)

    z2_d, z3_d, tr_d = compute_arclength_taylor3(
        _ab_fn, u_ab0, p_ab0, tau_ab, _ab_ju(u_ab0, p_ab0), _ab_jp(u_ab0, p_ab0)
    )
    z2_g, z3_g, tr_g = compute_arclength_taylor3(
        _ab_fn, u_ab0, p_ab0, tau_gmres, bordered_solver=gmres_solver
    )
    assert jnp.allclose(z2_g, z2_d, atol=1e-7)
    assert jnp.allclose(z3_g, z3_d, atol=1e-7)
    assert float(tr_g) == pytest.approx(float(tr_d), rel=1e-6)

    res_dense = jax.jit(
        lambda: track_arclength_scan(
            _ab_fn,
            _ab_ju,
            _ab_jp,
            u_ab0,
            p_ab0,
            tau_ab,
            ds=0.03,
            ds_max=0.06,
            n_steps=40,
            adaptive_step=True,
            predictor="pade21",
        )
    )()
    res_gmres = jax.jit(
        lambda: track_arclength_scan(
            _ab_fn,
            u0=u_ab0,
            p0=p_ab0,
            tau0=tau_gmres,
            ds=0.03,
            ds_max=0.06,
            n_steps=40,
            adaptive_step=True,
            predictor="pade21",
            bordered_solver=gmres_solver,
        )
    )()
    assert bool(jnp.all(res_gmres.converged))
    assert jnp.allclose(res_gmres.u, res_dense.u, atol=1e-7)
    assert jnp.allclose(res_gmres.p, res_dense.p, atol=1e-7)

    # Also verify vmap with GMRES bordered solver
    p_batch = jnp.linspace(0.5, 2.0, 6)
    u_batch = jnp.sqrt(p_batch)[:, None]
    fold_gmres = make_gmres_bordered_solver(_fold_fn, tol=1e-11)

    def _run_gmres_fold(u_s: jax.Array, p_s: jax.Array) -> TrackScanResult:
        tau_s = compute_tangent(u0=u_s, p0=p_s, direction=-1.0, bordered_solver=fold_gmres)
        return track_arclength_scan(
            _fold_fn,
            u0=u_s,
            p0=p_s,
            tau0=tau_s,
            ds=0.04,
            n_steps=15,
            adaptive_step=True,
            predictor="pade21",
            bordered_solver=fold_gmres,
        )

    batch_g = jax.jit(jax.vmap(_run_gmres_fold))(u_batch, p_batch)
    assert bool(jnp.all(batch_g.converged))
    assert jnp.allclose(batch_g.u[:, :, 0] ** 2, batch_g.p, atol=1e-7)

    # 3. Verify Kantorovich _do_final_check reuses factorization without re-factorizing (#25 note)
    counts = {"factor": 0, "solve": 0}

    def counting_factor(
        u: jax.Array, p: jax.Array, c: jax.Array, d: jax.Array
    ) -> BorderedFactorization:
        counts["factor"] += 1
        return factor_bordered_matrix(_fold_ju(u, p), _fold_jp(u, p), c, d)

    def counting_solve(fac: BorderedFactorization, f: jax.Array, g: jax.Array) -> BorderedSolution:
        counts["solve"] += 1
        return solve_factored_bordered(fac, f, g)

    counting_solver = BorderedOperatorSolver(factor=counting_factor, solve=counting_solve)
    u_prev = jnp.array([1.0], dtype=jnp.float64)
    p_prev = jnp.array(1.0, dtype=jnp.float64)
    tau = compute_tangent(_fold_ju, _fold_jp, u_prev, p_prev, direction=1.0)
    ds = jnp.array(0.05, dtype=jnp.float64)
    u_tan, p_tan = tangent_predict(u_prev, p_prev, tau, ds)
    res_exact = correct_arclength_step(
        _fold_fn,
        _fold_ju,
        _fold_jp,
        u_tan,
        p_tan,
        u_prev,
        p_prev,
        tau,
        ds,
        tol=1e-14,
        step_tol=1e-14,
    )
    u_kant = res_exact.u + jnp.array([5e-5], dtype=jnp.float64)
    p_kant = res_exact.p
    w_kant = init_weighted_norm(jnp.concatenate([u_kant, jnp.atleast_1d(p_kant)]))

    with jax.disable_jit():
        res_reuse = correct_arclength_step(
            _fold_fn,
            u_init=u_kant,
            p_init=p_kant,
            u_prev=u_prev,
            p_prev=p_prev,
            tau_prev=tau,
            ds=ds,
            weights=w_kant,
            omega_init=2.0,
            first_step=False,
            tol=1e-8,
            step_tol=1e-8,
            enforce_contraction=True,
            bordered_solver=counting_solver,
        )
    assert bool(res_reuse.converged)
    assert int(res_reuse.iterations) == 1
    assert counts["factor"] == 1
    assert counts["solve"] == 2
