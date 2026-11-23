from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bifurx.homotopy import (
    BorderedFactorization,
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
    factor_bordered_matrix,
    init_weighted_norm,
    kantorovich_h,
    pade21_predict,
    solve_bordered_block,
    solve_bordered_direct,
    solve_bordered_system,
    solve_factored_bordered,
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
