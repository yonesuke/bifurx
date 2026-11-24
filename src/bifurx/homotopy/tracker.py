from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.homotopy.bordered import BorderedOperatorSolver, solve_bordered_system
from bifurx.homotopy.corrector import correct_arclength_step
from bifurx.homotopy.norm import (
    WeightedNormOptions,
    init_weighted_norm,
    update_weighted_norm,
)
from bifurx.homotopy.predictor import (
    compute_arclength_taylor3,
    estimate_local_error,
    pade21_predict,
    tangent_predict,
)
from bifurx.homotopy.stepsize import TrackerParameters, update_step_size


class TrackScanResult(NamedTuple):
    """Result of pure-JAX pseudo-arclength or natural path tracking via `jax.lax.scan`."""

    u: jax.Array
    p: jax.Array
    tangents: jax.Array
    converged: jax.Array
    step_sizes: jax.Array
    iterations: jax.Array
    residual_norms: jax.Array
    omega: jax.Array
    trust_regions: jax.Array
    observer: Any


def _normalize_tangent(tau: jax.Array, w_arc: jax.Array | None = None) -> jax.Array:
    if w_arc is None:
        norm_val = jnp.linalg.norm(tau)
    else:
        norm_val = jnp.sqrt(jnp.maximum(jnp.dot(w_arc * tau, tau), 0.0))
    return jnp.where(norm_val > 1e-12, tau / norm_val, tau)


def compute_tangent(
    jac_u_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    jac_p_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    u0: ArrayLike | None = None,
    p0: float | jax.Array = 0.0,
    direction: float | jax.Array = 1.0,
    initial_tangent: jax.Array | None = None,
    arclength_weights: jax.Array | None = None,
    param_mode: str = "arclength",
    bordered_solver: BorderedOperatorSolver | None = None,
) -> jax.Array:
    """Compute or normalize the initial tangent vector `(du/ds, dp/ds)`."""
    if param_mode not in ("arclength", "natural"):
        msg = f"Unknown param_mode: {param_mode}. Choose from 'arclength' or 'natural'."
        raise ValueError(msg)

    w_arc = None if arclength_weights is None else jnp.asarray(arclength_weights, dtype=jnp.float64)
    if initial_tangent is not None:
        tau_in = jnp.asarray(initial_tangent, dtype=jnp.float64)
        return tau_in if param_mode == "natural" else _normalize_tangent(tau_in, w_arc)

    if u0 is None:
        msg = "u0 must be provided when initial_tangent is None."
        raise ValueError(msg)
    if bordered_solver is None and (jac_u_fn is None or jac_p_fn is None):
        msg = "Provide jac_u_fn and jac_p_fn when bordered_solver is None."
        raise ValueError(msg)

    u0_arr = jnp.asarray(u0, dtype=jnp.float64)
    p0_arr = jnp.squeeze(jnp.asarray(p0, dtype=jnp.float64))
    n = u0_arr.shape[0]
    dir_sgn = jnp.where(jnp.asarray(direction, dtype=jnp.float64) < 0.0, -1.0, 1.0)
    zero_u = jnp.zeros(n, dtype=jnp.float64)
    one_f = jnp.array(1.0, dtype=jnp.float64)

    if bordered_solver is not None:
        fac_op = bordered_solver.factor(u0_arr, p0_arr, zero_u, dir_sgn)
        tau_raw = bordered_solver.solve(fac_op, zero_u, one_f).z
        if param_mode == "natural":
            return tau_raw
        tau_aligned = tau_raw * jnp.where(tau_raw[-1] * dir_sgn < 0.0, -1.0, 1.0)
        tau_op = _normalize_tangent(tau_aligned, w_arc)
        if jac_u_fn is None or jac_p_fn is None:
            return tau_op

        def _fallback_svd() -> jax.Array:
            Ju_fb = jac_u_fn(u0_arr, p0_arr)
            Jp_fb = jac_p_fn(u0_arr, p0_arr)
            J_fb = jnp.hstack([Ju_fb, Jp_fb.reshape(-1, 1)])
            _, _, vh_fb = jnp.linalg.svd(J_fb)
            tau_fb = vh_fb[-1, :]
            tau_fb = tau_fb * jnp.where(tau_fb[-1] * dir_sgn < 0.0, -1.0, 1.0)
            return _normalize_tangent(tau_fb, w_arc)

        return jax.lax.cond(jnp.all(jnp.isfinite(tau_raw)), lambda: tau_op, _fallback_svd)

    assert jac_u_fn is not None and jac_p_fn is not None
    Ju = jac_u_fn(u0_arr, p0_arr)
    Jp = jac_p_fn(u0_arr, p0_arr)

    if param_mode == "natural":
        sol = solve_bordered_system(
            A=Ju,
            b=Jp,
            c=zero_u,
            d=dir_sgn,
            f=zero_u,
            g=1.0,
        )
        return sol.z

    J = jnp.hstack([Ju, Jp.reshape(-1, 1)])
    _, _, vh = jnp.linalg.svd(J)
    tau = vh[-1, :]
    tau = tau * jnp.where(tau[-1] * dir_sgn < 0.0, -1.0, 1.0)
    return _normalize_tangent(tau, w_arc)


def track_arclength_scan(
    residual_fn: Callable[..., jax.Array],
    jac_u_fn: Callable[..., jax.Array] | None = None,
    jac_p_fn: Callable[..., jax.Array] | None = None,
    u0: ArrayLike | None = None,
    p0: float | jax.Array = 0.0,
    tau0: jax.Array | None = None,
    ds: float | jax.Array = 0.05,
    ds_min: float | jax.Array = 1e-6,
    ds_max: float | jax.Array = 0.5,
    n_steps: int = 100,
    p_min: float | jax.Array = -jnp.inf,
    p_max: float | jax.Array = jnp.inf,
    adaptive_step: bool = False,
    predictor: str = "tangent",
    param_mode: str = "arclength",
    p_target: float | jax.Array | None = None,
    arclength_weights: jax.Array | None = None,
    chart_fn: Callable[[jax.Array, jax.Array], Any] | None = None,
    accept_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    stop_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    tol: float = 1e-8,
    step_tol: float = 1e-8,
    max_newton_iters: int = 15,
    solver_method: str = "auto",
    tracker_params: TrackerParameters | None = None,
    norm_options: WeightedNormOptions | None = None,
    step_observer_fn: Callable[[jax.Array, jax.Array, jax.Array, Any, Any], Any] | None = None,
    bordered_solver: BorderedOperatorSolver | Callable[[Any], BorderedOperatorSolver] | None = None,
) -> TrackScanResult:
    """Track a 1D solution curve `(u(s), p(s))` with `jax.lax.scan`."""
    if u0 is None or tau0 is None:
        msg = "u0 and tau0 must be provided."
        raise ValueError(msg)
    if bordered_solver is None and (jac_u_fn is None or jac_p_fn is None):
        msg = "Provide jac_u_fn and jac_p_fn when bordered_solver is None."
        raise ValueError(msg)
    if predictor not in ("tangent", "pade21"):
        msg = f"Unknown predictor: {predictor}. Choose from 'tangent' or 'pade21'."
        raise ValueError(msg)
    if param_mode not in ("arclength", "natural"):
        msg = f"Unknown param_mode: {param_mode}. Choose from 'arclength' or 'natural'."
        raise ValueError(msg)

    t_params = TrackerParameters() if tracker_params is None else tracker_params
    n_opts = WeightedNormOptions() if norm_options is None else norm_options

    u_cur = jnp.asarray(u0, dtype=jnp.float64)
    p_cur = jnp.squeeze(jnp.asarray(p0, dtype=jnp.float64))
    tau_cur = jnp.asarray(tau0, dtype=jnp.float64)
    n = u_cur.shape[0]
    w_arc = None if arclength_weights is None else jnp.asarray(arclength_weights, dtype=jnp.float64)
    is_natural = param_mode == "natural"
    has_target = is_natural and (p_target is not None)
    p_target_val = (
        jnp.array(0.0, dtype=jnp.float64)
        if p_target is None
        else jnp.squeeze(jnp.asarray(p_target, dtype=jnp.float64))
    )
    dir_sgn = jnp.where(tau_cur[-1] < 0.0, -1.0, 1.0)
    zero_u = jnp.zeros(n, dtype=jnp.float64)

    ds_cur = jnp.asarray(ds, dtype=jnp.float64)
    ds_min_val = jnp.asarray(ds_min, dtype=jnp.float64)
    ds_max_val = jnp.asarray(ds_max, dtype=jnp.float64)
    p_min_val = jnp.asarray(p_min, dtype=jnp.float64)
    p_max_val = jnp.asarray(p_max, dtype=jnp.float64)

    def _bind_res(chart: Any) -> Callable[[jax.Array, jax.Array], jax.Array]:
        if chart_fn is None:
            return residual_fn
        return lambda u_v, p_v: residual_fn(u_v, p_v, chart)

    def _bind_jacs(
        chart: Any,
    ) -> tuple[
        Callable[[jax.Array, jax.Array], jax.Array],
        Callable[[jax.Array, jax.Array], jax.Array],
    ]:
        assert jac_u_fn is not None and jac_p_fn is not None
        if chart_fn is None:
            return jac_u_fn, jac_p_fn
        return (
            lambda u_v, p_v: jac_u_fn(u_v, p_v, chart),
            lambda u_v, p_v: jac_p_fn(u_v, p_v, chart),
        )

    def _resolve_solver(chart: Any) -> BorderedOperatorSolver | None:
        if bordered_solver is None:
            return None
        if isinstance(bordered_solver, BorderedOperatorSolver):
            return bordered_solver
        return bordered_solver(chart)

    def _border_for(tau_val: jax.Array) -> tuple[jax.Array | None, jax.Array | None]:
        if is_natural:
            return zero_u, dir_sgn
        if w_arc is not None:
            return w_arc[:n] * tau_val[:n], w_arc[-1] * tau_val[-1]
        return None, None

    chart_0 = chart_fn(u_cur, p_cur) if chart_fn is not None else None
    pred_order = 4 if predictor == "pade21" else 2
    w0 = init_weighted_norm(jnp.concatenate([u_cur, jnp.atleast_1d(p_cur)]), n_opts)

    if predictor == "pade21":
        res0_fn = _bind_res(chart_0)
        solver_0 = _resolve_solver(chart_0)
        bc0, bd0 = _border_for(tau_cur)
        if solver_0 is not None:
            z2_0, z3_0, tr_0 = compute_arclength_taylor3(
                res0_fn,
                u_cur,
                p_cur,
                tau_cur,
                border_c=bc0,
                border_d=bd0,
                weights=w0,
                bordered_solver=solver_0,
            )
        else:
            ju0_fn, jp0_fn = _bind_jacs(chart_0)
            Ju0 = ju0_fn(u_cur, p_cur)
            Jp0 = jp0_fn(u_cur, p_cur)
            z2_0, z3_0, tr_0 = compute_arclength_taylor3(
                res0_fn,
                u_cur,
                p_cur,
                tau_cur,
                Ju0,
                Jp0,
                border_c=bc0,
                border_d=bd0,
                weights=w0,
            )
    else:
        z2_0 = jnp.zeros(n + 1, dtype=jnp.float64)
        z3_0 = jnp.zeros(n + 1, dtype=jnp.float64)
        tr_0 = jnp.array(jnp.inf, dtype=jnp.float64)

    init_active = (dir_sgn * (p_target_val - p_cur) > 1e-12) if has_target else jnp.bool_(True)
    init_carry = (
        u_cur,
        p_cur,
        tau_cur,
        z2_0,
        z3_0,
        tr_0,
        chart_0,
        ds_cur,
        w0,
        jnp.array(1.0, dtype=jnp.float64),
        jnp.array(1.0, dtype=jnp.float64),
        init_active,
        jnp.bool_(False),
        jnp.bool_(True),
    )

    def _compute_next_tau(
        chart_cand: Any,
        solver_cand: BorderedOperatorSolver | None,
        u_val: jax.Array,
        p_val: jax.Array,
        tau_prev: jax.Array,
    ) -> tuple[jax.Array, jax.Array | None, jax.Array | None]:
        c_bord = (
            zero_u if is_natural else (tau_prev[:n] if w_arc is None else w_arc[:n] * tau_prev[:n])
        )
        d_bord = (
            dir_sgn if is_natural else (tau_prev[-1] if w_arc is None else w_arc[-1] * tau_prev[-1])
        )

        if solver_cand is not None:
            fac_tau = solver_cand.factor(u_val, p_val, c_bord, d_bord)
            tau_raw = solver_cand.solve(fac_tau, zero_u, jnp.array(1.0, dtype=jnp.float64)).z
            Ju_cand, Jp_cand = None, None
        else:
            ju_step, jp_step = _bind_jacs(chart_cand)
            Ju_cand = ju_step(u_val, p_val)
            Jp_cand = jp_step(u_val, p_val)
            tau_raw = solve_bordered_system(
                A=Ju_cand,
                b=Jp_cand,
                c=c_bord,
                d=d_bord,
                f=zero_u,
                g=1.0,
                method=solver_method,
            ).z

        if is_natural:
            return tau_raw, Ju_cand, Jp_cand
        tau_cand = _normalize_tangent(tau_raw, w_arc)
        align_dot = (
            jnp.dot(tau_cand, tau_prev) if w_arc is None else jnp.dot(w_arc * tau_cand, tau_prev)
        )
        tau_cand = jnp.where(align_dot < 0.0, -tau_cand, tau_cand)
        return tau_cand, Ju_cand, Jp_cand

    def step_fn(carry: tuple[Any, ...], _: None) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        (
            u,
            p,
            tau,
            z2,
            z3,
            tr,
            chart,
            cur_ds,
            weights,
            omega,
            omega_prev,
            is_active,
            last_failed,
            is_first,
        ) = carry

        if has_target:
            rem = dir_sgn * (p_target_val - p)
            at_target = rem <= cur_ds
            step_ds = jnp.minimum(cur_ds, jnp.maximum(rem, 0.0))
        else:
            at_target = jnp.bool_(False)
            step_ds = cur_ds

        if predictor == "pade21":
            u_pred, p_pred = pade21_predict(u, p, tau, z2, z3, tr, step_ds)
        else:
            u_pred, p_pred = tangent_predict(u, p, tau, step_ds)

        if has_target:
            p_pred = jnp.where(at_target, p_target_val, p_pred)

        w_step = update_weighted_norm(
            weights, jnp.concatenate([u_pred, jnp.atleast_1d(p_pred)]), n_opts
        )
        res_step = _bind_res(chart)
        solver_step = _resolve_solver(chart)
        ju_step, jp_step = _bind_jacs(chart) if solver_step is None else (None, None)
        bc_step, bd_step = _border_for(tau)

        res = correct_arclength_step(
            residual_fn=res_step,
            jac_u_fn=ju_step,
            jac_p_fn=jp_step,
            u_init=u_pred,
            p_init=p_pred,
            u_prev=u,
            p_prev=p,
            tau_prev=tau,
            ds=step_ds,
            weights=w_step,
            omega_init=omega,
            first_step=is_first,
            tol=tol,
            step_tol=step_tol,
            max_iters=max_newton_iters,
            solver_method=solver_method,
            contraction_a=t_params.a,
            enforce_contraction=adaptive_step,
            track_contraction=adaptive_step,
            border_c=bc_step,
            border_d=bd_step,
            bordered_solver=solver_step,
        )

        u_cand, p_cand = res.u, res.p
        if has_target:
            p_cand = jnp.where(at_target, p_target_val, p_cand)

        step_ok = (
            res.converged & accept_fn(u_cand, p_cand) if accept_fn is not None else res.converged
        )
        converged_step = is_active & step_ok

        chart_cand = chart_fn(u_cand, p_cand) if chart_fn is not None else chart
        res_next_fn = _bind_res(chart_cand)
        solver_next = _resolve_solver(chart_cand)
        tau_cand, Ju_cand, Jp_cand = _compute_next_tau(chart_cand, solver_next, u_cand, p_cand, tau)

        if predictor == "pade21":
            bc_cand, bd_cand = _border_for(tau_cand)
            z2_cand, z3_cand, tr_cand = compute_arclength_taylor3(
                res_next_fn,
                u_cand,
                p_cand,
                tau_cand,
                Ju_cand,
                Jp_cand,
                border_c=bc_cand,
                border_d=bd_cand,
                weights=w_step,
                bordered_solver=solver_next,
            )
        else:
            z2_cand, z3_cand, tr_cand = z2, z3, tr

        obs_cand = (
            step_observer_fn(u_cand, p_cand, tau_cand, Ju_cand, Jp_cand)
            if step_observer_fn is not None
            else None
        )

        u_next = jnp.where(converged_step, u_cand, u)
        p_next = jnp.where(converged_step, p_cand, p)
        tau_next = jnp.where(converged_step, tau_cand, tau)
        z2_next = jnp.where(converged_step, z2_cand, z2)
        z3_next = jnp.where(converged_step, z3_cand, z3)
        tr_next = jnp.where(converged_step, tr_cand, tr)
        chart_next = (
            jax.tree_util.tree_map(
                lambda new_v, old_v: jnp.where(converged_step, new_v, old_v), chart_cand, chart
            )
            if chart_fn is not None
            else None
        )
        weights_next = jnp.where(converged_step, w_step, weights)

        in_bounds = (p_next >= p_min_val) & (p_next <= p_max_val)
        stopped_by_fn = stop_fn(u_next, p_next) if stop_fn is not None else jnp.bool_(False)
        still_going = (
            in_bounds
            & (~stopped_by_fn)
            & (~(dir_sgn * (p_target_val - p_next) <= 1e-12) if has_target else True)
        )
        is_first_next = jnp.where(converged_step, jnp.bool_(False), is_first)
        last_failed_next = jnp.where(is_active, ~step_ok, last_failed)

        if adaptive_step:
            omega_valid = jnp.isfinite(res.omega) & (res.omega > 0.0)
            omega_floor = jnp.maximum(0.5 * omega, 0.1)
            omega_cand = jnp.where(omega_valid, jnp.maximum(res.omega, omega_floor), omega_floor)
            omega_prev_next = jnp.where(converged_step, omega, omega_prev)
            omega_next = jnp.where(converged_step, omega_cand, omega)

            local_err = estimate_local_error(
                u_pred, p_pred, u_cand, p_cand, jnp.maximum(step_ds, ds_min_val), pred_order, w_step
            )
            ds_cand = update_step_size(
                ds=cur_ds,
                converged=step_ok,
                omega=omega_next,
                omega_prev=omega_prev_next,
                local_error=local_err,
                trust_region=tr_next,
                theta=res.theta,
                iterations=res.iterations,
                last_step_failed=last_failed,
                predictor_order=pred_order,
                ds_min=ds_min_val,
                ds_max=ds_max_val,
                params=t_params,
            )
            ds_next = jnp.where(is_active, ds_cand, cur_ds)
            can_retry = step_ok | (cur_ds > ds_min_val * (1.0 + 1e-12))
            is_active_next = is_active & jnp.where(step_ok, still_going, can_retry)
        else:
            omega_prev_next = omega_prev
            omega_next = omega
            ds_next = cur_ds
            is_active_next = converged_step & still_going

        carry_next = (
            u_next,
            p_next,
            tau_next,
            z2_next,
            z3_next,
            tr_next,
            chart_next,
            ds_next,
            weights_next,
            omega_next,
            omega_prev_next,
            is_active_next,
            last_failed_next,
            is_first_next,
        )
        output_step = (
            u_next,
            p_next,
            tau_next,
            converged_step,
            step_ds,
            res.iterations,
            res.residual_norm,
            omega_next,
            tr_next,
            obs_cand,
        )
        return carry_next, output_step

    _, outputs = jax.lax.scan(step_fn, init_carry, None, length=n_steps)
    return TrackScanResult(*outputs)
