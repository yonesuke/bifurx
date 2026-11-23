from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.homotopy.bordered import solve_bordered_system
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
    """Result of pure-JAX pseudo-arclength path tracking via `jax.lax.scan`."""

    u: jax.Array
    p: jax.Array
    tangents: jax.Array
    converged: jax.Array
    step_sizes: jax.Array
    iterations: jax.Array
    omega: jax.Array
    trust_regions: jax.Array
    observer: Any


def compute_tangent(
    jac_u_fn: Callable[[jax.Array, jax.Array], jax.Array],
    jac_p_fn: Callable[[jax.Array, jax.Array], jax.Array],
    u0: ArrayLike,
    p0: float | jax.Array,
    direction: float | jax.Array = 1.0,
    initial_tangent: jax.Array | None = None,
) -> jax.Array:
    """Compute or normalize the initial unit tangent vector `(du/ds, dp/ds)`."""
    if initial_tangent is not None:
        tau = jnp.asarray(initial_tangent, dtype=jnp.float64)
        norm_val = jnp.linalg.norm(tau)
        return jnp.where(norm_val > 1e-12, tau / norm_val, tau)

    u0_arr = jnp.asarray(u0, dtype=jnp.float64)
    p0_arr = jnp.squeeze(jnp.asarray(p0, dtype=jnp.float64))
    Ju = jac_u_fn(u0_arr, p0_arr)
    Jp = jac_p_fn(u0_arr, p0_arr)
    J = jnp.hstack([Ju, Jp.reshape(-1, 1)])
    _, _, vh = jnp.linalg.svd(J)
    tau = vh[-1, :]
    tau = tau * jnp.where(tau[-1] * direction < 0, -1.0, 1.0)
    norm_val = jnp.linalg.norm(tau)
    return jnp.where(norm_val > 1e-12, tau / norm_val, tau)


def track_arclength_scan(
    residual_fn: Callable[[jax.Array, jax.Array], jax.Array],
    jac_u_fn: Callable[[jax.Array, jax.Array], jax.Array],
    jac_p_fn: Callable[[jax.Array, jax.Array], jax.Array],
    u0: ArrayLike,
    p0: float | jax.Array,
    tau0: jax.Array,
    ds: float | jax.Array = 0.05,
    ds_min: float | jax.Array = 1e-6,
    ds_max: float | jax.Array = 0.5,
    n_steps: int = 100,
    p_min: float | jax.Array = -jnp.inf,
    p_max: float | jax.Array = jnp.inf,
    adaptive_step: bool = False,
    predictor: str = "tangent",
    tol: float = 1e-8,
    step_tol: float = 1e-8,
    rel_tol: float = 1e-8,
    max_newton_iters: int = 15,
    solver_method: str = "auto",
    tracker_params: TrackerParameters | None = None,
    norm_options: WeightedNormOptions | None = None,
    step_observer_fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], Any]
    | None = None,
) -> TrackScanResult:
    """Track a 1D solution curve `(u(s), p(s))` with `jax.lax.scan`."""
    if predictor not in ("tangent", "pade21"):
        msg = f"Unknown predictor: {predictor}. Choose from 'tangent' or 'pade21'."
        raise ValueError(msg)

    t_params = TrackerParameters() if tracker_params is None else tracker_params
    n_opts = WeightedNormOptions() if norm_options is None else norm_options

    u_cur = jnp.asarray(u0, dtype=jnp.float64)
    p_cur = jnp.squeeze(jnp.asarray(p0, dtype=jnp.float64))
    tau_cur = jnp.asarray(tau0, dtype=jnp.float64)
    n = u_cur.shape[0]

    ds_cur = jnp.asarray(ds, dtype=jnp.float64)
    ds_min_val = jnp.asarray(ds_min, dtype=jnp.float64)
    ds_max_val = jnp.asarray(ds_max, dtype=jnp.float64)
    p_min_val = jnp.asarray(p_min, dtype=jnp.float64)
    p_max_val = jnp.asarray(p_max, dtype=jnp.float64)

    pred_order = 4 if predictor == "pade21" else 2
    w0 = init_weighted_norm(jnp.concatenate([u_cur, jnp.atleast_1d(p_cur)]), n_opts)

    if predictor == "pade21":
        Ju0 = jac_u_fn(u_cur, p_cur)
        Jp0 = jac_p_fn(u_cur, p_cur)
        z2_0, z3_0, tr_0 = compute_arclength_taylor3(residual_fn, u_cur, p_cur, tau_cur, Ju0, Jp0)
    else:
        z2_0 = jnp.zeros(n + 1, dtype=jnp.float64)
        z3_0 = jnp.zeros(n + 1, dtype=jnp.float64)
        tr_0 = jnp.array(jnp.inf, dtype=jnp.float64)

    init_carry = (
        u_cur,
        p_cur,
        tau_cur,
        z2_0,
        z3_0,
        tr_0,
        ds_cur,
        w0,
        jnp.array(1.0, dtype=jnp.float64),
        jnp.array(1.0, dtype=jnp.float64),
        jnp.bool_(True),
        jnp.bool_(False),
        jnp.bool_(True),
    )

    def _compute_next_tau(
        u_val: jax.Array,
        p_val: jax.Array,
        tau_prev: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        Ju_cand = jac_u_fn(u_val, p_val)
        Jp_cand = jac_p_fn(u_val, p_val)
        sol_tau = solve_bordered_system(
            A=Ju_cand,
            b=Jp_cand,
            c=tau_prev[:n],
            d=tau_prev[-1],
            f=jnp.zeros(n, dtype=jnp.float64),
            g=1.0,
            method=solver_method,
        )
        tau_cand = sol_tau.z
        tau_norm = jnp.linalg.norm(tau_cand)
        tau_cand = jnp.where(tau_norm > 1e-12, tau_cand / tau_norm, tau_prev)
        tau_cand = jnp.where(jnp.dot(tau_cand, tau_prev) < 0.0, -tau_cand, tau_cand)
        return tau_cand, Ju_cand, Jp_cand

    def step_fn(carry: tuple[Any, ...], _: None) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        (
            u,
            p,
            tau,
            z2,
            z3,
            tr,
            cur_ds,
            weights,
            omega,
            omega_prev,
            is_active,
            last_failed,
            is_first,
        ) = carry

        if predictor == "pade21":
            u_pred, p_pred = pade21_predict(u, p, tau, z2, z3, tr, cur_ds)
        else:
            u_pred, p_pred = tangent_predict(u, p, tau, cur_ds)

        w_step = update_weighted_norm(
            weights, jnp.concatenate([u_pred, jnp.atleast_1d(p_pred)]), n_opts
        )

        res = correct_arclength_step(
            residual_fn=residual_fn,
            jac_u_fn=jac_u_fn,
            jac_p_fn=jac_p_fn,
            u_init=u_pred,
            p_init=p_pred,
            u_prev=u,
            p_prev=p,
            tau_prev=tau,
            ds=cur_ds,
            weights=w_step,
            omega_init=omega,
            first_step=is_first,
            tol=tol,
            step_tol=step_tol,
            rel_tol=rel_tol,
            max_iters=max_newton_iters,
            solver_method=solver_method,
            contraction_a=t_params.a,
            enforce_contraction=adaptive_step,
        )

        converged_step = is_active & res.converged
        u_cand, p_cand = res.u, res.p
        tau_cand, Ju_cand, Jp_cand = _compute_next_tau(u_cand, p_cand, tau)

        if predictor == "pade21":
            z2_cand, z3_cand, tr_cand = compute_arclength_taylor3(
                residual_fn, u_cand, p_cand, tau_cand, Ju_cand, Jp_cand
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

        z_next = jnp.concatenate([u_next, jnp.atleast_1d(p_next)])
        weights_next = jnp.where(
            converged_step, update_weighted_norm(w_step, z_next, n_opts), weights
        )

        omega_valid = jnp.isfinite(res.omega) & (res.omega > 0.0)
        omega_floor = jnp.maximum(0.5 * omega, 0.1)
        omega_cand = jnp.where(omega_valid, jnp.maximum(res.omega, omega_floor), omega_floor)
        omega_prev_next = jnp.where(converged_step, omega, omega_prev)
        omega_next = jnp.where(converged_step, omega_cand, omega)

        in_bounds = (p_next >= p_min_val) & (p_next <= p_max_val)
        is_first_next = jnp.where(converged_step, jnp.bool_(False), is_first)
        last_failed_next = jnp.where(is_active, ~res.converged, last_failed)

        if adaptive_step:
            local_err = estimate_local_error(
                u_pred, p_pred, u_cand, p_cand, cur_ds, pred_order, w_step
            )
            ds_cand = update_step_size(
                ds=cur_ds,
                converged=res.converged,
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
            can_retry = res.converged | (cur_ds > ds_min_val * (1.0 + 1e-12))
            is_active_next = is_active & jnp.where(res.converged, in_bounds, can_retry)
        else:
            ds_next = cur_ds
            is_active_next = converged_step & in_bounds

        carry_next = (
            u_next,
            p_next,
            tau_next,
            z2_next,
            z3_next,
            tr_next,
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
            cur_ds,
            res.iterations,
            omega_next,
            tr_next,
            obs_cand,
        )
        return carry_next, output_step

    _, outputs = jax.lax.scan(step_fn, init_carry, None, length=n_steps)
    return TrackScanResult(*outputs)
