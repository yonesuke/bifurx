from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from bifurx.homotopy.bordered import solve_bordered_system
from bifurx.homotopy.norm import weighted_inf_norm
from bifurx.homotopy.stepsize import kantorovich_h


class CorrectorResult(NamedTuple):
    """Result of pure-JAX bordered Newton arclength corrector."""

    converged: jax.Array
    u: jax.Array
    p: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array
    accuracy: jax.Array
    omega: jax.Array
    theta: jax.Array
    norm_dz0: jax.Array
    terminated_early: jax.Array


def correct_arclength_step(
    residual_fn: Callable[[jax.Array, jax.Array], jax.Array],
    jac_u_fn: Callable[[jax.Array, jax.Array], jax.Array],
    jac_p_fn: Callable[[jax.Array, jax.Array], jax.Array],
    u_init: jax.Array,
    p_init: float | jax.Array,
    u_prev: jax.Array,
    p_prev: float | jax.Array,
    tau_prev: jax.Array,
    ds: float | jax.Array,
    weights: jax.Array | None = None,
    omega_init: float | jax.Array = 0.0,
    first_step: bool | jax.Array = True,
    tol: float = 1e-8,
    step_tol: float = 1e-8,
    rel_tol: float = 1e-8,
    max_iters: int = 15,
    solver_method: str = "auto",
    contraction_a: float = 0.125,
    enforce_contraction: bool = False,
    track_contraction: bool = True,
    border_c: jax.Array | None = None,
    border_d: float | jax.Array | None = None,
) -> CorrectorResult:
    """Solve bordered arclength Newton system with optional contraction-rate early abort.

    Solves:
        residual_fn(u, p) = 0
        <c, u - u_prev> + d * (p - p_prev) - ds = 0
    where `(c, d)` defaults to `(tau_prev[:n], tau_prev[-1])`.
    """
    n = u_init.shape[0]
    u_0 = jnp.asarray(u_init, dtype=jnp.float64)
    p_0 = jnp.squeeze(jnp.asarray(p_init, dtype=jnp.float64))
    u_p = jnp.asarray(u_prev, dtype=jnp.float64)
    p_p = jnp.squeeze(jnp.asarray(p_prev, dtype=jnp.float64))
    tau_arr = jnp.asarray(tau_prev, dtype=jnp.float64)
    c_vec = (
        tau_arr[:n] if border_c is None else jnp.asarray(border_c, dtype=jnp.float64).reshape((n,))
    )
    d_val = (
        jnp.squeeze(tau_arr[-1])
        if border_d is None
        else jnp.squeeze(jnp.asarray(border_d, dtype=jnp.float64))
    )
    ds_val = jnp.squeeze(jnp.asarray(ds, dtype=jnp.float64))

    use_weighted_step = weights is not None
    weights_arr = (
        jnp.ones(n + 1, dtype=jnp.float64)
        if weights is None
        else jnp.asarray(weights, dtype=jnp.float64)
    )
    omega_init_arr = jnp.squeeze(jnp.asarray(omega_init, dtype=jnp.float64))
    first_step_arr = jnp.squeeze(jnp.asarray(first_step, dtype=bool))
    h_a = kantorovich_h(contraction_a)

    f_0 = residual_fn(u_0, p_0)
    arc_0 = jnp.dot(c_vec, u_0 - u_p) + d_val * (p_0 - p_p) - ds_val
    res_norm_0 = jnp.sqrt(jnp.sum(f_0**2) + arc_0**2)
    res_tol = tol + rel_tol * res_norm_0
    conv_0 = jnp.isfinite(res_norm_0) & (res_norm_0 <= tol)

    zero_f = jnp.array(0.0, dtype=jnp.float64)
    nan_f = jnp.array(jnp.nan, dtype=jnp.float64)
    init_omega = jnp.where(omega_init_arr > 0.0, omega_init_arr, zero_f)
    init_state = (
        u_0,
        p_0,
        f_0,
        arc_0,
        jnp.int32(0),
        conv_0,
        jnp.bool_(False),
        res_norm_0,
        zero_f,
        zero_f,
        init_omega,
        nan_f,
        jnp.array(contraction_a, dtype=jnp.float64),
    )

    def cond_fn(state: tuple[jax.Array, ...]) -> jax.Array:
        _u, _p, _f, _arc, it, conv, aborted, _rn, _sw, _dz0, _om, _th, _ab = state
        return (~conv) & (~aborted) & (it < max_iters)

    def body_fn(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        u, p, f_val, arc_res, it, _conv, _aborted, _rn, prev_sw, norm_dz0, omega, theta, a_bar = (
            state
        )

        Ju = jac_u_fn(u, p)
        Jp = jac_p_fn(u, p)

        sol = solve_bordered_system(
            A=Ju,
            b=Jp,
            c=c_vec,
            d=d_val,
            f=-f_val,
            g=-arc_res,
            method=solver_method,
        )
        dz = sol.z
        du = sol.x
        dp = jnp.squeeze(sol.y)

        step_norm_w = (
            weighted_inf_norm(dz, weights_arr) if use_weighted_step else jnp.max(jnp.abs(dz))
        )
        if use_weighted_step:
            step_ok = step_norm_w <= step_tol
        else:
            step_norm_l2 = jnp.sqrt(jnp.sum(du**2) + dp**2)
            z_norm_l2 = jnp.sqrt(jnp.sum(u**2) + p**2)
            step_ok = step_norm_l2 <= step_tol * (1.0 + z_norm_l2)

        u_cand = u + du
        p_cand = p + dp
        f_next = residual_fn(u_cand, p_cand)
        arc_next = jnp.dot(c_vec, u_cand - u_p) + d_val * (p_cand - p_p) - ds_val
        res_norm_next = jnp.sqrt(jnp.sum(f_next**2) + arc_next**2)

        is_valid = (
            jnp.isfinite(res_norm_next)
            & jnp.isfinite(step_norm_w)
            & jnp.all(jnp.isfinite(du))
            & jnp.isfinite(dp)
        )
        # Check convergence before contraction rejection so machine-precision iterates pass.
        conv_next = is_valid & (
            (res_norm_next <= res_tol) | (step_ok & (res_norm_next <= 10.0 * res_tol))
        )

        if track_contraction or enforce_contraction:
            norm_dz0_next = jnp.where(it == 0, step_norm_w, norm_dz0)
            theta_cand = step_norm_w / jnp.maximum(prev_sw, 1e-30)
            theta_next = jnp.where((it >= 1) & (prev_sw > 1e-30), theta_cand, theta)

            omega_from_i1 = 2.0 * step_norm_w / jnp.maximum(prev_sw**2, 1e-30)
            omega_next = jnp.where((it == 1) & (prev_sw > max(1e-7, tol)), omega_from_i1, omega)
            a_bar_next = jnp.where(it >= 1, a_bar * a_bar, a_bar)
        else:
            norm_dz0_next = norm_dz0
            theta_next = theta
            omega_next = omega
            a_bar_next = a_bar

        if enforce_contraction:
            abort_i0 = (
                (it == 0)
                & (~first_step_arr)
                & (omega_init_arr > 0.0)
                & (0.125 * step_norm_w * omega_init_arr > h_a)
            )
            abort_contraction = (it >= 1) & (theta_next > a_bar)
            aborted_next = (~is_valid) | ((~conv_next) & (abort_i0 | abort_contraction))
        else:
            aborted_next = ~is_valid

        u_next = jnp.where(is_valid, u_cand, u)
        p_next = jnp.where(is_valid, p_cand, p)
        f_out = jnp.where(is_valid, f_next, f_val)
        arc_out = jnp.where(is_valid, arc_next, arc_res)
        return (
            u_next,
            p_next,
            f_out,
            arc_out,
            it + jnp.int32(1),
            conv_next,
            aborted_next,
            res_norm_next,
            step_norm_w,
            norm_dz0_next,
            omega_next,
            theta_next,
            a_bar_next,
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    (
        u_fin,
        p_fin,
        _f_fin,
        _arc_fin,
        it_fin,
        conv_fin,
        aborted_fin,
        res_fin,
        step_fin,
        dz0_fin,
        omega_fin,
        theta_fin,
        _abar,
    ) = final_state

    omega_decay = jnp.where(omega_init_arr > 0.0, 0.25 * omega_init_arr, jnp.array(1.0))
    omega_out = jnp.where(conv_fin & (it_fin <= 1), omega_decay, omega_fin)

    return CorrectorResult(
        converged=conv_fin,
        u=u_fin,
        p=p_fin,
        iterations=it_fin,
        residual_norm=res_fin,
        accuracy=step_fin,
        omega=omega_out,
        theta=theta_fin,
        norm_dz0=dz0_fin,
        terminated_early=aborted_fin,
    )
