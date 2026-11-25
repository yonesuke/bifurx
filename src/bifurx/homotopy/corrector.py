from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from bifurx.homotopy.bordered import (
    BlockBorderedFactorization,
    BorderedFactorization,
    BorderedOperatorSolver,
    factor_bordered_block,
    factor_bordered_matrix,
    solve_factored_bordered,
    solve_factored_bordered_block,
)
from bifurx.homotopy.norm import weighted_inf_norm
from bifurx.homotopy.stepsize import kantorovich_h

_LIN_RES_FACTOR = 0.5
_LIN_SCALE_TOL = 1e-6
_LSTSQ_RCOND = 1e-12
_OMEGA_DECAY_FACTOR = 0.25


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


def _compute_acc_norm(
    step_norm_w: jax.Array,
    du: jax.Array,
    dp: jax.Array,
    u_base: jax.Array,
    p_base: jax.Array,
    use_weighted_step: bool,
) -> jax.Array:
    """Compute domain-space step accuracy in weighted or relative-L2 norm."""
    if use_weighted_step:
        return step_norm_w
    step_l2 = jnp.sqrt(jnp.sum(du**2) + dp**2)
    z_l2 = jnp.sqrt(jnp.sum(u_base**2) + p_base**2)
    return step_l2 / (1.0 + z_l2)


def _linear_consistency_ok(
    Ju: jax.Array,
    Jp: jax.Array,
    c_vec: jax.Array,
    d_val: jax.Array,
    weights_arr: jax.Array,
    f_val: jax.Array,
    arc_res: jax.Array,
    du: jax.Array,
    dp: jax.Array,
    step_norm_w: jax.Array,
) -> jax.Array:
    """Check row/column-scale-invariant linear residual consistency."""
    n = du.shape[0]
    f_lin = Ju @ du + Jp * dp + f_val
    row_scale = (
        jnp.sum(jnp.abs(Ju) * weights_arr[:n][None, :], axis=1) + jnp.abs(Jp) * weights_arr[-1]
    ) * step_norm_w
    f_lin_ok = jnp.all(
        jnp.abs(f_lin) <= _LIN_RES_FACTOR * jnp.abs(f_val) + _LIN_SCALE_TOL * row_scale
    )
    arc_lin = jnp.dot(c_vec, du) + d_val * dp + arc_res
    arc_scale = (
        jnp.sum(jnp.abs(c_vec) * weights_arr[:n]) + jnp.abs(d_val) * weights_arr[-1]
    ) * step_norm_w
    arc_lin_ok = jnp.abs(arc_lin) <= _LIN_RES_FACTOR * jnp.abs(arc_res) + _LIN_SCALE_TOL * arc_scale
    return (
        jnp.isfinite(step_norm_w)
        & jnp.all(jnp.isfinite(du))
        & jnp.isfinite(dp)
        & f_lin_ok
        & arc_lin_ok
    )


def _factor_bordered_step(
    Ju: jax.Array,
    Jp: jax.Array,
    c_vec: jax.Array,
    d_val: jax.Array,
    solver_method: str,
) -> BlockBorderedFactorization | BorderedFactorization:
    """Factorize the bordered linear system once per Newton iteration."""
    if solver_method == "block":
        return factor_bordered_block(Ju, Jp, c_vec, d_val)
    if solver_method in ("auto", "direct"):
        return factor_bordered_matrix(Ju, Jp, c_vec, d_val)
    msg = f"Unknown solver_method: {solver_method}. Choose from 'auto', 'block', or 'direct'."
    raise ValueError(msg)


def _solve_factored_with_fallback(
    fac: BlockBorderedFactorization | BorderedFactorization,
    Ju: jax.Array,
    Jp: jax.Array,
    c_vec: jax.Array,
    d_val: jax.Array,
    weights_arr: jax.Array,
    f_val: jax.Array,
    arc_res: jax.Array,
    solver_method: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve bordered linear system reusing `fac`, falling back to lstsq if `auto` loses accuracy."""
    if isinstance(fac, BlockBorderedFactorization):
        sol = solve_factored_bordered_block(fac, -f_val, -arc_res)
    else:
        sol = solve_factored_bordered(fac, -f_val, -arc_res)

    du = jnp.asarray(sol.x, dtype=jnp.float64)
    dp = jnp.asarray(jnp.squeeze(sol.y), dtype=jnp.float64)
    step_norm_w = jnp.asarray(weighted_inf_norm(sol.z, weights_arr), dtype=jnp.float64)
    lin_ok = jnp.asarray(
        _linear_consistency_ok(
            Ju, Jp, c_vec, d_val, weights_arr, f_val, arc_res, du, dp, step_norm_w
        ),
        dtype=bool,
    )
    if solver_method != "auto" or isinstance(fac, BlockBorderedFactorization):
        return du, dp, step_norm_w, lin_ok

    def _retry_lstsq() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        n = du.shape[0]
        r = jnp.concatenate([-f_val.ravel(), jnp.atleast_1d(-arc_res)])
        z_ls = jnp.linalg.lstsq(fac.M, r, rcond=_LSTSQ_RCOND)[0]
        du_d = jnp.asarray(z_ls[:n], dtype=jnp.float64)
        dp_d = jnp.asarray(jnp.squeeze(z_ls[n]), dtype=jnp.float64)
        sw_d = jnp.asarray(weighted_inf_norm(z_ls, weights_arr), dtype=jnp.float64)
        ok_d = jnp.asarray(
            _linear_consistency_ok(
                Ju, Jp, c_vec, d_val, weights_arr, f_val, arc_res, du_d, dp_d, sw_d
            ),
            dtype=bool,
        )
        return du_d, dp_d, sw_d, ok_d

    return jax.lax.cond(
        lin_ok,
        lambda: (du, dp, step_norm_w, lin_ok),
        _retry_lstsq,
    )


def _solve_operator_step(
    bordered_solver: BorderedOperatorSolver,
    residual_fn: Callable[[jax.Array, jax.Array], jax.Array],
    u_lin: jax.Array,
    p_lin: jax.Array,
    c_vec: jax.Array,
    d_val: jax.Array,
    weights_arr: jax.Array,
    fac: Any,
    f_rhs: jax.Array,
    arc_rhs: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve bordered linear system via pluggable `bordered_solver` and verify via JVP."""
    sol = bordered_solver.solve(fac, -f_rhs, -arc_rhs)
    du = jnp.asarray(sol.x, dtype=jnp.float64)
    dp = jnp.asarray(jnp.squeeze(sol.y), dtype=jnp.float64)
    step_norm_w = jnp.asarray(weighted_inf_norm(sol.z, weights_arr), dtype=jnp.float64)

    _, jvp_raw = jax.jvp(residual_fn, (u_lin, p_lin), (du, dp))
    jvp_f = jnp.atleast_1d(jnp.asarray(jvp_raw, dtype=jnp.float64)).ravel()
    arc_jvp = jnp.dot(c_vec, du) + d_val * dp
    jvp_all = jnp.concatenate([jvp_f, jnp.atleast_1d(arc_jvp)])
    r_rhs = jnp.concatenate([f_rhs, jnp.atleast_1d(arc_rhs)])
    r_lin = r_rhs + jvp_all
    lin_norm = jnp.linalg.norm(r_lin)
    tol_rhs = _LIN_RES_FACTOR * jnp.linalg.norm(r_rhs) + _LIN_SCALE_TOL * jnp.linalg.norm(jvp_all)
    lin_ok = (
        jnp.isfinite(step_norm_w)
        & jnp.all(jnp.isfinite(du))
        & jnp.isfinite(dp)
        & (lin_norm <= tol_rhs)
    )
    return du, dp, step_norm_w, jnp.asarray(lin_ok, dtype=bool)


def correct_arclength_step(
    residual_fn: Callable[[jax.Array, jax.Array], jax.Array],
    jac_u_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    jac_p_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
    u_init: jax.Array | None = None,
    p_init: float | jax.Array = 0.0,
    u_prev: jax.Array | None = None,
    p_prev: float | jax.Array = 0.0,
    tau_prev: jax.Array | None = None,
    ds: float | jax.Array = 0.0,
    weights: jax.Array | None = None,
    omega_init: float | jax.Array = 0.0,
    first_step: bool | jax.Array = True,
    tol: float = 1e-8,
    step_tol: float = 1e-8,
    max_iters: int = 15,
    solver_method: str = "auto",
    contraction_a: float = 0.125,
    enforce_contraction: bool = False,
    track_contraction: bool = True,
    border_c: jax.Array | None = None,
    border_d: float | jax.Array | None = None,
    bordered_solver: BorderedOperatorSolver | None = None,
) -> CorrectorResult:
    """Solve bordered arclength Newton system with optional contraction-rate early abort.

    Solves `residual_fn(u, p) = 0` and `<c, u - u_prev> + d * (p - p_prev) - ds = 0`, where
    `(c, d)` defaults to `(tau_prev[:n], tau_prev[-1])`. Domain-space Kantorovich stopping is
    governed by `tol` and `step_tol`.
    """
    if u_init is None or u_prev is None or tau_prev is None:
        msg = "u_init, u_prev, and tau_prev must be provided."
        raise ValueError(msg)
    if bordered_solver is None and (jac_u_fn is None or jac_p_fn is None):
        msg = "Provide jac_u_fn and jac_p_fn when bordered_solver is None."
        raise ValueError(msg)

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

    acc_tol = jnp.minimum(
        jnp.asarray(tol, dtype=jnp.float64),
        jnp.asarray(step_tol, dtype=jnp.float64),
    )
    h_a = kantorovich_h(contraction_a)
    kant_rhs = acc_tol * 2.0 * jnp.sqrt(jnp.maximum(1.0 - 2.0 * h_a, 0.0))
    eps_floor = 64.0 * float(jnp.finfo(jnp.float64).eps)

    def _eval_res(u_v: jax.Array, p_v: jax.Array) -> jax.Array:
        return jnp.atleast_1d(jnp.asarray(residual_fn(u_v, p_v), dtype=jnp.float64)).ravel()

    def _eval_arc(u_v: jax.Array, p_v: jax.Array) -> jax.Array:
        return jnp.squeeze(jnp.dot(c_vec, u_v - u_p) + d_val * (p_v - p_p) - ds_val)

    f_0 = _eval_res(u_0, p_0)
    arc_0 = _eval_arc(u_0, p_0)
    res_norm_0 = jnp.sqrt(jnp.sum(f_0**2) + arc_0**2)
    conv_0 = jnp.all(f_0 == 0.0) & (arc_0 == 0.0)

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
        jnp.bool_(False),
    )

    def cond_fn(state: tuple[jax.Array, ...]) -> jax.Array:
        _u, _p, _f, _arc, it, conv, aborted, _rn, _sw, _dz0, _om, _th, _ab, _has_om = state
        return (~conv) & (~aborted) & (it < max_iters)

    def body_fn(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        (
            u,
            p,
            f_val,
            arc_res,
            it,
            _conv,
            _aborted,
            rn_cur,
            prev_sw,
            norm_dz0,
            omega,
            theta,
            a_bar,
            has_om,
        ) = state

        if bordered_solver is not None:
            fac_op = bordered_solver.factor(u, p, c_vec, d_val)
            du, dp, step_norm_w, lin_ok = _solve_operator_step(
                bordered_solver,
                _eval_res,
                u,
                p,
                c_vec,
                d_val,
                weights_arr,
                fac_op,
                f_val,
                arc_res,
            )
        else:
            assert jac_u_fn is not None and jac_p_fn is not None
            Ju = jnp.asarray(jac_u_fn(u, p), dtype=jnp.float64)
            Jp = jnp.atleast_1d(jnp.asarray(jac_p_fn(u, p), dtype=jnp.float64)).ravel()
            fac_dense = _factor_bordered_step(Ju, Jp, c_vec, d_val, solver_method)
            du, dp, step_norm_w, lin_ok = _solve_factored_with_fallback(
                fac_dense,
                Ju,
                Jp,
                c_vec,
                d_val,
                weights_arr,
                f_val,
                arc_res,
                solver_method,
            )
        acc_norm = _compute_acc_norm(step_norm_w, du, dp, u, p, use_weighted_step)

        u_cand = u + du
        p_cand = p + dp
        f_next = _eval_res(u_cand, p_cand)
        arc_next = _eval_arc(u_cand, p_cand)
        res_norm_next = jnp.sqrt(jnp.sum(f_next**2) + arc_next**2)

        is_valid = jnp.all(jnp.isfinite(f_next)) & jnp.isfinite(arc_next) & lin_ok

        if track_contraction or enforce_contraction:
            norm_dz0_next = jnp.where(it == 0, step_norm_w, norm_dz0)
            theta_cand = step_norm_w / jnp.maximum(prev_sw, 1e-30)
            theta_next = jnp.where((it >= 1) & (prev_sw > 1e-30), theta_cand, theta)

            valid_omega_est = (
                is_valid
                & (it == 1)
                & (prev_sw > 0.0)
                & ((prev_sw * prev_sw) > 0.0)
                & (step_norm_w > eps_floor)
            )
            omega_from_i1 = 2.0 * step_norm_w / jnp.where(valid_omega_est, prev_sw * prev_sw, 1.0)
            omega_next = jnp.where(valid_omega_est, omega_from_i1, omega)
            has_om_next = has_om | valid_omega_est
            a_bar_next = jnp.where(it >= 1, a_bar * a_bar, a_bar)
        else:
            norm_dz0_next = norm_dz0
            theta_next = theta
            omega_next = omega
            has_om_next = has_om
            a_bar_next = a_bar

        direct_ok = is_valid & (acc_norm <= acc_tol)
        if track_contraction or enforce_contraction:
            has_kant_omega = (
                (omega_next > 0.0)
                & (omega_next * step_norm_w <= h_a)
                & jnp.where(it == 0, ~first_step_arr, theta_next <= a_bar)
            )
            need_final_check = (
                is_valid
                & (~direct_ok)
                & has_kant_omega
                & (omega_next * step_norm_w * step_norm_w < kant_rhs)
            )

            def _do_final_check() -> tuple[jax.Array, ...]:
                # Reuse factored operator / LU factorization for a zero-refactor simplified-Newton check.
                if bordered_solver is not None:
                    du_chk, dp_chk, sw_chk, lin_chk_ok = _solve_operator_step(
                        bordered_solver,
                        _eval_res,
                        u,
                        p,
                        c_vec,
                        d_val,
                        weights_arr,
                        fac_op,
                        f_next,
                        arc_next,
                    )
                else:
                    du_chk, dp_chk, sw_chk, lin_chk_ok = _solve_factored_with_fallback(
                        fac_dense,
                        Ju,
                        Jp,
                        c_vec,
                        d_val,
                        weights_arr,
                        f_next,
                        arc_next,
                        solver_method,
                    )
                acc_chk = _compute_acc_norm(
                    sw_chk, du_chk, dp_chk, u_cand, p_cand, use_weighted_step
                )

                u_chk = u_cand + du_chk
                p_chk = p_cand + dp_chk
                f_chk = _eval_res(u_chk, p_chk)
                arc_chk = _eval_arc(u_chk, p_chk)
                rn_chk = jnp.sqrt(jnp.sum(f_chk**2) + arc_chk**2)

                chk_valid = jnp.all(jnp.isfinite(f_chk)) & jnp.isfinite(arc_chk) & lin_chk_ok
                chk_ok = chk_valid & (acc_chk <= acc_tol)

                can_est_om_chk = (
                    chk_ok
                    & (it == 0)
                    & (step_norm_w > 0.0)
                    & ((step_norm_w * step_norm_w) > 0.0)
                    & (sw_chk > eps_floor)
                )
                om_chk = 2.0 * sw_chk / jnp.where(can_est_om_chk, step_norm_w * step_norm_w, 1.0)
                om_sel_chk = jnp.where(can_est_om_chk, om_chk, omega_next)
                has_om_sel_chk = has_om_next | can_est_om_chk

                return (
                    jnp.asarray(chk_ok, dtype=bool),
                    jnp.asarray(jnp.where(chk_ok, u_chk, u_cand), dtype=jnp.float64),
                    jnp.asarray(jnp.where(chk_ok, p_chk, p_cand), dtype=jnp.float64),
                    jnp.asarray(jnp.where(chk_ok, f_chk, f_next), dtype=jnp.float64),
                    jnp.asarray(jnp.where(chk_ok, arc_chk, arc_next), dtype=jnp.float64),
                    jnp.asarray(jnp.where(chk_ok, rn_chk, res_norm_next), dtype=jnp.float64),
                    jnp.asarray(jnp.where(chk_ok, sw_chk, step_norm_w), dtype=jnp.float64),
                    jnp.asarray(om_sel_chk, dtype=jnp.float64),
                    jnp.asarray(has_om_sel_chk, dtype=bool),
                )

            def _no_final_check() -> tuple[jax.Array, ...]:
                return (
                    jnp.bool_(False),
                    jnp.asarray(u_cand, dtype=jnp.float64),
                    jnp.asarray(p_cand, dtype=jnp.float64),
                    jnp.asarray(f_next, dtype=jnp.float64),
                    jnp.asarray(arc_next, dtype=jnp.float64),
                    jnp.asarray(res_norm_next, dtype=jnp.float64),
                    jnp.asarray(step_norm_w, dtype=jnp.float64),
                    jnp.asarray(omega_next, dtype=jnp.float64),
                    jnp.asarray(has_om_next, dtype=bool),
                )

            chk_ok, u_sel, p_sel, f_sel, arc_sel, rn_sel, sw_sel, om_sel, has_om_sel = jax.lax.cond(
                need_final_check, _do_final_check, _no_final_check
            )
        else:
            chk_ok = jnp.bool_(False)
            u_sel = u_cand
            p_sel = p_cand
            f_sel = f_next
            arc_sel = arc_next
            rn_sel = res_norm_next
            sw_sel = step_norm_w
            om_sel = omega_next
            has_om_sel = has_om_next

        conv_next = direct_ok | chk_ok

        if enforce_contraction:
            abort_i0 = (
                (it == 0)
                & (~first_step_arr)
                & (omega_init_arr > 0.0)
                & (contraction_a * step_norm_w * omega_init_arr > h_a)
            )
            abort_contraction = (it >= 1) & (theta_next > a_bar)
            aborted_next = (~is_valid) | ((~conv_next) & (abort_i0 | abort_contraction))
        else:
            aborted_next = ~is_valid

        u_next = jnp.where(is_valid, u_sel, u)
        p_next = jnp.where(is_valid, p_sel, p)
        f_out = jnp.where(is_valid, f_sel, f_val)
        arc_out = jnp.where(is_valid, arc_sel, arc_res)
        rn_out = jnp.where(is_valid, rn_sel, rn_cur)
        return (
            u_next,
            p_next,
            f_out,
            arc_out,
            it + jnp.int32(1),
            conv_next,
            aborted_next,
            rn_out,
            sw_sel,
            norm_dz0_next,
            om_sel,
            theta_next,
            a_bar_next,
            has_om_sel,
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
        has_om_fin,
    ) = final_state

    omega_decay = jnp.where(
        omega_init_arr > 0.0,
        _OMEGA_DECAY_FACTOR * omega_init_arr,
        jnp.array(1.0, dtype=jnp.float64),
    )
    omega_out = jnp.where(conv_fin & (~has_om_fin), omega_decay, omega_fin)

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
