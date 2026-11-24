from __future__ import annotations

import jax
import jax.numpy as jnp

from bifurx.bvp.collocation import (
    PeriodicOrbitResult,
    periodic_orbit_arclength_weights,
)
from bifurx.bvp.condensation import make_collocation_bordered_solver
from bifurx.bvp.floquet import (
    evaluate_floquet_diagnostics_from_factors,
    monodromy_factors_condensation,
)
from bifurx.homotopy.corrector import correct_arclength_step


def _eval_periodic_bif_test_func(
    orbit_a: PeriodicOrbitResult,
    X_val: jax.Array,
    p_val: jax.Array,
    mesh: jax.Array,
    tau_ref: jax.Array,
    w_arc: jax.Array,
    dot_u_ref: jax.Array,
    bif_type: str,
) -> jax.Array:
    """Evaluate scalar periodic orbit bifurcation test function (LPC, PD, or NS) in pure JAX."""
    problem = orbit_a.problem
    if bif_type == "LPC":
        bordered_solver = make_collocation_bordered_solver(problem, mesh, dot_u_ref)
        c_bord = w_arc[:-1] * tau_ref[:-1]
        d_bord = w_arc[-1] * tau_ref[-1]
        fac = bordered_solver.factor(X_val, p_val, c_bord, d_bord)
        zero_X = jnp.zeros_like(X_val)
        tau_raw = bordered_solver.solve(fac, zero_X, jnp.array(1.0, dtype=jnp.float64)).z
        tau_norm = jnp.sqrt(jnp.maximum(jnp.dot(w_arc * tau_raw, tau_raw), 1e-30))
        tau_unit = tau_raw / tau_norm
        align = jnp.dot(w_arc * tau_unit, tau_ref)
        tau_aligned = jnp.where(align < 0.0, -tau_unit, tau_unit)
        return tau_aligned[-1]

    _, u_g, T_val = problem.unpack_vars(X_val)
    factors = monodromy_factors_condensation(problem, mesh, u_g, T_val, p=p_val)
    _, tf_dict, _ = evaluate_floquet_diagnostics_from_factors(factors)
    return tf_dict[bif_type]


def refine_periodic_bifurcation(
    orbit_a: PeriodicOrbitResult,
    orbit_b: PeriodicOrbitResult,
    bif_type: str,
    *,
    tol: float = 1e-9,
    max_iters: int = 24,
    newton_tol: float = 1e-9,
    newton_max_iters: int = 10,
    period_weight: float = 1.0,
    param_weight: float = 1.0,
) -> PeriodicOrbitResult:
    """Refine a bracketed periodic orbit bifurcation (LPC, PD, or NS) in pure JAX.

    Uses safeguarded secant/bisection along the arclength segment connecting ``orbit_a``
    and ``orbit_b``, projecting each trial point onto the periodic orbit branch via
    block-condensed bordered Newton. Fully compatible with ``jax.jit`` (with ``bif_type``
    static) and performs zero Python host synchronization.
    """
    btype = bif_type.upper()
    if btype not in ("LPC", "PD", "NS"):
        msg = f"Unknown bif_type: {bif_type}. Must be 'LPC', 'PD', or 'NS'."
        raise ValueError(msg)
    if orbit_a.p is None or orbit_b.p is None:
        msg = "Both orbit_a and orbit_b must have non-None parameter values `p`."
        raise ValueError(msg)

    problem = orbit_a.problem
    mesh = jnp.asarray(orbit_a.mesh, dtype=jnp.float64)
    h_seg = mesh[1:] - mesh[:-1]

    X_a = problem.pack_vars(orbit_a.u_mesh, orbit_a.u_gauss, orbit_a.period)
    X_b = problem.pack_vars(orbit_b.u_mesh, orbit_b.u_gauss, orbit_b.period)
    p_a = jnp.squeeze(jnp.asarray(orbit_a.p, dtype=jnp.float64))
    p_b = jnp.squeeze(jnp.asarray(orbit_b.p, dtype=jnp.float64))

    w_arc = periodic_orbit_arclength_weights(
        problem, mesh, period_weight=period_weight, param_weight=param_weight
    )
    dz = jnp.concatenate([X_b - X_a, jnp.atleast_1d(p_b - p_a)])
    ds_seg = jnp.sqrt(jnp.maximum(jnp.dot(w_arc * dz, dz), 1e-30))
    tau_seg = dz / ds_seg

    U_ref = jnp.concatenate([orbit_a.u_mesh[:, None, :], orbit_a.u_gauss], axis=1)
    dot_u_ref = jnp.einsum("mj,Njd->Nmd", problem.D, U_ref) / h_seg[:, None, None]
    u_ref_g = jnp.asarray(orbit_a.u_gauss, dtype=jnp.float64)
    bordered_solver = make_collocation_bordered_solver(problem, mesh, dot_u_ref)

    def _res_fn(X_v: jax.Array, p_v: jax.Array) -> jax.Array:
        return problem.residual(X_v, p_v, mesh, u_ref_gauss=u_ref_g, dot_u_ref=dot_u_ref)

    f_lo0 = _eval_periodic_bif_test_func(orbit_a, X_a, p_a, mesh, tau_seg, w_arc, dot_u_ref, btype)
    f_hi0 = _eval_periodic_bif_test_func(orbit_a, X_b, p_b, mesh, tau_seg, w_arc, dot_u_ref, btype)
    tol_arr = jnp.asarray(tol, dtype=jnp.float64)

    init_state = (
        jnp.array(0.0, dtype=jnp.float64),  # a_lo
        jnp.array(1.0, dtype=jnp.float64),  # a_hi
        f_lo0,
        f_hi0,
        0.5 * (X_a + X_b),  # X_best
        0.5 * (p_a + p_b),  # p_best
        jnp.minimum(jnp.abs(f_lo0), jnp.abs(f_hi0)),  # abs_tf_best
        jnp.array(jnp.inf, dtype=jnp.float64),  # res_norm_best
        jnp.bool_(False),  # conv_best
        jnp.int32(0),  # last_side (-1: left/hi moved, +1: right/lo moved, 0: init)
        jnp.int32(0),  # it
        jnp.bool_(False),  # done
    )

    def cond_fn(state: tuple[jax.Array, ...]) -> jax.Array:
        a_lo, a_hi, _, _, _, _, _, _, _, _, it, done = state
        return (~done) & (it < max_iters) & ((a_hi - a_lo) > 1e-12)

    def body_fn(state: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
        (
            a_lo,
            a_hi,
            f_lo,
            f_hi,
            X_best,
            p_best,
            abs_tf_best,
            rn_best,
            conv_best,
            last_side,
            it,
            _,
        ) = state
        span = a_hi - a_lo
        denom = f_hi - f_lo
        safe_denom = jnp.where(jnp.abs(denom) > 1e-30, denom, 1.0)
        a_sec = jnp.where(
            jnp.abs(denom) > 1e-30, a_lo - f_lo * span / safe_denom, 0.5 * (a_lo + a_hi)
        )
        margin = 0.05 * span
        use_sec = (a_sec > a_lo + margin) & (a_sec < a_hi - margin)
        alpha = jnp.where(use_sec, a_sec, 0.5 * (a_lo + a_hi))

        X_guess = (1.0 - alpha) * X_a + alpha * X_b
        p_guess = (1.0 - alpha) * p_a + alpha * p_b
        pol = correct_arclength_step(
            residual_fn=_res_fn,
            u_init=X_guess,
            p_init=p_guess,
            u_prev=X_a,
            p_prev=p_a,
            tau_prev=tau_seg,
            ds=alpha * ds_seg,
            tol=newton_tol,
            step_tol=newton_tol,
            rel_tol=min(newton_tol, 1e-10),
            max_iters=newton_max_iters,
            border_c=w_arc[:-1] * tau_seg[:-1],
            border_d=w_arc[-1] * tau_seg[-1],
            bordered_solver=bordered_solver,
        )
        pol_ok = pol.converged & jnp.all(jnp.isfinite(pol.u)) & jnp.isfinite(pol.p)
        X_cand = jnp.where(pol_ok, pol.u, X_guess)
        p_cand = jnp.where(pol_ok, pol.p, p_guess)
        f_mid = _eval_periodic_bif_test_func(
            orbit_a, X_cand, p_cand, mesh, tau_seg, w_arc, dot_u_ref, btype
        )
        valid_mid = jnp.isfinite(f_mid)

        # Illinois-modified secant update prevents stalling on strongly convex test functions
        go_left = (f_lo * f_mid) <= 0.0
        cur_side = jnp.where(go_left, jnp.int32(-1), jnp.int32(1))
        same_side = cur_side == last_side
        a_lo_new = jnp.where(valid_mid & (~go_left), alpha, a_lo)
        f_lo_new = jnp.where(
            valid_mid & (~go_left),
            f_mid,
            jnp.where(valid_mid & go_left & same_side, 0.5 * f_lo, f_lo),
        )
        a_hi_new = jnp.where(valid_mid & go_left, alpha, a_hi)
        f_hi_new = jnp.where(
            valid_mid & go_left,
            f_mid,
            jnp.where(valid_mid & (~go_left) & same_side, 0.5 * f_hi, f_hi),
        )
        last_side_new = jnp.where(valid_mid, cur_side, last_side)

        abs_f_mid = jnp.abs(f_mid)
        better = valid_mid & (abs_f_mid <= abs_tf_best)
        X_best_new = jnp.where(better, X_cand, X_best)
        p_best_new = jnp.where(better, p_cand, p_best)
        abs_tf_new = jnp.where(better, abs_f_mid, abs_tf_best)
        rn_new = jnp.where(better, pol.residual_norm, rn_best)
        conv_new = jnp.where(better, pol_ok, conv_best)

        done_new = (~valid_mid) | (abs_f_mid <= tol_arr)
        return (
            a_lo_new,
            a_hi_new,
            f_lo_new,
            f_hi_new,
            X_best_new,
            p_best_new,
            abs_tf_new,
            rn_new,
            conv_new,
            last_side_new,
            it + jnp.int32(1),
            done_new,
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    _, _, _, _, X_fin, p_fin, _, rn_fin, conv_fin, _, it_fin, _ = final_state
    u_m_fin, u_g_fin, T_fin = problem.unpack_vars(X_fin)

    factors_fin = monodromy_factors_condensation(problem, mesh, u_g_fin, T_fin, p=p_fin)
    mults_fin, _, stable_fin = evaluate_floquet_diagnostics_from_factors(factors_fin)

    return PeriodicOrbitResult(
        mesh=mesh,
        u_mesh=u_m_fin,
        u_gauss=u_g_fin,
        period=T_fin,
        converged=conv_fin,
        iterations=it_fin,
        residual_norm=rn_fin,
        problem=problem,
        p=p_fin,
        floquet_multipliers=mults_fin,
        is_stable=stable_fin,
        bifurcations=(btype,),
    )
