from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from bifurx.options import NewtonOptions
from bifurx.problem import BifurcationProblem
from bifurx.solvers.bordered import solve_bordered_system


class NewtonResult(NamedTuple):
    """Result of pure-JAX bordered Newton corrector iterations."""

    converged: jax.Array
    u: jax.Array
    p: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array
    det_u: jax.Array


def bordered_newton_solve(
    problem: BifurcationProblem,
    u_init: jax.Array,
    p_init: float | jax.Array,
    u_prev: jax.Array,
    p_prev: float | jax.Array,
    tau_prev: jax.Array,
    ds: float | jax.Array,
    tol: float | None = None,
    step_tol: float | None = None,
    max_iters: int | None = None,
    solver_method: str | None = None,
    options: NewtonOptions | None = None,
) -> NewtonResult:
    """Pure-JAX bordered Newton corrector for pseudo-arclength continuation.

    Solves the augmented system:
        F(u, p) = 0
        <tau_u, u - u_prev> + tau_p * (p - p_prev) - ds = 0

    Implemented end-to-end using `jax.lax.while_loop` without host synchronization,
    enabling XLA compilation (`jax.jit`) and batching (`jax.vmap`).

    Parameters
    ----------
    problem : BifurcationProblem
        Nonlinear system definition F(u, p) = 0.
    u_init : jax.Array
        Predicted state vector for corrector start.
    p_init : float | jax.Array
        Predicted continuation parameter value.
    u_prev : jax.Array
        Previous converged state vector.
    p_prev : float | jax.Array
        Previous converged parameter value.
    tau_prev : jax.Array
        Tangent vector [tau_u, tau_p] normalized to unit length.
    ds : float | jax.Array
        Step size along arclength.
    tol : float | None, optional
        Residual norm convergence tolerance (defaults to options.tol).
    step_tol : float | None, optional
        Step size convergence tolerance (defaults to options.step_tol).
    max_iters : int | None, optional
        Maximum Newton iterations (defaults to options.max_iters).
    solver_method : str | None, optional
        Method for solving bordered linear systems: "auto", "block", or "direct".
    options : NewtonOptions | None, optional
        Newton solver options instance. Overridden by explicit kwargs.

    Returns
    -------
    NewtonResult
        NamedTuple containing JAX array results (converged, u, p, iterations, residual_norm, det_u).
    """
    base_opts = NewtonOptions() if options is None else options
    tol_val = base_opts.tol if tol is None else tol
    step_tol_val = base_opts.step_tol if step_tol is None else step_tol
    max_iters_val = base_opts.max_iters if max_iters is None else max_iters
    solver_method_val = base_opts.solver_method if solver_method is None else solver_method

    n = problem.dim
    u_0 = jnp.asarray(u_init, dtype=jnp.float64)
    p_0 = jnp.squeeze(jnp.asarray(p_init, dtype=jnp.float64))
    u_p = jnp.asarray(u_prev, dtype=jnp.float64)
    p_p = jnp.squeeze(jnp.asarray(p_prev, dtype=jnp.float64))
    tau_arr = jnp.asarray(tau_prev, dtype=jnp.float64)
    tau_u = tau_arr[:n]
    tau_p = tau_arr[-1]
    ds_val = jnp.squeeze(jnp.asarray(ds, dtype=jnp.float64))

    f_0 = problem.residual(u_0, p_0)
    arc_0 = jnp.dot(tau_u, u_0 - u_p) + tau_p * (p_0 - p_p) - ds_val
    res_norm_0 = jnp.sqrt(jnp.sum(f_0**2) + arc_0**2)
    conv_0 = res_norm_0 < tol_val

    init_state = (u_0, p_0, jnp.int32(0), conv_0, res_norm_0, jnp.array(0.0, dtype=jnp.float64))

    def cond_fn(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        _u, _p, it, conv, _rn, _sn = state
        return (~conv) & (it < max_iters_val)

    def body_fn(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        u, p, it, _conv, _rn, _sn = state
        f_val = problem.residual(u, p)
        arc_res = jnp.dot(tau_u, u - u_p) + tau_p * (p - p_p) - ds_val

        Ju = problem.jacobian_u(u, p)
        Jp = problem.jacobian_p(u, p)

        sol = solve_bordered_system(
            A=Ju,
            b=Jp,
            c=tau_u,
            d=tau_p,
            f=-f_val,
            g=-arc_res,
            method=solver_method_val,
        )
        du = sol.x
        dp = jnp.squeeze(sol.y)

        step_norm = jnp.sqrt(jnp.sum(du**2) + dp**2)
        u_next = u + du
        p_next = p + dp

        f_next = problem.residual(u_next, p_next)
        arc_next = jnp.dot(tau_u, u_next - u_p) + tau_p * (p_next - p_p) - ds_val
        res_norm_next = jnp.sqrt(jnp.sum(f_next**2) + arc_next**2)

        is_valid = (
            jnp.isfinite(res_norm_next)
            & jnp.isfinite(step_norm)
            & jnp.all(jnp.isfinite(du))
            & jnp.isfinite(dp)
        )
        conv_next = is_valid & (
            (res_norm_next < tol_val)
            | ((step_norm < step_tol_val) & (res_norm_next < 10.0 * tol_val))
        )
        it_next = jnp.where(is_valid, it + 1, jnp.int32(max_iters_val))
        return (u_next, p_next, it_next, conv_next, res_norm_next, step_norm)

    u_fin, p_fin, it_fin, conv_fin, res_fin, _step_fin = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )

    Ju_fin = problem.jacobian_u(u_fin, p_fin)
    sign, logdet = jnp.linalg.slogdet(Ju_fin)
    det_u = sign * jnp.exp(jnp.clip(logdet, -80.0, 80.0))

    return NewtonResult(
        converged=conv_fin,
        u=u_fin,
        p=p_fin,
        iterations=it_fin,
        residual_norm=res_fin,
        det_u=det_u,
    )
