from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array

from bifurx.problem import BifurcationProblem
from bifurx.solvers.bordered import solve_bordered_system


class NewtonResultJax(NamedTuple):
    """Result of pure JAX bordered Newton corrector iterations."""

    converged: Array
    u: Array
    p: Array
    iterations: Array
    residual_norm: Array
    det_u: Array


def bordered_newton_solve_jax(
    problem: BifurcationProblem,
    u_init: Array,
    p_init: float | Array,
    u_prev: Array,
    p_prev: float | Array,
    tau_prev: Array,
    ds: float | Array,
    tol: float = 1e-8,
    step_tol: float = 1e-8,
    max_iters: int = 15,
    solver_method: str = "auto",
) -> NewtonResultJax:
    """Pure JAX bordered Newton corrector for pseudo-arclength continuation.

    Solves the augmented system:
        F(u, p) = 0
        <tau_u, u - u_prev> + tau_p * (p - p_prev) - ds = 0

    Implemented using `jax.lax.while_loop` without host synchronization,
    enabling end-to-end XLA compilation and vmap batching.

    Parameters
    ----------
    problem : BifurcationProblem
        Nonlinear system definition F(u, p) = 0.
    u_init : Array
        Predicted state vector for corrector start.
    p_init : float | Array
        Predicted continuation parameter value.
    u_prev : Array
        Previous converged state vector.
    p_prev : float | Array
        Previous converged parameter value.
    tau_prev : Array
        Tangent vector [tau_u, tau_p] normalized to unit length.
    ds : float | Array
        Step size along arclength.
    tol : float, default=1e-8
        Residual norm convergence tolerance.
    step_tol : float, default=1e-8
        Step size convergence tolerance.
    max_iters : int, default=15
        Maximum Newton iterations.
    solver_method : str, default="auto"
        Method for solving bordered linear systems: "auto", "block", or "direct".

    Returns
    -------
    NewtonResultJax
        NamedTuple containing JAX array results (converged, u, p, iterations, residual_norm, det_u).
    """
    n = problem.dim
    u_0 = jnp.asarray(u_init, dtype=jnp.float64)
    p_0 = jnp.squeeze(jnp.asarray(p_init, dtype=jnp.float64))
    u_p = jnp.asarray(u_prev, dtype=jnp.float64)
    p_p = jnp.squeeze(jnp.asarray(p_prev, dtype=jnp.float64))
    tau_u = tau_prev[:n]
    tau_p = tau_prev[-1]
    ds_val = jnp.squeeze(jnp.asarray(ds, dtype=jnp.float64))

    f_0 = problem.residual(u_0, p_0)
    arc_0 = jnp.dot(tau_u, u_0 - u_p) + tau_p * (p_0 - p_p) - ds_val
    res_norm_0 = jnp.sqrt(jnp.sum(f_0**2) + arc_0**2)
    conv_0 = res_norm_0 < tol

    init_state = (u_0, p_0, jnp.int32(0), conv_0, res_norm_0, jnp.array(0.0, dtype=jnp.float64))

    def cond_fn(state: tuple[Array, Array, Array, Array, Array, Array]) -> Array:
        _u, _p, it, conv, _rn, _sn = state
        return (~conv) & (it < max_iters)

    def body_fn(
        state: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
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
            method=solver_method,
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
            (res_norm_next < tol) | ((step_norm < step_tol) & (res_norm_next < 10.0 * tol))
        )
        it_next = jnp.where(is_valid, it + 1, jnp.int32(max_iters))
        return (u_next, p_next, it_next, conv_next, res_norm_next, step_norm)

    u_fin, p_fin, it_fin, conv_fin, res_fin, _step_fin = jax.lax.while_loop(
        cond_fn, body_fn, init_state
    )

    Ju_fin = problem.jacobian_u(u_fin, p_fin)
    sign, logdet = jnp.linalg.slogdet(Ju_fin)
    det_u = sign * jnp.exp(jnp.clip(logdet, -100.0, 100.0))

    return NewtonResultJax(
        converged=conv_fin,
        u=u_fin,
        p=p_fin,
        iterations=it_fin,
        residual_norm=res_fin,
        det_u=det_u,
    )


@dataclass
class NewtonResult:
    """Result of bordered Newton corrector iterations."""

    converged: bool
    u: Array
    p: float
    iterations: int
    residual_norm: float
    residuals: list[float] = field(default_factory=list)
    det_u: float = 0.0
    quadratic_ratios: list[float] = field(default_factory=list)


def bordered_newton_solve(
    problem: BifurcationProblem,
    u_init: Array,
    p_init: float | Array,
    u_prev: Array,
    p_prev: float | Array,
    tau_prev: Array,
    ds: float | Array,
    tol: float = 1e-8,
    step_tol: float = 1e-8,
    max_iters: int = 15,
    solver_method: str = "auto",
    use_jit: bool = False,
) -> NewtonResult:
    """Bordered Newton corrector for pseudo-arclength continuation.

    Solves the augmented system:
        F(u, p) = 0
        <tau_u, u - u_prev> + tau_p * (p - p_prev) - ds = 0

    Parameters
    ----------
    problem : BifurcationProblem
        Nonlinear system definition F(u, p) = 0.
    u_init : Array
        Predicted state vector for corrector start.
    p_init : float
        Predicted continuation parameter value.
    u_prev : Array
        Previous converged state vector.
    p_prev : float
        Previous converged parameter value.
    tau_prev : Array
        Tangent vector [tau_u, tau_p] normalized to unit length.
    ds : float
        Step size along arclength.
    tol : float, default=1e-8
        Residual norm convergence tolerance.
    step_tol : float, default=1e-8
        Step size convergence tolerance.
    max_iters : int, default=15
        Maximum Newton iterations.
    solver_method : str, default="auto"
        Method for solving bordered linear systems: "auto", "block", or "direct".
    use_jit : bool, default=False
        Whether to use pure-JAX JIT-compiled corrector.

    Returns
    -------
    NewtonResult
        Dataclass containing convergence status, solution, iterations, and residuals.
    """
    if use_jit or isinstance(u_init, jax.core.Tracer) or isinstance(p_init, jax.core.Tracer):
        res_jax = bordered_newton_solve_jax(
            problem=problem,
            u_init=u_init,
            p_init=p_init,
            u_prev=u_prev,
            p_prev=p_prev,
            tau_prev=tau_prev,
            ds=ds,
            tol=tol,
            step_tol=step_tol,
            max_iters=max_iters,
            solver_method=solver_method,
        )
        if isinstance(u_init, jax.core.Tracer) or isinstance(p_init, jax.core.Tracer):
            return NewtonResult(
                converged=res_jax.converged,  # type: ignore[arg-type]
                u=res_jax.u,
                p=res_jax.p,  # type: ignore[arg-type]
                iterations=res_jax.iterations,  # type: ignore[arg-type]
                residual_norm=res_jax.residual_norm,  # type: ignore[arg-type]
                det_u=res_jax.det_u,  # type: ignore[arg-type]
            )
        return NewtonResult(
            converged=bool(res_jax.converged),
            u=res_jax.u,
            p=float(res_jax.p),
            iterations=int(res_jax.iterations),
            residual_norm=float(res_jax.residual_norm),
            det_u=float(res_jax.det_u),
        )

    u_cur = jnp.asarray(u_init, dtype=jnp.float64)
    p_cur = float(p_init)

    u_p = jnp.asarray(u_prev, dtype=jnp.float64)
    tau_u = tau_prev[: problem.dim]
    tau_p = float(tau_prev[-1])

    residuals: list[float] = []
    quadratic_ratios: list[float] = []
    converged = False
    det_u = 0.0

    for it in range(max_iters):
        # 1. Evaluate residuals
        f_val = problem.residual(u_cur, p_cur)
        arc_res = float(jnp.dot(tau_u, u_cur - u_p) + tau_p * (p_cur - p_prev) - ds)
        res_norm = float(jnp.sqrt(jnp.sum(f_val**2) + arc_res**2))
        residuals.append(res_norm)

        # Check quadratic convergence ratio r_{k+1} / r_k^2
        if it >= 1 and residuals[-2] > 1e-12:
            q_ratio = res_norm / (residuals[-2] ** 2)
            quadratic_ratios.append(float(q_ratio))

        # Check convergence on residual
        if res_norm < tol:
            converged = True
            break

        # 2. Evaluate Jacobians
        Ju = problem.jacobian_u(u_cur, p_cur)
        Jp = problem.jacobian_p(u_cur, p_cur)

        # 3. Solve bordered linear system:
        # [[Ju, Jp], [tau_u^T, tau_p]] [du, dp]^T = -[f_val, arc_res]^T
        sol = solve_bordered_system(
            A=Ju,
            b=Jp,
            c=tau_u,
            d=tau_p,
            f=-f_val,
            g=-arc_res,
            method=solver_method,
        )
        du = sol.x
        dp = float(sol.y)

        step_norm = float(jnp.sqrt(jnp.sum(du**2) + dp**2))
        if not (jnp.isfinite(step_norm) and jnp.all(jnp.isfinite(du)) and jnp.isfinite(dp)):
            converged = False
            break

        # Update
        u_cur = u_cur + du
        p_cur = p_cur + dp

        if step_norm < step_tol:
            # Re-evaluate residual at final step
            f_final = problem.residual(u_cur, p_cur)
            arc_final = float(jnp.dot(tau_u, u_cur - u_p) + tau_p * (p_cur - p_prev) - ds)
            final_res = float(jnp.sqrt(jnp.sum(f_final**2) + arc_final**2))
            residuals.append(final_res)
            converged = final_res < tol * 10
            break

    if converged:
        Ju_conv = problem.jacobian_u(u_cur, p_cur)
        sign, logdet = jnp.linalg.slogdet(Ju_conv)
        det_u = float(sign * jnp.exp(jnp.clip(logdet, -100.0, 100.0)))

    return NewtonResult(
        converged=converged,
        u=u_cur,
        p=p_cur,
        iterations=len(residuals) - 1 if converged else max_iters,
        residual_norm=residuals[-1] if residuals else float("inf"),
        residuals=residuals,
        det_u=det_u,
        quadratic_ratios=quadratic_ratios,
    )
