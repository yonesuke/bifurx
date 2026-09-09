from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
from jaxtyping import Array

from bifurx.problem import BifurcationProblem
from bifurx.solvers.bordered import solve_bordered_system


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
    p_init: float,
    u_prev: Array,
    p_prev: float,
    tau_prev: Array,
    ds: float,
    tol: float = 1e-8,
    step_tol: float = 1e-8,
    max_iters: int = 15,
    solver_method: str = "auto",
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

    Returns
    -------
    NewtonResult
        Dataclass containing convergence status, solution, iterations, and residuals.
    """
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
