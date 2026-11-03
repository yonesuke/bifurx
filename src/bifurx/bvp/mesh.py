from __future__ import annotations

import jax
import jax.numpy as jnp

from bifurx.bvp.collocation import PeriodicOrbitResult, solve_periodic_orbit


def equidistribute_mesh(
    sol: PeriodicOrbitResult,
    alpha: float = 1.0,
    beta: float = 0.0,
    damping: float = 0.7,
) -> jax.Array:
    """Compute an equidistributed adaptive mesh based on solution gradients (AUTO-07P ADPTDS).

    Parameters
    ----------
    sol : PeriodicOrbitResult
        Converged periodic orbit result.
    alpha : float, default=1.0
        Weight for solution gradient ||du/dt||^2 in monitor function.
    beta : float, default=0.0
        Weight for solution curvature ||d^2u/dt^2||^(2/3) in monitor function.
    damping : float, default=0.7
        Relaxation damping factor with previous mesh (in (0, 1]).

    Returns
    -------
    jax.Array
        New mesh of shape (N + 1,) on [0, 1].
    """
    mesh = jnp.asarray(sol.mesh)
    u_mesh = jnp.asarray(sol.u_mesh)
    u_gauss = jnp.asarray(sol.u_gauss)
    D = jnp.asarray(sol.problem.D)
    weights = jnp.asarray(sol.problem.weights)
    N = sol.problem.num_intervals
    m = sol.problem.num_gauss_points

    # Compute monitor function on each interval using Gauss points
    monitor = []
    for k in range(N):
        hk = mesh[k + 1] - mesh[k]
        U_k = jnp.vstack([u_mesh[k][None, :], u_gauss[k]])  # (m + 1, n)
        # Derivatives at Gauss points
        du_dt = (D @ U_k) / hk  # shape (m, n)

        # Norm of first derivative
        grad_norm2 = jnp.sum(du_dt**2, axis=1)  # shape (m,)

        # Approximate curvature
        if beta > 0.0:
            # Finite differences of gradient within interval
            d2u_dt2 = jnp.asarray(jnp.gradient(du_dt, axis=0)) / (hk / m)
            curv_term = beta * (jnp.sum(d2u_dt2**2, axis=1) ** (1.0 / 3.0))
        else:
            curv_term = 0.0

        w_gauss = jnp.sqrt(1.0 + alpha * grad_norm2 + curv_term)
        monitor.append(hk * jnp.dot(weights, w_gauss))
    I_k = jnp.stack(monitor)

    # Cumulative monitor function at mesh nodes
    W = jnp.concatenate([jnp.zeros(1, dtype=I_k.dtype), jnp.cumsum(I_k)])
    W_total = W[-1]

    if float(W_total) <= 0.0 or not bool(jnp.isfinite(W_total)):
        return mesh

    # Equidistant monitor targets
    W_targets = jnp.linspace(0.0, W_total, N + 1)

    # Invert W(t) by piecewise linear interpolation
    t_equi = jnp.interp(W_targets, W, mesh)
    t_equi = t_equi.at[0].set(0.0)
    t_equi = t_equi.at[-1].set(1.0)

    # Ensure strictly increasing
    t_equi = jnp.maximum.accumulate(t_equi)

    # Damping with current mesh: t_new = (1 - damping) * mesh + damping * t_equi
    damping = float(jnp.clip(damping, 0.0, 1.0))
    t_new = (1.0 - damping) * mesh + damping * t_equi
    t_new = t_new.at[0].set(0.0)
    t_new = t_new.at[-1].set(1.0)

    # Ensure minimum spacing to avoid degenerate intervals
    min_h = 1e-4 / N
    for i in range(1, N + 1):
        if float(t_new[i]) <= float(t_new[i - 1]) + min_h:
            t_new = t_new.at[i].set(t_new[i - 1] + min_h)
    t_new = t_new.at[-1].set(1.0)

    return t_new


def redistribute_mesh(
    sol: PeriodicOrbitResult,
    alpha: float = 1.0,
    beta: float = 0.0,
    damping: float = 0.7,
    p: float | jax.Array = 0.0,
    re_solve: bool = True,
    tol: float = 1e-8,
    max_iters: int = 15,
) -> PeriodicOrbitResult:
    """Adaptively redistribute collocation mesh and re-solve orbit on the new mesh.

    Parameters
    ----------
    sol : PeriodicOrbitResult
        Current converged periodic orbit solution.
    alpha : float, default=1.0
        Gradient weight in monitor function.
    beta : float, default=0.0
        Curvature weight in monitor function.
    damping : float, default=0.7
        Damping factor in [0, 1].
    p : float | jax.Array, default=0.0
        System parameter value.
    re_solve : bool, default=True
        Whether to run Newton polish on the new mesh.
    tol : float, default=1e-8
        Convergence tolerance if re-solving.
    max_iters : int, default=15
        Maximum iterations if re-solving.

    Returns
    -------
    PeriodicOrbitResult
        Orbit on the redistributed adaptive mesh.
    """
    new_mesh = equidistribute_mesh(sol, alpha=alpha, beta=beta, damping=damping)
    N = sol.problem.num_intervals
    m = sol.problem.num_gauss_points
    rho = jnp.asarray(sol.problem.rho)

    # Interpolate solution onto new mesh nodes and Gauss points
    u_mesh_new = jnp.stack([sol.evaluate(t) for t in new_mesh[:-1]])

    gauss_rows = []
    for k in range(N):
        hk = new_mesh[k + 1] - new_mesh[k]
        gauss_rows.append(jnp.stack([sol.evaluate(new_mesh[k] + hk * rho[i]) for i in range(m)]))
    u_gauss_new = jnp.stack(gauss_rows)

    if not re_solve:
        return PeriodicOrbitResult(
            mesh=new_mesh,
            u_mesh=u_mesh_new,
            u_gauss=u_gauss_new,
            period=sol.period,
            converged=sol.converged,
            iterations=0,
            residual_norm=sol.residual_norm,
            problem=sol.problem,
        )

    # Re-solve orbit on new mesh
    return solve_periodic_orbit(
        problem=sol.problem,
        u_mesh_init=u_mesh_new,
        T_init=sol.period,
        p=p,
        mesh=new_mesh,
        u_gauss_init=u_gauss_new,
        tol=tol,
        max_iters=max_iters,
    )
