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

    Vectorized across all intervals and Gauss nodes in pure JAX without host synchronization.

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
    mesh = jnp.asarray(sol.mesh, dtype=jnp.float64)
    u_mesh = jnp.asarray(sol.u_mesh, dtype=jnp.float64)
    u_gauss = jnp.asarray(sol.u_gauss, dtype=jnp.float64)
    D = jnp.asarray(sol.problem.D, dtype=jnp.float64)
    weights = jnp.asarray(sol.problem.weights, dtype=jnp.float64)
    N = sol.problem.num_intervals
    m = sol.problem.num_gauss_points

    h = mesh[1:] - mesh[:-1]  # (N,)
    U = jnp.concatenate([u_mesh[:, None, :], u_gauss], axis=1)  # (N, m + 1, n)
    du_dt = jnp.einsum("mj,Njd->Nmd", D, U) / h[:, None, None]  # (N, m, n)
    grad_norm2 = jnp.sum(du_dt**2, axis=-1)  # (N, m)

    if beta > 0.0:
        d2u_dt2 = jnp.asarray(jnp.gradient(du_dt, axis=1)) / (h[:, None, None] / m)
        curv_term = beta * (jnp.sum(d2u_dt2**2, axis=-1) ** (1.0 / 3.0))
    else:
        curv_term = 0.0

    w_gauss = jnp.sqrt(1.0 + alpha * grad_norm2 + curv_term)  # (N, m)
    I_k = h * (w_gauss @ weights)  # (N,)

    W = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(I_k)])
    W_total = W[-1]
    is_valid = (W_total > 0.0) & jnp.isfinite(W_total)

    W_targets = jnp.linspace(0.0, jnp.where(is_valid, W_total, 1.0), N + 1)
    t_equi = jnp.interp(W_targets, jnp.where(is_valid, W, mesh), mesh)
    t_equi = jnp.maximum.accumulate(t_equi.at[0].set(0.0).at[-1].set(1.0))

    damping_val = jnp.clip(jnp.asarray(damping, dtype=jnp.float64), 0.0, 1.0)
    t_new = (1.0 - damping_val) * mesh + damping_val * t_equi

    min_step = jnp.arange(N + 1, dtype=jnp.float64) * (1e-4 / N)
    t_new = jnp.maximum.accumulate(t_new - min_step) + min_step
    t_new = t_new.at[0].set(0.0).at[-1].set(1.0)

    return jnp.where(is_valid, t_new, mesh)


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
    """Adaptively redistribute collocation mesh and re-solve orbit on the new mesh in pure JAX.

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
    n = sol.problem.dim
    rho = jnp.asarray(sol.problem.rho, dtype=jnp.float64)

    u_mesh_new = sol.evaluate(new_mesh[:-1])
    hk = new_mesh[1:] - new_mesh[:-1]
    t_gauss = (new_mesh[:-1, None] + hk[:, None] * rho[None, :]).ravel()
    u_gauss_new = sol.evaluate(t_gauss).reshape(N, m, n)

    if not re_solve:
        return PeriodicOrbitResult(
            mesh=new_mesh,
            u_mesh=u_mesh_new,
            u_gauss=u_gauss_new,
            period=sol.period,
            converged=sol.converged,
            iterations=jnp.int32(0),
            residual_norm=sol.residual_norm,
            problem=sol.problem,
        )

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
