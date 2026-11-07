from __future__ import annotations

import jax
import jax.numpy as jnp

from bifurx.bvp.collocation import PeriodicOrbitResult, solve_periodic_orbit


def _equidistribute_mesh(
    sol: PeriodicOrbitResult,
    alpha: float = 1.0,
    beta: float = 0.0,
    damping: float = 0.7,
) -> jax.Array:
    """Compute an equidistributed adaptive mesh based on solution gradients (AUTO-07P ADPTDS)."""
    mesh = jnp.asarray(sol.mesh, dtype=jnp.float64)
    u_mesh = jnp.asarray(sol.u_mesh, dtype=jnp.float64)
    u_gauss = jnp.asarray(sol.u_gauss, dtype=jnp.float64)
    D = jnp.asarray(sol.problem.D, dtype=jnp.float64)
    weights = jnp.asarray(sol.problem.weights, dtype=jnp.float64)
    N = sol.problem.num_intervals
    m = sol.problem.num_gauss_points

    h = mesh[1:] - mesh[:-1]
    U = jnp.concatenate([u_mesh[:, None, :], u_gauss], axis=1)
    du_dt = jnp.einsum("mj,Njd->Nmd", D, U) / h[:, None, None]
    grad_norm2 = jnp.sum(du_dt**2, axis=-1)

    if beta > 0.0:
        d2u_dt2 = jnp.asarray(jnp.gradient(du_dt, axis=1)) / (h[:, None, None] / m)
        curv_term = beta * (jnp.sum(d2u_dt2**2, axis=-1) ** (1.0 / 3.0))
    else:
        curv_term = 0.0

    w_gauss = jnp.sqrt(1.0 + alpha * grad_norm2 + curv_term)
    I_k = h * (w_gauss @ weights)

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
    """Adaptively redistribute collocation mesh and re-solve orbit on the new mesh in pure JAX."""
    new_mesh = _equidistribute_mesh(sol, alpha=alpha, beta=beta, damping=damping)
    N = sol.problem.num_intervals
    m = sol.problem.num_gauss_points
    n = sol.problem.dim
    rho = jnp.asarray(sol.problem.rho, dtype=jnp.float64)

    u_mesh_new = sol.evaluate(new_mesh[:-1])
    hk = new_mesh[1:] - new_mesh[:-1]
    t_gauss = (new_mesh[:-1, None] + hk[:, None] * rho[None, :]).ravel()
    u_gauss_new = sol.evaluate(t_gauss).reshape(N, m, n)

    if not re_solve:
        p_val = sol.p if sol.p is not None else jnp.asarray(p, dtype=jnp.float64)
        return PeriodicOrbitResult(
            mesh=new_mesh,
            u_mesh=u_mesh_new,
            u_gauss=u_gauss_new,
            period=sol.period,
            converged=sol.converged,
            iterations=jnp.int32(0),
            residual_norm=sol.residual_norm,
            problem=sol.problem,
            p=p_val,
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
