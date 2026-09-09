from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from bifurx.bvp.collocation import PeriodicOrbitResult, solve_periodic_orbit


def equidistribute_mesh(
    sol: PeriodicOrbitResult,
    alpha: float = 1.0,
    beta: float = 0.0,
    damping: float = 0.7,
) -> np.ndarray:
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
    np.ndarray
        New mesh of shape (N + 1,) on [0, 1].
    """
    mesh = np.asarray(sol.mesh)
    u_mesh = np.asarray(sol.u_mesh)
    u_gauss = np.asarray(sol.u_gauss)
    D = np.asarray(sol.problem.D)
    weights = np.asarray(sol.problem.weights)
    N = sol.problem.num_intervals
    m = sol.problem.num_gauss_points

    # Compute monitor function on each interval using Gauss points
    I_k = np.zeros(N, dtype=np.float64)
    for k in range(N):
        hk = mesh[k + 1] - mesh[k]
        U_k = np.vstack([u_mesh[k][None, :], u_gauss[k]])  # (m + 1, n)
        # Derivatives at Gauss points
        du_dt = (D @ U_k) / hk  # shape (m, n)

        # Norm of first derivative
        grad_norm2 = np.sum(du_dt**2, axis=1)  # shape (m,)

        # Approximate curvature
        if beta > 0.0:
            # Finite differences of gradient within interval
            d2u_dt2 = np.gradient(du_dt, axis=0) / (hk / m)
            curv_term = beta * (np.sum(d2u_dt2**2, axis=1) ** (1.0 / 3.0))
        else:
            curv_term = 0.0

        w_gauss = np.sqrt(1.0 + alpha * grad_norm2 + curv_term)
        I_k[k] = hk * np.dot(weights, w_gauss)

    # Cumulative monitor function at mesh nodes
    W = np.zeros(N + 1, dtype=np.float64)
    W[1:] = np.cumsum(I_k)
    W_total = W[-1]

    if W_total <= 0.0 or not np.isfinite(W_total):
        return mesh

    # Equidistant monitor targets
    W_targets = np.linspace(0.0, W_total, N + 1)

    # Invert W(t) by piecewise linear interpolation
    t_equi = np.asarray(np.interp(W_targets, W, mesh), dtype=np.float64)
    t_equi[0] = 0.0
    t_equi[-1] = 1.0

    # Ensure strictly increasing
    t_equi = np.maximum.accumulate(t_equi)

    # Damping with current mesh: t_new = (1 - damping) * mesh + damping * t_equi
    damping = float(np.clip(damping, 0.0, 1.0))
    t_new = (1.0 - damping) * mesh + damping * t_equi
    t_new[0] = 0.0
    t_new[-1] = 1.0

    # Ensure minimum spacing to avoid degenerate intervals
    min_h = 1e-4 / N
    for i in range(1, N + 1):
        if t_new[i] <= t_new[i - 1] + min_h:
            t_new[i] = t_new[i - 1] + min_h
    t_new[-1] = 1.0

    return t_new


def redistribute_mesh(
    sol: PeriodicOrbitResult,
    alpha: float = 1.0,
    beta: float = 0.0,
    damping: float = 0.7,
    p: float | Array = 0.0,
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
    p : float | Array, default=0.0
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
    rho = np.asarray(sol.problem.rho)

    # Interpolate solution onto new mesh nodes and Gauss points
    u_mesh_new = np.array([sol.evaluate(t) for t in new_mesh[:-1]])

    u_gauss_new = np.zeros((N, m, sol.problem.dim), dtype=np.float64)
    for k in range(N):
        hk = new_mesh[k + 1] - new_mesh[k]
        for i in range(m):
            ti = new_mesh[k] + hk * rho[i]
            u_gauss_new[k, i] = sol.evaluate(ti)

    if not re_solve:
        return PeriodicOrbitResult(
            mesh=jnp.asarray(new_mesh),
            u_mesh=jnp.asarray(u_mesh_new),
            u_gauss=jnp.asarray(u_gauss_new),
            period=sol.period,
            converged=sol.converged,
            iterations=0,
            residual_norm=sol.residual_norm,
            problem=sol.problem,
        )

    # Re-solve orbit on new mesh
    return solve_periodic_orbit(
        problem=sol.problem,
        u_mesh_init=jnp.asarray(u_mesh_new),
        T_init=sol.period,
        p=p,
        mesh=jnp.asarray(new_mesh),
        u_gauss_init=jnp.asarray(u_gauss_new),
        tol=tol,
        max_iters=max_iters,
    )
