from __future__ import annotations

import jax
import jax.numpy as jnp

from bifurx.bvp.collocation import PeriodicOrbitResult


def compute_monodromy_condensation(
    sol: PeriodicOrbitResult,
    p: float | jax.Array = 0.0,
) -> jax.Array:
    """Compute the monodromy matrix M via de Boor-Swartz collocation Jacobian condensation.

    Linearizes the collocation equations on each interval k to obtain interval transition
    matrices M_k such that delta u_{k+1} = M_k delta u_k.
    The overall monodromy matrix is M = M_{N-1} * M_{N-2} * ... * M_0.

    Parameters
    ----------
    sol : PeriodicOrbitResult
        Converged periodic orbit result.
    p : float | jax.Array, default=0.0
        System parameter value.

    Returns
    -------
    jax.Array of shape (dim, dim)
        The monodromy matrix M.
    """
    mesh = jnp.asarray(sol.mesh, dtype=jnp.float64)
    u_gauss = jnp.asarray(sol.u_gauss, dtype=jnp.float64)
    T = jnp.squeeze(jnp.asarray(sol.period, dtype=jnp.float64))
    N = sol.problem.num_intervals
    m = sol.problem.num_gauss_points
    n = sol.problem.dim
    D = jnp.asarray(sol.problem.D, dtype=jnp.float64)
    C = jnp.asarray(sol.problem.C, dtype=jnp.float64)

    jac_fn = jax.jacobian(lambda u_: sol.problem.fn(u_, p, **sol.problem.kwargs))
    eye_n = jnp.eye(n, dtype=jnp.float64)
    eye_m = jnp.eye(m, dtype=jnp.float64)

    def interval_step(M_total: jax.Array, k: jax.Array) -> tuple[jax.Array, None]:
        """Accumulate the interval transition matrix M_k into the running product."""
        hk = mesh[k + 1] - mesh[k]
        Ju_k = jax.vmap(jac_fn)(u_gauss[k])  # (m, n, n)

        A_blocks = (D[:, 1:] / hk)[:, :, None, None] * eye_n[None, None, :, :] - T * eye_m[
            :, :, None, None
        ] * Ju_k[:, None, :, :]
        A_k = jnp.swapaxes(A_blocks, 1, 2).reshape(m * n, m * n)
        B_k = (-(D[:, 0] / hk)[:, None, None] * eye_n[None, :, :]).reshape(m * n, n)

        K_k = jnp.linalg.solve(A_k, B_k).reshape(m, n, n)
        M_k = C[0] * eye_n + jnp.einsum("j,jin->in", C[1:], K_k)
        return M_k @ M_total, None

    M_total, _ = jax.lax.scan(interval_step, eye_n, jnp.arange(N))
    return M_total


def compute_monodromy_variational(
    sol: PeriodicOrbitResult,
    p: float | jax.Array = 0.0,
    num_steps: int = 400,
) -> jax.Array:
    """Compute the monodromy matrix M by integrating the variational equation:

    dPhi/dt = T * (df/du)(u(t)) * Phi(t),  Phi(0) = I_n.
    """
    n = sol.problem.dim
    T = jnp.squeeze(jnp.asarray(sol.period, dtype=jnp.float64))
    dt = 1.0 / num_steps

    jac_fn = jax.jacobian(lambda u_: sol.problem.fn(u_, p, **sol.problem.kwargs))

    def rk4_step(Phi: jax.Array, step: jax.Array) -> tuple[jax.Array, None]:
        """Single RK4 step of dPhi/dt = T * (df/du)(u(t)) * Phi."""
        t0 = step * dt
        t_mid = t0 + 0.5 * dt
        t1 = t0 + dt

        A0 = T * jac_fn(sol.evaluate(t0))
        A_mid = T * jac_fn(sol.evaluate(t_mid))
        A1 = T * jac_fn(sol.evaluate(t1))

        k1 = A0 @ Phi
        k2 = A_mid @ (Phi + 0.5 * dt * k1)
        k3 = A_mid @ (Phi + 0.5 * dt * k2)
        k4 = A1 @ (Phi + dt * k3)

        return Phi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    Phi, _ = jax.lax.scan(rk4_step, jnp.eye(n, dtype=jnp.float64), jnp.arange(num_steps))
    return Phi


def compute_floquet_multipliers(
    sol: PeriodicOrbitResult,
    p: float | jax.Array = 0.0,
    method: str = "condensation",
) -> tuple[jax.Array, jax.Array]:
    """Compute Floquet multipliers and monodromy matrix for a periodic orbit.

    Parameters
    ----------
    sol : PeriodicOrbitResult
        Converged periodic orbit.
    p : float | jax.Array, default=0.0
        System parameter value.
    method : str, default="condensation"
        Method for monodromy matrix computation: "condensation" or "variational".

    Returns
    -------
    tuple[jax.Array, jax.Array]
        - multipliers: 1D complex array of Floquet multipliers, sorted by magnitude descending.
        - M: Monodromy matrix of shape (dim, dim).
    """
    if method == "condensation":
        M = compute_monodromy_condensation(sol, p=p)
    elif method == "variational":
        M = compute_monodromy_variational(sol, p=p)
    else:
        msg = f"Unknown method: {method}. Choose 'condensation' or 'variational'."
        raise ValueError(msg)

    multipliers = jnp.linalg.eigvals(M)
    idx = jnp.argsort(-jnp.abs(multipliers))
    return multipliers[idx], M
