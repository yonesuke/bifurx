from __future__ import annotations

import jax
import jax.numpy as jnp

from bifurx.bvp.collocation import PeriodicOrbitResult


def _monodromy_condensation(
    sol: PeriodicOrbitResult,
    p: float | jax.Array = 0.0,
) -> jax.Array:
    """Compute the monodromy matrix M via de Boor-Swartz collocation Jacobian condensation."""
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
        hk = mesh[k + 1] - mesh[k]
        Ju_k = jax.vmap(jac_fn)(u_gauss[k])

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


def _monodromy_variational(
    sol: PeriodicOrbitResult,
    p: float | jax.Array = 0.0,
    num_steps: int = 400,
) -> jax.Array:
    """Compute the monodromy matrix M by integrating the variational equation via RK4."""
    n = sol.problem.dim
    T = jnp.squeeze(jnp.asarray(sol.period, dtype=jnp.float64))
    dt = 1.0 / num_steps
    jac_fn = jax.jacobian(lambda u_: sol.problem.fn(u_, p, **sol.problem.kwargs))

    def rk4_step(Phi: jax.Array, step: jax.Array) -> tuple[jax.Array, None]:
        t0 = step * dt
        A0 = T * jac_fn(sol.evaluate(t0))
        A_mid = T * jac_fn(sol.evaluate(t0 + 0.5 * dt))
        A1 = T * jac_fn(sol.evaluate(t0 + dt))

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
    """Compute Floquet multipliers and monodromy matrix for a periodic orbit."""
    if method == "condensation":
        M = _monodromy_condensation(sol, p=p)
    elif method == "variational":
        M = _monodromy_variational(sol, p=p)
    else:
        msg = f"Unknown method: {method}. Choose 'condensation' or 'variational'."
        raise ValueError(msg)

    multipliers = jnp.linalg.eigvals(M)
    idx = jnp.argsort(-jnp.abs(multipliers))
    return multipliers[idx], M
