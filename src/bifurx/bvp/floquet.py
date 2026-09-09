from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from bifurx.bvp.collocation import PeriodicOrbitResult


def compute_monodromy_condensation(
    sol: PeriodicOrbitResult,
    p: float | Array = 0.0,
) -> np.ndarray:
    """Compute the monodromy matrix M via de Boor-Swartz collocation Jacobian condensation.

    Linearizes the collocation equations on each interval k to obtain interval transition
    matrices M_k such that delta u_{k+1} = M_k delta u_k.
    The overall monodromy matrix is M = M_{N-1} * M_{N-2} * ... * M_0.

    Parameters
    ----------
    sol : PeriodicOrbitResult
        Converged periodic orbit result.
    p : float | Array, default=0.0
        System parameter value.

    Returns
    -------
    np.ndarray of shape (dim, dim)
        The monodromy matrix M.
    """
    mesh = np.asarray(sol.mesh)
    u_gauss = np.asarray(sol.u_gauss)
    T = sol.period
    N = sol.problem.num_intervals
    m = sol.problem.num_gauss_points
    n = sol.problem.dim
    D = np.asarray(sol.problem.D)
    C = np.asarray(sol.problem.C)

    # Jacobian of vector field df/du
    jac_fn = jax.jacobian(lambda u_: sol.problem.fn(u_, p, **sol.problem.kwargs))

    # Overall monodromy accumulator
    M_total = np.eye(n, dtype=np.float64)

    for k in range(N):
        hk = mesh[k + 1] - mesh[k]

        # Build block matrix A_k of size (m*n, m*n) and RHS B_k of size (m*n, n)
        A_k = np.zeros((m * n, m * n), dtype=np.float64)
        B_k = np.zeros((m * n, n), dtype=np.float64)

        for i in range(m):
            u_i = jnp.asarray(u_gauss[k, i])
            Ju_i = np.asarray(jac_fn(u_i), dtype=np.float64)  # (n, n)

            # Block row i
            row_start = i * n
            row_end = (i + 1) * n

            # Contribution of u_{k, 0} (delta u_k) to RHS
            B_k[row_start:row_end, :] = -(D[i, 0] / hk) * np.eye(n)

            for j in range(m):
                col_start = j * n
                col_end = (j + 1) * n
                # D[i, j+1] / hk * I_n
                block_ij = (D[i, j + 1] / hk) * np.eye(n)
                if i == j:
                    block_ij -= T * Ju_i
                A_k[row_start:row_end, col_start:col_end] = block_ij

        # Solve A_k * K_k = B_k
        K_k = np.linalg.solve(A_k, B_k)  # shape (m*n, n)

        # Transition matrix M_k = C_0 * I_n + sum_{j=1}^m C_j * K_k[j]
        M_k = C[0] * np.eye(n, dtype=np.float64)
        for j in range(m):
            M_k += C[j + 1] * K_k[j * n : (j + 1) * n, :]

        M_total = M_k @ M_total

    return M_total


def compute_monodromy_variational(
    sol: PeriodicOrbitResult,
    p: float | Array = 0.0,
    num_steps: int = 400,
) -> np.ndarray:
    """Compute the monodromy matrix M by integrating the variational equation:

        dPhi/dt = T * (df/du)(u(t)) * Phi(t),  Phi(0) = I_n.

    Parameters
    ----------
    sol : PeriodicOrbitResult
        Converged periodic orbit result.
    p : float | Array, default=0.0
        System parameter value.
    num_steps : int, default=400
        Number of RK4 integration steps over [0, 1].

    Returns
    -------
    np.ndarray of shape (dim, dim)
        The monodromy matrix M = Phi(1).
    """
    n = sol.problem.dim
    T = sol.period
    dt = 1.0 / num_steps

    jac_fn = jax.jacobian(lambda u_: sol.problem.fn(u_, p, **sol.problem.kwargs))

    Phi = np.eye(n, dtype=np.float64)

    for step in range(num_steps):
        t0 = step * dt
        t_mid = t0 + 0.5 * dt
        t1 = t0 + dt

        u0 = np.asarray(sol.evaluate(t0))
        u_mid = np.asarray(sol.evaluate(t_mid))
        u1 = np.asarray(sol.evaluate(t1))

        A0 = T * np.asarray(jac_fn(jnp.asarray(u0)))
        A_mid = T * np.asarray(jac_fn(jnp.asarray(u_mid)))
        A1 = T * np.asarray(jac_fn(jnp.asarray(u1)))

        # RK4 step for dPhi/dt = A(t) * Phi
        k1 = A0 @ Phi
        k2 = A_mid @ (Phi + 0.5 * dt * k1)
        k3 = A_mid @ (Phi + 0.5 * dt * k2)
        k4 = A1 @ (Phi + dt * k3)

        Phi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return Phi


def compute_floquet_multipliers(
    sol: PeriodicOrbitResult,
    p: float | Array = 0.0,
    method: str = "condensation",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Floquet multipliers and monodromy matrix for a periodic orbit.

    Parameters
    ----------
    sol : PeriodicOrbitResult
        Converged periodic orbit.
    p : float | Array, default=0.0
        System parameter value.
    method : str, default="condensation"
        Method for monodromy matrix computation: "condensation" or "variational".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
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

    multipliers = np.linalg.eigvals(M)
    # Sort by magnitude descending
    idx = np.argsort(-np.abs(multipliers))
    return multipliers[idx], M
