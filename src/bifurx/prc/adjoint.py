from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from bifurx.bvp.collocation import (
    CollocationProblem,
    PeriodicOrbitResult,
    solve_periodic_orbit,
)


class IPRCResult(NamedTuple):
    """Result of an infinitesimal Phase Response Curve (iPRC) computation."""

    ts: Array  # Physical time grid in [0, T]
    theta: Array  # Normalized phase grid in [0, 1]
    orbit: Array  # Limit cycle trajectory u(t), shape (M, dim)
    Z: Array  # Phase response sensitivity Z(t) = grad_u theta(u(t)), shape (M, dim)
    period: float  # Limit cycle period T
    omega: float  # Angular frequency omega = 2*pi / T
    peak_amplitudes: Array  # Max absolute sensitivity for each state component, shape (dim,)
    zero_crossings: list[list[float]]  # Phase zero crossings for each state component


class IPRCContinuationResult(NamedTuple):
    """Result of iPRC parameter continuation."""

    p: Array  # Parameter values, shape (num_points,)
    periods: Array  # Periods T(p), shape (num_points,)
    peak_amplitudes: Array  # Peak amplitudes for each state, shape (num_points, dim)
    orbits: list[Array]  # List of orbit arrays
    Z: list[Array]  # List of iPRC arrays
    results: list[IPRCResult]  # Full list of IPRCResult objects

    @property
    def num_points(self) -> int:
        """Number of tracked continuation points."""
        return int(len(self.p))


def find_zero_crossings(signal: np.ndarray, time_grid: np.ndarray) -> list[float]:
    """Find linearly interpolated zero crossings of a 1D signal."""
    crossings = []
    n = len(signal)
    for i in range(n):
        s0 = signal[i]
        s1 = signal[(i + 1) % n]
        if s0 * s1 <= 0.0 and abs(s1 - s0) > 1e-12:
            t0 = time_grid[i]
            t1 = time_grid[(i + 1) % n]
            if t1 < t0:
                t1 = t1 + (time_grid[-1] - time_grid[0] + (time_grid[1] - time_grid[0]))
            t_cross = float(t0 - s0 * (t1 - t0) / (s1 - s0))
            crossings.append(t_cross)
    return crossings


def compute_iprc_from_collocation(
    sol: PeriodicOrbitResult,
    p: float | Array = 0.0,
    omega_target: float | None = None,
) -> IPRCResult:
    """Compute high-order infinitesimal Phase Response Curve Z(t) directly on Gauss collocation BVP.

    Solves the periodic adjoint BVP:
        dZ/dt + T * (df/du(u(t)))^T * Z(t) = 0,
        Z(0) = Z(1),
        <Z(0), f(u(0))> = omega = 2*pi / T.

    Parameters
    ----------
    sol : PeriodicOrbitResult
        Converged periodic orbit from CollocationProblem.
    p : float | Array, default=0.0
        System parameter value.
    omega_target : float | None, optional
        Target normalization value for <Z(0), f(u(0))>. Defaults to 2*pi / T.

    Returns
    -------
    IPRCResult
    """
    problem = sol.problem
    N = problem.num_intervals
    m = problem.num_gauss_points
    n = problem.dim
    T = sol.period
    mesh = np.asarray(sol.mesh)
    u_gauss = np.asarray(sol.u_gauss)
    u_mesh = np.asarray(sol.u_mesh)
    D = np.asarray(problem.D)
    C = np.asarray(problem.C)

    if omega_target is None:
        omega_val = 2.0 * np.pi / T
    else:
        omega_val = float(omega_target)

    jac_fn = jax.jacobian(lambda u: problem.fn(u, p, **problem.kwargs))

    dim_z = N * (m + 1) * n
    L = np.zeros((dim_z, dim_z), dtype=np.float64)
    rhs = np.zeros(dim_z, dtype=np.float64)

    def idx_mesh(k: int) -> int:
        return k * n

    def idx_gauss(k: int, i: int) -> int:
        return N * n + (k * m + i) * n

    row = 0
    # 1. Collocation equations at Gauss points
    for k in range(N):
        hk = mesh[k + 1] - mesh[k]
        for i in range(m):
            Ju_i = np.asarray(jac_fn(jnp.asarray(u_gauss[k, i])), dtype=np.float64)

            # Z_mesh[k] contribution
            L[row : row + n, idx_mesh(k) : idx_mesh(k) + n] = (D[i, 0] / hk) * np.eye(n)

            # Z_gauss[k, j] contributions
            for j in range(m):
                block_j = (D[i, j + 1] / hk) * np.eye(n)
                if j == i:
                    block_j += T * Ju_i.T
                L[row : row + n, idx_gauss(k, j) : idx_gauss(k, j) + n] = block_j
            row += n

    # 2. Continuity equations: Z_{k+1} - sum_{j=0}^m C_j Z_{k, j} = 0
    for k in range(N - 1):
        L[row : row + n, idx_mesh(k + 1) : idx_mesh(k + 1) + n] = np.eye(n)
        L[row : row + n, idx_mesh(k) : idx_mesh(k) + n] = -C[0] * np.eye(n)
        for j in range(m):
            L[row : row + n, idx_gauss(k, j) : idx_gauss(k, j) + n] = -C[j + 1] * np.eye(n)
        row += n

    # 3. Periodic boundary: for last interval k = N-1, replace one equation with normalization!
    L[row, idx_mesh(0)] = 1.0
    L[row, idx_mesh(N - 1) : idx_mesh(N - 1) + n] = -C[0] * np.eye(n)[0, :]
    for j in range(m):
        L[row, idx_gauss(N - 1, j) : idx_gauss(N - 1, j) + n] = -C[j + 1] * np.eye(n)[0, :]
    row += 1

    # Normalization: f(u0)^T * Z_0 = omega
    f0 = np.asarray(problem.fn(jnp.asarray(u_mesh[0]), p, **problem.kwargs), dtype=np.float64)
    L[row, idx_mesh(0) : idx_mesh(0) + n] = f0
    rhs[row] = omega_val
    row += 1

    Z_sol = np.linalg.solve(L, rhs)
    Z_mesh = Z_sol[: N * n].reshape((N, n))

    # Evaluation grid using the collocation mesh nodes
    theta_grid = mesh[:-1]
    ts_grid = theta_grid * T
    orbit_eval = u_mesh
    Z_eval = Z_mesh

    # Peak amplitudes and zero crossings
    peak_amps = np.max(np.abs(Z_eval), axis=0)
    zero_crossings = [find_zero_crossings(Z_eval[:, comp], theta_grid) for comp in range(n)]

    return IPRCResult(
        ts=jnp.asarray(ts_grid),
        theta=jnp.asarray(theta_grid),
        orbit=jnp.asarray(orbit_eval),
        Z=jnp.asarray(Z_eval),
        period=float(T),
        omega=float(omega_val),
        peak_amplitudes=jnp.asarray(peak_amps),
        zero_crossings=zero_crossings,
    )


def compute_iprc(
    sol_or_fn: PeriodicOrbitResult | Callable[..., Array],
    u_mesh: Array | np.ndarray | None = None,
    T: float | None = None,
    p: float | Array = 0.0,
    omega_target: float | None = None,
    M_mesh: int = 80,
    **kwargs: Any,
) -> IPRCResult:
    """High-level infinitesimal Phase Response Curve (iPRC) solver.

    Can be called either with a converged PeriodicOrbitResult:
        res = compute_iprc(sol)
    or directly with a vector field function and initial periodic orbit:
        res = compute_iprc(fn, u_mesh, T=period)

    Parameters
    ----------
    sol_or_fn : PeriodicOrbitResult or Callable
        Either a PeriodicOrbitResult from solve_periodic_orbit, or an ODE RHS function f(u, p).
    u_mesh : Array | np.ndarray | None, optional
        Periodic orbit mesh points if passing a function.
    T : float | None, optional
        Orbit period if passing a function.
    p : float | Array, default=0.0
        Parameter value.
    omega_target : float | None, optional
        Target normalization <Z(0), f(u(0))>. Defaults to 2*pi / T.
    M_mesh : int, default=80
        Number of mesh intervals for midpoint solver fallback.

    Returns
    -------
    IPRCResult
    """
    if isinstance(sol_or_fn, PeriodicOrbitResult):
        return compute_iprc_from_collocation(sol_or_fn, p=p, omega_target=omega_target)

    # Fallback to direct midpoint BVP solve
    fn = sol_or_fn
    if u_mesh is None or T is None:
        msg = "Must provide u_mesh and T when calling compute_iprc with a function."
        raise ValueError(msg)

    u_arr = np.asarray(u_mesh, dtype=np.float64)
    M = len(u_arr)
    n = u_arr.shape[1]
    h = 1.0 / M
    period = float(T)
    omega_val = float(2.0 * np.pi / period if omega_target is None else omega_target)

    jac_fn = jax.jacobian(lambda u: fn(u, p, **kwargs))

    u_next = np.roll(u_arr, -1, axis=0)
    u_mid = 0.5 * (u_arr + u_next)
    A_mid = np.array([np.asarray(jac_fn(jnp.asarray(u_mid[k]))) for k in range(M)])

    dim_z = M * n
    L = np.zeros((dim_z, dim_z), dtype=np.float64)
    rhs = np.zeros(dim_z, dtype=np.float64)

    for k in range(M - 1):
        AT = A_mid[k].T
        L[k * n : (k + 1) * n, k * n : (k + 1) * n] = -np.eye(n) / h + 0.5 * period * AT
        L[k * n : (k + 1) * n, (k + 1) * n : (k + 2) * n] = np.eye(n) / h + 0.5 * period * AT

    AT_last = A_mid[M - 1].T
    mat_last = -np.eye(n) / h + 0.5 * period * AT_last
    mat_0 = np.eye(n) / h + 0.5 * period * AT_last

    for r in range(n - 1):
        L[(M - 1) * n + r, (M - 1) * n : M * n] = mat_last[r, :]
        L[(M - 1) * n + r, 0:n] = mat_0[r, :]

    f0 = np.asarray(fn(jnp.asarray(u_arr[0]), p, **kwargs), dtype=np.float64)
    L[dim_z - 1, 0:n] = f0
    rhs[dim_z - 1] = omega_val

    Z_vec = np.linalg.solve(L, rhs)
    Z = Z_vec.reshape((M, n))

    theta_grid = np.linspace(0.0, 1.0, M, endpoint=False)
    ts_grid = theta_grid * period
    peak_amps = np.max(np.abs(Z), axis=0)
    zero_crossings = [find_zero_crossings(Z[:, comp], theta_grid) for comp in range(n)]

    return IPRCResult(
        ts=jnp.asarray(ts_grid),
        theta=jnp.asarray(theta_grid),
        orbit=jnp.asarray(u_arr),
        Z=jnp.asarray(Z),
        period=period,
        omega=omega_val,
        peak_amplitudes=jnp.asarray(peak_amps),
        zero_crossings=zero_crossings,
    )


def solve_coupled_orbit_prc(
    problem: CollocationProblem,
    u_mesh_init: Array | np.ndarray,
    T_init: float,
    p: float | Array,
    u_gauss_init: Array | np.ndarray | None = None,
    tol: float = 1e-8,
    max_iters: int = 25,
) -> tuple[PeriodicOrbitResult, IPRCResult]:
    """Solve periodic orbit and iPRC in a coupled fashion.

    First solves the nonlinear collocation BVP for (u, T), then performs the direct
    exact linear solve for the adjoint sensitivity Z(t).

    Parameters
    ----------
    problem : CollocationProblem
        BVP problem definition.
    u_mesh_init : Array | np.ndarray
        Initial guess for limit cycle.
    T_init : float
        Initial guess for period.
    p : float | Array
        Continuation parameter value.
    u_gauss_init : Array | np.ndarray | None, optional
        Initial guess at Gauss collocation nodes.
    tol : float, default=1e-8
        Convergence tolerance for periodic orbit.
    max_iters : int, default=25
        Maximum iterations.

    Returns
    -------
    tuple[PeriodicOrbitResult, IPRCResult]
    """
    sol_orbit = solve_periodic_orbit(
        problem=problem,
        u_mesh_init=u_mesh_init,
        T_init=T_init,
        p=p,
        u_gauss_init=u_gauss_init,
        tol=tol,
        max_iters=max_iters,
    )
    res_prc = compute_iprc_from_collocation(sol_orbit, p=p)
    return sol_orbit, res_prc


def continuation_iprc(
    problem: CollocationProblem,
    u_mesh_init: Array | np.ndarray,
    T_init: float,
    p_start: float,
    p_end: float,
    num_steps: int = 30,
    tol: float = 1e-8,
    max_iters: int = 20,
) -> IPRCContinuationResult:
    """Perform parameter continuation of the infinitesimal Phase Response Curve (iPRC).

    Tracks how limit cycle waveform, period, and phase response sensitivity Z(t) deform
    as a system parameter varies.

    Parameters
    ----------
    problem : CollocationProblem
        Collocation problem definition.
    u_mesh_init : Array
        Initial guess for limit cycle at p_start.
    T_init : float
        Initial period guess.
    p_start : float
        Starting parameter value.
    p_end : float
        Ending parameter value.
    num_steps : int, default=30
        Number of continuation steps.
    tol : float, default=1e-8
        Convergence tolerance per step.
    max_iters : int, default=20
        Maximum Newton iterations per step.

    Returns
    -------
    IPRCContinuationResult
    """
    p_values = np.linspace(p_start, p_end, num_steps)

    p_records: list[float] = []
    period_records: list[float] = []
    peak_amp_records: list[np.ndarray] = []
    orbit_records: list[Array] = []
    z_records: list[Array] = []
    result_records: list[IPRCResult] = []

    current_u = u_mesh_init
    current_u_gauss = None
    current_T = T_init

    for p_val in p_values:
        sol_orbit, sol_prc = solve_coupled_orbit_prc(
            problem=problem,
            u_mesh_init=current_u,
            T_init=current_T,
            p=p_val,
            u_gauss_init=current_u_gauss,
            tol=tol,
            max_iters=max_iters,
        )

        if not sol_orbit.converged:
            break

        p_records.append(float(p_val))
        period_records.append(sol_orbit.period)
        peak_amp_records.append(np.asarray(sol_prc.peak_amplitudes))
        orbit_records.append(sol_prc.orbit)
        z_records.append(sol_prc.Z)
        result_records.append(sol_prc)

        # Update predictor for next step
        current_u = sol_orbit.u_mesh
        current_u_gauss = sol_orbit.u_gauss
        current_T = sol_orbit.period

    return IPRCContinuationResult(
        p=jnp.asarray(p_records),
        periods=jnp.asarray(period_records),
        peak_amplitudes=jnp.asarray(np.array(peak_amp_records)),
        orbits=orbit_records,
        Z=z_records,
        results=result_records,
    )
