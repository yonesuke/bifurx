from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.bvp.collocation import (
    CollocationProblem,
    PeriodicOrbitResult,
    solve_periodic_orbit,
)


class IPRCResult(NamedTuple):
    """Result of an infinitesimal Phase Response Curve (iPRC) computation."""

    ts: jax.Array  # Physical time grid in [0, T]
    theta: jax.Array  # Normalized phase grid in [0, 1]
    orbit: jax.Array  # Limit cycle trajectory u(t), shape (M, dim)
    Z: jax.Array  # Phase response sensitivity Z(t) = grad_u theta(u(t)), shape (M, dim)
    period: float  # Limit cycle period T
    omega: float  # Angular frequency omega = 2*pi / T
    peak_amplitudes: jax.Array  # Max absolute sensitivity for each state component, shape (dim,)
    zero_crossings: list[list[float]]  # Phase zero crossings for each state component


class IPRCContinuationResult(NamedTuple):
    """Result of iPRC parameter continuation."""

    p: jax.Array  # Parameter values, shape (num_points,)
    periods: jax.Array  # Periods T(p), shape (num_points,)
    peak_amplitudes: jax.Array  # Peak amplitudes for each state, shape (num_points, dim)
    orbits: list[jax.Array]  # List of orbit arrays
    Z: list[jax.Array]  # List of iPRC arrays
    results: list[IPRCResult]  # Full list of IPRCResult objects

    @property
    def num_points(self) -> int:
        """Number of tracked continuation points."""
        return int(len(self.p))


def find_zero_crossings(signal: jax.Array, time_grid: jax.Array) -> list[float]:
    """Find linearly interpolated zero crossings of a 1D signal."""
    signal = jnp.asarray(signal)
    time_grid = jnp.asarray(time_grid)
    n = signal.shape[0]
    crossings = []
    for i in range(n):
        s0 = signal[i]
        s1 = signal[(i + 1) % n]
        if float(s0 * s1) <= 0.0 and abs(float(s1 - s0)) > 1e-12:
            t0 = time_grid[i]
            t1 = time_grid[(i + 1) % n]
            if float(t1) < float(t0):
                t1 = t1 + (time_grid[-1] - time_grid[0] + (time_grid[1] - time_grid[0]))
            t_cross = float(t0 - s0 * (t1 - t0) / (s1 - s0))
            crossings.append(t_cross)
    return crossings


def compute_iprc_from_collocation(
    sol: PeriodicOrbitResult,
    p: float | jax.Array = 0.0,
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
    p : float | jax.Array, default=0.0
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
    mesh = jnp.asarray(sol.mesh)
    u_gauss = jnp.asarray(sol.u_gauss)
    u_mesh = jnp.asarray(sol.u_mesh)
    D = jnp.asarray(problem.D)
    C = jnp.asarray(problem.C)

    if omega_target is None:
        omega_val = 2.0 * jnp.pi / T
    else:
        omega_val = jnp.asarray(omega_target)

    jac_fn = jax.jacobian(lambda u: problem.fn(u, p, **problem.kwargs))

    dim_z = N * (m + 1) * n
    L = jnp.zeros((dim_z, dim_z), dtype=jnp.float64)
    rhs = jnp.zeros(dim_z, dtype=jnp.float64)

    def idx_mesh(k: int) -> int:
        return k * n

    def idx_gauss(k: int, i: int) -> int:
        return N * n + (k * m + i) * n

    row = 0
    # 1. Collocation equations at Gauss points
    for k in range(N):
        hk = mesh[k + 1] - mesh[k]
        for i in range(m):
            Ju_i = jac_fn(u_gauss[k, i])

            # Z_mesh[k] contribution
            L = L.at[row : row + n, idx_mesh(k) : idx_mesh(k) + n].set((D[i, 0] / hk) * jnp.eye(n))

            # Z_gauss[k, j] contributions
            for j in range(m):
                block_j = (D[i, j + 1] / hk) * jnp.eye(n)
                if j == i:
                    block_j = block_j + T * Ju_i.T
                L = L.at[row : row + n, idx_gauss(k, j) : idx_gauss(k, j) + n].set(block_j)
            row += n

    # 2. Continuity equations: Z_{k+1} - sum_{j=0}^m C_j Z_{k, j} = 0
    for k in range(N - 1):
        L = L.at[row : row + n, idx_mesh(k + 1) : idx_mesh(k + 1) + n].set(jnp.eye(n))
        L = L.at[row : row + n, idx_mesh(k) : idx_mesh(k) + n].set(-C[0] * jnp.eye(n))
        for j in range(m):
            L = L.at[row : row + n, idx_gauss(k, j) : idx_gauss(k, j) + n].set(
                -C[j + 1] * jnp.eye(n)
            )
        row += n

    # 3. Periodic boundary: for last interval k = N-1, replace one equation with normalization!
    L = L.at[row, idx_mesh(0)].set(1.0)
    L = L.at[row, idx_mesh(N - 1) : idx_mesh(N - 1) + n].set(-C[0] * jnp.eye(n)[0, :])
    for j in range(m):
        L = L.at[row, idx_gauss(N - 1, j) : idx_gauss(N - 1, j) + n].set(
            -C[j + 1] * jnp.eye(n)[0, :]
        )
    row += 1

    # Normalization: f(u0)^T * Z_0 = omega
    f0 = problem.fn(u_mesh[0], p, **problem.kwargs)
    L = L.at[row, idx_mesh(0) : idx_mesh(0) + n].set(f0)
    rhs = rhs.at[row].set(omega_val)
    row += 1

    Z_sol = jnp.linalg.solve(L, rhs)
    Z_mesh = Z_sol[: N * n].reshape((N, n))

    # Evaluation grid using the collocation mesh nodes
    theta_grid = mesh[:-1]
    ts_grid = theta_grid * T
    orbit_eval = u_mesh
    Z_eval = Z_mesh

    # Peak amplitudes and zero crossings
    peak_amps = jnp.max(jnp.abs(Z_eval), axis=0)
    zero_crossings = [find_zero_crossings(Z_eval[:, comp], theta_grid) for comp in range(n)]

    return IPRCResult(
        ts=ts_grid,
        theta=theta_grid,
        orbit=orbit_eval,
        Z=Z_eval,
        period=float(T),
        omega=float(omega_val),
        peak_amplitudes=peak_amps,
        zero_crossings=zero_crossings,
    )


def compute_iprc(
    sol_or_fn: PeriodicOrbitResult | Callable[..., jax.Array],
    u_mesh: ArrayLike | None = None,
    T: float | None = None,
    p: float | jax.Array = 0.0,
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
    u_mesh : jax.Array | None, optional
        Periodic orbit mesh points if passing a function.
    T : float | None, optional
        Orbit period if passing a function.
    p : float | jax.Array, default=0.0
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

    u_arr = jnp.asarray(u_mesh, dtype=jnp.float64)
    M = u_arr.shape[0]
    n = u_arr.shape[1]
    h = 1.0 / M
    period = float(T)
    omega_val = float(2.0 * jnp.pi / period if omega_target is None else omega_target)

    jac_fn = jax.jacobian(lambda u: fn(u, p, **kwargs))

    u_next = jnp.roll(u_arr, -1, axis=0)
    u_mid = 0.5 * (u_arr + u_next)
    A_mid = jnp.stack([jac_fn(u_mid[k]) for k in range(M)])

    dim_z = M * n
    L = jnp.zeros((dim_z, dim_z), dtype=jnp.float64)
    rhs = jnp.zeros(dim_z, dtype=jnp.float64)

    for k in range(M - 1):
        AT = A_mid[k].T
        L = L.at[k * n : (k + 1) * n, k * n : (k + 1) * n].set(-jnp.eye(n) / h + 0.5 * period * AT)
        L = L.at[k * n : (k + 1) * n, (k + 1) * n : (k + 2) * n].set(
            jnp.eye(n) / h + 0.5 * period * AT
        )

    AT_last = A_mid[M - 1].T
    mat_last = -jnp.eye(n) / h + 0.5 * period * AT_last
    mat_0 = jnp.eye(n) / h + 0.5 * period * AT_last

    for r in range(n - 1):
        L = L.at[(M - 1) * n + r, (M - 1) * n : M * n].set(mat_last[r, :])
        L = L.at[(M - 1) * n + r, 0:n].set(mat_0[r, :])

    f0 = fn(u_arr[0], p, **kwargs)
    L = L.at[dim_z - 1, 0:n].set(f0)
    rhs = rhs.at[dim_z - 1].set(omega_val)

    Z_vec = jnp.linalg.solve(L, rhs)
    Z = Z_vec.reshape((M, n))

    theta_grid = jnp.linspace(0.0, 1.0, M, endpoint=False)
    ts_grid = theta_grid * period
    peak_amps = jnp.max(jnp.abs(Z), axis=0)
    zero_crossings = [find_zero_crossings(Z[:, comp], theta_grid) for comp in range(n)]

    return IPRCResult(
        ts=ts_grid,
        theta=theta_grid,
        orbit=u_arr,
        Z=Z,
        period=period,
        omega=omega_val,
        peak_amplitudes=peak_amps,
        zero_crossings=zero_crossings,
    )


def solve_coupled_orbit_prc(
    problem: CollocationProblem,
    u_mesh_init: ArrayLike,
    T_init: float,
    p: float | jax.Array,
    u_gauss_init: jax.Array | None = None,
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
    u_mesh_init : jax.Array
        Initial guess for limit cycle.
    T_init : float
        Initial guess for period.
    p : float | jax.Array
        Continuation parameter value.
    u_gauss_init : jax.Array | None, optional
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
    u_mesh_init: ArrayLike,
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
    u_mesh_init : jax.Array
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
    p_values = jnp.linspace(p_start, p_end, num_steps)

    p_records: list[float] = []
    period_records: list[float] = []
    peak_amp_records: list[jax.Array] = []
    orbit_records: list[jax.Array] = []
    z_records: list[jax.Array] = []
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
        period_records.append(float(sol_orbit.period))
        peak_amp_records.append(sol_prc.peak_amplitudes)
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
        peak_amplitudes=(
            jnp.stack(peak_amp_records) if peak_amp_records else jnp.zeros((0,), dtype=jnp.float64)
        ),
        orbits=orbit_records,
        Z=z_records,
        results=result_records,
    )
