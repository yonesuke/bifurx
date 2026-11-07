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
    orbit: jax.Array  # Limit cycle trajectory u(t), shape (N, dim)
    Z: jax.Array  # Phase response sensitivity Z(t) = grad_u theta(u(t)), shape (N, dim)
    period: jax.Array  # Limit cycle period T
    omega: jax.Array  # Angular frequency omega = 2*pi / T
    peak_amplitudes: jax.Array  # Max absolute sensitivity for each state component, shape (dim,)
    zero_crossings: list[list[float]]  # Phase zero crossings for each state component


class IPRCContinuationResult(NamedTuple):
    """Result of iPRC parameter continuation."""

    p: jax.Array  # Parameter values, shape (num_points,)
    periods: jax.Array  # Periods T(p), shape (num_points,)
    peak_amplitudes: jax.Array  # Peak amplitudes for each state, shape (num_points, dim)
    orbits: jax.Array  # Stacked orbit arrays, shape (num_points, N, dim)
    Z: jax.Array  # Stacked iPRC arrays, shape (num_points, N, dim)
    results: list[IPRCResult]  # Full list of IPRCResult objects

    @property
    def num_points(self) -> int:
        """Number of tracked continuation points."""
        return int(self.p.shape[0])


def _find_zero_crossings(signal: jax.Array, time_grid: jax.Array) -> list[float]:
    """Find linearly interpolated zero crossings of a 1D periodic signal."""
    s0 = jnp.asarray(signal, dtype=jnp.float64)
    t0 = jnp.asarray(time_grid, dtype=jnp.float64)
    s1 = jnp.roll(s0, -1)
    t1_raw = jnp.roll(t0, -1)
    period_span = t0[-1] - t0[0] + (t0[1] - t0[0])
    t1 = jnp.where(t1_raw < t0, t1_raw + period_span, t1_raw)

    ds = s1 - s0
    mask = ((s0 * s1) <= 0.0) & (jnp.abs(ds) > 1e-12)
    t_cross = t0 - s0 * (t1 - t0) / jnp.where(mask, ds, 1.0)
    return [float(v) for v in t_cross[mask]]


def _solve_adjoint_collocation_jax(
    problem: CollocationProblem,
    mesh: jax.Array,
    u_mesh: jax.Array,
    u_gauss: jax.Array,
    T: jax.Array,
    p: float | jax.Array,
    omega_val: jax.Array,
) -> jax.Array:
    """Pure-JAX vectorized linear collocation solve for the periodic adjoint BVP."""
    N, m, n = problem.num_intervals, problem.num_gauss_points, problem.dim
    D = jnp.asarray(problem.D, dtype=jnp.float64)
    C = jnp.asarray(problem.C, dtype=jnp.float64)
    h = mesh[1:] - mesh[:-1]
    p_val = jnp.squeeze(jnp.asarray(p, dtype=jnp.float64))

    jac_fn = jax.jacobian(lambda u: problem.fn(u, p_val, **problem.kwargs))
    Ju_all = jax.vmap(jax.vmap(jac_fn))(u_gauss)
    f0 = problem.fn(u_mesh[0], p_val, **problem.kwargs)

    n_mesh = N * n
    dim_z = N * (m + 1) * n

    def adjoint_operator(z_flat: jax.Array) -> jax.Array:
        Z_mesh = z_flat[:n_mesh].reshape((N, n))
        Z_gauss = z_flat[n_mesh:].reshape((N, m, n))
        Z_full = jnp.concatenate([Z_mesh[:, None, :], Z_gauss], axis=1)

        dZ_dt = jnp.einsum("mj,Njd->Nmd", D, Z_full) / h[:, None, None]
        jt_z = jnp.einsum("Nmji,Nmj->Nmi", Ju_all, Z_gauss)
        res_coll = (dZ_dt + T * jt_z).ravel()

        Z_end = jnp.einsum("j,Njd->Nd", C, Z_full)
        res_cont = (Z_mesh[1:] - Z_end[:-1]).ravel()
        res_bc = (Z_mesh[0, : n - 1] - Z_end[N - 1, : n - 1]).ravel()
        res_norm = jnp.atleast_1d(jnp.dot(f0, Z_mesh[0]))

        return jnp.concatenate([res_coll, res_cont, res_bc, res_norm])

    L = jax.jacobian(adjoint_operator)(jnp.zeros(dim_z, dtype=jnp.float64))
    rhs = jnp.zeros(dim_z, dtype=jnp.float64).at[-1].set(omega_val)
    return jnp.linalg.solve(L, rhs)[:n_mesh].reshape((N, n))


def _build_iprc_result(
    theta_grid: jax.Array,
    orbit: jax.Array,
    Z: jax.Array,
    period: jax.Array,
    omega: jax.Array,
) -> IPRCResult:
    n = int(Z.shape[1])
    return IPRCResult(
        ts=theta_grid * period,
        theta=theta_grid,
        orbit=orbit,
        Z=Z,
        period=period,
        omega=omega,
        peak_amplitudes=jnp.max(jnp.abs(Z), axis=0),
        zero_crossings=[_find_zero_crossings(Z[:, comp], theta_grid) for comp in range(n)],
    )


def compute_iprc(
    sol_or_fn: PeriodicOrbitResult | Callable[..., jax.Array],
    u_mesh: ArrayLike | None = None,
    T: float | jax.Array | None = None,
    p: float | jax.Array = 0.0,
    omega_target: float | jax.Array | None = None,
    **kwargs: Any,
) -> IPRCResult:
    """Compute infinitesimal Phase Response Curve Z(t) via Gauss-Legendre collocation adjoint BVP.

    Accepts either a converged ``PeriodicOrbitResult`` or a vector field ``fn`` with ``(u_mesh, T)``.
    """
    if isinstance(sol_or_fn, PeriodicOrbitResult):
        sol = sol_or_fn
    else:
        if u_mesh is None or T is None:
            msg = "Must provide u_mesh and T when calling compute_iprc with a function."
            raise ValueError(msg)
        u_arr = jnp.asarray(u_mesh, dtype=jnp.float64)
        prob = CollocationProblem(
            sol_or_fn, dim=int(u_arr.shape[1]), num_intervals=int(u_arr.shape[0]), **kwargs
        )
        sol = solve_periodic_orbit(prob, u_mesh_init=u_arr, T_init=T, p=p)

    problem = sol.problem
    T_arr = jnp.squeeze(jnp.asarray(sol.period, dtype=jnp.float64))
    mesh = jnp.asarray(sol.mesh, dtype=jnp.float64)
    u_gauss = jnp.asarray(sol.u_gauss, dtype=jnp.float64)
    u_mesh_arr = jnp.asarray(sol.u_mesh, dtype=jnp.float64)

    omega_val = (
        2.0 * jnp.pi / T_arr
        if omega_target is None
        else jnp.squeeze(jnp.asarray(omega_target, dtype=jnp.float64))
    )
    Z_eval = _solve_adjoint_collocation_jax(problem, mesh, u_mesh_arr, u_gauss, T_arr, p, omega_val)
    return _build_iprc_result(mesh[:-1], u_mesh_arr, Z_eval, T_arr, omega_val)


def continuation_iprc(
    problem: CollocationProblem,
    u_mesh_init: ArrayLike,
    T_init: float | jax.Array,
    p_start: float | jax.Array,
    p_end: float | jax.Array,
    num_steps: int = 30,
    tol: float = 1e-8,
    max_iters: int = 20,
) -> IPRCContinuationResult:
    """Perform parameter continuation of the infinitesimal Phase Response Curve (iPRC) via `jax.lax.scan`."""
    N = problem.num_intervals
    p_values = jnp.linspace(
        jnp.squeeze(jnp.asarray(p_start, dtype=jnp.float64)),
        jnp.squeeze(jnp.asarray(p_end, dtype=jnp.float64)),
        num_steps,
    )
    mesh = jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float64)

    u_mesh0 = jnp.asarray(u_mesh_init, dtype=jnp.float64)
    u_k1 = jnp.roll(u_mesh0, -1, axis=0)
    rho = problem.rho
    u_gauss0 = (1.0 - rho[None, :, None]) * u_mesh0[:, None, :] + rho[None, :, None] * u_k1[
        :, None, :
    ]
    T0 = jnp.squeeze(jnp.asarray(T_init, dtype=jnp.float64))

    init_carry = (u_mesh0, u_gauss0, T0, jnp.bool_(True))

    def scan_step(carry, p_val):
        u_m, u_g, cur_T, is_active = carry
        sol_orbit = solve_periodic_orbit(
            problem=problem,
            u_mesh_init=u_m,
            T_init=cur_T,
            p=p_val,
            mesh=mesh,
            u_gauss_init=u_g,
            tol=tol,
            max_iters=max_iters,
        )
        step_ok = is_active & sol_orbit.converged
        u_m_next = jnp.where(step_ok, sol_orbit.u_mesh, u_m)
        u_g_next = jnp.where(step_ok, sol_orbit.u_gauss, u_g)
        T_next = jnp.where(step_ok, sol_orbit.period, cur_T)

        omega_val = 2.0 * jnp.pi / T_next
        Z_step = _solve_adjoint_collocation_jax(
            problem, mesh, u_m_next, u_g_next, T_next, p_val, omega_val
        )
        peak_amp = jnp.max(jnp.abs(Z_step), axis=0)

        carry_next = (u_m_next, u_g_next, T_next, step_ok)
        out = (p_val, T_next, peak_amp, u_m_next, Z_step, omega_val, step_ok)
        return carry_next, out

    _, outs = jax.lax.scan(scan_step, init_carry, p_values)
    p_all, T_all, peak_all, orbit_all, Z_all, omega_all, ok_all = outs

    p_kept = p_all[ok_all]
    T_kept = T_all[ok_all]
    peak_kept = peak_all[ok_all]
    orbit_kept = orbit_all[ok_all]
    Z_kept = Z_all[ok_all]
    omega_kept = omega_all[ok_all]

    theta_grid = mesh[:-1]
    results_list = [
        _build_iprc_result(theta_grid, orbit_kept[k], Z_kept[k], T_kept[k], omega_kept[k])
        for k in range(int(p_kept.shape[0]))
    ]

    return IPRCContinuationResult(
        p=p_kept,
        periods=T_kept,
        peak_amplitudes=peak_kept,
        orbits=orbit_kept,
        Z=Z_kept,
        results=results_list,
    )
