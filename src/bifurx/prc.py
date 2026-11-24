from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.bvp.collocation import (
    CollocationProblem,
    PeriodicOrbitResult,
    continue_periodic_orbit,
    solve_periodic_orbit,
)
from bifurx.bvp.condensation import condense_local_blocks
from bifurx.homotopy.bordered import safe_lu_factor, safe_lu_solve
from bifurx.options import CollocationOptions, ContinuationOptions


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
    *,
    rcond: float = 1e-12,
) -> jax.Array:
    """Pure-JAX de Boor-Swartz block-condensed linear solve for the periodic adjoint BVP."""
    N, n = problem.num_intervals, problem.dim
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    u_mesh_arr = jnp.asarray(u_mesh, dtype=jnp.float64)
    u_gauss_arr = jnp.asarray(u_gauss, dtype=jnp.float64)
    T_val = jnp.squeeze(jnp.asarray(T, dtype=jnp.float64))
    p_val = jnp.squeeze(jnp.asarray(p, dtype=jnp.float64))
    omega_arr = jnp.squeeze(jnp.asarray(omega_val, dtype=jnp.float64))
    h = mesh_arr[1:] - mesh_arr[:-1]
    eye_n = jnp.eye(n, dtype=jnp.float64)

    jac_fn = jax.jacfwd(lambda u: problem.fn(u, p_val, **problem.kwargs))
    Ju_all = jax.vmap(jax.vmap(jac_fn))(u_gauss_arr)

    f0 = jnp.atleast_1d(
        jnp.asarray(problem.fn(u_mesh_arr[0], p_val, **problem.kwargs), dtype=jnp.float64)
    ).ravel()
    f0_norm = jnp.linalg.norm(f0)
    f0_unit = f0 / jnp.where(f0_norm > 1e-30, f0_norm, 1.0)

    # Local adjoint collocation condensation: endpoint map Z_end,k = M_k^adj Z_mesh,k
    _, _, _, M_k = condense_local_blocks(problem, h, T_val, Ju_all, adjoint=True)

    # Condensed (N*n + 1) x (N*n + 1) system for [Z_mesh, alpha]:
    #   Z_mesh,k+1 - M_k^adj Z_mesh,k = 0                      (k = 0..N-2)
    #   Z_mesh,0   - M_{N-1}^adj Z_mesh,N-1 + alpha f0_unit = 0
    #   f0^T Z_mesh,0 = omega_val
    n_mesh = N * n
    k_idx = jnp.arange(N)
    cont_mat = jnp.zeros((N, n, N, n), dtype=jnp.float64)
    cont_mat = cont_mat.at[k_idx, :, (k_idx + 1) % N, :].add(eye_n)
    cont_mat = cont_mat.at[k_idx, :, k_idx, :].add(-M_k)
    cont_mat = cont_mat.reshape(n_mesh, n_mesh)

    M_cond = jnp.zeros((n_mesh + 1, n_mesh + 1), dtype=jnp.float64)
    M_cond = M_cond.at[:n_mesh, :n_mesh].set(cont_mat)
    # Keep all n periodic BC rows with rank-1 border alpha * f0_unit so M_cond stays
    # nonsingular regardless of which component of f0 is zero.
    M_cond = M_cond.at[(N - 1) * n : n_mesh, n_mesh].set(f0_unit)
    M_cond = M_cond.at[n_mesh, :n].set(f0)

    rhs_cond = jnp.zeros(n_mesh + 1, dtype=jnp.float64).at[n_mesh].set(omega_arr)

    lu_cond, piv_cond, use_lu_cond = safe_lu_factor(M_cond, rcond=rcond)
    sol_cond = safe_lu_solve(M_cond, lu_cond, piv_cond, use_lu_cond, rhs_cond, rcond=rcond)
    return sol_cond[:n_mesh].reshape((N, n))


def _build_iprc_result(
    theta_grid: jax.Array,
    orbit: jax.Array,
    Z: jax.Array,
    period: jax.Array,
    omega: jax.Array,
    *,
    compute_zero_crossings: bool = True,
) -> IPRCResult:
    n = int(Z.shape[1])
    zcs = (
        [_find_zero_crossings(Z[:, comp], theta_grid) for comp in range(n)]
        if compute_zero_crossings
        else []
    )
    return IPRCResult(
        ts=theta_grid * period,
        theta=theta_grid,
        orbit=orbit,
        Z=Z,
        period=period,
        omega=omega,
        peak_amplitudes=jnp.max(jnp.abs(Z), axis=0),
        zero_crossings=zcs,
    )


def compute_iprc(
    sol_or_fn: PeriodicOrbitResult | Callable[..., jax.Array],
    u_mesh: ArrayLike | None = None,
    T: float | jax.Array | None = None,
    p: float | jax.Array | None = None,
    omega_target: float | jax.Array | None = None,
    *,
    compute_zero_crossings: bool = True,
    **kwargs: Any,
) -> IPRCResult:
    """Compute infinitesimal Phase Response Curve Z(t) via Gauss-Legendre collocation adjoint BVP.

    Accepts either a converged ``PeriodicOrbitResult`` or a vector field ``fn`` with ``(u_mesh, T)``.
    Set ``compute_zero_crossings=False`` for pure-JAX ``jax.jit`` / ``jax.vmap`` compatibility.
    """
    if isinstance(sol_or_fn, PeriodicOrbitResult):
        sol = sol_or_fn
        p_val: float | jax.Array = (sol.p if sol.p is not None else 0.0) if p is None else p
    else:
        if u_mesh is None or T is None:
            msg = "Must provide u_mesh and T when calling compute_iprc with a function."
            raise ValueError(msg)
        p_val = 0.0 if p is None else p
        u_arr = jnp.asarray(u_mesh, dtype=jnp.float64)
        prob = CollocationProblem(
            sol_or_fn, dim=int(u_arr.shape[1]), num_intervals=int(u_arr.shape[0]), **kwargs
        )
        sol = solve_periodic_orbit(prob, u_mesh_init=u_arr, T_init=T, p=p_val)

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
    Z_eval = _solve_adjoint_collocation_jax(
        problem, mesh, u_mesh_arr, u_gauss, T_arr, p_val, omega_val
    )
    return _build_iprc_result(
        mesh[:-1],
        u_mesh_arr,
        Z_eval,
        T_arr,
        omega_val,
        compute_zero_crossings=compute_zero_crossings,
    )


def continuation_iprc(
    problem: CollocationProblem,
    u_mesh_init: ArrayLike,
    T_init: float | jax.Array,
    p_start: float | jax.Array,
    p_end: float | jax.Array,
    num_steps: int = 30,
    tol: float = 1e-8,
    max_iters: int = 20,
    *,
    param_mode: str = "natural",
    dp: float | jax.Array | None = None,
    dp_min: float | None = None,
    dp_max: float | None = None,
    max_steps: int | None = None,
    param_weight: float | None = None,
    period_weight: float | None = None,
    predictor: str | None = None,
    adaptive_step: bool | None = None,
    non_trivial: bool = True,
    options: ContinuationOptions | CollocationOptions | None = None,
) -> IPRCContinuationResult:
    """Continue periodic orbits and evaluate iPRC Z(t) in parallel via `jax.vmap`."""
    N, n = problem.num_intervals, problem.dim
    init_orbit = solve_periodic_orbit(
        problem=problem,
        u_mesh_init=u_mesh_init,
        T_init=T_init,
        p=p_start,
        tol=tol,
        max_iters=max_iters,
        non_trivial=non_trivial,
    )
    if not bool(init_orbit.converged):
        return IPRCContinuationResult(
            p=jnp.zeros((0,), dtype=jnp.float64),
            periods=jnp.zeros((0,), dtype=jnp.float64),
            peak_amplitudes=jnp.zeros((0, n), dtype=jnp.float64),
            orbits=jnp.zeros((0, N, n), dtype=jnp.float64),
            Z=jnp.zeros((0, N, n), dtype=jnp.float64),
            results=[],
        )

    legacy_grid = (
        param_mode == "natural"
        and dp is None
        and max_steps is None
        and adaptive_step is None
        and options is None
    )
    if legacy_grid:
        steps_eff = max(int(num_steps) - 1, 0)
        dp_val = abs(float(p_end) - float(p_start)) / max(steps_eff, 1)
        dp_eff: float | jax.Array | None = dp_val
        dp_min_eff = min(dp_val, 1e-5) if (dp_min is None and dp_val > 0.0) else dp_min
        dp_max_eff = max(dp_val, 0.1) if dp_max is None else dp_max
        adapt_eff: bool | None = False
    else:
        steps_eff = int(num_steps) if max_steps is None else int(max_steps)
        dp_eff = dp
        dp_min_eff = dp_min
        dp_max_eff = dp_max
        adapt_eff = adaptive_step

    orbits = continue_periodic_orbit(
        problem=problem,
        initial_orbit=init_orbit,
        p_start=p_start,
        p_end=p_end,
        dp=dp_eff,
        dp_min=dp_min_eff,
        dp_max=dp_max_eff,
        max_steps=steps_eff,
        param_mode=param_mode,
        param_weight=param_weight,
        period_weight=period_weight,
        non_trivial=non_trivial,
        tol=tol,
        max_iters=max_iters,
        options=options,
        predictor=predictor,
        adaptive_step=adapt_eff,
    )

    mesh = jnp.asarray(init_orbit.mesh, dtype=jnp.float64)
    p_all = jnp.stack(
        [
            jnp.squeeze(jnp.asarray(p_start if orb.p is None else orb.p, dtype=jnp.float64))
            for orb in orbits
        ]
    )
    u_mesh_all = jnp.stack([jnp.asarray(orb.u_mesh, dtype=jnp.float64) for orb in orbits])
    u_gauss_all = jnp.stack([jnp.asarray(orb.u_gauss, dtype=jnp.float64) for orb in orbits])
    T_all = jnp.stack([jnp.squeeze(jnp.asarray(orb.period, dtype=jnp.float64)) for orb in orbits])
    omega_all = 2.0 * jnp.pi / T_all

    Z_all = jax.vmap(
        lambda um, ug, t_v, p_v, om_v: _solve_adjoint_collocation_jax(
            problem, mesh, um, ug, t_v, p_v, om_v
        )
    )(u_mesh_all, u_gauss_all, T_all, p_all, omega_all)
    peak_all = jnp.max(jnp.abs(Z_all), axis=1)

    theta_grid = mesh[:-1]
    results_list = [
        _build_iprc_result(theta_grid, u_mesh_all[k], Z_all[k], T_all[k], omega_all[k])
        for k in range(len(orbits))
    ]

    return IPRCContinuationResult(
        p=p_all,
        periods=T_all,
        peak_amplitudes=peak_all,
        orbits=u_mesh_all,
        Z=Z_all,
        results=results_list,
    )
