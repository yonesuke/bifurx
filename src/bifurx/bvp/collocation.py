from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.bifurcations import (
    compute_first_lyapunov_coefficient,
    extract_leading_hopf_pair,
    test_func_hopf,
)
from bifurx.bvp.condensation import (
    factor_condensed_collocation,
    make_collocation_bordered_solver,
    solve_condensed_collocation,
)
from bifurx.bvp.floquet import (
    evaluate_floquet_diagnostics,
    evaluate_floquet_diagnostics_from_factors,
    monodromy_condensation_raw,
    monodromy_factors_condensation,
)
from bifurx.bvp.mesh import equidistribute_mesh, interpolate_orbit_to_mesh
from bifurx.bvp.problem import (
    CollocationProblem,
    PeriodicOrbitResult,
    _orbit_amplitude,
)
from bifurx.homotopy import BorderedOperatorSolver, compute_tangent, track_arclength_scan
from bifurx.homotopy.corrector import correct_arclength_step
from bifurx.options import (
    CollocationOptions,
    ContinuationOptions,
    resolve_collocation_options,
)


def _hopf_harmonic_seed(
    problem: CollocationProblem,
    u_h: jax.Array,
    p_h: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Extract normalized Hopf eigenpair and unit-amplitude harmonic profiles on uniform mesh."""
    jac_u_fn = jax.jacobian(lambda u_, p_: problem.fn(u_, p_, **problem.kwargs), argnums=0)
    Ju = jac_u_fn(u_h, p_h)
    eigs, vecs = jnp.linalg.eig(Ju)
    _, leading_real, omega = extract_leading_hopf_pair(Ju, eigs=eigs)
    target_eig = leading_real + 1j * omega
    T0 = 2.0 * jnp.pi / omega
    vec = vecs[:, jnp.argmin(jnp.abs(eigs - target_eig))]

    v_r, v_i = jnp.real(vec), jnp.imag(vec)
    norm_v = jnp.sqrt(jnp.dot(v_r, v_r) + jnp.dot(v_i, v_i))
    v_r, v_i = v_r / norm_v, v_i / norm_v

    N = problem.num_intervals
    mesh = jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float64)
    rho = jnp.asarray(problem.rho, dtype=jnp.float64)
    t_mesh = mesh[:-1]
    theta_mesh = 2.0 * jnp.pi * t_mesh
    du_mesh = (
        jnp.cos(theta_mesh)[:, None] * v_r[None, :] - jnp.sin(theta_mesh)[:, None] * v_i[None, :]
    )

    hk = mesh[1:] - mesh[:-1]
    t_gauss = t_mesh[:, None] + hk[:, None] * rho[None, :]
    theta_gauss = 2.0 * jnp.pi * t_gauss
    du_gauss = (
        jnp.cos(theta_gauss)[:, :, None] * v_r[None, None, :]
        - jnp.sin(theta_gauss)[:, :, None] * v_i[None, None, :]
    )
    dot_du_gauss = (
        -2.0 * jnp.pi * jnp.sin(theta_gauss)[:, :, None] * v_r[None, None, :]
        - 2.0 * jnp.pi * jnp.cos(theta_gauss)[:, :, None] * v_i[None, None, :]
    )
    return Ju, eigs, omega, T0, mesh, du_mesh, du_gauss, dot_du_gauss


def initialize_from_hopf(
    problem: CollocationProblem,
    u_hopf: ArrayLike,
    p_hopf: float | jax.Array,
    eps: float | jax.Array | None = None,
    p_init: float | jax.Array | None = None,
    direction: float | str = -1.0,
    ds: float | None = None,
    options: CollocationOptions | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Construct an initial periodic orbit guess near a Hopf bifurcation in pure JAX."""
    opts = resolve_collocation_options(options, eps=eps)
    eps_val = jnp.squeeze(jnp.asarray(opts.eps, dtype=jnp.float64))

    u_h = jnp.asarray(u_hopf, dtype=jnp.float64)
    p_h = jnp.squeeze(jnp.asarray(p_hopf, dtype=jnp.float64))

    Ju, eigs, omega, T0, mesh, du_mesh, du_gauss, _ = _hopf_harmonic_seed(problem, u_h, p_h)

    complex_mask = jnp.abs(jnp.imag(eigs)) > 1e-4 * (1.0 + jnp.abs(eigs))
    crit_mask = complex_mask & (jnp.abs(jnp.real(eigs)) < 1e-2)
    crit_freqs = jnp.where(crit_mask, jnp.abs(jnp.imag(eigs)), -1.0)
    matches = (jnp.abs(crit_freqs - omega) < 1e-3) & (crit_freqs > 0)

    def _warn_degenerate() -> None:
        warnings.warn(
            "Degenerate Hopf point detected: critical imaginary eigenvalue has algebraic "
            "multiplicity >= 2. A single eigenvector seed may collapse onto the trivial "
            "equilibrium. Consider increasing eps, setting non_trivial=True, or providing a custom initial guess.",
            UserWarning,
            stacklevel=2,
        )

    jax.lax.cond(
        jnp.sum(matches) > 2,
        lambda: jax.debug.callback(_warn_degenerate),
        lambda: None,
    )

    u_mesh = u_h[None, :] + eps_val * du_mesh
    u_gauss = u_h[None, None, :] + eps_val * du_gauss

    if isinstance(direction, str):
        if direction != "auto":
            msg = f"Unknown direction: {direction}. Pass a float or 'auto'."
            raise ValueError(msg)
        jac_u_fn = jax.jacobian(lambda u_, p_: problem.fn(u_, p_, **problem.kwargs), argnums=0)
        l1 = compute_first_lyapunov_coefficient(
            lambda u_, p_: problem.fn(u_, p_, **problem.kwargs),
            u_h,
            p_h,
            Ju=Ju,
        )
        Jp = jax.jacfwd(lambda p_: problem.fn(u_h, p_, **problem.kwargs))(p_h)
        du_dp_direct = -jnp.linalg.solve(Ju, Jp)
        du_dp = jax.lax.cond(
            jnp.all(jnp.isfinite(du_dp_direct)),
            lambda: du_dp_direct,
            lambda: -jnp.linalg.lstsq(Ju, Jp, rcond=1e-12)[0],
        )
        dp_fd = 1e-6 * (1.0 + jnp.abs(p_h))
        alpha_plus = test_func_hopf(jac_u_fn(u_h + dp_fd * du_dp, p_h + dp_fd))
        alpha_minus = test_func_hopf(jac_u_fn(u_h - dp_fd * du_dp, p_h - dp_fd))
        alpha_prime = (alpha_plus - alpha_minus) / (2.0 * dp_fd)
        prod = alpha_prime * l1
        dir_val = jnp.where(
            jnp.isfinite(prod) & (jnp.abs(prod) > 1e-14),
            jnp.where(prod > 0.0, -1.0, 1.0),
            -1.0,
        )
    else:
        dir_val = jnp.squeeze(jnp.asarray(direction, dtype=jnp.float64))

    if p_init is not None:
        p_init_arr = jnp.squeeze(jnp.asarray(p_init, dtype=jnp.float64))
    else:
        step = eps_val if ds is None else jnp.squeeze(jnp.asarray(ds, dtype=jnp.float64))
        p_init_arr = p_h + dir_val * step

    return mesh, u_mesh, u_gauss, T0, p_init_arr


def branch_from_hopf(
    problem: CollocationProblem,
    u_hopf: ArrayLike | Sequence[float],
    p_hopf: float | jax.Array,
    ds: float | jax.Array = 0.05,
    period_weight: float | None = None,
    param_weight: float = 1.0,
    tol: float | None = None,
    max_iters: int | None = None,
    min_amplitude: float | None = None,
    options: CollocationOptions | None = None,
) -> PeriodicOrbitResult:
    """Switch from a Hopf point onto the periodic orbit branch via arclength-bordered Newton.

    Follows the AUTO-07P / MatCont Hopf branch-switching algorithm: starting from the
    equilibrium baseline `(u_hopf, T0, p_hopf)` with vertical tangent
    `tau0 = (cos(2*pi*t) v_r - sin(2*pi*t) v_i, dT=0, dp=0)`, solves jointly for
    `(u(t), T, p)` at arclength distance `ds`. Automatically handles both supercritical
    and subcritical Hopf bifurcations without fixed-parameter tuning.
    """
    opts = resolve_collocation_options(
        options,
        tol=tol,
        max_iters=max_iters,
        min_amplitude=min_amplitude,
        period_weight=period_weight,
    )
    tol_val = opts.tol
    max_iters_val = opts.max_iters
    min_amp_val = opts.min_amplitude
    t_w = opts.period_weight
    p_w = param_weight
    ds_val = jnp.squeeze(jnp.asarray(ds, dtype=jnp.float64))

    u_h = jnp.asarray(u_hopf, dtype=jnp.float64)
    p_h = jnp.squeeze(jnp.asarray(p_hopf, dtype=jnp.float64))

    _, _, _, T0, mesh, du_mesh, du_gauss, dot_u_ref = _hopf_harmonic_seed(problem, u_h, p_h)
    N, m, n = problem.num_intervals, problem.num_gauss_points, problem.dim

    u_mesh_eq = jnp.broadcast_to(u_h[None, :], (N, n))
    u_gauss_eq = jnp.broadcast_to(u_h[None, None, :], (N, m, n))
    X_eq = problem.pack_vars(u_mesh_eq, u_gauss_eq, T0)

    tau_X_raw = problem.pack_vars(du_mesh, du_gauss, 0.0)
    tau_raw = jnp.concatenate([tau_X_raw, jnp.zeros(1, dtype=jnp.float64)])
    w_arc = periodic_orbit_arclength_weights(problem, mesh, period_weight=t_w, param_weight=p_w)
    tau_norm = jnp.sqrt(jnp.maximum(jnp.dot(w_arc * tau_raw, tau_raw), 1e-30))
    tau0 = tau_raw / tau_norm

    X_pred = X_eq + ds_val * tau0[:-1]
    p_pred = p_h

    bordered_solver = make_collocation_bordered_solver(problem, mesh, dot_u_ref)

    def _res_fn(X: jax.Array, p_val: jax.Array) -> jax.Array:
        return problem.residual(X, p_val, mesh, dot_u_ref=dot_u_ref)

    res = correct_arclength_step(
        residual_fn=_res_fn,
        u_init=X_pred,
        p_init=p_pred,
        u_prev=X_eq,
        p_prev=p_h,
        tau_prev=tau0,
        ds=ds_val,
        tol=tol_val,
        step_tol=tol_val,
        max_iters=max_iters_val,
        border_c=w_arc[:-1] * tau0[:-1],
        border_d=w_arc[-1] * tau0[-1],
        bordered_solver=bordered_solver,
    )

    u_mesh_sol, u_gauss_sol, T_sol = problem.unpack_vars(res.u)
    conv = (
        res.converged
        & jnp.isfinite(T_sol)
        & (T_sol > 0.0)
        & (_orbit_amplitude(u_mesh_sol) >= min_amp_val)
    )
    return PeriodicOrbitResult(
        mesh=mesh,
        u_mesh=u_mesh_sol,
        u_gauss=u_gauss_sol,
        period=T_sol,
        converged=conv,
        iterations=res.iterations,
        residual_norm=res.residual_norm,
        problem=problem,
        p=res.p,
    )


def solve_periodic_orbit(
    problem: CollocationProblem,
    u_mesh_init: ArrayLike,
    T_init: float | jax.Array,
    p: float | jax.Array,
    mesh: ArrayLike | None = None,
    u_gauss_init: ArrayLike | None = None,
    u_ref_mesh: ArrayLike | None = None,
    u_ref_gauss: ArrayLike | None = None,
    tol: float | None = None,
    max_iters: int | None = None,
    non_trivial: bool = False,
    min_amplitude: float | None = None,
    options: CollocationOptions | None = None,
) -> PeriodicOrbitResult:
    """Solve for a periodic orbit (u(t), T) using pure-JAX Gauss-Legendre collocation."""
    opts = resolve_collocation_options(
        options, tol=tol, max_iters=max_iters, min_amplitude=min_amplitude
    )
    tol_val = opts.tol
    max_iters_val = opts.max_iters
    min_amp_val = opts.min_amplitude

    N = problem.num_intervals
    n_mesh = N * problem.dim

    mesh_arr = (
        jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float64)
        if mesh is None
        else jnp.asarray(mesh, dtype=jnp.float64)
    )
    u_mesh_arr = jnp.asarray(u_mesh_init, dtype=jnp.float64)
    amp_init = _orbit_amplitude(u_mesh_arr)

    if u_gauss_init is None:
        u_k1 = jnp.roll(u_mesh_arr, -1, axis=0)
        if problem.period_shift is not None:
            u_k1 = u_k1.at[-1].add(problem.period_shift)
        rho = problem.rho
        u_gauss_arr = (1.0 - rho[None, :, None]) * u_mesh_arr[:, None, :] + rho[
            None, :, None
        ] * u_k1[:, None, :]
    else:
        u_gauss_arr = jnp.asarray(u_gauss_init, dtype=jnp.float64)

    u_ref_mesh_arr = (
        u_mesh_arr if u_ref_mesh is None else jnp.asarray(u_ref_mesh, dtype=jnp.float64)
    )
    u_ref_gauss_arr = (
        u_gauss_arr if u_ref_gauss is None else jnp.asarray(u_ref_gauss, dtype=jnp.float64)
    )

    p_arr = jnp.squeeze(jnp.asarray(p, dtype=jnp.float64))
    T_init_arr = jnp.squeeze(jnp.asarray(T_init, dtype=jnp.float64))
    X_init = problem.pack_vars(u_mesh_arr, u_gauss_arr, T_init_arr)
    dot_u_ref = problem.derivative_at_gauss(u_ref_mesh_arr, u_ref_gauss_arr, mesh_arr)

    def eval_res(x_vec: jax.Array) -> jax.Array:
        return problem.residual(
            x_vec, p_arr, mesh_arr, u_ref_gauss=u_ref_gauss_arr, dot_u_ref=dot_u_ref
        )

    res_norm_0 = jnp.linalg.norm(eval_res(X_init))
    init_state = (X_init, jnp.int32(0), res_norm_0 < tol_val, res_norm_0)

    def cond_fn(state: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> jax.Array:
        _, it, conv, _ = state
        return (~conv) & (it < max_iters_val)

    def body_fn(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        X, it, _, res_norm = state
        res = eval_res(X)
        fac = factor_condensed_collocation(problem, X, p_arr, mesh_arr, dot_u_ref)
        dX = solve_condensed_collocation(fac, -res)

        def _step_score(x_cand: jax.Array) -> jax.Array:
            rn = jnp.linalg.norm(eval_res(x_cand))
            if non_trivial:
                amp_cand = _orbit_amplitude(x_cand[:n_mesh].reshape((N, problem.dim)))
                return jnp.where((amp_init >= min_amp_val) & (amp_cand < min_amp_val), jnp.inf, rn)
            return rn

        X1, X2, X3 = X + dX, X + 0.5 * dX, X + 0.25 * dX
        r1, r2, r3 = _step_score(X1), _step_score(X2), _step_score(X3)

        X_next = jnp.where(
            r1 < res_norm, X1, jnp.where(r2 < res_norm, X2, jnp.where(r3 < res_norm, X3, X1))
        )
        rn_next = jnp.where(
            r1 < res_norm, r1, jnp.where(r2 < res_norm, r2, jnp.where(r3 < res_norm, r3, r1))
        )

        is_valid = jnp.isfinite(rn_next) & jnp.all(jnp.isfinite(X_next))
        conv_next = is_valid & (rn_next < tol_val)
        it_next = jnp.where(is_valid, it + 1, jnp.int32(max_iters_val))
        return (X_next, it_next, conv_next, rn_next)

    X_fin, it_fin, conv_fin, rn_fin = jax.lax.while_loop(cond_fn, body_fn, init_state)
    u_mesh_sol, u_gauss_sol, T_sol = problem.unpack_vars(X_fin)

    if non_trivial:
        conv_fin = conv_fin & (_orbit_amplitude(u_mesh_sol) >= min_amp_val)

    return PeriodicOrbitResult(
        mesh=mesh_arr,
        u_mesh=u_mesh_sol,
        u_gauss=u_gauss_sol,
        period=T_sol,
        converged=conv_fin,
        iterations=it_fin,
        residual_norm=rn_fin,
        problem=problem,
        p=p_arr,
    )


def periodic_orbit_arclength_weights(
    problem: CollocationProblem,
    mesh: ArrayLike,
    period_weight: float | jax.Array = 1.0,
    param_weight: float | jax.Array = 1.0,
) -> jax.Array:
    """Build exact Gauss-Legendre L2[0, 1] arclength weights for `[u_mesh, u_gauss, T, p]`."""
    if not isinstance(period_weight, jax.core.Tracer) and period_weight < 0.0:
        msg = f"period_weight must be non-negative, got {period_weight}."
        raise ValueError(msg)
    if not isinstance(param_weight, jax.core.Tracer) and param_weight < 0.0:
        msg = f"param_weight must be non-negative, got {param_weight}."
        raise ValueError(msg)
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    h_arr = mesh_arr[1:] - mesh_arr[:-1]
    N, m, n = problem.num_intervals, problem.num_gauss_points, problem.dim
    w_mesh = jnp.zeros(N * n, dtype=jnp.float64)
    w_gauss = jnp.broadcast_to(
        (h_arr[:, None] * problem.weights[None, :])[:, :, None], (N, m, n)
    ).ravel()
    t_w_arr = jnp.squeeze(jnp.asarray(period_weight, dtype=jnp.float64))
    p_w_arr = jnp.squeeze(jnp.asarray(param_weight, dtype=jnp.float64))
    return jnp.concatenate(
        [
            w_mesh,
            w_gauss,
            jnp.stack([t_w_arr**2, p_w_arr**2]),
        ]
    )


def continue_periodic_orbit(
    problem: CollocationProblem,
    initial_orbit: PeriodicOrbitResult,
    p_start: float | jax.Array,
    p_end: float | jax.Array,
    dp: float | jax.Array | None = None,
    dp_min: float | None = None,
    dp_max: float | None = None,
    max_steps: int | None = None,
    param_mode: str = "natural",
    param_weight: float | None = None,
    period_weight: float | None = None,
    non_trivial: bool = True,
    tol: float | None = None,
    max_iters: int | None = None,
    verbose: bool = False,
    options: ContinuationOptions | CollocationOptions | None = None,
    collocation_options: CollocationOptions | None = None,
    predictor: str | None = None,
    adaptive_step: bool | None = None,
    compute_floquet: bool = False,
    floquet_method: str = "periodic_qr",
    remesh_interval: int | None = None,
    mesh_alpha: float = 1.0,
    mesh_damping: float = 0.7,
    period_bounds: tuple[float, float] | None = None,
) -> list[PeriodicOrbitResult]:
    """Continue a family of periodic orbits along parameter p via `bifurx.homotopy`."""
    if param_mode not in ("natural", "arclength"):
        msg = f"Unknown param_mode: {param_mode}. Choose from 'natural' or 'arclength'."
        raise ValueError(msg)
    if floquet_method not in ("periodic_qr", "periodic_schur", "condensation"):
        msg = (
            f"Unknown floquet_method: {floquet_method}. "
            "Choose from 'periodic_qr', 'periodic_schur', or 'condensation'."
        )
        raise ValueError(msg)
    use_periodic_schur = floquet_method in ("periodic_qr", "periodic_schur")
    if remesh_interval is not None and remesh_interval <= 0:
        msg = f"remesh_interval must be positive, got {remesh_interval}."
        raise ValueError(msg)
    if period_bounds is not None:
        t_lo_val, t_hi_val = float(period_bounds[0]), float(period_bounds[1])
        if t_lo_val < 0.0 or t_lo_val >= t_hi_val:
            msg = f"Invalid period_bounds={period_bounds}. Require 0 <= T_min < T_max."
            raise ValueError(msg)
    else:
        t_lo_val, t_hi_val = 0.0, float("inf")

    cont_opts = options if isinstance(options, ContinuationOptions) else ContinuationOptions()
    coll_opts = (
        options
        if isinstance(options, CollocationOptions)
        else (CollocationOptions() if collocation_options is None else collocation_options)
    )
    has_cont = isinstance(options, ContinuationOptions)
    has_coll = isinstance(options, CollocationOptions) or collocation_options is not None

    cur_dp = float(abs(float(dp if dp is not None else (cont_opts.ds if has_cont else 0.01))))
    dp_min_val = float(dp_min if dp_min is not None else (cont_opts.ds_min if has_cont else 1e-5))
    dp_max_val = float(dp_max if dp_max is not None else (cont_opts.ds_max if has_cont else 0.1))
    max_steps_val = int(cont_opts.max_steps if max_steps is None else max_steps)
    tol_val = float(
        tol
        if tol is not None
        else (coll_opts.tol if has_coll else (cont_opts.newton.tol if has_cont else 1e-7))
    )
    max_iters_val = int(
        max_iters
        if max_iters is not None
        else (
            coll_opts.max_iters
            if has_coll
            else (cont_opts.newton.max_iters if has_cont else coll_opts.max_iters)
        )
    )
    min_amp_val = float(coll_opts.min_amplitude)
    pred_val = cont_opts.predictor if predictor is None else predictor
    adapt_val = (
        cont_opts.adaptive_step
        if (has_cont and adaptive_step is None)
        else (True if adaptive_step is None else adaptive_step)
    )

    p_s, p_e = float(p_start), float(p_end)
    p_lo, p_hi = min(p_s, p_e), max(p_s, p_e)
    direction = 1.0 if p_e >= p_s else -1.0
    p_w = 1.0 if param_weight is None else float(param_weight)
    t_w = float(coll_opts.period_weight if period_weight is None else period_weight)

    if initial_orbit.p is None:
        initial_orbit = initial_orbit._replace(p=jnp.asarray(p_s, dtype=jnp.float64))

    init_mesh = jnp.asarray(initial_orbit.mesh, dtype=jnp.float64)
    if param_mode == "arclength":
        if t_w < 0.0:
            msg = f"period_weight must be non-negative, got {t_w}."
            raise ValueError(msg)
        if p_w < 0.0:
            msg = f"param_weight must be non-negative, got {p_w}."
            raise ValueError(msg)

    if compute_floquet:
        if use_periodic_schur:
            factors0 = monodromy_factors_condensation(
                problem, init_mesh, initial_orbit.u_gauss, initial_orbit.period, p=p_s
            )
            init_mults, init_tf_fl, init_stable = evaluate_floquet_diagnostics_from_factors(
                factors0
            )
        else:
            M0 = monodromy_condensation_raw(
                problem, init_mesh, initial_orbit.u_gauss, initial_orbit.period, p=p_s
            )
            init_mults, init_tf_fl, init_stable = evaluate_floquet_diagnostics(M0)
        initial_orbit = initial_orbit._replace(
            floquet_multipliers=init_mults, is_stable=init_stable
        )
    else:
        init_tf_fl = {}

    if max_steps_val <= 0 or abs(p_e - p_s) <= 1e-12:
        return [initial_orbit]

    def _build_segment_problem(
        mesh_seg: jax.Array,
        X_ref0: jax.Array,
        p_ref0: jax.Array,
        prev_tau_and_mesh: tuple[jax.Array, jax.Array] | None,
    ) -> tuple[
        Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]],
        Callable[..., jax.Array],
        Callable[[tuple[jax.Array, jax.Array]], BorderedOperatorSolver],
        jax.Array | None,
        jax.Array,
        Callable[[jax.Array, jax.Array, jax.Array, Any, Any], Any] | None,
    ]:
        h_seg = mesh_seg[1:] - mesh_seg[:-1]
        chart_fn = problem.make_phase_chart_fn(mesh_seg)

        def res_fn(X: jax.Array, p_val: jax.Array, chart: tuple[jax.Array, jax.Array]) -> jax.Array:
            dot_u_ref, u_ref_g = chart
            return problem.residual(X, p_val, mesh_seg, u_ref_gauss=u_ref_g, dot_u_ref=dot_u_ref)

        def solver_factory(chart: tuple[jax.Array, jax.Array]) -> BorderedOperatorSolver:
            dot_u_ref, _ = chart
            return make_collocation_bordered_solver(problem, mesh_seg, dot_u_ref)

        w_arc_seg = (
            periodic_orbit_arclength_weights(problem, mesh_seg, period_weight=t_w, param_weight=p_w)
            if param_mode == "arclength"
            else None
        )

        chart0_seg = chart_fn(X_ref0, p_ref0)
        tau_seg = compute_tangent(
            u0=X_ref0,
            p0=p_ref0,
            direction=direction,
            arclength_weights=w_arc_seg,
            param_mode=param_mode,
            bordered_solver=solver_factory(chart0_seg),
        )
        if prev_tau_and_mesh is not None:
            old_tau, old_mesh = prev_tau_and_mesh
            if param_mode == "natural":
                if float(old_tau[-1] * tau_seg[-1]) < 0.0:
                    tau_seg = -tau_seg
            else:
                du_m_old, du_g_old, dT_old = problem.unpack_vars(old_tau[:-1])
                old_tau_orbit = PeriodicOrbitResult(
                    mesh=old_mesh,
                    u_mesh=du_m_old,
                    u_gauss=du_g_old,
                    period=dT_old,
                    converged=jnp.bool_(True),
                    iterations=jnp.int32(0),
                    residual_norm=jnp.array(0.0, dtype=jnp.float64),
                    problem=problem,
                    p=p_ref0,
                )
                t_gauss_new = (mesh_seg[:-1, None] + h_seg[:, None] * problem.rho[None, :]).ravel()
                du_g_interp = old_tau_orbit.evaluate(t_gauss_new).reshape(
                    problem.num_intervals, problem.num_gauss_points, problem.dim
                )
                _, du_g_new, dT_new = problem.unpack_vars(tau_seg[:-1])
                dot_l2 = (
                    jnp.sum(
                        h_seg[:, None, None]
                        * problem.weights[None, :, None]
                        * du_g_interp
                        * du_g_new
                    )
                    + (t_w**2) * dT_old * dT_new
                    + (p_w**2) * old_tau[-1] * tau_seg[-1]
                )
                if float(dot_l2) < 0.0:
                    tau_seg = -tau_seg

        def _floquet_obs(
            X_c: jax.Array,
            p_c: jax.Array,
            tau_c: jax.Array,
            _ju_c: Any,
            _jp_c: Any,
        ) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
            _, u_g_c, T_c = problem.unpack_vars(X_c)
            if use_periodic_schur:
                factors_c = monodromy_factors_condensation(problem, mesh_seg, u_g_c, T_c, p=p_c)
                return evaluate_floquet_diagnostics_from_factors(factors_c, tangent_p=tau_c[-1])
            M_c = monodromy_condensation_raw(problem, mesh_seg, u_g_c, T_c, p=p_c)
            return evaluate_floquet_diagnostics(M_c, tangent_p=tau_c[-1])

        obs_fn = _floquet_obs if compute_floquet else None
        return chart_fn, res_fn, solver_factory, w_arc_seg, tau_seg, obs_fn

    def accept_fn(X: jax.Array, _p: jax.Array) -> jax.Array:
        u_m, _, T_val = problem.unpack_vars(X)
        ok = jnp.isfinite(T_val) & (T_val > 0.0)
        if non_trivial:
            ok = ok & (_orbit_amplitude(u_m) >= min_amp_val)
        return ok

    if period_bounds is not None:

        def stop_fn(X: jax.Array, _p: jax.Array) -> jax.Array:
            T_val = jnp.squeeze(X[-1])
            return (T_val <= t_lo_val) | (T_val >= t_hi_val)
    else:
        stop_fn = None

    p_target_arg = None if param_mode == "arclength" else p_e
    cur_mesh = init_mesh
    cur_X = problem.pack_vars(initial_orbit.u_mesh, initial_orbit.u_gauss, initial_orbit.period)
    cur_p = jnp.asarray(p_s, dtype=jnp.float64)
    cur_ds = cur_dp
    prev_tm: tuple[jax.Array, jax.Array] | None = None
    remaining = max_steps_val
    chunk_size = max_steps_val if remesh_interval is None else remesh_interval

    results: list[PeriodicOrbitResult] = [initial_orbit]
    tf_history: list[dict[str, float]] = []
    step_counter = 0

    while remaining > 0:
        seg_steps = min(chunk_size, remaining)
        chart_fn, res_fn, solver_factory, w_arc_seg, tau_seg, obs_fn = _build_segment_problem(
            cur_mesh, cur_X, cur_p, prev_tm
        )
        if compute_floquet and not tf_history:
            tf_history.append(
                {
                    "LPC": float(tau_seg[-1]),
                    "PD": float(init_tf_fl["PD"]),
                    "NS": float(init_tf_fl["NS"]),
                }
            )

        scan_res = track_arclength_scan(
            residual_fn=res_fn,
            u0=cur_X,
            p0=cur_p,
            tau0=tau_seg,
            ds=cur_ds,
            ds_min=dp_min_val,
            ds_max=dp_max_val,
            n_steps=seg_steps,
            p_min=p_lo,
            p_max=p_hi,
            adaptive_step=adapt_val,
            predictor=pred_val,
            param_mode=param_mode,
            p_target=p_target_arg,
            arclength_weights=w_arc_seg,
            chart_fn=chart_fn,
            accept_fn=accept_fn,
            stop_fn=stop_fn,
            tol=tol_val,
            step_tol=tol_val,
            max_newton_iters=max_iters_val,
            solver_method=cont_opts.newton.solver_method,
            tracker_params=cont_opts.tracker,
            norm_options=cont_opts.norm,
            step_observer_fn=obs_fn,
            bordered_solver=solver_factory,
        )

        kept_indices = jnp.nonzero(scan_res.converged)[0].tolist()
        if not kept_indices:
            break

        for idx in kept_indices:
            step_counter += 1
            u_m_sol, u_g_sol, T_sol = problem.unpack_vars(scan_res.u[idx])
            if compute_floquet:
                mult_arr, tf_dict, stab_arr = scan_res.observer
                mult_i = mult_arr[idx]
                stab_i = stab_arr[idx]
                tf_history.append(
                    {
                        "LPC": float(tf_dict["LPC"][idx]),
                        "PD": float(tf_dict["PD"][idx]),
                        "NS": float(tf_dict["NS"][idx]),
                    }
                )
            else:
                mult_i = None
                stab_i = None

            sol = PeriodicOrbitResult(
                mesh=cur_mesh,
                u_mesh=u_m_sol,
                u_gauss=u_g_sol,
                period=T_sol,
                converged=scan_res.converged[idx],
                iterations=scan_res.iterations[idx],
                residual_norm=scan_res.residual_norms[idx],
                problem=problem,
                p=scan_res.p[idx],
                floquet_multipliers=mult_i,
                is_stable=stab_i,
            )
            results.append(sol)
            if verbose:
                print(
                    f"Step {step_counter}: p={float(scan_res.p[idx]):.5f}, T={float(sol.T):.5f}, "
                    f"amp={float(sol.amplitude):.5f}, iters={int(sol.iterations)}"
                )

        remaining -= seg_steps
        last_idx = kept_indices[-1]
        last_orb = results[-1]
        last_p = float(scan_res.p[last_idx])
        last_T = float(last_orb.period)
        if (
            last_p < p_lo
            or last_p > p_hi
            or (period_bounds is not None and (last_T <= t_lo_val or last_T >= t_hi_val))
            or (param_mode == "natural" and abs(last_p - p_e) <= 1e-12)
        ):
            break
        if remaining <= 0 or remesh_interval is None:
            break

        cur_tau_raw = scan_res.tangents[last_idx]
        cur_ds = max(float(scan_res.step_sizes[last_idx]), dp_min_val)

        remeshed = redistribute_mesh(
            last_orb,
            alpha=mesh_alpha,
            damping=mesh_damping,
            p=last_p,
            re_solve=True,
            tol=tol_val,
            max_iters=max_iters_val,
        )
        prev_tm = (cur_tau_raw, cur_mesh)
        if bool(remeshed.converged):
            cur_mesh = jnp.asarray(remeshed.mesh, dtype=jnp.float64)
            cur_X = problem.pack_vars(remeshed.u_mesh, remeshed.u_gauss, remeshed.period)
        else:
            cur_X = problem.pack_vars(last_orb.u_mesh, last_orb.u_gauss, last_orb.period)
        cur_p = jnp.asarray(last_p, dtype=jnp.float64)

    if compute_floquet and len(results) >= 2:
        check_types = ("LPC", "PD", "NS") if param_mode == "arclength" else ("PD", "NS")
        for k in range(1, len(results)):
            prev_tf, cur_tf = tf_history[k - 1], tf_history[k]
            detected = [
                btype
                for btype in check_types
                if jnp.isfinite(prev_tf[btype])
                and jnp.isfinite(cur_tf[btype])
                and (prev_tf[btype] * cur_tf[btype]) < 0.0
            ]
            if detected:
                results[k] = results[k]._replace(bifurcations=tuple(detected))

    return results


def redistribute_mesh(
    sol: PeriodicOrbitResult,
    alpha: float = 1.0,
    beta: float = 0.0,
    damping: float = 0.7,
    p: float | jax.Array = 0.0,
    re_solve: bool = True,
    tol: float = 1e-8,
    max_iters: int = 15,
    min_step_ratio: float = 1e-4,
) -> PeriodicOrbitResult:
    """Adaptively redistribute collocation mesh and re-solve orbit on the new mesh in pure JAX."""
    new_mesh = equidistribute_mesh(
        sol, alpha=alpha, beta=beta, damping=damping, min_step_ratio=min_step_ratio
    )
    u_mesh_new, u_gauss_new = interpolate_orbit_to_mesh(sol, new_mesh)

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
