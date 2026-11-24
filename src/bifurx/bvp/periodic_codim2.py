from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from bifurx.bvp.collocation import (
    CollocationProblem,
    PeriodicOrbitResult,
    periodic_orbit_arclength_weights,
)
from bifurx.bvp.condensation import (
    CondensedCollocationFactor,
    _condense_collocation_blocks,
    factor_condensed_collocation,
    solve_condensed_bordered,
)
from bifurx.bvp.floquet import monodromy_condensation_raw
from bifurx.homotopy import BorderedOperatorSolver, compute_tangent, track_arclength_scan
from bifurx.options import ContinuationOptions, resolve_continuation_options


class FixedPeriodContinuationResult(NamedTuple):
    """Pure-JAX result of 2-parameter fixed-period periodic orbit continuation.

    Tracks a curve `(u(t), p1, p2)` at fixed period `T` (isoperiodic / homoclinic
    approximation curve in AUTO-07P). All fields are JAX arrays with no host sync.
    """

    p1: jax.Array  # shape (n_steps + 1,)
    p2: jax.Array  # shape (n_steps + 1,)
    u_mesh: jax.Array  # shape (n_steps + 1, N, dim)
    u_gauss: jax.Array  # shape (n_steps + 1, N, m, dim)
    mesh: jax.Array  # shape (N + 1,)
    period: jax.Array  # scalar fixed period T
    tangents: jax.Array  # shape (n_steps + 1, dim_X + 1)
    converged: jax.Array  # shape (n_steps + 1,) bool
    iterations: jax.Array  # shape (n_steps + 1,) int32
    residual_norms: jax.Array  # shape (n_steps + 1,) float64


def _resolve_2p_bounds_and_stop(
    p1_bounds: tuple[float, float] | None,
    p2_bounds: tuple[float, float] | None,
) -> tuple[jax.Array, jax.Array, Callable[[jax.Array, jax.Array], jax.Array] | None]:
    """Resolve p1 bounds and optional p2 stop callback for 2-parameter continuation scans."""
    p1_min = (
        jnp.array(-jnp.inf, dtype=jnp.float64)
        if p1_bounds is None
        else jnp.asarray(p1_bounds[0], dtype=jnp.float64)
    )
    p1_max = (
        jnp.array(jnp.inf, dtype=jnp.float64)
        if p1_bounds is None
        else jnp.asarray(p1_bounds[1], dtype=jnp.float64)
    )
    if p2_bounds is not None:
        p2_lo = jnp.asarray(p2_bounds[0], dtype=jnp.float64)
        p2_hi = jnp.asarray(p2_bounds[1], dtype=jnp.float64)

        def stop_fn(state_vec: jax.Array, _p1: jax.Array) -> jax.Array:
            p2_v = jnp.squeeze(state_vec[-1])
            return (p2_v < p2_lo) | (p2_v > p2_hi)
    else:
        stop_fn = None
    return p1_min, p1_max, stop_fn


def _prepend_initial_scan_point(
    state0: jax.Array,
    p1_0: jax.Array,
    tau0: jax.Array,
    init_res_norm: jax.Array,
    scan_res: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Prepend initial point arrays to a `TrackScanResult`."""
    return (
        jnp.concatenate([state0[None, :], scan_res.u], axis=0),
        jnp.concatenate([jnp.atleast_1d(p1_0), scan_res.p], axis=0),
        jnp.concatenate([tau0[None, :], scan_res.tangents], axis=0),
        jnp.concatenate([jnp.array([True]), scan_res.converged], axis=0),
        jnp.concatenate([jnp.array([0], dtype=jnp.int32), scan_res.iterations], axis=0),
        jnp.concatenate([jnp.atleast_1d(init_res_norm), scan_res.residual_norms], axis=0),
    )


def _fixed_period_residual(
    problem: CollocationProblem,
    fn_2p: Callable[..., jax.Array],
    X_2p: jax.Array,
    p1: jax.Array,
    T_fixed: jax.Array,
    mesh: jax.Array,
    dot_u_ref: jax.Array,
    u_ref_gauss: jax.Array,
    kwargs: dict[str, Any],
) -> jax.Array:
    """Evaluate fixed-period collocation BVP residual for X_2p = [u_mesh, u_gauss, p2]."""
    u_mesh, u_gauss, p2 = problem.unpack_vars(X_2p)
    f_val = jax.vmap(jax.vmap(lambda u_: fn_2p(u_, p1, p2, **kwargs)))(u_gauss)
    return problem._eval_residual_from_rhs(
        u_mesh,
        u_gauss,
        T_fixed,
        f_val,
        mesh,
        u_ref_gauss=u_ref_gauss,
        dot_u_ref=dot_u_ref,
    )


def _make_fixed_period_bordered_solver(
    problem: CollocationProblem,
    fn_2p: Callable[..., jax.Array],
    T_fixed: jax.Array,
    mesh: jax.Array,
    dot_u_ref: jax.Array,
    kwargs: dict[str, Any],
) -> BorderedOperatorSolver:
    """Build O(N n^3) block-condensed bordered solver for fixed-period 2-parameter BVP."""

    def factor(
        X_2p: jax.Array, p1: jax.Array, c: jax.Array, d: jax.Array
    ) -> CondensedCollocationFactor:
        _, u_gauss, p2 = problem.unpack_vars(X_2p)
        Ju_all = jax.vmap(jax.vmap(jax.jacfwd(lambda u_: fn_2p(u_, p1, p2, **kwargs))))(u_gauss)
        Jp2_all = jax.vmap(
            jax.vmap(lambda u_: jax.jacfwd(lambda p2_: fn_2p(u_, p1, p2_, **kwargs))(p2))
        )(u_gauss)
        Jp1_all = jax.vmap(
            jax.vmap(lambda u_: jax.jacfwd(lambda p1_: fn_2p(u_, p1_, p2, **kwargs))(p1))
        )(u_gauss)
        return _condense_collocation_blocks(
            problem,
            T=T_fixed,
            mesh=mesh,
            dot_u_ref=dot_u_ref,
            Ju_all=Ju_all,
            q_scalar=-T_fixed * Jp2_all,
            q_param=-T_fixed * Jp1_all,
            c_X=c,
            d_p=d,
        )

    def solve(fac: CondensedCollocationFactor, f: jax.Array, g: jax.Array) -> Any:
        return solve_condensed_bordered(fac, f, g)

    return BorderedOperatorSolver(factor=factor, solve=solve)


def continue_fixed_period_2p(
    fn_2p: Callable[..., jax.Array],
    initial_orbit: PeriodicOrbitResult,
    p1_init: float | jax.Array,
    p2_init: float | jax.Array,
    *,
    p1_bounds: tuple[float, float] | None = None,
    p2_bounds: tuple[float, float] | None = None,
    ds: float | jax.Array | None = None,
    ds_min: float | jax.Array | None = None,
    ds_max: float | jax.Array | None = None,
    max_steps: int | None = None,
    direction: float | jax.Array | None = None,
    p1_weight: float = 1.0,
    p2_weight: float = 1.0,
    adaptive_step: bool | None = None,
    predictor: str | None = None,
    tol: float | None = None,
    max_newton_iters: int | None = None,
    options: ContinuationOptions | None = None,
    **kwargs: Any,
) -> FixedPeriodContinuationResult:
    """Continue a periodic orbit in two parameters `(p1, p2)` at fixed period `T` in pure JAX.

    Implements the AUTO-07P fixed-large-period continuation method for tracing isoperiodic
    and approximate homoclinic (HC) curves in two parameters. Uses de Boor-Swartz block
    condensation and `jax.lax.scan` with zero Python host synchronization.
    """
    opts = resolve_continuation_options(
        options,
        default_options=ContinuationOptions(
            ds=0.02, ds_min=1e-5, ds_max=0.1, max_steps=100, adaptive_step=True
        ),
        ds=ds,
        ds_min=ds_min,
        ds_max=ds_max,
        max_steps=max_steps,
        direction=direction,
        adaptive_step=adaptive_step,
        predictor=predictor,
        newton_tol=tol,
        newton_max_iters=max_newton_iters,
    )

    problem = initial_orbit.problem
    mesh = jnp.asarray(initial_orbit.mesh, dtype=jnp.float64)
    T_fixed = jnp.squeeze(jnp.asarray(initial_orbit.period, dtype=jnp.float64))
    p1_0 = jnp.squeeze(jnp.asarray(p1_init, dtype=jnp.float64))
    p2_0 = jnp.squeeze(jnp.asarray(p2_init, dtype=jnp.float64))
    X0 = problem.pack_vars(initial_orbit.u_mesh, initial_orbit.u_gauss, p2_0)

    chart_fn = problem.make_phase_chart_fn(mesh)

    def res_fn(X_2p: jax.Array, p1_val: jax.Array, chart: tuple[jax.Array, jax.Array]) -> jax.Array:
        dot_u_ref, u_ref_g = chart
        return _fixed_period_residual(
            problem, fn_2p, X_2p, p1_val, T_fixed, mesh, dot_u_ref, u_ref_g, kwargs
        )

    def solver_factory(chart: tuple[jax.Array, jax.Array]) -> BorderedOperatorSolver:
        dot_u_ref, _ = chart
        return _make_fixed_period_bordered_solver(problem, fn_2p, T_fixed, mesh, dot_u_ref, kwargs)

    w_arc = periodic_orbit_arclength_weights(
        problem, mesh, period_weight=p2_weight, param_weight=p1_weight
    )
    chart0 = chart_fn(X0, p1_0)
    tau0 = compute_tangent(
        u0=X0,
        p0=p1_0,
        direction=opts.direction,
        arclength_weights=w_arc,
        param_mode="arclength",
        bordered_solver=solver_factory(chart0),
    )

    p1_min, p1_max, stop_fn = _resolve_2p_bounds_and_stop(p1_bounds, p2_bounds)

    scan_res = track_arclength_scan(
        residual_fn=res_fn,
        u0=X0,
        p0=p1_0,
        tau0=tau0,
        ds=opts.ds,
        ds_min=opts.ds_min,
        ds_max=opts.ds_max,
        n_steps=int(opts.max_steps),
        p_min=p1_min,
        p_max=p1_max,
        adaptive_step=bool(opts.adaptive_step),
        predictor=opts.predictor,
        param_mode="arclength",
        arclength_weights=w_arc,
        chart_fn=chart_fn,
        stop_fn=stop_fn,
        tol=opts.newton.tol,
        step_tol=opts.newton.step_tol,
        max_newton_iters=opts.newton.max_iters,
        solver_method=opts.newton.solver_method,
        tracker_params=opts.tracker,
        norm_options=opts.norm,
        bordered_solver=solver_factory,
    )

    X_all, p1_all, tangents_all, conv_all, iters_all, res_all = _prepend_initial_scan_point(
        X0, p1_0, tau0, initial_orbit.residual_norm, scan_res
    )
    u_mesh_all, u_gauss_all, p2_all = jax.vmap(problem.unpack_vars)(X_all)

    return FixedPeriodContinuationResult(
        p1=p1_all,
        p2=p2_all,
        u_mesh=u_mesh_all,
        u_gauss=u_gauss_all,
        mesh=mesh,
        period=T_fixed,
        tangents=tangents_all,
        converged=conv_all,
        iterations=iters_all,
        residual_norms=res_all,
    )


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class PeriodicCodim2Result:
    """Pure-JAX PyTree result of 2-parameter periodic orbit bifurcation (LPC, PD, or NS)."""

    p1: jax.Array  # shape (n_steps + 1,)
    p2: jax.Array  # shape (n_steps + 1,)
    period: jax.Array  # shape (n_steps + 1,)
    u_mesh: jax.Array  # shape (n_steps + 1, N, dim)
    u_gauss: jax.Array  # shape (n_steps + 1, N, m, dim)
    mesh: jax.Array  # shape (N + 1,)
    tangents: jax.Array  # shape (n_steps + 1, dim_X + 2)
    converged: jax.Array  # shape (n_steps + 1,) bool
    iterations: jax.Array  # shape (n_steps + 1,) int32
    residual_norms: jax.Array  # shape (n_steps + 1,) float64
    bif_type: str  # "LPC", "PD", or "NS"

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], tuple[str]]:
        children = (
            self.p1,
            self.p2,
            self.period,
            self.u_mesh,
            self.u_gauss,
            self.mesh,
            self.tangents,
            self.converged,
            self.iterations,
            self.residual_norms,
        )
        return children, (self.bif_type,)

    @classmethod
    def tree_unflatten(
        cls, aux_data: tuple[str], children: tuple[jax.Array, ...]
    ) -> PeriodicCodim2Result:
        return cls(*children, bif_type=aux_data[0])


def _exterior_square(A: jax.Array) -> jax.Array:
    """Compute 2nd exterior power Lambda^2(A) with eigenvalues {mu_i * mu_j : i < j}."""
    n = A.shape[0]
    idx_i, idx_j = jnp.triu_indices(n, k=1)
    return A[idx_i[:, None], idx_i[None, :]] * A[idx_j[:, None], idx_j[None, :]] - (
        A[idx_i[:, None], idx_j[None, :]] * A[idx_j[:, None], idx_i[None, :]]
    )


def _bordered_singularity_scalar(A: jax.Array, b: jax.Array, c: jax.Array) -> jax.Array:
    """Minimally-augmented scalar g(A) = 0 iff A is singular (Govaerts 2000)."""
    k = A.shape[0]
    M_aug = jnp.block([[A, b[:, None]], [c[None, :], jnp.zeros((1, 1), dtype=jnp.float64)]])
    rhs = jnp.zeros(k + 1, dtype=jnp.float64).at[-1].set(1.0)
    return jnp.linalg.solve(M_aug, rhs)[-1]


def _periodic_bif_matrix(
    base_problem: CollocationProblem,
    fn_2p: Callable[..., jax.Array],
    X: jax.Array,
    p1: jax.Array,
    p2: jax.Array,
    mesh: jax.Array,
    dot_u_ref: jax.Array,
    bif_type: str,
    kwargs: dict[str, Any],
) -> jax.Array:
    """Build the reduced matrix whose singularity defines LPC, PD, or NS in pure JAX."""
    prob_p2 = CollocationProblem(
        fn=lambda u_, p1_: fn_2p(u_, p1_, p2, **kwargs),
        dim=base_problem.dim,
        num_intervals=base_problem.num_intervals,
        num_gauss_points=base_problem.num_gauss_points,
        period_shift=base_problem.period_shift,
    )
    if bif_type == "LPC":
        fac = factor_condensed_collocation(prob_p2, X, p1, mesh, dot_u_ref)
        return fac.cyclic.M_red

    _, u_g, T_val = base_problem.unpack_vars(X)
    M_mono = monodromy_condensation_raw(prob_p2, mesh, u_g, T_val, p=p1)
    if bif_type == "PD":
        return M_mono + jnp.eye(base_problem.dim, dtype=jnp.float64)

    ext2 = _exterior_square(M_mono)
    return ext2 - jnp.eye(ext2.shape[0], dtype=jnp.float64)


def continue_periodic_bifurcation_2p(
    fn_2p: Callable[..., jax.Array],
    bif_type: str,
    initial_orbit: PeriodicOrbitResult,
    p1_init: float | jax.Array,
    p2_init: float | jax.Array,
    *,
    p1_bounds: tuple[float, float] | None = None,
    p2_bounds: tuple[float, float] | None = None,
    ds: float | jax.Array | None = None,
    ds_min: float | jax.Array | None = None,
    ds_max: float | jax.Array | None = None,
    max_steps: int | None = None,
    direction: float | jax.Array | None = None,
    adaptive_step: bool | None = None,
    predictor: str | None = None,
    tol: float | None = None,
    max_newton_iters: int | None = None,
    options: ContinuationOptions | None = None,
    **kwargs: Any,
) -> PeriodicCodim2Result:
    """Track a 2-parameter periodic orbit bifurcation curve (LPC, PD, or NS) in pure JAX.

    Uses minimally-augmented scalar defining equations (Govaerts 2000) built directly on
    the condensed collocation operator (`M_red` for LPC) and condensed monodromy matrix
    (`M + I` for PD, `Lambda^2(M) - I` for NS). Fully pure JAX and `jax.jit`-compatible.
    """
    btype = bif_type.upper()
    if btype not in ("LPC", "PD", "NS"):
        msg = f"Unknown bif_type: {bif_type}. Must be 'LPC', 'PD', or 'NS'."
        raise ValueError(msg)
    problem = initial_orbit.problem
    if btype == "NS" and problem.dim < 3:
        msg = f"NS bifurcation requires dim >= 3, got dim={problem.dim}."
        raise ValueError(msg)

    opts = resolve_continuation_options(
        options,
        default_options=ContinuationOptions(
            ds=0.04, ds_min=1e-5, ds_max=0.15, max_steps=60, adaptive_step=True
        ),
        ds=ds,
        ds_min=ds_min,
        ds_max=ds_max,
        max_steps=max_steps,
        direction=direction,
        adaptive_step=adaptive_step,
        predictor=predictor,
        newton_tol=tol,
        newton_max_iters=max_newton_iters,
    )

    mesh = jnp.asarray(initial_orbit.mesh, dtype=jnp.float64)
    p1_0 = jnp.squeeze(jnp.asarray(p1_init, dtype=jnp.float64))
    p2_0 = jnp.squeeze(jnp.asarray(p2_init, dtype=jnp.float64))
    X0 = problem.pack_vars(initial_orbit.u_mesh, initial_orbit.u_gauss, initial_orbit.period)
    Z0 = jnp.concatenate([X0, jnp.atleast_1d(p2_0)])

    _base_chart = problem.make_phase_chart_fn(mesh)

    def chart_fn(Z_ref: jax.Array, p1_v: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _base_chart(Z_ref[:-1], p1_v)

    chart0 = chart_fn(Z0, p1_0)
    A0 = _periodic_bif_matrix(problem, fn_2p, X0, p1_0, p2_0, mesh, chart0[0], btype, kwargs)
    u_svd0, _, vh_svd0 = jnp.linalg.svd(A0)
    b_border = u_svd0[:, -1]
    c_border = vh_svd0[-1, :]

    def res_fn(Z: jax.Array, p1_val: jax.Array, chart: tuple[jax.Array, jax.Array]) -> jax.Array:
        X_v, p2_v = Z[:-1], Z[-1]
        dot_u_ref, u_ref_g = chart
        prob_p2 = CollocationProblem(
            fn=lambda u_, p1_: fn_2p(u_, p1_, p2_v, **kwargs),
            dim=problem.dim,
            num_intervals=problem.num_intervals,
            num_gauss_points=problem.num_gauss_points,
            period_shift=problem.period_shift,
        )
        res_bvp = prob_p2.residual(X_v, p1_val, mesh, u_ref_gauss=u_ref_g, dot_u_ref=dot_u_ref)
        A_cur = _periodic_bif_matrix(
            problem, fn_2p, X_v, p1_val, p2_v, mesh, dot_u_ref, btype, kwargs
        )
        g_bif = _bordered_singularity_scalar(A_cur, b_border, c_border)
        return jnp.concatenate([res_bvp, jnp.atleast_1d(g_bif)])

    jac_u_fn = jax.jacfwd(res_fn, argnums=0)
    jac_p_fn = jax.jacfwd(res_fn, argnums=1)

    tau0 = compute_tangent(
        jac_u_fn=lambda z_, p_: jac_u_fn(z_, p_, chart0),
        jac_p_fn=lambda z_, p_: jac_p_fn(z_, p_, chart0),
        u0=Z0,
        p0=p1_0,
        direction=opts.direction,
        param_mode="arclength",
    )

    p1_min, p1_max, stop_fn = _resolve_2p_bounds_and_stop(p1_bounds, p2_bounds)

    scan_res = track_arclength_scan(
        residual_fn=res_fn,
        jac_u_fn=jac_u_fn,
        jac_p_fn=jac_p_fn,
        u0=Z0,
        p0=p1_0,
        tau0=tau0,
        ds=opts.ds,
        ds_min=opts.ds_min,
        ds_max=opts.ds_max,
        n_steps=int(opts.max_steps),
        p_min=p1_min,
        p_max=p1_max,
        adaptive_step=bool(opts.adaptive_step),
        predictor=opts.predictor,
        param_mode="arclength",
        chart_fn=chart_fn,
        stop_fn=stop_fn,
        tol=opts.newton.tol,
        step_tol=opts.newton.step_tol,
        max_newton_iters=opts.newton.max_iters,
        solver_method=opts.newton.solver_method,
        tracker_params=opts.tracker,
        norm_options=opts.norm,
    )

    Z_all, p1_all, tangents_all, conv_all, iters_all, res_all = _prepend_initial_scan_point(
        Z0, p1_0, tau0, initial_orbit.residual_norm, scan_res
    )
    u_mesh_all, u_gauss_all, T_all = jax.vmap(problem.unpack_vars)(Z_all[:, :-1])
    p2_all = Z_all[:, -1]

    return PeriodicCodim2Result(
        p1=p1_all,
        p2=p2_all,
        period=T_all,
        u_mesh=u_mesh_all,
        u_gauss=u_gauss_all,
        mesh=mesh,
        tangents=tangents_all,
        converged=conv_all,
        iterations=iters_all,
        residual_norms=res_all,
        bif_type=btype,
    )
