from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

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
from bifurx.homotopy import BorderedOperatorSolver, compute_tangent, track_arclength_scan
from bifurx.homotopy.corrector import correct_arclength_step
from bifurx.options import CollocationOptions, ContinuationOptions


def _gauss_legendre_nodes(m: int) -> tuple[jax.Array, jax.Array]:
    """Compute normalized Gauss-Legendre quadrature nodes and weights on [0, 1]."""
    if m == 1:
        return jnp.array([0.5], dtype=jnp.float64), jnp.array([1.0], dtype=jnp.float64)

    k = jnp.arange(1, m, dtype=jnp.float64)
    beta = k / jnp.sqrt(4.0 * k**2 - 1.0)
    nodes, vecs = jnp.linalg.eigh(jnp.diag(beta, 1) + jnp.diag(beta, -1))
    return 0.5 * (nodes + 1.0), vecs[0, :] ** 2


def _eval_lagrange_basis(xi: float | jax.Array, nodes: ArrayLike) -> jax.Array:
    """Evaluate Lagrange basis polynomials L_j(xi) for nodes s in pure vectorized JAX."""
    nodes_arr = jnp.asarray(nodes, dtype=jnp.float64)
    n_nodes = nodes_arr.shape[0]
    xi_arr = jnp.asarray(xi, dtype=jnp.float64)
    is_scalar = xi_arr.ndim == 0
    xi_1d = jnp.atleast_1d(xi_arr)

    keep = ~jnp.eye(n_nodes, dtype=bool)
    numer = xi_1d[:, None, None] - nodes_arr[None, None, :]
    denom = jnp.where(keep, nodes_arr[None, :, None] - nodes_arr[None, None, :], 1.0)
    L_vals = jnp.prod(jnp.where(keep, numer / denom, 1.0), axis=-1)
    return L_vals[0] if is_scalar else L_vals


def collocation_matrices(m: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute Lagrange differentiation and boundary extrapolation matrices for Gauss collocation."""
    rho, weights = _gauss_legendre_nodes(m)
    s = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), rho])
    D = jax.vmap(jax.jacfwd(lambda xi: _eval_lagrange_basis(xi, s)))(rho)
    C = _eval_lagrange_basis(jnp.array(1.0, dtype=jnp.float64), s)
    return D, C, rho, weights


def _orbit_amplitude(u_mesh: jax.Array) -> jax.Array:
    """Compute root-mean-square amplitude of an orbit around its mean state."""
    u_mean = jnp.mean(u_mesh, axis=0, keepdims=True)
    return jnp.sqrt(jnp.mean(jnp.sum((u_mesh - u_mean) ** 2, axis=-1)))


class PeriodicOrbitResult(NamedTuple):
    """Result of solving a periodic orbit via pure-JAX collocation BVP."""

    mesh: jax.Array
    u_mesh: jax.Array  # shape (N, dim)
    u_gauss: jax.Array  # shape (N, m, dim)
    period: jax.Array
    converged: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array
    problem: CollocationProblem
    p: jax.Array | None = None
    floquet_multipliers: jax.Array | None = None
    is_stable: jax.Array | None = None
    bifurcations: tuple[str, ...] = ()

    @property
    def T(self) -> jax.Array:
        """Alias for period."""
        return self.period

    @property
    def amplitude(self) -> jax.Array:
        """Root-mean-square amplitude of orbit around its mean state (#14)."""
        return _orbit_amplitude(self.u_mesh)

    def evaluate(self, t: float | jax.Array) -> jax.Array:
        """Evaluate continuous periodic orbit state u(t) at any normalized phase t in [0, 1]."""
        mesh = jnp.asarray(self.mesh, dtype=jnp.float64)
        u_mesh = jnp.asarray(self.u_mesh, dtype=jnp.float64)
        u_gauss = jnp.asarray(self.u_gauss, dtype=jnp.float64)
        N = self.problem.num_intervals
        s_nodes = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.float64), jnp.asarray(self.problem.rho, dtype=jnp.float64)]
        )

        is_scalar = jnp.ndim(t) == 0
        t_raw = jnp.atleast_1d(jnp.asarray(t, dtype=jnp.float64))
        n_wraps = jnp.floor(t_raw)
        t_arr = t_raw - n_wraps

        k = jnp.clip(jnp.searchsorted(mesh, t_arr, side="right") - 1, 0, N - 1)
        hk = mesh[k + 1] - mesh[k]
        xi = jnp.clip((t_arr - mesh[k]) / (hk + 1e-30), 0.0, 1.0)

        U_k = jnp.concatenate([u_mesh[k][:, None, :], u_gauss[k]], axis=1)
        L_vals = _eval_lagrange_basis(xi, s_nodes)
        u_eval = jnp.einsum("tj,tjn->tn", L_vals, U_k)
        if self.problem.period_shift is not None:
            u_eval = u_eval + n_wraps[:, None] * self.problem.period_shift[None, :]
        return u_eval[0] if is_scalar else u_eval


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class CollocationProblem:
    """PyTree dataclass representing the Gauss-Legendre collocation BVP for periodic orbits."""

    fn: Callable[..., jax.Array]
    dim: int
    D: jax.Array
    C: jax.Array
    rho: jax.Array
    weights: jax.Array
    num_intervals: int = 25
    num_gauss_points: int = 4
    period_shift: jax.Array | None = None
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        fn: Callable[..., jax.Array],
        dim: int,
        num_intervals: int | None = None,
        num_gauss_points: int | None = None,
        options: CollocationOptions | None = None,
        period_shift: ArrayLike | Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        base_opts = CollocationOptions() if options is None else options
        n_int = base_opts.num_intervals if num_intervals is None else int(num_intervals)
        n_gauss = base_opts.num_gauss_points if num_gauss_points is None else int(num_gauss_points)

        shift_arr: jax.Array | None = None
        if period_shift is not None:
            shift_arr = jnp.asarray(period_shift, dtype=jnp.float64).ravel()
            if shift_arr.shape != (dim,):
                msg = f"period_shift must have shape ({dim},), got {shift_arr.shape}."
                raise ValueError(msg)

        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "num_intervals", n_int)
        object.__setattr__(self, "num_gauss_points", n_gauss)
        object.__setattr__(self, "period_shift", shift_arr)
        object.__setattr__(self, "kwargs", kwargs)

        D_arr, C_arr, rho_arr, weights_arr = collocation_matrices(self.num_gauss_points)
        object.__setattr__(self, "D", D_arr)
        object.__setattr__(self, "C", C_arr)
        object.__setattr__(self, "rho", rho_arr)
        object.__setattr__(self, "weights", weights_arr)

    def tree_flatten(
        self,
    ) -> tuple[
        tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None], tuple[Any, ...]
    ]:
        """Flatten into JAX pytree leaves ``(D, C, rho, weights, period_shift)`` and static aux data."""
        children = (self.D, self.C, self.rho, self.weights, self.period_shift)
        aux = (self.fn, self.dim, self.num_intervals, self.num_gauss_points, self.kwargs)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[Any, ...],
        children: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array | None],
    ) -> CollocationProblem:
        """Rebuild a problem from JAX pytree leaves without re-running ``__init__``."""
        fn, dim, num_intervals, num_gauss_points, kwargs = aux
        D, C, rho, weights, period_shift = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "fn", fn)
        object.__setattr__(obj, "dim", dim)
        object.__setattr__(obj, "num_intervals", num_intervals)
        object.__setattr__(obj, "num_gauss_points", num_gauss_points)
        object.__setattr__(obj, "period_shift", period_shift)
        object.__setattr__(obj, "kwargs", kwargs)
        object.__setattr__(obj, "D", D)
        object.__setattr__(obj, "C", C)
        object.__setattr__(obj, "rho", rho)
        object.__setattr__(obj, "weights", weights)
        return obj

    def pack_vars(
        self, u_mesh: jax.Array, u_gauss: jax.Array, period: float | jax.Array
    ) -> jax.Array:
        """Pack (u_mesh, u_gauss, period) into a single 1D flat vector."""
        return jnp.concatenate(
            [
                jnp.asarray(u_mesh, dtype=jnp.float64).ravel(),
                jnp.asarray(u_gauss, dtype=jnp.float64).ravel(),
                jnp.atleast_1d(jnp.asarray(period, dtype=jnp.float64)),
            ]
        )

    def unpack_vars(self, X: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Unpack 1D flat vector into (u_mesh, u_gauss, period)."""
        N, m, n = self.num_intervals, self.num_gauss_points, self.dim
        n_mesh = N * n
        n_gauss = N * m * n
        return (
            X[:n_mesh].reshape((N, n)),
            X[n_mesh : n_mesh + n_gauss].reshape((N, m, n)),
            jnp.squeeze(X[-1]),
        )

    def residual(
        self,
        X: jax.Array,
        p: float | jax.Array,
        mesh: jax.Array,
        u_ref_mesh: jax.Array | None = None,
        u_ref_gauss: jax.Array | None = None,
        dot_u_ref: jax.Array | None = None,
    ) -> jax.Array:
        """Evaluate collocation BVP residual vector of dimension N*(m+1)*dim + 1."""
        u_mesh, u_gauss, T = self.unpack_vars(X)
        h = mesh[1:] - mesh[:-1]

        p_val = jnp.squeeze(p) if jnp.ndim(p) > 0 else p
        U = jnp.concatenate([u_mesh[:, None, :], u_gauss], axis=1)

        du_dt = jnp.einsum("mj,Njd->Nmd", self.D, U) / h[:, None, None]
        f_val = jax.vmap(jax.vmap(lambda u: self.fn(u, p_val, **self.kwargs)))(u_gauss)
        res_coll = (du_dt - T * f_val).ravel()

        u_next = jnp.roll(u_mesh, -1, axis=0)
        if self.period_shift is not None:
            u_next = u_next.at[-1].add(self.period_shift)
        u_end_approx = jnp.einsum("j,Njd->Nd", self.C, U)
        res_cont = (u_next - u_end_approx).ravel()

        if dot_u_ref is None:
            ref_mesh = u_mesh if u_ref_mesh is None else u_ref_mesh
            ref_gauss = u_gauss if u_ref_gauss is None else u_ref_gauss
            U_ref = jnp.concatenate([ref_mesh[:, None, :], ref_gauss], axis=1)
            dot_u_ref_val = jnp.einsum("mj,Njd->Nmd", self.D, U_ref) / h[:, None, None]
        else:
            dot_u_ref_val = dot_u_ref
        u_phase = u_gauss - u_ref_gauss if u_ref_gauss is not None else u_gauss
        inner_prod = jnp.sum(dot_u_ref_val * u_phase, axis=-1)
        phase_val = jnp.sum(h[:, None] * self.weights[None, :] * inner_prod)

        return jnp.concatenate([res_coll, res_cont, jnp.atleast_1d(phase_val)])


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
    base_eps = options.eps if options is not None else CollocationOptions().eps
    eps_val = jnp.squeeze(jnp.asarray(base_eps if eps is None else eps, dtype=jnp.float64))

    u_h = jnp.asarray(u_hopf, dtype=jnp.float64)
    p_h = jnp.squeeze(jnp.asarray(p_hopf, dtype=jnp.float64))

    jac_u_fn = jax.jacobian(lambda u_, p_: problem.fn(u_, p_, **problem.kwargs), argnums=0)
    Ju = jac_u_fn(u_h, p_h)
    eigs, vecs = jnp.linalg.eig(Ju)

    _, leading_real, omega = extract_leading_hopf_pair(Ju, eigs=eigs)
    target_eig = leading_real + 1j * omega
    T0 = 2.0 * jnp.pi / omega
    vec = vecs[:, jnp.argmin(jnp.abs(eigs - target_eig))]

    complex_mask = jnp.abs(jnp.imag(eigs)) > 1e-4 * (1.0 + jnp.abs(eigs))
    crit_mask = complex_mask & (jnp.abs(jnp.real(eigs)) < 1e-2)
    crit_freqs = jnp.where(crit_mask, jnp.abs(jnp.imag(eigs)), -1.0)
    matches = (jnp.abs(crit_freqs - omega) < 1e-3) & (crit_freqs > 0)
    if bool(jnp.sum(matches) > 2):
        warnings.warn(
            "Degenerate Hopf point detected: critical imaginary eigenvalue has algebraic "
            "multiplicity >= 2. A single eigenvector seed may collapse onto the trivial "
            "equilibrium. Consider increasing eps, setting non_trivial=True, or providing a custom initial guess.",
            UserWarning,
            stacklevel=2,
        )

    v_r, v_i = jnp.real(vec), jnp.imag(vec)
    norm_v = jnp.sqrt(jnp.dot(v_r, v_r) + jnp.dot(v_i, v_i))
    v_r, v_i = v_r / norm_v, v_i / norm_v

    N = problem.num_intervals
    mesh = jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float64)
    rho = jnp.asarray(problem.rho, dtype=jnp.float64)

    t_mesh = mesh[:-1]
    theta_mesh = 2.0 * jnp.pi * t_mesh
    u_mesh = u_h[None, :] + eps_val * (
        jnp.cos(theta_mesh)[:, None] * v_r[None, :] - jnp.sin(theta_mesh)[:, None] * v_i[None, :]
    )

    hk = mesh[1:] - mesh[:-1]
    t_gauss = t_mesh[:, None] + hk[:, None] * rho[None, :]
    theta_gauss = 2.0 * jnp.pi * t_gauss
    u_gauss = u_h[None, None, :] + eps_val * (
        jnp.cos(theta_gauss)[:, :, None] * v_r[None, None, :]
        - jnp.sin(theta_gauss)[:, :, None] * v_i[None, None, :]
    )

    if isinstance(direction, str):
        if direction != "auto":
            msg = f"Unknown direction: {direction}. Pass a float or 'auto'."
            raise ValueError(msg)
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
    base_opts = CollocationOptions() if options is None else options
    tol_val = base_opts.tol if tol is None else float(tol)
    max_iters_val = base_opts.max_iters if max_iters is None else int(max_iters)
    min_amp_val = base_opts.min_amplitude if min_amplitude is None else float(min_amplitude)
    t_w = float(base_opts.period_weight if period_weight is None else period_weight)
    p_w = float(param_weight)
    ds_val = jnp.squeeze(jnp.asarray(ds, dtype=jnp.float64))

    u_h = jnp.asarray(u_hopf, dtype=jnp.float64)
    p_h = jnp.squeeze(jnp.asarray(p_hopf, dtype=jnp.float64))

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

    N, m, n = problem.num_intervals, problem.num_gauss_points, problem.dim
    mesh = jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float64)
    rho = jnp.asarray(problem.rho, dtype=jnp.float64)
    hk = mesh[1:] - mesh[:-1]

    t_mesh = mesh[:-1]
    theta_mesh = 2.0 * jnp.pi * t_mesh
    du_mesh = (
        jnp.cos(theta_mesh)[:, None] * v_r[None, :] - jnp.sin(theta_mesh)[:, None] * v_i[None, :]
    )

    t_gauss = t_mesh[:, None] + hk[:, None] * rho[None, :]
    theta_gauss = 2.0 * jnp.pi * t_gauss
    du_gauss = (
        jnp.cos(theta_gauss)[:, :, None] * v_r[None, None, :]
        - jnp.sin(theta_gauss)[:, :, None] * v_i[None, None, :]
    )
    dot_u_ref = (
        -2.0 * jnp.pi * jnp.sin(theta_gauss)[:, :, None] * v_r[None, None, :]
        - 2.0 * jnp.pi * jnp.cos(theta_gauss)[:, :, None] * v_i[None, None, :]
    )

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
        rel_tol=min(tol_val, 1e-10),
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
    base_opts = CollocationOptions() if options is None else options
    tol_val = base_opts.tol if tol is None else tol
    max_iters_val = base_opts.max_iters if max_iters is None else max_iters
    min_amp_val = base_opts.min_amplitude if min_amplitude is None else min_amplitude

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
    h_arr = mesh_arr[1:] - mesh_arr[:-1]
    U_ref = jnp.concatenate([u_ref_mesh_arr[:, None, :], u_ref_gauss_arr], axis=1)
    dot_u_ref = jnp.einsum("mj,Njd->Nmd", problem.D, U_ref) / h_arr[:, None, None]

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
    period_weight: float = 1.0,
    param_weight: float = 1.0,
) -> jax.Array:
    """Build exact Gauss-Legendre L2[0, 1] arclength weights for `[u_mesh, u_gauss, T, p]`."""
    if period_weight < 0.0:
        msg = f"period_weight must be non-negative, got {period_weight}."
        raise ValueError(msg)
    if param_weight < 0.0:
        msg = f"param_weight must be non-negative, got {param_weight}."
        raise ValueError(msg)
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    h_arr = mesh_arr[1:] - mesh_arr[:-1]
    N, m, n = problem.num_intervals, problem.num_gauss_points, problem.dim
    w_mesh = jnp.zeros(N * n, dtype=jnp.float64)
    w_gauss = jnp.broadcast_to(
        (h_arr[:, None] * problem.weights[None, :])[:, :, None], (N, m, n)
    ).ravel()
    return jnp.concatenate(
        [
            w_mesh,
            w_gauss,
            jnp.array([float(period_weight) ** 2, float(param_weight) ** 2], dtype=jnp.float64),
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
    """Continue a family of periodic orbits along parameter p via `bifurx.homotopy` (#15)."""
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
    rel_tol_val = float(cont_opts.newton.rel_tol if has_cont else min(tol_val, 1e-10))
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
        _ = periodic_orbit_arclength_weights(
            problem, init_mesh, period_weight=t_w, param_weight=p_w
        )

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

        def chart_fn(X_ref: jax.Array, _p: jax.Array) -> tuple[jax.Array, jax.Array]:
            u_ref_m, u_ref_g, _ = problem.unpack_vars(X_ref)
            U_ref = jnp.concatenate([u_ref_m[:, None, :], u_ref_g], axis=1)
            dot_u_ref = jnp.einsum("mj,Njd->Nmd", problem.D, U_ref) / h_seg[:, None, None]
            return dot_u_ref, u_ref_g

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
            rel_tol=rel_tol_val,
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
        from bifurx.bvp.mesh import redistribute_mesh

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
