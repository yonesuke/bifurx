from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.bifurcations.test_funcs import extract_leading_hopf_pair
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
        t_arr = jnp.atleast_1d(jnp.asarray(t, dtype=jnp.float64)) % 1.0

        k = jnp.clip(jnp.searchsorted(mesh, t_arr, side="right") - 1, 0, N - 1)
        hk = mesh[k + 1] - mesh[k]
        xi = jnp.clip((t_arr - mesh[k]) / (hk + 1e-30), 0.0, 1.0)

        U_k = jnp.concatenate([u_mesh[k][:, None, :], u_gauss[k]], axis=1)
        L_vals = _eval_lagrange_basis(xi, s_nodes)
        u_eval = jnp.einsum("tj,tjn->tn", L_vals, U_k)
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
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        fn: Callable[..., jax.Array],
        dim: int,
        num_intervals: int | None = None,
        num_gauss_points: int | None = None,
        options: CollocationOptions | None = None,
        **kwargs: Any,
    ) -> None:
        base_opts = CollocationOptions() if options is None else options
        n_int = base_opts.num_intervals if num_intervals is None else int(num_intervals)
        n_gauss = base_opts.num_gauss_points if num_gauss_points is None else int(num_gauss_points)

        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "num_intervals", n_int)
        object.__setattr__(self, "num_gauss_points", n_gauss)
        object.__setattr__(self, "kwargs", kwargs)

        D_arr, C_arr, rho_arr, weights_arr = collocation_matrices(self.num_gauss_points)
        object.__setattr__(self, "D", D_arr)
        object.__setattr__(self, "C", C_arr)
        object.__setattr__(self, "rho", rho_arr)
        object.__setattr__(self, "weights", weights_arr)

    def tree_flatten(
        self,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array], tuple[Any, ...]]:
        """Flatten into JAX pytree leaves ``(D, C, rho, weights)`` and static auxiliary data."""
        children = (self.D, self.C, self.rho, self.weights)
        aux = (self.fn, self.dim, self.num_intervals, self.num_gauss_points, self.kwargs)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls,
        aux: tuple[Any, ...],
        children: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> CollocationProblem:
        """Rebuild a problem from JAX pytree leaves without re-running ``__init__``."""
        fn, dim, num_intervals, num_gauss_points, kwargs = aux
        D, C, rho, weights = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "fn", fn)
        object.__setattr__(obj, "dim", dim)
        object.__setattr__(obj, "num_intervals", num_intervals)
        object.__setattr__(obj, "num_gauss_points", num_gauss_points)
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
    ) -> jax.Array:
        """Evaluate collocation BVP residual vector of dimension N*(m+1)*dim + 1."""
        u_mesh, u_gauss, T = self.unpack_vars(X)
        h = mesh[1:] - mesh[:-1]

        if u_ref_mesh is None:
            u_ref_mesh = u_mesh
        if u_ref_gauss is None:
            u_ref_gauss = u_gauss

        p_val = jnp.squeeze(p) if jnp.ndim(p) > 0 else p
        U = jnp.concatenate([u_mesh[:, None, :], u_gauss], axis=1)

        du_dt = jnp.einsum("mj,Njd->Nmd", self.D, U) / h[:, None, None]
        f_val = jax.vmap(jax.vmap(lambda u: self.fn(u, p_val, **self.kwargs)))(u_gauss)
        res_coll = (du_dt - T * f_val).ravel()

        u_next = jnp.roll(u_mesh, -1, axis=0)
        u_end_approx = jnp.einsum("j,Njd->Nd", self.C, U)
        res_cont = (u_next - u_end_approx).ravel()

        U_ref = jnp.concatenate([u_ref_mesh[:, None, :], u_ref_gauss], axis=1)
        dot_u_ref = jnp.einsum("mj,Njd->Nmd", self.D, U_ref) / h[:, None, None]
        inner_prod = jnp.sum(dot_u_ref * u_gauss, axis=-1)
        phase_val = jnp.sum(h[:, None] * self.weights[None, :] * inner_prod)

        return jnp.concatenate([res_coll, res_cont, jnp.atleast_1d(phase_val)])


def initialize_from_hopf(
    problem: CollocationProblem,
    u_hopf: ArrayLike,
    p_hopf: float | jax.Array,
    eps: float | jax.Array | None = None,
    p_init: float | jax.Array | None = None,
    direction: float = -1.0,
    ds: float | None = None,
    options: CollocationOptions | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Construct an initial periodic orbit guess near a Hopf bifurcation in pure JAX."""
    base_eps = options.eps if options is not None else CollocationOptions().eps
    eps_val = jnp.squeeze(jnp.asarray(base_eps if eps is None else eps, dtype=jnp.float64))

    u_h = jnp.asarray(u_hopf, dtype=jnp.float64)
    p_h = jnp.squeeze(jnp.asarray(p_hopf, dtype=jnp.float64))

    Ju = jax.jacobian(lambda u_: problem.fn(u_, p_h, **problem.kwargs))(u_h)
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

    if p_init is not None:
        p_init_arr = jnp.squeeze(jnp.asarray(p_init, dtype=jnp.float64))
    else:
        step = eps_val if ds is None else jnp.squeeze(jnp.asarray(ds, dtype=jnp.float64))
        p_init_arr = p_h + direction * step

    return mesh, u_mesh, u_gauss, T0, p_init_arr


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

    def eval_res(x_vec: jax.Array) -> jax.Array:
        return problem.residual(
            x_vec, p_arr, mesh_arr, u_ref_mesh=u_ref_mesh_arr, u_ref_gauss=u_ref_gauss_arr
        )

    jac_fn = jax.jacobian(eval_res)
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
        J = jac_fn(X)

        dX_solve = -jnp.linalg.solve(J, res)
        dX = jax.lax.cond(
            jnp.all(jnp.isfinite(dX_solve)),
            lambda: dX_solve,
            lambda: -jnp.linalg.lstsq(J, res, rcond=1e-12)[0],
        )

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
    non_trivial: bool = True,
    tol: float | None = None,
    max_iters: int | None = None,
    verbose: bool = False,
    options: ContinuationOptions | CollocationOptions | None = None,
    collocation_options: CollocationOptions | None = None,
) -> list[PeriodicOrbitResult]:
    """Continue a family of periodic orbits along parameter p (#15)."""
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

    p_s, p_e = float(p_start), float(p_end)
    direction = 1.0 if p_e >= p_s else -1.0

    if initial_orbit.p is None:
        initial_orbit = initial_orbit._replace(p=jnp.asarray(p_s, dtype=jnp.float64))

    results: list[PeriodicOrbitResult] = [initial_orbit]
    current_orbit = initial_orbit
    p_cur = p_s

    step = 0
    while step < max_steps_val and (direction * (p_e - p_cur) > 1e-9):
        p_target = p_cur + direction * cur_dp
        if (direction > 0 and p_target > p_e) or (direction < 0 and p_target < p_e):
            p_target = p_e

        sol = solve_periodic_orbit(
            problem,
            u_mesh_init=current_orbit.u_mesh,
            T_init=current_orbit.period,
            p=p_target,
            mesh=current_orbit.mesh,
            u_gauss_init=current_orbit.u_gauss,
            u_ref_mesh=current_orbit.u_mesh,
            u_ref_gauss=current_orbit.u_gauss,
            tol=tol_val,
            max_iters=max_iters_val,
            non_trivial=non_trivial,
        )

        if bool(sol.converged):
            p_cur = p_target
            current_orbit = sol
            results.append(sol)
            if verbose:
                print(
                    f"Step {step + 1}: p={p_cur:.5f}, T={float(sol.T):.5f}, "
                    f"amp={float(sol.amplitude):.5f}, iters={int(sol.iterations)}"
                )
            if int(sol.iterations) <= 4:
                cur_dp = min(cur_dp * 1.25, dp_max_val)
            step += 1
        else:
            cur_dp *= 0.5
            if cur_dp < dp_min_val:
                if verbose:
                    print(
                        f"Step {step + 1}: step size {cur_dp} < dp_min ({dp_min_val}). Terminating."
                    )
                break

    return results
