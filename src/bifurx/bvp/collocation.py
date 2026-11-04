from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def gauss_legendre_nodes(m: int) -> tuple[jax.Array, jax.Array]:
    """Compute normalized Gauss-Legendre quadrature nodes and weights on [0, 1].

    Uses the Golub-Welsch algorithm (eigen-decomposition of the Legendre Jacobi matrix)
    in pure JAX.

    Parameters
    ----------
    m : int
        Number of Gauss collocation points per mesh interval.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (nodes, weights) where nodes are in (0, 1) and sum(weights) == 1.0.
    """
    if m == 1:
        return jnp.array([0.5], dtype=jnp.float64), jnp.array([1.0], dtype=jnp.float64)

    k = jnp.arange(1, m, dtype=jnp.float64)
    beta = k / jnp.sqrt(4.0 * k**2 - 1.0)
    jacobi = jnp.diag(beta, 1) + jnp.diag(beta, -1)
    nodes, vecs = jnp.linalg.eigh(jacobi)
    weights_raw = 2.0 * vecs[0, :] ** 2

    rho = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights_raw
    return rho, weights


def eval_lagrange_basis(xi: float | jax.Array, nodes: ArrayLike) -> jax.Array:
    """Evaluate Lagrange basis polynomials L_j(xi) for nodes s in pure vectorized JAX.

    Supports both scalar ``xi`` (returns shape ``(n_nodes,)``) and 1D array ``xi``
    (returns shape ``(len(xi), n_nodes)``).
    """
    nodes_arr = jnp.asarray(nodes, dtype=jnp.float64)
    n_nodes = nodes_arr.shape[0]
    xi_arr = jnp.asarray(xi, dtype=jnp.float64)
    is_scalar = xi_arr.ndim == 0
    xi_1d = jnp.atleast_1d(xi_arr)

    keep = ~jnp.eye(n_nodes, dtype=bool)
    numer = xi_1d[:, None, None] - nodes_arr[None, None, :]
    denom = jnp.where(keep, nodes_arr[None, :, None] - nodes_arr[None, None, :], 1.0)
    ratio = jnp.where(keep, numer / denom, 1.0)
    L_vals = jnp.prod(ratio, axis=-1)

    return L_vals[0] if is_scalar else L_vals


def collocation_matrices(m: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute Lagrange differentiation and boundary extrapolation matrices for Gauss collocation.

    Nodes on [0, 1] are s = [0, rho_1, ..., rho_m]. Differentiates ``eval_lagrange_basis``
    via ``jax.jacfwd`` in pure JAX without host loops.

    Parameters
    ----------
    m : int
        Number of Gauss collocation points per mesh interval.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array, jax.Array]
        - D: shape (m, m + 1), derivative matrix D_ij = L'_j(rho_i).
        - C: shape (m + 1,), continuity vector C_j = L_j(1.0).
        - rho: shape (m,), Gauss nodes in (0, 1).
        - weights: shape (m,), Gauss weights.
    """
    rho, weights = gauss_legendre_nodes(m)
    s = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), rho])
    D = jax.vmap(jax.jacfwd(lambda xi: eval_lagrange_basis(xi, s)))(rho)
    C = eval_lagrange_basis(jnp.array(1.0, dtype=jnp.float64), s)
    return D, C, rho, weights


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
        u_mean = jnp.mean(self.u_mesh, axis=0, keepdims=True)
        return jnp.sqrt(jnp.mean(jnp.sum((self.u_mesh - u_mean) ** 2, axis=-1)))

    def evaluate(self, t: float | jax.Array) -> jax.Array:
        """Evaluate continuous periodic orbit state u(t) at any normalized phase t in [0, 1]."""
        return self.problem.eval_solution(self, t)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class CollocationProblem:
    """PyTree dataclass representing the Gauss-Legendre collocation BVP for periodic orbits.

    Parameters
    ----------
    fn : Callable[..., jax.Array]
        Right-hand side function f(u, p, **kwargs) returning 1D state derivative.
    dim : int
        Dimension of state vector u.
    num_intervals : int, default=25
        Number of mesh intervals N.
    num_gauss_points : int, default=4
        Number of Gauss collocation points m per interval (de Boor-Swartz default m=4).
    kwargs : dict[str, Any]
        Additional parameters passed to `fn`.
    """

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
        num_intervals: int = 25,
        num_gauss_points: int = 4,
        **kwargs: Any,
    ) -> None:
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "dim", dim)
        object.__setattr__(self, "num_intervals", int(num_intervals))
        object.__setattr__(self, "num_gauss_points", int(num_gauss_points))
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

    @property
    def total_dim(self) -> int:
        """Total number of scalar unknowns: N * (m + 1) * dim + 1 (period T)."""
        return self.num_intervals * (self.num_gauss_points + 1) * self.dim + 1

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
        N = self.num_intervals
        m = self.num_gauss_points
        n = self.dim
        n_mesh = N * n
        n_gauss = N * m * n

        u_mesh = X[:n_mesh].reshape((N, n))
        u_gauss = X[n_mesh : n_mesh + n_gauss].reshape((N, m, n))
        period = jnp.squeeze(X[-1])
        return u_mesh, u_gauss, period

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

        # 1. Collocation at Gauss points: du/dt - T * f(u_gauss)
        du_dt = jnp.einsum("mj,Njd->Nmd", self.D, U) / h[:, None, None]
        f_val = jax.vmap(jax.vmap(lambda u: self.fn(u, p_val, **self.kwargs)))(u_gauss)
        res_coll = (du_dt - T * f_val).ravel()

        # 2. Continuity with next mesh node: u_{(k+1)%N} - C @ U_k
        u_next = jnp.roll(u_mesh, -1, axis=0)
        u_end_approx = jnp.einsum("j,Njd->Nd", self.C, U)
        res_cont = (u_next - u_end_approx).ravel()

        # 3. Integral phase condition: sum_k hk sum_i w_i <dot{u}_ref(t_i), u_gauss(t_i)>
        U_ref = jnp.concatenate([u_ref_mesh[:, None, :], u_ref_gauss], axis=1)
        dot_u_ref = jnp.einsum("mj,Njd->Nmd", self.D, U_ref) / h[:, None, None]
        inner_prod = jnp.sum(dot_u_ref * u_gauss, axis=-1)
        phase_val = jnp.sum(h[:, None] * self.weights[None, :] * inner_prod)

        return jnp.concatenate([res_coll, res_cont, jnp.atleast_1d(phase_val)])

    def eval_solution(self, sol: PeriodicOrbitResult, t: float | jax.Array) -> jax.Array:
        """Evaluate continuous solution u(t) at any phase t in [0, 1]."""
        mesh = jnp.asarray(sol.mesh, dtype=jnp.float64)
        u_mesh = jnp.asarray(sol.u_mesh, dtype=jnp.float64)
        u_gauss = jnp.asarray(sol.u_gauss, dtype=jnp.float64)
        N = self.num_intervals
        s_nodes = jnp.concatenate([jnp.zeros(1, dtype=jnp.float64), jnp.asarray(self.rho)])

        is_scalar = jnp.ndim(t) == 0
        t_arr = jnp.atleast_1d(jnp.asarray(t, dtype=jnp.float64)) % 1.0

        k = jnp.clip(jnp.searchsorted(mesh, t_arr, side="right") - 1, 0, N - 1)
        hk = mesh[k + 1] - mesh[k]
        xi = jnp.clip((t_arr - mesh[k]) / (hk + 1e-30), 0.0, 1.0)

        U_k = jnp.concatenate([u_mesh[k][:, None, :], u_gauss[k]], axis=1)
        L_vals = eval_lagrange_basis(xi, s_nodes)  # (T, m + 1)

        u_eval = jnp.einsum("tj,tjn->tn", L_vals, U_k)
        return u_eval[0] if is_scalar else u_eval


def initialize_from_hopf(
    problem: CollocationProblem,
    u_hopf: ArrayLike,
    p_hopf: float | jax.Array,
    eps: float | jax.Array = 0.02,
    p_init: float | jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Construct an initial periodic orbit guess near a Hopf bifurcation in pure JAX.

    Parameters
    ----------
    problem : CollocationProblem
        The collocation problem instance.
    u_hopf : jax.Array
        Equilibrium state vector at the Hopf point.
    p_hopf : float | jax.Array
        Parameter value at the Hopf point.
    eps : float | jax.Array, default=0.02
        Perturbation amplitude for the initial cycle.
    p_init : float | jax.Array | None, optional
        Initial parameter slightly detuned from Hopf. Defaults to p_hopf - 0.001.

    Returns
    -------
    tuple[mesh, u_mesh, u_gauss, T0, p_init]
    """
    u_h = jnp.asarray(u_hopf, dtype=jnp.float64)
    p_h = jnp.squeeze(jnp.asarray(p_hopf, dtype=jnp.float64))
    eps_val = jnp.squeeze(jnp.asarray(eps, dtype=jnp.float64))

    Ju = jax.jacobian(lambda u_: problem.fn(u_, p_h, **problem.kwargs))(u_h)
    eigs, vecs = jnp.linalg.eig(Ju)

    complex_mask = jnp.abs(jnp.imag(eigs)) > 1e-4
    real_abs = jnp.where(complex_mask, jnp.abs(jnp.real(eigs)), jnp.inf)
    chosen_idx = jnp.argmin(real_abs)
    val = eigs[chosen_idx]
    omega = jnp.abs(jnp.imag(val))
    T0 = 2.0 * jnp.pi / omega
    vec = vecs[:, chosen_idx]

    # Check for degenerate Hopf bifurcation (repeated critical imaginary eigenvalues) (#14)
    crit_mask = complex_mask & (jnp.abs(jnp.real(eigs)) < 1e-2)
    crit_freqs = jnp.where(crit_mask, jnp.abs(jnp.imag(eigs)), -1.0)
    freq_diffs = jnp.abs(crit_freqs - omega)
    matches = (freq_diffs < 1e-3) & (crit_freqs > 0)
    if bool(jnp.sum(matches) > 2):
        warnings.warn(
            "Degenerate Hopf point detected: critical imaginary eigenvalue has algebraic "
            "multiplicity >= 2. A single eigenvector seed may collapse onto the trivial "
            "equilibrium. Consider increasing eps, setting non_trivial=True, or providing a custom initial guess.",
            UserWarning,
            stacklevel=2,
        )

    v_r = jnp.real(vec)
    v_i = jnp.imag(vec)
    norm_v = jnp.sqrt(jnp.dot(v_r, v_r) + jnp.dot(v_i, v_i))
    v_r = v_r / norm_v
    v_i = v_i / norm_v

    N = problem.num_intervals
    mesh = jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float64)
    rho = jnp.asarray(problem.rho, dtype=jnp.float64)

    t_mesh = mesh[:-1]  # (N,)
    theta_mesh = 2.0 * jnp.pi * t_mesh
    u_mesh = u_h[None, :] + eps_val * (
        jnp.cos(theta_mesh)[:, None] * v_r[None, :] - jnp.sin(theta_mesh)[:, None] * v_i[None, :]
    )

    hk = mesh[1:] - mesh[:-1]  # (N,)
    t_gauss = t_mesh[:, None] + hk[:, None] * rho[None, :]  # (N, m)
    theta_gauss = 2.0 * jnp.pi * t_gauss
    u_gauss = u_h[None, None, :] + eps_val * (
        jnp.cos(theta_gauss)[:, :, None] * v_r[None, None, :]
        - jnp.sin(theta_gauss)[:, :, None] * v_i[None, None, :]
    )

    p_init_arr = (
        p_h - 0.001 if p_init is None else jnp.squeeze(jnp.asarray(p_init, dtype=jnp.float64))
    )

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
    tol: float = 1e-8,
    max_iters: int = 25,
    non_trivial: bool = False,
    min_amplitude: float = 1e-4,
) -> PeriodicOrbitResult:
    """Solve for a periodic orbit (u(t), T) using pure-JAX Gauss-Legendre collocation.

    Uses ``jax.lax.while_loop`` over damped Newton steps with fully vectorized
    collocation residual and Jacobian evaluations without host synchronization.

    Parameters
    ----------
    problem : CollocationProblem
        Collocation problem definition.
    u_mesh_init : jax.Array of shape (N, dim)
        Initial guess for state at mesh nodes.
    T_init : float | jax.Array
        Initial guess for orbit period T.
    p : float | jax.Array
        Continuation parameter value.
    mesh : jax.Array of shape (N + 1,) | None, optional
        Normalized time grid in [0, 1]. Defaults to uniform linspace.
    u_gauss_init : jax.Array of shape (N, m, dim) | None, optional
        Initial guess at Gauss collocation nodes. Defaults to vectorized linear interpolation.
    u_ref_mesh, u_ref_gauss : optional
        Reference solution for integral phase pinning condition.
    tol : float, default=1e-8
        Convergence tolerance for Newton residual norm.
    max_iters : int, default=25
        Maximum Newton iterations.
    non_trivial : bool, default=False
        If True, forbids collapsing onto the trivial constant equilibrium by penalizing
        zero amplitude during Newton iterations and requiring amplitude >= min_amplitude (#14).
    min_amplitude : float, default=1e-4
        Minimum RMS amplitude required for non_trivial=True.

    Returns
    -------
    PeriodicOrbitResult
        Result containing mesh, u_mesh, u_gauss, period, converged, iterations, residual_norm.
    """
    N = problem.num_intervals
    n_mesh = N * problem.dim

    if mesh is None:
        mesh_arr = jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float64)
    else:
        mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)

    u_mesh_arr = jnp.asarray(u_mesh_init, dtype=jnp.float64)
    u_mean_0 = jnp.mean(u_mesh_arr, axis=0, keepdims=True)
    amp_init = jnp.sqrt(jnp.mean(jnp.sum((u_mesh_arr - u_mean_0) ** 2, axis=-1)))

    if u_gauss_init is None:
        u_k0 = u_mesh_arr
        u_k1 = jnp.roll(u_mesh_arr, -1, axis=0)
        rho = problem.rho
        u_gauss_arr = (1.0 - rho[None, :, None]) * u_k0[:, None, :] + rho[None, :, None] * u_k1[
            :, None, :
        ]
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

    r0 = eval_res(X_init)
    res_norm_0 = jnp.linalg.norm(r0)
    conv_0 = res_norm_0 < tol

    init_state = (X_init, jnp.int32(0), conv_0, res_norm_0)

    def cond_fn(state: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> jax.Array:
        _, it, conv, _ = state
        return (~conv) & (it < max_iters)

    def body_fn(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        X, it, _, res_norm = state
        res = eval_res(X)
        J = jac_fn(X)

        s = jnp.linalg.svdvals(J)
        is_inv = (s[-1] / (s[0] + 1e-30)) > 1e-12

        def _solve() -> jax.Array:
            return -jnp.linalg.solve(J, res)

        def _lstsq() -> jax.Array:
            return -jnp.linalg.lstsq(J, res, rcond=1e-12)[0]

        dX = jax.lax.cond(is_inv, _solve, _lstsq)

        def _step_score(x_cand: jax.Array) -> jax.Array:
            rn = jnp.linalg.norm(eval_res(x_cand))
            if non_trivial:
                u_m = x_cand[:n_mesh].reshape((N, problem.dim))
                u_m_mean = jnp.mean(u_m, axis=0, keepdims=True)
                amp_cand = jnp.sqrt(jnp.mean(jnp.sum((u_m - u_m_mean) ** 2, axis=-1)))
                collapse_pen = jnp.where(
                    (amp_init > min_amplitude) & (amp_cand < 0.25 * amp_init),
                    1e4 * (0.25 * amp_init - amp_cand) ** 2,
                    0.0,
                )
                return rn + collapse_pen
            return rn

        X1 = X + dX
        r1 = _step_score(X1)
        X2 = X + 0.5 * dX
        r2 = _step_score(X2)
        X3 = X + 0.25 * dX
        r3 = _step_score(X3)

        X_next = jnp.where(
            r1 < res_norm, X1, jnp.where(r2 < res_norm, X2, jnp.where(r3 < res_norm, X3, X1))
        )
        rn_next = jnp.where(
            r1 < res_norm, r1, jnp.where(r2 < res_norm, r2, jnp.where(r3 < res_norm, r3, r1))
        )

        is_valid = jnp.isfinite(rn_next) & jnp.all(jnp.isfinite(X_next))
        conv_next = is_valid & (rn_next < tol)
        it_next = jnp.where(is_valid, it + 1, jnp.int32(max_iters))
        return (X_next, it_next, conv_next, rn_next)

    X_fin, it_fin, conv_fin, rn_fin = jax.lax.while_loop(cond_fn, body_fn, init_state)
    u_mesh_sol, u_gauss_sol, T_sol = problem.unpack_vars(X_fin)
    u_mean_sol = jnp.mean(u_mesh_sol, axis=0, keepdims=True)
    amp_sol = jnp.sqrt(jnp.mean(jnp.sum((u_mesh_sol - u_mean_sol) ** 2, axis=-1)))

    if non_trivial:
        conv_fin = conv_fin & (amp_sol >= min_amplitude)

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
    dp: float | jax.Array = 0.01,
    dp_min: float = 1e-5,
    dp_max: float = 0.1,
    max_steps: int = 100,
    param_mode: str = "natural",
    param_weight: float | None = None,
    non_trivial: bool = True,
    tol: float = 1e-7,
    max_iters: int = 25,
    verbose: bool = False,
) -> list[PeriodicOrbitResult]:
    """Continue a family of periodic orbits along parameter p (#15).

    Supports natural parameter stepping (``param_mode='natural'``) with adaptive step size
    control, keeping parameter progression strictly monotone and avoiding tangent wiggles
    inherent to high-dimensional collocation pseudo-arclength systems.

    Parameters
    ----------
    problem : CollocationProblem
        Collocation problem instance.
    initial_orbit : PeriodicOrbitResult
        A converged periodic orbit at p_start.
    p_start : float | jax.Array
        Starting parameter value.
    p_end : float | jax.Array
        Target end parameter value.
    dp : float | jax.Array, default=0.01
        Initial parameter step size.
    dp_min : float, default=1e-5
        Minimum allowed step size before terminating.
    dp_max : float, default=0.1
        Maximum allowed step size.
    max_steps : int, default=100
        Maximum continuation steps.
    param_mode : str, default='natural'
        Continuation mode: 'natural' for fixed-parameter stepping with adaptive reduction.
    param_weight : float | None, optional
        Scaling factor for parameter direction.
    non_trivial : bool, default=True
        Whether to enforce non-trivial periodic orbit constraint.
    tol : float, default=1e-7
        Convergence tolerance for collocation Newton solver.
    max_iters : int, default=25
        Maximum iterations per collocation solve.
    verbose : bool, default=False
        If True, print continuation step progress.

    Returns
    -------
    list[PeriodicOrbitResult]
        List of converged periodic orbit results along the parameter branch.
    """
    p_s = float(p_start)
    p_e = float(p_end)
    direction = 1.0 if p_e >= p_s else -1.0
    cur_dp = float(abs(float(dp)))

    if initial_orbit.p is None:
        initial_orbit = initial_orbit._replace(p=jnp.asarray(p_s, dtype=jnp.float64))

    results: list[PeriodicOrbitResult] = [initial_orbit]
    current_orbit = initial_orbit
    p_cur = p_s

    step = 0
    while step < max_steps and (direction * (p_e - p_cur) > 1e-9):
        p_target = p_cur + direction * cur_dp
        if direction > 0 and p_target > p_e:
            p_target = p_e
        elif direction < 0 and p_target < p_e:
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
            tol=tol,
            max_iters=max_iters,
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
                cur_dp = min(cur_dp * 1.25, dp_max)
            step += 1
        else:
            cur_dp *= 0.5
            if cur_dp < dp_min:
                if verbose:
                    print(f"Step {step + 1}: step size {cur_dp} < dp_min ({dp_min}). Terminating.")
                break

    return results
