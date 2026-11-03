from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def gauss_legendre_nodes(m: int) -> tuple[jax.Array, jax.Array]:
    """Compute normalized Gauss-Legendre quadrature nodes and weights on [0, 1].

    Uses the Golub-Welsch algorithm (eigen-decomposition of the Legendre Jacobi matrix)
    so no host NumPy/SciPy quadrature routine is required.

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


def collocation_matrices(m: int) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute Lagrange differentiation and boundary extrapolation matrices for Gauss collocation.

    Nodes on [0, 1] are s = [0, rho_1, ..., rho_m].

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
    rho_list = [float(v) for v in rho]
    s = [0.0, *rho_list]
    n_nodes = m + 1

    # D matrix: derivative of Lagrange basis at rho_i (host precomputation)
    D = [[0.0] * n_nodes for _ in range(m)]
    for i in range(m):
        xi = rho_list[i]
        for j in range(n_nodes):
            dL_j = 0.0
            for k in range(n_nodes):
                if k != j:
                    prod = 1.0
                    for l_idx in range(n_nodes):
                        if l_idx != j and l_idx != k:
                            prod *= (xi - s[l_idx]) / (s[j] - s[l_idx])
                    dL_j += prod / (s[j] - s[k])
            D[i][j] = dL_j

    # C vector: Lagrange basis evaluated at 1.0
    C = [0.0] * n_nodes
    for j in range(n_nodes):
        val = 1.0
        for k in range(n_nodes):
            if k != j:
                val *= (1.0 - s[k]) / (s[j] - s[k])
        C[j] = val

    return (
        jnp.asarray(D, dtype=jnp.float64),
        jnp.asarray(C, dtype=jnp.float64),
        rho,
        weights,
    )


def eval_lagrange_basis(xi: float | jax.Array, nodes: ArrayLike) -> jax.Array:
    """Evaluate Lagrange basis polynomials L_j(xi) for nodes s."""
    nodes_arr = jnp.asarray(nodes, dtype=jnp.float64)
    n_nodes = nodes_arr.shape[0]
    xi_val = jnp.asarray(xi, dtype=jnp.float64)

    # Vectorized evaluation
    def _basis_j(j: int) -> jax.Array:
        # Product over k != j: (xi - s[k]) / (s[j] - s[k])
        terms = jnp.where(
            jnp.arange(n_nodes) == j,
            1.0,
            (xi_val - nodes_arr) / (nodes_arr[j] - nodes_arr + 1e-30),
        )
        return jnp.prod(terms)

    return jnp.array([_basis_j(j) for j in range(n_nodes)])


class PeriodicOrbitResultJax(NamedTuple):
    """Result of pure JAX collocation BVP solve."""

    mesh: jax.Array
    u_mesh: jax.Array  # shape (N, dim)
    u_gauss: jax.Array  # shape (N, m, dim)
    period: jax.Array
    converged: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array
    problem: CollocationProblem

    @property
    def T(self) -> jax.Array:
        """Alias for period."""
        return self.period


class PeriodicOrbitResult(NamedTuple):
    """Result of solving a periodic orbit via collocation BVP."""

    mesh: jax.Array
    u_mesh: jax.Array  # shape (N, dim)
    u_gauss: jax.Array  # shape (N, m, dim)
    period: float
    converged: bool
    iterations: int
    residual_norm: float
    problem: CollocationProblem

    @property
    def T(self) -> float:
        """Alias for period."""
        return self.period

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

        # Mesh interval lengths h_k = t_{k+1} - t_k
        h = mesh[1:] - mesh[:-1]  # shape (N,)

        # Reference orbit for integral phase condition
        if u_ref_mesh is None:
            u_ref_mesh = u_mesh
        if u_ref_gauss is None:
            u_ref_gauss = u_gauss

        p_val = jnp.squeeze(p) if jnp.ndim(p) > 0 else p

        # U: shape (N, m + 1, dim) containing u_mesh at index 0 and u_gauss at indices 1..m
        U = jnp.concatenate([u_mesh[:, None, :], u_gauss], axis=1)

        # 1. Collocation at Gauss points: du/dt - T * f(u_gauss)
        # D has shape (m, m + 1); einsum computes (D @ U_k) / hk for each interval k
        du_dt = jnp.einsum("mj,Njd->Nmd", self.D, U) / h[:, None, None]
        f_val = jax.vmap(jax.vmap(lambda u: self.fn(u, p_val, **self.kwargs)))(u_gauss)
        res_coll = (du_dt - T * f_val).ravel()

        # 2. Continuity with next mesh node: u_{(k+1)%N} - C @ U_k
        u_next = jnp.roll(u_mesh, -1, axis=0)
        u_end_approx = jnp.einsum("j,Njd->Nd", self.C, U)
        res_cont = (u_next - u_end_approx).ravel()

        # 3. Integral phase condition contribution: sum_k hk sum_i w_i <dot{u}_ref(t_i), u_gauss(t_i)>
        U_ref = jnp.concatenate([u_ref_mesh[:, None, :], u_ref_gauss], axis=1)
        dot_u_ref = jnp.einsum("mj,Njd->Nmd", self.D, U_ref) / h[:, None, None]
        inner_prod = jnp.sum(dot_u_ref * u_gauss, axis=-1)  # shape (N, m)
        phase_val = jnp.sum(h[:, None] * self.weights[None, :] * inner_prod)

        return jnp.concatenate([res_coll, res_cont, jnp.atleast_1d(phase_val)])

    @jax.jit
    def eval_residual(
        self,
        X: jax.Array,
        p: float | jax.Array,
        mesh: jax.Array,
        u_ref_mesh: jax.Array,
        u_ref_gauss: jax.Array,
    ) -> jax.Array:
        """JIT-compiled evaluation of residual vector."""
        return self.residual(X, p, mesh, u_ref_mesh=u_ref_mesh, u_ref_gauss=u_ref_gauss)

    @jax.jit
    def eval_jacobian(
        self,
        X: jax.Array,
        p: float | jax.Array,
        mesh: jax.Array,
        u_ref_mesh: jax.Array,
        u_ref_gauss: jax.Array,
    ) -> jax.Array:
        """JIT-compiled evaluation of residual Jacobian dR/dX."""
        return jax.jacobian(
            lambda x_: self.residual(x_, p, mesh, u_ref_mesh=u_ref_mesh, u_ref_gauss=u_ref_gauss)
        )(X)

    def eval_solution(self, sol: PeriodicOrbitResult, t: float | jax.Array) -> jax.Array:
        """Evaluate continuous solution u(t) at any phase t in [0, 1]."""
        mesh = jnp.asarray(sol.mesh)
        u_mesh = jnp.asarray(sol.u_mesh)
        u_gauss = jnp.asarray(sol.u_gauss)
        m = self.num_gauss_points
        N = self.num_intervals
        s_nodes = jnp.concatenate([jnp.zeros(1), jnp.asarray(self.rho)])  # (m + 1,)

        is_scalar = jnp.ndim(t) == 0
        t_arr = jnp.atleast_1d(jnp.asarray(t, dtype=jnp.float64)) % 1.0

        # Locate interval k where mesh[k] <= t <= mesh[k + 1]
        k = jnp.clip(jnp.searchsorted(mesh, t_arr, side="right") - 1, 0, N - 1)
        hk = mesh[k + 1] - mesh[k]
        xi = jnp.clip((t_arr - mesh[k]) / (hk + 1e-30), 0.0, 1.0)

        # U: shape (T, m + 1, n) with u_mesh at index 0 and u_gauss at indices 1..m
        U_k = jnp.concatenate([u_mesh[k][:, None, :], u_gauss[k]], axis=1)

        # Lagrange basis L[t, j] = prod_{l != j} (xi_t - s_l) / (s_j - s_l)
        keep = ~jnp.eye(m + 1, dtype=bool)
        numer = xi[:, None, None] - s_nodes[None, None, :]
        denom = s_nodes[None, :, None] - s_nodes[None, None, :]
        ratio = jnp.where(keep, numer / (denom + 1e-30), 1.0)
        L_vals = jnp.prod(ratio, axis=-1)  # (T, m + 1)

        u_eval = jnp.einsum("tj,tjn->tn", L_vals, U_k)
        if is_scalar:
            return u_eval[0]
        return u_eval


def initialize_from_hopf(
    problem: CollocationProblem,
    u_hopf: ArrayLike,
    p_hopf: float,
    eps: float = 0.02,
    p_init: float | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, float, float]:
    """Construct an initial periodic orbit guess near a Hopf bifurcation.

    Parameters
    ----------
    problem : CollocationProblem
        The collocation problem instance.
    u_hopf : jax.Array
        Equilibrium state vector at the Hopf point.
    p_hopf : float
        Parameter value at the Hopf point.
    eps : float, default=0.02
        Perturbation amplitude for the initial cycle.
    p_init : float | None, optional
        Initial parameter slightly detuned from Hopf. Defaults to p_hopf - 0.001.

    Returns
    -------
    tuple[mesh, u_mesh, u_gauss, T0, p_init]
    """
    u_h = jnp.asarray(u_hopf, dtype=jnp.float64)
    # Compute Jacobian at Hopf
    jac_fn = jax.jacobian(lambda u_: problem.fn(u_, p_hopf, **problem.kwargs))
    Ju = jac_fn(u_h)

    eigs, vecs = jnp.linalg.eig(Ju)
    # Find pair with smallest |Re(lambda)| and non-zero Im(lambda)
    complex_mask = jnp.abs(jnp.imag(eigs)) > 1e-4
    if not bool(jnp.any(complex_mask)):
        msg = "No complex-conjugate eigenvalues found at provided equilibrium."
        raise ValueError(msg)

    real_abs = jnp.where(complex_mask, jnp.abs(jnp.real(eigs)), jnp.inf)
    chosen_idx = int(jnp.argmin(real_abs))
    val = eigs[chosen_idx]
    omega = float(jnp.abs(jnp.imag(val)))
    T0 = 2.0 * jnp.pi / omega
    vec = vecs[:, chosen_idx]

    v_r = jnp.real(vec)
    v_i = jnp.imag(vec)
    norm_v = jnp.sqrt(jnp.dot(v_r, v_r) + jnp.dot(v_i, v_i))
    v_r = v_r / norm_v
    v_i = v_i / norm_v

    N = problem.num_intervals
    m = problem.num_gauss_points
    mesh = jnp.linspace(0.0, 1.0, N + 1)
    rho = jnp.asarray(problem.rho)

    u_mesh = jnp.zeros((N, problem.dim))
    u_gauss = jnp.zeros((N, m, problem.dim))

    for k in range(N):
        t_k = mesh[k]
        theta_k = 2.0 * jnp.pi * t_k
        u_mesh = u_mesh.at[k].set(u_h + eps * (v_r * jnp.cos(theta_k) - v_i * jnp.sin(theta_k)))
        hk = mesh[k + 1] - mesh[k]
        for i in range(m):
            t_g = t_k + hk * rho[i]
            theta_g = 2.0 * jnp.pi * t_g
            u_gauss = u_gauss.at[k, i].set(
                u_h + eps * (v_r * jnp.cos(theta_g) - v_i * jnp.sin(theta_g))
            )

    if p_init is None:
        p_init = p_hopf - 0.001

    return (
        mesh,
        u_mesh,
        u_gauss,
        float(T0),
        float(p_init),
    )


def solve_periodic_orbit_jax(
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
) -> PeriodicOrbitResultJax:
    """Solve for a periodic orbit (u(t), T) using JIT-accelerated Gauss-Legendre collocation.

    Uses `jax.lax.while_loop` over Newton damping steps, with fully vectorized
    collocation residual and Jacobian evaluations in pure JAX without host synchronization.

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

    Returns
    -------
    PeriodicOrbitResult
        Result containing mesh, u_mesh, u_gauss, period, converged, iterations, residual_norm.
    """
    N = problem.num_intervals

    if mesh is None:
        mesh_arr = jnp.linspace(0.0, 1.0, N + 1)
    else:
        mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)

    u_mesh_arr = jnp.asarray(u_mesh_init, dtype=jnp.float64)

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

    r0 = problem.residual(
        X_init, p_arr, mesh_arr, u_ref_mesh=u_ref_mesh_arr, u_ref_gauss=u_ref_gauss_arr
    )
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
        res = problem.residual(
            X, p_arr, mesh_arr, u_ref_mesh=u_ref_mesh_arr, u_ref_gauss=u_ref_gauss_arr
        )
        J = jax.jacobian(
            lambda x_: problem.residual(
                x_, p_arr, mesh_arr, u_ref_mesh=u_ref_mesh_arr, u_ref_gauss=u_ref_gauss_arr
            )
        )(X)

        s = jnp.linalg.svdvals(J)
        is_inv = (s[-1] / (s[0] + 1e-30)) > 1e-12

        def _solve() -> jax.Array:
            return -jnp.linalg.solve(J, res)

        def _lstsq() -> jax.Array:
            return -jnp.linalg.lstsq(J, res, rcond=1e-12)[0]

        dX = jax.lax.cond(is_inv, _solve, _lstsq)

        X1 = X + dX
        r1 = jnp.linalg.norm(
            problem.residual(
                X1, p_arr, mesh_arr, u_ref_mesh=u_ref_mesh_arr, u_ref_gauss=u_ref_gauss_arr
            )
        )
        X2 = X + 0.5 * dX
        r2 = jnp.linalg.norm(
            problem.residual(
                X2, p_arr, mesh_arr, u_ref_mesh=u_ref_mesh_arr, u_ref_gauss=u_ref_gauss_arr
            )
        )
        X3 = X + 0.25 * dX
        r3 = jnp.linalg.norm(
            problem.residual(
                X3, p_arr, mesh_arr, u_ref_mesh=u_ref_mesh_arr, u_ref_gauss=u_ref_gauss_arr
            )
        )

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

    return PeriodicOrbitResultJax(
        mesh=mesh_arr,
        u_mesh=u_mesh_sol,
        u_gauss=u_gauss_sol,
        period=T_sol,
        converged=conv_fin,
        iterations=it_fin,
        residual_norm=rn_fin,
        problem=problem,
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
    tol: float = 1e-8,
    max_iters: int = 25,
    use_jit: bool = False,
) -> PeriodicOrbitResult:
    """Solve for a periodic orbit (u(t), T) using Gauss-Legendre collocation.

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
        Initial guess at Gauss collocation nodes. Defaults to linear interpolation.
    u_ref_mesh, u_ref_gauss : optional
        Reference solution for integral phase pinning condition.
    tol : float, default=1e-8
        Convergence tolerance for Newton residual norm.
    max_iters : int, default=25
        Maximum Newton iterations.
    use_jit : bool, default=False
        Whether to use JIT-compiled while_loop solver `solve_periodic_orbit_jax`.

    Returns
    -------
    PeriodicOrbitResult
    """
    if use_jit or isinstance(u_mesh_init, jax.core.Tracer) or isinstance(p, jax.core.Tracer):
        res = solve_periodic_orbit_jax(
            problem=problem,
            u_mesh_init=u_mesh_init,
            T_init=T_init,
            p=p,
            mesh=mesh,
            u_gauss_init=u_gauss_init,
            u_ref_mesh=u_ref_mesh,
            u_ref_gauss=u_ref_gauss,
            tol=tol,
            max_iters=max_iters,
        )
        if isinstance(u_mesh_init, jax.core.Tracer) or isinstance(p, jax.core.Tracer):
            return PeriodicOrbitResult(
                mesh=res.mesh,
                u_mesh=res.u_mesh,
                u_gauss=res.u_gauss,
                period=res.period,  # type: ignore[arg-type]
                converged=res.converged,  # type: ignore[arg-type]
                iterations=res.iterations,  # type: ignore[arg-type]
                residual_norm=res.residual_norm,  # type: ignore[arg-type]
                problem=problem,
            )
        return PeriodicOrbitResult(
            mesh=res.mesh,
            u_mesh=res.u_mesh,
            u_gauss=res.u_gauss,
            period=float(res.period),
            converged=bool(res.converged),
            iterations=int(res.iterations),
            residual_norm=float(res.residual_norm),
            problem=problem,
        )

    N = problem.num_intervals

    if mesh is None:
        mesh_arr = jnp.linspace(0.0, 1.0, N + 1)
    else:
        mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)

    u_mesh_arr = jnp.asarray(u_mesh_init, dtype=jnp.float64)

    if u_gauss_init is None:
        # Interpolate linearly between adjacent mesh nodes
        rho = problem.rho
        u_gauss_list = []
        for k in range(N):
            u_k0 = u_mesh_arr[k]
            u_k1 = u_mesh_arr[(k + 1) % N]
            u_gauss_list.append((1.0 - rho[:, None]) * u_k0[None, :] + rho[:, None] * u_k1[None, :])
        u_gauss_arr = jnp.stack(u_gauss_list)
    else:
        u_gauss_arr = jnp.asarray(u_gauss_init, dtype=jnp.float64)

    if u_ref_mesh is None:
        u_ref_mesh = u_mesh_arr
    if u_ref_gauss is None:
        u_ref_gauss = u_gauss_arr

    p_arr = jnp.asarray(p, dtype=jnp.float64)
    if p_arr.ndim > 0:
        p_arr = jnp.squeeze(p_arr)
    X = problem.pack_vars(u_mesh_arr, u_gauss_arr, T_init)
    converged = False
    iterations_taken = 0
    final_res = 1.0

    for it in range(1, max_iters + 1):
        iterations_taken = it
        res = problem.eval_residual(X, p_arr, mesh_arr, u_ref_mesh, u_ref_gauss)
        final_res = float(jnp.linalg.norm(res))
        if final_res < tol:
            converged = True
            break

        J = problem.eval_jacobian(X, p_arr, mesh_arr, u_ref_mesh, u_ref_gauss)
        # Finite-mask solve: fall back to least-squares on a singular/near-singular Jacobian
        s = jnp.linalg.svdvals(J)
        if float(s[-1] / (s[0] + 1e-30)) > 1e-12:
            dX = -jnp.linalg.solve(J, res)
        else:
            dX = -jnp.linalg.lstsq(J, res)[0]

        # Simple backtracking line search
        alpha = 1.0
        X_candidate = X + dX
        for _ in range(3):
            res_cand = float(
                jnp.linalg.norm(
                    problem.eval_residual(X_candidate, p_arr, mesh_arr, u_ref_mesh, u_ref_gauss)
                )
            )
            if res_cand < final_res or alpha < 0.25:
                break
            alpha *= 0.5
            X_candidate = X + alpha * dX

        X = X_candidate

    u_mesh_sol, u_gauss_sol, T_sol = problem.unpack_vars(X)

    return PeriodicOrbitResult(
        mesh=mesh_arr,
        u_mesh=u_mesh_sol,
        u_gauss=u_gauss_sol,
        period=float(T_sol),
        converged=converged,
        iterations=iterations_taken,
        residual_norm=final_res,
        problem=problem,
    )
