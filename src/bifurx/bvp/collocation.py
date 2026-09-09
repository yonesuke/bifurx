from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


def gauss_legendre_nodes(m: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute normalized Gauss-Legendre quadrature nodes and weights on [0, 1].

    Parameters
    ----------
    m : int
        Number of Gauss collocation points per mesh interval.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (nodes, weights) where nodes are in (0, 1) and sum(weights) == 1.0.
    """
    x, w = np.polynomial.legendre.leggauss(m)
    rho = 0.5 * (x + 1.0)
    weights = 0.5 * w
    return rho, weights


def collocation_matrices(m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Lagrange differentiation and boundary extrapolation matrices for Gauss collocation.

    Nodes on [0, 1] are s = [0, rho_1, ..., rho_m].

    Parameters
    ----------
    m : int
        Number of Gauss collocation points per mesh interval.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - D: shape (m, m + 1), derivative matrix D_ij = L'_j(rho_i).
        - C: shape (m + 1,), continuity vector C_j = L_j(1.0).
        - rho: shape (m,), Gauss nodes in (0, 1).
        - weights: shape (m,), Gauss weights.
    """
    rho, weights = gauss_legendre_nodes(m)
    s = np.concatenate([[0.0], rho])
    n_nodes = m + 1

    # D matrix: derivative of Lagrange basis at rho_i
    D = np.zeros((m, n_nodes), dtype=np.float64)
    for i in range(m):
        xi = rho[i]
        for j in range(n_nodes):
            dL_j = 0.0
            for k in range(n_nodes):
                if k != j:
                    prod = 1.0
                    for l_idx in range(n_nodes):
                        if l_idx != j and l_idx != k:
                            prod *= (xi - s[l_idx]) / (s[j] - s[l_idx])
                    dL_j += prod / (s[j] - s[k])
            D[i, j] = dL_j

    # C vector: Lagrange basis evaluated at 1.0
    C = np.zeros(n_nodes, dtype=np.float64)
    for j in range(n_nodes):
        val = 1.0
        for k in range(n_nodes):
            if k != j:
                val *= (1.0 - s[k]) / (s[j] - s[k])
        C[j] = val

    return D, C, rho, weights


def eval_lagrange_basis(xi: float | Array, nodes: np.ndarray | Array) -> Array:
    """Evaluate Lagrange basis polynomials L_j(xi) for nodes s."""
    nodes_arr = jnp.asarray(nodes, dtype=jnp.float64)
    n_nodes = nodes_arr.shape[0]
    xi_val = jnp.asarray(xi, dtype=jnp.float64)

    # Vectorized evaluation
    def _basis_j(j: int) -> Array:
        # Product over k != j: (xi - s[k]) / (s[j] - s[k])
        terms = jnp.where(
            jnp.arange(n_nodes) == j,
            1.0,
            (xi_val - nodes_arr) / (nodes_arr[j] - nodes_arr + 1e-30),
        )
        return jnp.prod(terms)

    return jnp.array([_basis_j(j) for j in range(n_nodes)])


class PeriodicOrbitResult(NamedTuple):
    """Result of solving a periodic orbit via collocation BVP."""

    mesh: Array
    u_mesh: Array  # shape (N, dim)
    u_gauss: Array  # shape (N, m, dim)
    period: float
    converged: bool
    iterations: int
    residual_norm: float
    problem: CollocationProblem

    @property
    def T(self) -> float:
        """Alias for period."""
        return self.period

    def evaluate(self, t: float | Array) -> Array:
        """Evaluate continuous periodic orbit state u(t) at any normalized phase t in [0, 1]."""
        return self.problem.eval_solution(self, t)


class CollocationProblem(eqx.Module):
    """PyTree dataclass representing the Gauss-Legendre collocation BVP for periodic orbits.

    Parameters
    ----------
    fn : Callable[..., Array]
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

    fn: Callable[..., Array] = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    D: Array
    C: Array
    rho: Array
    weights: Array
    num_intervals: int = eqx.field(default=25, static=True)
    num_gauss_points: int = eqx.field(default=4, static=True)
    kwargs: dict[str, Any] = eqx.field(default_factory=dict, static=True)

    def __init__(
        self,
        fn: Callable[..., Array],
        dim: int,
        num_intervals: int = 25,
        num_gauss_points: int = 4,
        **kwargs: Any,
    ) -> None:
        self.fn = fn
        self.dim = dim
        self.num_intervals = int(num_intervals)
        self.num_gauss_points = int(num_gauss_points)
        self.kwargs = kwargs

        D_np, C_np, rho_np, weights_np = collocation_matrices(self.num_gauss_points)
        self.D = jnp.asarray(D_np, dtype=jnp.float64)
        self.C = jnp.asarray(C_np, dtype=jnp.float64)
        self.rho = jnp.asarray(rho_np, dtype=jnp.float64)
        self.weights = jnp.asarray(weights_np, dtype=jnp.float64)

    @property
    def total_dim(self) -> int:
        """Total number of scalar unknowns: N * (m + 1) * dim + 1 (period T)."""
        return self.num_intervals * (self.num_gauss_points + 1) * self.dim + 1

    def pack_vars(self, u_mesh: Array, u_gauss: Array, period: float | Array) -> Array:
        """Pack (u_mesh, u_gauss, period) into a single 1D flat vector."""
        return jnp.concatenate([
            jnp.asarray(u_mesh, dtype=jnp.float64).ravel(),
            jnp.asarray(u_gauss, dtype=jnp.float64).ravel(),
            jnp.atleast_1d(jnp.asarray(period, dtype=jnp.float64)),
        ])

    def unpack_vars(self, X: Array) -> tuple[Array, Array, Array]:
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
        X: Array,
        p: float | Array,
        mesh: Array,
        u_ref_mesh: Array | None = None,
        u_ref_gauss: Array | None = None,
    ) -> Array:
        """Evaluate collocation BVP residual vector of dimension N*(m+1)*dim + 1."""
        u_mesh, u_gauss, T = self.unpack_vars(X)
        N = self.num_intervals
        m = self.num_gauss_points

        # Mesh interval lengths h_k = t_{k+1} - t_k
        h = mesh[1:] - mesh[:-1]  # shape (N,)

        # Reference orbit for integral phase condition
        if u_ref_mesh is None:
            u_ref_mesh = u_mesh
        if u_ref_gauss is None:
            u_ref_gauss = u_gauss

        colloc_residuals = []
        continuity_residuals = []
        phase_integrals = []

        p_val = jnp.squeeze(p) if jnp.ndim(p) > 0 else p
        for k in range(N):
            hk = h[k]
            u_k = u_mesh[k]  # (n,)
            u_g = u_gauss[k]  # (m, n)
            U_k = jnp.vstack([u_k[None, :], u_g])  # shape (m + 1, n)

            # 1. Collocation at Gauss points
            for i in range(m):
                du_dt_i = jnp.dot(self.D[i, :], U_k) / hk
                f_i = self.fn(u_g[i], p_val, **self.kwargs)
                r_coll = du_dt_i - T * f_i
                colloc_residuals.append(r_coll)

            # 2. Continuity with next mesh node
            u_next = u_mesh[(k + 1) % N]  # Periodic boundary condition: u_N = u_0
            u_end_approx = jnp.dot(self.C, U_k)
            r_cont = u_next - u_end_approx
            continuity_residuals.append(r_cont)

            # 3. Integral phase condition contribution: int <dot{u}_ref, u> dt
            # Use reference polynomial derivative at Gauss points
            U_ref_k = jnp.vstack([u_ref_mesh[k][None, :], u_ref_gauss[k]])
            for i in range(m):
                dot_u_ref_i = jnp.dot(self.D[i, :], U_ref_k) / hk
                phase_integrals.append(hk * self.weights[i] * jnp.dot(dot_u_ref_i, u_g[i]))

        res_coll = jnp.concatenate(colloc_residuals)
        res_cont = jnp.concatenate(continuity_residuals)
        phase_val = jnp.sum(jnp.array(phase_integrals))

        return jnp.concatenate([res_coll, res_cont, jnp.atleast_1d(phase_val)])

    @jax.jit
    def eval_residual(
        self,
        X: Array,
        p: float | Array,
        mesh: Array,
        u_ref_mesh: Array,
        u_ref_gauss: Array,
    ) -> Array:
        """JIT-compiled evaluation of residual vector."""
        return self.residual(X, p, mesh, u_ref_mesh=u_ref_mesh, u_ref_gauss=u_ref_gauss)

    @jax.jit
    def eval_jacobian(
        self,
        X: Array,
        p: float | Array,
        mesh: Array,
        u_ref_mesh: Array,
        u_ref_gauss: Array,
    ) -> Array:
        """JIT-compiled evaluation of residual Jacobian dR/dX."""
        return jax.jacobian(
            lambda x_: self.residual(x_, p, mesh, u_ref_mesh=u_ref_mesh, u_ref_gauss=u_ref_gauss)
        )(X)

    def eval_solution(self, sol: PeriodicOrbitResult, t: float | Array) -> Array:
        """Evaluate continuous solution u(t) at any phase t in [0, 1]."""
        mesh = np.asarray(sol.mesh)
        u_mesh = np.asarray(sol.u_mesh)
        u_gauss = np.asarray(sol.u_gauss)
        s_nodes = np.concatenate([[0.0], np.asarray(self.rho)])
        m = self.num_gauss_points
        N = self.num_intervals

        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64)) % 1.0

        results = []
        for ti in t_arr:
            # Find interval index k where mesh[k] <= ti <= mesh[k+1]
            k = int(np.searchsorted(mesh, ti, side="right") - 1)
            k = max(0, min(N - 1, k))
            hk = mesh[k + 1] - mesh[k]
            xi = (ti - mesh[k]) / (hk + 1e-30)
            xi = max(0.0, min(1.0, xi))

            U_k = np.vstack([u_mesh[k][None, :], u_gauss[k]])  # (m + 1, n)
            # Evaluate Lagrange basis
            L_vals = np.ones(m + 1, dtype=np.float64)
            for j in range(m + 1):
                for l_idx in range(m + 1):
                    if l_idx != j:
                        L_vals[j] *= (xi - s_nodes[l_idx]) / (s_nodes[j] - s_nodes[l_idx])

            u_eval = L_vals @ U_k
            results.append(u_eval)

        res_arr = jnp.asarray(np.array(results))
        if jnp.asarray(t).ndim == 0:
            return res_arr[0]
        return res_arr


def initialize_from_hopf(
    problem: CollocationProblem,
    u_hopf: Array | np.ndarray,
    p_hopf: float,
    eps: float = 0.02,
    p_init: float | None = None,
) -> tuple[Array, Array, Array, float, float]:
    """Construct an initial periodic orbit guess near a Hopf bifurcation.

    Parameters
    ----------
    problem : CollocationProblem
        The collocation problem instance.
    u_hopf : Array | np.ndarray
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
    u_h = np.asarray(u_hopf, dtype=np.float64)
    # Compute Jacobian at Hopf
    jac_fn = jax.jacobian(lambda u_: problem.fn(u_, p_hopf, **problem.kwargs))
    Ju = np.asarray(jac_fn(jnp.asarray(u_h)))

    eigs, vecs = np.linalg.eig(Ju)
    # Find pair with smallest |Re(lambda)| and non-zero Im(lambda)
    imag_indices = [i for i, val in enumerate(eigs) if abs(np.imag(val)) > 1e-4]
    if not imag_indices:
        msg = "No complex-conjugate eigenvalues found at provided equilibrium."
        raise ValueError(msg)

    chosen_idx = imag_indices[int(np.argmin(np.abs(np.real(eigs[imag_indices]))))]
    val = eigs[chosen_idx]
    omega = float(abs(np.imag(val)))
    T0 = 2.0 * np.pi / omega
    vec = vecs[:, chosen_idx]

    v_r = np.real(vec)
    v_i = np.imag(vec)
    norm_v = np.sqrt(np.dot(v_r, v_r) + np.dot(v_i, v_i))
    v_r /= norm_v
    v_i /= norm_v

    N = problem.num_intervals
    m = problem.num_gauss_points
    mesh = np.linspace(0.0, 1.0, N + 1)
    rho = np.asarray(problem.rho)

    u_mesh = np.zeros((N, problem.dim))
    u_gauss = np.zeros((N, m, problem.dim))

    for k in range(N):
        t_k = mesh[k]
        theta_k = 2.0 * np.pi * t_k
        u_mesh[k] = u_h + eps * (v_r * np.cos(theta_k) - v_i * np.sin(theta_k))
        hk = mesh[k + 1] - mesh[k]
        for i in range(m):
            t_g = t_k + hk * rho[i]
            theta_g = 2.0 * np.pi * t_g
            u_gauss[k, i] = u_h + eps * (v_r * np.cos(theta_g) - v_i * np.sin(theta_g))

    if p_init is None:
        p_init = p_hopf - 0.001

    return (
        jnp.asarray(mesh),
        jnp.asarray(u_mesh),
        jnp.asarray(u_gauss),
        float(T0),
        float(p_init),
    )


def solve_periodic_orbit(
    problem: CollocationProblem,
    u_mesh_init: Array | np.ndarray,
    T_init: float,
    p: float | Array,
    mesh: Array | np.ndarray | None = None,
    u_gauss_init: Array | np.ndarray | None = None,
    u_ref_mesh: Array | np.ndarray | None = None,
    u_ref_gauss: Array | np.ndarray | None = None,
    tol: float = 1e-8,
    max_iters: int = 25,
) -> PeriodicOrbitResult:
    """Solve for a periodic orbit (u(t), T) using Gauss-Legendre collocation.

    Parameters
    ----------
    problem : CollocationProblem
        Collocation problem definition.
    u_mesh_init : Array of shape (N, dim)
        Initial guess for state at mesh nodes.
    T_init : float
        Initial guess for orbit period T.
    p : float | Array
        Continuation parameter value.
    mesh : Array of shape (N + 1,) | None, optional
        Normalized time grid in [0, 1]. Defaults to uniform linspace.
    u_gauss_init : Array of shape (N, m, dim) | None, optional
        Initial guess at Gauss collocation nodes. Defaults to linear interpolation.
    u_ref_mesh, u_ref_gauss : optional
        Reference solution for integral phase pinning condition.
    tol : float, default=1e-8
        Convergence tolerance for Newton residual norm.
    max_iters : int, default=25
        Maximum Newton iterations.

    Returns
    -------
    PeriodicOrbitResult
    """
    N = problem.num_intervals

    if mesh is None:
        mesh_arr = jnp.linspace(0.0, 1.0, N + 1)
    else:
        mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)

    u_mesh_arr = jnp.asarray(u_mesh_init, dtype=jnp.float64)

    if u_gauss_init is None:
        # Interpolate linearly between adjacent mesh nodes
        u_gauss_list = []
        rho_np = np.asarray(problem.rho)
        for k in range(N):
            u_k0 = np.asarray(u_mesh_arr[k])
            u_k1 = np.asarray(u_mesh_arr[(k + 1) % N])
            u_g_k = [(1.0 - r) * u_k0 + r * u_k1 for r in rho_np]
            u_gauss_list.append(u_g_k)
        u_gauss_arr = jnp.asarray(np.array(u_gauss_list))
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
        # Solve linear Newton step
        try:
            dX = -jnp.linalg.solve(J, res)
        except Exception:
            # Fallback to least-squares in case of near-singular tangent
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
