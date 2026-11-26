from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.options import CollocationOptions


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
        """Root-mean-square amplitude of orbit around its mean state."""
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

    def derivative_at_gauss(
        self, u_mesh: jax.Array, u_gauss: jax.Array, mesh: jax.Array
    ) -> jax.Array:
        """Evaluate collocation polynomial time derivative du/dt at all Gauss points."""
        h = mesh[1:] - mesh[:-1]
        U = jnp.concatenate([u_mesh[:, None, :], u_gauss], axis=1)
        return jnp.einsum("mj,Njd->Nmd", self.D, U) / h[:, None, None]

    def make_phase_chart_fn(
        self, mesh: jax.Array
    ) -> Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
        """Build a chart callback returning ``(dot_u_ref, u_ref_gauss)`` for integral phase anchor."""
        mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)

        def chart_fn(X_ref: jax.Array, _p: jax.Array) -> tuple[jax.Array, jax.Array]:
            u_ref_m, u_ref_g, _ = self.unpack_vars(X_ref)
            return self.derivative_at_gauss(u_ref_m, u_ref_g, mesh_arr), u_ref_g

        return chart_fn

    def _eval_residual_from_rhs(
        self,
        u_mesh: jax.Array,
        u_gauss: jax.Array,
        T: jax.Array,
        f_val: jax.Array,
        mesh: jax.Array,
        u_ref_mesh: jax.Array | None = None,
        u_ref_gauss: jax.Array | None = None,
        dot_u_ref: jax.Array | None = None,
    ) -> jax.Array:
        """Assemble collocation, continuity, and integral phase residuals from pre-evaluated RHS."""
        h = mesh[1:] - mesh[:-1]
        U = jnp.concatenate([u_mesh[:, None, :], u_gauss], axis=1)

        du_dt = jnp.einsum("mj,Njd->Nmd", self.D, U) / h[:, None, None]
        res_coll = (du_dt - T * f_val).ravel()

        u_next = jnp.roll(u_mesh, -1, axis=0)
        if self.period_shift is not None:
            u_next = u_next.at[-1].add(self.period_shift)
        u_end_approx = jnp.einsum("j,Njd->Nd", self.C, U)
        res_cont = (u_next - u_end_approx).ravel()

        if dot_u_ref is None:
            ref_mesh = u_mesh if u_ref_mesh is None else u_ref_mesh
            ref_gauss = u_gauss if u_ref_gauss is None else u_ref_gauss
            dot_u_ref_val = self.derivative_at_gauss(ref_mesh, ref_gauss, mesh)
        else:
            dot_u_ref_val = dot_u_ref
        u_phase = u_gauss - u_ref_gauss if u_ref_gauss is not None else u_gauss
        inner_prod = jnp.sum(dot_u_ref_val * u_phase, axis=-1)
        phase_val = jnp.sum(h[:, None] * self.weights[None, :] * inner_prod)

        return jnp.concatenate([res_coll, res_cont, jnp.atleast_1d(phase_val)])

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
        p_val = jnp.squeeze(p) if jnp.ndim(p) > 0 else p
        f_val = jax.vmap(jax.vmap(lambda u: self.fn(u, p_val, **self.kwargs)))(u_gauss)
        return self._eval_residual_from_rhs(
            u_mesh,
            u_gauss,
            T,
            f_val,
            mesh,
            u_ref_mesh=u_ref_mesh,
            u_ref_gauss=u_ref_gauss,
            dot_u_ref=dot_u_ref,
        )
