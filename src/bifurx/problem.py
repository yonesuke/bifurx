from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class BifurcationProblem:
    """PyTree dataclass representing the nonlinear equilibrium system $F(u, p) = 0$.

    Parameters
    ----------
    fn : Callable[..., jax.Array]
        Residual function F(u, p, **kwargs) returning a 1D jax.Array of residuals.
    u0 : jax.Array
        Initial state vector u.
    p0 : float
        Initial continuation parameter value p.
    param_bounds : tuple[float, float] | None, optional
        Optional (p_min, p_max) bounds to terminate continuation.
    param_index : int | None, optional
        Optional index if parameter p is an element of a parameter vector.
    **kwargs : Any
        Additional keyword arguments passed to `fn`.
    """

    fn: Callable[..., jax.Array]
    u0: jax.Array
    p0: float
    param_bounds: tuple[float, float] | None = None
    param_index: int | None = None
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __init__(
        self,
        fn: Callable[..., jax.Array],
        u0: Any,
        p0: float | int,
        param_bounds: tuple[float, float] | None = None,
        param_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        object.__setattr__(self, "fn", fn)
        u_arr = jnp.asarray(u0, dtype=jnp.float64)
        if u_arr.ndim == 0:
            u_arr = jnp.atleast_1d(u_arr)
        object.__setattr__(self, "u0", u_arr)
        object.__setattr__(self, "p0", float(p0))
        object.__setattr__(self, "param_bounds", param_bounds)
        object.__setattr__(self, "param_index", param_index)
        object.__setattr__(self, "kwargs", kwargs)

    def tree_flatten(self) -> tuple[tuple[jax.Array, float], tuple[Any, ...]]:
        """Flatten into JAX pytree leaves ``(u0, p0)`` and static auxiliary data."""
        children = (self.u0, self.p0)
        aux = (self.fn, self.param_bounds, self.param_index, self.kwargs)
        return children, aux

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[Any, ...], children: tuple[jax.Array, float]
    ) -> BifurcationProblem:
        """Rebuild a problem from JAX pytree leaves without re-running ``__init__``."""
        fn, param_bounds, param_index, kwargs = aux
        u0, p0 = children
        obj = object.__new__(cls)
        object.__setattr__(obj, "fn", fn)
        object.__setattr__(obj, "u0", u0)
        object.__setattr__(obj, "p0", p0)
        object.__setattr__(obj, "param_bounds", param_bounds)
        object.__setattr__(obj, "param_index", param_index)
        object.__setattr__(obj, "kwargs", kwargs)
        return obj

    @property
    def dim(self) -> int:
        """Dimension of state vector u."""
        return int(self.u0.shape[0])

    def residual(self, u: jax.Array, p: float | jax.Array) -> jax.Array:
        """Evaluate nonlinear residual F(u, p)."""
        u_arr = jnp.asarray(u, dtype=jnp.float64)
        if u_arr.ndim == 0:
            u_arr = jnp.atleast_1d(u_arr)
        p_val = jnp.asarray(p, dtype=jnp.float64)
        if p_val.ndim == 0:
            p_val = float(p_val) if not isinstance(p, jax.core.Tracer) else p_val
        res = self.fn(u_arr, p_val, **self.kwargs)
        res_arr = jnp.asarray(res, dtype=jnp.float64)
        if res_arr.ndim == 0:
            res_arr = jnp.atleast_1d(res_arr)
        return res_arr

    def jacobian_u(self, u: jax.Array, p: float | jax.Array) -> jax.Array:
        """Compute Jacobian with respect to state u: $\\nabla_u F(u, p)$."""
        u_arr = jnp.asarray(u, dtype=jnp.float64)
        if u_arr.ndim == 0:
            u_arr = jnp.atleast_1d(u_arr)
        jac = jax.jacobian(lambda u_: self.residual(u_, p))(u_arr)
        if jac.ndim == 1:
            jac = jac.reshape(1, -1)
        return jac

    def jacobian_p(self, u: jax.Array, p: float | jax.Array) -> jax.Array:
        """Compute parameter derivative: $\\nabla_p F(u, p)$."""
        p_val = jnp.asarray(p, dtype=jnp.float64)
        if p_val.ndim > 0:
            p_val = p_val.squeeze()
        jac = jax.jacobian(lambda p_: self.residual(u, p_))(p_val)
        return jnp.asarray(jac, dtype=jnp.float64).reshape(self.dim)
