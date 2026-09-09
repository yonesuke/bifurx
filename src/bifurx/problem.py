from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


class BifurcationProblem(eqx.Module):
    """PyTree dataclass representing the nonlinear equilibrium system $F(u, p) = 0$.

    Parameters
    ----------
    fn : Callable[..., Array]
        Residual function F(u, p, **kwargs) returning a 1D Array of residuals.
    u0 : Array
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

    fn: Callable[..., Array] = eqx.field(static=True)
    u0: Array
    p0: float
    param_bounds: tuple[float, float] | None = eqx.field(default=None, static=True)
    param_index: int | None = eqx.field(default=None, static=True)
    kwargs: dict[str, Any] = eqx.field(default_factory=dict, static=True)

    def __init__(
        self,
        fn: Callable[..., Array],
        u0: Any,
        p0: float | int,
        param_bounds: tuple[float, float] | None = None,
        param_index: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.fn = fn
        u_arr = jnp.asarray(u0, dtype=jnp.float64)
        if u_arr.ndim == 0:
            u_arr = jnp.atleast_1d(u_arr)
        self.u0 = u_arr
        self.p0 = float(p0)
        self.param_bounds = param_bounds
        self.param_index = param_index
        self.kwargs = kwargs

    @property
    def dim(self) -> int:
        """Dimension of state vector u."""
        return int(self.u0.shape[0])

    def residual(self, u: Array, p: float | Array) -> Array:
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

    def jacobian_u(self, u: Array, p: float | Array) -> Array:
        """Compute Jacobian with respect to state u: $\\nabla_u F(u, p)$."""
        u_arr = jnp.asarray(u, dtype=jnp.float64)
        if u_arr.ndim == 0:
            u_arr = jnp.atleast_1d(u_arr)
        jac = jax.jacobian(lambda u_: self.residual(u_, p))(u_arr)
        if jac.ndim == 1:
            jac = jac.reshape(1, -1)
        return jac

    def jacobian_p(self, u: Array, p: float | Array) -> Array:
        """Compute parameter derivative: $\\nabla_p F(u, p)$."""
        p_val = jnp.asarray(p, dtype=jnp.float64)
        if p_val.ndim > 0:
            p_val = p_val.squeeze()
        jac = jax.jacobian(lambda p_: self.residual(u, p_))(p_val)
        return jnp.asarray(jac, dtype=jnp.float64).reshape(self.dim)
