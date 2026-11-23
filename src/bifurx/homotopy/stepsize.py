from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TrackerParameters:
    """Theory-grounded adaptive step-size and contraction control parameters (Timme 2020)."""

    _child_fields: ClassVar[tuple[str, ...]] = (
        "a",
        "beta_a",
        "beta_omega_p",
        "beta_tau",
        "max_growth_factor",
        "reject_fallback_factor",
    )

    a: float = 0.125
    beta_a: float = 1.0
    beta_omega_p: float = 3.0
    beta_tau: float = 0.4
    max_growth_factor: float = 2.0
    reject_fallback_factor: float = 0.25

    def replace(self, **changes: Any) -> Self:
        """Return a copy of tracker parameters with updated fields."""
        return dataclasses.replace(self, **changes)

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = tuple(getattr(self, f) for f in self._child_fields)
        return children, ()

    @classmethod
    def tree_unflatten(cls, _aux_data: tuple[Any, ...], children: tuple[Any, ...]) -> Self:
        kwargs = dict(zip(cls._child_fields, children, strict=True))
        return cls(**kwargs)


def kantorovich_h(a: float | jax.Array) -> jax.Array:
    """Evaluate Kantorovich contraction function h(a) = 2a(sqrt(4a^2 + 1) - 2a)."""
    a_arr = jnp.asarray(a, dtype=jnp.float64)
    return 2.0 * a_arr * (jnp.sqrt(4.0 * a_arr * a_arr + 1.0) - 2.0 * a_arr)


def _nth_root(x: jax.Array, order: int) -> jax.Array:
    """Fast real root avoiding slow transcendental `pow` for order 2 and 4."""
    if order == 2:
        return jnp.sqrt(x)
    if order == 4:
        return jnp.sqrt(jnp.sqrt(x))
    return x ** (1.0 / order)


def update_step_size(
    ds: jax.Array,
    converged: jax.Array,
    omega: jax.Array,
    omega_prev: jax.Array,
    local_error: jax.Array,
    trust_region: jax.Array,
    theta: jax.Array,
    iterations: jax.Array,
    last_step_failed: jax.Array,
    predictor_order: int,
    ds_min: jax.Array,
    ds_max: jax.Array,
    params: TrackerParameters | None = None,
) -> jax.Array:
    """Compute next step size using Timme (2020) local-error and contraction control."""
    p_opts = TrackerParameters() if params is None else params
    a_eff = p_opts.beta_a * p_opts.a
    h_a = kantorovich_h(a_eff)

    def _accepted_ds() -> jax.Array:
        omega_safe = jnp.maximum(jnp.where(jnp.isfinite(omega) & (omega > 0.0), omega, 1.0), 1e-12)
        omega_prev_safe = jnp.where(
            jnp.isfinite(omega_prev) & (omega_prev > 0.0), omega_prev, omega_safe
        )
        omega_est = jnp.clip(
            omega_safe + 2.0 * (omega_safe - omega_prev_safe), omega_safe, 8.0 * omega_safe
        )
        has_err = jnp.isfinite(local_error) & (local_error > 1e-30)
        denom = jnp.where(has_err, omega_est * local_error, 1.0)
        ds_err = jnp.where(
            has_err,
            _nth_root((jnp.sqrt(1.0 + 2.0 * h_a) - 1.0) / denom, predictor_order)
            / p_opts.beta_omega_p,
            ds * p_opts.max_growth_factor,
        )
        ds_tr = jnp.where(jnp.isfinite(trust_region), p_opts.beta_tau * trust_region, jnp.inf)
        ds_acc = jnp.minimum(jnp.minimum(ds_err, ds_tr), ds * p_opts.max_growth_factor)
        return jnp.where(last_step_failed, jnp.minimum(ds_acc, ds), ds_acc)

    def _rejected_ds() -> jax.Array:
        j = jnp.clip(iterations - 2, 0, 10)
        theta_valid = jnp.isfinite(theta) & (theta > 0.0) & (iterations >= 2)
        theta_safe = jnp.where(theta_valid, theta, 1.0)
        Theta_j = theta_safe ** (1.0 / (2.0**j))
        h_Theta = kantorovich_h(Theta_j)
        h_half = kantorovich_h(0.5 * a_eff)
        use_formula = theta_valid & (h_Theta > h_half)

        num = jnp.sqrt(1.0 + 2.0 * h_half) - 1.0
        den = jnp.maximum(jnp.sqrt(1.0 + 2.0 * h_Theta) - 1.0, 1e-30)
        ratio = jnp.clip(
            _nth_root(num / den, predictor_order),
            p_opts.reject_fallback_factor,
            0.75,
        )
        return jnp.where(use_formula, ratio * ds, p_opts.reject_fallback_factor * ds)

    ds_cand = jax.lax.cond(converged, _accepted_ds, _rejected_ds)
    return jnp.clip(ds_cand, ds_min, ds_max)
