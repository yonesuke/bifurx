from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp

from bifurx.homotopy.pytree import PyTreeOptionsBase


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WeightedNormOptions(PyTreeOptionsBase):
    """Options for dynamic component-wise weighted infinity norm."""

    _child_fields: ClassVar[tuple[str, ...]] = ("scale_min", "scale_abs_min", "scale_max")

    scale_min: float = 1e-4
    scale_abs_min: float = 1e-6
    scale_max: float = 1e150


def init_weighted_norm(
    z: jax.Array,
    options: WeightedNormOptions | None = None,
) -> jax.Array:
    """Initialize component-wise scaling weights from state-parameter vector `z`."""
    opts = WeightedNormOptions() if options is None else options
    z_abs = jnp.abs(z)
    point_norm = jnp.max(z_abs)
    w = jnp.clip(z_abs, opts.scale_min * point_norm, opts.scale_max * point_norm)
    return jnp.maximum(w, opts.scale_abs_min)


def update_weighted_norm(
    weights: jax.Array,
    z: jax.Array,
    options: WeightedNormOptions | None = None,
) -> jax.Array:
    """Smoothly update component-wise scaling weights using current iterate `z`."""
    opts = WeightedNormOptions() if options is None else options
    z_abs = jnp.abs(z)
    # Clamp against the unweighted scale of z so starting at z=0 never amplifies weights.
    point_norm = jnp.max(z_abs)
    w_cand = 0.5 * (z_abs + weights)
    w_cand = jnp.clip(w_cand, opts.scale_min * point_norm, opts.scale_max * point_norm)
    w_cand = jnp.maximum(w_cand, opts.scale_abs_min)
    return jnp.where(jnp.isfinite(w_cand), w_cand, weights)


def weighted_inf_norm(v: jax.Array, weights: jax.Array) -> jax.Array:
    """Compute weighted infinity norm ||D^{-1} v||_inf."""
    return jnp.max(jnp.abs(v) / weights)


def weighted_distance(a: jax.Array, b: jax.Array, weights: jax.Array) -> jax.Array:
    """Compute weighted infinity distance ||D^{-1} (a - b)||_inf."""
    return weighted_inf_norm(a - b, weights)
