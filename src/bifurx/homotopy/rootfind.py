from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

_SECANT_DENOM_EPS = 1e-30


class SafeguardedRootResult(NamedTuple):
    """Result of pure-JAX 1D safeguarded Illinois-secant / bisection root finding on [0, 1]."""

    alpha: jax.Array
    f_val: jax.Array
    payload: Any
    iterations: jax.Array
    converged: jax.Array


def safeguarded_secant_root(
    eval_fn: Callable[[jax.Array], tuple[jax.Array, Any]],
    f_lo: jax.Array,
    f_hi: jax.Array,
    init_payload: Any,
    *,
    tol: float | jax.Array = 1e-9,
    span_tol: float = 1e-12,
    margin_ratio: float = 0.05,
    max_iters: int = 24,
    illinois: bool = True,
) -> SafeguardedRootResult:
    """Find a scalar root of ``eval_fn(alpha)[0] = 0`` on ``[0, 1]`` in pure JAX.

    Combines Illinois-modified secant steps with a safeguarded bisection fallback inside
    ``jax.lax.while_loop``, tracking the best PyTree ``payload`` observed along the bracket.
    """
    f_lo0 = jnp.squeeze(jnp.asarray(f_lo, dtype=jnp.float64))
    f_hi0 = jnp.squeeze(jnp.asarray(f_hi, dtype=jnp.float64))
    tol_arr = jnp.squeeze(jnp.asarray(tol, dtype=jnp.float64))

    init_f_best = jnp.where(jnp.abs(f_lo0) <= jnp.abs(f_hi0), f_lo0, f_hi0)
    init_state = (
        jnp.array(0.0, dtype=jnp.float64),  # a_lo
        jnp.array(1.0, dtype=jnp.float64),  # a_hi
        f_lo0,
        f_hi0,
        jnp.array(0.5, dtype=jnp.float64),  # alpha_best
        init_f_best,  # f_best
        jnp.abs(init_f_best),  # abs_f_best
        init_payload,  # payload_best
        jnp.int32(0),  # last_side (-1: hi moved, +1: lo moved, 0: init)
        jnp.int32(0),  # it
        jnp.bool_(False),  # done
    )

    def cond_fn(state: tuple[Any, ...]) -> jax.Array:
        a_lo, a_hi, _, _, _, _, _, _, _, it, done = state
        return (~done) & (it < max_iters) & ((a_hi - a_lo) > span_tol)

    def body_fn(state: tuple[Any, ...]) -> tuple[Any, ...]:
        (
            a_lo,
            a_hi,
            f_lo_c,
            f_hi_c,
            alpha_best,
            f_best,
            abs_f_best,
            payload_best,
            last_side,
            it,
            _,
        ) = state
        span = a_hi - a_lo
        denom = f_hi_c - f_lo_c
        safe_denom = jnp.where(jnp.abs(denom) > _SECANT_DENOM_EPS, denom, 1.0)
        a_sec = jnp.where(
            jnp.abs(denom) > _SECANT_DENOM_EPS,
            a_lo - f_lo_c * span / safe_denom,
            0.5 * (a_lo + a_hi),
        )
        margin = margin_ratio * span
        use_sec = (a_sec > a_lo + margin) & (a_sec < a_hi - margin)
        alpha = jnp.where(use_sec, a_sec, 0.5 * (a_lo + a_hi))

        f_mid, payload_mid = eval_fn(alpha)
        f_mid = jnp.squeeze(jnp.asarray(f_mid, dtype=jnp.float64))
        valid_mid = jnp.isfinite(f_mid)

        go_left = (f_lo_c * f_mid) <= 0.0
        cur_side = jnp.where(go_left, jnp.int32(-1), jnp.int32(1))
        same_side = (cur_side == last_side) if illinois else jnp.bool_(False)

        a_lo_new = jnp.where(valid_mid & (~go_left), alpha, a_lo)
        f_lo_new = jnp.where(
            valid_mid & (~go_left),
            f_mid,
            jnp.where(valid_mid & go_left & same_side, 0.5 * f_lo_c, f_lo_c),
        )
        a_hi_new = jnp.where(valid_mid & go_left, alpha, a_hi)
        f_hi_new = jnp.where(
            valid_mid & go_left,
            f_mid,
            jnp.where(valid_mid & (~go_left) & same_side, 0.5 * f_hi_c, f_hi_c),
        )
        last_side_new = jnp.where(valid_mid, cur_side, last_side)

        abs_f_mid = jnp.abs(f_mid)
        better = valid_mid & (abs_f_mid <= abs_f_best)
        alpha_best_new = jnp.where(better, alpha, alpha_best)
        f_best_new = jnp.where(better, f_mid, f_best)
        abs_f_best_new = jnp.where(better, abs_f_mid, abs_f_best)
        payload_best_new = jax.tree_util.tree_map(
            lambda new_v, old_v: jnp.where(better, new_v, old_v),
            payload_mid,
            payload_best,
        )

        done_new = (~valid_mid) | (abs_f_mid <= tol_arr)
        return (
            a_lo_new,
            a_hi_new,
            f_lo_new,
            f_hi_new,
            alpha_best_new,
            f_best_new,
            abs_f_best_new,
            payload_best_new,
            last_side_new,
            it + jnp.int32(1),
            done_new,
        )

    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    _, _, _, _, alpha_fin, f_fin, abs_f_fin, payload_fin, _, it_fin, _ = final_state
    return SafeguardedRootResult(
        alpha=alpha_fin,
        f_val=f_fin,
        payload=payload_fin,
        iterations=it_fin,
        converged=jnp.isfinite(f_fin) & (abs_f_fin <= tol_arr),
    )
