from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from bifurx.homotopy.bordered import factor_bordered_matrix, solve_factored_bordered
from bifurx.homotopy.norm import weighted_distance


def tangent_predict(
    u: jax.Array,
    p: jax.Array,
    tau: jax.Array,
    ds: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """First-order Euler tangent predictor along arclength."""
    n = u.shape[0]
    return u + ds * tau[:n], jnp.squeeze(p + ds * tau[-1])


def estimate_pade_trust_region(
    z1: jax.Array,
    z2: jax.Array,
    z3: jax.Array,
) -> jax.Array:
    """Estimate Padé (2,1) trust region radius from component pole distances."""
    c1, c2, c3 = jnp.abs(z1), jnp.abs(z2), jnp.abs(z3)
    lam = jnp.maximum(1e-6, c1)
    s1 = c1 / lam
    s2 = c2 / (lam**2)
    s3 = c3 / (lam**3)
    tol = 1e-14 * jnp.maximum(s1, jnp.maximum(s2, s3))
    valid = (s2 > tol) & (s3 > tol)
    tau_i = jnp.where(valid, (s2 / jnp.where(valid, s3, 1.0)) / lam, jnp.inf)
    return jnp.min(tau_i)


def compute_arclength_taylor3(
    residual_fn: Callable[[jax.Array, jax.Array], jax.Array],
    u: jax.Array,
    p: jax.Array,
    tau: jax.Array,
    Ju: jax.Array,
    Jp: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute 2nd/3rd arclength Taylor coefficients and Padé trust region.

    Taylor series coefficients `z2 = z''(0)/2`, `z3 = z'''(0)/6` satisfy
    `z(s) = z0 + tau * s + z2 * s^2 + z3 * s^3 + O(s^4)` along the curve
    `residual_fn(u, p) = 0`, `<tau, z(s) - z0> = s`.
    """
    n = u.shape[0]
    z0 = jnp.concatenate([u, jnp.atleast_1d(p)])
    z1 = tau

    def G(z: jax.Array) -> jax.Array:
        return residual_fn(z[:n], jnp.squeeze(z[-1]))

    def g_along_z1(t: jax.Array) -> jax.Array:
        return G(z0 + t * z1)

    def d2g_along_z1(t: jax.Array) -> jax.Array:
        def dg(t_in: jax.Array) -> jax.Array:
            return jax.jvp(g_along_z1, (t_in,), (jnp.array(1.0, dtype=jnp.float64),))[1]

        return jax.jvp(dg, (t,), (jnp.array(1.0, dtype=jnp.float64),))[1]

    d2_11, d3_111 = jax.jvp(
        d2g_along_z1,
        (jnp.array(0.0, dtype=jnp.float64),),
        (jnp.array(1.0, dtype=jnp.float64),),
    )

    fac = factor_bordered_matrix(Ju, Jp, tau[:n], tau[-1])
    z2 = solve_factored_bordered(fac, -0.5 * d2_11, 0.0).z

    d2_12 = jax.jvp(lambda z: jax.jvp(G, (z,), (z1,))[1], (z0,), (z2,))[1]
    rhs3 = -d2_12 - (1.0 / 6.0) * d3_111
    z3 = solve_factored_bordered(fac, rhs3, 0.0).z

    trust_region = estimate_pade_trust_region(z1, z2, z3)
    return z2, z3, trust_region


def pade21_predict(
    u: jax.Array,
    p: jax.Array,
    z1: jax.Array,
    z2: jax.Array,
    z3: jax.Array,
    trust_region: jax.Array,
    ds: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Component-wise Padé (2,1) rational predictor with Taylor fallback near poles/zeros."""
    n = u.shape[0]
    z0 = jnp.concatenate([u, jnp.atleast_1d(p)])
    lam = jnp.where(jnp.isfinite(trust_region), jnp.clip(trust_region, 1e-6, 1e6), 1.0)
    c2_sq = (z2 * (lam**2)) ** 2
    c3_sq = (z3 * (lam**3)) ** 2
    tol_sq = (1e-12**2) * (z0**2 + (z1 * lam) ** 2 + c2_sq + c3_sq)

    z2_safe = jnp.where(jnp.abs(z2) > 1e-30, z2, 1.0)
    denom = 1.0 - ds * (z3 / z2_safe)
    use_taylor = (c3_sq <= tol_sq) | (c2_sq <= tol_sq) | (jnp.abs(denom) < 0.1)

    z_taylor = z0 + ds * (z1 + ds * z2)
    denom_safe = jnp.where(use_taylor, 1.0, denom)
    z_pade = z0 + ds * (z1 + ds * z2 / denom_safe)
    z_pred = jnp.where(use_taylor, z_taylor, z_pade)
    return z_pred[:n], jnp.squeeze(z_pred[-1])


def estimate_local_error(
    u_pred: jax.Array,
    p_pred: jax.Array,
    u_corr: jax.Array,
    p_corr: jax.Array,
    ds: jax.Array,
    order: int,
    weights: jax.Array,
) -> jax.Array:
    """Estimate local predictor error constant ||z_pred - z_corr||_w / |ds|^order."""
    z_pred = jnp.concatenate([u_pred, jnp.atleast_1d(p_pred)])
    z_corr = jnp.concatenate([u_corr, jnp.atleast_1d(p_corr)])
    dist = weighted_distance(z_pred, z_corr, weights)
    ds_abs = jnp.maximum(jnp.abs(ds), 1e-30)
    return dist / (ds_abs**order)
