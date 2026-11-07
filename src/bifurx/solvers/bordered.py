from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class BorderedSolution(NamedTuple):
    """Solution to a bordered linear system.

    Supports unpacking: `x, y = sol`.
    """

    x: jax.Array
    y: jax.Array

    @property
    def z(self) -> jax.Array:
        """Concatenated solution vector [x, y]."""
        return jnp.concatenate([self.x.ravel(), jnp.atleast_1d(self.y)])


def build_bordered_matrix(
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: float | jax.Array,
) -> jax.Array:
    """Construct full (n+1) x (n+1) bordered matrix M = [[A, b], [c^T, d]]."""
    return jnp.block([[A, b.reshape(-1, 1)], [c.reshape(1, -1), jnp.atleast_2d(d)]])


def _block_lu_components(
    A: jax.Array,
    b_flat: jax.Array,
    c_flat: jax.Array,
    d_scalar: jax.Array,
    f_flat: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rhs = jnp.stack([b_flat, f_flat], axis=-1)
    sol_vw = jnp.linalg.solve(A, rhs)
    v, w = sol_vw[:, 0], sol_vw[:, 1]
    delta = d_scalar - jnp.dot(c_flat, v)
    return v, w, delta


def solve_bordered_block(
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: float | jax.Array,
    f: jax.Array,
    g: float | jax.Array,
) -> BorderedSolution:
    """Solve bordered linear system using Block LU / Sherman-Morrison-Woodbury elimination."""
    c_flat = c.ravel()
    v, w, delta = _block_lu_components(A, b.ravel(), c_flat, jnp.squeeze(d), f.ravel())
    y = (jnp.squeeze(g) - jnp.dot(c_flat, w)) / delta
    return BorderedSolution(x=w - y * v, y=jnp.squeeze(y))


def solve_bordered_direct(
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: float | jax.Array,
    f: jax.Array,
    g: float | jax.Array,
    rcond: float = 1e-12,
) -> BorderedSolution:
    """Solve bordered linear system via direct factorization or SVD/least-squares fallback."""
    n = A.shape[0]
    M = build_bordered_matrix(A, b, c, d)
    r = jnp.concatenate([f.ravel(), jnp.atleast_1d(g)])

    s = jnp.linalg.svdvals(M)
    is_invertible = (s[-1] / (s[0] + 1e-30)) > rcond
    z = jax.lax.cond(
        is_invertible,
        lambda: jnp.linalg.solve(M, r),
        lambda: jnp.linalg.lstsq(M, r, rcond=rcond)[0],
    )
    return BorderedSolution(x=z[:n], y=jnp.squeeze(z[n]))


def solve_bordered_system(
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: float | jax.Array,
    f: jax.Array,
    g: float | jax.Array,
    method: str = "auto",
    rcond: float = 1e-10,
) -> BorderedSolution:
    """Solve bordered linear system [[A, b], [c^T, d]] [x, y]^T = [f, g]^T."""
    b_flat, c_flat, f_flat = b.ravel(), c.ravel(), f.ravel()
    d_scalar, g_scalar = jnp.squeeze(d), jnp.squeeze(g)

    if method == "block":
        return solve_bordered_block(A, b_flat, c_flat, d_scalar, f_flat, g_scalar)

    if method == "direct":
        return solve_bordered_direct(A, b_flat, c_flat, d_scalar, f_flat, g_scalar, rcond=rcond)

    if method == "auto":
        v, w, delta = _block_lu_components(A, b_flat, c_flat, d_scalar, f_flat)
        is_valid = (
            jnp.all(jnp.isfinite(v))
            & jnp.all(jnp.isfinite(w))
            & (jnp.abs(delta) > rcond * (1.0 + jnp.abs(d_scalar)))
        )
        delta_safe = jnp.where(is_valid, delta, 1.0)
        v_safe = jnp.where(is_valid, v, 0.0)
        w_safe = jnp.where(is_valid, w, 0.0)

        def _block_branch() -> tuple[jax.Array, jax.Array]:
            y_val = (g_scalar - jnp.dot(c_flat, w_safe)) / delta_safe
            return w_safe - y_val * v_safe, jnp.squeeze(y_val)

        def _direct_branch() -> tuple[jax.Array, jax.Array]:
            sol = solve_bordered_direct(A, b_flat, c_flat, d_scalar, f_flat, g_scalar, rcond=rcond)
            return sol.x, sol.y

        x_res, y_res = jax.lax.cond(is_valid, _block_branch, _direct_branch)
        return BorderedSolution(x=x_res, y=y_res)

    msg = f"Unknown method: {method}. Choose from 'auto', 'block', or 'direct'."
    raise ValueError(msg)
