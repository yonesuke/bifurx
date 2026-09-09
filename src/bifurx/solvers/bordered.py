from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array


class BorderedSolution(NamedTuple):
    """Solution to a bordered linear system.

    Supports unpacking: `x, y = sol`.
    """

    x: Array
    y: Array

    @property
    def z(self) -> Array:
        """Concatenated solution vector [x, y]."""
        return jnp.concatenate([self.x.ravel(), jnp.atleast_1d(self.y)])


def _build_bordered_matrix(
    A: Array, b: Array, c: Array, d: float | Array, f: Array, g: float | Array
) -> tuple[Array, Array]:
    """Construct full (n+1) x (n+1) matrix M and RHS vector r."""
    b_col = b.reshape(-1, 1)
    c_row = c.reshape(1, -1)
    d_elem = jnp.atleast_2d(d)
    M = jnp.block([[A, b_col], [c_row, d_elem]])
    r = jnp.concatenate([f.ravel(), jnp.atleast_1d(g)])
    return M, r


def solve_bordered_block(
    A: Array,
    b: Array,
    c: Array,
    d: float | Array,
    f: Array,
    g: float | Array,
) -> BorderedSolution:
    """Solve bordered linear system using Block LU / Sherman-Morrison-Woodbury elimination.

    Assumes matrix A is non-singular.
    """
    b_flat = b.ravel()
    f_flat = f.ravel()
    c_flat = c.ravel()

    # Solve A v = b and A w = f in a single batched call
    rhs = jnp.stack([b_flat, f_flat], axis=-1)
    sol_vw = jnp.linalg.solve(A, rhs)
    v = sol_vw[:, 0]
    w = sol_vw[:, 1]

    delta = d - jnp.dot(c_flat, v)
    y = (g - jnp.dot(c_flat, w)) / delta
    x = w - y * v
    return BorderedSolution(x=x, y=jnp.squeeze(y))


def solve_bordered_direct(
    A: Array,
    b: Array,
    c: Array,
    d: float | Array,
    f: Array,
    g: float | Array,
    rcond: float = 1e-12,
) -> BorderedSolution:
    """Solve bordered linear system via direct factorization or SVD/least-squares fallback."""
    n = A.shape[0]
    M, r = _build_bordered_matrix(A, b, c, d, f, g)

    s = jnp.linalg.svdvals(M)
    is_invertible = (s[-1] / (s[0] + 1e-30)) > rcond

    def _solve_inv() -> Array:
        return jnp.linalg.solve(M, r)

    def _solve_lstsq() -> Array:
        return jnp.linalg.lstsq(M, r, rcond=rcond)[0]

    z = jax.lax.cond(is_invertible, _solve_inv, _solve_lstsq)
    return BorderedSolution(x=z[:n], y=jnp.squeeze(z[n]))


def solve_bordered_system(
    A: Array,
    b: Array,
    c: Array,
    d: float | Array,
    f: Array,
    g: float | Array,
    method: str = "auto",
    rcond: float = 1e-10,
) -> BorderedSolution:
    """Solve bordered linear system:

    [[A, b], [c^T, d]] [x, y]^T = [f, g]^T

    Parameters
    ----------
    A : Array of shape (n, n)
        Leading block matrix.
    b : Array of shape (n,) or (n, 1)
        Right border vector.
    c : Array of shape (n,) or (1, n)
        Bottom border vector.
    d : float or Array
        Scalar corner element.
    f : Array of shape (n,)
        Top RHS vector.
    g : float or Array
        Bottom RHS scalar.
    method : str, default="auto"
        Solver method: "block" (Block LU), "direct" (full bordered solve), or "auto"
        (uses Block LU when A is non-singular, and fallback to direct when A is singular).
    rcond : float, default=1e-10
        Threshold for reciprocal condition number.

    Returns
    -------
    BorderedSolution
        Named tuple (x, y) containing the solution components.
    """
    n = A.shape[0]
    b_flat = b.ravel()
    c_flat = c.ravel()
    f_flat = f.ravel()
    d_scalar = jnp.squeeze(d)
    g_scalar = jnp.squeeze(g)

    if method == "block":
        return solve_bordered_block(A, b_flat, c_flat, d_scalar, f_flat, g_scalar)

    if method == "direct":
        return solve_bordered_direct(A, b_flat, c_flat, d_scalar, f_flat, g_scalar, rcond=rcond)

    if method == "auto":
        s = jnp.linalg.svdvals(A)
        rcond_A = s[-1] / (s[0] + 1e-30)
        is_A_good = rcond_A > rcond

        # Safe A to prevent NaNs in non-executed branch under autodiff/JIT
        A_safe = jnp.where(is_A_good, A, jnp.eye(n, dtype=A.dtype))

        def _block_branch() -> tuple[Array, Array]:
            rhs = jnp.stack([b_flat, f_flat], axis=-1)
            sol_vw = jnp.linalg.solve(A_safe, rhs)
            v = sol_vw[:, 0]
            w = sol_vw[:, 1]
            delta = d_scalar - jnp.dot(c_flat, v)
            # Check if delta is non-zero
            is_delta_good = jnp.abs(delta) > rcond

            def _compute_block() -> tuple[Array, Array]:
                y_val = (g_scalar - jnp.dot(c_flat, w)) / delta
                x_val = w - y_val * v
                return x_val, jnp.squeeze(y_val)

            def _compute_fallback_direct() -> tuple[Array, Array]:
                sol = solve_bordered_direct(
                    A, b_flat, c_flat, d_scalar, f_flat, g_scalar, rcond=rcond
                )
                return sol.x, sol.y

            return jax.lax.cond(is_delta_good, _compute_block, _compute_fallback_direct)

        def _direct_branch() -> tuple[Array, Array]:
            sol = solve_bordered_direct(A, b_flat, c_flat, d_scalar, f_flat, g_scalar, rcond=rcond)
            return sol.x, sol.y

        x_res, y_res = jax.lax.cond(is_A_good, _block_branch, _direct_branch)
        return BorderedSolution(x=x_res, y=y_res)

    msg = f"Unknown method: {method}. Choose from 'auto', 'block', or 'direct'."
    raise ValueError(msg)
