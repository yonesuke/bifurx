from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

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


class BorderedOperatorSolver(NamedTuple):
    """Pluggable factor-and-solve backend for bordered linear systems."""

    factor: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], Any]
    solve: Callable[[Any, jax.Array, jax.Array], BorderedSolution]


class BlockBorderedFactorization(NamedTuple):
    """Reusable block LU factorization for a bordered system."""

    lu: jax.Array
    piv: jax.Array
    v: jax.Array
    delta: jax.Array
    c: jax.Array
    is_valid: jax.Array


class BorderedFactorization(NamedTuple):
    """Reusable LU factorization (with lstsq fallback matrix) for a bordered system."""

    M: jax.Array
    lu: jax.Array
    piv: jax.Array
    use_lu: jax.Array


def build_bordered_matrix(
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: float | jax.Array,
) -> jax.Array:
    """Construct full (n+1) x (n+1) bordered matrix M = [[A, b], [c^T, d]]."""
    return jnp.block([[A, b.reshape(-1, 1)], [c.reshape(1, -1), jnp.atleast_2d(d)]])


def factor_bordered_matrix(
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: float | jax.Array,
    rcond: float = 1e-12,
) -> BorderedFactorization:
    """Factorize a bordered matrix once for solving multiple right-hand sides."""
    M = build_bordered_matrix(A, b, c, d)
    lu, piv = jax.scipy.linalg.lu_factor(M)
    u_diag = jnp.abs(jnp.diag(lu))
    u_max = jnp.max(u_diag)
    use_lu = jnp.all(jnp.isfinite(lu)) & (jnp.min(u_diag) > rcond * (u_max + 1e-30))
    return BorderedFactorization(M=M, lu=lu, piv=piv, use_lu=use_lu)


def solve_factored_bordered(
    fac: BorderedFactorization,
    f: jax.Array,
    g: float | jax.Array,
    rcond: float = 1e-12,
) -> BorderedSolution:
    """Solve M [x, y]^T = [f, g]^T reusing a `BorderedFactorization`."""
    n = fac.M.shape[0] - 1
    r = jnp.concatenate([f.ravel(), jnp.atleast_1d(g)])
    z = jax.lax.cond(
        fac.use_lu,
        lambda: jax.scipy.linalg.lu_solve((fac.lu, fac.piv), r),
        lambda: jnp.linalg.lstsq(fac.M, r, rcond=rcond)[0],
    )
    return BorderedSolution(x=z[:n], y=jnp.squeeze(z[n]))


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


def factor_bordered_block(
    A: jax.Array,
    b: jax.Array,
    c: jax.Array,
    d: float | jax.Array,
    rcond: float = 1e-10,
) -> BlockBorderedFactorization:
    """Factorize A once for Schur-complement block bordered solves."""
    b_flat, c_flat = b.ravel(), c.ravel()
    d_scalar = jnp.squeeze(jnp.asarray(d, dtype=jnp.float64))
    lu, piv = jax.scipy.linalg.lu_factor(A)
    v = jax.scipy.linalg.lu_solve((lu, piv), b_flat)
    delta = d_scalar - jnp.dot(c_flat, v)
    is_valid = (
        jnp.all(jnp.isfinite(lu))
        & jnp.all(jnp.isfinite(v))
        & (jnp.abs(delta) > rcond * (1.0 + jnp.abs(d_scalar)))
    )
    return BlockBorderedFactorization(lu=lu, piv=piv, v=v, delta=delta, c=c_flat, is_valid=is_valid)


def solve_factored_bordered_block(
    fac: BlockBorderedFactorization,
    f: jax.Array,
    g: float | jax.Array,
) -> BorderedSolution:
    """Solve [[A, b], [c^T, d]] [x, y]^T = [f, g]^T reusing `BlockBorderedFactorization`."""
    w = jax.scipy.linalg.lu_solve((fac.lu, fac.piv), f.ravel())
    delta_safe = jnp.where(jnp.abs(fac.delta) > 1e-30, fac.delta, 1.0)
    y = jnp.where(
        fac.is_valid,
        (jnp.squeeze(jnp.asarray(g, dtype=jnp.float64)) - jnp.dot(fac.c, w)) / delta_safe,
        jnp.nan,
    )
    return BorderedSolution(x=w - y * fac.v, y=jnp.squeeze(y))


def make_gmres_bordered_solver(
    residual_fn: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    tol: float = 1e-10,
    maxiter: int | None = None,
    restart: int = 20,
) -> BorderedOperatorSolver:
    """Build a matrix-free GMRES `BorderedOperatorSolver` backed by forward-mode JVPs."""

    def factor(u: jax.Array, p: jax.Array, c: jax.Array, d: jax.Array) -> tuple[jax.Array, ...]:
        return (
            jnp.asarray(u, dtype=jnp.float64),
            jnp.squeeze(jnp.asarray(p, dtype=jnp.float64)),
            jnp.asarray(c, dtype=jnp.float64).ravel(),
            jnp.squeeze(jnp.asarray(d, dtype=jnp.float64)),
        )

    def solve(fac: tuple[jax.Array, ...], f: jax.Array, g: jax.Array) -> BorderedSolution:
        u_arr, p_arr, c_arr, d_arr = fac
        n = u_arr.shape[0]
        rhs = jnp.concatenate(
            [
                jnp.atleast_1d(jnp.asarray(f, dtype=jnp.float64)).ravel(),
                jnp.atleast_1d(jnp.squeeze(jnp.asarray(g, dtype=jnp.float64))),
            ]
        )

        def matvec(z: jax.Array) -> jax.Array:
            dz_u, dz_p = z[:n], jnp.squeeze(z[n])
            _, jvp_f = jax.jvp(residual_fn, (u_arr, p_arr), (dz_u, dz_p))
            jvp_flat = jnp.atleast_1d(jnp.asarray(jvp_f, dtype=jnp.float64)).ravel()
            arc = jnp.dot(c_arr, dz_u) + d_arr * dz_p
            return jnp.concatenate([jvp_flat, jnp.atleast_1d(arc)])

        b_norm = jnp.linalg.norm(rhs)
        has_rhs = b_norm > 0.0
        rhs_unit = jnp.where(has_rhs, rhs / jnp.where(has_rhs, b_norm, 1.0), rhs)
        z_unit, _ = jax.scipy.sparse.linalg.gmres(
            matvec,
            rhs_unit,
            tol=tol,
            atol=0.0,
            restart=min(restart, n + 1),
            maxiter=maxiter,
        )
        z_sol = jnp.where(has_rhs, b_norm * z_unit, jnp.zeros_like(rhs))
        return BorderedSolution(x=z_sol[:n], y=jnp.squeeze(z_sol[n]))

    return BorderedOperatorSolver(factor=factor, solve=solve)
