from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from bifurx.problem import BifurcationProblem


def test_func_fold(tangent: jax.Array) -> jax.Array:
    """Limit Point / Fold test function: parameter component of tangent vector $\\dot{p}$."""
    return tangent[-1]


def test_func_branch_point(Ju: jax.Array) -> jax.Array:
    """Branch point test function: determinant of Jacobian $\\det(\\nabla_u F)$."""
    sign, logdet = jnp.linalg.slogdet(Ju)
    # Clip logdet to avoid overflow/underflow while preserving sign
    clipped_log = jnp.clip(logdet, -80.0, 80.0)
    return sign * jnp.exp(clipped_log)


def bialternate_matrix(A: jax.Array) -> jax.Array:
    """Compute the bialternate matrix product $2 A \\odot I$.

    For an n x n matrix A, produces an m x m matrix where m = n * (n - 1) / 2.
    Its eigenvalues are $\\lambda_i + \\lambda_j$ for all $0 \\le i < j < n$.
    A determinant of zero indicates a pair of eigenvalues summing to zero (e.g., $\\pm i \\omega$).
    """
    n = A.shape[0]
    if n < 2:
        return jnp.zeros((1, 1), dtype=A.dtype)

    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]
    m = len(pairs)
    B = jnp.zeros((m, m), dtype=A.dtype)

    for i, (p, q) in enumerate(pairs):
        for j, (r, s) in enumerate(pairs):
            if p == r and q == s:
                val = A[p, p] + A[q, q]
            elif p == r and q != s:
                val = A[q, s]
            elif q == s and p != r:
                val = A[p, r]
            elif p == s:
                val = -A[q, r]
            elif q == r:
                val = -A[p, s]
            else:
                val = jnp.array(0.0, dtype=A.dtype)
            B = B.at[i, j].set(val)

    return B


def test_func_hopf(Ju: jax.Array) -> jax.Array:
    """Hopf bifurcation test function.

    Returns the real part of the leading complex-conjugate eigenvalue pair with non-negligible
    imaginary component (|Im| > 1e-4). Returns NaN if all eigenvalues are purely real.
    """
    eigs = jnp.linalg.eigvals(Ju)
    complex_mask = jnp.abs(jnp.imag(eigs)) > 1e-4
    leading = jnp.max(jnp.where(complex_mask, jnp.real(eigs), -jnp.inf))
    return jnp.where(jnp.any(complex_mask), leading, jnp.nan)


def _bisect_root(
    f: Callable[[jax.Array], jax.Array],
    lo: jax.Array,
    hi: jax.Array,
    tol: float,
    max_iters: int = 100,
) -> jax.Array:
    """Find a scalar root of ``f`` on a bracketing interval ``[lo, hi]`` by bisection.

    Runs as a JAX ``lax.while_loop`` so the search is traceable/jittable, and terminates
    once the bracket is narrower than ``tol`` or ``max_iters`` iterations elapse.
    """

    def cond(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        lo_s, hi_s, i_s = state
        return jnp.logical_and((hi_s - lo_s) > tol, i_s < max_iters)

    def body(
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        lo_s, hi_s, i_s = state
        mid = 0.5 * (lo_s + hi_s)
        # Keep the sub-interval that still brackets the sign change.
        go_left = (f(lo_s) * f(mid)) <= 0.0
        new_lo = jnp.where(go_left, lo_s, mid)
        new_hi = jnp.where(go_left, mid, hi_s)
        return new_lo, new_hi, i_s + 1

    lo_f, hi_f, _ = jax.lax.while_loop(cond, body, (lo, hi, jnp.int32(0)))
    return 0.5 * (lo_f + hi_f)


def refine_fold_moore_spence(
    problem: BifurcationProblem,
    u_approx: jax.Array,
    p_approx: float,
    tol: float = 1e-11,
    max_iters: int = 15,
) -> tuple[jax.Array, float]:
    """Pinpoint exact Fold (Limit Point) coordinates using the Moore-Spence augmented system:

    F(u, p) = 0
    \\nabla_u F(u, p) * v = 0
    l^T * v - 1 = 0

    where v is the null vector of $\\nabla_u F$, and l is a fixed normalization vector.
    """
    n = problem.dim
    Ju0 = problem.jacobian_u(u_approx, p_approx)
    _, _, vh = jnp.linalg.svd(Ju0)
    v0 = vh[-1, :]
    l_norm = v0 / jnp.linalg.norm(v0)

    # Unknowns: x = [u, p, v] of length 2n + 1
    x0 = jnp.concatenate(
        [
            jnp.asarray(u_approx, dtype=jnp.float64).ravel(),
            jnp.atleast_1d(jnp.asarray(p_approx, dtype=jnp.float64)),
            v0,
        ]
    )

    def ms_res(x_vec: jax.Array) -> jax.Array:
        u_v = x_vec[:n]
        p_v = x_vec[n]
        v_v = x_vec[n + 1 :]

        f_val = problem.residual(u_v, p_v)
        Ju_v = problem.jacobian_u(u_v, p_v)
        v_res = Ju_v @ v_v
        norm_res = jnp.dot(l_norm, v_v) - 1.0
        return jnp.concatenate([f_val, v_res, jnp.atleast_1d(norm_res)])

    jac_fn = jax.jit(jax.jacobian(ms_res))

    def cond(state: tuple[jax.Array, jax.Array]) -> jax.Array:
        x_cur, i = state
        return jnp.logical_and(jnp.linalg.norm(ms_res(x_cur)) >= tol, i < max_iters)

    def body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x_cur, i = state
        res = ms_res(x_cur)
        J_ms = jac_fn(x_cur)
        # Finite-mask guard replaces the old exception-based singular-matrix guard.
        dx = -jnp.linalg.solve(J_ms, res)
        dx = jnp.where(jnp.all(jnp.isfinite(dx)), dx, jnp.zeros_like(dx))
        return x_cur + dx, i + 1

    x, _ = jax.lax.while_loop(cond, body, (x0, jnp.int32(0)))

    return x[:n], float(x[n])


def refine_bifurcation_point(
    problem: BifurcationProblem,
    u_a: jax.Array,
    p_a: float,
    tau_a: jax.Array,
    u_b: jax.Array,
    p_b: float,
    tau_b: jax.Array,
    bif_type: str,
    tol: float = 1e-10,
) -> tuple[jax.Array, float]:
    """Pinpoint exact bifurcation coordinates along a continuation step.

    Uses Moore-Spence for Fold (LP), and a JAX ``lax.while_loop`` bisection on the
    test functions for Branch Point (BP) and Hopf (HB).
    """
    if bif_type == "LP":
        try:
            u_fold, p_fold = refine_fold_moore_spence(problem, u_b, p_b, tol=tol)
            return u_fold, p_fold
        except Exception:
            pass

    u_a_j = jnp.asarray(u_a, dtype=jnp.float64)
    u_b_j = jnp.asarray(u_b, dtype=jnp.float64)
    tau_a_j = jnp.asarray(tau_a, dtype=jnp.float64)
    tau_b_j = jnp.asarray(tau_b, dtype=jnp.float64)
    p_a_j = jnp.asarray(p_a, dtype=jnp.float64)
    p_b_j = jnp.asarray(p_b, dtype=jnp.float64)

    def test_val_at(alpha: jax.Array) -> jax.Array:
        u_mid = (1.0 - alpha) * u_a_j + alpha * u_b_j
        p_mid = (1.0 - alpha) * p_a_j + alpha * p_b_j
        Ju_mid = problem.jacobian_u(u_mid, p_mid)

        if bif_type == "BP":
            return test_func_branch_point(Ju_mid)
        if bif_type == "HB":
            return test_func_hopf(Ju_mid)
        tau_mid = (1.0 - alpha) * tau_a_j + alpha * tau_b_j
        return test_func_fold(tau_mid)

    alpha_lo = jnp.array(0.0, dtype=jnp.float64)
    alpha_hi = jnp.array(1.0, dtype=jnp.float64)
    fa = test_val_at(alpha_lo)
    fb = test_val_at(alpha_hi)

    if bool(jnp.isfinite(fa)) and bool(jnp.isfinite(fb)) and bool(fa * fb <= 0.0):
        alpha_opt = _bisect_root(test_val_at, alpha_lo, alpha_hi, tol=tol)
    else:
        alpha_opt = jnp.array(0.5, dtype=jnp.float64)

    u_opt = (1.0 - alpha_opt) * u_a_j + alpha_opt * u_b_j
    p_opt = (1.0 - alpha_opt) * p_a_j + alpha_opt * p_b_j

    # Polish by projecting onto F(u, p) = 0 via bordered Newton
    from bifurx.solvers.newton import bordered_newton_solve

    tau_opt = (1.0 - alpha_opt) * tau_a_j + alpha_opt * tau_b_j
    tau_norm = float(jnp.linalg.norm(tau_opt))
    if tau_norm > 1e-12:
        tau_opt = tau_opt / tau_norm

    res_pol = bordered_newton_solve(
        problem=problem,
        u_init=u_opt,
        p_init=p_opt,
        u_prev=u_opt,
        p_prev=p_opt,
        tau_prev=tau_opt,
        ds=0.0,
        tol=tol,
        max_iters=5,
    )
    if res_pol.converged and jnp.all(jnp.isfinite(res_pol.u)) and jnp.isfinite(res_pol.p):
        return res_pol.u, res_pol.p

    return u_opt, float(p_opt)
