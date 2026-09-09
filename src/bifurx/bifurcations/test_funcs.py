from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from scipy.optimize import brentq

from bifurx.problem import BifurcationProblem


def test_func_fold(tangent: Array) -> float:
    """Limit Point / Fold test function: parameter component of tangent vector $\\dot{p}$."""
    return float(tangent[-1])


def test_func_branch_point(Ju: Array) -> float:
    """Branch point test function: determinant of Jacobian $\\det(\\nabla_u F)$."""
    sign, logdet = jnp.linalg.slogdet(Ju)
    # Clip logdet to avoid overflow/underflow while preserving sign
    clipped_log = jnp.clip(logdet, -80.0, 80.0)
    return float(sign * jnp.exp(clipped_log))


def bialternate_matrix(A: Array) -> Array:
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
    B = np.zeros((m, m), dtype=np.float64)
    A_np = np.asarray(A, dtype=np.float64)

    for i, (p, q) in enumerate(pairs):
        for j, (r, s) in enumerate(pairs):
            val = 0.0
            if p == r and q == s:
                val = A_np[p, p] + A_np[q, q]
            elif p == r and q != s:
                val = A_np[q, s]
            elif q == s and p != r:
                val = A_np[p, r]
            elif p == s:
                val = -A_np[q, r]
            elif q == r:
                val = -A_np[p, s]
            B[i, j] = val

    return jnp.asarray(B, dtype=A.dtype)


def test_func_hopf(Ju: Array) -> float:
    """Hopf bifurcation test function.

    Returns the real part of the leading complex-conjugate eigenvalue pair with non-negligible
    imaginary component (|Im| > 1e-4). Returns NaN if all eigenvalues are purely real.
    """
    Ju_np = np.asarray(Ju)
    eigs = np.linalg.eigvals(Ju_np)
    complex_mask = np.abs(np.imag(eigs)) > 1e-4
    if np.any(complex_mask):
        return float(np.max(np.real(eigs[complex_mask])))
    return float("nan")


def refine_fold_moore_spence(
    problem: BifurcationProblem,
    u_approx: Array,
    p_approx: float,
    tol: float = 1e-11,
    max_iters: int = 15,
) -> tuple[Array, float]:
    """Pinpoint exact Fold (Limit Point) coordinates using the Moore-Spence augmented system:

    F(u, p) = 0
    \\nabla_u F(u, p) * v = 0
    l^T * v - 1 = 0

    where v is the null vector of $\\nabla_u F$, and l is a fixed normalization vector.
    """
    n = problem.dim
    Ju0 = np.asarray(problem.jacobian_u(u_approx, p_approx))
    _, _, vh = np.linalg.svd(Ju0)
    v0 = vh[-1, :]
    l_norm = v0 / np.linalg.norm(v0)

    # Unknowns: x = [u, p, v] of length 2n + 1
    x = np.concatenate([np.asarray(u_approx).ravel(), [p_approx], v0])

    def ms_res(x_vec: Array) -> Array:
        u_v = x_vec[:n]
        p_v = x_vec[n]
        v_v = x_vec[n + 1 :]

        f_val = problem.residual(u_v, p_v)
        Ju_v = problem.jacobian_u(u_v, p_v)
        v_res = Ju_v @ v_v
        norm_res = jnp.dot(jnp.asarray(l_norm), v_v) - 1.0
        return jnp.concatenate([f_val, v_res, jnp.atleast_1d(norm_res)])

    jac_fn = jax.jit(jax.jacobian(ms_res))

    for _ in range(max_iters):
        res = np.asarray(ms_res(jnp.asarray(x)))
        if np.linalg.norm(res) < tol:
            break
        J_ms = np.asarray(jac_fn(jnp.asarray(x)))
        try:
            dx = -np.linalg.solve(J_ms, res)
            x += dx
        except np.linalg.LinAlgError:
            break

    u_fold = jnp.asarray(x[:n])
    p_fold = float(x[n])
    return u_fold, p_fold


def refine_bifurcation_point(
    problem: BifurcationProblem,
    u_a: Array,
    p_a: float,
    tau_a: Array,
    u_b: Array,
    p_b: float,
    tau_b: Array,
    bif_type: str,
    tol: float = 1e-10,
) -> tuple[Array, float]:
    """Pinpoint exact bifurcation coordinates along a continuation step.

    Uses Moore-Spence for Fold (LP), and bisection/Brent root finding on test functions
    for Branch Point (BP) and Hopf (HB).
    """
    if bif_type == "LP":
        try:
            u_fold, p_fold = refine_fold_moore_spence(problem, u_b, p_b, tol=tol)
            return u_fold, p_fold
        except Exception:
            pass

    u_a_np = np.asarray(u_a)
    u_b_np = np.asarray(u_b)

    def test_val_at(alpha: float) -> float:
        u_mid = (1.0 - alpha) * u_a_np + alpha * u_b_np
        p_mid = (1.0 - alpha) * p_a + alpha * p_b
        Ju_mid = problem.jacobian_u(jnp.asarray(u_mid), p_mid)

        if bif_type == "BP":
            return test_func_branch_point(Ju_mid)
        if bif_type == "HB":
            return test_func_hopf(Ju_mid)
        if bif_type == "LP":
            tau_mid = (1.0 - alpha) * np.asarray(tau_a) + alpha * np.asarray(tau_b)
            return test_func_fold(jnp.asarray(tau_mid))
        return 0.0

    fa = test_val_at(0.0)
    fb = test_val_at(1.0)

    if fa * fb <= 0:
        try:
            brent_res = brentq(test_val_at, 0.0, 1.0, xtol=tol)
            alpha_opt = (
                float(brent_res) if not isinstance(brent_res, tuple) else float(brent_res[0])
            )
        except ValueError:
            alpha_opt = 0.5
    else:
        alpha_opt = 0.5

    u_opt = (1.0 - alpha_opt) * u_a_np + alpha_opt * u_b_np
    p_opt = (1.0 - alpha_opt) * p_a + alpha_opt * p_b

    # Polish by projecting onto F(u, p) = 0 via bordered Newton
    from bifurx.solvers.newton import bordered_newton_solve

    tau_opt = (1.0 - alpha_opt) * np.asarray(tau_a) + alpha_opt * np.asarray(tau_b)
    tau_norm = float(np.linalg.norm(tau_opt))
    if tau_norm > 1e-12:
        tau_opt = tau_opt / tau_norm

    res_pol = bordered_newton_solve(
        problem=problem,
        u_init=jnp.asarray(u_opt),
        p_init=p_opt,
        u_prev=jnp.asarray(u_opt),
        p_prev=p_opt,
        tau_prev=jnp.asarray(tau_opt),
        ds=0.0,
        tol=tol,
        max_iters=5,
    )
    if res_pol.converged and jnp.all(jnp.isfinite(res_pol.u)) and jnp.isfinite(res_pol.p):
        return res_pol.u, res_pol.p

    return jnp.asarray(u_opt), float(p_opt)
