from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from bifurx.problem import BifurcationProblem
from bifurx.solvers.newton import bordered_newton_solve


def test_func_fold(tangent: jax.Array) -> jax.Array:
    """Limit Point / Fold test function: parameter component of tangent vector $\\dot{p}$."""
    return tangent[-1]


def test_func_branch_point(Ju: jax.Array) -> jax.Array:
    """Branch point test function: determinant of Jacobian $\\det(\\nabla_u F)$."""
    sign, logdet = jnp.linalg.slogdet(Ju)
    clipped_log = jnp.clip(logdet, -80.0, 80.0)
    return sign * jnp.exp(clipped_log)


def bialternate_matrix(A: jax.Array) -> jax.Array:
    """Compute the bialternate matrix product $2 A \\odot I$ in pure vectorized JAX.

    For an n x n matrix A, produces an m x m matrix where m = n * (n - 1) / 2.
    Its eigenvalues are $\\lambda_i + \\lambda_j$ for all $0 \\le i < j < n$.
    A determinant of zero indicates a pair of eigenvalues summing to zero (e.g., $\\pm i \\omega$).
    """
    n = A.shape[0]
    if n < 2:
        return jnp.zeros((1, 1), dtype=A.dtype)

    p, q = jnp.triu_indices(n, k=1)
    r, s = p, q

    pi, qi = p[:, None], q[:, None]
    rj, sj = r[None, :], s[None, :]

    diag_term = jnp.where((pi == rj) & (qi == sj), A[p, p][:, None] + A[q, q][:, None], 0.0)
    pr_term = jnp.where((pi == rj) & (qi != sj), A[qi, sj], 0.0)
    qs_term = jnp.where((qi == sj) & (pi != rj), A[pi, rj], 0.0)
    ps_term = jnp.where(pi == sj, -A[qi, rj], 0.0)
    qr_term = jnp.where(qi == rj, -A[pi, sj], 0.0)

    return diag_term + pr_term + qs_term + ps_term + qr_term


def test_func_hopf(Ju: jax.Array, eigs: jax.Array | None = None) -> jax.Array:
    """Hopf bifurcation test function.

    Returns the real part of the leading complex-conjugate eigenvalue pair with non-negligible
    imaginary component (|Im| > 1e-4). Returns NaN if all eigenvalues are purely real.
    """
    eigvals = jnp.linalg.eigvals(Ju) if eigs is None else eigs
    complex_mask = jnp.abs(jnp.imag(eigvals)) > 1e-4
    leading = jnp.max(jnp.where(complex_mask, jnp.real(eigvals), -jnp.inf))
    return jnp.where(jnp.any(complex_mask), leading, jnp.nan)


def evaluate_point_diagnostics(
    Ju: jax.Array, tangent: jax.Array
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Compute all codim-1 test functions (LP, BP, HB) and Hurwitz stability from a single eigensolve."""
    eigs = jnp.linalg.eigvals(Ju)
    is_stable = jnp.all(jnp.real(eigs) < 0.0)
    test_funcs = {
        "LP": test_func_fold(tangent),
        "BP": test_func_branch_point(Ju),
        "HB": test_func_hopf(Ju, eigs=eigs),
    }
    return test_funcs, is_stable


def initial_null_vector(Ju: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Extract normalized right singular vector corresponding to smallest singular value of Ju."""
    _, _, vh = jnp.linalg.svd(Ju)
    v0 = vh[-1, :]
    norm_v0 = jnp.linalg.norm(v0)
    l_norm = jnp.where(norm_v0 > 1e-30, v0 / norm_v0, v0)
    return v0, l_norm


def moore_spence_fold_residual(
    f_val: jax.Array,
    Ju: jax.Array,
    v: jax.Array,
    l_norm: jax.Array,
) -> jax.Array:
    """Evaluate Moore-Spence augmented fold system [F(u, p), Ju @ v, <l_norm, v> - 1]."""
    v_res = Ju @ v
    norm_res = jnp.dot(l_norm, v) - 1.0
    return jnp.concatenate([f_val, v_res, jnp.atleast_1d(norm_res)])


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
        return ((hi_s - lo_s) > tol) & (i_s < max_iters)

    def body(
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        lo_s, hi_s, i_s = state
        mid = 0.5 * (lo_s + hi_s)
        go_left = (f(lo_s) * f(mid)) <= 0.0
        new_lo = jnp.where(go_left, lo_s, mid)
        new_hi = jnp.where(go_left, mid, hi_s)
        return new_lo, new_hi, i_s + 1

    lo_f, hi_f, _ = jax.lax.while_loop(cond, body, (lo, hi, jnp.int32(0)))
    return 0.5 * (lo_f + hi_f)


def refine_fold_moore_spence(
    problem: BifurcationProblem,
    u_approx: jax.Array,
    p_approx: float | jax.Array,
    tol: float = 1e-11,
    max_iters: int = 15,
) -> tuple[jax.Array, jax.Array]:
    """Pinpoint exact Fold (Limit Point) coordinates using the Moore-Spence augmented system:

    F(u, p) = 0
    \\nabla_u F(u, p) * v = 0
    l^T * v - 1 = 0

    where v is the null vector of $\\nabla_u F$, and l is a fixed normalization vector.
    """
    n = problem.dim
    u_0 = jnp.asarray(u_approx, dtype=jnp.float64).ravel()
    p_0 = jnp.squeeze(jnp.asarray(p_approx, dtype=jnp.float64))
    Ju0 = problem.jacobian_u(u_0, p_0)
    v0, l_norm = initial_null_vector(Ju0)

    x0 = jnp.concatenate([u_0, jnp.atleast_1d(p_0), v0])

    def ms_res(x_vec: jax.Array) -> jax.Array:
        u_v = x_vec[:n]
        p_v = x_vec[n]
        v_v = x_vec[n + 1 :]
        f_val = problem.residual(u_v, p_v)
        Ju_v = problem.jacobian_u(u_v, p_v)
        return moore_spence_fold_residual(f_val, Ju_v, v_v, l_norm)

    jac_fn = jax.jacobian(ms_res)

    def cond(state: tuple[jax.Array, jax.Array]) -> jax.Array:
        x_cur, i = state
        return (jnp.linalg.norm(ms_res(x_cur)) >= tol) & (i < max_iters)

    def body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x_cur, i = state
        res = ms_res(x_cur)
        J_ms = jac_fn(x_cur)
        dx = -jnp.linalg.solve(J_ms, res)
        is_ok = jnp.all(jnp.isfinite(dx))
        dx_safe = jnp.where(is_ok, dx, jnp.zeros_like(dx))
        i_next = jnp.where(is_ok, i + 1, jnp.int32(max_iters))
        return x_cur + dx_safe, i_next

    x, _ = jax.lax.while_loop(cond, body, (x0, jnp.int32(0)))
    return x[:n], jnp.squeeze(x[n])


def refine_bifurcation_point(
    problem: BifurcationProblem,
    u_a: jax.Array,
    p_a: float | jax.Array,
    tau_a: jax.Array,
    u_b: jax.Array,
    p_b: float | jax.Array,
    tau_b: jax.Array,
    bif_type: str,
    tol: float = 1e-10,
) -> tuple[jax.Array, jax.Array]:
    """Pinpoint exact bifurcation coordinates along a continuation step in pure JAX.

    Uses Moore-Spence for Fold (LP) with bisection + bordered-Newton fallback, and
    ``jax.lax.while_loop`` bisection on test functions + bordered-Newton polish for
    Branch Point (BP) and Hopf (HB).
    """
    u_a_j = jnp.asarray(u_a, dtype=jnp.float64)
    u_b_j = jnp.asarray(u_b, dtype=jnp.float64)
    tau_a_j = jnp.asarray(tau_a, dtype=jnp.float64)
    tau_b_j = jnp.asarray(tau_b, dtype=jnp.float64)
    p_a_j = jnp.squeeze(jnp.asarray(p_a, dtype=jnp.float64))
    p_b_j = jnp.squeeze(jnp.asarray(p_b, dtype=jnp.float64))

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

    has_bracket = jnp.isfinite(fa) & jnp.isfinite(fb) & ((fa * fb) <= 0.0)
    alpha_bisect = _bisect_root(test_val_at, alpha_lo, alpha_hi, tol=tol)
    alpha_opt = jnp.where(has_bracket, alpha_bisect, jnp.array(0.5, dtype=jnp.float64))

    u_opt = (1.0 - alpha_opt) * u_a_j + alpha_opt * u_b_j
    p_opt = (1.0 - alpha_opt) * p_a_j + alpha_opt * p_b_j

    tau_opt = (1.0 - alpha_opt) * tau_a_j + alpha_opt * tau_b_j
    tau_norm = jnp.linalg.norm(tau_opt)
    tau_opt = jnp.where(tau_norm > 1e-12, tau_opt / tau_norm, tau_opt)

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
    pol_ok = res_pol.converged & jnp.all(jnp.isfinite(res_pol.u)) & jnp.isfinite(res_pol.p)
    u_pol = jnp.where(pol_ok, res_pol.u, u_opt)
    p_pol = jnp.where(pol_ok, res_pol.p, p_opt)

    if bif_type == "LP":
        u_fold, p_fold = refine_fold_moore_spence(problem, u_b_j, p_b_j, tol=tol)
        ms_res_norm = jnp.linalg.norm(problem.residual(u_fold, p_fold))
        ms_ok = jnp.all(jnp.isfinite(u_fold)) & jnp.isfinite(p_fold) & (ms_res_norm < 1e-6)
        return jnp.where(ms_ok, u_fold, u_pol), jnp.where(ms_ok, p_fold, p_pol)

    return u_pol, p_pol
