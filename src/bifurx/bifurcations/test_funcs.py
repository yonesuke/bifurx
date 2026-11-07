from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from bifurx.problem import BifurcationProblem
from bifurx.solvers.bordered import build_bordered_matrix
from bifurx.solvers.newton import bordered_newton_solve


def _test_func_branch_point(
    Ju: jax.Array,
    Jp: jax.Array | None = None,
    tangent: jax.Array | None = None,
) -> jax.Array:
    """Branch point test function: determinant of the augmented continuation Jacobian."""
    n = Ju.shape[0]
    b = jnp.zeros(n, dtype=Ju.dtype) if Jp is None else jnp.reshape(Jp, (n,))
    c = jnp.zeros(n, dtype=Ju.dtype) if tangent is None else jnp.reshape(tangent[:n], (n,))
    d = jnp.array(1.0, dtype=Ju.dtype) if tangent is None else jnp.squeeze(tangent[-1])
    return jnp.linalg.det(build_bordered_matrix(Ju, b, c, d))


def extract_leading_hopf_pair(
    Ju: jax.Array,
    eigs: jax.Array | None = None,
    rel_tol: float = 1e-4,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Extract leading complex-conjugate eigenvalue pair for Hopf bifurcation in pure JAX.

    Returns: (has_hopf_pair, leading_real, leading_imag)
    """
    eigvals = jnp.linalg.eigvals(Ju) if eigs is None else eigs
    complex_mask = jnp.abs(jnp.imag(eigvals)) > rel_tol * (1.0 + jnp.abs(eigvals))
    has_hopf_pair = jnp.any(complex_mask)

    masked_real = jnp.where(complex_mask, jnp.real(eigvals), -jnp.inf)
    idx = jnp.argmax(masked_real)
    leading_real = jnp.where(has_hopf_pair, jnp.real(eigvals[idx]), jnp.nan)
    leading_imag = jnp.where(has_hopf_pair, jnp.abs(jnp.imag(eigvals[idx])), jnp.nan)
    return has_hopf_pair, leading_real, leading_imag


def test_func_hopf(
    Ju: jax.Array,
    eigs: jax.Array | None = None,
    rel_tol: float = 1e-4,
) -> jax.Array:
    """Hopf bifurcation test function: real part of the leading complex-conjugate eigenvalue pair."""
    has_hopf_pair, leading_real, _ = extract_leading_hopf_pair(Ju, eigs=eigs, rel_tol=rel_tol)
    return jnp.where(has_hopf_pair, leading_real, jnp.nan)


def evaluate_point_diagnostics(
    Ju: jax.Array,
    tangent: jax.Array,
    Jp: jax.Array | None = None,
) -> tuple[dict[str, jax.Array], jax.Array]:
    """Compute all codim-1 test functions (LP, BP, HB), stability, and diagnostics in pure JAX."""
    eigs = jnp.linalg.eigvals(Ju)
    real_parts = jnp.real(eigs)
    is_stable = jnp.all(real_parts < 0.0)
    test_funcs = {
        "LP": tangent[-1],
        "BP": _test_func_branch_point(Ju, Jp=Jp, tangent=tangent),
        "HB": test_func_hopf(Ju, eigs=eigs),
        "st_idx": jnp.sum(real_parts > 0.0),
        "min_eig": jnp.min(jnp.abs(eigs)),
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
    return jnp.concatenate([f_val, Ju @ v, jnp.atleast_1d(jnp.dot(l_norm, v) - 1.0)])


def _bisect_root(
    f: Callable[[jax.Array], jax.Array],
    lo: jax.Array,
    hi: jax.Array,
    tol: float,
    max_iters: int = 100,
) -> jax.Array:
    """Find a scalar root of ``f`` on a bracketing interval ``[lo, hi]`` by bisection in pure JAX."""

    def cond(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        lo_s, hi_s, i_s = state
        return ((hi_s - lo_s) > tol) & (i_s < max_iters) & jnp.isfinite(lo_s) & jnp.isfinite(hi_s)

    def body(
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        lo_s, hi_s, i_s = state
        mid = 0.5 * (lo_s + hi_s)
        f_mid, f_lo = f(mid), f(lo_s)
        go_left = jnp.isfinite(f_mid) & jnp.isfinite(f_lo) & ((f_lo * f_mid) <= 0.0)
        return jnp.where(go_left, lo_s, mid), jnp.where(go_left, mid, hi_s), i_s + 1

    lo_f, hi_f, _ = jax.lax.while_loop(cond, body, (lo, hi, jnp.int32(0)))
    return 0.5 * (lo_f + hi_f)


def _ternary_search_min(
    f: Callable[[jax.Array], jax.Array],
    lo: jax.Array,
    hi: jax.Array,
    tol: float = 1e-8,
    max_iters: int = 60,
) -> jax.Array:
    """Find the minimizer of scalar function f on [lo, hi] by ternary search in pure JAX."""

    def cond(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        lo_s, hi_s, i_s = state
        return ((hi_s - lo_s) > tol) & (i_s < max_iters) & jnp.isfinite(lo_s) & jnp.isfinite(hi_s)

    def body(
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        lo_s, hi_s, i_s = state
        span = hi_s - lo_s
        m1, m2 = lo_s + span / 3.0, hi_s - span / 3.0
        go_left = f(m1) < f(m2)
        return jnp.where(go_left, lo_s, m1), jnp.where(go_left, m2, hi_s), i_s + 1

    lo_f, hi_f, _ = jax.lax.while_loop(cond, body, (lo, hi, jnp.int32(0)))
    return 0.5 * (lo_f + hi_f)


def _refine_fold_moore_spence(
    problem: BifurcationProblem,
    u_approx: jax.Array,
    p_approx: float | jax.Array,
    v_init: jax.Array | None = None,
    tol: float = 1e-11,
    max_iters: int = 15,
) -> tuple[jax.Array, jax.Array]:
    """Pinpoint exact Fold (Limit Point) coordinates using the Moore-Spence augmented system."""
    n = problem.dim
    u_0 = jnp.asarray(u_approx, dtype=jnp.float64).ravel()
    p_0 = jnp.squeeze(jnp.asarray(p_approx, dtype=jnp.float64))
    Ju0 = problem.jacobian_u(u_0, p_0)
    if v_init is None:
        v0, l_norm = initial_null_vector(Ju0)
    else:
        v_norm = jnp.linalg.norm(v_init)
        v_cand = jnp.where(v_norm > 1e-30, v_init / v_norm, v_init)
        v_svd, _ = initial_null_vector(Ju0)
        v0 = jnp.where(v_norm > 1e-12, v_cand, v_svd)
        l_norm = v0

    x0 = jnp.concatenate([u_0, jnp.atleast_1d(p_0), v0])

    def ms_res(x_vec: jax.Array) -> jax.Array:
        u_v, p_v, v_v = x_vec[:n], x_vec[n], x_vec[n + 1 :]
        return moore_spence_fold_residual(
            problem.residual(u_v, p_v), problem.jacobian_u(u_v, p_v), v_v, l_norm
        )

    jac_fn = jax.jacobian(ms_res)

    def cond(state: tuple[jax.Array, jax.Array]) -> jax.Array:
        x_cur, i = state
        return (jnp.linalg.norm(ms_res(x_cur)) >= tol) & (i < max_iters)

    def body(state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        x_cur, i = state
        dx = -jnp.linalg.solve(jac_fn(x_cur), ms_res(x_cur))
        is_ok = jnp.all(jnp.isfinite(dx))
        return (
            x_cur + jnp.where(is_ok, dx, jnp.zeros_like(dx)),
            jnp.where(is_ok, i + 1, jnp.int32(max_iters)),
        )

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
    """Pinpoint exact bifurcation coordinates along a continuation step in pure JAX."""
    u_a_j = jnp.asarray(u_a, dtype=jnp.float64)
    u_b_j = jnp.asarray(u_b, dtype=jnp.float64)
    tau_a_j = jnp.asarray(tau_a, dtype=jnp.float64)
    tau_b_j = jnp.asarray(tau_b, dtype=jnp.float64)
    p_a_j = jnp.squeeze(jnp.asarray(p_a, dtype=jnp.float64))
    p_b_j = jnp.squeeze(jnp.asarray(p_b, dtype=jnp.float64))

    def test_val_at(alpha: jax.Array) -> jax.Array:
        u_mid = (1.0 - alpha) * u_a_j + alpha * u_b_j
        p_mid = (1.0 - alpha) * p_a_j + alpha * p_b_j
        tau_mid = (1.0 - alpha) * tau_a_j + alpha * tau_b_j
        if bif_type == "LP":
            return tau_mid[-1]
        Ju_mid = problem.jacobian_u(u_mid, p_mid)
        if bif_type == "BP":
            Jp_mid = problem.jacobian_p(u_mid, p_mid)
            tau_norm = jnp.linalg.norm(tau_mid)
            tau_mid = jnp.where(tau_norm > 1e-12, tau_mid / tau_norm, tau_mid)
            return _test_func_branch_point(Ju_mid, Jp=Jp_mid, tangent=tau_mid)
        return test_func_hopf(Ju_mid)

    alpha_lo = jnp.array(0.0, dtype=jnp.float64)
    alpha_hi = jnp.array(1.0, dtype=jnp.float64)
    fa, fb = test_val_at(alpha_lo), test_val_at(alpha_hi)

    has_bracket = jnp.isfinite(fa) & jnp.isfinite(fb) & ((fa * fb) <= 0.0)
    alpha_bisect = _bisect_root(test_val_at, alpha_lo, alpha_hi, tol=tol)
    alpha_min = _ternary_search_min(lambda a: jnp.abs(test_val_at(a)), alpha_lo, alpha_hi, tol=tol)
    alpha_opt = jnp.where(
        has_bracket & jnp.isfinite(alpha_bisect),
        alpha_bisect,
        jnp.where(jnp.isfinite(alpha_min), alpha_min, jnp.array(0.5, dtype=jnp.float64)),
    )

    u_mid = 0.5 * (u_a_j + u_b_j)
    p_mid = 0.5 * (p_a_j + p_b_j)

    u_opt = (1.0 - alpha_opt) * u_a_j + alpha_opt * u_b_j
    p_opt = (1.0 - alpha_opt) * p_a_j + alpha_opt * p_b_j
    u_opt = jnp.where(jnp.all(jnp.isfinite(u_opt)), u_opt, u_mid)
    p_opt = jnp.where(jnp.isfinite(p_opt), p_opt, p_mid)

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
        u_ms, p_ms = _refine_fold_moore_spence(
            problem, u_opt, p_opt, v_init=tau_opt[: problem.dim], tol=tol
        )
        res_ms = jnp.linalg.norm(problem.residual(u_ms, p_ms))
        ms_ok = jnp.all(jnp.isfinite(u_ms)) & jnp.isfinite(p_ms) & (res_ms < 1e-6)
        u_res = jnp.where(ms_ok, u_ms, u_opt)
        p_res = jnp.where(ms_ok, p_ms, p_opt)
    else:
        u_res, p_res = u_pol, p_pol

    return (
        jnp.where(jnp.all(jnp.isfinite(u_res)), u_res, u_mid),
        jnp.where(jnp.isfinite(p_res), p_res, p_mid),
    )
