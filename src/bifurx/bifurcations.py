from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.homotopy.bordered import build_bordered_matrix, solve_bordered_system
from bifurx.homotopy.corrector import correct_arclength_step
from bifurx.problem import BifurcationProblem


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


def canonical_hopf_eigenvector(
    Ju: jax.Array,
    eigs: jax.Array | None = None,
    vecs: jax.Array | None = None,
    rel_tol: float = 1e-4,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Extract phase-normalized `(has_pair, vr, vi, omega)` for the leading Hopf eigenpair."""
    if eigs is None or vecs is None:
        eigvals, eigvecs = jnp.linalg.eig(Ju)
    else:
        eigvals, eigvecs = eigs, vecs

    has_pair, leading_real, omega0 = extract_leading_hopf_pair(Ju, eigs=eigvals, rel_tol=rel_tol)
    target_eig = jnp.where(has_pair, leading_real + 1j * omega0, 0.0 + 0.0j)
    vec0 = eigvecs[:, jnp.argmin(jnp.abs(eigvals - target_eig))]
    vr, vi = jnp.real(vec0), jnp.imag(vec0)

    # Rotate v -> exp(-i*theta)*v so <vr, vi> == 0 and ||vr|| >= ||vi||, then normalize ||vr|| == 1.
    theta = 0.5 * jnp.arctan2(2.0 * jnp.dot(vr, vi), jnp.dot(vr, vr) - jnp.dot(vi, vi))
    c, s = jnp.cos(theta), jnp.sin(theta)
    vr_rot = c * vr + s * vi
    vi_rot = -s * vr + c * vi
    swap = jnp.dot(vr_rot, vr_rot) < jnp.dot(vi_rot, vi_rot)
    vr_p = jnp.where(swap, vi_rot, vr_rot)
    vi_p = jnp.where(swap, -vr_rot, vi_rot)

    norm_vr = jnp.linalg.norm(vr_p)
    scale = jnp.where(norm_vr > 1e-30, 1.0 / norm_vr, 1.0)
    omega_safe = jnp.where(has_pair, omega0, 0.0)
    return has_pair, vr_p * scale, vi_p * scale, omega_safe


def hopf_augmented_residual(
    f_val: jax.Array,
    Ju: jax.Array,
    vr: jax.Array,
    vi: jax.Array,
    omega: jax.Array,
    l_R: jax.Array,
) -> jax.Array:
    """Evaluate Hopf defining system [F, Ju*vr + omega*vi, Ju*vi - omega*vr, <l_R, vr>-1, <l_R, vi>]."""
    return jnp.concatenate(
        [
            f_val,
            Ju @ vr + omega * vi,
            Ju @ vi - omega * vr,
            jnp.atleast_1d(jnp.dot(l_R, vr) - 1.0),
            jnp.atleast_1d(jnp.dot(l_R, vi)),
        ]
    )


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
    # Carry f(lo) across iterations to preserve bracket sign consistency at exact zeros
    # and halve function evaluations inside the while_loop.
    f_lo_0 = f(lo)

    def cond(state: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> jax.Array:
        lo_s, hi_s, _, i_s = state
        return ((hi_s - lo_s) > tol) & (i_s < max_iters) & jnp.isfinite(lo_s) & jnp.isfinite(hi_s)

    def body(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        lo_s, hi_s, f_lo_s, i_s = state
        mid = 0.5 * (lo_s + hi_s)
        f_mid = f(mid)
        go_left = jnp.isfinite(f_mid) & jnp.isfinite(f_lo_s) & ((f_lo_s * f_mid) <= 0.0)
        return (
            jnp.where(go_left, lo_s, mid),
            jnp.where(go_left, mid, hi_s),
            jnp.where(go_left, f_lo_s, f_mid),
            i_s + 1,
        )

    lo_f, hi_f, _, _ = jax.lax.while_loop(cond, body, (lo, hi, f_lo_0, jnp.int32(0)))
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


def _solve_linear(mat: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve a square linear system with least-squares fallback if non-finite."""
    sol = jnp.linalg.solve(mat, rhs)
    return jax.lax.cond(
        jnp.all(jnp.isfinite(sol)),
        lambda: sol,
        lambda: jnp.linalg.lstsq(mat, rhs, rcond=1e-12)[0],
    )


def _augmented_newton_solve(
    res_fn: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    tol: float = 1e-11,
    max_iters: int = 15,
) -> tuple[jax.Array, jax.Array]:
    """Solve square augmented system F(x) = 0 in pure JAX with scale-invariant stopping."""
    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    r0 = jnp.atleast_1d(jnp.asarray(res_fn(x0_arr), dtype=jnp.float64)).ravel()
    jac_fn = jax.jacfwd(res_fn)
    init_ok = jnp.all(jnp.isfinite(x0_arr)) & jnp.all(jnp.isfinite(r0))
    conv0 = init_ok & jnp.all(r0 == 0.0)
    init_state = (x0_arr, r0, jnp.int32(0), conv0, ~init_ok)

    def cond(state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]) -> jax.Array:
        _, _, i, conv, aborted = state
        return (~conv) & (~aborted) & (i < max_iters)

    def body(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        x_cur, r_cur, i, _, _ = state
        J = jnp.asarray(jac_fn(x_cur), dtype=jnp.float64)
        dx = _solve_linear(J, -r_cur)
        step_norm = jnp.linalg.norm(dx)
        lin_res = J @ dx + r_cur
        row_scale = jnp.linalg.norm(J, axis=1) * step_norm
        lin_ok = jnp.all(jnp.isfinite(dx)) & jnp.all(
            jnp.abs(lin_res) <= 0.5 * jnp.abs(r_cur) + 1e-6 * row_scale
        )
        x_cand = x_cur + dx
        r_cand = jnp.atleast_1d(jnp.asarray(res_fn(x_cand), dtype=jnp.float64)).ravel()
        cand_ok = lin_ok & jnp.all(jnp.isfinite(x_cand)) & jnp.all(jnp.isfinite(r_cand))
        step_ok = step_norm <= tol * (1.0 + jnp.linalg.norm(x_cur))
        conv_next = cand_ok & (step_ok | jnp.all(r_cand == 0.0))
        return (
            jnp.where(cand_ok, x_cand, x_cur),
            jnp.where(cand_ok, r_cand, r_cur),
            i + jnp.int32(1),
            conv_next,
            ~cand_ok,
        )

    x_fin, _, _, conv_fin, _ = jax.lax.while_loop(cond, body, init_state)
    return x_fin, conv_fin


def _refine_fold_moore_spence(
    problem: BifurcationProblem,
    u_approx: jax.Array,
    p_approx: float | jax.Array,
    v_init: jax.Array | None = None,
    tol: float = 1e-11,
    max_iters: int = 15,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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

    x_fin, conv = _augmented_newton_solve(ms_res, x0, tol=tol, max_iters=max_iters)
    return x_fin[:n], jnp.squeeze(x_fin[n]), conv


def _refine_hopf_augmented(
    problem: BifurcationProblem,
    u_approx: jax.Array,
    p_approx: float | jax.Array,
    tol: float = 1e-11,
    max_iters: int = 15,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Pinpoint exact Hopf (HB) coordinates via the (3n+2)-dim augmented defining system."""
    n = problem.dim
    u_0 = jnp.asarray(u_approx, dtype=jnp.float64).ravel()
    p_0 = jnp.squeeze(jnp.asarray(p_approx, dtype=jnp.float64))
    if n < 2:
        return u_0, p_0, jnp.bool_(False)

    Ju0 = problem.jacobian_u(u_0, p_0)
    has_pair, vr0, vi0, omega0 = canonical_hopf_eigenvector(Ju0)
    l_R = vr0
    x0 = jnp.concatenate([u_0, jnp.atleast_1d(p_0), vr0, vi0, jnp.atleast_1d(omega0)])

    def hb_res(x_vec: jax.Array) -> jax.Array:
        u_v = x_vec[:n]
        p_v = x_vec[n]
        vr_v = x_vec[n + 1 : 2 * n + 1]
        vi_v = x_vec[2 * n + 1 : 3 * n + 1]
        om_v = x_vec[3 * n + 1]
        return hopf_augmented_residual(
            problem.residual(u_v, p_v),
            problem.jacobian_u(u_v, p_v),
            vr_v,
            vi_v,
            om_v,
            l_R,
        )

    x_fin, conv = _augmented_newton_solve(hb_res, x0, tol=tol, max_iters=max_iters)
    u_fin = x_fin[:n]
    p_fin = jnp.squeeze(x_fin[n])
    om_fin = x_fin[3 * n + 1]
    ok = (
        has_pair
        & conv
        & jnp.all(jnp.isfinite(u_fin))
        & jnp.isfinite(p_fin)
        & (jnp.abs(om_fin) > 1e-8)
    )
    return u_fin, p_fin, ok


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
    """Pinpoint exact bifurcation coordinates on the curve in pure JAX."""
    n = problem.dim
    u_a_j = jnp.asarray(u_a, dtype=jnp.float64)
    u_b_j = jnp.asarray(u_b, dtype=jnp.float64)
    tau_a_j = jnp.asarray(tau_a, dtype=jnp.float64)
    tau_b_j = jnp.asarray(tau_b, dtype=jnp.float64)
    p_a_j = jnp.squeeze(jnp.asarray(p_a, dtype=jnp.float64))
    p_b_j = jnp.squeeze(jnp.asarray(p_b, dtype=jnp.float64))

    z_a = jnp.concatenate([u_a_j, jnp.atleast_1d(p_a_j)])
    z_b = jnp.concatenate([u_b_j, jnp.atleast_1d(p_b_j)])
    dz_ab = z_b - z_a
    ds_ab = jnp.linalg.norm(dz_ab)
    tau_chord = 0.5 * (tau_a_j + tau_b_j)
    tau_chord_norm = jnp.linalg.norm(tau_chord)
    tau_chord = jnp.where(tau_chord_norm > 1e-12, tau_chord / tau_chord_norm, tau_a_j)
    tau_sec = jnp.where(ds_ab > 1e-12, dz_ab / jnp.maximum(ds_ab, 1e-30), tau_chord)

    def _project_on_curve(alpha: jax.Array) -> tuple[jax.Array, jax.Array]:
        u_ch = (1.0 - alpha) * u_a_j + alpha * u_b_j
        p_ch = (1.0 - alpha) * p_a_j + alpha * p_b_j
        res = correct_arclength_step(
            residual_fn=problem.residual,
            jac_u_fn=problem.jacobian_u,
            jac_p_fn=problem.jacobian_p,
            u_init=u_ch,
            p_init=p_ch,
            u_prev=u_a_j,
            p_prev=p_a_j,
            tau_prev=tau_sec,
            ds=alpha * ds_ab,
            tol=tol,
            step_tol=tol,
            max_iters=6,
        )
        ok = res.converged & jnp.all(jnp.isfinite(res.u)) & jnp.isfinite(res.p)
        return jnp.where(ok, res.u, u_ch), jnp.where(ok, res.p, p_ch)

    def _curve_tangent(Ju_c: jax.Array, Jp_c: jax.Array, fallback: jax.Array) -> jax.Array:
        sol_tau = solve_bordered_system(
            A=Ju_c,
            b=Jp_c,
            c=tau_sec[:n],
            d=tau_sec[-1],
            f=jnp.zeros(n, dtype=jnp.float64),
            g=1.0,
        )
        t_raw = sol_tau.z
        t_norm = jnp.linalg.norm(t_raw)
        ok = jnp.all(jnp.isfinite(t_raw)) & (t_norm > 1e-12)
        t_unit = jnp.where(ok, t_raw / jnp.maximum(t_norm, 1e-30), fallback)
        return jnp.where(jnp.dot(t_unit, tau_sec) < 0.0, -t_unit, t_unit)

    def test_val_at(alpha: jax.Array) -> jax.Array:
        u_c, p_c = _project_on_curve(alpha)
        tau_ch_alpha = (1.0 - alpha) * tau_a_j + alpha * tau_b_j
        tau_ch_n = jnp.linalg.norm(tau_ch_alpha)
        tau_fb = jnp.where(tau_ch_n > 1e-12, tau_ch_alpha / tau_ch_n, tau_ch_alpha)
        Ju_c = problem.jacobian_u(u_c, p_c)
        if bif_type == "HB":
            return test_func_hopf(Ju_c)
        Jp_c = problem.jacobian_p(u_c, p_c)
        tau_c = _curve_tangent(Ju_c, Jp_c, tau_fb)
        if bif_type == "LP":
            return tau_c[-1]
        return _test_func_branch_point(Ju_c, Jp=Jp_c, tangent=tau_c)

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
    u_pol, p_pol = _project_on_curve(alpha_opt)
    z_pol = jnp.concatenate([u_pol, jnp.atleast_1d(p_pol)])
    tol_radius = 2.0 * jnp.maximum(ds_ab, 1.0 + 0.1 * jnp.linalg.norm(z_pol))

    tau_opt = (1.0 - alpha_opt) * tau_a_j + alpha_opt * tau_b_j
    tau_norm = jnp.linalg.norm(tau_opt)
    tau_opt = jnp.where(tau_norm > 1e-12, tau_opt / tau_norm, tau_opt)
    tau_opt = _curve_tangent(
        problem.jacobian_u(u_pol, p_pol), problem.jacobian_p(u_pol, p_pol), tau_opt
    )

    if bif_type == "LP":
        u_ms, p_ms, ms_conv = _refine_fold_moore_spence(
            problem, u_pol, p_pol, v_init=tau_opt[:n], tol=tol
        )
        z_ms = jnp.concatenate([u_ms, jnp.atleast_1d(p_ms)])
        res_ms = jnp.linalg.norm(problem.residual(u_ms, p_ms))
        res_pol = jnp.linalg.norm(problem.residual(u_pol, p_pol))
        ms_ok = (
            ms_conv
            & jnp.all(jnp.isfinite(z_ms))
            & (res_ms <= jnp.maximum(1e-6, 10.0 * res_pol))
            & (jnp.linalg.norm(z_ms - z_pol) <= tol_radius)
        )
        u_res = jnp.where(ms_ok, u_ms, u_pol)
        p_res = jnp.where(ms_ok, p_ms, p_pol)
    elif bif_type == "HB":
        u_hb, p_hb, hb_conv = _refine_hopf_augmented(problem, u_pol, p_pol, tol=tol)
        z_hb = jnp.concatenate([u_hb, jnp.atleast_1d(p_hb)])
        hb_ok = (
            hb_conv & jnp.all(jnp.isfinite(z_hb)) & (jnp.linalg.norm(z_hb - z_pol) <= tol_radius)
        )
        u_res = jnp.where(hb_ok, u_hb, u_pol)
        p_res = jnp.where(hb_ok, p_hb, p_pol)
    else:
        u_res, p_res = u_pol, p_pol

    return (
        jnp.where(jnp.all(jnp.isfinite(u_res)), u_res, u_mid),
        jnp.where(jnp.isfinite(p_res), p_res, p_mid),
    )


def compute_first_lyapunov_coefficient(
    fn_or_problem: BifurcationProblem | Callable[..., jax.Array],
    u_hopf: ArrayLike,
    p_hopf: float | jax.Array,
    *,
    Ju: jax.Array | None = None,
) -> jax.Array:
    """Compute Kuznetsov's first Lyapunov coefficient l_1 at a Hopf point via JAX autodiff."""
    u_h = jnp.atleast_1d(jnp.asarray(u_hopf, dtype=jnp.float64)).ravel()
    p_h = jnp.squeeze(jnp.asarray(p_hopf, dtype=jnp.float64))
    n = int(u_h.shape[0])
    if n < 2:
        return jnp.array(jnp.nan, dtype=jnp.float64)

    if isinstance(fn_or_problem, BifurcationProblem):

        def _f_u(u_val: jax.Array) -> jax.Array:
            return fn_or_problem.residual(u_val, p_h)

        A = fn_or_problem.jacobian_u(u_h, p_h) if Ju is None else jnp.asarray(Ju, dtype=jnp.float64)
    else:

        def _f_u(u_val: jax.Array) -> jax.Array:
            return jnp.atleast_1d(jnp.asarray(fn_or_problem(u_val, p_h), dtype=jnp.float64)).ravel()

        A = jax.jacfwd(_f_u)(u_h) if Ju is None else jnp.asarray(Ju, dtype=jnp.float64)

    eigs, vecs = jnp.linalg.eig(A)
    has_pair, leading_real, omega0 = extract_leading_hopf_pair(A, eigs=eigs)
    target_eig = jnp.where(has_pair, leading_real + 1j * omega0, 0.0 + 0.0j)
    q_raw = vecs[:, jnp.argmin(jnp.abs(eigs - target_eig))]
    q_norm = jnp.linalg.norm(q_raw)
    q = q_raw / jnp.where(q_norm > 1e-30, q_norm, 1.0)

    eigs_t, vecs_t = jnp.linalg.eig(A.T)
    p_raw = vecs_t[:, jnp.argmin(jnp.abs(eigs_t - jnp.conj(target_eig)))]
    c = jnp.vdot(p_raw, q)
    p_vec = p_raw / jnp.where(jnp.abs(c) > 1e-30, jnp.conj(c), 1.0)

    def _b_real(v1: jax.Array, v2: jax.Array) -> jax.Array:
        def d1(u_val: jax.Array) -> jax.Array:
            return jax.jvp(_f_u, (u_val,), (v1,))[1]

        return jax.jvp(d1, (u_h,), (v2,))[1]

    def _b_complex(x: jax.Array, y: jax.Array) -> jax.Array:
        xr, xi = jnp.real(x), jnp.imag(x)
        yr, yi = jnp.real(y), jnp.imag(y)
        return (_b_real(xr, yr) - _b_real(xi, yi)) + 1j * (_b_real(xr, yi) + _b_real(xi, yr))

    def _c_real(v1: jax.Array, v2: jax.Array, v3: jax.Array) -> jax.Array:
        def d2(u_val: jax.Array) -> jax.Array:
            def d1(w_val: jax.Array) -> jax.Array:
                return jax.jvp(_f_u, (w_val,), (v1,))[1]

            return jax.jvp(d1, (u_val,), (v2,))[1]

        return jax.jvp(d2, (u_h,), (v3,))[1]

    qr, qi = jnp.real(q), jnp.imag(q)
    b_qq = (_b_real(qr, qr) - _b_real(qi, qi)) + 2j * _b_real(qr, qi)
    b_qqbar = _b_real(qr, qr) + _b_real(qi, qi)
    c_qqqbar = (_c_real(qr, qr, qr) + _c_real(qr, qi, qi)) + 1j * (
        _c_real(qi, qr, qr) + _c_real(qi, qi, qi)
    )

    w11 = _solve_linear(A, b_qqbar)
    omega_safe = jnp.where(has_pair & (omega0 > 1e-30), omega0, 1.0)
    M2 = 2j * omega_safe * jnp.eye(n, dtype=jnp.complex128) - A.astype(jnp.complex128)
    w20 = _solve_linear(M2, b_qq)

    term1 = jnp.vdot(p_vec, c_qqqbar)
    term2 = -2.0 * jnp.vdot(p_vec, _b_complex(q, w11.astype(jnp.complex128)))
    term3 = jnp.vdot(p_vec, _b_complex(jnp.conj(q), w20))
    l1 = jnp.real(term1 + term2 + term3) / (2.0 * omega_safe)

    valid = has_pair & (omega0 > 1e-12) & (jnp.abs(c) > 1e-12) & jnp.isfinite(l1)
    return jnp.where(valid, l1, jnp.nan)


def _compute_secondary_tangent(
    problem: BifurcationProblem,
    u_star: jax.Array,
    p_star: jax.Array,
    incoming_tangent: jax.Array,
) -> jax.Array:
    """Solve the Algebraic Bifurcation Equation (ABE) for the secondary branch unit tangent."""
    n = problem.dim
    Ju = problem.jacobian_u(u_star, p_star)
    Jp = problem.jacobian_p(u_star, p_star)

    u_svd, _, vh = jnp.linalg.svd(Ju)
    phi = vh[-1, :]
    psi = u_svd[:, -1]

    v0_raw = jnp.linalg.lstsq(Ju, -Jp)[0].ravel()
    v0 = v0_raw - jnp.dot(v0_raw, phi) * phi

    zero_p = jnp.array(0.0, dtype=jnp.float64)
    one_p = jnp.array(1.0, dtype=jnp.float64)
    xi1 = (phi, zero_p)
    xi2 = (v0, one_p)

    def d2F_directional(
        v_dir: tuple[jax.Array, jax.Array], w_dir: tuple[jax.Array, jax.Array]
    ) -> jax.Array:
        v_u, v_p = v_dir
        w_u, w_p = w_dir

        def dir1(u_val: jax.Array, p_val: jax.Array) -> jax.Array:
            return jax.jvp(problem.residual, (u_val, p_val), (v_u, v_p))[1]

        return jax.jvp(dir1, (u_star, p_star), (w_u, w_p))[1]

    c11 = jnp.dot(psi, d2F_directional(xi1, xi1))
    c12 = jnp.dot(psi, d2F_directional(xi1, xi2))
    c22 = jnp.dot(psi, d2F_directional(xi2, xi2))

    tau_in = jnp.asarray(incoming_tangent, dtype=jnp.float64)
    pdot_in = tau_in[-1]
    udot_in = tau_in[:n]
    alpha_in = jnp.dot(phi, udot_in - pdot_in * v0)

    c11_safe = jnp.where(jnp.abs(c11) > 1e-8, c11, one_p)
    sqrt_disc = jnp.sqrt(jnp.maximum(0.0, c12**2 - c11 * c22))
    r1 = (-c12 + sqrt_disc) / c11_safe
    r2 = (-c12 - sqrt_disc) / c11_safe

    pdot_safe = jnp.where(jnp.abs(pdot_in) > 1e-6, pdot_in, one_p)
    r_in = alpha_in / pdot_safe
    r_sec = jnp.where(jnp.abs(r1 - r_in) < jnp.abs(r2 - r_in), r2, r1)
    alpha_case1 = jnp.where(jnp.abs(pdot_in) > 1e-6, r_sec, r1)

    alpha_case2 = jnp.where(jnp.abs(pdot_in) < 1e-4, -c22, one_p)
    pdot_case2 = jnp.where(jnp.abs(pdot_in) < 1e-4, 2.0 * c12, zero_p)

    has_c11 = jnp.abs(c11) > 1e-8
    has_c12 = jnp.abs(c12) > 1e-8
    alpha_sec = jnp.where(has_c11, alpha_case1, jnp.where(has_c12, alpha_case2, one_p))
    pdot_sec = jnp.where(has_c11, one_p, jnp.where(has_c12, pdot_case2, zero_p))

    udot_sec = alpha_sec * phi + pdot_sec * v0
    tau_sec_vec = jnp.concatenate([udot_sec, jnp.atleast_1d(pdot_sec)])
    tau_sec_norm = jnp.linalg.norm(tau_sec_vec)
    phi_fallback = jnp.concatenate([phi, jnp.zeros(1, dtype=jnp.float64)])
    phi_fallback = phi_fallback / jnp.linalg.norm(phi_fallback)

    tau_sec_vec = jnp.where(tau_sec_norm > 1e-12, tau_sec_vec / tau_sec_norm, phi_fallback)
    return jnp.where(jnp.abs(jnp.dot(tau_sec_vec, tau_in)) > 0.999, phi_fallback, tau_sec_vec)


def switch_branch(
    problem: BifurcationProblem,
    bp_u: jax.Array,
    bp_p: float | jax.Array,
    incoming_tangent: jax.Array,
    ds: float | jax.Array = 0.05,
    branch_direction: float | jax.Array = 1.0,
    tol: float = 1e-8,
    max_iters: int = 15,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Switch to the intersecting branch at a simple branch point (BP) in pure JAX."""
    n = problem.dim
    bp_u_arr = jnp.asarray(bp_u, dtype=jnp.float64)
    bp_p_arr = jnp.squeeze(jnp.asarray(bp_p, dtype=jnp.float64))
    ds_arr = jnp.squeeze(jnp.asarray(ds, dtype=jnp.float64))

    tau_sec = _compute_secondary_tangent(problem, bp_u_arr, bp_p_arr, incoming_tangent)
    dir_sgn = jnp.where(jnp.asarray(branch_direction, dtype=jnp.float64) < 0, -1.0, 1.0)
    tau_target = dir_sgn * tau_sec

    u_pred = bp_u_arr + ds_arr * tau_target[:n]
    p_pred = bp_p_arr + ds_arr * tau_target[-1]

    res = correct_arclength_step(
        residual_fn=problem.residual,
        jac_u_fn=problem.jacobian_u,
        jac_p_fn=problem.jacobian_p,
        u_init=u_pred,
        p_init=p_pred,
        u_prev=bp_u_arr,
        p_prev=bp_p_arr,
        tau_prev=tau_target,
        ds=ds_arr,
        tol=tol,
        max_iters=max_iters,
    )

    u_out = jnp.where(res.converged, res.u, u_pred)
    p_out = jnp.where(res.converged, res.p, p_pred)
    return u_out, p_out, tau_target
