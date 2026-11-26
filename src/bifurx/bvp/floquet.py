from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from bifurx.bvp.condensation import condense_local_blocks

_SPLIT_2X2_MAX_ITERS = 8
_EXCEPTIONAL_SHIFT_PERIOD = 10
_ZERO_SHIFT_LOG_GAP = 30.0
_DEFAULT_QR_ITERS_PER_DIM = 60
_DEFAULT_VARIATIONAL_STEPS = 400

if TYPE_CHECKING:
    from bifurx.bvp.problem import CollocationProblem, PeriodicOrbitResult


def monodromy_factors_condensation(
    problem: CollocationProblem,
    mesh: jax.Array,
    u_gauss: jax.Array,
    period: jax.Array,
    p: float | jax.Array = 0.0,
) -> jax.Array:
    """Compute per-interval transition matrices ``M_k`` of shape ``(N, n, n)`` via condensation.

    The full monodromy matrix is the cyclic product ``M = M_{N-1} @ ... @ M_1 @ M_0``.
    """
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    u_gauss_arr = jnp.asarray(u_gauss, dtype=jnp.float64)
    T = jnp.squeeze(jnp.asarray(period, dtype=jnp.float64))
    p_val = jnp.squeeze(jnp.asarray(p, dtype=jnp.float64))
    h = mesh_arr[1:] - mesh_arr[:-1]
    jac_fn = jax.jacobian(lambda u_: problem.fn(u_, p_val, **problem.kwargs))
    Ju_all = jax.vmap(jax.vmap(jac_fn))(u_gauss_arr)
    _, _, _, M_k = condense_local_blocks(problem, h, T, Ju_all, adjoint=False)
    return M_k


def monodromy_condensation_raw(
    problem: CollocationProblem,
    mesh: jax.Array,
    u_gauss: jax.Array,
    period: jax.Array,
    p: float | jax.Array = 0.0,
) -> jax.Array:
    """Compute monodromy matrix M via de Boor-Swartz condensation directly from collocation arrays."""
    factors = monodromy_factors_condensation(problem, mesh, u_gauss, period, p=p)
    eye_n = jnp.eye(problem.dim, dtype=jnp.float64)
    M_total, _ = jax.lax.scan(lambda acc, M_k: (M_k @ acc, None), eye_n, factors)
    return M_total


def _monodromy_factors_from_sol(
    sol: PeriodicOrbitResult,
    p: float | jax.Array = 0.0,
) -> jax.Array:
    """Extract per-interval monodromy factors ``M_k`` from a ``PeriodicOrbitResult``."""
    return monodromy_factors_condensation(
        sol.problem,
        jnp.asarray(sol.mesh, dtype=jnp.float64),
        jnp.asarray(sol.u_gauss, dtype=jnp.float64),
        jnp.squeeze(jnp.asarray(sol.period, dtype=jnp.float64)),
        p=p,
    )


def _givens(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Compute Givens rotation ``(c, s)`` such that ``[[c, s], [-s, c]] @ [a, b]^T = [r, 0]^T``."""
    r = jnp.hypot(a, b)
    pos = r > 0.0
    safe_r = jnp.where(pos, r, 1.0)
    c = jnp.where(pos, a / safe_r, 1.0)
    s = jnp.where(pos, b / safe_r, 0.0)
    return c, s


def _apply_givens_left(A: jax.Array, i: jax.Array, c: jax.Array, s: jax.Array) -> jax.Array:
    """Apply ``G^T = [[c, s], [-s, c]]`` from the left to rows ``(i-1, i)`` of ``A``."""
    r0 = A[i - 1, :]
    r1 = A[i, :]
    return A.at[i - 1, :].set(c * r0 + s * r1).at[i, :].set(-s * r0 + c * r1)


def _apply_givens_right(A: jax.Array, i: jax.Array, c: jax.Array, s: jax.Array) -> jax.Array:
    """Apply ``G = [[c, -s], [s, c]]`` from the right to columns ``(i-1, i)`` of ``A``."""
    c0 = A[:, i - 1]
    c1 = A[:, i]
    return A.at[:, i - 1].set(c * c0 + s * c1).at[:, i].set(-s * c0 + c * c1)


def _propagate_givens(
    H: jax.Array,
    R_all: jax.Array,
    i: jax.Array,
    c: jax.Array,
    s: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Apply orthogonal similarity ``G^T (H R_{N-2} ... R_0) G`` preserving upper-triangular ``R_k``."""
    H = _apply_givens_left(H, i, c, s)
    if R_all.shape[0] > 0:

        def step(
            carry: tuple[jax.Array, jax.Array], R_k: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            ck, sk = carry
            R_k = _apply_givens_right(R_k, i, ck, sk)
            c_next, s_next = _givens(R_k[i - 1, i - 1], R_k[i, i - 1])
            R_k = _apply_givens_left(R_k, i, c_next, s_next)
            R_k = R_k.at[i, i - 1].set(0.0)
            return (c_next, s_next), R_k

        (c_end, s_end), R_all = jax.lax.scan(step, (c, s), R_all)
        H = _apply_givens_right(H, i, c_end, s_end)
    else:
        H = _apply_givens_right(H, i, c, s)
    return H, R_all


def _normalized_2x2_product(
    B_all: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute ``B_{K-1} @ ... @ B_0 = exp(log_scale) * P_hat`` with ``||P_hat||_F = 1``."""
    eye2 = jnp.eye(2, dtype=jnp.float64)

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
        B_k: jax.Array,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], None]:
        P, log_s, is_zero = carry
        P_new = B_k @ P
        nrm = jnp.linalg.norm(P_new)
        pos = nrm > 0.0
        safe_nrm = jnp.where(pos, nrm, 1.0)
        P_out = jnp.where(pos, P_new / safe_nrm, jnp.zeros((2, 2), dtype=jnp.float64))
        log_s_out = jnp.where(pos, log_s + jnp.log(safe_nrm), log_s)
        return (P_out, log_s_out, is_zero | (~pos)), None

    (P_hat, log_scale, is_zero), _ = jax.lax.scan(
        step,
        (eye2, jnp.array(0.0, dtype=jnp.float64), jnp.array(False)),
        B_all,
    )
    return P_hat, log_scale, is_zero


def _periodic_hessenberg_triangular(factors: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Reduce ``factors`` of shape ``(N, n, n)`` to periodic Hessenberg-triangular form.

    Returns ``(H, R_all)`` where ``H`` is ``(n, n)`` upper Hessenberg and ``R_all`` is
    ``(N - 1, n, n)`` with every ``R_all[k]`` upper triangular, such that
    ``H @ R_all[N-2] @ ... @ R_all[0]`` is orthogonally similar to
    ``factors[N-1] @ ... @ factors[0]``.
    """
    N, n, _ = factors.shape
    eye_n = jnp.eye(n, dtype=jnp.float64)

    if N > 1:

        def qr_sweep(Q: jax.Array, A_k: jax.Array) -> tuple[jax.Array, jax.Array]:
            Q_next, R_k = jnp.linalg.qr(A_k @ Q)
            return Q_next, jnp.triu(R_k)

        Q_last, R_all = jax.lax.scan(qr_sweep, eye_n, factors[:-1])
        H = factors[-1] @ Q_last
    else:
        R_all = jnp.zeros((0, n, n), dtype=jnp.float64)
        H = factors[0]

    if n <= 2:
        return H, R_all

    def outer_col(j: jax.Array, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        def inner_row(
            step_idx: jax.Array, inner_carry: tuple[jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            H_c, R_c = inner_carry
            i = (n - 1) - step_idx
            c, s = _givens(H_c[i - 1, j], H_c[i, j])
            H_c, R_c = _propagate_givens(H_c, R_c, i, c, s)
            H_c = H_c.at[i, j].set(0.0)
            return H_c, R_c

        return jax.lax.fori_loop(0, (n - 2) - j, inner_row, carry)

    H, R_all = jax.lax.fori_loop(0, n - 2, outer_col, (H, R_all))
    H = jnp.triu(H, k=-1)
    return H, R_all


def _is_negligible_subdiag(H: jax.Array, i: jax.Array, tol: float) -> jax.Array:
    """Return True if subdiagonal entry ``H[i, i-1]`` is negligible relative to its diagonal."""
    diag_scale = jnp.abs(H[i - 1, i - 1]) + jnp.abs(H[i, i])
    thresh = jnp.where(diag_scale > 0.0, tol * diag_scale, jnp.finfo(jnp.float64).tiny)
    return jnp.abs(H[i, i - 1]) <= thresh


def _split_real_2x2_block(
    H: jax.Array,
    R_all: jax.Array,
    i: jax.Array,
    tol: float,
) -> tuple[jax.Array, jax.Array]:
    """Triangularize a decoupled 2x2 diagonal block ``(i, i+1)`` if its eigenvalues are real."""
    n_r = R_all.shape[0]
    eye2 = jnp.eye(2, dtype=jnp.float64)

    def cond_fn(state: tuple[jax.Array, jax.Array, jax.Array, jax.Array]) -> jax.Array:
        H_c, _, it, done = state
        return (~done) & (it < _SPLIT_2X2_MAX_ITERS) & (~_is_negligible_subdiag(H_c, i + 1, tol))

    def body_fn(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        H_c, R_c, it, _ = state
        B_H = jax.lax.dynamic_slice(H_c, (i, i), (2, 2))
        if n_r > 0:
            B_R = jax.lax.dynamic_slice(R_c, (jnp.int32(0), i, i), (n_r, 2, 2))
            B_all = jnp.concatenate([B_R, B_H[None, :, :]], axis=0)
        else:
            B_all = B_H[None, :, :]

        P_hat, _, is_zero = _normalized_2x2_product(B_all)
        half_tr = 0.5 * (P_hat[0, 0] + P_hat[1, 1])
        diff = 0.5 * (P_hat[0, 0] - P_hat[1, 1])
        disc = diff * diff + P_hat[0, 1] * P_hat[1, 0]
        is_complex_or_zero = is_zero | (disc < 0.0)

        def do_step(
            args: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            H_in, R_in = args
            sq = jnp.sqrt(jnp.maximum(disc, 0.0))
            lam_big = half_tr + jnp.where(half_tr >= 0.0, sq, -sq)
            det_P = P_hat[0, 0] * P_hat[1, 1] - P_hat[0, 1] * P_hat[1, 0]
            lam_small = jnp.where(
                jnp.abs(lam_big) > 0.0,
                det_P / jnp.where(jnp.abs(lam_big) > 0.0, lam_big, 1.0),
                0.0,
            )
            W = P_hat - lam_small * eye2
            n0 = jnp.hypot(W[0, 0], W[1, 0])
            n1 = jnp.hypot(W[0, 1], W[1, 1])
            v = jnp.where(n0 >= n1, W[:, 0], W[:, 1])
            c, s = _givens(v[0], v[1])
            return _propagate_givens(H_in, R_in, i + 1, c, s)

        H_next, R_next = jax.lax.cond(
            is_complex_or_zero,
            lambda args: args,
            do_step,
            (H_c, R_c),
        )
        return H_next, R_next, it + 1, is_complex_or_zero

    H_fin, R_fin, _, _ = jax.lax.while_loop(
        cond_fn, body_fn, (H, R_all, jnp.int32(0), jnp.array(False))
    )
    H_fin = jnp.where(
        _is_negligible_subdiag(H_fin, i + 1, tol),
        H_fin.at[i + 1, i].set(0.0),
        H_fin,
    )
    return H_fin, R_fin


def _francis_periodic_qr_step(
    H: jax.Array,
    R_all: jax.Array,
    m1: jax.Array,
    m2: jax.Array,
    its: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Perform one overflow-free periodic implicit Francis double-shift QR sweep on ``[m1, m2]``."""
    n_r = R_all.shape[0]

    # 1. Leading 3x2 block of M = exp(beta) * M_hat (without overflow/underflow)
    H_lead = jax.lax.dynamic_slice(H, (m1, m1), (3, 2))
    if n_r > 0:
        R_lead = jax.lax.dynamic_slice(R_all, (jnp.int32(0), m1, m1), (n_r, 2, 2))
        R_hat, beta, lead_zero = _normalized_2x2_product(R_lead)
        M_raw = H_lead @ R_hat
    else:
        beta = jnp.array(0.0, dtype=jnp.float64)
        lead_zero = jnp.array(False)
        M_raw = H_lead

    m_nrm = jnp.linalg.norm(M_raw)
    m_pos = m_nrm > 0.0
    safe_m_nrm = jnp.where(m_pos, m_nrm, 1.0)
    M_hat = jnp.where(m_pos, M_raw / safe_m_nrm, jnp.zeros((3, 2), dtype=jnp.float64))
    beta = jnp.where(m_pos, beta + jnp.log(safe_m_nrm), beta)
    lead_zero = lead_zero | (~m_pos)

    # 2. Trailing 2x2 shift block T_2 = exp(alpha) * T_hat
    H_trail = jax.lax.dynamic_slice(H, (m2 - 1, m2 - 1), (2, 2))
    if n_r > 0:
        R_trail = jax.lax.dynamic_slice(R_all, (jnp.int32(0), m2 - 1, m2 - 1), (n_r, 2, 2))
        B_trail = jnp.concatenate([R_trail, H_trail[None, :, :]], axis=0)
    else:
        B_trail = H_trail[None, :, :]
    T_hat, alpha, trail_zero = _normalized_2x2_product(B_trail)

    tr_T = T_hat[0, 0] + T_hat[1, 1]
    det_T = T_hat[0, 0] * T_hat[1, 1] - T_hat[0, 1] * T_hat[1, 0]

    # Exceptional shift every 10th iteration without deflation to break cyclic symmetries
    use_exceptional = (its > 0) & ((its % _EXCEPTIONAL_SHIFT_PERIOD) == 0)
    s_exc = jnp.abs(M_hat[0, 0]) + jnp.abs(M_hat[1, 0]) + jnp.abs(M_hat[1, 1])
    tr_use = jnp.where(use_exceptional, 1.5 * s_exc, tr_T)
    det_use = jnp.where(use_exceptional, s_exc * s_exc, det_T)
    alpha_use = jnp.where(use_exceptional, beta, alpha)

    # If trailing shift vastly dominates leading block, fall back to zero shift so large
    # eigenvalues move to the top and small eigenvalues deflate at the bottom.
    use_zero_shift = (~use_exceptional) & (trail_zero | ((alpha_use - beta) > _ZERO_SHIFT_LOG_GAP))
    gamma = jnp.maximum(alpha_use, beta)
    c_M = jnp.where(lead_zero, 0.0, jnp.exp(jnp.clip(beta - gamma, -700.0, 0.0)))
    c_T = jnp.where(use_zero_shift, 0.0, jnp.exp(jnp.clip(alpha_use - gamma, -700.0, 0.0)))

    # First column of (M - sigma_1 I)(M - sigma_2 I) / exp(2 * gamma)
    m00, m10, m01, m11, m21 = M_hat[0, 0], M_hat[1, 0], M_hat[0, 1], M_hat[1, 1], M_hat[2, 1]
    v0 = c_M * c_M * (m00 * m00 + m01 * m10) - c_M * c_T * tr_use * m00 + c_T * c_T * det_use
    v1 = c_M * m10 * (c_M * (m00 + m11) - c_T * tr_use)
    v2 = c_M * c_M * m21 * m10

    # Guard against degenerate shift vector
    degenerate = (jnp.abs(v1) + jnp.abs(v2)) <= 1e-14 * jnp.abs(v0)
    v0 = jnp.where(degenerate, m00 * m00 + m01 * m10, v0)
    v1 = jnp.where(degenerate, m10 * (m00 + m11), v1)
    v2 = jnp.where(degenerate, m21 * m10, v2)

    # 3. Introduce bulge at m1 via two Givens rotations: G_b on (m1+1, m1+2), G_a on (m1, m1+1)
    c_b, s_b = _givens(v1, v2)
    r1 = jnp.hypot(v1, v2)
    c_a, s_a = _givens(v0, r1)
    H, R_all = _propagate_givens(H, R_all, m1 + 2, c_b, s_b)
    H, R_all = _propagate_givens(H, R_all, m1 + 1, c_a, s_a)

    # 4. Chase bulge down columns k = m1 .. m2 - 2
    def chase_body(k: jax.Array, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        H_c, R_c = carry

        def two_bulge(args: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            H_i, R_i = args
            cb, sb = _givens(H_i[k + 2, k], H_i[k + 3, k])
            H_i, R_i = _propagate_givens(H_i, R_i, k + 3, cb, sb)
            H_i = H_i.at[k + 3, k].set(0.0)
            ca, sa = _givens(H_i[k + 1, k], H_i[k + 2, k])
            H_i, R_i = _propagate_givens(H_i, R_i, k + 2, ca, sa)
            H_i = H_i.at[k + 2, k].set(0.0)
            return H_i, R_i

        def one_bulge(args: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            H_i, R_i = args
            ca, sa = _givens(H_i[k + 1, k], H_i[k + 2, k])
            H_i, R_i = _propagate_givens(H_i, R_i, k + 2, ca, sa)
            H_i = H_i.at[k + 2, k].set(0.0)
            return H_i, R_i

        return jax.lax.cond(k + 3 <= m2, two_bulge, one_bulge, (H_c, R_c))

    H, R_all = jax.lax.fori_loop(m1, m2 - 1, chase_body, (H, R_all))
    return H, R_all


def periodic_real_schur(
    factors: jax.Array,
    max_iters: int | None = None,
    tol: float = 1e-14,
) -> jax.Array:
    """Compute the Periodic Real Schur Form of a cyclic matrix product without multiplying factors.

    Given ``factors`` of shape ``(N, n, n)`` representing ``M = M_{N-1} @ ... @ M_1 @ M_0``,
    returns ``schur_factors`` of shape ``(N, n, n)`` such that:

    - ``schur_factors[:-1]`` are upper triangular,
    - ``schur_factors[-1]`` is upper quasi-triangular (1x1 real blocks and 2x2 complex
      conjugate blocks), and
    - ``schur_factors[N-1] @ ... @ schur_factors[0]`` is orthogonally similar to ``M``.

    Pure JAX, 64-bit, JIT-compatible (``jax.lax.scan`` / ``jax.lax.while_loop``).
    """
    factors_arr = jnp.asarray(factors, dtype=jnp.float64)
    N, n, _ = factors_arr.shape
    if n <= 1:
        return factors_arr

    H, R_all = _periodic_hessenberg_triangular(factors_arr)

    if n >= 3:
        max_it = _DEFAULT_QR_ITERS_PER_DIM * n if max_iters is None else int(max_iters)

        def qr_cond(
            state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> jax.Array:
            _, _, m2, _, total_its = state
            return (m2 > 1) & (total_its < max_it)

        def qr_body(
            state: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            H_c, R_c, m2, its, total_its = state

            # Check deflation at bottom of active window: 1x1 or 2x2 block
            deflate_1 = _is_negligible_subdiag(H_c, m2, tol)
            deflate_2 = (~deflate_1) & _is_negligible_subdiag(H_c, m2 - 1, tol)

            def do_deflate_1() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                return (
                    H_c.at[m2, m2 - 1].set(0.0),
                    R_c,
                    m2 - 1,
                    jnp.int32(0),
                    total_its,
                )

            def do_deflate_2() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                return (
                    H_c.at[m2 - 1, m2 - 2].set(0.0),
                    R_c,
                    m2 - 2,
                    jnp.int32(0),
                    total_its,
                )

            def do_qr_sweep() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                # Find start m1 of unreduced active block [m1, m2]
                def find_m1_cond(k: jax.Array) -> jax.Array:
                    return (k > 0) & (~_is_negligible_subdiag(H_c, k, tol))

                m1 = jax.lax.while_loop(find_m1_cond, lambda k: k - 1, m2 - 1)
                H_clean = jax.lax.cond(
                    m1 > 0,
                    lambda mat: mat.at[m1, m1 - 1].set(0.0),
                    lambda mat: mat,
                    H_c,
                )
                H_next, R_next = _francis_periodic_qr_step(H_clean, R_c, m1, m2, its)
                return H_next, R_next, m2, its + 1, total_its + 1

            return jax.lax.cond(
                deflate_1,
                do_deflate_1,
                lambda: jax.lax.cond(deflate_2, do_deflate_2, do_qr_sweep),
            )

        init_state = (H, R_all, jnp.int32(n - 1), jnp.int32(0), jnp.int32(0))
        H, R_all, _, _, _ = jax.lax.while_loop(qr_cond, qr_body, init_state)

    # Split any decoupled 2x2 diagonal blocks that have real eigenvalues into 1x1 blocks
    def split_cond(state: tuple[jax.Array, jax.Array, jax.Array]) -> jax.Array:
        _, _, i = state
        return i < n - 1

    def split_body(
        state: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        H_c, R_c, i = state
        is_2x2 = ~_is_negligible_subdiag(H_c, i + 1, tol)

        def handle_2x2(args: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
            H_new, R_new = _split_real_2x2_block(args[0], args[1], i, tol)
            still_2x2 = ~_is_negligible_subdiag(H_new, i + 1, tol)
            return H_new, R_new, jnp.where(still_2x2, i + 2, i + 1)

        def handle_1x1(args: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array, jax.Array]:
            return args[0].at[i + 1, i].set(0.0), args[1], i + 1

        return jax.lax.cond(is_2x2, handle_2x2, handle_1x1, (H_c, R_c))

    H, R_all, _ = jax.lax.while_loop(split_cond, split_body, (H, R_all, jnp.int32(0)))
    H = jnp.triu(H, k=-1)

    if N > 1:
        return jnp.concatenate([R_all, H[None, :, :]], axis=0)
    return H[None, :, :]


def _extract_eigenvalues_from_schur_factors(
    schur_factors: jax.Array,
    tol: float = 1e-14,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Extract ``(multipliers, log_abs, phases)`` from periodic real Schur factors."""
    _, n, _ = schur_factors.shape
    if n == 0:
        return (
            jnp.zeros(0, dtype=jnp.complex128),
            jnp.zeros(0, dtype=jnp.float64),
            jnp.zeros(0, dtype=jnp.float64),
        )
    if n == 1:
        d_k = schur_factors[:, 0, 0]
        any_zero = jnp.any(d_k == 0.0)
        safe_abs = jnp.where(d_k == 0.0, 1.0, jnp.abs(d_k))
        log_m = jnp.where(any_zero, -jnp.inf, jnp.sum(jnp.log(safe_abs)))
        sign = jnp.where(any_zero, 1.0, jnp.prod(jnp.where(d_k < 0.0, -1.0, 1.0)))
        phase_val = jnp.where(sign < 0.0, jnp.pi, 0.0)
        mag = jnp.where(any_zero, 0.0, jnp.exp(log_m))
        mult_val = (sign * mag).astype(jnp.complex128)
        return jnp.atleast_1d(mult_val), jnp.atleast_1d(log_m), jnp.atleast_1d(phase_val)

    H = schur_factors[-1]

    log_abs_arr = jnp.zeros(n, dtype=jnp.float64)
    phases_arr = jnp.zeros(n, dtype=jnp.float64)
    mults_arr = jnp.zeros(n, dtype=jnp.complex128)

    def cond_fn(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> jax.Array:
        i, _, _, _ = state
        return i < n

    def body_fn(
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        i, la, ph, mu = state
        has_next = i + 1 < n
        is_2x2 = has_next & (~_is_negligible_subdiag(H, jnp.minimum(i + 1, n - 1), tol))

        def extract_1x1() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            d_k = schur_factors[:, i, i]
            any_zero = jnp.any(d_k == 0.0)
            safe_abs = jnp.where(d_k == 0.0, 1.0, jnp.abs(d_k))
            log_m = jnp.where(any_zero, -jnp.inf, jnp.sum(jnp.log(safe_abs)))
            sign = jnp.where(any_zero, 1.0, jnp.prod(jnp.where(d_k < 0.0, -1.0, 1.0)))
            phase_val = jnp.where(sign < 0.0, jnp.pi, 0.0)
            mag = jnp.where(any_zero, 0.0, jnp.exp(log_m))
            mult_val = (sign * mag).astype(jnp.complex128)
            return (
                i + 1,
                la.at[i].set(log_m),
                ph.at[i].set(phase_val),
                mu.at[i].set(mult_val),
            )

        def extract_2x2() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            idx = jnp.minimum(i, jnp.int32(n - 2))
            B_all = jax.lax.dynamic_slice(
                schur_factors, (jnp.int32(0), idx, idx), (schur_factors.shape[0], 2, 2)
            )
            P_hat, log_scale, is_zero = _normalized_2x2_product(B_all)
            half_tr = 0.5 * (P_hat[0, 0] + P_hat[1, 1])
            diff = 0.5 * (P_hat[0, 0] - P_hat[1, 1])
            disc = diff * diff + P_hat[0, 1] * P_hat[1, 0]

            # Exact log|det(B_k)| sum for complex conjugate pairs (avoids cancellation)
            dets_k = B_all[:, 0, 0] * B_all[:, 1, 1] - B_all[:, 0, 1] * B_all[:, 1, 0]
            any_det_zero = jnp.any(dets_k == 0.0)
            safe_dets = jnp.where(dets_k == 0.0, 1.0, jnp.abs(dets_k))
            log_m_cplx = jnp.where(
                is_zero | any_det_zero,
                -jnp.inf,
                0.5 * jnp.sum(jnp.log(safe_dets)),
            )
            im_part = jnp.sqrt(jnp.maximum(-disc, 0.0))
            theta_cplx = jnp.arctan2(im_part, half_tr)
            mag_cplx = jnp.where(is_zero | any_det_zero, 0.0, jnp.exp(log_m_cplx))
            mu1_cplx = mag_cplx * (jnp.cos(theta_cplx) + 1j * jnp.sin(theta_cplx))
            mu2_cplx = jnp.conj(mu1_cplx)

            # Fallback if a 2x2 block still had real eigenvalues
            sq = jnp.sqrt(jnp.maximum(disc, 0.0))
            l1_hat = half_tr + jnp.where(half_tr >= 0.0, sq, -sq)
            det_P = P_hat[0, 0] * P_hat[1, 1] - P_hat[0, 1] * P_hat[1, 0]
            l2_hat = jnp.where(
                jnp.abs(l1_hat) > 0.0,
                det_P / jnp.where(jnp.abs(l1_hat) > 0.0, l1_hat, 1.0),
                0.0,
            )
            l1_zero = is_zero | (l1_hat == 0.0)
            l2_zero = is_zero | (l2_hat == 0.0)
            log_m1_real = jnp.where(
                l1_zero, -jnp.inf, log_scale + jnp.log(jnp.where(l1_zero, 1.0, jnp.abs(l1_hat)))
            )
            log_m2_real = jnp.where(
                l2_zero, -jnp.inf, log_scale + jnp.log(jnp.where(l2_zero, 1.0, jnp.abs(l2_hat)))
            )
            ph1_real = jnp.where(l1_hat < 0.0, jnp.pi, 0.0)
            ph2_real = jnp.where(l2_hat < 0.0, jnp.pi, 0.0)
            mu1_real = jnp.where(l1_zero, 0.0, jnp.sign(l1_hat) * jnp.exp(log_m1_real)).astype(
                jnp.complex128
            )
            mu2_real = jnp.where(l2_zero, 0.0, jnp.sign(l2_hat) * jnp.exp(log_m2_real)).astype(
                jnp.complex128
            )

            is_cplx = disc < 0.0
            lm1 = jnp.where(is_cplx, log_m_cplx, log_m1_real)
            lm2 = jnp.where(is_cplx, log_m_cplx, log_m2_real)
            p1 = jnp.where(is_cplx, theta_cplx, ph1_real)
            p2 = jnp.where(is_cplx, -theta_cplx, ph2_real)
            m1_val = jnp.where(is_cplx, mu1_cplx, mu1_real)
            m2_val = jnp.where(is_cplx, mu2_cplx, mu2_real)

            return (
                i + 2,
                la.at[idx].set(lm1).at[idx + 1].set(lm2),
                ph.at[idx].set(p1).at[idx + 1].set(p2),
                mu.at[idx].set(m1_val).at[idx + 1].set(m2_val),
            )

        return jax.lax.cond(is_2x2, extract_2x2, extract_1x1)

    _, log_abs_arr, phases_arr, mults_arr = jax.lax.while_loop(
        cond_fn, body_fn, (jnp.int32(0), log_abs_arr, phases_arr, mults_arr)
    )
    order = jnp.argsort(-log_abs_arr)
    return mults_arr[order], log_abs_arr[order], phases_arr[order]


def periodic_schur_eigenvalues(
    factors: jax.Array,
    max_iters: int | None = None,
    tol: float = 1e-14,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute eigenvalues of ``M_{N-1} @ ... @ M_0`` stably via Periodic Real Schur Form.

    Returns ``(multipliers, log_abs, phases)`` sorted in descending order of ``log_abs``:

    - ``multipliers``: complex Floquet multipliers ``mu_i`` of shape ``(n,)`` (``complex128``)
    - ``log_abs``: exact ``log(|mu_i|)`` of shape ``(n,)`` (``float64``), overflow/underflow-free
    - ``phases``: argument ``arg(mu_i)`` in ``(-pi, pi]`` of shape ``(n,)`` (``float64``)
    """
    schur_factors = periodic_real_schur(factors, max_iters=max_iters, tol=tol)
    return _extract_eigenvalues_from_schur_factors(schur_factors, tol=tol)


def evaluate_floquet_diagnostics(
    M: jax.Array,
    tangent_p: jax.Array | None = None,
    rel_tol: float = 1e-4,
) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
    """Compute sorted Floquet multipliers, LPC/PD/NS test functions, and orbital stability."""
    n = int(M.shape[0])
    eigvals = jnp.linalg.eigvals(M)
    idx = jnp.argsort(-jnp.abs(eigvals))
    multipliers = eigvals[idx]

    idx_triv = jnp.argmin(jnp.abs(multipliers - 1.0))
    nontriv_mask = jnp.arange(n) != idx_triv
    is_stable = jnp.all(jnp.where(nontriv_mask, jnp.abs(multipliers) < 1.0, True))

    tf_pd = jnp.linalg.det(M + jnp.eye(n, dtype=jnp.float64))
    complex_mask = nontriv_mask & (
        jnp.abs(jnp.imag(multipliers)) > rel_tol * (1.0 + jnp.abs(multipliers))
    )
    has_complex = jnp.any(complex_mask)
    ns_vals = jnp.where(complex_mask, jnp.abs(multipliers) - 1.0, -jnp.inf)
    tf_ns = jnp.where(has_complex, jnp.max(ns_vals), jnp.nan)
    tf_lpc = (
        jnp.array(jnp.nan, dtype=jnp.float64)
        if tangent_p is None
        else jnp.squeeze(jnp.asarray(tangent_p, dtype=jnp.float64))
    )
    return multipliers, {"LPC": tf_lpc, "PD": tf_pd, "NS": tf_ns}, is_stable


def evaluate_floquet_diagnostics_from_factors(
    factors: jax.Array,
    tangent_p: jax.Array | None = None,
    rel_tol: float = 1e-4,
) -> tuple[jax.Array, dict[str, jax.Array], jax.Array]:
    """Compute Floquet diagnostics stably from interval factors ``M_k`` via Periodic Real Schur."""
    n = int(factors.shape[1])
    multipliers, log_abs, phases = periodic_schur_eigenvalues(factors)

    # Trivial multiplier mu = 1 corresponds to log|mu| = 0 and arg(mu) = 0
    idx_triv = jnp.argmin(jnp.hypot(log_abs, phases))
    nontriv_mask = jnp.arange(n) != idx_triv
    is_stable = jnp.all(jnp.where(nontriv_mask, log_abs < 0.0, True))

    # det(M + I) = prod_i (mu_i + 1); safe even when multipliers span huge ranges
    tf_pd = jnp.real(jnp.prod(multipliers + 1.0))
    complex_mask = nontriv_mask & (jnp.abs(jnp.sin(phases)) > rel_tol)
    has_complex = jnp.any(complex_mask)
    ns_vals = jnp.where(complex_mask, jnp.expm1(jnp.clip(log_abs, -700.0, 700.0)), -jnp.inf)
    tf_ns = jnp.where(has_complex, jnp.max(ns_vals), jnp.nan)
    tf_lpc = (
        jnp.array(jnp.nan, dtype=jnp.float64)
        if tangent_p is None
        else jnp.squeeze(jnp.asarray(tangent_p, dtype=jnp.float64))
    )
    return multipliers, {"LPC": tf_lpc, "PD": tf_pd, "NS": tf_ns}, is_stable


def _monodromy_variational(
    sol: PeriodicOrbitResult,
    p: float | jax.Array = 0.0,
    num_steps: int = _DEFAULT_VARIATIONAL_STEPS,
) -> jax.Array:
    """Compute the monodromy matrix M by integrating the variational equation via RK4."""
    n = sol.problem.dim
    T = jnp.squeeze(jnp.asarray(sol.period, dtype=jnp.float64))
    dt = 1.0 / num_steps
    jac_fn = jax.jacobian(lambda u_: sol.problem.fn(u_, p, **sol.problem.kwargs))

    def rk4_step(Phi: jax.Array, step: jax.Array) -> tuple[jax.Array, None]:
        t0 = step * dt
        A0 = T * jac_fn(sol.evaluate(t0))
        A_mid = T * jac_fn(sol.evaluate(t0 + 0.5 * dt))
        A1 = T * jac_fn(sol.evaluate(t0 + dt))

        k1 = A0 @ Phi
        k2 = A_mid @ (Phi + 0.5 * dt * k1)
        k3 = A_mid @ (Phi + 0.5 * dt * k2)
        k4 = A1 @ (Phi + dt * k3)
        return Phi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    Phi, _ = jax.lax.scan(rk4_step, jnp.eye(n, dtype=jnp.float64), jnp.arange(num_steps))
    return Phi


def compute_floquet_multipliers(
    sol: PeriodicOrbitResult,
    p: float | jax.Array = 0.0,
    method: str = "condensation",
) -> tuple[jax.Array, jax.Array]:
    """Compute Floquet multipliers and monodromy representation for a periodic orbit.

    Parameters
    ----------
    sol : PeriodicOrbitResult
        Converged periodic orbit solution.
    p : float or jax.Array, optional
        Parameter value at which to evaluate the variational Jacobian.
    method : {"condensation", "periodic_qr", "periodic_schur", "variational"}, optional
        - ``"condensation"``: explicitly multiplies condensed interval factors ``M_{N-1}...M_0``
          and returns ``(multipliers, M)``.
        - ``"periodic_qr"`` / ``"periodic_schur"``: applies Periodic Real Schur decomposition
          directly to the condensed interval factors without multiplying them, returning
          ``(multipliers, schur_factors)`` where ``schur_factors`` has shape ``(N, n, n)``.
        - ``"variational"``: integrates the variational ODE via RK4 and returns
          ``(multipliers, M)``.
    """
    if method == "condensation":
        factors = _monodromy_factors_from_sol(sol, p=p)
        eye_n = jnp.eye(sol.problem.dim, dtype=jnp.float64)
        M, _ = jax.lax.scan(lambda acc, M_k: (M_k @ acc, None), eye_n, factors)
        multipliers = jnp.linalg.eigvals(M)
        idx = jnp.argsort(-jnp.abs(multipliers))
        return multipliers[idx], M
    if method in ("periodic_qr", "periodic_schur"):
        factors = _monodromy_factors_from_sol(sol, p=p)
        schur_factors = periodic_real_schur(factors)
        multipliers, _, _ = _extract_eigenvalues_from_schur_factors(schur_factors)
        return multipliers, schur_factors
    if method == "variational":
        M = _monodromy_variational(sol, p=p)
        multipliers = jnp.linalg.eigvals(M)
        idx = jnp.argsort(-jnp.abs(multipliers))
        return multipliers[idx], M

    msg = (
        f"Unknown method: {method}. "
        "Choose 'condensation', 'periodic_qr', 'periodic_schur', or 'variational'."
    )
    raise ValueError(msg)


def compute_floquet_exponents(
    sol: PeriodicOrbitResult,
    p: float | jax.Array = 0.0,
    method: str = "periodic_qr",
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute real Floquet exponents ``sigma_i = log(|mu_i|) / T`` stably via Periodic Schur.

    Returns ``(exponents, multipliers, phases)`` sorted in descending order of ``exponents``.
    """
    T = jnp.squeeze(jnp.asarray(sol.period, dtype=jnp.float64))
    if method in ("periodic_qr", "periodic_schur"):
        factors = _monodromy_factors_from_sol(sol, p=p)
        multipliers, log_abs, phases = periodic_schur_eigenvalues(factors)
        return log_abs / T, multipliers, phases
    if method in ("condensation", "variational"):
        multipliers, _ = compute_floquet_multipliers(sol, p=p, method=method)
        mags = jnp.abs(multipliers)
        log_abs = jnp.where(mags > 0.0, jnp.log(jnp.where(mags > 0.0, mags, 1.0)), -jnp.inf)
        phases = jnp.angle(multipliers)
        return log_abs / T, multipliers, phases

    msg = (
        f"Unknown method: {method}. "
        "Choose 'periodic_qr', 'periodic_schur', 'condensation', or 'variational'."
    )
    raise ValueError(msg)
