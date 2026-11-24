from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from bifurx.homotopy.bordered import (
    BorderedOperatorSolver,
    BorderedSolution,
    safe_lu_factor,
    safe_lu_solve,
)

if TYPE_CHECKING:
    from bifurx.bvp.collocation import CollocationProblem


class CyclicBlockBidiagonalFactor(NamedTuple):
    """Structured O(N n^3) orthogonal elimination factor for cyclic block-bidiagonal + border systems.

    Represents the factored system:
        -M_k u_k + u_{(k+1) mod N} + S_k w = r_k,   k = 0, ..., N - 1
        sum_{k=0}^{N-1} P_k u_k + D_border w = g
    """

    Q: jax.Array
    R: jax.Array
    E: jax.Array
    G: jax.Array
    W: jax.Array
    M_red: jax.Array
    lu_red: jax.Array
    piv_red: jax.Array
    use_lu_red: jax.Array


def factor_cyclic_block_bidiagonal(
    M_blocks: jax.Array,
    S_blocks: jax.Array,
    P_blocks: jax.Array,
    D_border: jax.Array,
    *,
    rcond: float = 1e-12,
) -> CyclicBlockBidiagonalFactor:
    """Factor a cyclic block-bidiagonal system with dense border rows/columns in O(N n^3) time.

    Uses sequential block-QR row reduction (Wright 1992 / Ascher-Mattheij-Russell) to eliminate
    interior mesh nodes u_1, ..., u_{N-1} without forming unscaled monodromy products, reducing
    the (N*n + b) x (N*n + b) system to an (n + b) x (n + b) system for (u_0, w).
    """
    N, n, _ = M_blocks.shape
    b = S_blocks.shape[2]
    eye_n = jnp.eye(n, dtype=jnp.float64)

    if N == 1:
        Q = jnp.empty((0, 2 * n, 2 * n), dtype=jnp.float64)
        R = jnp.empty((0, n, n), dtype=jnp.float64)
        E = jnp.empty((0, n, n), dtype=jnp.float64)
        G = jnp.empty((0, n, b), dtype=jnp.float64)
        W = jnp.empty((0, b, n), dtype=jnp.float64)
        M_red = jnp.block([[eye_n - M_blocks[0], S_blocks[0]], [P_blocks[0], D_border]])
    else:
        # P_next[k] supplies P_{k+2} for k = 0..N-3; for k = N-2, u_{(k+2)%N} = u_0 is folded
        # into P0 after the scan via P0_total = P0_fin + P_cur_fin.
        P_next = jnp.concatenate(
            [P_blocks[2:], jnp.zeros((1, b, n), dtype=jnp.float64)],
            axis=0,
        )
        init_carry = (-M_blocks[0], eye_n, S_blocks[0], P_blocks[0], P_blocks[1], D_border)

        def _elim_step(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            x: tuple[jax.Array, jax.Array, jax.Array],
        ) -> tuple[
            tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ]:
            A_cur, B_cur, S_cur, P0_cur, P_cur, D_cur = carry
            M_next, S_next, P_nxt = x

            X_k = jnp.concatenate([B_cur, -M_next], axis=0)
            Q_k, R_full = jnp.linalg.qr(X_k, mode="complete")
            R_k = R_full[:n, :]

            EA = Q_k[:n, :].T @ A_cur
            E_k = EA[:n, :]
            A_new = EA[n:, :]

            F_k = Q_k[n:, :n].T
            B_new = Q_k[n:, n:].T

            GS = Q_k.T @ jnp.concatenate([S_cur, S_next], axis=0)
            G_k = GS[:n, :]
            S_new = GS[n:, :]

            # Solve R_k^T W_k^T = P_cur^T  <=>  W_k R_k = P_cur
            W_k = jax.scipy.linalg.solve_triangular(R_k, P_cur.T, trans="T", lower=False).T
            P0_new = P0_cur - W_k @ E_k
            P_cur_new = P_nxt - W_k @ F_k
            D_new = D_cur - W_k @ G_k

            return (A_new, B_new, S_new, P0_new, P_cur_new, D_new), (Q_k, R_k, E_k, G_k, W_k)

        (A_fin, B_fin, S_fin, P0_fin, P_cur_fin, D_fin), (Q, R, E, G, W) = jax.lax.scan(
            _elim_step,
            init_carry,
            (M_blocks[1:], S_blocks[1:], P_next),
        )
        P0_total = P0_fin + P_cur_fin
        M_red = jnp.block([[A_fin + B_fin, S_fin], [P0_total, D_fin]])

    lu_red, piv_red, use_lu_red = safe_lu_factor(M_red, rcond=rcond)
    return CyclicBlockBidiagonalFactor(
        Q=Q,
        R=R,
        E=E,
        G=G,
        W=W,
        M_red=M_red,
        lu_red=lu_red,
        piv_red=piv_red,
        use_lu_red=use_lu_red,
    )


def solve_cyclic_block_bidiagonal(
    cyc_fac: CyclicBlockBidiagonalFactor,
    r_cont: jax.Array,
    g_border: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solve a factored cyclic block-bidiagonal + border system in O(N n^2 + (n + b)^3) time."""
    N, n = r_cont.shape

    if N == 1:
        rhs_red = jnp.concatenate([r_cont[0], g_border], axis=0)
        y_red = safe_lu_solve(cyc_fac.M_red, cyc_fac.lu_red, cyc_fac.piv_red, cyc_fac.use_lu_red, rhs_red)
        return y_red[:n][None, :], y_red[n:]

    def _fwd_step(
        carry: tuple[jax.Array, jax.Array],
        x: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        r_cur, g_cur = carry
        Q_k, W_k, r_next = x
        r_rot = Q_k.T @ jnp.concatenate([r_cur, r_next], axis=0)
        r_top = r_rot[:n]
        r_new = r_rot[n:]
        g_new = g_cur - W_k @ r_top
        return (r_new, g_new), r_top

    (r_fin, g_fin), r_top_all = jax.lax.scan(
        _fwd_step,
        (r_cont[0], g_border),
        (cyc_fac.Q, cyc_fac.W, r_cont[1:]),
    )

    rhs_red = jnp.concatenate([r_fin, g_fin], axis=0)
    y_red = safe_lu_solve(cyc_fac.M_red, cyc_fac.lu_red, cyc_fac.piv_red, cyc_fac.use_lu_red, rhs_red)
    u0 = y_red[:n]
    w = y_red[n:]

    def _bwd_step(
        u_next: jax.Array,
        x: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        Q_k, R_k, E_k, G_k, r_top_k = x
        F_k = Q_k[n:, :n].T
        rhs_u = r_top_k - E_k @ u0 - F_k @ u_next - G_k @ w
        u_k1 = jax.scipy.linalg.solve_triangular(R_k, rhs_u, lower=False)
        return u_k1, u_k1

    _, u_rest = jax.lax.scan(
        _bwd_step,
        u0,
        (cyc_fac.Q, cyc_fac.R, cyc_fac.E, cyc_fac.G, r_top_all),
        reverse=True,
    )
    u_mesh = jnp.concatenate([u0[None, :], u_rest], axis=0)
    return u_mesh, w


class CondensedCollocationFactor(NamedTuple):
    """Reusable de Boor-Swartz block-condensed factorization for collocation BVP systems."""

    lu_A: jax.Array
    piv_A: jax.Array
    K_all: jax.Array
    gamma_T: jax.Array
    gamma_p: jax.Array | None
    phi: jax.Array
    c_gauss: jax.Array | None
    C: jax.Array
    cyclic: CyclicBlockBidiagonalFactor


def condense_local_blocks(
    problem: CollocationProblem,
    h: jax.Array,
    T: jax.Array,
    Ju_all: jax.Array,
    *,
    adjoint: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Assemble and LU-condense per-interval collocation blocks for primal or adjoint BVPs.

    Returns ``(lu_A, piv_A, K_all, M_k)`` where ``K_all`` maps interval mesh nodes to Gauss
    collocation stages (``u_gauss,k = -K_k u_mesh,k + ...``) and ``M_k`` is the interval
    transition matrix (``u_end,k = M_k u_mesh,k + ...``).
    """
    N, m, n = problem.num_intervals, problem.num_gauss_points, problem.dim
    D = jnp.asarray(problem.D, dtype=jnp.float64)
    C = jnp.asarray(problem.C, dtype=jnp.float64)
    eye_n = jnp.eye(n, dtype=jnp.float64)
    eye_m = jnp.eye(m, dtype=jnp.float64)

    if adjoint:
        Ju_eff = jnp.swapaxes(Ju_all, -2, -1)
        rhs_sign = 1.0
    else:
        Ju_eff = Ju_all
        rhs_sign = -1.0

    A_blocks = (D[None, :, 1:] / h[:, None, None])[:, :, :, None, None] * eye_n[
        None, None, None, :, :
    ] + (rhs_sign * T) * eye_m[None, :, :, None, None] * Ju_eff[:, :, None, :, :]
    A_all = jnp.swapaxes(A_blocks, 2, 3).reshape(N, m * n, m * n)
    B_all = ((D[None, :, 0] / h[:, None])[:, :, None, None] * eye_n[None, None, :, :]).reshape(
        N, m * n, n
    )

    lu_A, piv_A = jax.vmap(jax.scipy.linalg.lu_factor)(A_all)
    K_all = jax.vmap(lambda lu_k, piv_k, b_k: jax.scipy.linalg.lu_solve((lu_k, piv_k), b_k))(
        lu_A, piv_A, B_all
    )
    M_k = C[0] * eye_n[None, :, :] - jnp.einsum("j,Njab->Nab", C[1:], K_all.reshape(N, m, n, n))
    return lu_A, piv_A, K_all, M_k


def _condense_collocation_blocks(
    problem: CollocationProblem,
    T: jax.Array,
    mesh: jax.Array,
    dot_u_ref: jax.Array,
    Ju_all: jax.Array,
    q_scalar: jax.Array,
    *,
    q_param: jax.Array | None = None,
    c_X: jax.Array | None = None,
    d_p: float | jax.Array | None = None,
    rcond: float = 1e-12,
) -> CondensedCollocationFactor:
    """Shared de Boor-Swartz block condensation for 1-parameter and fixed-period 2-parameter BVPs."""
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    h = mesh_arr[1:] - mesh_arr[:-1]
    N, m, n = problem.num_intervals, problem.num_gauss_points, problem.dim
    C = jnp.asarray(problem.C, dtype=jnp.float64)
    weights = jnp.asarray(problem.weights, dtype=jnp.float64)

    lu_A, piv_A, K_all, M_k = condense_local_blocks(problem, h, T, Ju_all, adjoint=False)
    q_sc = q_scalar.reshape(N, m * n)
    gamma_T = jax.vmap(lambda lu_k, piv_k, q_k: jax.scipy.linalg.lu_solve((lu_k, piv_k), q_k))(
        lu_A, piv_A, q_sc
    )
    s_T = jnp.einsum("j,Nja->Na", C[1:], gamma_T.reshape(N, m, n))

    phi = (h[:, None, None] * weights[None, :, None] * dot_u_ref).reshape(N, m * n)
    phase_u = -jnp.einsum("Ni,Nia->Na", phi, K_all)
    phase_T = -jnp.sum(phi * gamma_T)

    if c_X is not None and q_param is not None:
        q_p = q_param.reshape(N, m * n)
        gamma_p = jax.vmap(lambda lu_k, piv_k, q_k: jax.scipy.linalg.lu_solve((lu_k, piv_k), q_k))(
            lu_A, piv_A, q_p
        )
        s_p = jnp.einsum("j,Nja->Na", C[1:], gamma_p.reshape(N, m, n))
        phase_p = -jnp.sum(phi * gamma_p)

        c_m, c_g, c_T = problem.unpack_vars(c_X)
        c_gauss = c_g.reshape(N, m * n)
        arc_u = c_m - jnp.einsum("Ni,Nia->Na", c_gauss, K_all)
        arc_T = c_T - jnp.sum(c_gauss * gamma_T)
        d_scalar = (
            jnp.array(0.0, dtype=jnp.float64)
            if d_p is None
            else jnp.squeeze(jnp.asarray(d_p, dtype=jnp.float64))
        )
        arc_p = d_scalar - jnp.sum(c_gauss * gamma_p)

        S_blocks = jnp.stack([s_T, s_p], axis=-1)
        P_blocks = jnp.stack([phase_u, arc_u], axis=1)
        D_border = jnp.array([[phase_T, phase_p], [arc_T, arc_p]], dtype=jnp.float64)
    else:
        gamma_p = None
        c_gauss = None
        S_blocks = s_T[:, :, None]
        P_blocks = phase_u[:, None, :]
        D_border = jnp.array([[phase_T]], dtype=jnp.float64)

    cyclic = factor_cyclic_block_bidiagonal(M_k, S_blocks, P_blocks, D_border, rcond=rcond)
    return CondensedCollocationFactor(
        lu_A=lu_A,
        piv_A=piv_A,
        K_all=K_all,
        gamma_T=gamma_T,
        gamma_p=gamma_p,
        phi=phi,
        c_gauss=c_gauss,
        C=C,
        cyclic=cyclic,
    )


def factor_condensed_collocation(
    problem: CollocationProblem,
    X: jax.Array,
    p: float | jax.Array,
    mesh: jax.Array,
    dot_u_ref: jax.Array,
    *,
    c_X: jax.Array | None = None,
    d_p: float | jax.Array | None = None,
    rcond: float = 1e-12,
) -> CondensedCollocationFactor:
    """Factorize collocation BVP Jacobian (square or bordered) via block condensation."""
    _, u_gauss, T = problem.unpack_vars(X)
    p_val = jnp.squeeze(jnp.asarray(p, dtype=jnp.float64))

    f_all = jax.vmap(jax.vmap(lambda u_: problem.fn(u_, p_val, **problem.kwargs)))(u_gauss)
    Ju_all = jax.vmap(jax.vmap(jax.jacfwd(lambda u_: problem.fn(u_, p_val, **problem.kwargs))))(
        u_gauss
    )
    q_param = (
        -T
        * jax.vmap(
            jax.vmap(lambda u_: jax.jacfwd(lambda p_: problem.fn(u_, p_, **problem.kwargs))(p_val))
        )(u_gauss)
        if c_X is not None
        else None
    )
    return _condense_collocation_blocks(
        problem,
        T=T,
        mesh=mesh,
        dot_u_ref=dot_u_ref,
        Ju_all=Ju_all,
        q_scalar=-f_all,
        q_param=q_param,
        c_X=c_X,
        d_p=d_p,
        rcond=rcond,
    )


def _solve_condensed_core(
    fac: CondensedCollocationFactor,
    rhs_X: jax.Array,
    rhs_g: jax.Array | None,
) -> tuple[jax.Array, jax.Array | None]:
    N, m_n, n = fac.K_all.shape
    m = m_n // n
    n_coll = N * m_n
    n_mesh = N * n

    rhs_arr = jnp.asarray(rhs_X, dtype=jnp.float64).ravel()
    r_coll = rhs_arr[:n_coll].reshape(N, m_n)
    r_cont = rhs_arr[n_coll : n_coll + n_mesh].reshape(N, n)
    r_phase = rhs_arr[-1]

    r_hat = jax.vmap(lambda lu_k, piv_k, r_k: jax.scipy.linalg.lu_solve((lu_k, piv_k), r_k))(
        fac.lu_A, fac.piv_A, r_coll
    )
    rhs_cont = r_cont + jnp.einsum("j,Nja->Na", fac.C[1:], r_hat.reshape(N, m, n))
    rhs_phase = r_phase - jnp.sum(fac.phi * r_hat)

    if rhs_g is not None and fac.gamma_p is not None and fac.c_gauss is not None:
        rhs_arc = jnp.squeeze(rhs_g) - jnp.sum(fac.c_gauss * r_hat)
        g_border = jnp.stack([rhs_phase, rhs_arc])
    else:
        g_border = jnp.atleast_1d(rhs_phase)

    du_mesh, w = solve_cyclic_block_bidiagonal(fac.cyclic, rhs_cont, g_border)
    dT = w[0]
    du_gauss = r_hat - jnp.einsum("Nia,Na->Ni", fac.K_all, du_mesh) - fac.gamma_T * dT
    if rhs_g is not None and fac.gamma_p is not None:
        dp = jnp.squeeze(w[1])
        du_gauss = du_gauss - fac.gamma_p * dp
        dX = jnp.concatenate([du_mesh.ravel(), du_gauss.ravel(), jnp.atleast_1d(dT)])
        return dX, dp

    dX = jnp.concatenate([du_mesh.ravel(), du_gauss.ravel(), jnp.atleast_1d(dT)])
    return dX, None


def solve_condensed_collocation(
    fac: CondensedCollocationFactor,
    rhs_X: jax.Array,
) -> jax.Array:
    """Solve the square collocation Newton system reusing a `CondensedCollocationFactor`."""
    dX, _ = _solve_condensed_core(fac, rhs_X, None)
    return dX


def solve_condensed_bordered(
    fac: CondensedCollocationFactor,
    rhs_X: jax.Array,
    rhs_g: float | jax.Array,
) -> BorderedSolution:
    """Solve the bordered collocation system reusing a `CondensedCollocationFactor`."""
    rhs_g_arr = jnp.squeeze(jnp.asarray(rhs_g, dtype=jnp.float64))
    dX, dp = _solve_condensed_core(fac, rhs_X, rhs_g_arr)
    assert dp is not None
    return BorderedSolution(x=dX, y=dp)


def make_collocation_bordered_solver(
    problem: CollocationProblem,
    mesh: jax.Array,
    dot_u_ref: jax.Array,
) -> BorderedOperatorSolver:
    """Build a block-condensed `BorderedOperatorSolver` for a collocation BVP."""

    def factor(
        X: jax.Array, p: jax.Array, c: jax.Array, d: jax.Array
    ) -> CondensedCollocationFactor:
        return factor_condensed_collocation(problem, X, p, mesh, dot_u_ref, c_X=c, d_p=d)

    def solve(fac: CondensedCollocationFactor, f: jax.Array, g: jax.Array) -> BorderedSolution:
        return solve_condensed_bordered(fac, f, g)

    return BorderedOperatorSolver(factor=factor, solve=solve)
