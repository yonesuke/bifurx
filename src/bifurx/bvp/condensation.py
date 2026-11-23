from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp

from bifurx.homotopy.bordered import BorderedOperatorSolver, BorderedSolution

if TYPE_CHECKING:
    from bifurx.bvp.collocation import CollocationProblem


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
    M_cond: jax.Array
    lu_cond: jax.Array
    piv_cond: jax.Array
    use_lu_cond: jax.Array


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
    mesh_arr = jnp.asarray(mesh, dtype=jnp.float64)
    h = mesh_arr[1:] - mesh_arr[:-1]
    p_val = jnp.squeeze(jnp.asarray(p, dtype=jnp.float64))

    N, m, n = problem.num_intervals, problem.num_gauss_points, problem.dim
    D = jnp.asarray(problem.D, dtype=jnp.float64)
    C = jnp.asarray(problem.C, dtype=jnp.float64)
    weights = jnp.asarray(problem.weights, dtype=jnp.float64)
    eye_n = jnp.eye(n, dtype=jnp.float64)
    eye_m = jnp.eye(m, dtype=jnp.float64)

    f_all = jax.vmap(jax.vmap(lambda u_: problem.fn(u_, p_val, **problem.kwargs)))(u_gauss)
    Ju_all = jax.vmap(jax.vmap(jax.jacfwd(lambda u_: problem.fn(u_, p_val, **problem.kwargs))))(
        u_gauss
    )

    A_blocks = (D[None, :, 1:] / h[:, None, None])[:, :, :, None, None] * eye_n[
        None, None, None, :, :
    ] - T * eye_m[None, :, :, None, None] * Ju_all[:, :, None, :, :]
    A_all = jnp.swapaxes(A_blocks, 2, 3).reshape(N, m * n, m * n)
    B_all = ((D[None, :, 0] / h[:, None])[:, :, None, None] * eye_n[None, None, :, :]).reshape(
        N, m * n, n
    )
    q_T = (-f_all).reshape(N, m * n)

    lu_A, piv_A = jax.vmap(jax.scipy.linalg.lu_factor)(A_all)
    K_all = jax.vmap(lambda lu_k, piv_k, b_k: jax.scipy.linalg.lu_solve((lu_k, piv_k), b_k))(
        lu_A, piv_A, B_all
    )
    gamma_T = jax.vmap(lambda lu_k, piv_k, q_k: jax.scipy.linalg.lu_solve((lu_k, piv_k), q_k))(
        lu_A, piv_A, q_T
    )

    M_k = C[0] * eye_n[None, :, :] - jnp.einsum("j,Njab->Nab", C[1:], K_all.reshape(N, m, n, n))
    s_T = jnp.einsum("j,Nja->Na", C[1:], gamma_T.reshape(N, m, n))

    phi = (h[:, None, None] * weights[None, :, None] * dot_u_ref).reshape(N, m * n)
    phase_u = -jnp.einsum("Ni,Nia->Na", phi, K_all)
    phase_T = -jnp.sum(phi * gamma_T)

    n_mesh = N * n
    k_idx = jnp.arange(N)
    cont_mat = jnp.zeros((N, n, N, n), dtype=jnp.float64)
    cont_mat = cont_mat.at[k_idx, :, (k_idx + 1) % N, :].add(eye_n)
    cont_mat = cont_mat.at[k_idx, :, k_idx, :].add(-M_k)
    cont_mat = cont_mat.reshape(n_mesh, n_mesh)

    if c_X is not None:
        Jp_all = jax.vmap(
            jax.vmap(lambda u_: jax.jacfwd(lambda p_: problem.fn(u_, p_, **problem.kwargs))(p_val))
        )(u_gauss)
        q_p = (-T * Jp_all).reshape(N, m * n)
        gamma_p = jax.vmap(lambda lu_k, piv_k, q_k: jax.scipy.linalg.lu_solve((lu_k, piv_k), q_k))(
            lu_A, piv_A, q_p
        )
        s_p = jnp.einsum("j,Nja->Na", C[1:], gamma_p.reshape(N, m, n))
        phase_p = -jnp.sum(phi * gamma_p)

        c_m, c_g, c_T = problem.unpack_vars(c_X)
        c_gauss = c_g.reshape(N, m * n)
        arc_u = c_m - jnp.einsum("Ni,Nia->Na", c_gauss, K_all)
        arc_T = c_T - jnp.sum(c_gauss * gamma_T)
        d_scalar = 0.0 if d_p is None else jnp.squeeze(jnp.asarray(d_p, dtype=jnp.float64))
        arc_p = d_scalar - jnp.sum(c_gauss * gamma_p)

        M_cond = jnp.zeros((n_mesh + 2, n_mesh + 2), dtype=jnp.float64)
        M_cond = M_cond.at[:n_mesh, :n_mesh].set(cont_mat)
        M_cond = M_cond.at[:n_mesh, n_mesh].set(s_T.ravel())
        M_cond = M_cond.at[:n_mesh, n_mesh + 1].set(s_p.ravel())
        M_cond = M_cond.at[n_mesh, :n_mesh].set(phase_u.ravel())
        M_cond = M_cond.at[n_mesh, n_mesh].set(phase_T)
        M_cond = M_cond.at[n_mesh, n_mesh + 1].set(phase_p)
        M_cond = M_cond.at[n_mesh + 1, :n_mesh].set(arc_u.ravel())
        M_cond = M_cond.at[n_mesh + 1, n_mesh].set(arc_T)
        M_cond = M_cond.at[n_mesh + 1, n_mesh + 1].set(arc_p)
    else:
        gamma_p = None
        c_gauss = None
        M_cond = jnp.zeros((n_mesh + 1, n_mesh + 1), dtype=jnp.float64)
        M_cond = M_cond.at[:n_mesh, :n_mesh].set(cont_mat)
        M_cond = M_cond.at[:n_mesh, n_mesh].set(s_T.ravel())
        M_cond = M_cond.at[n_mesh, :n_mesh].set(phase_u.ravel())
        M_cond = M_cond.at[n_mesh, n_mesh].set(phase_T)

    lu_cond, piv_cond = jax.scipy.linalg.lu_factor(M_cond)
    u_diag = jnp.abs(jnp.diag(lu_cond))
    use_lu_cond = jnp.all(jnp.isfinite(lu_cond)) & (
        jnp.min(u_diag) > rcond * (jnp.max(u_diag) + 1e-30)
    )
    return CondensedCollocationFactor(
        lu_A=lu_A,
        piv_A=piv_A,
        K_all=K_all,
        gamma_T=gamma_T,
        gamma_p=gamma_p,
        phi=phi,
        c_gauss=c_gauss,
        C=C,
        M_cond=M_cond,
        lu_cond=lu_cond,
        piv_cond=piv_cond,
        use_lu_cond=use_lu_cond,
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
    rhs_cont = (r_cont + jnp.einsum("j,Nja->Na", fac.C[1:], r_hat.reshape(N, m, n))).ravel()
    rhs_phase = r_phase - jnp.sum(fac.phi * r_hat)

    if rhs_g is not None and fac.gamma_p is not None and fac.c_gauss is not None:
        rhs_arc = jnp.squeeze(rhs_g) - jnp.sum(fac.c_gauss * r_hat)
        rhs_cond = jnp.concatenate([rhs_cont, jnp.atleast_1d(rhs_phase), jnp.atleast_1d(rhs_arc)])
    else:
        rhs_cond = jnp.concatenate([rhs_cont, jnp.atleast_1d(rhs_phase)])

    y = jax.lax.cond(
        fac.use_lu_cond,
        lambda: jax.scipy.linalg.lu_solve((fac.lu_cond, fac.piv_cond), rhs_cond),
        lambda: jnp.linalg.lstsq(fac.M_cond, rhs_cond, rcond=1e-12)[0],
    )
    du_mesh = y[:n_mesh].reshape(N, n)
    dT = y[n_mesh]
    du_gauss = r_hat - jnp.einsum("Nia,Na->Ni", fac.K_all, du_mesh) - fac.gamma_T * dT
    if rhs_g is not None and fac.gamma_p is not None:
        dp = jnp.squeeze(y[n_mesh + 1])
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
