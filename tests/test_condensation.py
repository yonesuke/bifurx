from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bifurx.bvp.collocation import (
    CollocationProblem,
    solve_periodic_orbit,
)
from bifurx.bvp.condensation import (
    CondensedCollocationFactor,
    CyclicBlockBidiagonalFactor,
    factor_condensed_collocation,
    factor_cyclic_block_bidiagonal,
    make_collocation_bordered_solver,
    solve_condensed_bordered,
    solve_condensed_collocation,
    solve_cyclic_block_bidiagonal,
)
from bifurx.bvp.mesh import redistribute_mesh
from bifurx.homotopy.bordered import solve_bordered_system


def _assemble_dense_cyclic_bordered(
    M_blocks: jax.Array,
    S_blocks: jax.Array,
    P_blocks: jax.Array,
    D_border: jax.Array,
) -> jax.Array:
    """Reference dense (N*n + b) x (N*n + b) assembly for testing."""
    N, n, _ = M_blocks.shape
    b = S_blocks.shape[2]
    n_mesh = N * n
    eye_n = jnp.eye(n, dtype=jnp.float64)
    k_idx = jnp.arange(N)

    cont = jnp.zeros((N, n, N, n), dtype=jnp.float64)
    cont = cont.at[k_idx, :, (k_idx + 1) % N, :].add(eye_n)
    cont = cont.at[k_idx, :, k_idx, :].add(-M_blocks)
    cont = cont.reshape(n_mesh, n_mesh)

    S_mat = S_blocks.reshape(n_mesh, b)
    P_mat = jnp.transpose(P_blocks, (1, 0, 2)).reshape(b, n_mesh)

    return jnp.block([[cont, S_mat], [P_mat, D_border]])


def _f_vdp(u: jax.Array, mu: float | jax.Array) -> jax.Array:
    x, y = u[0], u[1]
    return jnp.array([y, mu * (1.0 - x * x) * y - x])


def _f_stuart_landau(u: jax.Array, mu: float | jax.Array) -> jax.Array:
    x, y = u[0], u[1]
    r2 = x * x + y * y
    return jnp.array([mu * x - y - x * r2, x + mu * y - y * r2])


@pytest.mark.parametrize("N", [1, 2, 3, 5, 17, 25, 64, 100])
@pytest.mark.parametrize("n", [1, 2, 3, 6])
@pytest.mark.parametrize("b", [1, 2])
def test_cyclic_block_bidiagonal_arbitrary_sizes(N: int, n: int, b: int):
    """Exact agreement against dense solve for arbitrary N (non-power-of-2), n, and b."""
    key = jax.random.PRNGKey(1000 * N + 100 * n + b)
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    # Mildly conditioned random blocks around identity to mimic discrete ODE flow maps
    M_blocks = jnp.eye(n, dtype=jnp.float64)[None, :, :] + 0.15 * jax.random.normal(
        k1, (N, n, n), dtype=jnp.float64
    )
    S_blocks = jax.random.normal(k2, (N, n, b), dtype=jnp.float64)
    P_blocks = jax.random.normal(k3, (N, b, n), dtype=jnp.float64)
    D_border = jax.random.normal(k4, (b, b), dtype=jnp.float64)
    r_cont = jax.random.normal(k5, (N, n), dtype=jnp.float64)
    g_border = jax.random.normal(k6, (b,), dtype=jnp.float64)

    fac = factor_cyclic_block_bidiagonal(M_blocks, S_blocks, P_blocks, D_border)
    assert isinstance(fac, CyclicBlockBidiagonalFactor)
    u_mesh, w = solve_cyclic_block_bidiagonal(fac, r_cont, g_border)

    # Verify residual directly in O(N n^2) for all sizes
    u_next = jnp.roll(u_mesh, shift=-1, axis=0)
    res_cont = u_next - jnp.einsum("Nij,Nj->Ni", M_blocks, u_mesh) + jnp.einsum(
        "Nib,b->Ni", S_blocks, w
    ) - r_cont
    res_border = jnp.einsum("Nbi,Ni->b", P_blocks, u_mesh) + D_border @ w - g_border
    rhs_norm = float(jnp.linalg.norm(jnp.concatenate([r_cont.ravel(), g_border])))
    sol_norm = float(jnp.linalg.norm(jnp.concatenate([u_mesh.ravel(), w])))
    res_norm = float(jnp.linalg.norm(jnp.concatenate([res_cont.ravel(), res_border])))
    assert res_norm <= 1e-11 * (rhs_norm + sol_norm + 1.0)

    # Compare against dense LU for moderate sizes where dense assembly is fast
    if N * n <= 160:
        A_dense = _assemble_dense_cyclic_bordered(M_blocks, S_blocks, P_blocks, D_border)
        rhs_dense = jnp.concatenate([r_cont.ravel(), g_border])
        sol_dense = jnp.linalg.solve(A_dense, rhs_dense)
        sol_cyc = jnp.concatenate([u_mesh.ravel(), w])
        assert jnp.allclose(sol_cyc, sol_dense, rtol=1e-10, atol=1e-10)


def test_cyclic_solver_singular_continuity_floquet_one():
    """Verify solver handles exact 1D nullspace in C_cont (Floquet multiplier = 1)."""
    N, n = 19, 3
    key = jax.random.PRNGKey(77)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # Build M_k such that prod_k M_k has an exact eigenvalue 1 along e_0:
    # Each M_k = diag(1.0, 0.92, 1.04) conjugated by random orthogonal shifts Q_k^T D Q_{k-1}
    Qs = jax.vmap(lambda k: jnp.linalg.qr(jax.random.normal(k, (n, n), dtype=jnp.float64))[0])(
        jax.random.split(k1, N)
    )
    D_step = jnp.diag(jnp.array([1.0, 0.92, 1.04], dtype=jnp.float64))
    Q_next = jnp.roll(Qs, shift=-1, axis=0)
    M_blocks = jnp.einsum("Nij,jk,Nlk->Nil", Q_next, D_step, Qs)

    # Confirm C_cont alone is singular (smallest singular value ~ 0)
    A_cont = _assemble_dense_cyclic_bordered(
        M_blocks,
        jnp.zeros((N, n, 1), dtype=jnp.float64),
        jnp.zeros((N, 1, n), dtype=jnp.float64),
        jnp.zeros((1, 1), dtype=jnp.float64),
    )[: N * n, : N * n]
    svals = jnp.linalg.svd(A_cont, compute_uv=False)
    assert float(svals[-1]) < 1e-12

    for b in (1, 2):
        S_blocks = jax.random.normal(k2, (N, n, b), dtype=jnp.float64)
        P_blocks = jax.random.normal(k3, (N, b, n), dtype=jnp.float64)
        D_border = jax.random.normal(k4, (b, b), dtype=jnp.float64)
        r_cont = jax.random.normal(k5, (N, n), dtype=jnp.float64)
        g_border = jnp.arange(1, b + 1, dtype=jnp.float64)

        fac = factor_cyclic_block_bidiagonal(M_blocks, S_blocks, P_blocks, D_border)
        assert bool(fac.use_lu_red)
        u_mesh, w = solve_cyclic_block_bidiagonal(fac, r_cont, g_border)

        A_dense = _assemble_dense_cyclic_bordered(M_blocks, S_blocks, P_blocks, D_border)
        sol_dense = jnp.linalg.solve(A_dense, jnp.concatenate([r_cont.ravel(), g_border]))
        sol_cyc = jnp.concatenate([u_mesh.ravel(), w])
        assert jnp.allclose(sol_cyc, sol_dense, rtol=1e-11, atol=1e-11)


def test_condensed_collocation_exact_periodic_orbit_nullspace():
    """At an exact converged periodic orbit (C_cont singular), square and bordered solves match dense."""
    N, m = 15, 4
    prob = CollocationProblem(_f_stuart_landau, dim=2, num_intervals=N, num_gauss_points=m)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False, dtype=jnp.float64)
    u_init = jnp.stack([jnp.cos(t_grid), jnp.sin(t_grid)], axis=1)
    mu = jnp.array(1.0, dtype=jnp.float64)

    sol = solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=mu, tol=1e-12)
    assert bool(sol.converged)

    X = prob.pack_vars(sol.u_mesh, sol.u_gauss, sol.period)
    mesh = sol.mesh
    h = mesh[1:] - mesh[:-1]
    U_ref = jnp.concatenate([sol.u_mesh[:, None, :], sol.u_gauss], axis=1)
    dot_u_ref = jnp.einsum("mj,Njd->Nmd", prob.D, U_ref) / h[:, None, None]

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(123), 4)
    rhs_X = jax.random.normal(k1, shape=X.shape, dtype=jnp.float64)
    c_X = jax.random.normal(k2, shape=X.shape, dtype=jnp.float64)
    d_p = jax.random.normal(k3, shape=(), dtype=jnp.float64)
    rhs_g = jax.random.normal(k4, shape=(), dtype=jnp.float64)

    J_dense = jax.jacfwd(lambda x_: prob.residual(x_, mu, mesh, dot_u_ref=dot_u_ref))(X)
    Jp_dense = jax.jacfwd(lambda p_: prob.residual(X, p_, mesh, dot_u_ref=dot_u_ref))(mu)

    # Square solve
    dX_dense = jnp.linalg.solve(J_dense, rhs_X)
    fac_sq = factor_condensed_collocation(prob, X, mu, mesh, dot_u_ref)
    dX_cond = solve_condensed_collocation(fac_sq, rhs_X)
    assert jnp.allclose(dX_cond, dX_dense, rtol=1e-10, atol=1e-10)

    # Bordered solve
    sol_dense = solve_bordered_system(J_dense, Jp_dense, c_X, d_p, rhs_X, rhs_g, method="direct")
    fac_bd = factor_condensed_collocation(prob, X, mu, mesh, dot_u_ref, c_X=c_X, d_p=d_p)
    sol_cond = solve_condensed_bordered(fac_bd, rhs_X, rhs_g)
    assert jnp.allclose(sol_cond.z, sol_dense.z, rtol=1e-10, atol=1e-10)


def test_stiff_cyclic_dichotomy_stability():
    """Verify orthogonal block-QR stays accurate when prod(M_k) has extreme dichotomy (1e60 / 1e-60)."""
    N, n, b = 60, 3, 2
    key = jax.random.PRNGKey(999)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # Per-step eigenvalues: 10.0 (grows to 1e60), 0.1 (shrinks to 1e-60), 1.0 (neutral/periodic)
    # Naive forward product M_{N-1}...M_0 would have condition number ~1e120 and lose all digits.
    Qs = jax.vmap(lambda k: jnp.linalg.qr(jax.random.normal(k, (n, n), dtype=jnp.float64))[0])(
        jax.random.split(k1, N)
    )
    D_step = jnp.diag(jnp.array([10.0, 0.1, 1.0], dtype=jnp.float64))
    Q_next = jnp.roll(Qs, shift=-1, axis=0)
    M_blocks = jnp.einsum("Nij,jk,Nlk->Nil", Q_next, D_step, Qs)

    S_blocks = jax.random.normal(k2, (N, n, b), dtype=jnp.float64)
    P_blocks = jax.random.normal(k3, (N, b, n), dtype=jnp.float64)
    D_border = jax.random.normal(k4, (b, b), dtype=jnp.float64)

    # Manufactured ground truth to verify forward error directly
    u_true = jax.random.normal(k5, (N, n), dtype=jnp.float64)
    w_true = jnp.array([1.25, -0.75], dtype=jnp.float64)
    r_cont = (
        jnp.roll(u_true, shift=-1, axis=0)
        - jnp.einsum("Nij,Nj->Ni", M_blocks, u_true)
        + jnp.einsum("Nib,b->Ni", S_blocks, w_true)
    )
    g_border = jnp.einsum("Nbi,Ni->b", P_blocks, u_true) + D_border @ w_true

    fac = factor_cyclic_block_bidiagonal(M_blocks, S_blocks, P_blocks, D_border)
    assert jnp.all(jnp.isfinite(fac.M_red))
    u_sol, w_sol = solve_cyclic_block_bidiagonal(fac, r_cont, g_border)

    assert jnp.allclose(u_sol, u_true, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(w_sol, w_true, rtol=1e-11, atol=1e-11)


def test_stiff_van_der_pol_periodic_orbit():
    """Verify condensed cyclic solver converges and matches dense Jacobian on stiff Van der Pol (mu=10)."""
    N, m = 40, 4
    prob = CollocationProblem(_f_vdp, dim=2, num_intervals=N, num_gauss_points=m)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False, dtype=jnp.float64)
    u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)

    sol = solve_periodic_orbit(prob, u_init, T_init=6.66, p=1.0, tol=1e-10)
    assert bool(sol.converged)

    for mu_val in (2.0, 3.5, 5.0, 7.5, 10.0):
        sol = solve_periodic_orbit(
            prob,
            sol.u_mesh,
            T_init=sol.period,
            p=mu_val,
            mesh=sol.mesh,
            u_gauss_init=sol.u_gauss,
            tol=1e-10,
            max_iters=30,
        )
        assert bool(sol.converged), f"Failed to converge at mu={mu_val}"
        sol = redistribute_mesh(sol, p=mu_val, tol=1e-10)
        assert bool(sol.converged), f"Failed to converge after remesh at mu={mu_val}"

    sol_stiff = sol
    mu_stiff = jnp.array(10.0, dtype=jnp.float64)
    # Known asymptotic period of Van der Pol at mu=10 is ~19.08
    assert 18.5 < float(sol_stiff.period) < 19.5

    # Verify condensed solve matches dense Jacobian solve at the stiff mu=10 orbit
    X = prob.pack_vars(sol_stiff.u_mesh, sol_stiff.u_gauss, sol_stiff.period)
    mesh = sol_stiff.mesh
    h = mesh[1:] - mesh[:-1]
    U_ref = jnp.concatenate([sol_stiff.u_mesh[:, None, :], sol_stiff.u_gauss], axis=1)
    dot_u_ref = jnp.einsum("mj,Njd->Nmd", prob.D, U_ref) / h[:, None, None]

    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(555), 4)
    rhs_X = jax.random.normal(k1, shape=X.shape, dtype=jnp.float64)
    c_X = jax.random.normal(k2, shape=X.shape, dtype=jnp.float64)
    d_p = jax.random.normal(k3, shape=(), dtype=jnp.float64)
    rhs_g = jax.random.normal(k4, shape=(), dtype=jnp.float64)

    J_dense = jax.jacfwd(lambda x_: prob.residual(x_, mu_stiff, mesh, dot_u_ref=dot_u_ref))(X)
    Jp_dense = jax.jacfwd(lambda p_: prob.residual(X, p_, mesh, dot_u_ref=dot_u_ref))(mu_stiff)

    dX_dense = jnp.linalg.solve(J_dense, rhs_X)
    fac_sq = factor_condensed_collocation(prob, X, mu_stiff, mesh, dot_u_ref)
    dX_cond = solve_condensed_collocation(fac_sq, rhs_X)
    assert jnp.allclose(dX_cond, dX_dense, rtol=1e-9, atol=1e-9)

    sol_dense = solve_bordered_system(J_dense, Jp_dense, c_X, d_p, rhs_X, rhs_g, method="direct")
    solver = make_collocation_bordered_solver(prob, mesh, dot_u_ref)
    sol_cond = solver.solve(solver.factor(X, mu_stiff, c_X, d_p), rhs_X, rhs_g)
    assert jnp.allclose(sol_cond.z, sol_dense.z, rtol=1e-9, atol=1e-9)


def test_jit_compatibility_and_no_dense_nn_matrix():
    """Verify jax.jit works end-to-end and no O((N*n)^2) matrix is materialized in factor or jaxpr."""
    N, m, n = 64, 3, 4
    n_mesh = N * n  # 256 -> dense M_cond would be 257 x 257 = 66,049 elements

    def f_4d(u: jax.Array, p: float | jax.Array) -> jax.Array:
        return jnp.array([u[1], -u[0] + p * u[2], u[3], -u[2] - 0.1 * u[0]])

    prob = CollocationProblem(f_4d, dim=n, num_intervals=N, num_gauss_points=m)
    mesh = jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float64)
    key = jax.random.PRNGKey(2026)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    u_m = jax.random.normal(k1, (N, n), dtype=jnp.float64)
    u_g = jax.random.normal(k2, (N, m, n), dtype=jnp.float64)
    X = prob.pack_vars(u_m, u_g, 2.0 * jnp.pi)
    dot_u_ref = jax.random.normal(k3, (N, m, n), dtype=jnp.float64)
    p_val = jnp.array(0.2, dtype=jnp.float64)
    rhs_X = jax.random.normal(k4, X.shape, dtype=jnp.float64)
    c_X = jax.random.normal(k5, X.shape, dtype=jnp.float64)
    d_p = jnp.array(0.5, dtype=jnp.float64)
    rhs_g = jnp.array(-0.3, dtype=jnp.float64)

    @jax.jit
    def jit_factor_and_solve_square(x_in: jax.Array, r_in: jax.Array) -> tuple[CondensedCollocationFactor, jax.Array]:
        fac = factor_condensed_collocation(prob, x_in, p_val, mesh, dot_u_ref)
        dX = solve_condensed_collocation(fac, r_in)
        return fac, dX

    @jax.jit
    def jit_factor_and_solve_bordered(
        x_in: jax.Array, r_in: jax.Array, rg_in: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        fac = factor_condensed_collocation(prob, x_in, p_val, mesh, dot_u_ref, c_X=c_X, d_p=d_p)
        sol = solve_condensed_bordered(fac, r_in, rg_in)
        return sol.x, sol.y

    fac, dX = jit_factor_and_solve_square(X, rhs_X)
    dX_b, dp_b = jit_factor_and_solve_bordered(X, rhs_X, rhs_g)
    assert jnp.all(jnp.isfinite(dX))
    assert jnp.all(jnp.isfinite(dX_b))
    assert jnp.isfinite(dp_b)

    # Check that no leaf in CondensedCollocationFactor has O((N*n)^2) size
    dense_threshold = n_mesh * n_mesh
    for leaf in jax.tree_util.tree_leaves(fac):
        assert leaf.size < dense_threshold, f"Unexpected large array of shape {leaf.shape}"

    # Check that the traced jaxpr for factor_cyclic_block_bidiagonal never allocates (N*n, N*n)
    M_dummy = jnp.zeros((N, n, n), dtype=jnp.float64)
    S_dummy = jnp.zeros((N, n, 2), dtype=jnp.float64)
    P_dummy = jnp.zeros((N, 2, n), dtype=jnp.float64)
    D_dummy = jnp.zeros((2, 2), dtype=jnp.float64)
    closed_jaxpr = jax.make_jaxpr(factor_cyclic_block_bidiagonal)(
        M_dummy, S_dummy, P_dummy, D_dummy
    )
    for eqn in closed_jaxpr.jaxpr.eqns:
        for outvar in eqn.outvars:
            aval = outvar.aval
            if isinstance(aval, jax.core.ShapedArray):
                assert aval.size < dense_threshold, (
                    f"Dense O((N*n)^2) allocation detected in jaxpr: {eqn.primitive} -> {aval.shape}"
                )


def test_near_singular_reduced_system_lstsq_fallback():
    """Verify lstsq fallback triggers cleanly when reduced (n+b)x(n+b) system is rank-deficient."""
    N, n, b = 4, 2, 1
    # M_k = I for all k -> I - prod M_k = 0; border row P_k = 0, D = 0 -> singular reduced system
    M_blocks = jnp.broadcast_to(jnp.eye(n, dtype=jnp.float64), (N, n, n))
    S_blocks = jnp.zeros((N, n, b), dtype=jnp.float64)
    P_blocks = jnp.zeros((N, b, n), dtype=jnp.float64)
    D_border = jnp.zeros((b, b), dtype=jnp.float64)

    fac = factor_cyclic_block_bidiagonal(M_blocks, S_blocks, P_blocks, D_border)
    assert not bool(fac.use_lu_red)

    r_cont = jnp.zeros((N, n), dtype=jnp.float64)
    g_border = jnp.zeros((b,), dtype=jnp.float64)
    u_mesh, w = solve_cyclic_block_bidiagonal(fac, r_cont, g_border)
    assert jnp.all(jnp.isfinite(u_mesh))
    assert jnp.all(jnp.isfinite(w))
    assert jnp.allclose(u_mesh, 0.0)
    assert jnp.allclose(w, 0.0)
