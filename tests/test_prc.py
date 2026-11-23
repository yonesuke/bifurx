from __future__ import annotations

import jax
import jax.numpy as jnp

from bifurx.bvp import CollocationProblem, solve_periodic_orbit
from bifurx.prc import _solve_adjoint_collocation_jax, compute_iprc, continuation_iprc


def f_stuart_landau(u: jnp.ndarray, p: float = 0.0) -> jnp.ndarray:
    """Stuart-Landau oscillator with exact analytical limit cycle and iPRC.

    Equations:
      dx/dt = x - y - x * (x^2 + y^2)
      dy/dt = x + y - y * (x^2 + y^2)

    Exact limit cycle:
      x(t) = cos(t), y(t) = sin(t), Period T = 2*pi

    Exact iPRC (phase theta = atan2(y, x)):
      Z_x(t) = -sin(t), Z_y(t) = cos(t)
    """
    x, y = u[0], u[1]
    r2 = x * x + y * y
    return jnp.array([x - y - x * r2, x + y - y * r2])


def f_vdp(u: jnp.ndarray, mu: float) -> jnp.ndarray:
    """Van der Pol oscillator."""
    x, y = u[0], u[1]
    return jnp.array([y, mu * (1.0 - x * x) * y - x])


def test_stuart_landau_analytical_iprc():
    """Test iPRC on Stuart-Landau oscillator where exact analytical iPRC is known."""
    N = 25
    m = 4
    prob = CollocationProblem(f_stuart_landau, dim=2, num_intervals=N, num_gauss_points=m)

    t_grid = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
    u_init = jnp.stack([jnp.cos(t_grid), jnp.sin(t_grid)], axis=1)

    sol_orbit = solve_periodic_orbit(prob, u_init, T_init=6.28, p=0.0, tol=1e-9)
    res_prc = compute_iprc(sol_orbit, p=0.0)

    assert bool(sol_orbit.converged)
    assert abs(float(sol_orbit.period) - 2.0 * float(jnp.pi)) < 1e-6

    # Verify iPRC against analytical sinusoidal solution: Z_x = -sin(t), Z_y = cos(t)
    ts = res_prc.ts
    zx_exact = -jnp.sin(ts)
    zy_exact = jnp.cos(ts)

    Z_num = res_prc.Z
    err_zx = float(jnp.max(jnp.abs(Z_num[:, 0] - zx_exact)))
    err_zy = float(jnp.max(jnp.abs(Z_num[:, 1] - zy_exact)))

    assert err_zx < 1e-4, f"Z_x mismatch with analytical formula: {err_zx}"
    assert err_zy < 1e-4, f"Z_y mismatch with analytical formula: {err_zy}"

    # Verify normalization <Z(0), f(u(0))> == 2*pi / T
    u0 = res_prc.orbit[0]
    f0 = f_stuart_landau(u0)
    z0 = res_prc.Z[0]
    inner_prod = float(jnp.dot(z0, f0))
    expected_norm = 2.0 * float(jnp.pi) / float(sol_orbit.period)
    assert abs(inner_prod - expected_norm) < 1e-5, (
        f"Normalization mismatch: {inner_prod} vs {expected_norm}"
    )


def test_van_der_pol_iprc_and_continuation():
    """Test iPRC computation and parameter continuation for Van der Pol oscillator."""
    mu = 0.2
    N = 20
    m = 4
    prob = CollocationProblem(f_vdp, dim=2, num_intervals=N, num_gauss_points=m)

    t_grid = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
    u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)

    sol = solve_periodic_orbit(prob, u_init, T_init=2 * jnp.pi, p=mu, tol=1e-8)
    assert bool(sol.converged)

    # Test standalone compute_iprc
    res_prc = compute_iprc(sol, p=mu)
    assert float(res_prc.period) > 6.0
    assert len(res_prc.Z) == N

    # Verify periodicity of iPRC: Z(0) approx Z(T)
    # At mesh nodes, Z[0] and extrapolated Z[end] match
    f0 = f_vdp(sol.u_mesh[0], mu)
    inner_prod = float(jnp.dot(res_prc.Z[0], f0))
    expected_norm = 2.0 * float(jnp.pi) / float(sol.period)
    assert abs(inner_prod - expected_norm) < 1e-4, (
        f"Normalization error: {inner_prod} vs {expected_norm}"
    )

    # Test parameter continuation of iPRC across mu in [0.1, 0.4]
    res_cont = continuation_iprc(
        prob,
        u_mesh_init=sol.u_mesh,
        T_init=sol.period,
        p_start=0.1,
        p_end=0.4,
        num_steps=4,
        tol=1e-7,
    )

    assert res_cont.num_points == 4
    # Period should increase monotonically with mu for Van der Pol
    periods = res_cont.periods
    assert bool(jnp.all(jnp.diff(periods) > 0.0)), f"Period should increase with mu: {periods}"


def test_iprc_singular_last_velocity_component():
    """Verify bordered adjoint solve stays nonsingular when f0[-1] == 0 (#22)."""
    N, m = 25, 4
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)

    # 1. 2D Stuart-Landau shifted by pi/2 so u0 = (0, 1) and f0 = (-1, 0), hence f0[-1] == 0
    prob_2d = CollocationProblem(f_stuart_landau, dim=2, num_intervals=N, num_gauss_points=m)
    u_init_2d = jnp.stack([-jnp.sin(t_grid), jnp.cos(t_grid)], axis=1)
    sol_2d = solve_periodic_orbit(prob_2d, u_init_2d, T_init=6.28, p=0.0, tol=1e-9)
    assert bool(sol_2d.converged)
    assert abs(float(f_stuart_landau(sol_2d.u_mesh[0])[-1])) < 1e-8

    prc_2d = compute_iprc(sol_2d)  # p=None infers sol_2d.p
    assert bool(jnp.all(jnp.isfinite(prc_2d.Z)))
    ts = prc_2d.ts
    assert float(jnp.max(jnp.abs(prc_2d.Z[:, 0] - (-jnp.cos(ts))))) < 1e-4
    assert float(jnp.max(jnp.abs(prc_2d.Z[:, 1] - (-jnp.sin(ts))))) < 1e-4
    f0_2d = f_stuart_landau(prc_2d.orbit[0])
    assert abs(float(jnp.dot(prc_2d.Z[0], f0_2d)) - float(prc_2d.omega)) < 1e-5

    # 2. 3D decoupled system with z(t) == 0 -> f0 = (-1, 0, 0)
    def f_3d(u: jnp.ndarray, p: float = 0.0) -> jnp.ndarray:
        x, y, z = u[0], u[1], u[2]
        r2 = x * x + y * y
        return jnp.array([x - y - x * r2, x + y - y * r2, -(1.0 + p) * z])

    prob_3d = CollocationProblem(f_3d, dim=3, num_intervals=N, num_gauss_points=m)
    u_init_3d = jnp.stack([-jnp.sin(t_grid), jnp.cos(t_grid), jnp.zeros_like(t_grid)], axis=1)
    sol_3d = solve_periodic_orbit(prob_3d, u_init_3d, T_init=6.28, p=0.5, tol=1e-9)
    assert bool(sol_3d.converged)
    assert abs(float(f_3d(sol_3d.u_mesh[0], 0.5)[-1])) < 1e-12

    prc_3d = compute_iprc(sol_3d)  # verifies p=0.5 inferred from sol_3d.p
    assert bool(jnp.all(jnp.isfinite(prc_3d.Z)))
    assert float(jnp.max(jnp.abs(prc_3d.Z[:, 0] - (-jnp.cos(prc_3d.ts))))) < 1e-4
    assert float(jnp.max(jnp.abs(prc_3d.Z[:, 1] - (-jnp.sin(prc_3d.ts))))) < 1e-4
    assert float(jnp.max(jnp.abs(prc_3d.Z[:, 2]))) < 1e-4
    f0_3d = f_3d(prc_3d.orbit[0], 0.5)
    assert abs(float(jnp.dot(prc_3d.Z[0], f0_3d)) - float(prc_3d.omega)) < 1e-5


def test_continuation_iprc_arclength_adaptive():
    """Verify continuation_iprc supports arclength mode, adaptive Padé predictor, and exact iPRC norm."""
    mu = 0.2
    N, m = 20, 4
    prob = CollocationProblem(f_vdp, dim=2, num_intervals=N, num_gauss_points=m)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)
    sol = solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=mu, tol=1e-8)
    assert bool(sol.converged)

    res_arc = continuation_iprc(
        prob,
        u_mesh_init=sol.u_mesh,
        T_init=sol.period,
        p_start=mu,
        p_end=0.8,
        dp=0.08,
        max_steps=6,
        tol=1e-7,
        param_mode="arclength",
        adaptive_step=True,
        predictor="pade21",
    )
    assert res_arc.num_points >= 4
    assert bool(jnp.all(jnp.isfinite(res_arc.Z)))
    assert bool(jnp.all(jnp.diff(res_arc.periods) > 0.0))
    for k, item in enumerate(res_arc.results):
        f0 = f_vdp(item.orbit[0], float(res_arc.p[k]))
        expected_omega = 2.0 * float(jnp.pi) / float(item.period)
        assert abs(float(jnp.dot(item.Z[0], f0)) - expected_omega) < 1e-4


def _solve_adjoint_dense_reference(
    problem: CollocationProblem,
    mesh: jax.Array,
    u_mesh: jax.Array,
    u_gauss: jax.Array,
    T: jax.Array,
    p: float | jax.Array,
    omega_val: jax.Array,
) -> jax.Array:
    """Direct dense (N*(m+1)*n + 1) adjoint collocation reference solver."""
    N, m, n = problem.num_intervals, problem.num_gauss_points, problem.dim
    D = jnp.asarray(problem.D, dtype=jnp.float64)
    C = jnp.asarray(problem.C, dtype=jnp.float64)
    h = mesh[1:] - mesh[:-1]
    p_val = jnp.squeeze(jnp.asarray(p, dtype=jnp.float64))

    jac_fn = jax.jacobian(lambda u: problem.fn(u, p_val, **problem.kwargs))
    Ju_all = jax.vmap(jax.vmap(jac_fn))(u_gauss)
    f0 = jnp.atleast_1d(
        jnp.asarray(problem.fn(u_mesh[0], p_val, **problem.kwargs), dtype=jnp.float64)
    ).ravel()
    f0_norm = jnp.linalg.norm(f0)
    f0_unit = f0 / jnp.where(f0_norm > 1e-30, f0_norm, 1.0)

    n_mesh = N * n
    dim_z = N * (m + 1) * n
    dim_aug = dim_z + 1

    def adjoint_operator(z_aug: jax.Array) -> jax.Array:
        Z_mesh = z_aug[:n_mesh].reshape((N, n))
        Z_gauss = z_aug[n_mesh:dim_z].reshape((N, m, n))
        alpha = z_aug[dim_z]
        Z_full = jnp.concatenate([Z_mesh[:, None, :], Z_gauss], axis=1)

        dZ_dt = jnp.einsum("mj,Njd->Nmd", D, Z_full) / h[:, None, None]
        jt_z = jnp.einsum("Nmji,Nmj->Nmi", Ju_all, Z_gauss)
        res_coll = (dZ_dt + T * jt_z).ravel()

        Z_end = jnp.einsum("j,Njd->Nd", C, Z_full)
        res_cont = (Z_mesh[1:] - Z_end[:-1]).ravel()
        res_bc = (Z_mesh[0] - Z_end[N - 1] + alpha * f0_unit).ravel()
        res_norm = jnp.atleast_1d(jnp.dot(f0, Z_mesh[0]))

        return jnp.concatenate([res_coll, res_cont, res_bc, res_norm])

    L = jax.jacfwd(adjoint_operator)(jnp.zeros(dim_aug, dtype=jnp.float64))
    rhs = jnp.zeros(dim_aug, dtype=jnp.float64).at[-1].set(omega_val)
    sol = jnp.linalg.solve(L, rhs)
    return sol[:n_mesh].reshape((N, n))


def test_condensed_adjoint_matches_dense_reference():
    """Verify block-condensed adjoint solver matches dense reference to machine precision (#28)."""
    N, m = 20, 4
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)

    # 1. Van der Pol (moderately nonlinear mu = 1.5)
    prob_vdp = CollocationProblem(f_vdp, dim=2, num_intervals=N, num_gauss_points=m)
    u_init_vdp = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)
    sol_vdp = solve_periodic_orbit(prob_vdp, u_init_vdp, T_init=7.0, p=1.5, tol=1e-10)
    assert bool(sol_vdp.converged)

    omega_vdp = 2.0 * jnp.pi / sol_vdp.period
    Z_cond_vdp = _solve_adjoint_collocation_jax(
        prob_vdp, sol_vdp.mesh, sol_vdp.u_mesh, sol_vdp.u_gauss, sol_vdp.period, 1.5, omega_vdp
    )
    Z_ref_vdp = _solve_adjoint_dense_reference(
        prob_vdp, sol_vdp.mesh, sol_vdp.u_mesh, sol_vdp.u_gauss, sol_vdp.period, 1.5, omega_vdp
    )
    assert bool(jnp.allclose(Z_cond_vdp, Z_ref_vdp, rtol=1e-10, atol=1e-11)), (
        f"VdP max diff: {float(jnp.max(jnp.abs(Z_cond_vdp - Z_ref_vdp)))}"
    )

    # 2. Stuart-Landau shifted so f0[-1] == 0 (tests rank-1 border alpha * f0_unit)
    prob_sl = CollocationProblem(f_stuart_landau, dim=2, num_intervals=N, num_gauss_points=m)
    u_init_sl = jnp.stack([-jnp.sin(t_grid), jnp.cos(t_grid)], axis=1)
    sol_sl = solve_periodic_orbit(prob_sl, u_init_sl, T_init=6.28, p=0.0, tol=1e-10)
    assert bool(sol_sl.converged)
    assert abs(float(f_stuart_landau(sol_sl.u_mesh[0])[-1])) < 1e-8

    omega_sl = 2.0 * jnp.pi / sol_sl.period
    Z_cond_sl = _solve_adjoint_collocation_jax(
        prob_sl, sol_sl.mesh, sol_sl.u_mesh, sol_sl.u_gauss, sol_sl.period, 0.0, omega_sl
    )
    Z_ref_sl = _solve_adjoint_dense_reference(
        prob_sl, sol_sl.mesh, sol_sl.u_mesh, sol_sl.u_gauss, sol_sl.period, 0.0, omega_sl
    )
    assert bool(jnp.allclose(Z_cond_sl, Z_ref_sl, rtol=1e-10, atol=1e-11)), (
        f"Stuart-Landau max diff: {float(jnp.max(jnp.abs(Z_cond_sl - Z_ref_sl)))}"
    )


def test_condensed_adjoint_large_n_and_jit():
    """Verify block-condensed iPRC solver scales to N=60, m=4 and is jax.jit-compatible (#28)."""
    N, m = 60, 4
    mu = 0.5
    prob = CollocationProblem(f_vdp, dim=2, num_intervals=N, num_gauss_points=m)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)
    sol = solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=mu, tol=1e-10)
    assert bool(sol.converged)

    res_prc = compute_iprc(sol, p=mu)
    assert res_prc.Z.shape == (N, 2)
    assert bool(jnp.all(jnp.isfinite(res_prc.Z)))

    f0 = f_vdp(sol.u_mesh[0], mu)
    assert abs(float(jnp.dot(res_prc.Z[0], f0)) - float(res_prc.omega)) < 1e-11

    solve_jit = jax.jit(
        lambda mesh, um, ug, T, p_v, om: _solve_adjoint_collocation_jax(
            prob, mesh, um, ug, T, p_v, om
        )
    )
    Z_jit = solve_jit(
        sol.mesh,
        sol.u_mesh,
        sol.u_gauss,
        sol.period,
        jnp.asarray(mu, dtype=jnp.float64),
        res_prc.omega,
    )
    assert bool(jnp.allclose(Z_jit, res_prc.Z, rtol=1e-12, atol=1e-13))
