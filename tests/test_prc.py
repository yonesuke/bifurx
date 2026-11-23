from __future__ import annotations

import jax.numpy as jnp

from bifurx.bvp import CollocationProblem, solve_periodic_orbit
from bifurx.prc import compute_iprc, continuation_iprc


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
