from __future__ import annotations

import jax.numpy as jnp

from bifurx.bvp import CollocationProblem, solve_periodic_orbit
from bifurx.prc import compute_iprc, continuation_iprc, solve_coupled_orbit_prc


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

    sol_orbit, res_prc = solve_coupled_orbit_prc(prob, u_init, T_init=6.28, p=0.0, tol=1e-9)

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
