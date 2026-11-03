from __future__ import annotations

import jax.numpy as jnp

from bifurx.bvp import (
    CollocationProblem,
    compute_floquet_multipliers,
    initialize_from_hopf,
    redistribute_mesh,
    solve_periodic_orbit,
)


def f_vdp(u: jnp.ndarray, mu: float) -> jnp.ndarray:
    """Van der Pol oscillator."""
    x, y = u[0], u[1]
    return jnp.array([y, mu * (1.0 - x * x) * y - x])


def f_ab(u: jnp.ndarray, p1: float) -> jnp.ndarray:
    """A-B reaction model (AUTO-07P demo 'ab')."""
    P2 = 14.0
    P3 = 2.0
    u1, u2 = u[0], u[1]
    e = jnp.exp(u2)
    f1 = -u1 + p1 * (1.0 - u1) * e
    f2 = -u2 + p1 * P2 * (1.0 - u1) * e - P3 * u2
    return jnp.array([f1, f2])


def test_vdp_periodic_orbit_collocation():
    """Test solving periodic orbit for Van der Pol (mu = 0.2) via Gauss collocation."""
    mu = 0.2
    N = 25
    m = 4
    prob = CollocationProblem(f_vdp, dim=2, num_intervals=N, num_gauss_points=m)

    t_grid = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
    u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)

    sol = solve_periodic_orbit(prob, u_init, T_init=2 * jnp.pi, p=mu, tol=1e-8)

    assert bool(sol.converged), "Van der Pol periodic orbit should converge"
    assert int(sol.iterations) <= 6, f"Expected convergence in <= 6 iters, got {sol.iterations}"
    # Period of VdP at mu=0.2 is approximately 2*pi * (1 + mu^2 / 16) ~ 6.298
    assert 6.28 < float(sol.period) < 6.35, f"Period out of expected range: {sol.period}"
    assert float(sol.residual_norm) < 1e-8, f"Residual norm too high: {sol.residual_norm}"

    # Test continuous evaluation
    u_0 = sol.evaluate(0.0)
    u_1 = sol.evaluate(1.0)
    assert jnp.allclose(u_0, u_1, atol=1e-6), "Periodic boundary condition violated"

    # Test Floquet multipliers
    mults, _M = compute_floquet_multipliers(sol, p=mu, method="condensation")
    # One multiplier must be 1.0 (phase neutrality)
    assert abs(float(jnp.abs(mults[0])) - 1.0) < 1e-4, (
        f"Trivial multiplier should be 1.0, got {mults[0]}"
    )
    # Second multiplier must have magnitude < 1.0 (orbital stability)
    assert float(jnp.abs(mults[1])) < 1.0, (
        f"Limit cycle should be stable, got multiplier {mults[1]}"
    )

    # Test adaptive mesh redistribution
    sol_adapt = redistribute_mesh(sol, alpha=1.0, p=mu, re_solve=True)
    assert bool(sol_adapt.converged), "Redistributed orbit should converge"
    assert abs(float(sol_adapt.period - sol.period)) < 1e-4, (
        "Period should be consistent after remeshing"
    )


def test_ab_reaction_periodic_orbit_from_hopf():
    """Test seeding and solving periodic orbit from Hopf bifurcation in A-B reaction."""
    # Official AUTO-07P Hopf point for A-B model:
    u_hopf = jnp.array([0.8950809, 4.1770443])
    p1_hopf = 0.1309003

    N = 25
    prob = CollocationProblem(f_ab, dim=2, num_intervals=N, num_gauss_points=4)

    mesh, u_mesh_init, u_gauss_init, T0, p_init = initialize_from_hopf(
        prob, u_hopf, p1_hopf, eps=0.1, p_init=0.130
    )

    assert float(T0) > 0.0, f"Hopf period should be positive, got {T0}"

    sol = solve_periodic_orbit(
        prob,
        u_mesh_init=u_mesh_init,
        T_init=T0,
        p=p_init,
        mesh=mesh,
        u_gauss_init=u_gauss_init,
        tol=1e-7,
        non_trivial=True,
    )

    assert bool(sol.converged), "Periodic orbit seeded from Hopf should converge"
    assert float(sol.period) > 0.0
    # Check max amplitude of u1 is in reasonable limit cycle range (~0.9 - 1.0)
    max_u1 = float(jnp.max(sol.u_mesh[:, 0]))
    assert 0.85 < max_u1 < 1.05, f"Unexpected u1 amplitude: {max_u1}"
    assert float(sol.amplitude) > 0.05, f"Expected non-trivial amplitude, got {sol.amplitude}"


def test_collocation_non_trivial_and_degenerate_hopf():
    """Test non_trivial option and degenerate Hopf warning in collocation BVP (#14)."""
    import pytest

    # 1. Non-trivial constraint on trivial equilibrium guess
    prob = CollocationProblem(f_vdp, dim=2, num_intervals=15, num_gauss_points=3)
    u_zero = jnp.zeros((15, 2))
    # Trivial equilibrium u=0 satisfies collocation BVP
    sol_trivial = solve_periodic_orbit(
        prob, u_mesh_init=u_zero, T_init=2 * jnp.pi, p=0.2, non_trivial=False
    )
    assert bool(sol_trivial.converged)
    assert float(sol_trivial.amplitude) < 1e-6

    # With non_trivial=True, trivial solution must NOT be accepted as converged limit cycle
    sol_reject = solve_periodic_orbit(
        prob, u_mesh_init=u_zero, T_init=2 * jnp.pi, p=0.2, non_trivial=True, min_amplitude=1e-3
    )
    assert not bool(sol_reject.converged), (
        "Trivial equilibrium should not be accepted as limit cycle"
    )

    # 2. Degenerate Hopf warning when critical eigenvalue has multiplicity >= 2
    # Two identical oscillators experiencing Hopf simultaneously:
    def f_double_hopf(u, p):
        # 4D system: two uncoupled Stuart-Landau oscillators
        x1, y1, x2, y2 = u[0], u[1], u[2], u[3]
        dx1 = p * x1 - y1 - (x1**2 + y1**2) * x1
        dy1 = x1 + p * y1 - (x1**2 + y1**2) * y1
        dx2 = p * x2 - y2 - (x2**2 + y2**2) * x2
        dy2 = x2 + p * y2 - (x2**2 + y2**2) * y2
        return jnp.array([dx1, dy1, dx2, dy2])

    prob_deg = CollocationProblem(f_double_hopf, dim=4, num_intervals=10, num_gauss_points=2)
    with pytest.warns(UserWarning, match="Degenerate Hopf point detected"):
        initialize_from_hopf(prob_deg, u_hopf=jnp.zeros(4), p_hopf=0.0, eps=0.01)
