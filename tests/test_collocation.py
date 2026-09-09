from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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

    t_grid = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u_init = np.stack([2.0 * np.cos(t_grid), -2.0 * np.sin(t_grid)], axis=1)

    sol = solve_periodic_orbit(prob, u_init, T_init=2 * np.pi, p=mu, tol=1e-8)

    assert sol.converged, "Van der Pol periodic orbit should converge"
    assert sol.iterations <= 6, f"Expected convergence in <= 6 iters, got {sol.iterations}"
    # Period of VdP at mu=0.2 is approximately 2*pi * (1 + mu^2 / 16) ~ 6.298
    assert 6.28 < sol.period < 6.35, f"Period out of expected range: {sol.period}"
    assert sol.residual_norm < 1e-8, f"Residual norm too high: {sol.residual_norm}"

    # Test continuous evaluation
    u_0 = np.asarray(sol.evaluate(0.0))
    u_1 = np.asarray(sol.evaluate(1.0))
    np.testing.assert_allclose(u_0, u_1, atol=1e-6, err_msg="Periodic boundary condition violated")

    # Test Floquet multipliers
    mults, M = compute_floquet_multipliers(sol, p=mu, method="condensation")
    # One multiplier must be 1.0 (phase neutrality)
    assert abs(abs(mults[0]) - 1.0) < 1e-4, f"Trivial multiplier should be 1.0, got {mults[0]}"
    # Second multiplier must have magnitude < 1.0 (orbital stability)
    assert abs(mults[1]) < 1.0, f"Limit cycle should be stable, got multiplier {mults[1]}"

    # Test adaptive mesh redistribution
    sol_adapt = redistribute_mesh(sol, alpha=1.0, p=mu, re_solve=True)
    assert sol_adapt.converged, "Redistributed orbit should converge"
    assert abs(sol_adapt.period - sol.period) < 1e-4, "Period should be consistent after remeshing"


def test_ab_reaction_periodic_orbit_from_hopf():
    """Test seeding and solving periodic orbit from Hopf bifurcation in A-B reaction."""
    # Official AUTO-07P Hopf point for A-B model:
    u_hopf = np.array([0.8950809, 4.1770443])
    p1_hopf = 0.1309003

    N = 25
    prob = CollocationProblem(f_ab, dim=2, num_intervals=N, num_gauss_points=4)

    mesh, u_mesh_init, u_gauss_init, T0, p_init = initialize_from_hopf(
        prob, u_hopf, p1_hopf, eps=0.03, p_init=0.128
    )

    assert T0 > 0.0, f"Hopf period should be positive, got {T0}"

    sol = solve_periodic_orbit(
        prob,
        u_mesh_init=u_mesh_init,
        T_init=T0,
        p=p_init,
        mesh=mesh,
        u_gauss_init=u_gauss_init,
        tol=1e-7,
    )

    assert sol.converged, "Periodic orbit seeded from Hopf should converge"
    assert sol.period > 0.0
    # Check max amplitude of u1 is in reasonable limit cycle range (~0.9 - 1.0)
    max_u1 = float(np.max(np.asarray(sol.u_mesh)[:, 0]))
    assert 0.85 < max_u1 < 1.05, f"Unexpected u1 amplitude: {max_u1}"
