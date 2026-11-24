from __future__ import annotations

import jax
import jax.numpy as jnp

import bifurx as bx


def test_fixed_period_2p_isoperiodic_exact_and_jit():
    """Verify continue_fixed_period_2p tracks an exact isoperiodic curve in pure JAX (and JIT)."""

    # Stuart-Landau with amplitude-dependent frequency:
    # r^2 = p1, angular frequency = p2 + r^2 = p1 + p2.
    # Constant period T = 2*pi / omega_c <=> p1 + p2 = omega_c.
    def f_sl_2p(u: jnp.ndarray, p1: float | jax.Array, p2: float | jax.Array) -> jnp.ndarray:
        x, y = u[0], u[1]
        r2 = x * x + y * y
        w = p2 + r2
        return jnp.array([p1 * x - w * y - x * r2, w * x + p1 * y - y * r2])

    p1_0, p2_0 = 0.5, 1.0  # omega_c = 1.5
    omega_c = p1_0 + p2_0
    T_exact = 2.0 * float(jnp.pi) / omega_c
    N = 20
    prob_1p = bx.CollocationProblem(
        lambda u, p: f_sl_2p(u, p, p2_0), dim=2, num_intervals=N, num_gauss_points=4
    )
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    r0 = float(jnp.sqrt(p1_0))
    u_init = jnp.stack([r0 * jnp.cos(t_grid), r0 * jnp.sin(t_grid)], axis=1)
    sol0 = bx.solve_periodic_orbit(prob_1p, u_init, T_init=T_exact, p=p1_0, tol=1e-10)
    assert bool(sol0.converged)

    # Run under jax.jit to guarantee pure-JAX compatibility (no host float/bool syncs)
    @jax.jit
    def run_jit(orb: bx.PeriodicOrbitResult) -> bx.FixedPeriodContinuationResult:
        return bx.continue_fixed_period_2p(
            f_sl_2p,
            orb,
            p1_init=p1_0,
            p2_init=p2_0,
            p1_bounds=(0.2, 1.2),
            p2_bounds=(0.2, 1.5),
            ds=0.08,
            ds_max=0.12,
            max_steps=12,
            direction=1.0,
            tol=1e-9,
        )

    res = run_jit(sol0)
    conv = res.converged
    assert int(jnp.sum(conv)) >= 6
    p1_v = res.p1[conv]
    p2_v = res.p2[conv]
    assert float(jnp.max(jnp.abs((p1_v + p2_v) - omega_c))) < 1e-8
    assert float(p1_v[-1]) > p1_0 + 0.2
    assert float(p2_v[-1]) < p2_0 - 0.2


def test_fixed_period_2p_bimodal_kuramoto_homoclinic_curve():
    """Verify fixed-large-period 2p continuation traces the Martens (2009) HC curve."""

    def f_bimodal_2p(
        u: jnp.ndarray, omega0: float | jax.Array, delta: float | jax.Array
    ) -> jnp.ndarray:
        x, y = u[0], u[1]
        return jnp.array(
            [
                (1.0 - 0.5 * delta) * x - 0.5 * omega0 * y - x**3 + x * y**2,
                -0.5 * delta * y + 0.5 * omega0 * x - 2.0 * x**2 * y,
            ]
        )

    w0_start = 1.092
    prob_1p = bx.CollocationProblem(
        lambda u, p: f_bimodal_2p(u, w0_start, p), dim=2, num_intervals=30, num_gauss_points=4
    )
    orb0 = bx.branch_from_hopf(prob_1p, u_hopf=[0.0, 0.0], p_hopf=1.0, ds=0.1)
    assert bool(orb0.converged) and orb0.p is not None

    branch_1p = bx.continue_periodic_orbit(
        prob_1p,
        orb0,
        p_start=float(orb0.p),
        p_end=0.90,
        dp=0.02,
        dp_max=0.04,
        max_steps=80,
        param_mode="arclength",
        period_weight=0.01,
        remesh_interval=15,
        period_bounds=(0.1, 80.0),
    )
    large_T_orb = branch_1p[-1]
    assert float(large_T_orb.period) >= 60.0 and large_T_orb.p is not None

    res_hc = bx.continue_fixed_period_2p(
        f_bimodal_2p,
        large_T_orb,
        p1_init=w0_start,
        p2_init=large_T_orb.p,
        p1_bounds=(1.02, 1.35),
        p2_bounds=(0.70, 1.00),
        ds=0.03,
        ds_max=0.06,
        max_steps=15,
        direction=1.0,
        tol=1e-7,
    )
    conv = res_hc.converged
    assert int(jnp.sum(conv)) >= 5
    w_v = res_hc.p1[conv]
    d_v = res_hc.p2[conv]
    # Along the Martens (2009) HC curve, Delta decreases monotonically as omega0 increases
    assert float(w_v[-1]) > w0_start + 0.05
    assert float(d_v[-1]) < float(d_v[0]) - 0.02
