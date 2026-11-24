from __future__ import annotations

import jax
import jax.numpy as jnp

import bifurx as bx


def test_refine_periodic_bifurcation_lpc_bautin_and_jit():
    """Verify pure-JAX LPC (Fold of Limit Cycles) refinement on Bautin oscillator under jax.jit."""

    # Bautin oscillator: dr/dt = r*(p + r^2 - r^4), dtheta/dt = 1.
    # Exact Fold of Limit Cycles (LPC) at r^2 = 1/2 => p_LPC = -0.25, T = 2*pi.
    def f_bautin(u: jnp.ndarray, p: float | jax.Array) -> jnp.ndarray:
        x, y = u[0], u[1]
        r2 = x * x + y * y
        rad = p + r2 - r2 * r2
        return jnp.array([rad * x - y, x + rad * y])

    N = 20
    prob = bx.CollocationProblem(f_bautin, dim=2, num_intervals=N, num_gauss_points=4)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    r0 = 1.0  # upper stable branch at p = 0
    u_init = jnp.stack([r0 * jnp.cos(t_grid), r0 * jnp.sin(t_grid)], axis=1)
    sol0 = bx.solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=0.0, tol=1e-10)
    assert bool(sol0.converged)

    branch = bx.continue_periodic_orbit(
        prob,
        sol0,
        p_start=0.0,
        p_end=-0.35,
        dp=0.08,
        dp_max=0.12,
        max_steps=25,
        param_mode="arclength",
        compute_floquet=True,
    )
    lpc_indices = [k for k, o in enumerate(branch) if "LPC" in o.bifurcations]
    assert len(lpc_indices) >= 1
    k_lpc = lpc_indices[0]

    @jax.jit
    def _refine_jit(
        oa: bx.PeriodicOrbitResult, ob: bx.PeriodicOrbitResult
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        ref = bx.refine_periodic_bifurcation(oa, ob, "LPC", tol=1e-10)
        assert ref.p is not None
        return ref.p, ref.period, ref.converged

    p_ref, T_ref, conv_ref = _refine_jit(
        branch[k_lpc - 1]._replace(bifurcations=()),
        branch[k_lpc]._replace(bifurcations=()),
    )
    assert bool(conv_ref)
    assert abs(float(p_ref) - (-0.25)) < 1e-8, f"LPC p mismatch: {p_ref}"
    assert abs(float(T_ref) - 2.0 * float(jnp.pi)) < 1e-6


def test_refine_periodic_bifurcation_ns_exact():
    """Verify pure-JAX NS (Neimark-Sacker / Torus) refinement to machine precision."""

    # 4D system: unit-circle limit cycle in (x, y) + transverse oscillator in (z1, z2)
    # with Floquet multipliers exp(2*pi*(p +/- i*0.7)), crossing |mu| = 1 at exact p_NS = 0.0.
    def f_ns(u: jnp.ndarray, p: float | jax.Array) -> jnp.ndarray:
        x, y, z1, z2 = u[0], u[1], u[2], u[3]
        r2 = x * x + y * y
        w = 0.7
        return jnp.array([x * (1.0 - r2) - y, y * (1.0 - r2) + x, p * z1 - w * z2, w * z1 + p * z2])

    N = 16
    prob = bx.CollocationProblem(f_ns, dim=4, num_intervals=N, num_gauss_points=4)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    zeros = jnp.zeros_like(t_grid)
    u_init = jnp.stack([jnp.cos(t_grid), jnp.sin(t_grid), zeros, zeros], axis=1)

    sol_a = bx.solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=-0.12, tol=1e-10)
    sol_b = bx.solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=0.15, tol=1e-10)
    assert bool(sol_a.converged) and bool(sol_b.converged)

    refined = bx.refine_periodic_bifurcation(sol_a, sol_b, "NS", tol=1e-10)
    assert bool(refined.converged)
    assert refined.p is not None
    assert abs(float(refined.p)) < 1e-9, f"NS p mismatch: {refined.p}"
    assert refined.bifurcations == ("NS",)


def test_refine_periodic_bifurcation_pd_rossler():
    """Verify pure-JAX PD (Period-Doubling) refinement on the classic Rossler system."""

    def f_rossler(u: jnp.ndarray, c: float | jax.Array) -> jnp.ndarray:
        x, y, z = u[0], u[1], u[2]
        return jnp.array([-y - z, x + 0.2 * y, 0.2 + z * (x - c)])

    # Seed near the well-known period-1 limit cycle of Rossler at c = 2.5 via RK4
    def _rk4_step(u: jnp.ndarray, dt: float, c_val: float) -> jnp.ndarray:
        k1 = f_rossler(u, c_val)
        k2 = f_rossler(u + 0.5 * dt * k1, c_val)
        k3 = f_rossler(u + 0.5 * dt * k2, c_val)
        k4 = f_rossler(u + dt * k3, c_val)
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    u_cur = jnp.array([1.0, 0.0, 0.0])
    u_cur, _ = jax.lax.scan(lambda u, _: (_rk4_step(u, 0.02, 2.5), None), u_cur, jnp.arange(3000))
    # Integrate roughly one period (T ~ 5.88)
    N = 30
    T_guess = 5.88
    substeps = 10
    dt_sub = T_guess / (N * substeps)

    def _chunk(u: jnp.ndarray, _idx: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        u_next, _ = jax.lax.scan(
            lambda s, __: (_rk4_step(s, dt_sub, 2.5), None), u, jnp.arange(substeps)
        )
        return u_next, u

    _, u_mesh_init = jax.lax.scan(_chunk, u_cur, jnp.arange(N))
    prob = bx.CollocationProblem(f_rossler, dim=3, num_intervals=N, num_gauss_points=4)
    sol0 = bx.solve_periodic_orbit(prob, u_mesh_init, T_init=T_guess, p=2.5, tol=1e-9)
    assert bool(sol0.converged)

    branch = bx.continue_periodic_orbit(
        prob,
        sol0,
        p_start=2.5,
        p_end=3.2,
        dp=0.08,
        dp_max=0.15,
        max_steps=20,
        param_mode="arclength",
        compute_floquet=True,
    )
    pd_indices = [k for k, o in enumerate(branch) if "PD" in o.bifurcations]
    assert len(pd_indices) >= 1, "Expected PD bifurcation on Rossler period-1 branch"
    k_pd = pd_indices[0]

    refined = bx.refine_periodic_bifurcation(branch[k_pd - 1], branch[k_pd], "PD", tol=1e-9)
    assert bool(refined.converged)
    assert refined.p is not None
    assert refined.floquet_multipliers is not None
    # One Floquet multiplier must be -1 to high precision
    dist_neg1 = float(jnp.min(jnp.abs(refined.floquet_multipliers + 1.0)))
    assert dist_neg1 < 1e-7, f"Expected Floquet multiplier at -1, min dist={dist_neg1}"
    assert refined.bifurcations == ("PD",)

    # Switch onto the period-2 branch in pure JAX (and verify jax.jit compatibility)
    @jax.jit
    def _pd_switch_jit(orb: bx.PeriodicOrbitResult) -> tuple[jax.Array, jax.Array, jax.Array]:
        p2_orb = bx.branch_from_period_doubling(orb, ds=0.4, compute_floquet=True)
        assert p2_orb.p is not None and p2_orb.is_stable is not None
        return p2_orb.p, p2_orb.period, p2_orb.converged

    p2_val, T2_val, conv2 = _pd_switch_jit(refined._replace(bifurcations=()))
    assert bool(conv2)
    assert float(p2_val) > float(refined.p)
    assert abs(float(T2_val) - 2.0 * float(refined.period)) < 0.2

    sol_p2 = bx.branch_from_period_doubling(refined, ds=0.4, compute_floquet=True)
    assert bool(sol_p2.converged) and sol_p2.p is not None and sol_p2.is_stable is not None
    assert bool(sol_p2.is_stable), "Supercritical Rossler period-2 branch should be stable"
    # Verify genuine period-2 asymmetry (first half != second half)
    half_diff = float(jnp.max(jnp.abs(sol_p2.u_mesh[:N] - sol_p2.u_mesh[N:])))
    assert half_diff > 0.05, f"Orbit collapsed onto period-1 branch: half_diff={half_diff}"

    # Continue the new period-2 branch seamlessly
    branch_p2 = bx.continue_periodic_orbit(
        sol_p2.problem,
        sol_p2,
        p_start=float(sol_p2.p),
        p_end=float(sol_p2.p) + 0.5,
        dp=0.1,
        max_steps=6,
        param_mode="arclength",
        compute_floquet=True,
    )
    assert len(branch_p2) >= 4
    assert all(bool(o.converged) for o in branch_p2)
