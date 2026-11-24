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


def test_periodic_codim2_lpc_bautin_exact_and_jit():
    """Verify pure-JAX 2p LPC curve tracking matches exact p1 = -p2^2 / 4 under jax.jit."""

    def f_bautin_2p(u: jnp.ndarray, p1: float | jax.Array, p2: float | jax.Array) -> jnp.ndarray:
        x, y = u[0], u[1]
        r2 = x * x + y * y
        rad = p1 + p2 * r2 - r2 * r2
        return jnp.array([rad * x - y, x + rad * y])

    # Exact LPC at p2 = 1.0 => p1 = -0.25, r = 1/sqrt(2), T = 2*pi
    N = 12
    prob = bx.CollocationProblem(
        lambda u, p: f_bautin_2p(u, p, 1.0), dim=2, num_intervals=N, num_gauss_points=4
    )
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    r0 = float(jnp.sqrt(0.5))
    u_init = jnp.stack([r0 * jnp.cos(t_grid), r0 * jnp.sin(t_grid)], axis=1)
    sol_lpc = bx.solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=-0.25, tol=1e-10)
    assert bool(sol_lpc.converged)

    @jax.jit
    def _track_lpc_jit(orb: bx.PeriodicOrbitResult) -> bx.PeriodicCodim2Result:
        return bx.continue_periodic_bifurcation_2p(
            f_bautin_2p,
            "LPC",
            orb,
            p1_init=-0.25,
            p2_init=1.0,
            p1_bounds=(-1.5, 0.0),
            p2_bounds=(0.4, 2.0),
            ds=0.1,
            ds_max=0.15,
            max_steps=8,
            direction=-1.0,
            tol=1e-8,
        )

    res = _track_lpc_jit(sol_lpc)
    conv = res.converged
    assert int(jnp.sum(conv)) >= 5
    p1_v = res.p1[conv]
    p2_v = res.p2[conv]
    assert float(jnp.max(jnp.abs(p1_v + 0.25 * p2_v**2))) < 1e-7
    assert float(p2_v[-1]) > 1.15


def test_periodic_codim2_ns_exact():
    """Verify pure-JAX 2p NS (Torus) curve tracking matches exact p1 + p2 = 0."""

    def f_ns_2p(u: jnp.ndarray, p1: float | jax.Array, p2: float | jax.Array) -> jnp.ndarray:
        x, y, z1, z2 = u[0], u[1], u[2], u[3]
        r2 = x * x + y * y
        s = p1 + p2
        w = 0.7
        return jnp.array([x * (1.0 - r2) - y, y * (1.0 - r2) + x, s * z1 - w * z2, w * z1 + s * z2])

    N = 10
    prob = bx.CollocationProblem(
        lambda u, p: f_ns_2p(u, p, 0.2), dim=4, num_intervals=N, num_gauss_points=4
    )
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    zeros = jnp.zeros_like(t_grid)
    u_init = jnp.stack([jnp.cos(t_grid), jnp.sin(t_grid), zeros, zeros], axis=1)
    sol_ns = bx.solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=-0.2, tol=1e-10)
    assert bool(sol_ns.converged)

    res = bx.continue_periodic_bifurcation_2p(
        f_ns_2p,
        "NS",
        sol_ns,
        p1_init=-0.2,
        p2_init=0.2,
        p1_bounds=(-1.0, 1.0),
        p2_bounds=(-1.0, 1.0),
        ds=0.1,
        ds_max=0.15,
        max_steps=6,
        direction=1.0,
        tol=1e-8,
    )
    conv = res.converged
    assert int(jnp.sum(conv)) >= 4
    p1_v = res.p1[conv]
    p2_v = res.p2[conv]
    assert float(jnp.max(jnp.abs(p1_v + p2_v))) < 1e-7
    assert float(p1_v[-1]) > -0.05


def test_periodic_codim2_pd_rossler():
    """Verify pure-JAX 2p PD curve tracking on the 2-parameter Rossler system (c, a)."""

    def f_rossler_2p(u: jnp.ndarray, c: float | jax.Array, a: float | jax.Array) -> jnp.ndarray:
        x, y, z = u[0], u[1], u[2]
        return jnp.array([-y - z, x + a * y, 0.2 + z * (x - c)])

    def _rk4_step(u: jnp.ndarray, dt: float, c_val: float) -> jnp.ndarray:
        k1 = f_rossler_2p(u, c_val, 0.2)
        k2 = f_rossler_2p(u + 0.5 * dt * k1, c_val, 0.2)
        k3 = f_rossler_2p(u + 0.5 * dt * k2, c_val, 0.2)
        k4 = f_rossler_2p(u + dt * k3, c_val, 0.2)
        return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    u_cur = jnp.array([1.0, 0.0, 0.0])
    u_cur, _ = jax.lax.scan(lambda u, _: (_rk4_step(u, 0.02, 2.5), None), u_cur, jnp.arange(3000))
    N = 18
    T_guess = 5.88
    substeps = 10
    dt_sub = T_guess / (N * substeps)

    def _chunk(u: jnp.ndarray, _idx: jax.Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        u_next, _ = jax.lax.scan(
            lambda s, __: (_rk4_step(s, dt_sub, 2.5), None), u, jnp.arange(substeps)
        )
        return u_next, u

    _, u_mesh_init = jax.lax.scan(_chunk, u_cur, jnp.arange(N))
    prob = bx.CollocationProblem(
        lambda u, c: f_rossler_2p(u, c, 0.2), dim=3, num_intervals=N, num_gauss_points=4
    )
    sol0 = bx.solve_periodic_orbit(prob, u_mesh_init, T_init=T_guess, p=2.5, tol=1e-9)
    branch = bx.continue_periodic_orbit(
        prob, sol0, p_start=2.5, p_end=3.2, dp=0.1, max_steps=16, compute_floquet=True
    )
    k_pd = next(k for k, o in enumerate(branch) if "PD" in o.bifurcations)
    pd_orb = bx.refine_periodic_bifurcation(branch[k_pd - 1], branch[k_pd], "PD", tol=1e-9)
    assert bool(pd_orb.converged) and pd_orb.p is not None

    res_pd = bx.continue_periodic_bifurcation_2p(
        f_rossler_2p,
        "PD",
        pd_orb,
        p1_init=pd_orb.p,
        p2_init=0.2,
        p1_bounds=(2.0, 4.5),
        p2_bounds=(0.1, 0.35),
        ds=0.15,
        ds_max=0.25,
        max_steps=5,
        direction=1.0,
        tol=1e-7,
    )
    conv = res_pd.converged
    assert int(jnp.sum(conv)) >= 4
    c_last = res_pd.p1[conv][-1]
    a_last = res_pd.p2[conv][-1]
    assert abs(float(a_last) - 0.2) > 0.005

    # Verify the tracked endpoint still has a Floquet multiplier at -1
    prob_end = bx.CollocationProblem(
        lambda u, c: f_rossler_2p(u, c, a_last), dim=3, num_intervals=N, num_gauss_points=4
    )
    orb_end = bx.PeriodicOrbitResult(
        mesh=res_pd.mesh,
        u_mesh=res_pd.u_mesh[conv][-1],
        u_gauss=res_pd.u_gauss[conv][-1],
        period=res_pd.period[conv][-1],
        converged=jnp.bool_(True),
        iterations=jnp.int32(0),
        residual_norm=res_pd.residual_norms[conv][-1],
        problem=prob_end,
        p=c_last,
    )
    mults_end, _ = bx.compute_floquet_multipliers(orb_end, p=c_last, method="periodic_qr")
    assert float(jnp.min(jnp.abs(mults_end + 1.0))) < 1e-5
