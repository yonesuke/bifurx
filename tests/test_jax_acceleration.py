from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from bifurx.bvp.collocation import (
    CollocationProblem,
    solve_periodic_orbit,
)
from bifurx.continuation.arclength import continuation, continuation_scan
from bifurx.problem import BifurcationProblem
from bifurx.solvers.newton import bordered_newton_solve


def f_pitchfork(u: jnp.ndarray, p: float | jnp.ndarray) -> jnp.ndarray:
    x = u[0]
    return jnp.array([p * x - x**3])


def f_ab(u: jnp.ndarray, p1: float | jnp.ndarray) -> jnp.ndarray:
    P2, P3 = 14.0, 2.0
    u1, u2 = u[0], u[1]
    e = jnp.exp(u2)
    f1 = -u1 + p1 * (1.0 - u1) * e
    f2 = -u2 + p1 * P2 * (1.0 - u1) * e - P3 * u2
    return jnp.array([f1, f2])


def f_stuart_landau(u: jnp.ndarray, p: float | jnp.ndarray = 0.0) -> jnp.ndarray:
    x, y = u[0], u[1]
    r2 = x * x + y * y
    return jnp.array([x - y - x * r2, x + y - y * r2])


def test_bordered_newton_solve_analytical():
    """Test unified pure-JAX bordered_newton_solve against analytical pitchfork solution."""
    prob = BifurcationProblem(f_pitchfork, u0=[0.0], p0=1.0)
    u_prev = jnp.array([1.0])
    p_prev = jnp.array(1.0)
    tau_prev = jnp.array([0.0, 1.0])
    u_init = jnp.array([1.0])
    p_init = jnp.array(1.05)
    ds = 0.05

    # 1. Direct execution
    res = bordered_newton_solve(prob, u_init, p_init, u_prev, p_prev, tau_prev, ds=ds)
    assert bool(res.converged)
    assert int(res.iterations) <= 4
    assert float(res.residual_norm) < 1e-8
    assert abs(float(res.u[0]) ** 2 - float(res.p)) < 1e-8

    # 2. Under JIT compilation
    jitted_solve = jax.jit(
        lambda ui, pi: bordered_newton_solve(prob, ui, pi, u_prev, p_prev, tau_prev, ds=ds)
    )
    res_jit = jitted_solve(u_init, p_init)
    assert bool(res_jit.converged)
    assert jnp.allclose(res_jit.u, res.u)
    assert jnp.allclose(res_jit.p, res.p)


def test_continuation_scan_pitchfork():
    """Test continuation_scan reproduces pitchfork bifurcation branch."""
    prob = BifurcationProblem(f_pitchfork, u0=[0.5], p0=0.25)
    n_steps = 40
    ds = 0.03

    scan_res = continuation_scan(prob, u0=[0.5], p0=0.25, n_steps=n_steps, ds=ds)

    assert scan_res.num_points == n_steps
    assert jnp.all(scan_res.converged), "All steps along upper branch should converge"

    # Verify u^2 == p along upper branch
    u_vals = scan_res.u[:, 0]
    p_vals = scan_res.p
    assert jnp.all(u_vals > 0)
    assert jnp.allclose(u_vals**2, p_vals, atol=1e-3)

    # Verify tuple unpacking support
    u_traj, p_traj, tau_traj, tf_traj, conv_traj = scan_res[:5]
    assert u_traj.shape == (n_steps, 1)
    assert p_traj.shape == (n_steps,)
    assert "LP" in tf_traj
    assert "BP" in tf_traj


def test_continuation_scan_ab_reaction():
    """Test continuation_scan reproduces A-B reaction S-curve folds and Hopf point."""
    prob = BifurcationProblem(f_ab, u0=[0.0, 0.0], p0=0.0)
    n_steps = 160
    ds = 0.035

    jitted_scan = jax.jit(
        lambda: continuation_scan(prob, u0=[0.0, 0.0], p0=0.0, n_steps=n_steps, ds=ds)
    )
    res = jitted_scan()

    p_arr = res.p
    tf_lp = res.test_funcs["LP"]
    tf_hb = res.test_funcs["HB"]

    # Verify trajectory explores the full S-curve (p goes beyond 0.10, down below 0.09, and up to 0.14)
    assert float(jnp.max(p_arr)) > 0.13
    assert float(jnp.min(p_arr[20:])) < 0.095

    # Check Fold zero-crossings in tf_lp
    lp_crossings = jnp.where(tf_lp[:-1] * tf_lp[1:] < 0)[0]
    assert len(lp_crossings) >= 2, f"Expected at least 2 Fold crossings, got {len(lp_crossings)}"

    # First LP near p ~ 0.1057
    p_lp1 = float(p_arr[lp_crossings[0]])
    assert abs(p_lp1 - 0.1057) < 0.015

    # Second LP near p ~ 0.0889
    p_lp2 = float(p_arr[lp_crossings[1]])
    assert abs(p_lp2 - 0.0889) < 0.015

    # Check Hopf point in tf_hb (real part of complex eigenvalue crosses zero)
    valid_hb = ~jnp.isnan(tf_hb)
    hb_indices = jnp.where(valid_hb)[0]
    assert len(hb_indices) > 0
    hb_crossings = jnp.where(tf_hb[hb_indices[:-1]] * tf_hb[hb_indices[1:]] < 0)[0]
    assert len(hb_crossings) >= 1, "Expected Hopf zero crossing in leading eigenvalue real part"


def test_vmap_continuation_scan_100_trajectories():
    """Test jax.vmap(continuation_scan) running a batch of 100 continuation trajectories in parallel."""
    prob = BifurcationProblem(f_pitchfork, u0=[0.0], p0=1.0)
    batch_size = 100
    n_steps = 30

    p_batch = jnp.linspace(0.5, 2.0, batch_size)
    u_batch = jnp.sqrt(p_batch)[:, None]

    vmapped_continuation = jax.jit(
        jax.vmap(lambda u, p: continuation_scan(prob, u0=u, p0=p, n_steps=n_steps, ds=0.02))
    )

    t0 = time.perf_counter()
    batch_res = vmapped_continuation(u_batch, p_batch)
    batch_res.u.block_until_ready()
    t_vmap = time.perf_counter() - t0

    assert batch_res.u.shape == (batch_size, n_steps, 1)
    assert batch_res.p.shape == (batch_size, n_steps)
    assert batch_res.tangents.shape == (batch_size, n_steps, 2)
    assert batch_res.converged.shape == (batch_size, n_steps)

    assert jnp.all(batch_res.converged)

    u_all = batch_res.u[:, :, 0]
    p_all = batch_res.p
    assert jnp.allclose(u_all**2, p_all, atol=1e-3)

    time_per_traj_ms = (t_vmap / batch_size) * 1000
    assert time_per_traj_ms < 50.0


def test_solve_periodic_orbit_stuart_landau():
    """Test unified pure-JAX collocation solver on Stuart-Landau oscillator with exact limit cycle."""
    N = 15
    m = 4
    prob = CollocationProblem(f_stuart_landau, dim=2, num_intervals=N, num_gauss_points=m)

    t_grid = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
    u_mesh = jnp.stack([jnp.cos(t_grid), jnp.sin(t_grid)], axis=1)

    sol = solve_periodic_orbit(prob, u_mesh, T_init=6.2, p=0.0, tol=1e-9)
    assert bool(sol.converged)
    assert int(sol.iterations) <= 5
    assert abs(float(sol.period) - 2.0 * float(jnp.pi)) < 1e-5

    sol_jit = jax.jit(lambda um: solve_periodic_orbit(prob, um, T_init=6.2, p=0.0, tol=1e-9))(
        u_mesh
    )
    assert bool(sol_jit.converged)
    assert abs(float(sol_jit.period) - 2.0 * float(jnp.pi)) < 1e-5


def test_benchmark_jit_scan_speedup():
    """Benchmark test measuring execution time and speedup of JIT continuation_scan vs un-jitted call."""
    prob = BifurcationProblem(f_ab, u0=[0.0, 0.0], p0=0.0)
    n_steps = 50
    ds = 0.03

    scan_jit = jax.jit(
        lambda: continuation_scan(prob, u0=[0.0, 0.0], p0=0.0, n_steps=n_steps, ds=ds)
    )
    warmup_res = scan_jit()
    warmup_res.u.block_until_ready()

    t0 = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        out = scan_jit()
        out.u.block_until_ready()
    t_scan = (time.perf_counter() - t0) / n_runs

    t0 = time.perf_counter()
    res_py = continuation(
        prob,
        u0=[0.0, 0.0],
        p0=0.0,
        max_steps=n_steps,
        ds=ds,
        detect_bifurcations=False,
        adaptive_step=False,
    )
    t_py = time.perf_counter() - t0

    assert res_py.num_points > 10
    speedup = t_py / t_scan
    assert speedup > 5.0, (
        f"Expected speedup > 5x, got {speedup:.1f}x (py={t_py * 1e3:.1f}ms, scan={t_scan * 1e3:.2f}ms)"
    )
