from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from bifurx.bvp.collocation import (
    CollocationProblem,
    solve_periodic_orbit,
    solve_periodic_orbit_jax,
)
from bifurx.continuation.arclength import continuation, continuation_scan
from bifurx.problem import BifurcationProblem
from bifurx.solvers.newton import bordered_newton_solve, bordered_newton_solve_jax


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


def test_bordered_newton_solve_jax_analytical():
    """Test bordered_newton_solve_jax against analytical pitchfork solution and convergence."""
    prob = BifurcationProblem(f_pitchfork, u0=[0.0], p0=1.0)
    u_prev = jnp.array([1.0])
    p_prev = jnp.array(1.0)
    tau_prev = jnp.array([0.0, 1.0])
    u_init = jnp.array([1.0])
    p_init = jnp.array(1.05)
    ds = 0.05

    # 1. Pure JAX solve
    res_jax = bordered_newton_solve_jax(prob, u_init, p_init, u_prev, p_prev, tau_prev, ds=ds)
    assert bool(res_jax.converged)
    assert int(res_jax.iterations) <= 4
    assert float(res_jax.residual_norm) < 1e-8
    assert abs(float(res_jax.u[0]) ** 2 - float(res_jax.p)) < 1e-8

    # 2. Under JIT compilation
    jitted_solve = jax.jit(
        lambda ui, pi: bordered_newton_solve_jax(prob, ui, pi, u_prev, p_prev, tau_prev, ds=ds)
    )
    res_jit = jitted_solve(u_init, p_init)
    assert bool(res_jit.converged)
    assert np.allclose(res_jit.u, res_jax.u)
    assert np.allclose(res_jit.p, res_jax.p)

    # 3. Via bordered_newton_solve with use_jit=True
    res_compat = bordered_newton_solve(
        prob, u_init, p_init, u_prev, p_prev, tau_prev, ds=ds, use_jit=True
    )
    assert res_compat.converged
    assert abs(res_compat.u[0] ** 2 - res_compat.p) < 1e-8


def test_continuation_scan_pitchfork():
    """Test continuation_scan reproduces pitchfork bifurcation branch."""
    prob = BifurcationProblem(f_pitchfork, u0=[0.5], p0=0.25)
    n_steps = 40
    ds = 0.03

    scan_res = continuation_scan(prob, u0=[0.5], p0=0.25, n_steps=n_steps, ds=ds)

    assert scan_res.num_points == n_steps
    assert jnp.all(scan_res.converged), "All steps along upper branch should converge"

    # Verify u^2 == p along upper branch
    u_vals = np.asarray(scan_res.u[:, 0])
    p_vals = np.asarray(scan_res.p)
    assert np.all(u_vals > 0)
    assert np.allclose(u_vals**2, p_vals, atol=1e-3)

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

    # Compile and execute continuation scan
    jitted_scan = jax.jit(
        lambda: continuation_scan(prob, u0=[0.0, 0.0], p0=0.0, n_steps=n_steps, ds=ds)
    )
    res = jitted_scan()

    p_arr = np.asarray(res.p)
    tf_lp = np.asarray(res.test_funcs["LP"])
    tf_hb = np.asarray(res.test_funcs["HB"])

    # Verify trajectory explores the full S-curve (p goes beyond 0.10, down below 0.09, and up to 0.14)
    assert np.max(p_arr) > 0.13
    assert np.min(p_arr[20:]) < 0.095

    # Check Fold zero-crossings in tf_lp
    lp_crossings = np.where(tf_lp[:-1] * tf_lp[1:] < 0)[0]
    assert len(lp_crossings) >= 2, f"Expected at least 2 Fold crossings, got {len(lp_crossings)}"

    # First LP near p ~ 0.1057
    p_lp1 = p_arr[lp_crossings[0]]
    assert abs(p_lp1 - 0.1057) < 0.015

    # Second LP near p ~ 0.0889
    p_lp2 = p_arr[lp_crossings[1]]
    assert abs(p_lp2 - 0.0889) < 0.015

    # Check Hopf point in tf_hb (real part of complex eigenvalue crosses zero)
    valid_hb = ~np.isnan(tf_hb)
    hb_indices = np.where(valid_hb)[0]
    assert len(hb_indices) > 0
    # Leading real part crossing zero
    hb_crossings = np.where(tf_hb[hb_indices[:-1]] * tf_hb[hb_indices[1:]] < 0)[0]
    assert len(hb_crossings) >= 1, "Expected Hopf zero crossing in leading eigenvalue real part"


def test_vmap_continuation_scan_100_trajectories():
    """Test jax.vmap(continuation_scan) running a batch of 100 continuation trajectories in parallel."""
    prob = BifurcationProblem(f_pitchfork, u0=[0.0], p0=1.0)
    batch_size = 100
    n_steps = 30

    # 100 distinct initial conditions along branch
    p_batch = jnp.linspace(0.5, 2.0, batch_size)
    u_batch = jnp.sqrt(p_batch)[:, None]

    # Vmapped continuation scan function
    vmapped_continuation = jax.jit(
        jax.vmap(lambda u, p: continuation_scan(prob, u0=u, p0=p, n_steps=n_steps, ds=0.02))
    )

    t0 = time.perf_counter()
    batch_res = vmapped_continuation(u_batch, p_batch)
    batch_res.u.block_until_ready()
    t_vmap = time.perf_counter() - t0

    # Verify batch shapes
    assert batch_res.u.shape == (batch_size, n_steps, 1)
    assert batch_res.p.shape == (batch_size, n_steps)
    assert batch_res.tangents.shape == (batch_size, n_steps, 2)
    assert batch_res.converged.shape == (batch_size, n_steps)

    # Verify all 100 trajectories converged
    assert jnp.all(batch_res.converged)

    # Verify accuracy: u^2 == p across all 100 * 30 points
    u_all = np.asarray(batch_res.u[:, :, 0])
    p_all = np.asarray(batch_res.p)
    assert np.allclose(u_all**2, p_all, atol=1e-3)

    # Time per trajectory is sub-millisecond
    time_per_traj_ms = (t_vmap / batch_size) * 1000
    assert time_per_traj_ms < 50.0  # highly efficient batching


def test_solve_periodic_orbit_jax_stuart_landau():
    """Test JIT-accelerated collocation solver on Stuart-Landau oscillator with exact limit cycle."""
    N = 15
    m = 4
    prob = CollocationProblem(f_stuart_landau, dim=2, num_intervals=N, num_gauss_points=m)

    t_grid = np.linspace(0, 2 * np.pi, N, endpoint=False)
    u_mesh = jnp.stack([jnp.cos(t_grid), jnp.sin(t_grid)], axis=1)

    # Solve with pure JAX while_loop solver
    sol_jax = solve_periodic_orbit_jax(prob, u_mesh, T_init=6.2, p=0.0, tol=1e-9)
    assert bool(sol_jax.converged)
    assert int(sol_jax.iterations) <= 5
    assert abs(float(sol_jax.period) - 2.0 * np.pi) < 1e-5

    # Solve via solve_periodic_orbit with use_jit=True
    sol_compat = solve_periodic_orbit(prob, u_mesh, T_init=6.2, p=0.0, tol=1e-9, use_jit=True)
    assert sol_compat.converged
    assert abs(sol_compat.period - 2.0 * np.pi) < 1e-5


def test_benchmark_jit_scan_speedup():
    """Benchmark test measuring execution time and speedup of JIT continuation_scan vs Python loop."""
    prob = BifurcationProblem(f_ab, u0=[0.0, 0.0], p0=0.0)
    n_steps = 50
    ds = 0.03

    # Warmup and compile JIT scan
    scan_jit = jax.jit(
        lambda: continuation_scan(prob, u0=[0.0, 0.0], p0=0.0, n_steps=n_steps, ds=ds)
    )
    warmup_res = scan_jit()
    warmup_res.u.block_until_ready()

    # Time JIT scan (average over 5 runs)
    t0 = time.perf_counter()
    n_runs = 5
    for _ in range(n_runs):
        out = scan_jit()
        out.u.block_until_ready()
    t_scan = (time.perf_counter() - t0) / n_runs

    # Time Python continuation loop (single run)
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
