from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bifurx.bvp import (
    CollocationProblem,
    CondensedCollocationFactor,
    compute_floquet_multipliers,
    continue_periodic_orbit,
    evaluate_floquet_diagnostics,
    factor_condensed_collocation,
    initialize_from_hopf,
    make_collocation_bordered_solver,
    periodic_orbit_arclength_weights,
    redistribute_mesh,
    solve_condensed_bordered,
    solve_condensed_collocation,
    solve_periodic_orbit,
)
from bifurx.homotopy import compute_tangent, solve_bordered_system
from bifurx.options import CollocationOptions


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


def test_continue_periodic_orbit_natural_mode():
    """Test natural parameter continuation of periodic orbits (#15)."""
    from bifurx.bvp import continue_periodic_orbit

    N = 20
    m = 3
    prob = CollocationProblem(f_vdp, dim=2, num_intervals=N, num_gauss_points=m)

    t_grid = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
    u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)
    sol0 = solve_periodic_orbit(prob, u_init, T_init=2 * jnp.pi, p=0.2, tol=1e-7)
    assert bool(sol0.converged)
    assert sol0.p is not None, "PeriodicOrbitResult must have parameter field p (#17)"
    assert float(sol0.p) == 0.2

    # Continue from mu=0.2 to mu=0.35 with natural parameter stepping
    branch = continue_periodic_orbit(
        prob,
        initial_orbit=sol0,
        p_start=0.2,
        p_end=0.35,
        dp=0.05,
        param_mode="natural",
        tol=1e-7,
        max_steps=10,
    )

    assert len(branch) >= 3, (
        f"Expected at least 3 steps on periodic orbit branch, got {len(branch)}"
    )
    p_vals = [float(sol.p) for sol in branch if sol.p is not None]
    assert len(p_vals) == len(branch)
    assert p_vals[0] == 0.2
    assert p_vals[-1] == 0.35
    assert all(p2 > p1 for p1, p2 in zip(p_vals[:-1], p_vals[1:], strict=True))
    for sol in branch:
        assert bool(sol.converged)
        assert sol.p is not None
        assert float(sol.amplitude) > 1.0, (
            f"Expected limit cycle amplitude > 1.0, got {sol.amplitude}"
        )
        assert 6.2 < float(sol.T) < 6.5, f"Expected VdP period near 2*pi, got {sol.T}"


def test_initialize_from_hopf_detuning():
    """Verify direction- and step-scaled detuning in initialize_from_hopf (#21)."""
    prob = CollocationProblem(f_ab, dim=2, num_intervals=10, num_gauss_points=2)
    u_hopf = jnp.array([0.8950809, 4.1770443])
    p_hopf = 0.1309003

    # 1. Custom direction and step size ds
    _, _, _, T0, p_init_custom = initialize_from_hopf(
        prob, u_hopf, p_hopf, eps=0.05, direction=1.0, ds=0.003
    )
    assert abs(float(p_init_custom) - (p_hopf + 0.003)) < 1e-12
    assert float(T0) > 0.0

    # 2. Default direction=-1.0 with ds=None (scaled by eps)
    _, _, _, _, p_init_eps = initialize_from_hopf(
        prob, u_hopf, p_hopf, eps=0.05, direction=-1.0, ds=None
    )
    assert abs(float(p_init_eps) - (p_hopf - 0.05)) < 1e-12


def test_continue_periodic_orbit_arclength_and_param_weight():
    """Verify weighted arclength and Padé (2,1) continuation of periodic orbits (#15)."""
    from bifurx.bvp import continue_periodic_orbit

    # 1. Mesh-size invariant dp/ds & continuation on Van der Pol (also with pade21)
    p_ends: list[float] = []
    for n_int in (15, 30):
        prob = CollocationProblem(f_vdp, dim=2, num_intervals=n_int, num_gauss_points=3)
        t_grid = jnp.linspace(0, 2 * jnp.pi, n_int, endpoint=False)
        u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)
        sol0 = solve_periodic_orbit(prob, u_init, T_init=2 * jnp.pi, p=0.2, tol=1e-7)
        assert bool(sol0.converged)

        branch = continue_periodic_orbit(
            prob,
            initial_orbit=sol0,
            p_start=0.2,
            p_end=0.35,
            dp=0.05,
            dp_max=0.08,
            param_mode="arclength",
            param_weight=1.0,
            predictor="pade21",
            tol=1e-7,
            max_steps=8,
        )
        assert len(branch) >= 3
        assert all(bool(s.converged) for s in branch)
        assert all(float(s.residual_norm) < 1e-6 for s in branch[1:])
        p_ends.append(float(branch[-1].p))  # type: ignore[arg-type]

    # Mesh refinement N=15 -> N=30 produces consistent parameter advancement (not crushed by state dim)
    assert all(p_val >= 0.34 for p_val in p_ends)

    # 2. A-B reaction limit cycle branch from Hopf (descending p1, growing period)
    u_hopf = jnp.array([0.8950809, 4.1770443])
    p1_hopf = 0.1309003
    prob_ab = CollocationProblem(f_ab, dim=2, num_intervals=20, num_gauss_points=3)
    mesh, u_m0, u_g0, T0, p0 = initialize_from_hopf(prob_ab, u_hopf, p1_hopf, eps=0.1, p_init=0.130)
    sol_ab0 = solve_periodic_orbit(
        prob_ab, u_mesh_init=u_m0, T_init=T0, p=p0, mesh=mesh, u_gauss_init=u_g0, tol=1e-7
    )
    assert bool(sol_ab0.converged)

    branch_ab = continue_periodic_orbit(
        prob_ab,
        initial_orbit=sol_ab0,
        p_start=0.130,
        p_end=0.115,
        dp=0.006,
        dp_max=0.015,
        param_mode="arclength",
        param_weight=1.0,
        tol=1e-7,
        max_steps=12,
    )
    assert len(branch_ab) >= 3
    assert all(bool(s.converged) for s in branch_ab)
    assert float(branch_ab[-1].p) < float(branch_ab[0].p)  # type: ignore[arg-type]
    assert float(branch_ab[-1].period) > float(branch_ab[0].period)


def test_periodic_orbit_arclength_l2_mesh_invariance():
    """Verify exact Gauss-Legendre L2[0,1] weights and mesh-invariant arclength tangents (#24)."""

    def eval_smooth_orbit(t: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([jnp.cos(2.0 * jnp.pi * t), 2.0 * jnp.sin(2.0 * jnp.pi * t)], axis=-1)

    # Exact integral \int_0^1 (cos^2(2*pi*t) + 4*sin^2(2*pi*t)) dt = 2.5
    exact_l2 = 2.5
    pw, par_w = 0.5, 0.7
    T_test, p_test = 1.2, -0.4

    # 1. Exact quadrature across uniform (N=15, m=3), refined (N=30, m=4), and stretched mesh
    stretched_mesh = jnp.linspace(0.0, 1.0, 21, dtype=jnp.float64) ** 2
    assert float(jnp.max(jnp.diff(stretched_mesh)) / jnp.min(jnp.diff(stretched_mesh))) > 30.0

    for n_int, n_gauss, mesh in (
        (15, 3, jnp.linspace(0.0, 1.0, 16, dtype=jnp.float64)),
        (30, 4, jnp.linspace(0.0, 1.0, 31, dtype=jnp.float64)),
        (20, 4, stretched_mesh),
    ):
        prob = CollocationProblem(f_vdp, dim=2, num_intervals=n_int, num_gauss_points=n_gauss)
        u_m = eval_smooth_orbit(mesh[:-1])
        hk = mesh[1:] - mesh[:-1]
        t_g = mesh[:-1, None] + hk[:, None] * prob.rho[None, :]
        u_g = eval_smooth_orbit(t_g)

        X_zero = prob.pack_vars(u_m, u_g, 0.0)
        Z_zero = jnp.concatenate([X_zero, jnp.array([0.0], dtype=jnp.float64)])
        w_arc = periodic_orbit_arclength_weights(prob, mesh, period_weight=pw, param_weight=par_w)
        quad_zero = float(jnp.dot(w_arc * Z_zero, Z_zero))
        assert abs(quad_zero - exact_l2) < 1e-6

        X_full = prob.pack_vars(u_m, u_g, T_test)
        Z_full = jnp.concatenate([X_full, jnp.array([p_test], dtype=jnp.float64)])
        quad_sub = float(jnp.dot(w_arc * Z_full, Z_full)) - (
            pw**2 * T_test**2 + par_w**2 * p_test**2
        )
        assert abs(quad_sub - exact_l2) < 1e-6

    # 2. Initial arclength tangent (dT/ds, dp/ds) invariance on Van der Pol (mu=1.0)
    mu = 1.0
    prob15 = CollocationProblem(f_vdp, dim=2, num_intervals=15, num_gauss_points=3)
    t15 = jnp.linspace(0.0, 2.0 * jnp.pi, 15, endpoint=False)
    u15_init = jnp.stack([2.0 * jnp.cos(t15), -2.0 * jnp.sin(t15)], axis=1)
    sol15 = solve_periodic_orbit(prob15, u15_init, T_init=6.66, p=mu, tol=1e-9)
    assert bool(sol15.converged)

    prob30 = CollocationProblem(f_vdp, dim=2, num_intervals=30, num_gauss_points=4)
    t30 = jnp.linspace(0.0, 2.0 * jnp.pi, 30, endpoint=False)
    u30_init = jnp.stack([2.0 * jnp.cos(t30), -2.0 * jnp.sin(t30)], axis=1)
    sol30 = solve_periodic_orbit(prob30, u30_init, T_init=6.66, p=mu, tol=1e-9)
    assert bool(sol30.converged)

    sol15_adapt = redistribute_mesh(sol15, alpha=2.0, p=mu, re_solve=True, tol=1e-9)
    assert bool(sol15_adapt.converged)
    h_adapt = jnp.diff(sol15_adapt.mesh)
    assert float(jnp.max(h_adapt) / jnp.min(h_adapt)) > 2.0

    # Also verify exact L2 quadrature on the Van der Pol adapted mesh (with m=4 collocation)
    prob15_m4 = CollocationProblem(f_vdp, dim=2, num_intervals=15, num_gauss_points=4)
    hk_a = sol15_adapt.mesh[1:] - sol15_adapt.mesh[:-1]
    u_ma = eval_smooth_orbit(sol15_adapt.mesh[:-1])
    u_ga = eval_smooth_orbit(sol15_adapt.mesh[:-1, None] + hk_a[:, None] * prob15_m4.rho[None, :])
    Z_adapt = jnp.concatenate(
        [prob15_m4.pack_vars(u_ma, u_ga, 0.0), jnp.array([0.0], dtype=jnp.float64)]
    )
    w_adapt = periodic_orbit_arclength_weights(prob15_m4, sol15_adapt.mesh)
    assert abs(float(jnp.dot(w_adapt * Z_adapt, Z_adapt)) - exact_l2) < 1e-6

    tangents: list[tuple[float, float]] = []
    for prob_i, sol_i in ((prob15, sol15), (prob30, sol30), (prob15, sol15_adapt)):
        mesh_i = sol_i.mesh
        h_i = mesh_i[1:] - mesh_i[:-1]
        X0_i = prob_i.pack_vars(sol_i.u_mesh, sol_i.u_gauss, sol_i.period)
        U_ref = jnp.concatenate([sol_i.u_mesh[:, None, :], sol_i.u_gauss], axis=1)
        dot_u_ref = jnp.einsum("mj,Njd->Nmd", prob_i.D, U_ref) / h_i[:, None, None]
        res_fn = lambda X, p_v, m_arr=mesh_i, d_ref=dot_u_ref, p_obj=prob_i: p_obj.residual(
            X, p_v, m_arr, dot_u_ref=d_ref
        )
        w_arc_i = periodic_orbit_arclength_weights(
            prob_i, mesh_i, period_weight=1.0, param_weight=1.0
        )
        tau = compute_tangent(
            jac_u_fn=jax.jacfwd(res_fn, argnums=0),
            jac_p_fn=jax.jacfwd(res_fn, argnums=1),
            u0=X0_i,
            p0=jnp.array(mu, dtype=jnp.float64),
            direction=1.0,
            arclength_weights=w_arc_i,
            param_mode="arclength",
        )
        tangents.append((float(tau[-2]), float(tau[-1])))

    dT_ref, dp_ref = tangents[1]  # refined N=30, m=4 reference
    for dT_i, dp_i in tangents:
        assert abs(dT_i - dT_ref) < 1e-2
        assert abs(dp_i - dp_ref) < 1e-2

    # 3. Arclength continuation starting from redistribute_mesh-adapted non-uniform orbit
    branch_adapt = continue_periodic_orbit(
        prob15,
        initial_orbit=sol15_adapt,
        p_start=mu,
        p_end=1.4,
        dp=0.1,
        param_mode="arclength",
        tol=1e-8,
        max_steps=6,
    )
    assert len(branch_adapt) >= 4
    assert all(bool(s.converged) for s in branch_adapt)
    assert all(float(s.residual_norm) < 1e-6 for s in branch_adapt[1:])
    assert all(jnp.allclose(s.mesh, sol15_adapt.mesh) for s in branch_adapt)
    p_vals = [float(s.p) for s in branch_adapt if s.p is not None]
    assert all(p2 > p1 for p1, p2 in zip(p_vals[:-1], p_vals[1:], strict=True))


def test_periodic_orbit_configurable_period_weight():
    """Verify configurable period_weight and validation in periodic-orbit arclength (#24)."""
    u_hopf = jnp.array([0.8950809, 4.1770443])
    p1_hopf = 0.1309003
    prob_ab = CollocationProblem(f_ab, dim=2, num_intervals=20, num_gauss_points=3)
    mesh, u_m0, u_g0, T0, p0 = initialize_from_hopf(prob_ab, u_hopf, p1_hopf, eps=0.1, p_init=0.130)
    sol_ab0 = solve_periodic_orbit(
        prob_ab, u_mesh_init=u_m0, T_init=T0, p=p0, mesh=mesh, u_gauss_init=u_g0, tol=1e-7
    )
    assert bool(sol_ab0.converged)

    branch_pw0 = continue_periodic_orbit(
        prob_ab,
        initial_orbit=sol_ab0,
        p_start=0.130,
        p_end=0.115,
        dp=0.008,
        dp_max=0.015,
        param_mode="arclength",
        period_weight=0.0,
        tol=1e-7,
        max_steps=8,
    )
    branch_pw2 = continue_periodic_orbit(
        prob_ab,
        initial_orbit=sol_ab0,
        p_start=0.130,
        p_end=0.115,
        dp=0.008,
        dp_max=0.015,
        param_mode="arclength",
        period_weight=2.0,
        tol=1e-7,
        max_steps=8,
    )

    for br in (branch_pw0, branch_pw2):
        assert len(br) >= 4
        assert all(bool(s.converged) for s in br)
        assert float(br[-1].period) > float(br[0].period)
        assert float(br[-1].p) < float(br[0].p)  # type: ignore[arg-type]

    # With period_weight=0.0, period growth does not consume arclength budget
    assert float(branch_pw0[-1].p) < float(branch_pw2[-1].p)  # type: ignore[arg-type]

    # Passing via CollocationOptions and overriding with explicit period_weight
    branch_opts0 = continue_periodic_orbit(
        prob_ab,
        initial_orbit=sol_ab0,
        p_start=0.130,
        p_end=0.115,
        dp=0.008,
        dp_max=0.015,
        param_mode="arclength",
        collocation_options=CollocationOptions(period_weight=0.0),
        tol=1e-7,
        max_steps=8,
    )
    branch_override = continue_periodic_orbit(
        prob_ab,
        initial_orbit=sol_ab0,
        p_start=0.130,
        p_end=0.115,
        dp=0.008,
        dp_max=0.015,
        param_mode="arclength",
        period_weight=2.0,
        collocation_options=CollocationOptions(period_weight=0.0),
        tol=1e-7,
        max_steps=8,
    )
    assert abs(float(branch_opts0[-1].p) - float(branch_pw0[-1].p)) < 1e-12  # type: ignore[arg-type]
    assert abs(float(branch_override[-1].p) - float(branch_pw2[-1].p)) < 1e-12  # type: ignore[arg-type]

    # Negative weights must raise ValueError
    with pytest.raises(ValueError, match="period_weight"):
        periodic_orbit_arclength_weights(prob_ab, mesh, period_weight=-0.1)
    with pytest.raises(ValueError, match="param_weight"):
        periodic_orbit_arclength_weights(prob_ab, mesh, param_weight=-0.1)
    with pytest.raises(ValueError, match="period_weight"):
        continue_periodic_orbit(
            prob_ab,
            initial_orbit=sol_ab0,
            p_start=0.130,
            p_end=0.115,
            param_mode="arclength",
            period_weight=-0.5,
        )
    with pytest.raises(ValueError, match="param_weight"):
        continue_periodic_orbit(
            prob_ab,
            initial_orbit=sol_ab0,
            p_start=0.130,
            p_end=0.115,
            param_mode="arclength",
            param_weight=-0.5,
        )


def test_continue_periodic_orbit_floquet_monitoring_and_lpc():
    """Verify in-scan Floquet monitoring detects stability flip and LPC fold (#B5)."""

    def f_subcrit_sl(u: jnp.ndarray, p: float) -> jnp.ndarray:
        x, y = u[0], u[1]
        r2 = x * x + y * y
        amp = p + 2.0 * r2 - r2 * r2
        return jnp.array([amp * x - y, x + amp * y])

    N, m = 20, 4
    prob = CollocationProblem(f_subcrit_sl, dim=2, num_intervals=N, num_gauss_points=m)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    r0 = float(jnp.sqrt(2.0))
    u_init = jnp.stack([r0 * jnp.cos(t_grid), r0 * jnp.sin(t_grid)], axis=1)
    sol0 = solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=0.0, tol=1e-9)
    assert bool(sol0.converged)

    branch = continue_periodic_orbit(
        prob,
        initial_orbit=sol0,
        p_start=0.0,
        p_end=-1.5,
        dp=0.15,
        dp_max=0.25,
        max_steps=35,
        param_mode="arclength",
        compute_floquet=True,
    )
    assert len(branch) >= 10
    assert all(bool(s.converged) for s in branch)
    assert branch[0].floquet_multipliers is not None
    assert branch[0].is_stable is not None and bool(branch[0].is_stable)

    p_vals = [float(s.p) for s in branch if s.p is not None]
    assert min(p_vals) < -0.95
    assert p_vals[-1] > min(p_vals) + 0.1
    assert any(s.is_stable is not None and (not bool(s.is_stable)) for s in branch)

    lpc_orbits = [s for s in branch if "LPC" in s.bifurcations]
    assert len(lpc_orbits) >= 1
    assert abs(float(lpc_orbits[0].p) - (-1.0)) < 0.2  # type: ignore[arg-type]


def test_evaluate_floquet_diagnostics_pd_and_ns():
    """Verify evaluate_floquet_diagnostics classifies stability, PD (-1), and NS (|mu|=1)."""
    M_stable = jnp.diag(jnp.array([1.0, 0.5], dtype=jnp.float64))
    mults_s, tf_s, is_s = evaluate_floquet_diagnostics(M_stable)
    assert mults_s.shape == (2,)
    assert bool(is_s)
    assert float(tf_s["PD"]) > 0.0
    assert bool(jnp.isnan(tf_s["NS"]))

    # Period-doubling crossing: mu_2 = -0.8 -> -1.2
    _, tf_pd_before, st_pd_before = evaluate_floquet_diagnostics(
        jnp.diag(jnp.array([1.0, -0.8], dtype=jnp.float64))
    )
    _, tf_pd_after, st_pd_after = evaluate_floquet_diagnostics(
        jnp.diag(jnp.array([1.0, -1.2], dtype=jnp.float64))
    )
    assert bool(st_pd_before) and not bool(st_pd_after)
    assert float(tf_pd_before["PD"]) * float(tf_pd_after["PD"]) < 0.0

    # Neimark-Sacker crossing: complex pair modulus r = 0.9 -> 1.1
    th = 1.0
    c, s = float(jnp.cos(th)), float(jnp.sin(th))
    M_ns_before = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 0.9 * c, -0.9 * s], [0.0, 0.9 * s, 0.9 * c]],
        dtype=jnp.float64,
    )
    M_ns_after = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.1 * c, -1.1 * s], [0.0, 1.1 * s, 1.1 * c]],
        dtype=jnp.float64,
    )
    _, tf_ns_before, st_ns_before = evaluate_floquet_diagnostics(M_ns_before)
    _, tf_ns_after, st_ns_after = evaluate_floquet_diagnostics(M_ns_after)
    assert bool(st_ns_before) and not bool(st_ns_after)
    assert float(tf_ns_before["NS"]) * float(tf_ns_after["NS"]) < 0.0
    assert float(tf_ns_before["PD"]) > 0.0 and float(tf_ns_after["PD"]) > 0.0


def test_continue_periodic_orbit_in_loop_adaptive_remeshing():
    """Verify in-loop adaptive remeshing maintains smooth arclength progress on Van der Pol."""
    mu0 = 0.5
    N, m = 20, 4
    prob = CollocationProblem(f_vdp, dim=2, num_intervals=N, num_gauss_points=m)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)
    sol0 = solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=mu0, tol=1e-8)
    assert bool(sol0.converged)

    branch = continue_periodic_orbit(
        prob,
        initial_orbit=sol0,
        p_start=mu0,
        p_end=3.0,
        dp=0.25,
        dp_max=0.4,
        max_steps=16,
        param_mode="arclength",
        remesh_interval=4,
    )
    assert len(branch) >= 8
    assert all(bool(s.converged) for s in branch)
    p_vals = [float(s.p) for s in branch if s.p is not None]
    assert all(p2 > p1 for p1, p2 in zip(p_vals[:-1], p_vals[1:], strict=True))
    assert p_vals[-1] > 2.0
    assert float(branch[-1].period) > float(branch[0].period)
    dh = jnp.diff(branch[-1].mesh)
    assert float(jnp.max(dh) / jnp.min(dh)) > 1.4

    with pytest.raises(ValueError, match="remesh_interval"):
        continue_periodic_orbit(
            prob,
            initial_orbit=sol0,
            p_start=mu0,
            p_end=3.0,
            remesh_interval=0,
        )


def test_initialize_from_hopf_auto_direction():
    """Verify initialize_from_hopf(..., direction='auto') selects branch direction from l_1 * alpha'."""

    def sl_super(u: jnp.ndarray, p: float) -> jnp.ndarray:
        x, y = u[0], u[1]
        r2 = x * x + y * y
        return jnp.array([p * x - y - x * r2, x + p * y - y * r2])

    def sl_sub(u: jnp.ndarray, p: float) -> jnp.ndarray:
        x, y = u[0], u[1]
        r2 = x * x + y * y
        return jnp.array([p * x - y + x * r2, x + p * y + y * r2])

    prob_super = CollocationProblem(sl_super, dim=2, num_intervals=15, num_gauss_points=3)
    _, _, _, _, p_super = initialize_from_hopf(
        prob_super, jnp.zeros(2), 0.0, eps=0.05, direction="auto"
    )
    assert float(p_super) > 0.0

    prob_sub = CollocationProblem(sl_sub, dim=2, num_intervals=15, num_gauss_points=3)
    _, _, _, _, p_sub = initialize_from_hopf(
        prob_sub, jnp.zeros(2), 0.0, eps=0.05, direction="auto"
    )
    assert float(p_sub) < 0.0

    prob_ab = CollocationProblem(f_ab, dim=2, num_intervals=20, num_gauss_points=3)
    p1_hopf = 0.1309003
    _, _, _, _, p_ab = initialize_from_hopf(
        prob_ab,
        jnp.array([0.8950809, 4.1770443]),
        p1_hopf,
        eps=0.05,
        direction="auto",
    )
    assert float(p_ab) < p1_hopf


def test_condensed_collocation_matches_dense_exact():
    """Verify de Boor-Swartz block condensation matches dense Newton and bordered solves exactly."""
    rng = jax.random.PRNGKey(42)
    mu = jnp.array(1.0, dtype=jnp.float64)

    for N, nonuniform in ((12, True), (1, False)):
        prob = CollocationProblem(f_vdp, dim=2, num_intervals=N, num_gauss_points=4)
        mesh = (
            jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float64) ** 1.3
            if nonuniform
            else jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float64)
        )
        t_m = 2.0 * jnp.pi * mesh[:-1]
        h = mesh[1:] - mesh[:-1]
        t_g = 2.0 * jnp.pi * (mesh[:-1, None] + h[:, None] * prob.rho[None, :])
        u_m = jnp.stack([2.0 * jnp.cos(t_m), -2.0 * jnp.sin(t_m)], axis=-1)
        u_g = jnp.stack([2.0 * jnp.cos(t_g), -2.0 * jnp.sin(t_g)], axis=-1)
        X = prob.pack_vars(u_m, u_g, 6.66)

        U_ref = jnp.concatenate([u_m[:, None, :], u_g], axis=1)
        dot_u_ref = jnp.einsum("mj,Njd->Nmd", prob.D, U_ref) / h[:, None, None]

        k1, k2, k3, k4 = jax.random.split(rng, 4)
        rhs_X = jax.random.normal(k1, shape=X.shape, dtype=jnp.float64)
        c_X = jax.random.normal(k2, shape=X.shape, dtype=jnp.float64)
        d_p = jax.random.normal(k3, shape=(), dtype=jnp.float64)
        rhs_g = jax.random.normal(k4, shape=(), dtype=jnp.float64)

        def _res(
            x_v: jnp.ndarray,
            p_v: jnp.ndarray,
            p_obj: CollocationProblem = prob,
            m_arr: jnp.ndarray = mesh,
            d_ref: jnp.ndarray = dot_u_ref,
        ) -> jnp.ndarray:
            return p_obj.residual(x_v, p_v, m_arr, dot_u_ref=d_ref)

        J_dense = jax.jacfwd(_res, argnums=0)(X, mu)
        Jp_dense = jax.jacfwd(_res, argnums=1)(X, mu)

        # 1. Square solve equivalence
        dX_dense = jnp.linalg.solve(J_dense, rhs_X)
        fac_sq = factor_condensed_collocation(prob, X, mu, mesh, dot_u_ref)
        assert isinstance(fac_sq, CondensedCollocationFactor)
        dX_cond = solve_condensed_collocation(fac_sq, rhs_X)
        assert jnp.allclose(dX_cond, dX_dense, atol=1e-11)

        # 2. Bordered solve equivalence (direct + BorderedOperatorSolver wrapper)
        sol_dense = solve_bordered_system(
            J_dense, Jp_dense, c_X, d_p, rhs_X, rhs_g, method="direct"
        )
        fac_bd = factor_condensed_collocation(prob, X, mu, mesh, dot_u_ref, c_X=c_X, d_p=d_p)
        sol_cond = solve_condensed_bordered(fac_bd, rhs_X, rhs_g)
        assert jnp.allclose(sol_cond.z, sol_dense.z, atol=1e-11)

        solver = make_collocation_bordered_solver(prob, mesh, dot_u_ref)
        sol_op = solver.solve(solver.factor(X, mu, c_X, d_p), rhs_X, rhs_g)
        assert jnp.allclose(sol_op.z, sol_dense.z, atol=1e-11)


def test_condensed_collocation_high_dim_periodic_orbit():
    """Verify condensed collocation solves and continues a 601-unknown 4D periodic orbit."""

    def f_coupled_sl(u: jnp.ndarray, p: float) -> jnp.ndarray:
        x1, y1, x2, y2 = u[0], u[1], u[2], u[3]
        r1 = x1 * x1 + y1 * y1
        r2 = x2 * x2 + y2 * y2
        mu1 = 1.0 + p
        mu2 = 1.0 + 0.5 * p
        return jnp.array(
            [
                mu1 * x1 - y1 - x1 * r1 + 0.05 * (x2 - x1),
                x1 + mu1 * y1 - y1 * r1 + 0.05 * (y2 - y1),
                mu2 * x2 - y2 - x2 * r2 + 0.05 * (x1 - x2),
                x2 + mu2 * y2 - y2 * r2 + 0.05 * (y1 - y2),
            ]
        )

    N, m = 30, 4  # n = 4 -> 30 * 5 * 4 + 1 = 601 variables
    prob = CollocationProblem(f_coupled_sl, dim=4, num_intervals=N, num_gauss_points=m)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    u_init = jnp.stack([jnp.cos(t_grid), jnp.sin(t_grid), jnp.cos(t_grid), jnp.sin(t_grid)], axis=1)
    sol0 = solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=0.0, tol=1e-9)
    assert bool(sol0.converged)
    assert abs(float(sol0.period) - 2.0 * float(jnp.pi)) < 1e-4

    branch = continue_periodic_orbit(
        prob,
        initial_orbit=sol0,
        p_start=0.0,
        p_end=0.5,
        dp=0.1,
        dp_max=0.2,
        max_steps=6,
        param_mode="arclength",
        predictor="pade21",
        compute_floquet=True,
    )
    assert len(branch) >= 4
    assert all(bool(s.converged) for s in branch)
    assert float(branch[-1].p) > 0.2  # type: ignore[arg-type]
    assert branch[-1].floquet_multipliers is not None
    assert bool(branch[-1].is_stable)
