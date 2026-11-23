from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from bifurx.bvp import (
    CollocationProblem,
    compute_floquet_exponents,
    compute_floquet_multipliers,
    continue_periodic_orbit,
    evaluate_floquet_diagnostics,
    evaluate_floquet_diagnostics_from_factors,
    initialize_from_hopf,
    monodromy_factors_condensation,
    periodic_real_schur,
    periodic_schur_eigenvalues,
    solve_periodic_orbit,
)
from bifurx.bvp.floquet import monodromy_condensation_raw


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


def f_3d_complex_floquet(u: jnp.ndarray, p: float) -> jnp.ndarray:
    """4D/3D-style limit cycle with complex conjugate transverse Floquet multipliers.

    Planar Stuart-Landau limit cycle in (x, y) coupled to a damped 2D harmonic oscillator
    in (z1, z2) with damping -alpha and frequency omega_z, giving exact Floquet multipliers
    1, exp(-2*pi*2), and exp(-alpha*2*pi) * exp(+-i*omega_z*2*pi).
    """
    x, y, z1, z2 = u[0], u[1], u[2], u[3]
    r2 = x * x + y * y
    alpha = 0.25 + p
    omega_z = 0.37
    return jnp.array(
        [
            x - y - x * r2,
            x + y - y * r2,
            -alpha * z1 - omega_z * z2 + 0.05 * x * z1,
            omega_z * z1 - alpha * z2 + 0.05 * y * z2,
        ]
    )


def _sort_complex_eigs(eigs: jax.Array) -> jax.Array:
    """Deterministic sort of complex eigenvalues by rounded (-|mu|, -Re, -Im)."""
    arr = jnp.asarray(eigs, dtype=jnp.complex128)
    vals = [complex(v) for v in arr]
    vals.sort(key=lambda z: (-round(abs(z), 8), -round(z.real, 8), -round(z.imag, 8)))
    return jnp.array(vals, dtype=jnp.complex128)


def _make_random_orthogonal(key: jax.Array, N: int, n: int) -> jax.Array:
    """Generate N random orthogonal n x n matrices."""
    A = jax.random.normal(key, (N, n, n), dtype=jnp.float64)
    Q, _ = jax.vmap(jnp.linalg.qr)(A)
    return Q


def test_periodic_schur_matches_explicit_well_conditioned_orbits():
    """Verify periodic_qr matches explicit condensation on VdP, A-B, and 4D complex-pair orbits."""
    # 1. Van der Pol (mu = 0.2)
    prob_vdp = CollocationProblem(f_vdp, dim=2, num_intervals=25, num_gauss_points=4)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, 25, endpoint=False)
    u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)
    sol_vdp = solve_periodic_orbit(prob_vdp, u_init, T_init=2.0 * jnp.pi, p=0.2, tol=1e-10)
    assert bool(sol_vdp.converged)

    mults_exp, M_exp = compute_floquet_multipliers(sol_vdp, p=0.2, method="condensation")
    mults_qr, schur_vdp = compute_floquet_multipliers(sol_vdp, p=0.2, method="periodic_qr")
    assert schur_vdp.shape == (25, 2, 2)
    assert jnp.allclose(_sort_complex_eigs(mults_qr), _sort_complex_eigs(mults_exp), atol=1e-11)

    exps_qr, mults_exp_qr, _ = compute_floquet_exponents(sol_vdp, p=0.2, method="periodic_qr")
    exps_cond, _, _ = compute_floquet_exponents(sol_vdp, p=0.2, method="condensation")
    assert jnp.allclose(exps_qr, exps_cond, atol=1e-11)
    assert jnp.allclose(_sort_complex_eigs(mults_exp_qr), _sort_complex_eigs(mults_exp), atol=1e-11)

    # 2. Hopf-seeded A-B reaction model
    u_hopf = jnp.array([0.8950809, 4.1770443])
    p1_hopf = 0.1309003
    prob_ab = CollocationProblem(f_ab, dim=2, num_intervals=25, num_gauss_points=4)
    mesh, u_m0, u_g0, T0, p0 = initialize_from_hopf(prob_ab, u_hopf, p1_hopf, eps=0.1, p_init=0.130)
    sol_ab = solve_periodic_orbit(
        prob_ab, u_mesh_init=u_m0, T_init=T0, p=p0, mesh=mesh, u_gauss_init=u_g0, tol=1e-9
    )
    assert bool(sol_ab.converged)

    mults_ab_exp, _ = compute_floquet_multipliers(sol_ab, p=p0, method="condensation")
    mults_ab_qr, _ = compute_floquet_multipliers(sol_ab, p=p0, method="periodic_schur")
    assert jnp.allclose(
        _sort_complex_eigs(mults_ab_qr), _sort_complex_eigs(mults_ab_exp), atol=1e-10
    )

    # 3. 4D system with complex conjugate Floquet multipliers (exercises 2x2 Schur blocks)
    prob_4d = CollocationProblem(f_3d_complex_floquet, dim=4, num_intervals=20, num_gauss_points=4)
    t4 = jnp.linspace(0.0, 2.0 * jnp.pi, 20, endpoint=False)
    u4_init = jnp.stack([jnp.cos(t4), jnp.sin(t4), jnp.zeros_like(t4), jnp.zeros_like(t4)], axis=1)
    sol_4d = solve_periodic_orbit(prob_4d, u4_init, T_init=2.0 * jnp.pi, p=0.0, tol=1e-10)
    assert bool(sol_4d.converged)

    mults_4d_exp, _ = compute_floquet_multipliers(sol_4d, p=0.0, method="condensation")
    mults_4d_qr, schur_4d = compute_floquet_multipliers(sol_4d, p=0.0, method="periodic_qr")
    assert schur_4d.shape == (20, 4, 4)
    assert jnp.allclose(
        _sort_complex_eigs(mults_4d_qr), _sort_complex_eigs(mults_4d_exp), atol=1e-10
    )
    # Confirm complex conjugate pair is genuinely present
    assert jnp.any(jnp.abs(jnp.imag(mults_4d_qr)) > 0.05)


def test_periodic_schur_extreme_stiff_synthetic_factors():
    """Verify overflow/underflow-free log-magnitudes on synthetic 10^{+200} / 10^{-200} products."""
    N = 100
    key = jax.random.PRNGKey(1234)

    # Case 1: 2x2 with multipliers 10^{+200} and 10^{-200} (explicit product overflows/loses small)
    Q2 = _make_random_orthogonal(key, N, 2)
    Q2_next = jnp.roll(Q2, -1, axis=0)
    # Each factor has diagonal (10^2, 10^-2) with random upper-triangular shear
    D2 = jnp.zeros((N, 2, 2), dtype=jnp.float64)
    D2 = D2.at[:, 0, 0].set(100.0)
    D2 = D2.at[:, 1, 1].set(0.01)
    D2 = D2.at[:, 0, 1].set(0.7)
    factors_2 = jax.vmap(lambda qn, dk, qk: qn @ dk @ qk.T)(Q2_next, D2, Q2)

    mults_2, log_abs_2, phases_2 = periodic_schur_eigenvalues(factors_2)
    expected_log_2 = jnp.array([200.0 * jnp.log(10.0), -200.0 * jnp.log(10.0)], dtype=jnp.float64)
    assert jnp.allclose(log_abs_2, expected_log_2, rtol=1e-12, atol=1e-11)
    assert jnp.allclose(phases_2, jnp.zeros(2), atol=1e-12)
    assert jnp.all(jnp.isfinite(mults_2))

    # Case 2: 3x3 with multipliers 10^{+200}, 1.0, 10^{-200} (explicit product fails completely)
    Q3 = _make_random_orthogonal(jax.random.PRNGKey(5678), N, 3)
    Q3_next = jnp.roll(Q3, -1, axis=0)
    D3 = jnp.zeros((N, 3, 3), dtype=jnp.float64)
    D3 = D3.at[:, 0, 0].set(100.0)
    D3 = D3.at[:, 1, 1].set(1.0)
    D3 = D3.at[:, 2, 2].set(0.01)
    D3 = D3.at[:, 0, 1].set(0.5)
    D3 = D3.at[:, 0, 2].set(-0.3)
    D3 = D3.at[:, 1, 2].set(0.4)
    factors_3 = jax.vmap(lambda qn, dk, qk: qn @ dk @ qk.T)(Q3_next, D3, Q3)

    # Confirm explicit product loses the 10^{-200} eigenvalue
    M3_explicit, _ = jax.lax.scan(lambda acc, mk: (mk @ acc, None), jnp.eye(3), factors_3)
    eigs_explicit = jnp.sort(jnp.abs(jnp.linalg.eigvals(M3_explicit)))
    assert not jnp.isfinite(eigs_explicit[-1]) or float(eigs_explicit[0]) > 1e-100

    _, log_abs_3, _ = periodic_schur_eigenvalues(factors_3)
    expected_log_3 = jnp.array(
        [200.0 * jnp.log(10.0), 0.0, -200.0 * jnp.log(10.0)], dtype=jnp.float64
    )
    assert jnp.allclose(log_abs_3, expected_log_3, rtol=1e-12, atol=1e-10)

    # Case 3: 4x4 with 10^{+400} (beyond float64 max!), 1.0, and complex pair 10^{-400} * e^{+-i*0.8}
    N4 = 200
    Q4 = _make_random_orthogonal(jax.random.PRNGKey(9999), N4, 4)
    Q4_next = jnp.roll(Q4, -1, axis=0)
    theta_step = 0.8 / N4
    c_th, s_th = jnp.cos(theta_step), jnp.sin(theta_step)
    scale_small = 1e-2  # (10^-2)^200 = 10^-400
    D4 = jnp.zeros((N4, 4, 4), dtype=jnp.float64)
    D4 = D4.at[:, 0, 0].set(1e2)  # (10^2)^200 = 10^+400
    D4 = D4.at[:, 1, 1].set(1.0)
    D4 = D4.at[:, 2, 2].set(scale_small * c_th)
    D4 = D4.at[:, 2, 3].set(-scale_small * s_th)
    D4 = D4.at[:, 3, 2].set(scale_small * s_th)
    D4 = D4.at[:, 3, 3].set(scale_small * c_th)
    D4 = D4.at[:, 0, 2].set(0.2)
    D4 = D4.at[:, 1, 3].set(-0.3)
    factors_4 = jax.vmap(lambda qn, dk, qk: qn @ dk @ qk.T)(Q4_next, D4, Q4)

    _, log_abs_4, phases_4 = periodic_schur_eigenvalues(factors_4)
    expected_log_4 = jnp.array(
        [400.0 * jnp.log(10.0), 0.0, -400.0 * jnp.log(10.0), -400.0 * jnp.log(10.0)],
        dtype=jnp.float64,
    )
    assert jnp.allclose(log_abs_4, expected_log_4, rtol=1e-11, atol=1e-10)
    assert jnp.allclose(jnp.sort(jnp.abs(phases_4[2:])), jnp.array([0.8, 0.8]), atol=1e-10)


def test_periodic_schur_edge_cases_and_equal_moduli():
    """Test N=1, n=1, period-doubling (-1), Neimark-Sacker (equal moduli |mu|=1), cyclic matrix."""
    # n = 1
    f1 = jnp.array([[[2.0]], [[-3.0]], [[0.5]]], dtype=jnp.float64)
    m1, la1, ph1 = periodic_schur_eigenvalues(f1)
    assert jnp.allclose(m1, jnp.array([-3.0 + 0.0j]), atol=1e-13)
    assert jnp.allclose(la1, jnp.array([jnp.log(3.0)]), atol=1e-13)
    assert jnp.allclose(ph1, jnp.array([jnp.pi]), atol=1e-13)

    # N = 1 with complex conjugate eigenvalues
    th = 1.2
    M_single = jnp.array(
        [
            [1.0, 0.2, -0.1],
            [0.0, 0.8 * jnp.cos(th), -0.8 * jnp.sin(th)],
            [0.0, 0.8 * jnp.sin(th), 0.8 * jnp.cos(th)],
        ],
        dtype=jnp.float64,
    )
    m_s, _, _ = periodic_schur_eigenvalues(M_single[None, :, :])
    assert jnp.allclose(
        _sort_complex_eigs(m_s), _sort_complex_eigs(jnp.linalg.eigvals(M_single)), atol=1e-12
    )

    # 3D Neimark-Sacker critical point: all 3 eigenvalues have |mu| = 1 (1 and e^{+-i*1.1})
    N = 10
    Q = _make_random_orthogonal(jax.random.PRNGKey(77), N, 3)
    Q_next = jnp.roll(Q, -1, axis=0)
    dth = 1.1 / N
    D_ns = jnp.zeros((N, 3, 3), dtype=jnp.float64)
    D_ns = D_ns.at[:, 0, 0].set(1.0)
    D_ns = D_ns.at[:, 1, 1].set(jnp.cos(dth))
    D_ns = D_ns.at[:, 1, 2].set(-jnp.sin(dth))
    D_ns = D_ns.at[:, 2, 1].set(jnp.sin(dth))
    D_ns = D_ns.at[:, 2, 2].set(jnp.cos(dth))
    factors_ns = jax.vmap(lambda qn, dk, qk: qn @ dk @ qk.T)(Q_next, D_ns, Q)

    mults_ns, log_abs_ns, _ = periodic_schur_eigenvalues(factors_ns)
    expected_ns = jnp.array([1.0 + 0.0j, jnp.exp(1.1j), jnp.exp(-1.1j)], dtype=jnp.complex128)
    assert jnp.allclose(log_abs_ns, jnp.zeros(3), atol=1e-12)
    assert jnp.allclose(_sort_complex_eigs(mults_ns), _sort_complex_eigs(expected_ns), atol=1e-11)

    # Cyclic 3x3 permutation matrix (tests exceptional shift breaking symmetry)
    P_cyc = jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    m_cyc, _, _ = periodic_schur_eigenvalues(P_cyc[None, :, :])
    assert jnp.allclose(
        _sort_complex_eigs(m_cyc), _sort_complex_eigs(jnp.linalg.eigvals(P_cyc)), atol=1e-11
    )


def test_stiff_van_der_pol_floquet_exponent_matches_liouville():
    """Verify periodic_qr recovers the exact ~1e-52 multiplier of stiff Van der Pol (mu = 6)."""
    N, m = 80, 4
    prob = CollocationProblem(f_vdp, dim=2, num_intervals=N, num_gauss_points=m)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, N, endpoint=False)
    u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)
    sol0 = solve_periodic_orbit(prob, u_init, T_init=6.66, p=1.0, tol=1e-10)
    assert bool(sol0.converged)

    branch = continue_periodic_orbit(
        prob,
        initial_orbit=sol0,
        p_start=1.0,
        p_end=6.0,
        dp=0.5,
        dp_max=1.0,
        max_steps=20,
        param_mode="natural",
        tol=1e-9,
    )
    sol_stiff = branch[-1]
    assert bool(sol_stiff.converged)
    mu_val = float(sol_stiff.p)  # type: ignore[arg-type]
    assert abs(mu_val - 6.0) < 1e-9

    # Exact Liouville integral: sigma_1 + sigma_2 = \int_0^1 div(f(u(s))) ds = \int_0^1 mu*(1 - x^2) ds
    h = sol_stiff.mesh[1:] - sol_stiff.mesh[:-1]
    x_g = sol_stiff.u_gauss[:, :, 0]
    div_avg = float(jnp.sum(h[:, None] * prob.weights[None, :] * (mu_val * (1.0 - x_g * x_g))))

    exps_qr, mults_qr, _ = compute_floquet_exponents(sol_stiff, p=mu_val, method="periodic_qr")
    exps_cond, _, _ = compute_floquet_exponents(sol_stiff, p=mu_val, method="condensation")

    # Trivial exponent ~ 0, nontrivial multiplier ~ 1.56e-52, matching Liouville to ~1e-3
    assert abs(float(exps_qr[0])) < 1e-3
    assert float(jnp.abs(mults_qr[1])) < 1e-45
    assert abs(float(exps_qr[1]) - div_avg) < 1e-3, (
        f"Stiff VdP Floquet exponent {float(exps_qr[1])} disagrees with Liouville {div_avg}"
    )
    # Explicit condensation loses accuracy completely on the ~1e-52 multiplier
    assert abs(float(exps_cond[1]) - div_avg) > 1.0


def test_evaluate_floquet_diagnostics_from_factors_and_jit():
    """Verify diagnostics from factors match matrix diagnostics and all core functions JIT cleanly."""
    # 1. Stability, PD, and NS parity with evaluate_floquet_diagnostics
    th = 1.0
    c, s = float(jnp.cos(th)), float(jnp.sin(th))
    M_ns = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 1.1 * c, -1.1 * s], [0.0, 1.1 * s, 1.1 * c]],
        dtype=jnp.float64,
    )
    m_mat, tf_mat, st_mat = evaluate_floquet_diagnostics(M_ns, tangent_p=jnp.array(0.5))
    m_fac, tf_fac, st_fac = evaluate_floquet_diagnostics_from_factors(
        M_ns[None, :, :], tangent_p=jnp.array(0.5)
    )
    assert bool(st_fac) == bool(st_mat)
    assert jnp.allclose(_sort_complex_eigs(m_fac), _sort_complex_eigs(m_mat), atol=1e-12)
    assert abs(float(tf_fac["PD"]) - float(tf_mat["PD"])) < 1e-11
    assert abs(float(tf_fac["NS"]) - float(tf_mat["NS"])) < 1e-11
    assert abs(float(tf_fac["LPC"]) - float(tf_mat["LPC"])) < 1e-12

    # 2. JIT compilation check for all core periodic Schur functions
    prob = CollocationProblem(f_vdp, dim=2, num_intervals=15, num_gauss_points=3)
    t_grid = jnp.linspace(0.0, 2.0 * jnp.pi, 15, endpoint=False)
    u_init = jnp.stack([2.0 * jnp.cos(t_grid), -2.0 * jnp.sin(t_grid)], axis=1)
    sol = solve_periodic_orbit(prob, u_init, T_init=2.0 * jnp.pi, p=0.2, tol=1e-9)

    jit_factors = jax.jit(monodromy_factors_condensation)
    jit_schur = jax.jit(periodic_real_schur)
    jit_eigs = jax.jit(periodic_schur_eigenvalues)
    jit_diag = jax.jit(evaluate_floquet_diagnostics_from_factors)

    factors = jit_factors(prob, sol.mesh, sol.u_gauss, sol.period, jnp.array(0.2))
    M_raw = monodromy_condensation_raw(prob, sol.mesh, sol.u_gauss, sol.period, 0.2)
    M_prod, _ = jax.lax.scan(lambda acc, mk: (mk @ acc, None), jnp.eye(2), factors)
    assert jnp.allclose(M_prod, M_raw, atol=1e-13)

    schur = jit_schur(factors)
    assert schur.shape == factors.shape
    mults_jit, log_abs_jit, _ = jit_eigs(factors)
    assert jnp.all(jnp.isfinite(log_abs_jit))
    assert jnp.allclose(
        _sort_complex_eigs(mults_jit), _sort_complex_eigs(jnp.linalg.eigvals(M_raw)), atol=1e-11
    )

    _, _, is_st = jit_diag(factors)
    assert bool(is_st)

    # 3. Unknown method raises ValueError
    with pytest.raises(ValueError, match="Unknown method"):
        compute_floquet_multipliers(sol, p=0.2, method="invalid")
    with pytest.raises(ValueError, match="Unknown method"):
        compute_floquet_exponents(sol, p=0.2, method="invalid")
