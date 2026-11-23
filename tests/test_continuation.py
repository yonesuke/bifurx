import jax
import jax.numpy as jnp
import pytest

from bifurx.bifurcations import (
    compute_first_lyapunov_coefficient,
    refine_bifurcation_point,
    switch_branch,
)
from bifurx.continuation import continuation
from bifurx.problem import BifurcationProblem


def test_pitchfork_bifurcation_and_branch_switching():
    """Test pitchfork bifurcation F(u, p) = p * u - u^3.

    1. Trivial branch u = 0 passes through BP at p = 0.
    2. BP is detected near p = 0, u = 0.
    3. Branch switching leads to stable upper branch u = +sqrt(p) and lower branch u = -sqrt(p).
    """

    def f_pitchfork(u, p):
        x = u[0]
        return jnp.array([p * x - x**3])

    prob = BifurcationProblem(f_pitchfork, u0=[0.0], p0=-0.5)

    # 1. Track trivial branch
    res_trivial = continuation(
        problem=prob,
        ds=0.04,
        max_steps=35,
        param_bounds=(-0.6, 0.6),
        branch_name="Trivial",
    )

    # Detect BP
    bps = res_trivial.get_bifurcations("BP")
    assert len(bps) >= 1, "Expected at least one branch point (BP) detected"

    bp0 = bps[0]
    assert abs(bp0.p) < 1e-2, f"BP parameter should be close to 0.0, got {bp0.p}"
    assert abs(float(bp0.u[0])) < 1e-2, f"BP state should be close to 0.0, got {bp0.u[0]}"

    # Stability of trivial branch: stable for p < 0, unstable for p > 0
    p_arr = res_trivial.p
    st_arr = res_trivial.stability
    # Check stable before BP
    stable_before = st_arr[p_arr < -0.05]
    assert bool(jnp.all(stable_before)), "Trivial branch should be stable for p < 0"
    # Check unstable after BP
    unstable_after = st_arr[p_arr > 0.05]
    assert bool(jnp.all(~unstable_after)), "Trivial branch should be unstable for p > 0"

    # 2. Switch to Upper branch (+u)
    u_up, p_up, tau_up = switch_branch(
        prob, bp0.u, bp0.p, bp0.tangent, ds=0.04, branch_direction=1.0
    )
    res_up = continuation(
        problem=prob,
        u0=u_up,
        p0=p_up,
        initial_tangent=tau_up,
        ds=0.03,
        max_steps=30,
        branch_name="Upper",
    )
    assert res_up.num_points > 5
    # Verify u^2 approx p along upper branch
    p_up_arr = res_up.p
    u_up_arr = res_up.u[:, 0]
    assert bool(jnp.all(u_up_arr > 0)), "Upper branch should have positive u"
    # Check u^2 close to p
    assert jnp.allclose(u_up_arr**2, p_up_arr, atol=1e-3)

    # 3. Switch to Lower branch (-u)
    u_dn, p_dn, tau_dn = switch_branch(
        prob, bp0.u, bp0.p, bp0.tangent, ds=0.04, branch_direction=-1.0
    )
    res_dn = continuation(
        problem=prob,
        u0=u_dn,
        p0=p_dn,
        initial_tangent=tau_dn,
        ds=0.03,
        max_steps=30,
        branch_name="Lower",
    )
    assert res_dn.num_points > 5
    p_dn_arr = res_dn.p
    u_dn_arr = res_dn.u[:, 0]
    assert bool(jnp.all(u_dn_arr < 0)), "Lower branch should have negative u"
    assert jnp.allclose(u_dn_arr**2, p_dn_arr, atol=1e-3)


def test_fold_bifurcation():
    """Test fold / saddle-node bifurcation F(u, p) = p - u^2.

    Equilibrium curve is p = u^2.
    Continuing from u=1.0, p=1.0 with direction=-1.0 passes through fold at (u=0, p=0).
    Tangent parameter component dot{p} crosses zero.
    """

    def f_fold(u, p):
        return jnp.array([p - u[0] ** 2])

    prob = BifurcationProblem(f_fold, u0=[1.0], p0=1.0)

    # Continue past the fold
    res = continuation(prob, ds=0.05, max_steps=25, direction=-1.0, branch_name="FoldCurve")

    lps = res.get_bifurcations("LP")
    assert len(lps) >= 1, "Expected Limit Point (LP) detected"

    lp0 = lps[0]
    assert abs(float(lp0.p)) < 1e-4, f"Fold p should be close to 0.0, got {lp0.p}"
    assert abs(float(lp0.u[0])) < 1e-3, f"Fold u should be close to 0.0, got {lp0.u[0]}"

    # Verify that p values turned around (min p is near 0, then increases again)
    p_arr = res.p
    min_p_idx = int(jnp.argmin(p_arr))
    assert 0 < min_p_idx < len(p_arr) - 1, "Curve should turn around in p"
    assert float(p_arr[min_p_idx]) < 0.05, (
        f"Sampled p near fold should be small, got {p_arr[min_p_idx]}"
    )
    assert abs(float(lp0.p)) < 1e-6, f"Refined fold p should be 0.0, got {lp0.p}"

    # Check stability: u > 0 is stable (df/du = -2u < 0), u < 0 is unstable (df/du = -2u > 0)
    u_arr = res.u[:, 0]
    st_arr = res.stability
    assert bool(jnp.all(st_arr[u_arr > 0.1])), "Upper arm (u > 0) should be stable"
    assert bool(jnp.all(~st_arr[u_arr < -0.1])), "Lower arm (u < 0) should be unstable"


def test_param_bounds_precedence():
    """Test precedence of param_bounds argument over problem.param_bounds."""
    fn = lambda u, p: jnp.array([u[0] ** 2 - p])
    # problem default bounds: (0.0, 5.0)
    prob = BifurcationProblem(fn=fn, u0=[1.0], p0=1.0, param_bounds=(0.0, 5.0))

    # continuation with default bounds reaches past 4.0
    res_default = continuation(prob, ds=0.1, max_steps=50)
    assert float(jnp.max(res_default.p)) > 4.0

    # continuation with override bounds terminates around 2.0
    res_override = continuation(prob, ds=0.1, param_bounds=(0.5, 2.0), max_steps=50)
    assert float(jnp.max(res_override.p)) < 3.0
    assert len(res_override.p) < len(res_default.p)


def test_continuation_stop_reason():
    """Test stop_reason reporting on ContinuationResult."""

    def fn(u, p):
        return jnp.array([u[0] ** 2 - p])

    prob = BifurcationProblem(fn=fn, u0=[1.0], p0=1.0)

    # 1. max_steps
    res_max = continuation(prob, ds=0.01, max_steps=5, param_bounds=(0.0, 10.0))
    assert res_max.stop_reason == "max_steps"

    # 2. bounds
    res_bounds = continuation(prob, ds=0.1, max_steps=50, param_bounds=(0.5, 2.0))
    assert res_bounds.stop_reason == "bounds"

    # 3. corrector_failed
    def fn_fail(u, p):
        return jnp.array([jnp.sqrt(jnp.maximum(p - u[0], -1.0))])

    prob_fail = BifurcationProblem(fn=fn_fail, u0=[0.0], p0=0.0)
    res_fail = continuation(prob_fail, ds=1.0, max_steps=10, adaptive_step=False)
    assert res_fail.stop_reason == "corrector_failed"


def test_user_supplied_jacobians():
    """Test custom jac_u and jac_p support in BifurcationProblem."""

    def fn(u, p):
        return jnp.array([u[0] ** 2 - p])

    def custom_jac_u(u, p):
        return jnp.array([[2.0 * u[0]]])

    def custom_jac_p(u, p):
        return jnp.array([-1.0])

    prob = BifurcationProblem(
        fn=fn,
        u0=[1.0],
        p0=1.0,
        jac_u=custom_jac_u,
        jac_p=custom_jac_p,
    )

    # Test direct jacobian evaluation
    ju = prob.jacobian_u(jnp.array([3.0]), 1.0)
    jp = prob.jacobian_p(jnp.array([3.0]), 1.0)
    assert jnp.allclose(ju, jnp.array([[6.0]]))
    assert jnp.allclose(jp, jnp.array([-1.0]))

    # Test continuation
    res = continuation(prob, ds=0.1, max_steps=15)
    assert res.num_points > 5
    assert jnp.allclose(res.u[:, 0] ** 2, res.p, atol=1e-3)


def test_refinement_nan_validation():
    """Test that bifurcation refinement never returns or stores NaN coordinates (#3)."""

    def f_singular(u, p):
        # Function that becomes NaN/singular near certain values
        return jnp.array([jnp.where(u[0] > 0.5, jnp.nan, u[0] ** 2 - p)])

    prob = BifurcationProblem(fn=f_singular, u0=[0.0], p0=0.0)

    u_a = jnp.array([0.0])
    p_a = jnp.array(0.0)
    tau_a = jnp.array([1.0, 0.0])
    u_b = jnp.array([1.0])  # will produce NaN in f_singular
    p_b = jnp.array(1.0)
    tau_b = jnp.array([0.0, 1.0])

    u_ref, p_ref = refine_bifurcation_point(prob, u_a, p_a, tau_a, u_b, p_b, tau_b, bif_type="LP")
    assert bool(jnp.all(jnp.isfinite(u_ref))), f"u_ref should be finite, got {u_ref}"
    assert bool(jnp.isfinite(p_ref)), f"p_ref should be finite, got {p_ref}"

    # Also verify that continuation never stores NaN coordinates in bifurcation_points
    def f_fold(u, p):
        return jnp.array([p - u[0] ** 2])

    prob_fold = BifurcationProblem(fn=f_fold, u0=[1.0], p0=1.0)
    res = continuation(prob_fold, ds=0.05, max_steps=40, direction=-1.0)
    for bp in res.bifurcation_points:
        assert not jnp.isnan(bp.p), f"Found NaN parameter in bifurcation point: {bp}"
        assert bool(jnp.all(jnp.isfinite(bp.u))), (
            f"Found non-finite state in bifurcation point: {bp}"
        )


def test_lp_refinement_bracket_selection():
    """Test that LP refinement selects the fold closest to the bracket (#12)."""

    # F(u, p) = (u - 2)^2 - p  -> Fold at u = 2, p = 0
    # Secondary branch/fold could attract Newton if seeded from u_b
    def f_two_folds(u, p):
        # Has two folds: one at u=2.0 (p=0.0) and one at u=-2.0 (p=0.0)
        return jnp.array([(u[0] ** 2 - 4.0) ** 2 - p])

    prob = BifurcationProblem(fn=f_two_folds, u0=[2.1], p0=0.1)

    # Bracket around u = 2 fold:
    # u_a is at 1.9, p_a = (1.9^2 - 4)^2 = 0.1521
    # Suppose u_b was stepped past the turn towards 2.1, p_b = 0.1681
    u_a = jnp.array([1.9])
    p_a = jnp.array(0.1521)
    tau_a = jnp.array([0.9, -0.1])
    u_b = jnp.array([2.1])
    p_b = jnp.array(0.1681)
    tau_b = jnp.array([0.9, 0.1])

    u_ref, p_ref = refine_bifurcation_point(prob, u_a, p_a, tau_a, u_b, p_b, tau_b, bif_type="LP")
    assert jnp.allclose(u_ref, jnp.array([2.0]), atol=1e-3)
    assert jnp.allclose(p_ref, 0.0, atol=1e-3)


def test_degenerate_multiplicity_two_zero_eigenvalue():
    """Test detection of branch point with even multiplicity zero eigenvalue (#13).

    det(Ju) = p^2 >= 0 never changes sign, but tracking real eigenvalue closest to 0
    successfully detects the bifurcation at p = 0.
    """

    def f_mult2(u, p):
        return jnp.array([p * u[0] - u[0] ** 3, p * u[1] - u[1] ** 3])

    prob = BifurcationProblem(fn=f_mult2, u0=[0.0, 0.0], p0=-0.4)
    res = continuation(prob, ds=0.04, max_steps=25, param_bounds=(-0.5, 0.5))

    bps = res.get_bifurcations("BP")
    assert len(bps) == 1, f"Expected exactly 1 BP detected, got {len(bps)}: {bps}"
    assert abs(bps[0].p) < 1e-2, f"BP expected near p=0, got {bps[0].p}"


def test_kuramoto_4d_degenerate_branch_point():
    """Test detection of even-multiplicity (multiplicity 4) branch point on symmetric manifold (#13, #18).

    det(Ju) = p^4 >= 0 touches zero quartically without changing sign.
    Canonical bordered determinant + instability index transition cleanly pinpoints the BP at p = 0.
    """

    def f_kuramoto_4d(u, p):
        r2 = jnp.sum(u**2)
        return p * u - r2 * u

    prob = BifurcationProblem(fn=f_kuramoto_4d, u0=[0.0, 0.0, 0.0, 0.0], p0=-0.4)
    res = continuation(prob, ds=0.04, max_steps=25, param_bounds=(-0.5, 0.5))

    bps = res.get_bifurcations("BP")
    assert len(bps) == 1, f"Expected exactly 1 BP detected, got {len(bps)}: {bps}"
    assert abs(bps[0].p) < 1e-2, f"BP expected near p=0, got {bps[0].p}"
    assert float(jnp.linalg.norm(bps[0].u)) < 1e-3, f"BP state expected near 0, got {bps[0].u}"


def test_persistently_singular_jacobian_no_spurious_bp():
    """Test that branch with persistent kernel does not flood spurious BPs (#2)."""

    # System on R^2 where second coordinate has persistent zero row:
    # J_u = [[1, 0], [0, 0]] has determinant 0 everywhere along branch
    def f_singular_branch(u, p):
        return jnp.array([u[0] - p, 0.0 * u[1]])

    prob = BifurcationProblem(fn=f_singular_branch, u0=[0.1, 0.0], p0=0.1)
    res = continuation(prob, ds=0.05, max_steps=20, param_bounds=(0.0, 1.0))

    # Should NOT have any spurious BPs
    bps = res.get_bifurcations("BP")
    assert len(bps) == 0, f"Expected 0 BPs on persistently singular branch, got {len(bps)}"


def test_branch_point_no_spurious_discontinuity_jumps():
    """Test that eigenvalue swaps and real-to-complex transitions do not produce spurious BPs (#16)."""
    # Mechanism 1: Equal-magnitude opposite-sign real eigenvalues swap (det = -1 constant)
    s, p0_val = 1.0, 1.0

    def a_mat(p):
        return jnp.array([[(p0_val - p) / 2.0 + s, 0.0], [0.0, (p0_val - p) / 2.0 - s]])

    prob1 = BifurcationProblem(
        fn=lambda u, p: a_mat(p) @ u,
        u0=[0.0, 0.0],
        p0=0.0,
        jac_u=lambda u, p: a_mat(p),
    )
    res1 = continuation(
        prob1,
        u0=jnp.zeros(2),
        p0=0.0,
        direction=1.0,
        ds=0.1,
        max_steps=40,
        param_bounds=(0.0, 2.0),
        adaptive_step=False,
    )
    bps1 = res1.get_bifurcations("BP")
    assert len(bps1) == 0, f"Expected 0 BPs for constant det=-1 system, got {bps1}"

    # Mechanism 2: Real eigenvalues complexify (Re = -1/2 for p > 0.25, no zero eigenvalue)
    def f_complexify(u, p):
        return jnp.array([u[1], -p * u[0] - u[1]])

    prob2 = BifurcationProblem(
        fn=f_complexify,
        u0=[0.0, 0.0],
        p0=0.1,
        jac_u=lambda u, p: jnp.array([[0.0, 1.0], [-p, -1.0]]),
    )
    res2 = continuation(
        prob2,
        u0=jnp.zeros(2),
        p0=0.1,
        direction=1.0,
        ds=0.05,
        max_steps=40,
        param_bounds=(0.1, 1.0),
        adaptive_step=False,
    )
    bps2 = res2.get_bifurcations("BP")
    assert len(bps2) == 0, f"Expected 0 BPs during complexification, got {bps2}"


def test_continuation_pade21_predictor():
    """Verify continuation with predictor='pade21' accurately detects fold and Hopf points."""

    def f_fold(u, p):
        return jnp.array([p - u[0] ** 2])

    prob_fold = BifurcationProblem(f_fold, u0=[1.0], p0=1.0)
    res_fold = continuation(
        prob_fold,
        ds=0.05,
        max_steps=30,
        direction=-1.0,
        predictor="pade21",
        branch_name="FoldPade21",
    )
    lps_fold = res_fold.get_bifurcations("LP")
    assert len(lps_fold) >= 1
    assert abs(float(lps_fold[0].p)) < 1e-6

    def f_ab(u, p1):
        u1, u2 = u[0], u[1]
        e = jnp.exp(u2)
        return jnp.array([-u1 + p1 * (1.0 - u1) * e, -u2 + p1 * 14.0 * (1.0 - u1) * e - 2.0 * u2])

    prob_ab = BifurcationProblem(f_ab, u0=[0.0, 0.0], p0=0.0)
    res_ab = continuation(
        prob_ab,
        ds=0.03,
        ds_max=0.05,
        max_steps=140,
        predictor="pade21",
        branch_name="DemoABPade21",
    )
    lps = res_ab.get_bifurcations("LP")
    hbs = res_ab.get_bifurcations("HB")
    assert len(lps) == 2
    assert abs(lps[0].p - 0.1057390) < 1e-4
    assert abs(lps[1].p - 0.08893185) < 1e-4
    assert len(hbs) >= 1
    assert abs(hbs[0].p - 0.1308998) < 1e-3


def test_hopf_direct_refinement_curved_branch_large_step():
    """Verify HB direct augmented Newton refinement recovers exact (u*, p*) on curved branches."""

    def f_curved_hopf(u: jnp.ndarray, p: float | jnp.ndarray) -> jnp.ndarray:
        x, y, z = u[0], u[1], u[2]
        e1 = x - p**2
        e3 = z - p
        return jnp.array(
            [
                (p - 1.0) * e1 - 2.0 * y - e1**2,
                2.0 * e1 + (p - 1.0) * y + e1 * e3,
                -e3 + e1**2 + y**2,
            ]
        )

    prob = BifurcationProblem(f_curved_hopf, u0=[0.36, 0.0, 0.6], p0=0.6)
    p_a, p_b = 0.6, 1.3
    u_a = jnp.array([p_a**2, 0.0, p_a], dtype=jnp.float64)
    u_b = jnp.array([p_b**2, 0.0, p_b], dtype=jnp.float64)
    tau_a = jnp.array([2.0 * p_a, 0.0, 1.0, 1.0], dtype=jnp.float64)
    tau_a = tau_a / jnp.linalg.norm(tau_a)
    tau_b = jnp.array([2.0 * p_b, 0.0, 1.0, 1.0], dtype=jnp.float64)
    tau_b = tau_b / jnp.linalg.norm(tau_b)

    u_ref, p_ref = refine_bifurcation_point(prob, u_a, p_a, tau_a, u_b, p_b, tau_b, bif_type="HB")
    assert abs(float(p_ref) - 1.0) < 1e-10
    assert jnp.allclose(u_ref, jnp.array([1.0, 0.0, 1.0]), atol=1e-10)


def test_bp_on_curve_refinement_curved_branch():
    """Verify BP on-curve projection refines curved branch points accurately on large brackets."""

    def f_curved_bp(u: jnp.ndarray, p: float | jnp.ndarray) -> jnp.ndarray:
        x, y = u[0], u[1]
        return jnp.array([x - p**2 - y**2, y * (x + p - 2.0 - y)])

    prob = BifurcationProblem(f_curved_bp, u0=[0.36, 0.0], p0=0.6)
    p_a, p_b = 0.6, 1.25
    u_a = jnp.array([p_a**2, 0.0], dtype=jnp.float64)
    u_b = jnp.array([p_b**2, 0.0], dtype=jnp.float64)
    tau_a = jnp.array([2.0 * p_a, 0.0, 1.0], dtype=jnp.float64)
    tau_a = tau_a / jnp.linalg.norm(tau_a)
    tau_b = jnp.array([2.0 * p_b, 0.0, 1.0], dtype=jnp.float64)
    tau_b = tau_b / jnp.linalg.norm(tau_b)

    u_ref, p_ref = refine_bifurcation_point(prob, u_a, p_a, tau_a, u_b, p_b, tau_b, bif_type="BP")
    assert abs(float(p_ref) - 1.0) < 1e-8
    assert jnp.allclose(u_ref, jnp.array([1.0, 0.0]), atol=1e-8)


def test_first_lyapunov_coefficient_analytical_and_jit():
    """Verify Kuznetsov first Lyapunov coefficient l_1 against analytical cases and jax.jit."""

    # 1. Supercritical Stuart-Landau (sigma = -1, omega = 2 -> l_1 = 2*sigma/omega = -1.0)
    def sl_super(u: jnp.ndarray, p: float | jnp.ndarray) -> jnp.ndarray:
        x, y = u[0], u[1]
        r2 = x * x + y * y
        return jnp.array([p * x - 2.0 * y - x * r2, 2.0 * x + p * y - y * r2])

    l1_super = compute_first_lyapunov_coefficient(sl_super, jnp.zeros(2), 0.0)
    assert float(l1_super) == pytest.approx(-1.0, rel=1e-10)

    # 2. Subcritical Stuart-Landau (sigma = +1.5, omega = 1 -> l_1 = 3.0)
    def sl_sub(u: jnp.ndarray, p: float | jnp.ndarray) -> jnp.ndarray:
        x, y = u[0], u[1]
        r2 = x * x + y * y
        return jnp.array([p * x - y + 1.5 * x * r2, x + p * y + 1.5 * y * r2])

    l1_sub = compute_first_lyapunov_coefficient(sl_sub, jnp.zeros(2), 0.0)
    assert float(l1_sub) == pytest.approx(3.0, rel=1e-10)

    # 3. 3D quadratic center-manifold coupling (tests w_11 = A^{-1} B(q, qbar) -> l_1 = 2*gamma = -1.0)
    def f_3d_quad(u: jnp.ndarray, p: float | jnp.ndarray) -> jnp.ndarray:
        x, y, z = u[0], u[1], u[2]
        return jnp.array(
            [
                p * x - y - 0.5 * x * z,
                x + p * y - 0.5 * y * z,
                -z + x * x + y * y,
            ]
        )

    l1_3d = compute_first_lyapunov_coefficient(f_3d_quad, jnp.zeros(3), 0.0)
    assert float(l1_3d) == pytest.approx(-1.0, rel=1e-10)

    # 4. 2D quadratic oscillator (tests both w_11 and w_20 = (2i*omega*I - A)^{-1} B(q, q))
    def f_quad_2d(u: jnp.ndarray, _p: float | jnp.ndarray) -> jnp.ndarray:
        x, y = u[0], u[1]
        return jnp.array([y + x * x, -x - x * x])

    l1_q2 = compute_first_lyapunov_coefficient(f_quad_2d, jnp.zeros(2), 0.0)
    assert float(l1_q2) == pytest.approx(-0.5, rel=1e-10)

    # 5. Brusselator (a=1, b_H=2 -> supercritical l_1 < 0) and jax.jit consistency
    def f_bruss(u: jnp.ndarray, b: float | jnp.ndarray) -> jnp.ndarray:
        a = 1.0
        x, y = u[0], u[1]
        return jnp.array([a - (b + 1.0) * x + x * x * y, b * x - x * x * y])

    prob_bruss = BifurcationProblem(f_bruss, u0=[1.0, 2.0], p0=2.0)
    l1_eager = compute_first_lyapunov_coefficient(prob_bruss, jnp.array([1.0, 2.0]), 2.0)
    l1_jit = jax.jit(lambda u_h, p_h: compute_first_lyapunov_coefficient(prob_bruss, u_h, p_h))(
        jnp.array([1.0, 2.0]), jnp.array(2.0)
    )
    assert bool(jnp.isfinite(l1_eager))
    assert float(l1_eager) < 0.0
    assert float(l1_jit) == pytest.approx(float(l1_eager), rel=1e-12)

    # 6. AB reaction continuation attaches first Lyapunov metadata on refined HB
    def f_ab(u: jnp.ndarray, p1: float | jnp.ndarray) -> jnp.ndarray:
        u1, u2 = u[0], u[1]
        e = jnp.exp(u2)
        return jnp.array([-u1 + p1 * (1.0 - u1) * e, -u2 + p1 * 14.0 * (1.0 - u1) * e - 2.0 * u2])

    res_ab = continuation(
        BifurcationProblem(f_ab, u0=[0.0, 0.0], p0=0.0), ds=0.03, ds_max=0.04, max_steps=140
    )
    hbs = res_ab.get_bifurcations("HB")
    assert len(hbs) >= 1
    assert abs(hbs[0].p - 0.1308998) < 1e-6
    assert "first_lyapunov" in hbs[0].details
    assert hbs[0].details["first_lyapunov"] < 0.0
    assert hbs[0].details["hopf_type"] == "supercritical"
