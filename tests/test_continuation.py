import jax.numpy as jnp

from bifurx.bifurcations.switch import switch_branch
from bifurx.continuation.arclength import continuation
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
    from bifurx.bifurcations.test_funcs import refine_bifurcation_point

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
    from bifurx.bifurcations.test_funcs import refine_bifurcation_point

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
