import jax.numpy as jnp
import numpy as np

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
    p_arr = np.asarray(res_trivial.p)
    st_arr = np.asarray(res_trivial.stability)
    # Check stable before BP
    stable_before = st_arr[p_arr < -0.05]
    assert np.all(stable_before), "Trivial branch should be stable for p < 0"
    # Check unstable after BP
    unstable_after = st_arr[p_arr > 0.05]
    assert np.all(~unstable_after), "Trivial branch should be unstable for p > 0"

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
    p_up_arr = np.asarray(res_up.p)
    u_up_arr = np.asarray(res_up.u)[:, 0]
    assert np.all(u_up_arr > 0), "Upper branch should have positive u"
    # Check u^2 close to p
    assert np.allclose(u_up_arr**2, p_up_arr, atol=1e-3)

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
    p_dn_arr = np.asarray(res_dn.p)
    u_dn_arr = np.asarray(res_dn.u)[:, 0]
    assert np.all(u_dn_arr < 0), "Lower branch should have negative u"
    assert np.allclose(u_dn_arr**2, p_dn_arr, atol=1e-3)


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
    assert abs(lp0.p) < 1e-4, f"Fold p should be close to 0.0, got {lp0.p}"
    assert abs(float(lp0.u[0])) < 1e-3, f"Fold u should be close to 0.0, got {lp0.u[0]}"

    # Verify that p values turned around (min p is near 0, then increases again)
    p_arr = np.asarray(res.p)
    min_p_idx = int(np.argmin(p_arr))
    assert 0 < min_p_idx < len(p_arr) - 1, "Curve should turn around in p"
    assert p_arr[min_p_idx] < 0.05, f"Sampled p near fold should be small, got {p_arr[min_p_idx]}"
    assert abs(lp0.p) < 1e-6, f"Refined fold p should be 0.0, got {lp0.p}"

    # Check stability: u > 0 is stable (df/du = -2u < 0), u < 0 is unstable (df/du = -2u > 0)
    u_arr = np.asarray(res.u)[:, 0]
    st_arr = np.asarray(res.stability)
    assert np.all(st_arr[u_arr > 0.1]), "Upper arm (u > 0) should be stable"
    assert np.all(~st_arr[u_arr < -0.1]), "Lower arm (u < 0) should be unstable"
