import jax.numpy as jnp
import numpy as np

from bifurx.continuation.arclength import continuation
from bifurx.plot.diagram import plot_diagram
from bifurx.problem import BifurcationProblem

P2 = 14.0
P3 = 2.0


def f_ab(u: jnp.ndarray, p1: float) -> jnp.ndarray:
    """A-B reaction model from AUTO-07P official demo 'ab'."""
    u1, u2 = u[0], u[1]
    e = jnp.exp(u2)
    f1 = -u1 + p1 * (1.0 - u1) * e
    f2 = -u2 + p1 * P2 * (1.0 - u1) * e - P3 * u2
    return jnp.array([f1, f2])


def test_ab_reaction_bifurcations_auto_benchmarks():
    """Validate bifurx on the AUTO-07P official benchmark demo 'ab' (A -> B reaction).

    Official AUTO-07P reference values (from doc/tutorial.tex):
      - LP 1: p1 = 0.1057390,  u1 = 0.3110230,  u2 = 1.451441
      - LP 2: p1 = 0.08893185, u1 = 0.6889822,  u2 = 3.215250
      - HB:   p1 = 0.1308998,  u1 = 0.8950803,  u2 = 4.177042
    """
    prob = BifurcationProblem(f_ab, u0=[0.0, 0.0], p0=0.0)

    res = continuation(
        problem=prob,
        ds=0.03,
        ds_max=0.04,
        max_steps=140,
        branch_name="DemoAB",
    )

    assert res.num_points >= 100, f"Expected at least 100 points, got {res.num_points}"

    # Verify Fold (LP) points
    lps = res.get_bifurcations("LP")
    assert len(lps) == 2, f"Expected exactly 2 LP points, got {len(lps)}"

    lp1, lp2 = lps[0], lps[1]

    # Check LP 1
    assert abs(lp1.p - 0.1057390) < 1e-4, f"LP 1 p1 mismatch: {lp1.p}"
    assert abs(float(lp1.u[0]) - 0.3110230) < 1e-3, f"LP 1 u1 mismatch: {lp1.u[0]}"
    assert abs(float(lp1.u[1]) - 1.451441) < 1e-3, f"LP 1 u2 mismatch: {lp1.u[1]}"

    # Check LP 2
    assert abs(lp2.p - 0.08893185) < 1e-4, f"LP 2 p1 mismatch: {lp2.p}"
    assert abs(float(lp2.u[0]) - 0.6889822) < 1e-3, f"LP 2 u1 mismatch: {lp2.u[0]}"
    assert abs(float(lp2.u[1]) - 3.215250) < 1e-3, f"LP 2 u2 mismatch: {lp2.u[1]}"

    # Verify Hopf (HB) point
    hb_pts = res.get_bifurcations("HB")
    assert len(hb_pts) >= 1, f"Expected at least 1 HB point, got {len(hb_pts)}"
    hb = hb_pts[0]
    assert abs(hb.p - 0.1308998) < 1e-3, f"HB p1 mismatch: {hb.p}"
    assert abs(float(hb.u[0]) - 0.8950803) < 1e-3, f"HB u1 mismatch: {hb.u[0]}"
    assert abs(float(hb.u[1]) - 4.177042) < 1e-3, f"HB u2 mismatch: {hb.u[1]}"

    # Verify stability structure:
    # 1. Lower branch (before LP 1, u2 < 1.4) is stable
    # 2. Middle & upper pre-Hopf branch (between LP 1 and HB, 1.5 < u2 < 4.1) is unstable
    # 3. Upper branch beyond HB (u2 > 4.3) is stable
    u2_arr = np.asarray(res.u)[:, 1]
    st_arr = np.asarray(res.stability)

    assert np.all(st_arr[u2_arr < 1.4]), "Lower branch (before LP 1) should be stable"
    assert np.all(~st_arr[(u2_arr > 1.5) & (u2_arr < 4.1)]), (
        "Branch between LP 1 and HB should be unstable"
    )
    assert np.all(st_arr[u2_arr > 4.3]), "Upper branch beyond HB should be stable"

    # Test plotting utility
    fig, ax = plot_diagram(res, state_index=0, title="Demo AB S-Curve")
    assert fig is not None
    assert ax is not None
