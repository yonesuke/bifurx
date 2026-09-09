from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from bifurx.codim2 import continuation_codim2

P2 = 14.0


def f_ab_2p(u: jnp.ndarray, p1: float, p3: float) -> jnp.ndarray:
    """A-B reaction model with two parameters (p1, p3)."""
    u1, u2 = u[0], u[1]
    e = jnp.exp(u2)
    f1 = -u1 + p1 * (1.0 - u1) * e
    f2 = -u2 + p1 * P2 * (1.0 - u1) * e - p3 * u2
    return jnp.array([f1, f2])


def test_fold_continuation_codim2_auto_run3_benchmark():
    """Test 2-parameter Fold (LP) continuation on AUTO-07P demo 'ab' (Runs 3 & 4).

    Validates tracking the locus of folds in (p1, p3) parameter space and checks
    accuracy against the official AUTO-07P benchmark at p3 = 2.5000:
      p1 = 0.1353352, u1 = 0.4996530, u2 = 1.998613
    """
    # LP 1 starting point at p3 = 2.0
    u_lp1 = np.array([0.3110178, 1.4514162])
    p1_lp1 = 0.1057390
    p3_init = 2.0

    res = continuation_codim2(
        f_ab_2p,
        bif_type="LP",
        u_init=u_lp1,
        p1_init=p1_lp1,
        p2_init=p3_init,
        ds=0.03,
        direction=1.0,
        max_steps=50,
        p2_bounds=(1.5, 3.2),
    )

    assert res.num_points >= 20, f"Expected at least 20 steps, got {res.num_points}"
    assert res.bif_type == "LP"

    # Locate point on Fold locus closest to p3 = 2.5000
    p3_arr = np.asarray(res.p2)
    idx_target = int(np.argmin(np.abs(p3_arr - 2.5000)))
    p1_found = float(res.p1[idx_target])
    p3_found = float(res.p2[idx_target])
    u1_found = float(res.u[idx_target, 0])
    u2_found = float(res.u[idx_target, 1])

    assert abs(p3_found - 2.5000) < 0.05, f"p3 should be close to 2.5, got {p3_found}"
    # Verify agreement with AUTO-07P benchmark values
    assert abs(p1_found - 0.1353352) < 5e-3, f"Fold p1 mismatch: {p1_found}"
    assert abs(u1_found - 0.4996530) < 1e-2, f"Fold u1 mismatch: {u1_found}"
    assert abs(u2_found - 1.998613) < 5e-2, f"Fold u2 mismatch: {u2_found}"


def test_hopf_continuation_codim2():
    """Test 2-parameter Hopf (HB) continuation in (p1, p3) parameter space."""
    # Starting Hopf point at p3 = 2.0
    u_hb = np.array([0.8950809, 4.1770443])
    p1_hb = 0.1309003
    p3_init = 2.0

    res_hb = continuation_codim2(
        f_ab_2p,
        bif_type="HB",
        u_init=u_hb,
        p1_init=p1_hb,
        p2_init=p3_init,
        ds=0.03,
        direction=-1.0,
        max_steps=35,
        p2_bounds=(0.2, 2.5),
    )

    assert res_hb.num_points >= 15, f"Expected at least 15 steps, got {res_hb.num_points}"
    assert res_hb.bif_type == "HB"

    # Verify that p1 remains positive and finite along the Hopf curve
    p1_arr = np.asarray(res_hb.p1)
    p3_arr = np.asarray(res_hb.p2)
    assert np.all(p1_arr > 0.0)
    assert np.all(np.isfinite(p1_arr))
    assert np.all(np.isfinite(p3_arr))
