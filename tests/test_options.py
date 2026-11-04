from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from bifurx.bvp.collocation import (
    CollocationProblem,
    continue_periodic_orbit,
    solve_periodic_orbit,
)
from bifurx.codim2.moore_spence import continuation_codim2
from bifurx.continuation.arclength import (
    continuation,
    continuation_scan,
)
from bifurx.options import (
    CollocationOptions,
    ContinuationOptions,
    NewtonOptions,
)
from bifurx.problem import BifurcationProblem
from bifurx.solvers.newton import bordered_newton_solve


def f_pitchfork(u: jnp.ndarray, p: float | jnp.ndarray) -> jnp.ndarray:
    return jnp.array([p * u[0] - u[0] ** 3])


def f_ab_2p(u: jnp.ndarray, p1: float, p3: float) -> jnp.ndarray:
    u1, u2 = u[0], u[1]
    e = jnp.exp(u2)
    f1 = -u1 + p1 * (1.0 - u1) * e
    f2 = -u2 + p1 * 14.0 * (1.0 - u1) * e - p3 * u2
    return jnp.array([f1, f2])


def f_stuart_landau(u: jnp.ndarray, p: float | jnp.ndarray = 0.0) -> jnp.ndarray:
    x, y = u[0], u[1]
    r2 = x * x + y * y
    return jnp.array([x - y - x * r2, x + y - y * r2])


def test_options_default_values():
    """Verify default values and types for all centralized options."""
    newton = NewtonOptions()
    assert newton.tol == 1e-8
    assert newton.step_tol == 1e-8
    assert newton.max_iters == 15
    assert newton.solver_method == "auto"

    cont = ContinuationOptions()
    assert cont.ds == 0.05
    assert cont.ds_min == 1e-6
    assert cont.ds_max == 0.5
    assert cont.max_steps == 100
    assert cont.adaptive_step is False
    assert cont.detect_bifurcations is True
    assert cont.direction == 1.0
    assert cont.newton == NewtonOptions()

    colloc = CollocationOptions()
    assert colloc.num_intervals == 25
    assert colloc.num_gauss_points == 4
    assert colloc.eps == 0.02
    assert colloc.tol == 1e-8
    assert colloc.max_iters == 25
    assert colloc.min_amplitude == 1e-4


def test_options_immutability():
    """Verify that options dataclasses are frozen and immutable."""
    newton = NewtonOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        newton.tol = 1e-6  # type: ignore[misc]

    cont = ContinuationOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cont.ds = 0.1  # type: ignore[misc]

    colloc = CollocationOptions()
    with pytest.raises(dataclasses.FrozenInstanceError):
        colloc.num_intervals = 50  # type: ignore[misc]


def test_options_replace_method():
    """Verify that .replace() returns a new updated instance without modifying original."""
    newton0 = NewtonOptions()
    newton1 = newton0.replace(tol=1e-5, max_iters=30)
    assert newton0.tol == 1e-8
    assert newton0.max_iters == 15
    assert newton1.tol == 1e-5
    assert newton1.max_iters == 30

    cont0 = ContinuationOptions()
    cont1 = cont0.replace(ds=0.01, max_steps=200, newton=newton1)
    assert cont0.ds == 0.05
    assert cont0.max_steps == 100
    assert cont1.ds == 0.01
    assert cont1.max_steps == 200
    assert cont1.newton.tol == 1e-5

    colloc0 = CollocationOptions()
    colloc1 = colloc0.replace(num_intervals=10, min_amplitude=1e-3)
    assert colloc0.num_intervals == 25
    assert colloc1.num_intervals == 10
    assert colloc1.min_amplitude == 1e-3


def test_options_pytree_registration():
    """Verify that options instances can be flattened and unflattened as JAX PyTrees."""
    cont = ContinuationOptions(
        ds=0.02,
        max_steps=50,
        newton=NewtonOptions(tol=1e-6, max_iters=20),
    )
    leaves, treedef = jax.tree_util.tree_flatten(cont)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt == cont

    colloc = CollocationOptions(num_intervals=30, tol=1e-7)
    leaves_c, treedef_c = jax.tree_util.tree_flatten(colloc)
    rebuilt_c = jax.tree_util.tree_unflatten(treedef_c, leaves_c)
    assert rebuilt_c == colloc


def test_bordered_newton_solve_with_options():
    """Verify bordered_newton_solve accepts NewtonOptions and kwargs override."""
    prob = BifurcationProblem(f_pitchfork, u0=[0.0], p0=1.0)
    u_prev = jnp.array([1.0])
    p_prev = jnp.array(1.0)
    tau_prev = jnp.array([0.0, 1.0])
    u_init = jnp.array([1.0])
    p_init = jnp.array(1.05)
    ds = 0.05

    # 1. Using NewtonOptions
    opts = NewtonOptions(tol=1e-7, max_iters=10)
    res1 = bordered_newton_solve(
        prob, u_init, p_init, u_prev, p_prev, tau_prev, ds=ds, options=opts
    )
    assert bool(res1.converged)

    # 2. Overriding kwarg takes precedence over options
    res2 = bordered_newton_solve(
        prob, u_init, p_init, u_prev, p_prev, tau_prev, ds=ds, options=opts, max_iters=20
    )
    assert bool(res2.converged)


def test_continuation_scan_with_options_and_overrides():
    """Verify continuation_scan supports options object, kwargs override, and kwargs fallback."""
    prob = BifurcationProblem(f_pitchfork, u0=[0.5], p0=0.25)

    # 1. Custom ContinuationOptions
    opts = ContinuationOptions(ds=0.02, max_steps=25)
    res_opts = continuation_scan(prob, u0=[0.5], p0=0.25, options=opts)
    assert res_opts.num_points == 25
    assert jnp.all(res_opts.converged)

    # 2. Explicit kwarg overrides options field
    res_override = continuation_scan(prob, u0=[0.5], p0=0.25, options=opts, max_steps=15)
    assert res_override.num_points == 15

    # 3. Kwargs fallback without options object (backwards compatibility)
    res_legacy = continuation_scan(prob, u0=[0.5], p0=0.25, ds=0.03, n_steps=20)
    assert res_legacy.num_points == 20
    assert jnp.all(res_legacy.converged)


def test_continuation_with_options_and_overrides():
    """Verify continuation supports options object, kwargs override, and kwargs fallback."""
    prob = BifurcationProblem(f_pitchfork, u0=[0.0], p0=-0.5)

    # 1. Custom options
    opts = ContinuationOptions(ds=0.04, max_steps=20, adaptive_step=True)
    res_opts = continuation(problem=prob, options=opts)
    assert res_opts.num_points > 5
    assert len(res_opts.get_bifurcations("BP")) >= 1

    # 2. Options with overriding kwarg
    res_override = continuation(problem=prob, options=opts, max_steps=10)
    assert res_override.num_points <= 11

    # 3. Legacy kwargs without options
    res_legacy = continuation(problem=prob, ds=0.04, max_steps=25)
    assert res_legacy.num_points > 5


def test_continuation_codim2_with_options_and_overrides():
    """Verify continuation_codim2 supports options object, kwargs override, and kwargs fallback."""
    u_lp1 = jnp.array([0.3110178, 1.4514162])
    p1_lp1 = 0.1057390
    p3_init = 2.0

    # 1. With ContinuationOptions
    opts = ContinuationOptions(ds=0.03, ds_max=0.1, max_steps=30, adaptive_step=True)
    res_opts = continuation_codim2(
        f_ab_2p,
        bif_type="LP",
        u_init=u_lp1,
        p1_init=p1_lp1,
        p2_init=p3_init,
        options=opts,
        p2_bounds=(1.5, 3.2),
    )
    assert res_opts.bif_type == "LP"
    assert res_opts.num_points >= 15

    # 2. Kwargs override options
    res_override = continuation_codim2(
        f_ab_2p,
        bif_type="LP",
        u_init=u_lp1,
        p1_init=p1_lp1,
        p2_init=p3_init,
        options=opts,
        max_steps=15,
        p2_bounds=(1.5, 3.2),
    )
    assert res_override.num_points <= 16


def test_collocation_and_continue_periodic_orbit_with_options():
    """Verify CollocationProblem, solve_periodic_orbit, and continue_periodic_orbit with options."""
    colloc_opts = CollocationOptions(num_intervals=15, num_gauss_points=3, tol=1e-8)

    # 1. CollocationProblem using CollocationOptions
    prob = CollocationProblem(f_stuart_landau, dim=2, options=colloc_opts)
    assert prob.num_intervals == 15
    assert prob.num_gauss_points == 3

    t_grid = jnp.linspace(0, 2 * jnp.pi, 15, endpoint=False)
    u_mesh = jnp.stack([jnp.cos(t_grid), jnp.sin(t_grid)], axis=1)

    # 2. solve_periodic_orbit using CollocationOptions
    sol = solve_periodic_orbit(prob, u_mesh, T_init=6.2, p=0.0, options=colloc_opts)
    assert bool(sol.converged)

    # 3. continue_periodic_orbit using ContinuationOptions
    cont_opts = ContinuationOptions(ds=0.02, max_steps=5)
    branch = continue_periodic_orbit(
        prob,
        initial_orbit=sol,
        p_start=0.0,
        p_end=0.05,
        options=cont_opts,
    )
    assert len(branch) >= 2
