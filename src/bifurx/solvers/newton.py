from __future__ import annotations

from typing import NamedTuple

import jax

from bifurx.homotopy.corrector import correct_arclength_step
from bifurx.options import NewtonOptions
from bifurx.problem import BifurcationProblem


class NewtonResult(NamedTuple):
    """Result of pure-JAX bordered Newton corrector iterations."""

    converged: jax.Array
    u: jax.Array
    p: jax.Array
    iterations: jax.Array
    residual_norm: jax.Array


def bordered_newton_solve(
    problem: BifurcationProblem,
    u_init: jax.Array,
    p_init: float | jax.Array,
    u_prev: jax.Array,
    p_prev: float | jax.Array,
    tau_prev: jax.Array,
    ds: float | jax.Array,
    tol: float | None = None,
    step_tol: float | None = None,
    rel_tol: float | None = None,
    max_iters: int | None = None,
    solver_method: str | None = None,
    options: NewtonOptions | None = None,
) -> NewtonResult:
    """Pure-JAX bordered Newton corrector for pseudo-arclength continuation."""
    base_opts = NewtonOptions() if options is None else options
    tol_val = base_opts.tol if tol is None else tol
    step_tol_val = base_opts.step_tol if step_tol is None else step_tol
    if rel_tol is not None:
        rel_tol_val = rel_tol
    elif options is not None:
        rel_tol_val = base_opts.rel_tol
    elif tol is not None:
        rel_tol_val = tol_val
    else:
        rel_tol_val = base_opts.rel_tol
    max_iters_val = base_opts.max_iters if max_iters is None else max_iters
    solver_method_val = base_opts.solver_method if solver_method is None else solver_method

    res = correct_arclength_step(
        residual_fn=problem.residual,
        jac_u_fn=problem.jacobian_u,
        jac_p_fn=problem.jacobian_p,
        u_init=u_init,
        p_init=p_init,
        u_prev=u_prev,
        p_prev=p_prev,
        tau_prev=tau_prev,
        ds=ds,
        tol=tol_val,
        step_tol=step_tol_val,
        rel_tol=rel_tol_val,
        max_iters=max_iters_val,
        solver_method=solver_method_val,
    )
    return NewtonResult(
        converged=res.converged,
        u=res.u,
        p=res.p,
        iterations=res.iterations,
        residual_norm=res.residual_norm,
    )
