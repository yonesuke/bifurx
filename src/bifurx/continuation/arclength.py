from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from bifurx.bifurcations.test_funcs import (
    refine_bifurcation_point,
    test_func_branch_point,
    test_func_fold,
    test_func_hopf,
)
from bifurx.problem import BifurcationProblem
from bifurx.solvers.bordered import solve_bordered_system
from bifurx.solvers.newton import bordered_newton_solve


@dataclass
class BifurcationPoint:
    """Detected and refined bifurcation point."""

    bif_type: str  # "LP", "BP", "HB"
    u: Array
    p: float
    step: int
    tangent: Array
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinuationResult:
    """Complete result trajectory of pseudo-arclength continuation."""

    problem: BifurcationProblem
    u: Array
    p: Array
    tangents: Array
    stability: Array
    bifurcation_points: list[BifurcationPoint]
    step_sizes: Array
    iterations: Array
    test_function_values: dict[str, Array]
    branch_name: str = "Branch"

    @property
    def num_points(self) -> int:
        return int(self.p.shape[0])

    def get_bifurcations(self, bif_type: str | None = None) -> list[BifurcationPoint]:
        """Filter detected bifurcation points by type."""
        if bif_type is None:
            return self.bifurcation_points
        return [pt for pt in self.bifurcation_points if pt.bif_type == bif_type]


def compute_initial_tangent(
    problem: BifurcationProblem,
    u0: Array,
    p0: float,
    direction: float = 1.0,
) -> Array:
    """Compute normalized initial tangent vector (dot{u}, dot{p}) at starting equilibrium."""
    Ju = problem.jacobian_u(u0, p0)
    Jp = problem.jacobian_p(u0, p0)
    J = jnp.hstack([Ju, Jp.reshape(-1, 1)])
    _, _, vh = jnp.linalg.svd(J)
    tau = vh[-1, :]
    if float(tau[-1] * direction) < 0:
        tau = -tau
    norm_val = float(jnp.linalg.norm(tau))
    if norm_val > 1e-12:
        tau = tau / norm_val
    return tau


def continuation(
    problem: BifurcationProblem,
    u0: Array | None = None,
    p0: float | None = None,
    direction: float = 1.0,
    initial_tangent: Array | None = None,
    ds: float = 0.05,
    ds_min: float = 1e-6,
    ds_max: float = 0.5,
    max_steps: int = 100,
    newton_tol: float = 1e-8,
    newton_max_iters: int = 15,
    detect_bifurcations: bool = True,
    param_bounds: tuple[float, float] | None = None,
    adaptive_step: bool = True,
    branch_name: str = "Branch",
    verbose: bool = False,
) -> ContinuationResult:
    """Keller pseudo-arclength continuation loop.

    Parameters
    ----------
    problem : BifurcationProblem
        Nonlinear system F(u, p) = 0.
    u0 : Array | None, optional
        Starting state vector (defaults to problem.u0).
    p0 : float | None, optional
        Starting parameter value (defaults to problem.p0).
    direction : float, default=1.0
        Direction of parameter continuation (+1.0 for increasing p, -1.0 for decreasing).
    initial_tangent : Array | None, optional
        Custom initial tangent vector.
    ds : float, default=0.05
        Initial arclength step size.
    ds_min : float, default=1e-6
        Minimum allowed arclength step size.
    ds_max : float, default=0.5
        Maximum allowed arclength step size.
    max_steps : int, default=100
        Maximum continuation steps to execute.
    newton_tol : float, default=1e-8
        Convergence tolerance for Newton corrector.
    newton_max_iters : int, default=15
        Maximum iterations per Newton solve.
    detect_bifurcations : bool, default=True
        Whether to monitor test functions and pinpoint LP, BP, and HB points.
    param_bounds : tuple[float, float] | None, optional
        Bounds (p_min, p_max) to terminate continuation.
    adaptive_step : bool, default=True
        Whether to dynamically adjust step size based on corrector convergence speed.
    branch_name : str, default="Branch"
        Identifier for this continuation branch.
    verbose : bool, default=False
        Whether to print step-by-step progress.

    Returns
    -------
    ContinuationResult
        Structured result containing trajectory points, stability, and bifurcations.
    """
    u_cur = problem.u0 if u0 is None else jnp.asarray(u0, dtype=jnp.float64)
    p_cur = problem.p0 if p0 is None else float(p0)
    bounds = param_bounds if param_bounds is not None else problem.param_bounds

    # Initial tangent
    if initial_tangent is not None:
        tau_cur = jnp.asarray(initial_tangent, dtype=jnp.float64)
        tau_cur = tau_cur / jnp.linalg.norm(tau_cur)
    else:
        tau_cur = compute_initial_tangent(problem, u_cur, p_cur, direction=direction)

    # Initial stability & test functions
    Ju_init = problem.jacobian_u(u_cur, p_cur)
    eigs_init = np.linalg.eigvals(np.asarray(Ju_init))
    is_stable = bool(np.all(np.real(eigs_init) < 0))

    tf_lp_init = test_func_fold(tau_cur)
    tf_bp_init = test_func_branch_point(Ju_init)
    tf_hb_init = test_func_hopf(Ju_init)

    u_history = [u_cur]
    p_history = [p_cur]
    tangent_history = [tau_cur]
    stability_history = [is_stable]
    step_sizes = [ds]
    iterations_history = [0]
    tf_lp_history = [tf_lp_init]
    tf_bp_history = [tf_bp_init]
    tf_hb_history = [tf_hb_init]
    bifurcation_points: list[BifurcationPoint] = []

    cur_ds = ds
    n = problem.dim

    if verbose:
        print(f"[{branch_name}] Start: p={p_cur:.4f}, u={np.asarray(u_cur)}")

    step = 0
    while step < max_steps:
        step += 1

        # 1. Tangent predictor
        u_pred = u_cur + cur_ds * tau_cur[:n]
        p_pred = p_cur + cur_ds * float(tau_cur[-1])

        # 2. Bordered Newton corrector
        res = bordered_newton_solve(
            problem=problem,
            u_init=u_pred,
            p_init=p_pred,
            u_prev=u_cur,
            p_prev=p_cur,
            tau_prev=tau_cur,
            ds=cur_ds,
            tol=newton_tol,
            max_iters=newton_max_iters,
        )

        if not res.converged:
            if adaptive_step and cur_ds > 2 * ds_min:
                cur_ds = max(cur_ds * 0.5, ds_min)
                if verbose:
                    print(
                        f"[{branch_name}] Step {step}: Corrector failed, reducing ds to {cur_ds:.4e}"
                    )
                continue
            else:
                if verbose:
                    print(
                        f"[{branch_name}] Corrector failed at step {step} with ds={cur_ds:.4e}. Stopping."
                    )
                break

        u_next = res.u
        p_next = res.p

        # 3. Tangent update at converged solution
        Ju_next = problem.jacobian_u(u_next, p_next)
        Jp_next = problem.jacobian_p(u_next, p_next)

        sol_tau = solve_bordered_system(
            A=Ju_next,
            b=Jp_next,
            c=tau_cur[:n],
            d=float(tau_cur[-1]),
            f=jnp.zeros(n, dtype=jnp.float64),
            g=1.0,
        )
        tau_next = sol_tau.z
        tau_norm = float(jnp.linalg.norm(tau_next))
        if tau_norm > 1e-12:
            tau_next = tau_next / tau_norm
        else:
            tau_next = tau_cur

        # Orientation preservation: <tau_next, tau_cur> > 0
        if float(jnp.dot(tau_next, tau_cur)) < 0:
            tau_next = -tau_next

        # 4. Stability & Test functions
        eigs_next = np.linalg.eigvals(np.asarray(Ju_next))
        is_stable_next = bool(np.all(np.real(eigs_next) < 0))

        tf_lp_next = test_func_fold(tau_next)
        tf_bp_next = test_func_branch_point(Ju_next)
        tf_hb_next = test_func_hopf(Ju_next)

        # 5. Bifurcation Detection & Refinement
        if detect_bifurcations:
            # Test 1: Limit Point / Fold (LP) - dot{p} zero crossing
            lp_detected = False
            if tf_lp_history[-1] * tf_lp_next < 0:
                lp_detected = True
                u_lp, p_lp = refine_bifurcation_point(
                    problem, u_cur, p_cur, tau_cur, u_next, p_next, tau_next, "LP"
                )
                bif_pt = BifurcationPoint(
                    bif_type="LP",
                    u=u_lp,
                    p=p_lp,
                    step=step,
                    tangent=tau_next,
                    details={"step": step},
                )
                bifurcation_points.append(bif_pt)
                if verbose:
                    print(f"  >>> [FOLD / LP] at step {step}: p={p_lp:.6f}, u={np.asarray(u_lp)}")

            # Test 2: Branch Point (BP) - det(Ju) sign change (excluding Folds)
            if not lp_detected and (tf_bp_history[-1] * tf_bp_next < 0):
                u_bp, p_bp = refine_bifurcation_point(
                    problem, u_cur, p_cur, tau_cur, u_next, p_next, tau_next, "BP"
                )
                bif_pt = BifurcationPoint(
                    bif_type="BP",
                    u=u_bp,
                    p=p_bp,
                    step=step,
                    tangent=tau_next,
                    details={"step": step, "Ju": Ju_next},
                )
                bifurcation_points.append(bif_pt)
                if verbose:
                    print(
                        f"  >>> [BRANCH POINT / BP] at step {step}: p={p_bp:.6f}, u={np.asarray(u_bp)}"
                    )

            # Test 3: Hopf Bifurcation (HB) - complex eigenvalue real part crossing
            if not np.isnan(tf_hb_history[-1]) and not np.isnan(tf_hb_next):
                if tf_hb_history[-1] * tf_hb_next < 0:
                    u_hb, p_hb = refine_bifurcation_point(
                        problem, u_cur, p_cur, tau_cur, u_next, p_next, tau_next, "HB"
                    )
                    bif_pt = BifurcationPoint(
                        bif_type="HB",
                        u=u_hb,
                        p=p_hb,
                        step=step,
                        tangent=tau_next,
                        details={"step": step},
                    )
                    bifurcation_points.append(bif_pt)
                    if verbose:
                        print(
                            f"  >>> [HOPF / HB] at step {step}: p={p_hb:.6f}, u={np.asarray(u_hb)}"
                        )

        # Append state
        u_history.append(u_next)
        p_history.append(p_next)
        tangent_history.append(tau_next)
        stability_history.append(is_stable_next)
        step_sizes.append(cur_ds)
        iterations_history.append(res.iterations)
        tf_lp_history.append(tf_lp_next)
        tf_bp_history.append(tf_bp_next)
        tf_hb_history.append(tf_hb_next)

        # 6. Adaptive step control
        if adaptive_step:
            if res.iterations <= 3:
                cur_ds = min(cur_ds * 2.0, ds_max)
            elif res.iterations >= 7:
                cur_ds = max(cur_ds * 0.5, ds_min)

        # Check bounds
        if bounds is not None:
            p_min, p_max = bounds
            if p_next < p_min or p_next > p_max:
                if verbose:
                    print(
                        f"[{branch_name}] Parameter reached bounds [{p_min}, {p_max}]: p={p_next:.4f}"
                    )
                break

        # Advance current state
        u_cur, p_cur, tau_cur = u_next, p_next, tau_next

    return ContinuationResult(
        problem=problem,
        u=jnp.stack(u_history, axis=0),
        p=jnp.array(p_history),
        tangents=jnp.stack(tangent_history, axis=0),
        stability=jnp.array(stability_history, dtype=bool),
        bifurcation_points=bifurcation_points,
        step_sizes=jnp.array(step_sizes),
        iterations=jnp.array(iterations_history),
        test_function_values={
            "LP": jnp.array(tf_lp_history),
            "BP": jnp.array(tf_bp_history),
            "HB": jnp.array(tf_hb_history),
        },
        branch_name=branch_name,
    )
