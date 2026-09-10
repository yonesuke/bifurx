from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
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
from bifurx.solvers.newton import bordered_newton_solve, bordered_newton_solve_jax


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


class ContinuationScanResult(NamedTuple):
    """Result of pure-JAX JIT continuation with `jax.lax.scan`.

    Supports unpacking: `(u_trajectory, p_trajectory, tau_trajectory, test_funcs, converged) = res[:5]`.
    """

    u: Array  # shape (n_steps, dim)
    p: Array  # shape (n_steps,)
    tangents: Array  # shape (n_steps, dim + 1)
    test_funcs: dict[str, Array]  # {"LP": Array, "BP": Array, "HB": Array}
    converged: Array  # shape (n_steps,) bool
    step_sizes: Array  # shape (n_steps,)
    iterations: Array  # shape (n_steps,)

    @property
    def test_function_values(self) -> dict[str, Array]:
        """Alias matching ContinuationResult interface."""
        return self.test_funcs

    @property
    def num_points(self) -> int:
        """Total points in scan trajectory."""
        return int(self.p.shape[0])


def compute_initial_tangent(
    problem: BifurcationProblem,
    u0: Array | np.ndarray | list[float] | Any,
    p0: float | Array,
    direction: float | Array = 1.0,
) -> Array:
    """Compute normalized initial tangent vector (dot{u}, dot{p}) at starting equilibrium."""
    u0_arr = jnp.asarray(u0, dtype=jnp.float64)
    Ju = problem.jacobian_u(u0_arr, p0)
    Jp = problem.jacobian_p(u0_arr, p0)
    J = jnp.hstack([Ju, Jp.reshape(-1, 1)])
    _, _, vh = jnp.linalg.svd(J)
    tau = vh[-1, :]
    sgn = jnp.where(tau[-1] * direction < 0, -1.0, 1.0)
    tau = tau * sgn
    norm_val = jnp.linalg.norm(tau)
    return jnp.where(norm_val > 1e-12, tau / norm_val, tau)


def continuation_scan(
    problem: BifurcationProblem,
    u0: Array | np.ndarray | list[float] | Any | None = None,
    p0: float | Array | None = None,
    direction: float | Array = 1.0,
    initial_tangent: Array | None = None,
    ds: float | Array = 0.05,
    ds_min: float | Array = 1e-6,
    ds_max: float | Array = 0.5,
    n_steps: int = 100,
    newton_tol: float = 1e-8,
    newton_step_tol: float = 1e-8,
    newton_max_iters: int = 15,
    param_bounds: tuple[float, float] | Array | None = None,
    adaptive_step: bool = False,
    solver_method: str = "auto",
) -> ContinuationScanResult:
    """End-to-end JIT pseudo-arclength continuation using `jax.lax.scan`.

    Formulates the entire continuation trajectory as a pure JAX scan:
        carry_next, output_step = step_fn(carry, None)
    with Carry PyTree: (u, p, tau, ds, is_active).

    Can be compiled end-to-end with `jax.jit` and batched across thousands
    of parameters/initial conditions with `jax.vmap`.

    Parameters
    ----------
    problem : BifurcationProblem
        Nonlinear system definition F(u, p) = 0.
    u0 : Array | None, optional
        Starting state vector (defaults to problem.u0).
    p0 : float | Array | None, optional
        Starting parameter value (defaults to problem.p0).
    direction : float | Array, default=1.0
        Direction of continuation (+1.0 or -1.0).
    initial_tangent : Array | None, optional
        Custom initial tangent vector.
    ds : float | Array, default=0.05
        Initial arclength step size.
    ds_min : float | Array, default=1e-6
        Minimum step size.
    ds_max : float | Array, default=0.5
        Maximum step size.
    n_steps : int, default=100
        Number of continuation steps.
    newton_tol : float, default=1e-8
        Residual norm convergence tolerance for Newton corrector.
    newton_step_tol : float, default=1e-8
        Step size convergence tolerance for Newton corrector.
    newton_max_iters : int, default=15
        Maximum Newton iterations per step.
    param_bounds : tuple[float, float] | Array | None, optional
        Bounds (p_min, p_max) to terminate active continuation.
    adaptive_step : bool, default=False
        Whether to adjust step size dynamically.
    solver_method : str, default="auto"
        Linear solver method for bordered systems.

    Returns
    -------
    ContinuationScanResult
        Stacked arrays: (u, p, tangents, test_funcs, converged, step_sizes, iterations).
    """
    n = problem.dim
    u_cur = problem.u0 if u0 is None else jnp.asarray(u0, dtype=jnp.float64)
    p_cur = jnp.asarray(problem.p0 if p0 is None else p0, dtype=jnp.float64)
    p_cur = jnp.squeeze(p_cur)
    ds_cur = jnp.asarray(ds, dtype=jnp.float64)

    if initial_tangent is not None:
        tau_cur = jnp.asarray(initial_tangent, dtype=jnp.float64)
        tau_norm = jnp.linalg.norm(tau_cur)
        tau_cur = jnp.where(tau_norm > 1e-12, tau_cur / tau_norm, tau_cur)
    else:
        tau_cur = compute_initial_tangent(problem, u_cur, p_cur, direction=direction)

    if param_bounds is not None:
        p_min = jnp.asarray(param_bounds[0], dtype=jnp.float64)
        p_max = jnp.asarray(param_bounds[1], dtype=jnp.float64)
    elif problem.param_bounds is not None:
        p_min = jnp.asarray(problem.param_bounds[0], dtype=jnp.float64)
        p_max = jnp.asarray(problem.param_bounds[1], dtype=jnp.float64)
    else:
        p_min = jnp.array(-jnp.inf, dtype=jnp.float64)
        p_max = jnp.array(jnp.inf, dtype=jnp.float64)

    init_carry = (u_cur, p_cur, tau_cur, ds_cur, jnp.bool_(True))

    def step_fn(carry, _):
        u, p, tau, cur_ds, is_active = carry

        # 1. Tangent predictor
        u_pred = u + cur_ds * tau[:n]
        p_pred = p + cur_ds * tau[-1]

        # 2. Pure JAX Newton corrector
        res = bordered_newton_solve_jax(
            problem=problem,
            u_init=u_pred,
            p_init=p_pred,
            u_prev=u,
            p_prev=p,
            tau_prev=tau,
            ds=cur_ds,
            tol=newton_tol,
            step_tol=newton_step_tol,
            max_iters=newton_max_iters,
            solver_method=solver_method,
        )

        u_cand = res.u
        p_cand = res.p
        converged_step = is_active & res.converged

        # 3. Tangent update at candidate solution
        Ju_cand = problem.jacobian_u(u_cand, p_cand)
        Jp_cand = problem.jacobian_p(u_cand, p_cand)
        sol_tau = solve_bordered_system(
            A=Ju_cand,
            b=Jp_cand,
            c=tau[:n],
            d=tau[-1],
            f=jnp.zeros(n, dtype=jnp.float64),
            g=1.0,
            method=solver_method,
        )
        tau_cand = sol_tau.z
        tau_cand_norm = jnp.linalg.norm(tau_cand)
        tau_cand = jnp.where(tau_cand_norm > 1e-12, tau_cand / tau_cand_norm, tau)
        tau_cand = jnp.where(jnp.dot(tau_cand, tau) < 0, -tau_cand, tau_cand)

        # 4. Test functions in pure jnp
        tf_lp = tau_cand[-1]
        sign, logdet = jnp.linalg.slogdet(Ju_cand)
        tf_bp = sign * jnp.exp(jnp.clip(logdet, -80.0, 80.0))

        eigs = jnp.linalg.eigvals(Ju_cand)
        complex_mask = jnp.abs(jnp.imag(eigs)) > 1e-4
        has_complex = jnp.any(complex_mask)
        leading_real = jnp.max(jnp.where(complex_mask, jnp.real(eigs), -1e9))
        tf_hb = jnp.where(has_complex, leading_real, jnp.nan)

        test_funcs = {
            "LP": tf_lp,
            "BP": tf_bp,
            "HB": tf_hb,
        }

        # Advance state if step converged and active
        u_next = jnp.where(converged_step, u_cand, u)
        p_next = jnp.where(converged_step, p_cand, p)
        tau_next = jnp.where(converged_step, tau_cand, tau)

        # Bounds check
        in_bounds = (p_next >= p_min) & (p_next <= p_max)
        is_active_next = converged_step & in_bounds

        # 5. Step size adjustment
        if adaptive_step:
            ds_inc = jnp.minimum(cur_ds * 1.5, ds_max)
            ds_dec = jnp.maximum(cur_ds * 0.5, ds_min)
            ds_next = jnp.where(
                res.converged,
                jnp.where(
                    res.iterations <= 3, ds_inc, jnp.where(res.iterations >= 7, ds_dec, cur_ds)
                ),
                ds_dec,
            )
        else:
            ds_next = cur_ds

        carry_next = (u_next, p_next, tau_next, ds_next, is_active_next)
        output_step = (
            u_next,
            p_next,
            tau_next,
            test_funcs,
            converged_step,
            cur_ds,
            res.iterations,
        )
        return carry_next, output_step

    _, outputs = jax.lax.scan(step_fn, init_carry, None, length=n_steps)
    u_traj, p_traj, tau_traj, tf_traj, conv_traj, ds_traj, it_traj = outputs

    return ContinuationScanResult(
        u=u_traj,
        p=p_traj,
        tangents=tau_traj,
        test_funcs=tf_traj,
        converged=conv_traj,
        step_sizes=ds_traj,
        iterations=it_traj,
    )


def continuation(
    problem: BifurcationProblem,
    u0: Array | np.ndarray | list[float] | Any | None = None,
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
    use_scan: bool = False,
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
    use_scan : bool, default=False
        Whether to use pure-JAX jax.lax.scan implementation for end-to-end acceleration.

    Returns
    -------
    ContinuationResult
        Structured result containing trajectory points, stability, and bifurcations.
    """
    u_cur = problem.u0 if u0 is None else jnp.asarray(u0, dtype=jnp.float64)
    p_cur = problem.p0 if p0 is None else float(p0)
    bounds = param_bounds if param_bounds is not None else problem.param_bounds

    if use_scan:
        scan_res = continuation_scan(
            problem=problem,
            u0=u_cur,
            p0=p_cur,
            direction=direction,
            initial_tangent=initial_tangent,
            ds=ds,
            ds_min=ds_min,
            ds_max=ds_max,
            n_steps=max_steps,
            newton_tol=newton_tol,
            newton_max_iters=newton_max_iters,
            param_bounds=bounds,
            adaptive_step=adaptive_step,
        )
        u_full = jnp.concatenate([u_cur[None, :], scan_res.u], axis=0)
        p_full = jnp.concatenate([jnp.atleast_1d(p_cur), scan_res.p], axis=0)
        tau_init = compute_initial_tangent(problem, u_cur, p_cur, direction=direction)
        tau_full = jnp.concatenate([tau_init[None, :], scan_res.tangents], axis=0)
        conv_full = jnp.concatenate([jnp.array([True]), scan_res.converged], axis=0)
        ds_full = jnp.concatenate([jnp.atleast_1d(ds), scan_res.step_sizes], axis=0)
        it_full = jnp.concatenate([jnp.array([0]), scan_res.iterations], axis=0)

        Ju_init = problem.jacobian_u(u_cur, p_cur)
        tf_lp_init = test_func_fold(tau_init)
        tf_bp_init = test_func_branch_point(Ju_init)
        tf_hb_init = test_func_hopf(Ju_init)

        tf_lp_full = jnp.concatenate(
            [jnp.atleast_1d(tf_lp_init), scan_res.test_funcs["LP"]], axis=0
        )
        tf_bp_full = jnp.concatenate(
            [jnp.atleast_1d(tf_bp_init), scan_res.test_funcs["BP"]], axis=0
        )
        tf_hb_full = jnp.concatenate(
            [jnp.atleast_1d(tf_hb_init), scan_res.test_funcs["HB"]], axis=0
        )

        # Truncate at first non-converged point if any
        valid_mask = np.asarray(conv_full)
        if not np.all(valid_mask):
            first_fail = int(np.argmin(valid_mask))
            u_full = u_full[:first_fail]
            p_full = p_full[:first_fail]
            tau_full = tau_full[:first_fail]
            ds_full = ds_full[:first_fail]
            it_full = it_full[:first_fail]
            tf_lp_full = tf_lp_full[:first_fail]
            tf_bp_full = tf_bp_full[:first_fail]
            tf_hb_full = tf_hb_full[:first_fail]

        bif_points: list[BifurcationPoint] = []
        if detect_bifurcations:
            n_pts = len(p_full)
            for s in range(1, n_pts):
                # LP
                if float(tf_lp_full[s - 1] * tf_lp_full[s]) < 0:
                    u_lp, p_lp = refine_bifurcation_point(
                        problem,
                        u_full[s - 1],
                        float(p_full[s - 1]),
                        tau_full[s - 1],
                        u_full[s],
                        float(p_full[s]),
                        tau_full[s],
                        "LP",
                    )
                    bif_points.append(
                        BifurcationPoint("LP", u_lp, p_lp, s, tau_full[s], {"step": s})
                    )
                # BP
                elif float(tf_bp_full[s - 1] * tf_bp_full[s]) < 0:
                    u_bp, p_bp = refine_bifurcation_point(
                        problem,
                        u_full[s - 1],
                        float(p_full[s - 1]),
                        tau_full[s - 1],
                        u_full[s],
                        float(p_full[s]),
                        tau_full[s],
                        "BP",
                    )
                    bif_points.append(
                        BifurcationPoint("BP", u_bp, p_bp, s, tau_full[s], {"step": s})
                    )
                # HB
                elif not np.isnan(float(tf_hb_full[s - 1])) and not np.isnan(float(tf_hb_full[s])):
                    if float(tf_hb_full[s - 1] * tf_hb_full[s]) < 0:
                        u_hb, p_hb = refine_bifurcation_point(
                            problem,
                            u_full[s - 1],
                            float(p_full[s - 1]),
                            tau_full[s - 1],
                            u_full[s],
                            float(p_full[s]),
                            tau_full[s],
                            "HB",
                        )
                        bif_points.append(
                            BifurcationPoint("HB", u_hb, p_hb, s, tau_full[s], {"step": s})
                        )

        return ContinuationResult(
            problem=problem,
            u=u_full,
            p=p_full,
            tangents=tau_full,
            stability=jnp.ones(len(p_full), dtype=bool),
            bifurcation_points=bif_points,
            step_sizes=ds_full,
            iterations=it_full,
            test_function_values={
                "LP": tf_lp_full,
                "BP": tf_bp_full,
                "HB": tf_hb_full,
            },
            branch_name=branch_name,
        )

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
