from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.bifurcations.test_funcs import (
    evaluate_point_diagnostics,
    refine_bifurcation_point,
)
from bifurx.problem import BifurcationProblem
from bifurx.solvers.bordered import solve_bordered_system
from bifurx.solvers.newton import bordered_newton_solve


@dataclass
class BifurcationPoint:
    """Detected and refined bifurcation point."""

    bif_type: str  # "LP", "BP", "HB"
    u: jax.Array
    p: float | jax.Array
    step: int
    tangent: jax.Array
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinuationResult:
    """Complete result trajectory of pseudo-arclength continuation."""

    problem: BifurcationProblem
    u: jax.Array
    p: jax.Array
    tangents: jax.Array
    stability: jax.Array
    bifurcation_points: list[BifurcationPoint]
    step_sizes: jax.Array
    iterations: jax.Array
    test_function_values: dict[str, jax.Array]
    branch_name: str = "Branch"
    stop_reason: str = "max_steps"  # "max_steps", "bounds", or "corrector_failed"

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

    u: jax.Array  # shape (n_steps, dim)
    p: jax.Array  # shape (n_steps,)
    tangents: jax.Array  # shape (n_steps, dim + 1)
    test_funcs: dict[str, jax.Array]  # {"LP": jax.Array, "BP": jax.Array, "HB": jax.Array}
    converged: jax.Array  # shape (n_steps,) bool
    step_sizes: jax.Array  # shape (n_steps,)
    iterations: jax.Array  # shape (n_steps,)
    stability: jax.Array  # shape (n_steps,) bool

    @property
    def test_function_values(self) -> dict[str, jax.Array]:
        """Alias matching ContinuationResult interface."""
        return self.test_funcs

    @property
    def num_points(self) -> int:
        """Total points in scan trajectory."""
        return int(self.p.shape[0])


def compute_initial_tangent(
    problem: BifurcationProblem,
    u0: ArrayLike | list[float] | Any,
    p0: float | jax.Array,
    direction: float | jax.Array = 1.0,
) -> jax.Array:
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


def _resolve_param_bounds(
    problem: BifurcationProblem,
    param_bounds: tuple[float, float] | jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Resolve (p_min, p_max) bounds with explicit override taking precedence over problem default."""
    bounds = param_bounds if param_bounds is not None else problem.param_bounds
    if bounds is not None:
        p_min = jnp.asarray(bounds[0], dtype=jnp.float64)
        p_max = jnp.asarray(bounds[1], dtype=jnp.float64)
    else:
        p_min = jnp.array(-jnp.inf, dtype=jnp.float64)
        p_max = jnp.array(jnp.inf, dtype=jnp.float64)
    return p_min, p_max


def continuation_scan(
    problem: BifurcationProblem,
    u0: ArrayLike | list[float] | Any | None = None,
    p0: float | jax.Array | None = None,
    direction: float | jax.Array = 1.0,
    initial_tangent: jax.Array | None = None,
    ds: float | jax.Array = 0.05,
    ds_min: float | jax.Array = 1e-6,
    ds_max: float | jax.Array = 0.5,
    n_steps: int = 100,
    newton_tol: float = 1e-8,
    newton_step_tol: float = 1e-8,
    newton_max_iters: int = 15,
    param_bounds: tuple[float, float] | jax.Array | None = None,
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
    u0 : jax.Array | None, optional
        Starting state vector (defaults to problem.u0).
    p0 : float | jax.Array | None, optional
        Starting parameter value (defaults to problem.p0).
    direction : float | jax.Array, default=1.0
        Direction of continuation (+1.0 or -1.0).
    initial_tangent : jax.Array | None, optional
        Custom initial tangent vector.
    ds : float | jax.Array, default=0.05
        Initial arclength step size.
    ds_min : float | jax.Array, default=1e-6
        Minimum step size.
    ds_max : float | jax.Array, default=0.5
        Maximum step size.
    n_steps : int, default=100
        Number of continuation steps.
    newton_tol : float, default=1e-8
        Residual norm convergence tolerance for Newton corrector.
    newton_step_tol : float, default=1e-8
        Step size convergence tolerance for Newton corrector.
    newton_max_iters : int, default=15
        Maximum Newton iterations per step.
    param_bounds : tuple[float, float] | jax.Array | None, optional
        Bounds (p_min, p_max) to terminate active continuation.
        Overrides `problem.param_bounds` if provided.
    adaptive_step : bool, default=False
        Whether to adjust step size dynamically.
    solver_method : str, default="auto"
        Linear solver method for bordered systems.

    Returns
    -------
    ContinuationScanResult
        Stacked arrays: (u, p, tangents, test_funcs, converged, step_sizes, iterations, stability).
    """
    n = problem.dim
    u_cur = problem.u0 if u0 is None else jnp.asarray(u0, dtype=jnp.float64)
    p_cur = jnp.squeeze(jnp.asarray(problem.p0 if p0 is None else p0, dtype=jnp.float64))
    ds_cur = jnp.asarray(ds, dtype=jnp.float64)

    if initial_tangent is not None:
        tau_cur = jnp.asarray(initial_tangent, dtype=jnp.float64)
        tau_norm = jnp.linalg.norm(tau_cur)
        tau_cur = jnp.where(tau_norm > 1e-12, tau_cur / tau_norm, tau_cur)
    else:
        tau_cur = compute_initial_tangent(problem, u_cur, p_cur, direction=direction)

    p_min, p_max = _resolve_param_bounds(problem, param_bounds)

    init_carry = (u_cur, p_cur, tau_cur, ds_cur, jnp.bool_(True))

    def step_fn(carry, _):
        u, p, tau, cur_ds, is_active = carry

        # 1. Tangent predictor
        u_pred = u + cur_ds * tau[:n]
        p_pred = p + cur_ds * tau[-1]

        # 2. Pure JAX Newton corrector
        res = bordered_newton_solve(
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

        # 4. Canonical test functions & stability from a single eigensolve
        test_funcs, is_stable = evaluate_point_diagnostics(Ju_cand, tau_cand)

        # Advance state if step converged and active
        u_next = jnp.where(converged_step, u_cand, u)
        p_next = jnp.where(converged_step, p_cand, p)
        tau_next = jnp.where(converged_step, tau_cand, tau)

        in_bounds = (p_next >= p_min) & (p_next <= p_max)
        is_active_next = jnp.where(converged_step, in_bounds, is_active)

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
            is_stable,
        )
        return carry_next, output_step

    _, outputs = jax.lax.scan(step_fn, init_carry, None, length=n_steps)
    u_traj, p_traj, tau_traj, tf_traj, conv_traj, ds_traj, it_traj, st_traj = outputs

    return ContinuationScanResult(
        u=u_traj,
        p=p_traj,
        tangents=tau_traj,
        test_funcs=tf_traj,
        converged=conv_traj,
        step_sizes=ds_traj,
        iterations=it_traj,
        stability=st_traj,
    )


def continuation(
    problem: BifurcationProblem,
    u0: ArrayLike | list[float] | Any | None = None,
    p0: float | jax.Array | None = None,
    direction: float = 1.0,
    initial_tangent: jax.Array | None = None,
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

    The trajectory, test functions, and stability are generated in a single pass by
    the pure-JAX ``continuation_scan`` kernel (``jax.lax.scan``); this function
    retains converged steps and pinpoints LP/BP/HB bifurcations by refinement.
    """
    u_cur = problem.u0 if u0 is None else jnp.asarray(u0, dtype=jnp.float64)
    p_cur = jnp.squeeze(jnp.asarray(problem.p0 if p0 is None else p0, dtype=jnp.float64))

    if verbose:
        print(f"[{branch_name}] Start: p={float(p_cur):.4f}, u={u_cur}")

    if initial_tangent is not None:
        tau_init = jnp.asarray(initial_tangent, dtype=jnp.float64)
        tau_norm = jnp.linalg.norm(tau_init)
        tau_init = jnp.where(tau_norm > 1e-12, tau_init / tau_norm, tau_init)
    else:
        tau_init = compute_initial_tangent(problem, u_cur, p_cur, direction=direction)

    # 1. Pure-JAX `lax.scan` continuation kernel
    scan_res = continuation_scan(
        problem=problem,
        u0=u_cur,
        p0=p_cur,
        direction=direction,
        initial_tangent=tau_init,
        ds=ds,
        ds_min=ds_min,
        ds_max=ds_max,
        n_steps=max_steps,
        newton_tol=newton_tol,
        newton_max_iters=newton_max_iters,
        param_bounds=param_bounds,
        adaptive_step=adaptive_step,
    )

    # 2. Assemble the full trajectory including the starting point
    Ju_init = problem.jacobian_u(u_cur, p_cur)
    tf_init, st_init = evaluate_point_diagnostics(Ju_init, tau_init)

    keep = jnp.concatenate([jnp.array([True]), scan_res.converged], axis=0)
    u_full = jnp.concatenate([u_cur[None, :], scan_res.u], axis=0)[keep]
    p_full = jnp.concatenate([jnp.atleast_1d(p_cur), scan_res.p], axis=0)[keep]
    tau_full = jnp.concatenate([tau_init[None, :], scan_res.tangents], axis=0)[keep]
    ds_full = jnp.concatenate([jnp.atleast_1d(ds), scan_res.step_sizes], axis=0)[keep]
    it_full = jnp.concatenate([jnp.array([0], dtype=jnp.int32), scan_res.iterations], axis=0)[keep]
    st_full = jnp.concatenate([jnp.atleast_1d(st_init), scan_res.stability], axis=0)[keep]

    tf_lp_full = jnp.concatenate(
        [jnp.atleast_1d(tf_init["LP"]), scan_res.test_funcs["LP"]], axis=0
    )[keep]
    tf_bp_full = jnp.concatenate(
        [jnp.atleast_1d(tf_init["BP"]), scan_res.test_funcs["BP"]], axis=0
    )[keep]
    tf_hb_full = jnp.concatenate(
        [jnp.atleast_1d(tf_init["HB"]), scan_res.test_funcs["HB"]], axis=0
    )[keep]

    # 3. Vectorized sign-change detection & pure-JAX refinement
    bif_points: list[BifurcationPoint] = []
    if detect_bifurcations and int(p_full.shape[0]) >= 2:
        lp_cross = (tf_lp_full[:-1] * tf_lp_full[1:]) < 0.0
        bp_cross = (~lp_cross) & ((tf_bp_full[:-1] * tf_bp_full[1:]) < 0.0)
        hb_valid = (~jnp.isnan(tf_hb_full[:-1])) & (~jnp.isnan(tf_hb_full[1:]))
        hb_cross = (~lp_cross) & (~bp_cross) & hb_valid & ((tf_hb_full[:-1] * tf_hb_full[1:]) < 0.0)

        for bif_type, mask, label in (
            ("LP", lp_cross, "FOLD / LP"),
            ("BP", bp_cross, "BRANCH POINT / BP"),
            ("HB", hb_cross, "HOPF / HB"),
        ):
            steps = jnp.nonzero(mask)[0] + 1
            for s_idx in steps.tolist():
                u_ref, p_ref = refine_bifurcation_point(
                    problem,
                    u_full[s_idx - 1],
                    p_full[s_idx - 1],
                    tau_full[s_idx - 1],
                    u_full[s_idx],
                    p_full[s_idx],
                    tau_full[s_idx],
                    bif_type,
                )
                p_val = float(p_ref)
                u_finite = bool(jnp.all(jnp.isfinite(u_ref)))
                p_finite = bool(jnp.isfinite(p_ref))

                if not (u_finite and p_finite):
                    if verbose:
                        print(
                            f"  >>> [{label}] at step {s_idx}: dropped non-finite bifurcation point"
                        )
                    continue

                res_norm = float(jnp.linalg.norm(problem.residual(u_ref, p_ref)))
                details: dict[str, Any] = {"step": s_idx}
                if res_norm > 1e-3:
                    details["refinement_failed"] = True
                    details["residual_norm"] = res_norm
                if bif_type == "BP":
                    details["Ju"] = problem.jacobian_u(u_full[s_idx], p_full[s_idx])
                bif_points.append(
                    BifurcationPoint(
                        bif_type=bif_type,
                        u=u_ref,
                        p=p_val,
                        step=s_idx,
                        tangent=tau_full[s_idx],
                        details=details,
                    )
                )
                if verbose:
                    print(f"  >>> [{label}] at step {s_idx}: p={float(p_ref):.6f}, u={u_ref}")

        bif_points.sort(key=lambda pt: pt.step)

    # 4. Determine continuation termination reason
    num_conv = int(jnp.sum(scan_res.converged))
    p_min, p_max = _resolve_param_bounds(problem, param_bounds)
    if num_conv == max_steps:
        stop_reason = "max_steps"
    else:
        # Check if stopped because boundary was crossed
        p_last = p_full[-1]
        if (p_last < p_min) or (p_last > p_max):
            stop_reason = "bounds"
        else:
            stop_reason = "corrector_failed"

    if verbose and stop_reason == "corrector_failed":
        print(
            f"[{branch_name}] Warning: Newton corrector failed to converge at step {num_conv + 1} "
            f"(p={float(p_full[-1]):.4f}). Terminated before reaching param_bounds or max_steps."
        )

    return ContinuationResult(
        problem=problem,
        u=u_full,
        p=p_full,
        tangents=tau_full,
        stability=st_full,
        bifurcation_points=bif_points,
        step_sizes=ds_full,
        iterations=it_full,
        test_function_values={
            "LP": tf_lp_full,
            "BP": tf_bp_full,
            "HB": tf_hb_full,
        },
        branch_name=branch_name,
        stop_reason=stop_reason,
    )
