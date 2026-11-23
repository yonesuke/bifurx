from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.bifurcations import (
    evaluate_point_diagnostics,
    refine_bifurcation_point,
)
from bifurx.homotopy.corrector import correct_arclength_step
from bifurx.homotopy.tracker import compute_tangent, track_arclength_scan
from bifurx.options import ContinuationOptions, NewtonOptions, resolve_continuation_options
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
    test_funcs: dict[str, jax.Array]
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
    def num_points(self) -> int:
        """Total points in scan trajectory."""
        return int(self.p.shape[0])


def compute_initial_tangent(
    problem: BifurcationProblem,
    u0: ArrayLike | list[float] | Any,
    p0: float | jax.Array,
    direction: float | jax.Array = 1.0,
    initial_tangent: jax.Array | None = None,
) -> jax.Array:
    """Compute or normalize the initial tangent vector (dot{u}, dot{p}) at starting equilibrium."""
    return compute_tangent(
        jac_u_fn=problem.jacobian_u,
        jac_p_fn=problem.jacobian_p,
        u0=jnp.asarray(u0, dtype=jnp.float64),
        p0=p0,
        direction=direction,
        initial_tangent=initial_tangent,
    )


def _resolve_param_bounds(
    problem: BifurcationProblem,
    param_bounds: tuple[float, float] | jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Resolve (p_min, p_max) bounds with explicit override taking precedence over problem default."""
    bounds = param_bounds if param_bounds is not None else problem.param_bounds
    if bounds is not None:
        return jnp.asarray(bounds[0], dtype=jnp.float64), jnp.asarray(bounds[1], dtype=jnp.float64)
    return jnp.array(-jnp.inf, dtype=jnp.float64), jnp.array(jnp.inf, dtype=jnp.float64)


def continuation_scan(
    problem: BifurcationProblem,
    u0: ArrayLike | list[float] | Any | None = None,
    p0: float | jax.Array | None = None,
    direction: float | jax.Array | None = None,
    initial_tangent: jax.Array | None = None,
    ds: float | jax.Array | None = None,
    ds_min: float | jax.Array | None = None,
    ds_max: float | jax.Array | None = None,
    n_steps: int | None = None,
    newton_tol: float | None = None,
    newton_step_tol: float | None = None,
    newton_rel_tol: float | None = None,
    newton_max_iters: int | None = None,
    param_bounds: tuple[float, float] | jax.Array | None = None,
    adaptive_step: bool | None = None,
    predictor: str | None = None,
    solver_method: str | None = None,
    options: ContinuationOptions | None = None,
    max_steps: int | None = None,
    bif_p_tol: float | None = None,
    bif_u_tol: float | None = None,
) -> ContinuationScanResult:
    """End-to-end JIT pseudo-arclength continuation using `jax.lax.scan`."""
    opts = resolve_continuation_options(
        options,
        ds=ds,
        ds_min=ds_min,
        ds_max=ds_max,
        max_steps=max_steps,
        n_steps=n_steps,
        direction=direction,
        adaptive_step=adaptive_step,
        predictor=predictor,
        bif_p_tol=bif_p_tol,
        bif_u_tol=bif_u_tol,
        newton_tol=newton_tol,
        newton_step_tol=newton_step_tol,
        newton_rel_tol=newton_rel_tol,
        newton_max_iters=newton_max_iters,
        solver_method=solver_method,
    )

    u_cur = problem.u0 if u0 is None else jnp.asarray(u0, dtype=jnp.float64)
    p_cur = jnp.squeeze(jnp.asarray(problem.p0 if p0 is None else p0, dtype=jnp.float64))
    tau_cur = compute_initial_tangent(
        problem, u_cur, p_cur, direction=opts.direction, initial_tangent=initial_tangent
    )
    p_min, p_max = _resolve_param_bounds(problem, param_bounds)

    scan_res = track_arclength_scan(
        residual_fn=problem.residual,
        jac_u_fn=problem.jacobian_u,
        jac_p_fn=problem.jacobian_p,
        u0=u_cur,
        p0=p_cur,
        tau0=tau_cur,
        ds=opts.ds,
        ds_min=opts.ds_min,
        ds_max=opts.ds_max,
        n_steps=int(opts.max_steps),
        p_min=p_min,
        p_max=p_max,
        adaptive_step=bool(opts.adaptive_step),
        predictor=opts.predictor,
        tol=opts.newton.tol,
        step_tol=opts.newton.step_tol,
        rel_tol=opts.newton.rel_tol,
        max_newton_iters=opts.newton.max_iters,
        solver_method=opts.newton.solver_method,
        tracker_params=opts.tracker,
        norm_options=opts.norm,
        step_observer_fn=lambda _u, _p, tau_c, Ju_c, Jp_c: evaluate_point_diagnostics(
            Ju_c, tau_c, Jp=Jp_c
        ),
    )
    test_funcs, is_stable = scan_res.observer
    return ContinuationScanResult(
        u=scan_res.u,
        p=scan_res.p,
        tangents=scan_res.tangents,
        test_funcs=test_funcs,
        converged=scan_res.converged,
        step_sizes=scan_res.step_sizes,
        iterations=scan_res.iterations,
        stability=is_stable,
    )


def continuation(
    problem: BifurcationProblem,
    u0: ArrayLike | list[float] | Any | None = None,
    p0: float | jax.Array | None = None,
    direction: float | None = None,
    initial_tangent: jax.Array | None = None,
    ds: float | None = None,
    ds_min: float | None = None,
    ds_max: float | None = None,
    max_steps: int | None = None,
    newton_tol: float | None = None,
    newton_step_tol: float | None = None,
    newton_rel_tol: float | None = None,
    newton_max_iters: int | None = None,
    detect_bifurcations: bool | None = None,
    param_bounds: tuple[float, float] | None = None,
    adaptive_step: bool | None = None,
    predictor: str | None = None,
    branch_name: str = "Branch",
    verbose: bool = False,
    options: ContinuationOptions | None = None,
    bif_p_tol: float | None = None,
    bif_u_tol: float | None = None,
    **kwargs: Any,
) -> ContinuationResult:
    """Keller pseudo-arclength continuation loop with bifurcation detection and refinement."""
    opts = resolve_continuation_options(
        options,
        default_options=ContinuationOptions(adaptive_step=True),
        ds=ds,
        ds_min=ds_min,
        ds_max=ds_max,
        max_steps=max_steps,
        direction=direction,
        adaptive_step=adaptive_step,
        detect_bifurcations=detect_bifurcations,
        predictor=predictor,
        bif_p_tol=bif_p_tol,
        bif_u_tol=bif_u_tol,
        newton_tol=newton_tol,
        newton_step_tol=newton_step_tol,
        newton_rel_tol=newton_rel_tol,
        newton_max_iters=newton_max_iters,
        extra_kwargs=kwargs,
    )

    u_cur = problem.u0 if u0 is None else jnp.asarray(u0, dtype=jnp.float64)
    p_cur = jnp.squeeze(jnp.asarray(problem.p0 if p0 is None else p0, dtype=jnp.float64))

    if verbose:
        print(f"[{branch_name}] Start: p={float(p_cur):.4f}, u={u_cur}")

    tau_init = compute_initial_tangent(
        problem, u_cur, p_cur, direction=opts.direction, initial_tangent=initial_tangent
    )

    scan_res = continuation_scan(
        problem=problem,
        u0=u_cur,
        p0=p_cur,
        direction=opts.direction,
        initial_tangent=tau_init,
        param_bounds=param_bounds,
        options=opts,
    )

    Ju_init = problem.jacobian_u(u_cur, p_cur)
    Jp_init = problem.jacobian_p(u_cur, p_cur)
    tf_init, st_init = evaluate_point_diagnostics(Ju_init, tau_init, Jp=Jp_init)

    keep = jnp.concatenate([jnp.array([True]), scan_res.converged], axis=0)
    u_full = jnp.concatenate([u_cur[None, :], scan_res.u], axis=0)[keep]
    p_full = jnp.concatenate([jnp.atleast_1d(p_cur), scan_res.p], axis=0)[keep]
    tau_full = jnp.concatenate([tau_init[None, :], scan_res.tangents], axis=0)[keep]
    ds_full = jnp.concatenate([jnp.atleast_1d(opts.ds), scan_res.step_sizes], axis=0)[keep]
    it_full = jnp.concatenate([jnp.array([0], dtype=jnp.int32), scan_res.iterations], axis=0)[keep]
    st_full = jnp.concatenate([jnp.atleast_1d(st_init), scan_res.stability], axis=0)[keep]

    tf_full = {
        k: jnp.concatenate([jnp.atleast_1d(tf_init[k]), scan_res.test_funcs[k]], axis=0)[keep]
        for k in ("LP", "BP", "HB", "st_idx")
    }
    tf_lp_full, tf_bp_full, tf_hb_full, tf_st_idx_full = (
        tf_full["LP"],
        tf_full["BP"],
        tf_full["HB"],
        tf_full["st_idx"],
    )

    bif_points: list[BifurcationPoint] = []
    if opts.detect_bifurcations and int(p_full.shape[0]) >= 2:
        lp_cross = (tf_lp_full[:-1] * tf_lp_full[1:]) < 0.0
        hb_valid = (~jnp.isnan(tf_hb_full[:-1])) & (~jnp.isnan(tf_hb_full[1:]))
        hb_cross = (~lp_cross) & hb_valid & ((tf_hb_full[:-1] * tf_hb_full[1:]) < 0.0)
        bp_det_cross = (~lp_cross) & ((tf_bp_full[:-1] * tf_bp_full[1:]) < 0.0)
        st_cross = (~lp_cross) & (~hb_cross) & (tf_st_idx_full[:-1] != tf_st_idx_full[1:])
        bp_cross = bp_det_cross | st_cross

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
                if not (bool(jnp.all(jnp.isfinite(u_ref))) and bool(jnp.isfinite(p_ref))):
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
                    Ju_ref = problem.jacobian_u(u_ref, p_ref)
                    details["Ju"] = Ju_ref
                    details["min_eig"] = float(jnp.min(jnp.abs(jnp.linalg.eigvals(Ju_ref))))
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

        deduped: list[BifurcationPoint] = []
        for bp in bif_points:
            if not any(
                bp.bif_type == prev.bif_type
                and abs(bp.p - prev.p) < opts.bif_p_tol * (1.0 + abs(prev.p))
                and float(jnp.linalg.norm(bp.u - prev.u))
                < opts.bif_u_tol * (1.0 + float(jnp.linalg.norm(prev.u)))
                for prev in deduped
            ):
                deduped.append(bp)
        bif_points = deduped

    num_conv = int(jnp.sum(scan_res.converged))
    p_min, p_max = _resolve_param_bounds(problem, param_bounds)
    if (p_full[-1] < p_min) or (p_full[-1] > p_max):
        stop_reason = "bounds"
    elif opts.max_steps == 0 or bool(scan_res.converged[-1]):
        stop_reason = "max_steps"
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
        test_funcs={
            "LP": tf_lp_full,
            "BP": tf_bp_full,
            "HB": tf_hb_full,
        },
        branch_name=branch_name,
        stop_reason=stop_reason,
    )
