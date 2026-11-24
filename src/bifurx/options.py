from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax

from bifurx.homotopy import PyTreeOptionsBase, TrackerParameters, WeightedNormOptions


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NewtonOptions(PyTreeOptionsBase):
    """Options for bordered Newton corrector iterations."""

    _child_fields: ClassVar[tuple[str, ...]] = ("tol", "step_tol")
    _aux_fields: ClassVar[tuple[str, ...]] = ("max_iters", "solver_method")

    tol: float = 1e-8
    step_tol: float = 1e-8
    max_iters: int = 15
    solver_method: str = "auto"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ContinuationOptions(PyTreeOptionsBase):
    """Options for pseudo-arclength numerical continuation."""

    _child_fields: ClassVar[tuple[str, ...]] = (
        "ds",
        "ds_min",
        "ds_max",
        "direction",
        "newton",
        "tracker",
        "norm",
        "bif_p_tol",
        "bif_u_tol",
    )
    _aux_fields: ClassVar[tuple[str, ...]] = (
        "max_steps",
        "adaptive_step",
        "detect_bifurcations",
        "predictor",
    )

    ds: float = 0.05
    ds_min: float = 1e-6
    ds_max: float = 0.5
    max_steps: int = 100
    adaptive_step: bool = False
    detect_bifurcations: bool = True
    predictor: str = "tangent"
    direction: float = 1.0
    newton: NewtonOptions = field(default_factory=NewtonOptions)
    tracker: TrackerParameters = field(default_factory=TrackerParameters)
    norm: WeightedNormOptions = field(default_factory=WeightedNormOptions)
    bif_p_tol: float = 1e-4
    bif_u_tol: float = 1e-3


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CollocationOptions(PyTreeOptionsBase):
    """Options for boundary value problem (BVP) orthogonal collocation."""

    _child_fields: ClassVar[tuple[str, ...]] = ("eps", "tol", "min_amplitude", "period_weight")
    _aux_fields: ClassVar[tuple[str, ...]] = ("num_intervals", "num_gauss_points", "max_iters")

    num_intervals: int = 25
    num_gauss_points: int = 4
    eps: float = 0.02
    tol: float = 1e-8
    max_iters: int = 25
    min_amplitude: float = 1e-4
    period_weight: float = 1.0


def resolve_continuation_options(
    options: ContinuationOptions | None = None,
    *,
    default_options: ContinuationOptions | None = None,
    ds: Any = None,
    ds_min: Any = None,
    ds_max: Any = None,
    max_steps: int | None = None,
    n_steps: int | None = None,
    direction: Any = None,
    adaptive_step: bool | None = None,
    detect_bifurcations: bool | None = None,
    predictor: str | None = None,
    tracker: TrackerParameters | None = None,
    norm: WeightedNormOptions | None = None,
    bif_p_tol: float | None = None,
    bif_u_tol: float | None = None,
    newton_tol: float | None = None,
    newton_step_tol: float | None = None,
    newton_max_iters: int | None = None,
    solver_method: str | None = None,
    extra_kwargs: dict[str, Any] | None = None,
) -> ContinuationOptions:
    """Resolve ContinuationOptions and nested NewtonOptions from explicit overrides and kwargs."""
    extra = extra_kwargs or {}
    base_opts = (
        options
        if options is not None
        else (default_options if default_options is not None else ContinuationOptions())
    )

    tol_val = newton_tol if newton_tol is not None else extra.get("tol")
    step_tol_val = (
        newton_step_tol
        if newton_step_tol is not None
        else extra.get("newton_step_tol", extra.get("step_tol"))
    )
    max_iters_val = newton_max_iters if newton_max_iters is not None else extra.get("max_iters")
    solver_val = solver_method if solver_method is not None else extra.get("solver_method")

    newton_updates = {
        k: v
        for k, v in (
            ("tol", tol_val),
            ("step_tol", step_tol_val),
            ("max_iters", max_iters_val),
            ("solver_method", solver_val),
        )
        if v is not None
    }
    resolved_newton = (
        base_opts.newton.replace(**newton_updates) if newton_updates else base_opts.newton
    )

    steps = (
        max_steps
        if max_steps is not None
        else (n_steps if n_steps is not None else extra.get("n_steps"))
    )
    cont_updates = {
        k: v
        for k, v in (
            ("ds", ds),
            ("ds_min", ds_min),
            ("ds_max", ds_max),
            ("max_steps", steps),
            ("direction", direction),
            ("adaptive_step", adaptive_step),
            ("detect_bifurcations", detect_bifurcations),
            ("predictor", predictor if predictor is not None else extra.get("predictor")),
            ("tracker", tracker if tracker is not None else extra.get("tracker")),
            ("norm", norm if norm is not None else extra.get("norm")),
            ("bif_p_tol", bif_p_tol if bif_p_tol is not None else extra.get("bif_p_tol")),
            ("bif_u_tol", bif_u_tol if bif_u_tol is not None else extra.get("bif_u_tol")),
        )
        if v is not None
    }
    return base_opts.replace(newton=resolved_newton, **cont_updates)
