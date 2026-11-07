from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import jax


class _OptionsBase:
    """Shared PyTree serialization and immutable update methods for option dataclasses."""

    _child_fields: ClassVar[tuple[str, ...]] = ()
    _aux_fields: ClassVar[tuple[str, ...]] = ()

    def replace(self, **changes: Any) -> Self:
        """Return a copy of the options with updated fields."""
        return dataclasses.replace(self, **changes)  # type: ignore[arg-type]

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = tuple(getattr(self, f) for f in self._child_fields)
        aux_data = tuple(getattr(self, f) for f in self._aux_fields)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[Any, ...], children: tuple[Any, ...]) -> Self:
        kwargs = dict(zip(cls._child_fields, children, strict=True))
        kwargs.update(zip(cls._aux_fields, aux_data, strict=True))
        return cls(**kwargs)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NewtonOptions(_OptionsBase):
    """Options for bordered Newton corrector iterations."""

    _child_fields: ClassVar[tuple[str, ...]] = ("tol", "step_tol", "rel_tol")
    _aux_fields: ClassVar[tuple[str, ...]] = ("max_iters", "solver_method")

    tol: float = 1e-8
    step_tol: float = 1e-8
    rel_tol: float = 1e-8
    max_iters: int = 15
    solver_method: str = "auto"


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ContinuationOptions(_OptionsBase):
    """Options for pseudo-arclength numerical continuation."""

    _child_fields: ClassVar[tuple[str, ...]] = (
        "ds",
        "ds_min",
        "ds_max",
        "direction",
        "newton",
        "bif_p_tol",
        "bif_u_tol",
    )
    _aux_fields: ClassVar[tuple[str, ...]] = (
        "max_steps",
        "adaptive_step",
        "detect_bifurcations",
    )

    ds: float = 0.05
    ds_min: float = 1e-6
    ds_max: float = 0.5
    max_steps: int = 100
    adaptive_step: bool = False
    detect_bifurcations: bool = True
    direction: float = 1.0
    newton: NewtonOptions = field(default_factory=NewtonOptions)
    bif_p_tol: float = 1e-4
    bif_u_tol: float = 1e-3


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CollocationOptions(_OptionsBase):
    """Options for boundary value problem (BVP) orthogonal collocation."""

    _child_fields: ClassVar[tuple[str, ...]] = ("eps", "tol", "min_amplitude")
    _aux_fields: ClassVar[tuple[str, ...]] = ("num_intervals", "num_gauss_points", "max_iters")

    num_intervals: int = 25
    num_gauss_points: int = 4
    eps: float = 0.02
    tol: float = 1e-8
    max_iters: int = 25
    min_amplitude: float = 1e-4


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
    bif_p_tol: float | None = None,
    bif_u_tol: float | None = None,
    newton_tol: float | None = None,
    newton_step_tol: float | None = None,
    newton_rel_tol: float | None = None,
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
    rel_tol_val = (
        newton_rel_tol
        if newton_rel_tol is not None
        else extra.get("newton_rel_tol", extra.get("rel_tol"))
    )
    max_iters_val = newton_max_iters if newton_max_iters is not None else extra.get("max_iters")
    solver_val = solver_method if solver_method is not None else extra.get("solver_method")

    newton_updates = {
        k: v
        for k, v in (
            ("tol", tol_val),
            ("step_tol", step_tol_val),
            ("rel_tol", rel_tol_val),
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
            ("bif_p_tol", bif_p_tol if bif_p_tol is not None else extra.get("bif_p_tol")),
            ("bif_u_tol", bif_u_tol if bif_u_tol is not None else extra.get("bif_u_tol")),
        )
        if v is not None
    }
    return base_opts.replace(newton=resolved_newton, **cont_updates)
