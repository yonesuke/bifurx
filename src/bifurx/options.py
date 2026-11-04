from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Self

import jax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NewtonOptions:
    """Options for bordered Newton corrector iterations.

    Attributes
    ----------
    tol : float, default=1e-8
        Residual norm convergence tolerance.
    step_tol : float, default=1e-8
        Step size norm convergence tolerance.
    rel_tol : float, default=1e-8
        Relative residual norm convergence tolerance.
    max_iters : int, default=15
        Maximum number of Newton iterations per step.
    solver_method : str, default="auto"
        Linear solver method for bordered systems ("auto", "block_elimination", "direct").
    """

    tol: float = 1e-8
    step_tol: float = 1e-8
    rel_tol: float = 1e-8
    max_iters: int = 15
    solver_method: str = "auto"

    def replace(self, **changes: Any) -> Self:
        """Return a copy of the options with updated fields."""
        return dataclasses.replace(self, **changes)

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = (self.tol, self.step_tol, self.rel_tol)
        aux_data = (self.max_iters, self.solver_method)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[Any, ...], children: tuple[Any, ...]) -> Self:
        tol, step_tol, rel_tol = children
        max_iters, solver_method = aux_data
        return cls(
            tol=tol,
            step_tol=step_tol,
            rel_tol=rel_tol,
            max_iters=max_iters,
            solver_method=solver_method,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ContinuationOptions:
    """Options for pseudo-arclength numerical continuation.

    Attributes
    ----------
    ds : float, default=0.05
        Initial arclength step size.
    ds_min : float, default=1e-6
        Minimum allowable step size.
    ds_max : float, default=0.5
        Maximum allowable step size.
    max_steps : int, default=100
        Maximum number of continuation steps.
    adaptive_step : bool, default=False
        Whether to dynamically adapt the arclength step size.
    detect_bifurcations : bool, default=True
        Whether to monitor test functions and locate bifurcation points.
    direction : float, default=1.0
        Continuation direction (+1.0 forward, -1.0 backward).
    newton : NewtonOptions
        Newton corrector options.
    bif_p_tol : float, default=1e-4
        Relative tolerance in parameter for bifurcation point de-duplication.
    bif_u_tol : float, default=1e-3
        Relative tolerance in state norm for bifurcation point de-duplication.
    """

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

    def replace(self, **changes: Any) -> Self:
        """Return a copy of the options with updated fields."""
        return dataclasses.replace(self, **changes)

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = (
            self.ds,
            self.ds_min,
            self.ds_max,
            self.direction,
            self.newton,
            self.bif_p_tol,
            self.bif_u_tol,
        )
        aux_data = (self.max_steps, self.adaptive_step, self.detect_bifurcations)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[Any, ...], children: tuple[Any, ...]) -> Self:
        ds, ds_min, ds_max, direction, newton, bif_p_tol, bif_u_tol = children
        max_steps, adaptive_step, detect_bifurcations = aux_data
        return cls(
            ds=ds,
            ds_min=ds_min,
            ds_max=ds_max,
            max_steps=max_steps,
            adaptive_step=adaptive_step,
            detect_bifurcations=detect_bifurcations,
            direction=direction,
            newton=newton,
            bif_p_tol=bif_p_tol,
            bif_u_tol=bif_u_tol,
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CollocationOptions:
    """Options for boundary value problem (BVP) orthogonal collocation.

    Attributes
    ----------
    num_intervals : int, default=25
        Number of mesh intervals (N).
    num_gauss_points : int, default=4
        Number of Gauss collocation points per mesh interval (m).
    eps : float, default=0.02
        Perturbation amplitude for cycle initialization from Hopf bifurcation.
    tol : float, default=1e-8
        Convergence tolerance for collocation Newton solver.
    max_iters : int, default=25
        Maximum iterations for collocation Newton solver.
    min_amplitude : float, default=1e-4
        Minimum RMS amplitude required for non-trivial orbit constraint.
    """

    num_intervals: int = 25
    num_gauss_points: int = 4
    eps: float = 0.02
    tol: float = 1e-8
    max_iters: int = 25
    min_amplitude: float = 1e-4

    def replace(self, **changes: Any) -> Self:
        """Return a copy of the options with updated fields."""
        return dataclasses.replace(self, **changes)

    def tree_flatten(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = (self.eps, self.tol, self.min_amplitude)
        aux_data = (self.num_intervals, self.num_gauss_points, self.max_iters)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple[Any, ...], children: tuple[Any, ...]) -> Self:
        eps, tol, min_amplitude = children
        num_intervals, num_gauss_points, max_iters = aux_data
        return cls(
            num_intervals=num_intervals,
            num_gauss_points=num_gauss_points,
            eps=eps,
            tol=tol,
            max_iters=max_iters,
            min_amplitude=min_amplitude,
        )
