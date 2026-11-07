from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.bifurcations.test_funcs import (
    extract_leading_hopf_pair,
    initial_null_vector,
    moore_spence_fold_residual,
)
from bifurx.continuation.arclength import compute_initial_tangent, continuation_scan
from bifurx.options import ContinuationOptions, resolve_continuation_options
from bifurx.problem import BifurcationProblem


class Codim2Result(NamedTuple):
    """Result of a Codimension-2 bifurcation curve continuation."""

    p1: jax.Array
    p2: jax.Array
    u: jax.Array  # shape (num_points, dim)
    tangents: jax.Array  # shape (num_points, dim_total)
    bif_type: str  # "LP" or "HB"

    @property
    def num_points(self) -> int:
        """Number of tracked bifurcation points along the curve."""
        return int(self.p1.shape[0])


def continuation_codim2(
    fn: Callable[..., jax.Array],
    bif_type: str,
    u_init: ArrayLike,
    p1_init: float | jax.Array,
    p2_init: float | jax.Array,
    options: ContinuationOptions | None = None,
    ds: float | None = None,
    ds_min: float | None = None,
    ds_max: float | None = None,
    max_steps: int | None = None,
    direction: float | None = None,
    param_orient_index: int = 1,  # 0 for orienting along p1, 1 for p2
    p1_bounds: tuple[float, float] | None = None,
    p2_bounds: tuple[float, float] | None = None,
    tol: float | None = None,
    max_newton_iters: int | None = None,
    adaptive_step: bool | None = None,
    **kwargs: Any,
) -> Codim2Result:
    """Track a Codimension-2 bifurcation curve (Fold LP or Hopf HB) in two parameters (p1, p2)."""
    extra_opts = {
        k: kwargs.pop(k) for k in ("n_steps", "newton_step_tol", "solver_method") if k in kwargs
    }
    opts = resolve_continuation_options(
        options,
        default_options=ContinuationOptions(
            ds=0.02, ds_min=1e-4, ds_max=0.1, max_steps=120, adaptive_step=True
        ),
        ds=ds,
        ds_min=ds_min,
        ds_max=ds_max,
        max_steps=max_steps,
        direction=direction,
        adaptive_step=adaptive_step,
        newton_tol=tol,
        newton_max_iters=max_newton_iters,
        extra_kwargs=extra_opts,
    )

    bif_type_upper = bif_type.upper()
    u0 = jnp.asarray(u_init, dtype=jnp.float64).ravel()
    p1_0 = jnp.squeeze(jnp.asarray(p1_init, dtype=jnp.float64))
    p2_0 = jnp.squeeze(jnp.asarray(p2_init, dtype=jnp.float64))
    n = int(u0.shape[0])

    jac_u_fn = jax.jacobian(lambda u, p1, p2: fn(u, p1, p2, **kwargs), argnums=0)

    if bif_type_upper == "LP":
        Ju0 = jac_u_fn(u0, p1_0, p2_0)
        v0, l_norm = initial_null_vector(Ju0)
        aux0 = v0

        def aug_res(
            u_v: jax.Array, p1_v: jax.Array, p2_v: jax.Array, aux_v: jax.Array
        ) -> jax.Array:
            return moore_spence_fold_residual(
                fn(u_v, p1_v, p2_v, **kwargs), jac_u_fn(u_v, p1_v, p2_v), aux_v, l_norm
            )

    elif bif_type_upper == "HB":
        if n == 2:
            aux0 = jnp.zeros((0,), dtype=jnp.float64)

            def aug_res(
                u_v: jax.Array, p1_v: jax.Array, p2_v: jax.Array, _aux_v: jax.Array
            ) -> jax.Array:
                f_val = fn(u_v, p1_v, p2_v, **kwargs)
                Ju = jac_u_fn(u_v, p1_v, p2_v)
                return jnp.concatenate([f_val, jnp.atleast_1d(jnp.trace(Ju))])

        else:
            Ju0 = jac_u_fn(u0, p1_0, p2_0)
            eigs, vecs = jnp.linalg.eig(Ju0)
            _, leading_real, omega0 = extract_leading_hopf_pair(Ju0, eigs=eigs)
            target_eig = leading_real + 1j * omega0
            vec0 = vecs[:, jnp.argmin(jnp.abs(eigs - target_eig))]
            v_r0, v_i0 = jnp.real(vec0), jnp.imag(vec0)
            norm_v = jnp.sqrt(jnp.dot(v_r0, v_r0) + jnp.dot(v_i0, v_i0))
            v_r0, v_i0 = v_r0 / norm_v, v_i0 / norm_v
            l_R = v_r0
            aux0 = jnp.concatenate([v_r0, v_i0, jnp.atleast_1d(omega0)])

            def aug_res(
                u_v: jax.Array, p1_v: jax.Array, p2_v: jax.Array, aux_v: jax.Array
            ) -> jax.Array:
                vr_v, vi_v, om_v = aux_v[:n], aux_v[n : 2 * n], aux_v[2 * n]
                f_val = fn(u_v, p1_v, p2_v, **kwargs)
                Ju = jac_u_fn(u_v, p1_v, p2_v)
                return jnp.concatenate(
                    [
                        f_val,
                        Ju @ vr_v + om_v * vi_v,
                        Ju @ vi_v - om_v * vr_v,
                        jnp.atleast_1d(jnp.dot(l_R, vr_v) - 1.0),
                        jnp.atleast_1d(jnp.dot(l_R, vi_v)),
                    ]
                )

    else:
        msg = f"Unknown bif_type: {bif_type}. Must be 'LP' or 'HB'."
        raise ValueError(msg)

    if param_orient_index == 0:
        u_state0 = jnp.concatenate([u0, jnp.atleast_1d(p2_0), aux0])
        p_cont0 = p1_0
        active_bounds = p1_bounds

        def problem_fn(u_state: jax.Array, p_cont: jax.Array) -> jax.Array:
            return aug_res(u_state[:n], p_cont, u_state[n], u_state[n + 1 :])

    else:
        u_state0 = jnp.concatenate([u0, jnp.atleast_1d(p1_0), aux0])
        p_cont0 = p2_0
        active_bounds = p2_bounds

        def problem_fn(u_state: jax.Array, p_cont: jax.Array) -> jax.Array:
            return aug_res(u_state[:n], u_state[n], p_cont, u_state[n + 1 :])

    prob = BifurcationProblem(
        fn=problem_fn,
        u0=u_state0,
        p0=float(p_cont0),
        param_bounds=active_bounds,
    )

    tau0 = compute_initial_tangent(prob, u_state0, p_cont0, direction=opts.direction)
    scan_res = continuation_scan(
        problem=prob,
        u0=u_state0,
        p0=p_cont0,
        direction=opts.direction,
        initial_tangent=tau0,
        param_bounds=active_bounds,
        options=opts,
    )

    keep = jnp.concatenate([jnp.array([True]), scan_res.converged], axis=0)
    u_state_full = jnp.concatenate([u_state0[None, :], scan_res.u], axis=0)[keep]
    p_cont_full = jnp.concatenate([jnp.atleast_1d(p_cont0), scan_res.p], axis=0)[keep]
    tau_scan_full = jnp.concatenate([tau0[None, :], scan_res.tangents], axis=0)[keep]

    u_traj = u_state_full[:, :n]
    if param_orient_index == 0:
        p1_traj, p2_traj = p_cont_full, u_state_full[:, n]
        tau_traj = jnp.concatenate(
            [
                tau_scan_full[:, :n],
                tau_scan_full[:, -1:],
                tau_scan_full[:, n : n + 1],
                tau_scan_full[:, n + 1 : -1],
            ],
            axis=1,
        )
    else:
        p1_traj, p2_traj = u_state_full[:, n], p_cont_full
        tau_traj = jnp.concatenate(
            [
                tau_scan_full[:, : n + 1],
                tau_scan_full[:, -1:],
                tau_scan_full[:, n + 1 : -1],
            ],
            axis=1,
        )

    in_bounds = jnp.ones(p1_traj.shape[0], dtype=bool)
    if p1_bounds is not None:
        in_bounds = in_bounds & (p1_traj >= p1_bounds[0]) & (p1_traj <= p1_bounds[1])
    if p2_bounds is not None:
        in_bounds = in_bounds & (p2_traj >= p2_bounds[0]) & (p2_traj <= p2_bounds[1])

    still_in = jnp.cumprod(in_bounds.astype(jnp.int32)).astype(bool)
    valid_mask = still_in | jnp.concatenate([jnp.array([True]), still_in[:-1]])

    return Codim2Result(
        p1=p1_traj[valid_mask],
        p2=p2_traj[valid_mask],
        u=u_traj[valid_mask],
        tangents=tau_traj[valid_mask],
        bif_type=bif_type_upper,
    )
