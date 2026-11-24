from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from bifurx.bifurcations import (
    canonical_hopf_eigenvector,
    compute_first_lyapunov_coefficient,
    hopf_augmented_residual,
    initial_null_vector,
    moore_spence_fold_residual,
    test_func_hopf,
)
from bifurx.homotopy import compute_tangent, safeguarded_secant_root, track_arclength_scan
from bifurx.homotopy.corrector import correct_arclength_step
from bifurx.options import ContinuationOptions, resolve_continuation_options


class Codim2Point(NamedTuple):
    """Detected Codimension-2 singularity along an LP or HB curve."""

    bif_type: str  # "CP", "BT", "GH", "ZH"
    u: jax.Array
    p1: float
    p2: float
    step: int
    details: dict[str, Any]


class Codim2Result(NamedTuple):
    """Result of a Codimension-2 bifurcation curve continuation."""

    p1: jax.Array
    p2: jax.Array
    u: jax.Array  # shape (num_points, dim)
    tangents: jax.Array  # shape (num_points, dim_total)
    bif_type: str  # "LP" or "HB"
    singularities: tuple[Codim2Point, ...] = ()
    test_funcs: dict[str, jax.Array] | None = None

    @property
    def num_points(self) -> int:
        """Number of tracked bifurcation points along the curve."""
        return int(self.p1.shape[0])

    def get_singularities(self, bif_type: str | None = None) -> list[Codim2Point]:
        """Filter detected Codim-2 singularities by type."""
        if bif_type is None:
            return list(self.singularities)
        return [pt for pt in self.singularities if pt.bif_type == bif_type]


def _align_null_vectors(w_arr: jax.Array) -> jax.Array:
    """Continuously align SVD null-vector signs along a tracked curve."""
    if int(w_arr.shape[0]) <= 1:
        return w_arr

    def _step(prev_w: jax.Array, cur_w: jax.Array) -> tuple[jax.Array, jax.Array]:
        aligned = jnp.where(jnp.dot(prev_w, cur_w) < 0.0, -cur_w, cur_w)
        return aligned, aligned

    _, rest = jax.lax.scan(_step, w_arr[0], w_arr[1:])
    return jnp.concatenate([w_arr[:1], rest], axis=0)


def continuation_codim2(
    fn: Callable[..., jax.Array],
    bif_type: str,
    u_init: ArrayLike | Sequence[float],
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
    detect_singularities: bool = True,
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
            _, v_r0, v_i0, omega0 = canonical_hopf_eigenvector(Ju0)
            l_R = v_r0
            aux0 = jnp.concatenate([v_r0, v_i0, jnp.atleast_1d(omega0)])

            def aug_res(
                u_v: jax.Array, p1_v: jax.Array, p2_v: jax.Array, aux_v: jax.Array
            ) -> jax.Array:
                vr_v, vi_v, om_v = aux_v[:n], aux_v[n : 2 * n], aux_v[2 * n]
                return hopf_augmented_residual(
                    fn(u_v, p1_v, p2_v, **kwargs),
                    jac_u_fn(u_v, p1_v, p2_v),
                    vr_v,
                    vi_v,
                    om_v,
                    l_R,
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

        def _unpack_step(
            u_state: jax.Array, p_cont: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            return u_state[:n], p_cont, u_state[n], u_state[n + 1 :]

    else:
        u_state0 = jnp.concatenate([u0, jnp.atleast_1d(p1_0), aux0])
        p_cont0 = p2_0
        active_bounds = p2_bounds

        def problem_fn(u_state: jax.Array, p_cont: jax.Array) -> jax.Array:
            return aug_res(u_state[:n], u_state[n], p_cont, u_state[n + 1 :])

        def _unpack_step(
            u_state: jax.Array, p_cont: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            return u_state[:n], u_state[n], p_cont, u_state[n + 1 :]

    jac_u_aug = jax.jacfwd(problem_fn, argnums=0)
    jac_p_aug = jax.jacfwd(problem_fn, argnums=1)
    tau0 = compute_tangent(jac_u_aug, jac_p_aug, u_state0, p_cont0, direction=opts.direction)
    p_min = (
        jnp.array(-jnp.inf, dtype=jnp.float64)
        if active_bounds is None
        else jnp.asarray(active_bounds[0], dtype=jnp.float64)
    )
    p_max = (
        jnp.array(jnp.inf, dtype=jnp.float64)
        if active_bounds is None
        else jnp.asarray(active_bounds[1], dtype=jnp.float64)
    )

    if detect_singularities:
        if bif_type_upper == "LP":

            def _observer_fn(
                u_s: jax.Array,
                p_c: jax.Array,
                _tau_c: jax.Array,
                _ju_c: jax.Array,
                _jp_c: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                u_v, p1_v, p2_v, v_vec = _unpack_step(u_s, p_c)
                A = jac_u_fn(u_v, p1_v, p2_v)
                u_svd, _, _ = jnp.linalg.svd(A)
                w_raw = u_svd[:, -1]

                def _d_u(z: jax.Array) -> jax.Array:
                    return jax.jvp(lambda y: fn(y, p1_v, p2_v, **kwargs), (z,), (v_vec,))[1]

                b_vv = jax.jvp(_d_u, (u_v,), (v_vec,))[1]
                tf_zh = test_func_hopf(A) if n >= 3 else jnp.array(jnp.nan, dtype=jnp.float64)
                return w_raw, v_vec, b_vv, tf_zh

        else:

            def _observer_fn(
                u_s: jax.Array,
                p_c: jax.Array,
                _tau_c: jax.Array,
                _ju_c: jax.Array,
                _jp_c: jax.Array,
            ) -> tuple[jax.Array, ...]:
                u_v, p1_v, p2_v, aux_v = _unpack_step(u_s, p_c)
                A = jac_u_fn(u_v, p1_v, p2_v)
                det_A = jnp.linalg.det(A)
                l1 = compute_first_lyapunov_coefficient(
                    lambda u_, _p: fn(u_, p1_v, p2_v, **kwargs), u_v, p2_v, Ju=A
                )
                if n == 2:
                    tf_bt = det_A
                    tf_gh = jnp.where(det_A > 1e-10, l1, jnp.nan)
                    return (tf_gh, tf_bt)
                omega_v = aux_v[2 * n]
                tf_bt = omega_v
                tf_zh = jnp.where(jnp.abs(omega_v) > 1e-6, det_A, jnp.nan)
                tf_gh = jnp.where(jnp.abs(omega_v) > 1e-6, l1, jnp.nan)
                return (tf_gh, tf_bt, tf_zh)

        step_obs = _observer_fn
    else:
        step_obs = None

    scan_res = track_arclength_scan(
        residual_fn=problem_fn,
        jac_u_fn=jac_u_aug,
        jac_p_fn=jac_p_aug,
        u0=u_state0,
        p0=p_cont0,
        tau0=tau0,
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
        max_newton_iters=opts.newton.max_iters,
        solver_method=opts.newton.solver_method,
        tracker_params=opts.tracker,
        norm_options=opts.norm,
        step_observer_fn=step_obs,
    )

    keep = jnp.concatenate([jnp.array([True]), scan_res.converged], axis=0)
    u_state_full = jnp.concatenate([u_state0[None, :], scan_res.u], axis=0)[keep]
    p_cont_full = jnp.concatenate([jnp.atleast_1d(p_cont0), scan_res.p], axis=0)[keep]
    tau_scan_full = jnp.concatenate([tau0[None, :], scan_res.tangents], axis=0)[keep]

    if param_orient_index == 0:
        p1_all, p2_all = p_cont_full, u_state_full[:, n]
    else:
        p1_all, p2_all = u_state_full[:, n], p_cont_full

    in_bounds = jnp.ones(p1_all.shape[0], dtype=bool)
    if p1_bounds is not None:
        in_bounds = in_bounds & (p1_all >= p1_bounds[0]) & (p1_all <= p1_bounds[1])
    if p2_bounds is not None:
        in_bounds = in_bounds & (p2_all >= p2_bounds[0]) & (p2_all <= p2_bounds[1])

    still_in = jnp.cumprod(in_bounds.astype(jnp.int32)).astype(bool)
    valid_mask = still_in | jnp.concatenate([jnp.array([True]), still_in[:-1]])

    u_state_valid = u_state_full[valid_mask]
    p_cont_valid = p_cont_full[valid_mask]
    tau_scan_valid = tau_scan_full[valid_mask]
    p1_out = p1_all[valid_mask]
    p2_out = p2_all[valid_mask]
    u_traj = u_state_valid[:, :n]

    if param_orient_index == 0:
        tau_traj = jnp.concatenate(
            [
                tau_scan_valid[:, :n],
                tau_scan_valid[:, -1:],
                tau_scan_valid[:, n : n + 1],
                tau_scan_valid[:, n + 1 : -1],
            ],
            axis=1,
        )
    else:
        tau_traj = jnp.concatenate(
            [
                tau_scan_valid[:, : n + 1],
                tau_scan_valid[:, -1:],
                tau_scan_valid[:, n + 1 : -1],
            ],
            axis=1,
        )

    singularities: list[Codim2Point] = []
    test_funcs_out: dict[str, jax.Array] | None = None
    if detect_singularities and step_obs is not None:
        init_obs = step_obs(
            u_state0, p_cont0, tau0, jac_u_aug(u_state0, p_cont0), jac_p_aug(u_state0, p_cont0)
        )
        full_obs = tuple(
            jnp.concatenate([jnp.expand_dims(i_v, axis=0), o_v], axis=0)[keep][valid_mask]
            for i_v, o_v in zip(init_obs, scan_res.observer, strict=True)
        )
        w_aligned: jax.Array | None = None
        if bif_type_upper == "LP":
            w_aligned = _align_null_vectors(full_obs[0])
            v_valid, b_vv_valid = full_obs[1], full_obs[2]
            test_funcs_out = {
                "CP": 0.5 * jnp.sum(w_aligned * b_vv_valid, axis=-1),
                "BT": jnp.sum(w_aligned * v_valid, axis=-1),
            }
            if n >= 3:
                test_funcs_out["ZH"] = full_obs[3]
        elif n == 2:
            test_funcs_out = {"GH": full_obs[0], "BT": full_obs[1]}
        else:
            test_funcs_out = {"GH": full_obs[0], "BT": full_obs[1], "ZH": full_obs[2]}

        n_pts = int(p1_out.shape[0])
        if n_pts >= 2:

            def _eval_codim2_tf_jax(
                label_name: str,
                us_pt: jax.Array,
                pc_pt: jax.Array,
                w_ref: jax.Array | None,
            ) -> jax.Array:
                obs_pt = step_obs(
                    us_pt,
                    pc_pt,
                    tau0,
                    jac_u_aug(us_pt, pc_pt),
                    jac_p_aug(us_pt, pc_pt),
                )
                if bif_type_upper == "LP":
                    w_raw_pt, v_pt, b_vv_pt, tf_zh_pt = obs_pt
                    if label_name == "ZH":
                        return tf_zh_pt
                    w_use = (
                        jnp.where(jnp.dot(w_ref, w_raw_pt) < 0.0, -w_raw_pt, w_raw_pt)
                        if w_ref is not None
                        else w_raw_pt
                    )
                    if label_name == "CP":
                        return 0.5 * jnp.sum(w_use * b_vv_pt)
                    return jnp.sum(w_use * v_pt)
                if label_name == "GH":
                    return obs_pt[0]
                if label_name == "BT":
                    return obs_pt[1]
                return obs_pt[2]

            tf_tol = max(float(opts.newton.tol), 1e-10)
            for label, tf_arr in test_funcs_out.items():
                valid_pairs = (
                    jnp.isfinite(tf_arr[:-1])
                    & jnp.isfinite(tf_arr[1:])
                    & ((tf_arr[:-1] * tf_arr[1:]) < 0.0)
                )
                for k in jnp.nonzero(valid_pairs)[0].tolist():
                    us_k, pc_k = u_state_valid[k], p_cont_valid[k]
                    us_k1, pc_k1 = u_state_valid[k + 1], p_cont_valid[k + 1]
                    w_ref_k = w_aligned[k] if w_aligned is not None else None

                    dz_seg = jnp.concatenate([us_k1, jnp.atleast_1d(pc_k1)]) - jnp.concatenate(
                        [us_k, jnp.atleast_1d(pc_k)]
                    )
                    ds_seg = jnp.linalg.norm(dz_seg)
                    tau_seg = jnp.where(
                        ds_seg > 1e-12, dz_seg / jnp.maximum(ds_seg, 1e-30), tau_scan_valid[k]
                    )

                    def _eval_seg(
                        alpha: jax.Array,
                        *,
                        _label: str = label,
                        _us_k: jax.Array = us_k,
                        _pc_k: jax.Array = pc_k,
                        _us_k1: jax.Array = us_k1,
                        _pc_k1: jax.Array = pc_k1,
                        _tau_seg: jax.Array = tau_seg,
                        _ds_seg: jax.Array = ds_seg,
                        _w_ref_k: jax.Array | None = w_ref_k,
                    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
                        us_guess = (1.0 - alpha) * _us_k + alpha * _us_k1
                        pc_guess = (1.0 - alpha) * _pc_k + alpha * _pc_k1
                        pol = correct_arclength_step(
                            residual_fn=problem_fn,
                            jac_u_fn=jac_u_aug,
                            jac_p_fn=jac_p_aug,
                            u_init=us_guess,
                            p_init=pc_guess,
                            u_prev=_us_k,
                            p_prev=_pc_k,
                            tau_prev=_tau_seg,
                            ds=alpha * _ds_seg,
                            tol=opts.newton.tol,
                            step_tol=opts.newton.step_tol,
                            max_iters=8,
                        )
                        pol_ok = pol.converged & jnp.all(jnp.isfinite(pol.u)) & jnp.isfinite(pol.p)
                        us_cand = jnp.where(pol_ok, pol.u, us_guess)
                        pc_cand = jnp.where(pol_ok, pol.p, pc_guess)
                        f_mid = _eval_codim2_tf_jax(_label, us_cand, pc_cand, _w_ref_k)
                        return f_mid, (us_cand, pc_cand)

                    root_res = safeguarded_secant_root(
                        _eval_seg,
                        tf_arr[k],
                        tf_arr[k + 1],
                        (0.5 * (us_k + us_k1), 0.5 * (pc_k + pc_k1)),
                        tol=tf_tol,
                        max_iters=15,
                    )
                    us_best, pc_best = root_res.payload
                    u_pt, p1_pt, p2_pt, _ = _unpack_step(us_best, pc_best)
                    singularities.append(
                        Codim2Point(
                            bif_type=label,
                            u=u_pt,
                            p1=float(p1_pt),
                            p2=float(p2_pt),
                            step=k + 1,
                            details={
                                "test_func_before": float(tf_arr[k]),
                                "test_func_after": float(tf_arr[k + 1]),
                                "test_func_refined": float(root_res.f_val),
                                "refinement_iters": int(root_res.iterations),
                            },
                        )
                    )
        singularities.sort(key=lambda pt: pt.step)

    return Codim2Result(
        p1=p1_out,
        p2=p2_out,
        u=u_traj,
        tangents=tau_traj,
        bif_type=bif_type_upper,
        singularities=tuple(singularities),
        test_funcs=test_funcs_out,
    )
