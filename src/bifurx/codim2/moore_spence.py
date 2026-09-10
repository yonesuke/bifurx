from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array


class Codim2Result(NamedTuple):
    """Result of a Codimension-2 bifurcation curve continuation."""

    p1: Array
    p2: Array
    u: Array  # shape (num_points, dim)
    tangents: Array  # shape (num_points, dim_total)
    bif_type: str  # "LP" or "HB"

    @property
    def num_points(self) -> int:
        """Number of tracked bifurcation points along the curve."""
        return int(len(self.p1))


def continuation_codim2(
    fn: Callable[..., Array],
    bif_type: str,
    u_init: Array | np.ndarray,
    p1_init: float,
    p2_init: float,
    ds: float = 0.02,
    ds_min: float = 1e-4,
    ds_max: float = 0.1,
    max_steps: int = 120,
    direction: float = 1.0,
    param_orient_index: int = 1,  # 0 for orienting along p1, 1 for p2
    p1_bounds: tuple[float, float] | None = None,
    p2_bounds: tuple[float, float] | None = None,
    tol: float = 1e-8,
    max_newton_iters: int = 15,
    **kwargs: Any,
) -> Codim2Result:
    """Track a Codimension-2 bifurcation curve (Fold LP or Hopf HB) in two parameters (p1, p2).

    Parameters
    ----------
    fn : Callable[..., Array]
        Residual function f(u, p1, p2, **kwargs) returning 1D array of residuals.
    bif_type : str
        Type of bifurcation to track: "LP" (Fold / Saddle-node) or "HB" (Hopf).
    u_init : Array
        Initial state vector at the bifurcation point.
    p1_init : float
        Initial value of first parameter.
    p2_init : float
        Initial value of second parameter.
    ds : float, default=0.02
        Initial pseudo-arclength continuation step size.
    ds_min : float, default=1e-4
        Minimum allowed step size.
    ds_max : float, default=0.1
        Maximum allowed step size.
    max_steps : int, default=120
        Maximum continuation steps.
    direction : float, default=1.0
        Continuation direction (+1.0 or -1.0).
    param_orient_index : int, default=1
        Which parameter to orient initial tangent along (0 for p1, 1 for p2).
    p1_bounds : tuple[float, float] | None, optional
        (p1_min, p1_max) termination bounds.
    p2_bounds : tuple[float, float] | None, optional
        (p2_min, p2_max) termination bounds.
    tol : float, default=1e-8
        Newton corrector convergence tolerance.
    max_newton_iters : int, default=15
        Maximum Newton iterations per step.
    **kwargs : Any
        Additional keyword arguments passed to `fn`.

    Returns
    -------
    Codim2Result
        Result containing (p1, p2, u, tangents, bif_type).
    """
    bif_type_upper = bif_type.upper()
    u0_np = np.asarray(u_init, dtype=np.float64)
    n = len(u0_np)

    # Autodiff Jacobian df/du
    jac_u_fn = jax.jit(jax.jacobian(lambda u, p1, p2: fn(u, p1, p2, **kwargs), argnums=0))

    if bif_type_upper == "LP":
        # Moore-Spence Fold augmented system:
        # Unknowns: z = [u (n), p1 (1), p2 (1), v (n)]  (dim = 2n + 2)
        # Equations:
        #   f(u, p1, p2) = 0        (n)
        #   df/du(u, p1, p2) * v = 0 (n)
        #   l^T * v - 1 = 0          (1)
        # Total equations = 2n + 1 in 2n + 2 unknowns -> 1D curve!

        Ju0 = np.asarray(jac_u_fn(jnp.asarray(u0_np), p1_init, p2_init))
        _, _, vh = np.linalg.svd(Ju0)
        v0 = vh[-1, :]
        l_norm = jnp.asarray(v0 / np.linalg.norm(v0))

        @jax.jit
        def res_fn(z: Array) -> Array:
            u_v = z[:n]
            p1_v = z[n]
            p2_v = z[n + 1]
            v_v = z[n + 2 :]

            f_val = fn(u_v, p1_v, p2_v, **kwargs)
            Ju = jac_u_fn(u_v, p1_v, p2_v)
            v_res = Ju @ v_v
            norm_res = jnp.dot(l_norm, v_v) - 1.0
            return jnp.concatenate([f_val, v_res, jnp.atleast_1d(norm_res)])

        dim_z = 2 * n + 2
        z_cur = np.concatenate([u0_np, [p1_init, p2_init], v0])
        idx_param = n if param_orient_index == 0 else n + 1

    elif bif_type_upper == "HB":
        if n == 2:
            # 2D systems: trace(df/du) = 0
            # Unknowns: w = [u (2), p1 (1), p2 (1)] (dim = 4)
            # Equations:
            #   f(u, p1, p2) = 0 (2)
            #   trace(df/du) = 0 (1)
            # Total equations = 3 in 4 unknowns!
            @jax.jit
            def res_fn(w: Array) -> Array:
                u_v = w[:2]
                p1_v = w[2]
                p2_v = w[3]
                f_val = fn(u_v, p1_v, p2_v, **kwargs)
                Ju = jac_u_fn(u_v, p1_v, p2_v)
                tr_val = jnp.trace(Ju)
                return jnp.concatenate([f_val, jnp.atleast_1d(tr_val)])

            dim_z = 4
            z_cur = np.concatenate([u0_np, [p1_init, p2_init]])
            idx_param = 2 if param_orient_index == 0 else 3
        else:
            # General n > 2: Complex eigenvector system
            # df/du * v_R + omega * v_I = 0
            # df/du * v_I - omega * v_R = 0
            # l_R^T * v_R - 1 = 0
            # l_R^T * v_I = 0
            # Unknowns: [u (n), p1 (1), p2 (1), v_R (n), v_I (n), omega (1)] (dim = 3n + 3)
            Ju0 = np.asarray(jac_u_fn(jnp.asarray(u0_np), p1_init, p2_init))
            eigs, vecs = np.linalg.eig(Ju0)
            imag_indices = [i for i, val in enumerate(eigs) if abs(np.imag(val)) > 1e-4]
            if not imag_indices:
                msg = "No complex-conjugate eigenvalues found for Hopf continuation."
                raise ValueError(msg)
            idx_hb = imag_indices[int(np.argmin(np.abs(np.real(eigs[imag_indices]))))]
            omega0 = float(abs(np.imag(eigs[idx_hb])))
            vec0 = vecs[:, idx_hb]
            v_r0 = np.real(vec0)
            v_i0 = np.imag(vec0)
            norm_v = np.sqrt(np.dot(v_r0, v_r0) + np.dot(v_i0, v_i0))
            v_r0 /= norm_v
            v_i0 /= norm_v
            l_R = jnp.asarray(v_r0)

            @jax.jit
            def res_fn(w: Array) -> Array:
                u_v = w[:n]
                p1_v = w[n]
                p2_v = w[n + 1]
                vr_v = w[n + 2 : 2 * n + 2]
                vi_v = w[2 * n + 2 : 3 * n + 2]
                om_v = w[3 * n + 2]

                f_val = fn(u_v, p1_v, p2_v, **kwargs)
                Ju = jac_u_fn(u_v, p1_v, p2_v)
                r_real = Ju @ vr_v + om_v * vi_v
                r_imag = Ju @ vi_v - om_v * vr_v
                n_real = jnp.dot(l_R, vr_v) - 1.0
                n_imag = jnp.dot(l_R, vi_v)

                return jnp.concatenate(
                    [
                        f_val,
                        r_real,
                        r_imag,
                        jnp.atleast_1d(n_real),
                        jnp.atleast_1d(n_imag),
                    ]
                )

            dim_z = 3 * n + 3
            z_cur = np.concatenate([u0_np, [p1_init, p2_init], v_r0, v_i0, [omega0]])
            idx_param = n if param_orient_index == 0 else n + 1
    else:
        msg = f"Unknown bif_type: {bif_type}. Must be 'LP' or 'HB'."
        raise ValueError(msg)

    jac_res_fn = jax.jit(jax.jacobian(res_fn))

    # Initial tangent from SVD of (dim_z - 1) x dim_z Jacobian
    J_init = np.asarray(jac_res_fn(jnp.asarray(z_cur)))
    _, _, vh_init = np.linalg.svd(J_init)
    tau = vh_init[-1, :]
    if tau[idx_param] * direction < 0:
        tau = -tau
    tau /= np.linalg.norm(tau)

    # History tracking
    p1_history = [p1_init]
    p2_history = [p2_init]
    u_history = [np.copy(u0_np)]
    tangents_history = [np.copy(tau)]

    current_ds = ds

    for _step in range(1, max_steps + 1):
        z_prev = np.copy(z_cur)
        tau_prev = np.copy(tau)

        # Predictor
        z = z_prev + current_ds * tau_prev
        converged = False
        iters_taken = 0

        # Bordered Newton Corrector
        for it in range(1, max_newton_iters + 1):
            iters_taken = it
            res_val = np.asarray(res_fn(jnp.asarray(z)))
            arc_res = np.dot(tau_prev, z - z_prev) - current_ds
            G = np.concatenate([res_val, [arc_res]])

            res_norm = np.linalg.norm(G)
            if res_norm < tol:
                converged = True
                break

            J_sub = np.asarray(jac_res_fn(jnp.asarray(z)))
            J_aug = np.zeros((dim_z, dim_z), dtype=np.float64)
            J_aug[:-1, :] = J_sub
            J_aug[-1, :] = tau_prev

            try:
                dz = -np.linalg.solve(J_aug, G)
            except np.linalg.LinAlgError:
                break
            z += dz

        if not converged:
            # Reduce step size and retry once
            current_ds = max(current_ds * 0.5, ds_min)
            z = z_prev + current_ds * tau_prev
            for it in range(1, max_newton_iters + 1):
                iters_taken = it
                res_val = np.asarray(res_fn(jnp.asarray(z)))
                arc_res = np.dot(tau_prev, z - z_prev) - current_ds
                G = np.concatenate([res_val, [arc_res]])
                if np.linalg.norm(G) < tol:
                    converged = True
                    break
                J_sub = np.asarray(jac_res_fn(jnp.asarray(z)))
                J_aug = np.zeros((dim_z, dim_z), dtype=np.float64)
                J_aug[:-1, :] = J_sub
                J_aug[-1, :] = tau_prev
                try:
                    dz = -np.linalg.solve(J_aug, G)
                except np.linalg.LinAlgError:
                    break
                z += dz

            if not converged:
                # Continuation halted
                break

        # AUTO ADPTDS step size adaptation
        if iters_taken <= 4 and current_ds < ds_max:
            current_ds = min(current_ds * 1.15, ds_max)
        elif iters_taken >= 8 and current_ds > ds_min:
            current_ds = max(current_ds * 0.7, ds_min)

        # Tangent update
        J_sub = np.asarray(jac_res_fn(jnp.asarray(z)))
        J_aug = np.zeros((dim_z, dim_z), dtype=np.float64)
        J_aug[:-1, :] = J_sub
        J_aug[-1, :] = tau_prev
        rhs = np.zeros(dim_z, dtype=np.float64)
        rhs[-1] = 1.0

        try:
            tau_new = np.linalg.solve(J_aug, rhs)
            tau_new /= np.linalg.norm(tau_new)
            if np.dot(tau_new, tau_prev) < 0:
                tau_new = -tau_new
            tau = tau_new
        except np.linalg.LinAlgError:
            pass

        z_cur = z
        p1_val = float(z_cur[n])
        p2_val = float(z_cur[n + 1])
        u_val = np.copy(z_cur[:n])

        p1_history.append(p1_val)
        p2_history.append(p2_val)
        u_history.append(u_val)
        tangents_history.append(np.copy(tau))

        # Check termination bounds
        if p1_bounds is not None and not (p1_bounds[0] <= p1_val <= p1_bounds[1]):
            break
        if p2_bounds is not None and not (p2_bounds[0] <= p2_val <= p2_bounds[1]):
            break

    return Codim2Result(
        p1=jnp.asarray(p1_history),
        p2=jnp.asarray(p2_history),
        u=jnp.asarray(np.array(u_history)),
        tangents=jnp.asarray(np.array(tangents_history)),
        bif_type=bif_type_upper,
    )
