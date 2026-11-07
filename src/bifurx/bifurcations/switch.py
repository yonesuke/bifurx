from __future__ import annotations

import jax
import jax.numpy as jnp

from bifurx.problem import BifurcationProblem
from bifurx.solvers.newton import bordered_newton_solve


def _compute_secondary_tangent(
    problem: BifurcationProblem,
    u_star: jax.Array,
    p_star: jax.Array,
    incoming_tangent: jax.Array,
) -> jax.Array:
    """Solve the Algebraic Bifurcation Equation (ABE) for the secondary branch unit tangent."""
    n = problem.dim
    Ju = problem.jacobian_u(u_star, p_star)
    Jp = problem.jacobian_p(u_star, p_star)

    u_svd, _, vh = jnp.linalg.svd(Ju)
    phi = vh[-1, :]
    psi = u_svd[:, -1]

    v0_raw = jnp.linalg.lstsq(Ju, -Jp)[0].ravel()
    v0 = v0_raw - jnp.dot(v0_raw, phi) * phi

    zero_p = jnp.array(0.0, dtype=jnp.float64)
    one_p = jnp.array(1.0, dtype=jnp.float64)
    xi1 = (phi, zero_p)
    xi2 = (v0, one_p)

    def d2F_directional(
        v_dir: tuple[jax.Array, jax.Array], w_dir: tuple[jax.Array, jax.Array]
    ) -> jax.Array:
        v_u, v_p = v_dir
        w_u, w_p = w_dir

        def dir1(u_val: jax.Array, p_val: jax.Array) -> jax.Array:
            return jax.jvp(problem.residual, (u_val, p_val), (v_u, v_p))[1]

        return jax.jvp(dir1, (u_star, p_star), (w_u, w_p))[1]

    c11 = jnp.dot(psi, d2F_directional(xi1, xi1))
    c12 = jnp.dot(psi, d2F_directional(xi1, xi2))
    c22 = jnp.dot(psi, d2F_directional(xi2, xi2))

    tau_in = jnp.asarray(incoming_tangent, dtype=jnp.float64)
    pdot_in = tau_in[-1]
    udot_in = tau_in[:n]
    alpha_in = jnp.dot(phi, udot_in - pdot_in * v0)

    c11_safe = jnp.where(jnp.abs(c11) > 1e-8, c11, one_p)
    sqrt_disc = jnp.sqrt(jnp.maximum(0.0, c12**2 - c11 * c22))
    r1 = (-c12 + sqrt_disc) / c11_safe
    r2 = (-c12 - sqrt_disc) / c11_safe

    pdot_safe = jnp.where(jnp.abs(pdot_in) > 1e-6, pdot_in, one_p)
    r_in = alpha_in / pdot_safe
    r_sec = jnp.where(jnp.abs(r1 - r_in) < jnp.abs(r2 - r_in), r2, r1)
    alpha_case1 = jnp.where(jnp.abs(pdot_in) > 1e-6, r_sec, r1)

    alpha_case2 = jnp.where(jnp.abs(pdot_in) < 1e-4, -c22, one_p)
    pdot_case2 = jnp.where(jnp.abs(pdot_in) < 1e-4, 2.0 * c12, zero_p)

    has_c11 = jnp.abs(c11) > 1e-8
    has_c12 = jnp.abs(c12) > 1e-8
    alpha_sec = jnp.where(has_c11, alpha_case1, jnp.where(has_c12, alpha_case2, one_p))
    pdot_sec = jnp.where(has_c11, one_p, jnp.where(has_c12, pdot_case2, zero_p))

    udot_sec = alpha_sec * phi + pdot_sec * v0
    tau_sec_vec = jnp.concatenate([udot_sec, jnp.atleast_1d(pdot_sec)])
    tau_sec_norm = jnp.linalg.norm(tau_sec_vec)
    phi_fallback = jnp.concatenate([phi, jnp.zeros(1, dtype=jnp.float64)])
    phi_fallback = phi_fallback / jnp.linalg.norm(phi_fallback)

    tau_sec_vec = jnp.where(tau_sec_norm > 1e-12, tau_sec_vec / tau_sec_norm, phi_fallback)
    return jnp.where(jnp.abs(jnp.dot(tau_sec_vec, tau_in)) > 0.999, phi_fallback, tau_sec_vec)


def switch_branch(
    problem: BifurcationProblem,
    bp_u: jax.Array,
    bp_p: float | jax.Array,
    incoming_tangent: jax.Array,
    ds: float | jax.Array = 0.05,
    branch_direction: float | jax.Array = 1.0,
    tol: float = 1e-8,
    max_iters: int = 15,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Switch to the intersecting branch at a simple branch point (BP) in pure JAX."""
    n = problem.dim
    bp_u_arr = jnp.asarray(bp_u, dtype=jnp.float64)
    bp_p_arr = jnp.squeeze(jnp.asarray(bp_p, dtype=jnp.float64))
    ds_arr = jnp.squeeze(jnp.asarray(ds, dtype=jnp.float64))

    tau_sec = _compute_secondary_tangent(problem, bp_u_arr, bp_p_arr, incoming_tangent)
    dir_sgn = jnp.where(jnp.asarray(branch_direction, dtype=jnp.float64) < 0, -1.0, 1.0)
    tau_target = dir_sgn * tau_sec

    u_pred = bp_u_arr + ds_arr * tau_target[:n]
    p_pred = bp_p_arr + ds_arr * tau_target[-1]

    res = bordered_newton_solve(
        problem=problem,
        u_init=u_pred,
        p_init=p_pred,
        u_prev=bp_u_arr,
        p_prev=bp_p_arr,
        tau_prev=tau_target,
        ds=ds_arr,
        tol=tol,
        max_iters=max_iters,
    )

    u_out = jnp.where(res.converged, res.u, u_pred)
    p_out = jnp.where(res.converged, res.p, p_pred)
    return u_out, p_out, tau_target
