from __future__ import annotations

import jax
import jax.numpy as jnp

from bifurx.problem import BifurcationProblem
from bifurx.solvers.newton import bordered_newton_solve


def compute_singular_direction(
    problem: BifurcationProblem,
    u_star: jax.Array,
    p_star: float | jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the right null vector phi, left null vector psi, and particular vector v0.

    Parameters
    ----------
    problem : BifurcationProblem
        Nonlinear system F(u, p) = 0.
    u_star : jax.Array
        State at the branch point.
    p_star : float | jax.Array
        Parameter at the branch point.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        (phi, psi, v0) where:
        - phi is the right null vector: Ju * phi = 0
        - psi is the left null vector: psi^T * Ju = 0
        - v0 satisfies Ju * v0 = -Jp with <phi, v0> = 0
    """
    Ju = problem.jacobian_u(u_star, p_star)
    Jp = problem.jacobian_p(u_star, p_star)

    u_svd, _, vh = jnp.linalg.svd(Ju)
    phi = vh[-1, :]
    psi = u_svd[:, -1]

    v0_raw = jnp.linalg.lstsq(Ju, -Jp)[0].ravel()
    v0 = v0_raw - jnp.dot(v0_raw, phi) * phi

    return phi, psi, v0


def compute_bifurcation_tangents(
    problem: BifurcationProblem,
    u_star: jax.Array,
    p_star: float | jax.Array,
    incoming_tangent: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Solve the Algebraic Bifurcation Equation (ABE) using second directional derivatives in pure JAX.

    At a simple branch point (BP), any tangent vector (dot{u}, dot{p}) satisfies:
        dot{u} = alpha * phi + dot{p} * v0

    Differentiating F(u, p) = 0 twice and projecting onto the adjoint null vector psi
    yields the quadratic equation:
        c11 * alpha^2 + 2 * c12 * alpha * dot{p} + c22 * dot{p}^2 = 0

    where directional derivatives are computed via nested JAX jvp calls.

    Parameters
    ----------
    problem : BifurcationProblem
        The nonlinear system.
    u_star : jax.Array
        State at the branch point.
    p_star : float | jax.Array
        Parameter at the branch point.
    incoming_tangent : jax.Array
        Tangent of the current incoming branch [tau_u, tau_p].

    Returns
    -------
    tuple[jax.Array, jax.Array]
        (tau_incoming, tau_secondary) unit tangent vectors for both intersecting branches.
    """
    n = problem.dim
    u_s = jnp.asarray(u_star, dtype=jnp.float64)
    p_s = jnp.squeeze(jnp.asarray(p_star, dtype=jnp.float64))
    phi, psi, v0 = compute_singular_direction(problem, u_s, p_s)

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

        return jax.jvp(dir1, (u_s, p_s), (w_u, w_p))[1]

    c11 = jnp.dot(psi, d2F_directional(xi1, xi1))
    c12 = jnp.dot(psi, d2F_directional(xi1, xi2))
    c22 = jnp.dot(psi, d2F_directional(xi2, xi2))

    tau_in = jnp.asarray(incoming_tangent, dtype=jnp.float64)
    pdot_in = tau_in[-1]
    udot_in = tau_in[:n]
    alpha_in = jnp.dot(phi, udot_in - pdot_in * v0)

    # Case 1: |c11| > 1e-8 -> quadratic roots r = alpha / pdot
    c11_safe = jnp.where(jnp.abs(c11) > 1e-8, c11, one_p)
    disc = jnp.maximum(0.0, c12**2 - c11 * c22)
    sqrt_disc = jnp.sqrt(disc)
    r1 = (-c12 + sqrt_disc) / c11_safe
    r2 = (-c12 - sqrt_disc) / c11_safe

    pdot_safe = jnp.where(jnp.abs(pdot_in) > 1e-6, pdot_in, one_p)
    r_in = alpha_in / pdot_safe
    r_sec = jnp.where(jnp.abs(r1 - r_in) < jnp.abs(r2 - r_in), r2, r1)
    alpha_case1 = jnp.where(jnp.abs(pdot_in) > 1e-6, r_sec, r1)
    pdot_case1 = one_p

    # Case 2: |c11| <= 1e-8 and |c12| > 1e-8 -> pdot * (2 * c12 * alpha + c22 * pdot) = 0
    alpha_case2 = jnp.where(jnp.abs(pdot_in) < 1e-4, -c22, one_p)
    pdot_case2 = jnp.where(jnp.abs(pdot_in) < 1e-4, 2.0 * c12, zero_p)

    # Select active case
    has_c11 = jnp.abs(c11) > 1e-8
    has_c12 = jnp.abs(c12) > 1e-8
    alpha_sec = jnp.where(has_c11, alpha_case1, jnp.where(has_c12, alpha_case2, one_p))
    pdot_sec = jnp.where(has_c11, pdot_case1, jnp.where(has_c12, pdot_case2, zero_p))

    udot_sec = alpha_sec * phi + pdot_sec * v0
    tau_sec_vec = jnp.concatenate([udot_sec, jnp.atleast_1d(pdot_sec)])
    tau_sec_norm = jnp.linalg.norm(tau_sec_vec)
    phi_fallback = jnp.concatenate([phi, jnp.zeros(1, dtype=jnp.float64)])
    phi_fallback = phi_fallback / jnp.linalg.norm(phi_fallback)

    tau_sec_vec = jnp.where(tau_sec_norm > 1e-12, tau_sec_vec / tau_sec_norm, phi_fallback)

    overlap = jnp.dot(tau_sec_vec, tau_in)
    tau_sec_vec = jnp.where(jnp.abs(overlap) > 0.999, phi_fallback, tau_sec_vec)

    return tau_in, tau_sec_vec


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
    """Switch to the intersecting branch at a simple branch point (BP) in pure JAX.

    Computes the secondary branch tangent via ABE, steps off the branch point by
    `branch_direction * ds * tau_sec`, and corrects the predictor using bordered Newton.

    Parameters
    ----------
    problem : BifurcationProblem
        The nonlinear system F(u, p) = 0.
    bp_u : jax.Array
        State at the detected branch point.
    bp_p : float | jax.Array
        Parameter at the detected branch point.
    incoming_tangent : jax.Array
        Tangent of the incoming branch.
    ds : float | jax.Array, default=0.05
        Initial step size along the new branch.
    branch_direction : float | jax.Array, default=1.0
        Direction along the intersecting branch (+1.0 or -1.0).
    tol : float, default=1e-8
        Convergence tolerance for the corrector.
    max_iters : int, default=15
        Maximum Newton corrector iterations.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        (u_start, p_start, tau_start) initial point and tangent for continuing the new branch.
    """
    n = problem.dim
    bp_u_arr = jnp.asarray(bp_u, dtype=jnp.float64)
    bp_p_arr = jnp.squeeze(jnp.asarray(bp_p, dtype=jnp.float64))
    ds_arr = jnp.squeeze(jnp.asarray(ds, dtype=jnp.float64))

    _, tau_sec = compute_bifurcation_tangents(problem, bp_u_arr, bp_p_arr, incoming_tangent)

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
