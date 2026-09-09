from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from bifurx.problem import BifurcationProblem
from bifurx.solvers.newton import bordered_newton_solve


def compute_singular_direction(
    problem: BifurcationProblem,
    u_star: Array,
    p_star: float,
) -> tuple[Array, Array, Array]:
    """Compute the right null vector phi, left null vector psi, and particular vector v0.

    Parameters
    ----------
    problem : BifurcationProblem
        Nonlinear system F(u, p) = 0.
    u_star : Array
        State at the branch point.
    p_star : float
        Parameter at the branch point.

    Returns
    -------
    tuple[Array, Array, Array]
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

    # Particular solution v0 for Ju * v0 = -Jp
    v0_raw = jnp.linalg.lstsq(Ju, -Jp)[0].ravel()
    # Ensure v0 is orthogonal to phi
    v0 = v0_raw - jnp.dot(v0_raw, phi) * phi

    return phi, psi, v0


def compute_bifurcation_tangents(
    problem: BifurcationProblem,
    u_star: Array,
    p_star: float,
    incoming_tangent: Array,
) -> tuple[Array, Array]:
    """Solve the Algebraic Bifurcation Equation (ABE) using second directional derivatives.

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
    u_star : Array
        State at the branch point.
    p_star : float
        Parameter at the branch point.
    incoming_tangent : Array
        Tangent of the current incoming branch [tau_u, tau_p].

    Returns
    -------
    tuple[Array, Array]
        (tau_incoming, tau_secondary) unit tangent vectors for both intersecting branches.
    """
    n = problem.dim
    phi, psi, v0 = compute_singular_direction(problem, u_star, p_star)

    xi1 = (phi, 0.0)
    xi2 = (v0, 1.0)

    # Directional second derivative via nested jvp
    def d2F_directional(v_dir: tuple[Array, float], w_dir: tuple[Array, float]) -> Array:
        v_u, v_p = v_dir
        w_u, w_p = w_dir

        def dir1(u_val: Array, p_val: float) -> Array:
            return jax.jvp(problem.residual, (u_val, p_val), (v_u, v_p))[1]

        return jax.jvp(dir1, (u_star, p_star), (w_u, w_p))[1]

    c11 = float(jnp.dot(psi, d2F_directional(xi1, xi1)))
    c12 = float(jnp.dot(psi, d2F_directional(xi1, xi2)))
    c22 = float(jnp.dot(psi, d2F_directional(xi2, xi2)))

    # Deconstruct incoming tangent in basis (xi1, xi2)
    tau_in_np = np.asarray(incoming_tangent)
    pdot_in = float(tau_in_np[-1])
    udot_in = tau_in_np[:n]
    alpha_in = float(np.dot(np.asarray(phi), udot_in - pdot_in * np.asarray(v0)))

    # Solve the quadratic ABE: c11 * alpha^2 + 2 * c12 * alpha * p_dot + c22 * p_dot^2 = 0
    if abs(c11) > 1e-8:
        # Divide by p_dot^2 -> c11 * r^2 + 2 * c12 * r + c22 = 0 with r = alpha / p_dot
        D = max(0.0, c12**2 - c11 * c22)
        r1 = (-c12 + np.sqrt(D)) / c11
        r2 = (-c12 - np.sqrt(D)) / c11

        if abs(pdot_in) > 1e-6:
            r_in = alpha_in / pdot_in
            r_sec = r2 if abs(r1 - r_in) < abs(r2 - r_in) else r1
            alpha_sec, pdot_sec = r_sec, 1.0
        else:
            # Incoming branch had pdot ~ 0
            alpha_sec, pdot_sec = r1, 1.0
    elif abs(c12) > 1e-8:
        # p_dot * (2 * c12 * alpha + c22 * p_dot) = 0
        # Root A: p_dot = 0, alpha = 1
        # Root B: p_dot = 2 * c12, alpha = -c22
        if abs(pdot_in) < 1e-4:
            alpha_sec, pdot_sec = -c22, 2.0 * c12
        else:
            alpha_sec, pdot_sec = 1.0, 0.0
    else:
        # Fallback: singular direction orthogonal to parameter axis
        alpha_sec, pdot_sec = 1.0, 0.0

    udot_sec = alpha_sec * np.asarray(phi) + pdot_sec * np.asarray(v0)
    tau_sec_vec = np.concatenate([udot_sec, [pdot_sec]])
    tau_sec_norm = np.linalg.norm(tau_sec_vec)
    if tau_sec_norm > 1e-12:
        tau_sec_vec = tau_sec_vec / tau_sec_norm
    else:
        tau_sec_vec = np.concatenate([np.asarray(phi), [0.0]])

    # Make secondary tangent linearly independent / orthogonalized to incoming tangent
    overlap = float(np.dot(tau_sec_vec, tau_in_np))
    if abs(overlap) > 0.999:
        # Fallback orthogonal perturbation along singular vector phi
        tau_sec_vec = np.concatenate([np.asarray(phi), [0.0]])
        tau_sec_vec = tau_sec_vec / np.linalg.norm(tau_sec_vec)

    return jnp.asarray(incoming_tangent), jnp.asarray(tau_sec_vec)


def switch_branch(
    problem: BifurcationProblem,
    bp_u: Array,
    bp_p: float,
    incoming_tangent: Array,
    ds: float = 0.05,
    branch_direction: float = 1.0,
    tol: float = 1e-8,
    max_iters: int = 15,
) -> tuple[Array, float, Array]:
    """Switch to the intersecting branch at a simple branch point (BP).

    Computes the secondary branch tangent via ABE, steps off the branch point by
    `branch_direction * ds * tau_sec`, and corrects the predictor using bordered Newton.

    Parameters
    ----------
    problem : BifurcationProblem
        The nonlinear system F(u, p) = 0.
    bp_u : Array
        State at the detected branch point.
    bp_p : float
        Parameter at the detected branch point.
    incoming_tangent : Array
        Tangent of the incoming branch.
    ds : float, default=0.05
        Initial step size along the new branch.
    branch_direction : float, default=1.0
        Direction along the intersecting branch (+1.0 or -1.0).
    tol : float, default=1e-8
        Convergence tolerance for the corrector.
    max_iters : int, default=15
        Maximum Newton corrector iterations.

    Returns
    -------
    tuple[Array, float, Array]
        (u_start, p_start, tau_start) initial point and tangent for continuing the new branch.
    """
    n = problem.dim
    _, tau_sec = compute_bifurcation_tangents(problem, bp_u, bp_p, incoming_tangent)

    # Orient in desired direction
    tau_target = float(np.sign(branch_direction)) * tau_sec

    # Predictor step onto new branch
    u_pred = bp_u + ds * tau_target[:n]
    p_pred = bp_p + ds * float(tau_target[-1])

    res = bordered_newton_solve(
        problem=problem,
        u_init=u_pred,
        p_init=p_pred,
        u_prev=bp_u,
        p_prev=bp_p,
        tau_prev=tau_target,
        ds=ds,
        tol=tol,
        max_iters=max_iters,
    )

    if res.converged:
        return res.u, res.p, tau_target

    # Fallback to predictor if corrector didn't converge within max_iters
    return u_pred, p_pred, tau_target
