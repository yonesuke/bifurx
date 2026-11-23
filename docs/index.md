# bifurx

**Numerical continuation and bifurcation analysis in JAX.**  
*A modern reimagining of AUTO-07P for the differentiable scientific computing era.*

---

## Overview

`bifurx` is a modern Python package for **numerical continuation** and **bifurcation analysis** of nonlinear dynamical systems and algebraic equilibrium problems $F(u, p) = 0$, built natively on [JAX](https://github.com/jax-ml/jax).

For decades, software packages such as **AUTO-07P**, **MATCONT**, and **COCO** have served as the foundational workhorses of nonlinear dynamics and bifurcation theory. While remarkably robust, classical packages often require manual Jacobian derivations or finite-difference approximations, and they lack seamless integration into modern machine learning workflows.

`bifurx` reimagines these classical continuation algorithms using modern differentiable programming:
- **Automatic Differentiation**: Exact state and parameter Jacobians ($\nabla_u F$, $\nabla_p F$) via `jax.jacobian`—no hand-derived matrices or numerical errors.
- **Directional Higher-Order Derivatives**: High-order directional derivatives for branch switching computed via nested JAX Jacobian-vector products (`jax.jvp`).
- **Hardware Acceleration & JIT Compilation**: Core bordered solvers and residuals compile to high-performance machine code via XLA.
- **Pythonic & Composable API**: Pure functional design built on native JAX PyTrees, compatible with `jax.jit`, `jax.vmap`, and modern numerical packages.

---

## Key Features

- **Standalone Pure-JAX Homotopy Core (`bifurx.homotopy`)**:
  - Keller pseudo-arclength and natural parameter continuation compiled end-to-end with `jax.lax.scan`.
  - Euler tangent and 3rd-order **Padé (2,1)** rational predictors with scale-invariant trust regions via nested `jax.jvp`.
  - Scale-invariant weighted norms, affine-invariant Kantorovich Newton corrector, and theory-grounded adaptive step-size control (Timme, 2020).
- **Bordered Linear Solvers & Pluggable Operators**:
  - Block LU (Schur complement), Direct LU with SVD/least-squares fallback, matrix-free GMRES via `jax.jvp`, and two-stage de Boor–Swartz + $\mathcal{O}(N n^3)$ orthogonal cyclic block-QR condensation for collocation BVPs.
- **Codimension-1 & Codimension-2 Bifurcation Analysis**:
  - **Limit Points (LP)**, **Branch Points (BP)** (via bordered Jacobian determinant), and **Hopf Bifurcations (HB)** (with first Lyapunov coefficient $l_1$).
  - **Branch Switching** via the Algebraic Bifurcation Equation (ABE) using nested `jax.jvp`.
  - **Two-Parameter Continuation (`bifurx.codim2`)**: Tracks LP and HB curves and detects Cusp (CP), Bogdanov-Takens (BT), Generalized Hopf (GH), and Zero-Hopf (ZH) singularities.
- **Periodic Orbits, Floquet Stability & iPRCs (`bifurx.bvp`, `bifurx.prc`)**:
  - Superconvergent Gauss-Legendre orthogonal collocation with adaptive mesh redistribution, $\mathcal{O}(N n^3)$ cyclic block-QR condensation, and overflow-free Periodic Real Schur Floquet analysis (LPC, PD, NS).
  - Block-condensed, non-iterative linear adjoint collocation BVP for infinitesimal Phase Response Curves (iPRCs) and their parameter continuation.

---

## Installation

Install `bifurx` using `pip` or `uv`:

```bash
# Clone and install locally
git clone https://github.com/ryosuke-yoneda/bifurx.git
cd bifurx
pip install -e .
```

Or with dependencies for documentation and testing:

```bash
uv sync --all-groups
```

---

## Quick Start

Here is a minimal example demonstrating branch continuation and branch-point detection on a pitchfork bifurcation $F(u, p) = p u - u^3 = 0$:

```python
import jax
import jax.numpy as jnp

# Enable 64-bit precision for numerical continuation
jax.config.update("jax_enable_x64", True)

from bifurx import BifurcationProblem, continuation, switch_branch


# 1. Define nonlinear equilibrium problem F(u, p) = 0
def f_pitchfork(u, p):
    x = u[0]
    return jnp.array([p * x - x**3])


prob = BifurcationProblem(f_pitchfork, u0=[0.0], p0=-0.5)

# 2. Track trivial branch u = 0
res_trivial = continuation(
    problem=prob,
    ds=0.04,
    max_steps=35,
    param_bounds=(-0.6, 0.6),
    branch_name="Trivial Branch",
)

# 3. Detect branch point
bp = res_trivial.get_bifurcations("BP")[0]
print(f"Detected Branch Point at p = {bp.p:.4f}, u = {bp.u[0]:.4f}")

# 4. Switch onto non-trivial branch u = +sqrt(p)
u_up, p_up, tau_up = switch_branch(prob, bp.u, bp.p, bp.tangent, ds=0.04, branch_direction=+1.0)
res_up = continuation(
    problem=prob,
    u0=u_up,
    p0=p_up,
    initial_tangent=tau_up,
    ds=0.03,
    max_steps=30,
    branch_name="Upper Branch",
)

# 5. Extract trajectory results
print(f"Upper branch computed {res_up.num_points} points.")
```

---

## Documentation Structure

- **Theory**:
  - [Homotopy Continuation: General Theory & Path-Tracking Core](theory/homotopy_continuation.md): General theory of `bifurx.homotopy`—Keller pseudo-arclength, scale-invariant weighted norms, Padé (2,1) predictors, bordered/operator solvers, affine-invariant Kantorovich Newton, and Timme (2020) step-size control.
  - [Applying Homotopy Continuation to Dynamical Systems](theory/dynamical_systems.md): How equilibria, codimension-2 curves, Gauss-Legendre collocation periodic orbits, and iPRCs map onto the homotopy core.
  - [Structured Linear Algebra for Periodic-Orbit Collocation BVPs](theory/collocation_condensation.md): Two-stage de Boor–Swartz block condensation and $\mathcal{O}(N n^3)$ orthogonal cyclic block-QR elimination (Wright 1992 / Ascher–Mattheij–Russell).
  - [Periodic Real Schur Decomposition for Stiff Floquet Analysis](theory/periodic_schur_floquet.md): Overflow/underflow-free Bojanczyk–Golub–Van Dooren Periodic Real Schur decomposition for Floquet multipliers, exponents, and cycle bifurcations on stiff orbits.
  - [Bifurcation Theory & Branch Switching](theory/bifurcations.md): Mathematical formulations of Fold, Hopf, and Branch Points, the Moore-Spence & Hopf defining systems, first Lyapunov coefficient, and the Algebraic Bifurcation Equation.
  - [Phase Reduction Theory & iPRC](theory/prc_theory.md): Isochrons, block-condensed linear adjoint collocation BVP, Malkin weak-coupling theory, and iPRC parameter continuation.
  - [CR3BP & KAM Torus Theory](theory/cr3bp_kam.md): Hamiltonian continuation, Lyapunov Center Theorem, symplectic Floquet theory, and KAM invariant tori.
- **Interactive Tutorials & Demos**:
  - [Demo 1: Supercritical Pitchfork Bifurcation](../notebooks/01_pitchfork_bifurcation.ipynb): Step-by-step branch switching and diagram plotting.
  - [Demo 2: AUTO-07P Demo `ab` Chemical Reactor](../notebooks/02_demo_ab_equilibrium.ipynb): S-shaped curve, fold detection, Hopf bifurcation, and eigenvalue spectrum analysis.
  - [Demo 3: Periodic Orbits in Demo `ab`](../notebooks/03_demo_ab_periodic.ipynb): Hopf initialization, Gauss-Legendre collocation, and Floquet stability.
  - [Demo 4: Codimension-2 Continuation in Demo `ab`](../notebooks/04_demo_ab_moore_spence.ipynb): Two-parameter Fold/Hopf curves and codim-2 singularities.
  - [Demo 5: CR3BP Lyapunov Orbits](../notebooks/05_demo_r3b_lyapunov.ipynb): Hamiltonian periodic orbit families and symplectic Floquet multipliers.
  - [Demo 6: Infinitesimal Phase Response Curves](../notebooks/06_phase_response_curve.ipynb): Adjoint BVP iPRC computation and continuation.
  - [Demo 7: Bratu BVP](../notebooks/07_demo_bratu_bvp.ipynb): Nonlinear boundary value problem continuation.
