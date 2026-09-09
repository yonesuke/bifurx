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
- **Pythonic & Composable API**: Pure functional design compatible with `equinox` PyTrees and modern numerical packages.

---

## Key Features

- **Keller Pseudo-Arclength Continuation**:
  - Traverses turning points (folds/limit points) where natural parameter continuation fails ($\partial u / \partial p \to \infty$).
  - Dynamic adaptive step size control based on Newton convergence rates.
- **Robust Bordered Linear System Solvers**:
  - Fast Block LU elimination (Sherman-Morrison-Woodbury formula) for regular Jacobians.
  - Direct augmented matrix factorization with SVD least-squares fallback for singular bordered systems.
- **Codimension-1 Bifurcation Detection & Refinement**:
  - **Limit Points / Folds (LP)**: Detected via tangent parameter velocity sign changes $\dot{p}=0$ and refined using the Moore-Spence augmented system.
  - **Branch Points (BP)**: Detected via determinant zero-crossings $\det(\nabla_u F) = 0$ and refined via bisection/Brent root-finding.
  - **Hopf Bifurcations (HB)**: Monitored via bialternate matrix products $2 \nabla_u F \odot I$ and imaginary eigenvalue transitions.
- **Branch Switching via Algebraic Bifurcation Equation (ABE)**:
  - Solves the quadratic ABE using directional second derivatives evaluated via nested `jax.jvp`, switching onto intersecting branches without heuristic perturbation.
- **Built-in Visualization**:
  - Integrated publication-quality bifurcation diagrams via `plot_diagram` with automatic stability distinction (solid for stable, dashed for unstable) and annotated bifurcation markers.

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

from bifurx import BifurcationProblem, continuation, plot_diagram, switch_branch

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
u_up, p_up, tau_up = switch_branch(
    prob, bp.u, bp.p, bp.tangent, ds=0.04, branch_direction=+1.0
)
res_up = continuation(
    problem=prob,
    u0=u_up,
    p0=p_up,
    initial_tangent=tau_up,
    ds=0.03,
    max_steps=30,
    branch_name="Upper Branch",
)

# 5. Plot bifurcation diagram
fig, ax = plot_diagram([res_trivial, res_up], title="Pitchfork Bifurcation")
```

---

## Documentation Structure

- **Theory**:
  - [Keller Pseudo-Arclength Continuation](theory/keller_arclength.md): Complete derivation of the bordered Jacobian equations and block solvers.
  - [Bifurcation Theory & Detection](theory/bifurcations.md): Mathematical formulations of Fold, Hopf, and Branch Points, the Moore-Spence system, and the Algebraic Bifurcation Equation.
- **Interactive Tutorials & Demos**:
  - [Demo 1: Supercritical Pitchfork Bifurcation](../notebooks/01_pitchfork_bifurcation.ipynb): Step-by-step branch switching and diagram plotting.
  - [Demo 2: AUTO-07P Demo `ab` Chemical Reactor](../notebooks/02_demo_ab_equilibrium.ipynb): S-shaped curve, fold detection, Hopf bifurcation, and eigenvalue spectrum analysis.
