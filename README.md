# bifurx

[![CI](https://github.com/yonesuke/bifurx/actions/workflows/ci.yml/badge.svg)](https://github.com/yonesuke/bifurx/actions/workflows/ci.yml)
[![Docs](https://github.com/yonesuke/bifurx/actions/workflows/docs.yml/badge.svg)](https://yonesuke.github.io/bifurx/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Numerical continuation and bifurcation analysis in JAX.**  
*A modern, differentiable reimagining of AUTO-07P.*

📖 **Documentation & Interactive Demos**: [https://yonesuke.github.io/bifurx/](https://yonesuke.github.io/bifurx/)

---

## Key Features

- **Zero Hand-Coded Jacobians**: Fully powered by JAX automatic differentiation (`jax.jacobian`, `jax.jvp`, `jax.vjp`).
- **Pseudo-Arclength Continuation**: Robust Keller continuation with bordered linear solvers (Block LU / Sherman-Morrison-Woodbury + SVD fallback).
- **Automated Bifurcation Detection**: Exact detection of Limit Points (Fold/LP), Branch Points (BP), and Hopf Bifurcations (HB).
- **Branch Switching**: Directional branch switching at algebraic bifurcation points via nested second-order forward-mode directional derivatives (`jax.jvp`).
- **Moore-Spence Augmented Systems**: Machine-precision root refinement and 2-parameter fold curve tracking.
- **High-Performance & Type-Safe**: JIT-compiled with XLA, strictly typed and checked with Meta's **Pyrefly** type checker.

---

## Quickstart

```python
import jax.numpy as jnp
import bifurx as bx


# 1. Define nonlinear problem F(u, p) = 0 (Supercritical pitchfork)
def pitchfork(u, p):
    return p * u - u**3


problem = bx.BifurcationProblem(fn=pitchfork, u0=jnp.array([0.0]), p0=-1.0)

# 2. Run pseudo-arclength continuation
res = bx.continuation(problem, ds=0.05, max_steps=60, p_bounds=(-1.0, 2.0))

print(f"Computed {len(res.points)} points. Detected {len(res.bifurcations)} bifurcations.")
for bif in res.bifurcations:
    print(f"Found {bif.kind.value} at p = {bif.p:.4f}, u = {bif.u[0]:.4f}")
```

---

## Interactive Demo Notebooks

Check out the interactive notebooks directly in [`notebooks/`](notebooks/):
1. [`notebooks/01_pitchfork_bifurcation.ipynb`](notebooks/01_pitchfork_bifurcation.ipynb) — Pitchfork bifurcation, BP detection, and branch switching
2. [`notebooks/02_demo_ab_equilibrium.ipynb`](notebooks/02_demo_ab_equilibrium.ipynb) — AUTO-07P demo `ab` run 1 (Exothermic CSTR S-curve & Fold/Hopf detection)
3. [`notebooks/03_demo_ab_periodic.ipynb`](notebooks/03_demo_ab_periodic.ipynb) — AUTO-07P demo `ab` run 2 (Periodic orbit continuation via Gauss collocation BVP & homoclinic period blowup)
4. [`notebooks/04_demo_ab_moore_spence.ipynb`](notebooks/04_demo_ab_moore_spence.ipynb) — AUTO-07P demo `ab` run 3 (2-Parameter Codim-2 Moore-Spence continuation, Cusp point, and BT singularity)
5. [`notebooks/05_demo_r3b_lyapunov.ipynb`](notebooks/05_demo_r3b_lyapunov.ipynb) — **Circular Restricted Three-Body Problem (CR3BP)**, Lyapunov orbit family, symplectic Floquet multipliers, and **KAM Torus Theory** in astrodynamics (JWST / SOHO)
6. [`notebooks/06_phase_response_curve.ipynb`](notebooks/06_phase_response_curve.ipynb) — **Modern PRAX**: Infinitesimal Phase Response Curves (iPRC) via superconvergent adjoint BVP, parameter continuation, Type I / II excitability, and Arnold tongues
7. [`notebooks/07_demo_bratu_bvp.ipynb`](notebooks/07_demo_bratu_bvp.ipynb) — Gelfand-Bratu nonlinear elliptic BVP (Thermal runaway and turning point)

---

## Development

```bash
# Clone repository
git clone https://github.com/yonesuke/bifurx.git
cd bifurx

# Install dependencies and sync environment with uv
uv sync --group dev --group test --group docs

# Run test suite
uv run pytest

# Run type checks with Pyrefly
uv run pyrefly check

# Build documentation locally
cd docs && uv run jupyter-book build --html
```
