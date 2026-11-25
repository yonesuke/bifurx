# AGENTS.md

Guidelines for coding agents working on `bifurx`.

## Pure JAX & Control Flow
- Keep numerical core functions pure JAX (`jax.jit` and `jax.vmap` compatible).
- Avoid Python host-sync casts (`float()`, `bool()`, `int()`, `.tolist()`, `.item()`) on traced arrays; use `jnp.where` / `jax.lax.cond` (and `jax.debug.callback` if runtime warnings are required).
- Use `jax.lax.scan`, `jax.lax.while_loop`, or `jax.lax.fori_loop` instead of Python `for` / `while` loops for numerical iterations.
- Enable/preserve 64-bit precision (`jnp.float64`) throughout numerical routines.

## DRY & Reuse Existing Primitives
Before writing new helpers, reuse existing shared abstractions:
- **PyTree option dataclasses**: `bifurx.homotopy.PyTreeOptionsBase`
- **Condition-checked LU with `lstsq` fallback**: `bifurx.homotopy.safe_lu_factor`, `safe_lu_solve`
- **1D bracketed root finding**: `bifurx.homotopy.safeguarded_secant_root`
- **Option resolution**: `resolve_continuation_options`, `resolve_collocation_options` (`bifurx.options`)
- **Collocation derivatives & phase charts**: `CollocationProblem.derivative_at_gauss`, `CollocationProblem.make_phase_chart_fn`
- **Local collocation block condensation (primal / adjoint / Floquet)**: `bifurx.bvp.condensation.condense_local_blocks`

## Hyperparameters & Dead Code
- Avoid implicit magic numbers inside algorithms: expose tunable parameters via option dataclasses/arguments, and define internal numerical constants as named module-level constants.
- Remove dead code immediately: no unused functions/classes, no unreferenced dataclass fields, no no-op compatibility arguments, and no leftover debug/PR-number comments.

## Architecture Constraints
- `bifurx.homotopy` must remain standalone (depend only on Python stdlib, `jax`, and `bifurx.homotopy`).
- All constraints in `tests/test_architecture.py` (independence, no unused definitions, no duplicate definitions) must pass.

## Git Workflow & Verification
- Work on a dedicated branch per task/phase, commit frequently in logical units, and merge locally into `main` once verified.
- Before committing or pushing, always run:
  ```bash
  uv run ruff check
  uv run ruff format --check
  uv run pyrefly check
  uv run pytest
  ```
