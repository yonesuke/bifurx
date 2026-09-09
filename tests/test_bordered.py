import jax
import jax.numpy as jnp

from bifurx.solvers.bordered import (
    solve_bordered_block,
    solve_bordered_direct,
    solve_bordered_system,
)


def test_bordered_nonsingular_block():
    """Test bordered solver when leading block A is non-singular."""
    A = jnp.array([[2.0, 1.0], [1.0, 3.0]], dtype=jnp.float64)
    b = jnp.array([1.0, 2.0], dtype=jnp.float64)
    c = jnp.array([3.0, 1.0], dtype=jnp.float64)
    d = 4.0
    f = jnp.array([5.0, 6.0], dtype=jnp.float64)
    g = 7.0

    # Test Block LU
    sol_block = solve_bordered_block(A, b, c, d, f, g)
    assert jnp.all(jnp.isfinite(sol_block.x))
    assert jnp.isfinite(sol_block.y)

    # Test Direct solve
    sol_direct = solve_bordered_direct(A, b, c, d, f, g)

    # Test Auto solve
    sol_auto = solve_bordered_system(A, b, c, d, f, g, method="auto")

    # Verify equivalence
    assert jnp.allclose(sol_block.x, sol_direct.x, atol=1e-10)
    assert jnp.allclose(sol_block.y, sol_direct.y, atol=1e-10)
    assert jnp.allclose(sol_auto.x, sol_direct.x, atol=1e-10)
    assert jnp.allclose(sol_auto.y, sol_direct.y, atol=1e-10)

    # Verify residual: M * z = r
    M = jnp.block([[A, b.reshape(-1, 1)], [c.reshape(1, -1), jnp.array([[d]])]])
    r = jnp.concatenate([f, jnp.array([g])])
    assert jnp.linalg.norm(M @ sol_auto.z - r) < 1e-10

    # Verify tuple unpacking
    x, y = sol_auto
    assert jnp.allclose(x, sol_auto.x)
    assert jnp.allclose(y, sol_auto.y)


def test_bordered_singular_block():
    """Test bordered solver when block A is singular (rank deficient) but M is non-singular."""
    # A is singular (row 2 = 2 * row 1), but b breaks singularity
    A = jnp.array([[1.0, 2.0], [2.0, 4.0]], dtype=jnp.float64)
    b = jnp.array([1.0, 0.0], dtype=jnp.float64)
    c = jnp.array([1.0, 0.0], dtype=jnp.float64)
    d = 0.0
    f = jnp.array([3.0, 4.0], dtype=jnp.float64)
    g = 1.0

    M = jnp.block([[A, b.reshape(-1, 1)], [c.reshape(1, -1), jnp.array([[d]])]])
    r = jnp.concatenate([f, jnp.array([g])])
    assert abs(float(jnp.linalg.det(A))) < 1e-12
    assert abs(float(jnp.linalg.det(M))) > 1.0

    # Auto solver should detect ill-conditioned A and fallback to direct
    sol_auto = solve_bordered_system(A, b, c, d, f, g, method="auto")
    assert jnp.all(jnp.isfinite(sol_auto.x))
    assert jnp.isfinite(sol_auto.y)

    res_norm = float(jnp.linalg.norm(M @ sol_auto.z - r))
    assert res_norm < 1e-10
    assert jnp.allclose(sol_auto.z, jnp.array([1.0, 0.5, 1.0]), atol=1e-10)


def test_bordered_jit_and_vmap():
    """Test JIT compilation and vectorization (vmap) of bordered solve."""
    jit_solve = jax.jit(solve_bordered_system)

    A = jnp.array([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float64)
    b = jnp.array([1.0, 1.0], dtype=jnp.float64)
    c = jnp.array([1.0, 1.0], dtype=jnp.float64)
    d = 0.0
    f = jnp.array([2.0, 3.0], dtype=jnp.float64)
    g = 1.0

    sol = jit_solve(A, b, c, d, f, g)
    assert jnp.all(jnp.isfinite(sol.x))

    # Test vmap over batch of f vectors
    f_batch = jnp.stack([f, 2 * f, 3 * f], axis=0)

    def solve_single(f_vec):
        return jit_solve(A, b, c, d, f_vec, g).z

    vmapped_solve = jax.vmap(solve_single)
    sol_batch = vmapped_solve(f_batch)
    assert sol_batch.shape == (3, 3)
    assert jnp.all(jnp.isfinite(sol_batch))
