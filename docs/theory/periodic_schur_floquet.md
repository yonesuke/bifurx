# Periodic Real Schur Decomposition for Stiff Floquet Analysis

The linear stability and local bifurcations of a $T$-periodic orbit $u^*(t + T) = u^*(t)$ of $\dot{u} = f(u, p)$ are governed by its **Floquet multipliers** $\mu_1, \dots, \mu_n \in \mathbb{C}$—the eigenvalues of the **monodromy matrix** $M = \Phi(T)$, where $\Phi(t)$ solves the variational equation:

$$\frac{d\Phi}{dt} = J_u(u^*(t), p) \, \Phi(t), \qquad \Phi(0) = I_n$$

Equivalently, the real **Floquet exponents** $\sigma_i = \frac{1}{T} \log |\mu_i|$ measure the average exponential growth or decay rates per unit time and satisfy **Liouville's formula**:

$$\sum_{i=1}^n \sigma_i = \frac{1}{T} \int_0^T \operatorname{tr}\!\left(J_u(u^*(t), p)\right) dt$$

In Gauss-Legendre collocation, Stage 1 block condensation ([Collocation Condensation](collocation_condensation.md)) directly yields the $N$ per-interval transition matrices $M_0, M_1, \dots, M_{N-1} \in \mathbb{R}^{n \times n}$ whose cyclic product is the discrete monodromy matrix:

$$M = M_{N-1} M_{N-2} \cdots M_1 M_0$$

---

## 1. Why Explicit Monodromy Multiplication Fails on Stiff Orbits

The straightforward way to compute Floquet multipliers is to multiply the factors explicitly—`M = M_{N-1} @ ... @ M_0` (`monodromy_condensation_raw`)—and call `jnp.linalg.eigvals(M)`.

On well-conditioned, weakly nonlinear orbits this works cleanly. On **stiff relaxation oscillations** or **long-period orbits near homoclinic bifurcations**, however, the eigenvalues of $M$ routinely span tens to hundreds of orders of magnitude. When $M$ is formed explicitly in IEEE-754 `float64` arithmetic ($\epsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$):

- **Catastrophic roundoff loss**: Any multiplier smaller than $\epsilon_{\text{mach}} \max_i |\mu_i|$ is completely swamped by rounding noise in the dominant directions.
- **Overflow / underflow**: Multipliers with $|\mu_i| > 10^{308}$ overflow to `inf`/`nan`, while strongly attracting modes ($|\mu_i| < 10^{-308}$) underflow to exact zero.

### Concrete Example: Stiff Van der Pol Oscillator ($\mu = 6.0$)
For the Van der Pol oscillator $\ddot{x} - \mu(1 - x^2)\dot{x} + x = 0$ at $\mu = 6.0$ ($N = 80, m = 4$), the exact nontrivial Floquet multiplier is $\mu_2 \approx 1.56 \times 10^{-52}$ (corresponding to the Liouville divergence integral $\sigma_2 \approx -9.1328$):

| Method | Trivial $\mu_1$ | Nontrivial $\mu_2$ | Nontrivial Exponent $\sigma_2$ | Exact Liouville Integral |
| :--- | :--- | :--- | :--- | :--- |
| **Explicit product** (`method="condensation"`) | $1.000000$ | $-2.88 \times 10^{-18}$ *(wrong sign & magnitude)* | $-3.092$ *(catastrophic error)* | $-9.1328$ |
| **Periodic Real Schur** (`method="periodic_qr"`) | $1.000000$ | $+1.56 \times 10^{-52}$ | **$-9.1325$** *(matches Liouville)* | $-9.1328$ |

---

## 2. Bojanczyk–Golub–Van Dooren Periodic Real Schur Decomposition

To compute all Floquet multipliers and exponents accurately **without ever forming the product $M_{N-1} \cdots M_0$**, `bifurx.bvp.floquet` implements the **Periodic Real Schur Decomposition** (Bojanczyk, Golub & Van Dooren, 1992) in pure, JIT-compiled `float64` JAX.

### Theorem (Periodic Real Schur Form)
For any sequence of real $n \times n$ matrices $(M_0, M_1, \dots, M_{N-1})$, there exist orthogonal matrices $Q_0, Q_1, \dots, Q_{N-1} \in O(n)$ such that the transformed factors

$$R_k = Q_k^T M_k Q_{(k-1)\bmod N}, \qquad k = 0, \dots, N-1$$

satisfy:
- $R_0, R_1, \dots, R_{N-2}$ are **upper triangular**, and
- $R_{N-1}$ is **upper quasi-triangular** (real Schur form: $1 \times 1$ diagonal blocks for real eigenvalues and $2 \times 2$ diagonal blocks for complex conjugate pairs).

Because the cyclic orthogonal transformations telescope:

$$R_{N-1} R_{N-2} \cdots R_0 = Q_{N-1}^T \left(M_{N-1} M_{N-2} \cdots M_0\right) Q_{N-1}$$

the product $R_{N-1} \cdots R_0$ is **orthogonally similar** to $M$, so its eigenvalues are identical to the Floquet multipliers—and they can be read directly from the $1 \times 1$ and $2 \times 2$ diagonal blocks of the factors $\{R_k\}$ without ever multiplying the full $n \times n$ matrices!

---

## 3. Pure-JAX Algorithm in `bifurx.bvp.floquet`

`periodic_real_schur(factors)` executes three JAX-traceable stages using `jax.lax.scan`, `jax.lax.fori_loop`, and `jax.lax.while_loop`:

### Stage 1: Periodic Hessenberg–Triangular Reduction
1. Sweep through $k = 0, \dots, N-2$ with standard QR factorizations to make $R_0, \dots, R_{N-2}$ upper triangular while accumulating the orthogonal transformations into $H \leftarrow M_{N-1}$.
2. Reduce $H$ to **upper Hessenberg** form using Givens rotations from the left on rows $(i-1, i)$ of $H$. Each rotation is propagated cyclically through $R_0, \dots, R_{N-2}$ (applying from the right and immediately restoring upper triangularity via a left Givens rotation—*Givens bulge chasing*), and finally applied from the right to $H$.

### Stage 2: Overflow-Free Periodic Implicit Francis Double-Shift QR
On the active unreduced Hessenberg subblock $[i_{\text{lo}}, i_{\text{hi}}]$ of $H$:
- **Deflation**: Subdiagonal entries $|H_{i, i-1}| \le \text{tol} \cdot (|H_{i-1, i-1}| + |H_{i, i}|)$ are zeroed out to split the problem into smaller blocks.
- **Overflow-free shift computation**: To choose the Francis double shift from the trailing $2 \times 2$ block of the product $H R_{N-2} \cdots R_0$ without overflowing, `bifurx` computes the **normalized $2 \times 2$ product** (`_normalized_2x2_product`):
  $$B_{N-1} B_{N-2} \cdots B_0 = e^{s_{\log}} \hat{P}, \qquad \|\hat{P}\|_F = 1$$
  rescaling by the Frobenius norm at every interval while accumulating $s_{\log}$ separately. When the trailing shifts dominate or cyclic symmetries stall convergence, zero-shift and exceptional shifts are triggered automatically.
- **Implicit bulge chase**: A 3-element Householder/Givens reflector injects the double shift at the top of the active block, and Givens rotations chase the bulge down the subdiagonal of $H$ while propagating cyclically through all $R_k$ to keep them upper triangular.

### Stage 3: Post-Pass Splitting of Real $2 \times 2$ Blocks
When a converged $2 \times 2$ block on the diagonal of $H$ actually has **real** eigenvalues (which commonly happens in $n = 2$ stiff systems), a dedicated post-pass (`_split_real_2x2_block`) computes its normalized $2 \times 2$ product $\hat{P}$, rotates the real eigenvector onto $e_1$, and propagates that single Givens rotation cyclically through all $R_k$—splitting the $2 \times 2$ block cleanly into two $1 \times 1$ real blocks without rank-1 roundoff contamination.

---

## 4. Overflow/Underflow-Free Eigenvalue & Exponent Extraction

Once the Periodic Real Schur factors $\{R_k\}_{k=0}^{N-1}$ are obtained, `periodic_schur_eigenvalues` extracts `(multipliers, log_abs, phases)` directly from the diagonal blocks:

- **For a $1 \times 1$ real block at index $i$**:
  $$\log |\mu_i| = \sum_{k=0}^{N-1} \log |R_{k, ii}|, \qquad \operatorname{sign}(\mu_i) = \prod_{k=0}^{N-1} \operatorname{sign}(R_{k, ii})$$
  $$\arg(\mu_i) = \begin{cases} 0 & \text{if } \operatorname{sign}(\mu_i) \ge 0 \\ \pi & \text{if } \operatorname{sign}(\mu_i) < 0 \end{cases}$$

- **For a $2 \times 2$ complex conjugate block at indices $(i, i+1)$**:
  Using the $2 \times 2$ diagonal subblocks $B_k = R_k[i:i+2, i:i+2]$:
  $$\log |\mu_{i, i+1}| = \frac{1}{2} \sum_{k=0}^{N-1} \log |\det(B_k)|$$
  and the exact complex phase $\pm \theta$ is recovered from the eigenvalues of the normalized $2 \times 2$ product $\hat{P}$ ($\|\hat{P}\|_F = 1$).

Because the log-magnitudes are summed **in log space**, `log_abs` and the Floquet exponents $\sigma_i = \frac{1}{T} \log |\mu_i|$ remain accurate to near machine precision even when $|\mu_i| = 10^{\pm 400}$—far beyond the representable range of `float64`!

---

## 5. Stability & Bifurcation Diagnostics

`evaluate_floquet_diagnostics_from_factors(factors, tangent_p=None)` evaluates orbital stability and codimension-1 cycle bifurcation test functions directly from `(multipliers, log_abs, phases)`:

- **Trivial multiplier identification**: Finds the index minimizing $\sqrt{(\log |\mu_i|)^2 + (\arg \mu_i)^2}$ (i.e., closest to $\mu = 1$ in log-polar coordinates).
- **Linear orbital stability**: `is_stable` is `True` iff $\log |\mu_j| < 0$ for all nontrivial multipliers $j$.
- **Bifurcation test functions**:
  - **LPC (Fold of Cycles)**: $\psi_{\text{LPC}} = \dot{p}$.
  - **PD (Period-Doubling / Flip)**: $\psi_{\text{PD}} = \operatorname{Re}\!\left(\prod_i (\mu_i + 1)\right)$.
  - **NS (Neimark–Sacker / Torus)**: $\psi_{\text{NS}} = \max_{|\sin \arg(\mu_j)| > \text{tol}} \operatorname{expm1}(\log |\mu_j|)$.

---

## 6. Implementation Mapping in `bifurx`

| Function (`bifurx.bvp.floquet`) | Purpose |
| :--- | :--- |
| `monodromy_factors_condensation` | Extracts `(N, n, n)` per-interval transition factors $M_k$ via `jax.vmap` |
| `periodic_real_schur` | Pure-JAX Bojanczyk–Golub–Van Dooren Periodic Real Schur decomposition |
| `periodic_schur_eigenvalues` | Returns `(multipliers, log_abs, phases)` sorted by descending `log_abs` |
| `compute_floquet_multipliers(..., method="periodic_qr")` | Stabilized multiplier computation (alias: `"periodic_schur"`) |
| `compute_floquet_exponents` | Returns real Floquet exponents $\sigma_i = \log|\mu_i| / T$, multipliers, and phases |
| `evaluate_floquet_diagnostics_from_factors` | Factor-level stability and LPC / PD / NS bifurcation diagnostics |

---

## References

1. A. Bojanczyk, G. H. Golub & P. Van Dooren, *The periodic Schur decomposition: algorithms and applications*, Proc. SPIE **1770**, 31–42 (1992).
2. T. F. Fairgrieve & A. D. Jepson, *O.K. Floquet multipliers*, SIAM J. Numer. Anal. **28**(5), 1446–1462 (1991).
3. K. Lust, *Improved numerical Floquet multipliers*, Int. J. Bifurcation Chaos **11**(9), 2389–2410 (2001).
