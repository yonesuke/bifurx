# Bifurcation Theory and Branch Switching

In parameterized dynamical systems and equilibrium problems $F(u, p) = 0$, **bifurcations** mark qualitative structural changes in the number, topology, or stability of equilibrium solutions as parameter $p$ varies.

---

## 1. Classification of Codimension-1 Bifurcations

Let $J_u = \nabla_u F(u, p) \in \mathbb{R}^{n \times n}$ and $J_p = \nabla_p F(u, p) \in \mathbb{R}^n$.

### 1.1 Limit Point / Fold (LP)
A **Limit Point** (also called a saddle-node or turning point) occurs when the equilibrium manifold exhibits a fold with respect to the parameter $p$:

- **Singularity**: $J_u$ has a simple eigenvalue $\lambda_1 = 0$.
- **Transversality**: $J_p \notin \text{Range}(J_u)$, which means $\psi^T J_p \ne 0$, where $\psi$ is the left null vector ($J_u^T \psi = 0$).

#### Test Function
Along the pseudo-arclength curve parameterized by $s$, the parameter velocity $\dot{p} = \frac{dp}{ds}$ changes sign:

$$\psi_{\text{LP}}(\tau) = \dot{p}$$

A fold point is detected when $\psi_{\text{LP}}$ crosses zero between consecutive continuation steps:

$$\psi_{\text{LP}}(\tau_k) \cdot \psi_{\text{LP}}(\tau_{k+1}) < 0$$

#### Moore-Spence Augmented System for Fold Refinement
Because $J_u$ is singular at the fold, refining the point by continuing along $p$ fails. Instead, `bifurx` solves the $(2n+1)$-dimensional **Moore-Spence augmented system**:

$$G(u, p, v) = \begin{pmatrix} F(u, p) \\ J_u(u, p) v \\ l^T v - 1 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ 0 \end{pmatrix}$$

where $v \in \mathbb{R}^n$ is the right null vector ($J_u v = 0$), and $l \in \mathbb{R}^n$ is a fixed normalization vector (chosen as $l = v_0 / \|v_0\|$).

The Jacobian matrix of $G(u, p, v)$ at a simple fold point is:

$$\nabla_{(u, p, v)} G = \begin{pmatrix} J_u & J_p & 0 \\ \nabla_u(J_u v) & \nabla_p(J_u v) & J_u \\ 0 & 0 & l^T \end{pmatrix}$$

This augmented Jacobian is **guaranteed to be non-singular** at an isolated fold point. Applying Newton's method to $G(u, p, v) = 0$ converges quadratically to machine precision without numerical ill-conditioning.

---

### 1.2 Branch Point (BP)
A **Branch Point** (transcritical, supercritical/subcritical pitchfork, or symmetry-breaking bifurcation) occurs where two distinct solution branches intersect transversally:

- **Singularity**: $J_u$ has a simple eigenvalue $\lambda_1 = 0$.
- **Transversality**: $J_p \in \text{Range}(J_u)$, so $\psi^T J_p = 0$.

#### Test Function
A naive determinant $\det(J_u)$ vanishes at both folds (LP) and branch points (BP). To isolate branch points cleanly without firing at regular folds, `bifurx` monitors the determinant of the **bordered continuation Jacobian**:

$$\psi_{\text{BP}}(u, p, \tau) = \det \begin{pmatrix} J_u(u, p) & J_p(u, p) \\ \dot{u}^T & \dot{p} \end{pmatrix}$$

At a regular fold, the bordered matrix remains nonsingular ($\psi_{\text{BP}} \ne 0$). At a simple branch point, $J_p \in \operatorname{Range}(J_u)$, so the extended Jacobian $[J_u \;\; J_p]$ drops rank and $\psi_{\text{BP}}$ crosses zero cleanly: $\psi_{\text{BP}, k} \cdot \psi_{\text{BP}, k+1} < 0$.

---

### 1.3 Hopf Bifurcation (HB)
A **Hopf Bifurcation** marks the emergence or disappearance of small-amplitude limit cycles (periodic orbits):

- **Singularity**: $J_u$ has a pair of purely imaginary complex conjugate eigenvalues:
  $$\lambda_{1, 2} = \pm i \omega \quad (\omega > 0)$$
  while all other $n - 2$ eigenvalues have non-zero real parts.

#### Test Functions
1. **Direct Eigenvalue Monitoring**:
   $$\psi_{\text{HB}}(u, p) = \max \left\{ \text{Re}(\lambda) : |\text{Im}(\lambda)| > \epsilon_{\text{imag}} \right\}$$
   A zero crossing indicates that the real part of the leading oscillatory pair changes sign.

2. **Bialternate Product Matrix**:
   The bialternate product matrix $2 J_u \odot I$ of dimension $m = \frac{n(n-1)}{2}$ has eigenvalues:
   $$\mu_{j, k} = \lambda_j + \lambda_k \quad (1 \le j < k \le n)$$
   At a Hopf point, $\mu_{1, 2} = (i\omega) + (-i\omega) = 0$. Hence:
   $$\det(2 J_u \odot I) = 0$$

#### Hopf Defining System & First Lyapunov Coefficient
To refine a Hopf bifurcation point (or continue a Hopf curve in two parameters via `bifurx.codim2`), `bifurx` solves the real-valued $(3n+2)$-dimensional **Hopf defining system** for $(u, p, v_r, v_i, \omega)$:

$$G_{\text{HB}}(u, p, v_r, v_i, \omega) = \begin{pmatrix} F(u, p) \\ J_u v_r + \omega v_i \\ J_u v_i - \omega v_r \\ \langle l_R, v_r \rangle - 1 \\ \langle l_R, v_i \rangle \end{pmatrix} = 0$$

Once located, `compute_first_lyapunov_coefficient` evaluates the **first Lyapunov coefficient** $l_1$ using nested `jax.jvp` calls for the multilinear forms $B(q, q)$ and $C(q, q, \bar{q})$:
- $l_1 < 0$: **supercritical Hopf** (stable limit cycle bifurcates).
- $l_1 > 0$: **subcritical Hopf** (unstable limit cycle bifurcates).
- $l_1 = 0$: **Generalized Hopf / Bautin (GH)** codimension-2 point.

---

## 2. Branch Switching via the Algebraic Bifurcation Equation (ABE)

At a simple branch point $(u^*, p^*)$, both intersecting branches satisfy the tangent equation:

$$J_u \dot{u} + J_p \dot{p} = 0$$

Let $\phi$ be the right null vector ($J_u \phi = 0$, $\|\phi\|=1$) and $\psi$ be the left null vector ($J_u^T \psi = 0$, $\|\psi\|=1$). Since $J_p \in \text{Range}(J_u)$, there exists a particular vector $v_0$ satisfying:

$$J_u v_0 = -J_p, \quad \langle \phi, v_0 \rangle = 0$$

Any solution tangent $(\dot{u}, \dot{p})$ at the branch point can be written as:

$$\dot{u} = \alpha \phi + \dot{p} v_0$$

for some scalar $\alpha \in \mathbb{R}$.

### Derivation of the ABE
Differentiating $F(u(s), p(s)) = 0$ twice with respect to arclength $s$ at $(u^*, p^*)$:

$$J_u \ddot{u} + J_p \ddot{p} + d^2F\left((\dot{u}, \dot{p}), (\dot{u}, \dot{p})\right) = 0$$

Projecting onto the left null vector $\psi^T$ eliminates the unknown second derivative $J_u \ddot{u}$ since $\psi^T J_u = 0$, and eliminates $\psi^T J_p \ddot{p}$ since $\psi^T J_p = 0$:

$$\psi^T d^2F\left((\dot{u}, \dot{p}), (\dot{u}, \dot{p})\right) = 0$$

Defining the basis vectors:

$$\xi_1 = \begin{pmatrix} \phi \\ 0 \end{pmatrix}, \quad \xi_2 = \begin{pmatrix} v_0 \\ 1 \end{pmatrix}$$

we substitute $(\dot{u}, \dot{p}) = \alpha \xi_1 + \dot{p} \xi_2$ to obtain the **Algebraic Bifurcation Equation**:

$$c_{11} \alpha^2 + 2 c_{12} \alpha \dot{p} + c_{22} \dot{p}^2 = 0$$

where the scalar coefficients are:

$$c_{11} = \psi^T d^2F(\xi_1, \xi_1)$$

$$c_{12} = \psi^T d^2F(\xi_1, \xi_2)$$

$$c_{22} = \psi^T d^2F(\xi_2, \xi_2)$$

---

## 3. Directional Second Derivatives via Nested JAX JVPs

In `bifurx`, these directional second derivatives are computed **analytically and exactly** without assembling large third-order derivative tensors ($n \times (n+1) \times (n+1)$).

Using nested Jacobian-Vector Products (`jax.jvp`):

```python
def d2F_directional(v_dir: tuple[Array, float], w_dir: tuple[Array, float]) -> Array:
    v_u, v_p = v_dir
    w_u, w_p = w_dir

    def dir1(u_val: Array, p_val: float) -> Array:
        # First directional derivative along v_dir
        return jax.jvp(problem.residual, (u_val, p_val), (v_u, v_p))[1]

    # Second directional derivative along w_dir
    return jax.jvp(dir1, (u_star, p_star), (w_u, w_p))[1]
```

### Solving the ABE
The discriminant of the quadratic equation:

$$\Delta_{\text{ABE}} = c_{12}^2 - c_{11} c_{22}$$

is strictly positive at a transversal branch crossing ($\Delta_{\text{ABE}} > 0$). The two roots $(\alpha^{(1)}, \dot{p}^{(1)})$ and $(\alpha^{(2)}, \dot{p}^{(2)})$ define two distinct unit tangents $\tau_1$ and $\tau_2$:

$$\tau = \frac{1}{\sqrt{\|\dot{u}\|^2 + \dot{p}^2}} \begin{pmatrix} \alpha \phi + \dot{p} v_0 \\ \dot{p} \end{pmatrix}$$

One root corresponds to the incoming primary branch; the second root gives the exact tangent $\tau_{\text{secondary}}$ of the intersecting branch!

### Branch Switch Step
To jump onto the secondary branch, `bifurx.switch_branch` sets:

$$u_{\text{init}} = u^* \pm \Delta s \, \tau_{\text{secondary}, u}$$

$$p_{\text{init}} = p^* \pm \Delta s \, \tau_{\text{secondary}, p}$$

and performs a standard bordered Newton solve to converge onto the new equilibrium branch.
