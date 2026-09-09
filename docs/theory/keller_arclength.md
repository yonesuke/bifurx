# Keller Pseudo-Arclength Continuation

Pseudo-arclength continuation, introduced by Herbert B. Keller (1977), is the canonical numerical method for tracing solution branches of parameterized nonlinear systems:

$$F(u, p) = 0, \quad u \in \mathbb{R}^n, \quad p \in \mathbb{R}$$

where $F: \mathbb{R}^n \times \mathbb{R} \to \mathbb{R}^n$ is a sufficiently smooth nonlinear residual.

---

## 1. Failure of Natural Parameter Continuation

In naive (natural parameter) continuation, the parameter $p$ is incremented by a fixed step size $\Delta p$:

$$p_{k+1} = p_k + \Delta p$$

and standard Newton-Raphson iterations are applied to find $u_{k+1}$:

$$\nabla_u F(u^{(i)}, p_{k+1}) \Delta u^{(i)} = -F(u^{(i)}, p_{k+1})$$

Differentiating the equilibrium relation $F(u(p), p) = 0$ with respect to $p$ gives:

$$\nabla_u F(u, p) \frac{du}{dp} + \nabla_p F(u, p) = 0 \implies \frac{du}{dp} = -\left[\nabla_u F(u, p)\right]^{-1} \nabla_p F(u, p)$$

### The Limit Point Problem
At a **Fold / Limit Point (LP)** where the solution curve turns around with respect to $p$, the tangent becomes vertical in $(p, u)$-space:

$$\frac{dp}{ds} = 0 \implies \left\|\frac{du}{dp}\right\| \to \infty$$

At such points, the state Jacobian $\nabla_u F$ becomes rank-deficient ($\det \nabla_u F = 0$). As a result:
1. The Newton iteration matrix becomes ill-conditioned and diverges.
2. Natural parameter continuation cannot step past the fold to trace the returning branch.

---

## 2. Pseudo-Arclength Formulation

Keller's key insight is to parameterize both the state vector $u$ and the continuation parameter $p$ by an intrinsic arclength parameter $s \in \mathbb{R}$:

$$s \mapsto (u(s), p(s)) \in \mathbb{R}^n \times \mathbb{R}$$

The tangent vector along the solution manifold is:

$$\tau(s) = \begin{pmatrix} \dot{u}(s) \\ \dot{p}(s) \end{pmatrix} = \begin{pmatrix} \frac{du}{ds} \\ \frac{dp}{ds} \end{pmatrix} \in \mathbb{R}^{n+1}$$

Differentiating $F(u(s), p(s)) = 0$ with respect to $s$ yields the tangent equation:

$$\nabla_u F(u, p) \dot{u} + \nabla_p F(u, p) \dot{p} = 0 \iff [\nabla_u F \quad \nabla_p F] \tau = 0$$

normalized by the Euclidean unit length:

$$\|\dot{u}\|^2 + \dot{p}^2 = 1$$

### Pseudo-Arclength Normalization
To advance by a step $\Delta s$ from a known solution $(u_k, p_k)$ with unit tangent $\tau_k = (\dot{u}_k, \dot{p}_k)$, we impose an affine hyperplane constraint perpendicular to the previous tangent vector:

$$N(u, p; s) = \langle \dot{u}_k, u - u_k \rangle + \dot{p}_k (p - p_k) - \Delta s = 0$$

Geometrically, this constrains the new point $(u, p)$ to lie on a plane at distance $\Delta s$ along the tangent direction $\tau_k$. Even if $\dot{p}_k = 0$ (at a turning point), the constraint remains fully regular as long as $\|\tau_k\| = 1$.

---

## 3. Predictor-Corrector Architecture

### Step 1: Tangent Predictor
The continuation step begins with an explicit Euler tangent predictor:

$$\begin{pmatrix} u^{(0)} \\ p^{(0)} \end{pmatrix} = \begin{pmatrix} u_k \\ p_k \end{pmatrix} + \Delta s \begin{pmatrix} \dot{u}_k \\ \dot{p}_k \end{pmatrix}$$

### Step 2: Newton Corrector on the Bordered System
To correct the predicted point $(u^{(0)}, p^{(0)})$ onto the true solution curve $F(u, p) = 0$, we solve the augmented $(n+1)$-dimensional nonlinear system:

$$H(u, p) = \begin{pmatrix} F(u, p) \\ N(u, p; s) \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$

Applying Newton's method leads to the bordered linear system:

$$\begin{pmatrix} \nabla_u F(u^{(i)}, p^{(i)}) & \nabla_p F(u^{(i)}, p^{(i)}) \\ \dot{u}_k^T & \dot{p}_k \end{pmatrix} \begin{pmatrix} \Delta u^{(i)} \\ \Delta p^{(i)} \end{pmatrix} = - \begin{pmatrix} F(u^{(i)}, p^{(i)}) \\ N(u^{(i)}, p^{(i)}; s) \end{pmatrix}$$

with updates:

$$u^{(i+1)} = u^{(i)} + \Delta u^{(i)}, \quad p^{(i+1)} = p^{(i)} + \Delta p^{(i)}$$

---

## 4. Solving Bordered Linear Systems

A general bordered linear system has the form:

$$\begin{pmatrix} A & b \\ c^T & d \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} f \\ g \end{pmatrix}$$

where $A \in \mathbb{R}^{n \times n}$, $b, c, f \in \mathbb{R}^n$, and $d, g, y \in \mathbb{R}$.

### Block Elimination (Sherman-Morrison-Woodbury)
When the leading submatrix $A = \nabla_u F$ is well-conditioned (away from bifurcations), the system can be decoupled into two standard $n \times n$ solves:

$$A v = b, \quad A w = f$$

Substituting $x = w - y v$ into the bottom row $c^T x + d y = g$ gives:

$$c^T(w - y v) + d y = g \implies y = \frac{g - c^T w}{d - c^T v}$$

$$x = w - y v$$

This method requires only one factorization of $A$ to solve for both $v$ and $w$ simultaneously:

$$\text{cost} = \mathcal{O}\left(\frac{1}{3} n^3 + 2 n^2\right)$$

### Direct Bordered Solve
At a fold point, $A = \nabla_u F$ is singular. However, the bordered matrix:

$$M = \begin{pmatrix} A & b \\ c^T & d \end{pmatrix}$$

remains nonsingular because:
1. $b = \nabla_p F \notin \text{Range}(A)$ (transversality condition).
2. $c = \dot{u}_k$ has a non-zero component along the nullspace $\ker(A)$.

In `bifurx`, `solve_bordered_system(..., method="auto")` dynamically checks the condition number of $M$ and $A$, using block elimination when possible and falling back to a direct solve with SVD regularization if ill-conditioning is detected.

---

## 5. Tangent Update and Orientation

Once the corrector converges to $(u_{k+1}, p_{k+1})$, the new tangent vector $\tau_{k+1} = (\dot{u}_{k+1}, \dot{p}_{k+1})$ is obtained by solving:

$$\begin{pmatrix} \nabla_u F(u_{k+1}, p_{k+1}) & \nabla_p F(u_{k+1}, p_{k+1}) \\ \dot{u}_k^T & \dot{p}_k \end{pmatrix} \begin{pmatrix} \dot{u}_{k+1} \\ \dot{p}_{k+1} \end{pmatrix} = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

and normalizing:

$$\tau_{k+1} \leftarrow \frac{\tau_{k+1}}{\|\tau_{k+1}\|}$$

### Orientation Preservation
To prevent the continuation algorithm from accidentally reversing direction along the curve, we verify the inner product:

$$\langle \tau_{k+1}, \tau_k \rangle > 0$$

If $\langle \tau_{k+1}, \tau_k \rangle < 0$, we invert $\tau_{k+1} \leftarrow -\tau_{k+1}$.

---

## 6. Dynamic Step Size Adaptation

To efficiently negotiate sharp turns (high curvature) while accelerating along straight segments, `bifurx` adapts the arclength step size $\Delta s$:

$$\Delta s_{k+1} = \begin{cases}
\min(\Delta s_k \cdot 1.3, \, \Delta s_{\max}) & \text{if } N_{\text{iters}} \le 3 \\
\Delta s_k & \text{if } 3 < N_{\text{iters}} \le 6 \\
\max(\Delta s_k \cdot 0.5, \, \Delta s_{\min}) & \text{if } N_{\text{iters}} > 6 \text{ or retry on failure}
\end{cases}$$
