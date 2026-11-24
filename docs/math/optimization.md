# 5. Optimization

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 1](linear-algebra.md) (eigenvalues), [Ch. 2](calculus.md) (gradients, Taylor, Hessian, convexity), [Ch. 3](probability.md) (expectation, variance).

    **You will learn:**

    - gradient descent, first on a parabola with numbers, then in general
    - convergence proofs for smooth, convex, strongly convex, and non-convex losses
    - why ill-conditioning slows gradient descent down
    - stochastic gradient descent and batch size
    - momentum and its $\sqrt\kappa$ speed-up
    - Adam derived from scratch, including bias correction
    - why AdamW decouples weight decay
    - warmup, cosine and WSD schedules; gradient clipping; newer optimizers such as Muon

    **Why it matters:** every LLM is trained by AdamW (or a close relative) with warmup, a decay schedule and gradient clipping. These choices decide whether a \$10M training run converges or diverges. You should know *why* each piece is there.

!!! watch "Watch"
    - 3Blue1Brown, [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w).
    - Andrew Ng (DeepLearning.AI), [Gradient descent with momentum](https://www.youtube.com/watch?v=k8fTYJPd3_I) and [Adam optimization algorithm](https://www.youtube.com/watch?v=JXQT_vxqwIs). Short and clear.

    Reading: G. Goh, [Why Momentum Really Works](https://distill.pub/2017/momentum/) (Distill, 2017), an interactive masterpiece. S. Boyd & L. Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), Ch. 9. Goodfellow et al., [*Deep Learning*](https://www.deeplearningbook.org/), Ch. 8.

---

## Roadmap

| § | Question it answers | Used later in |
|---|---|---|
| 5.1 | What problem is training solving? | everything |
| 5.2 | How big can a gradient step be, and does GD converge? | learning-rate choice |
| 5.3 | Why does GD zig-zag and crawl? | motivation for momentum and Adam |
| 5.4 | What changes when gradients are noisy? | batch size, LR decay (Ch. 15) |
| 5.5 | How does momentum help? | SGD with momentum |
| 5.6 | How does Adam work, line by line? | every LLM |
| 5.7 | Why warmup and decay? | training recipes (Ch. 14, 15) |
| 5.8 | Why clip gradients? | stability |
| 5.9 | What comes after Adam? | Muon, Shampoo, Lion |

---

## 5.1 The problem

Training means solving

$$
\min_{\theta\in\R^n}\ \mathcal{L}(\theta) = \frac1N\sum_{i=1}^N\ell(\theta;\mathbf{x}^{(i)}).
$$

| Symbol | Meaning | Size for an LLM |
|---|---|---|
| $\theta$ | all parameters, flattened into one long vector | $n\sim10^9$–$10^{12}$ |
| $\ell(\theta;\mathbf{x})$ | loss on one example | e.g. next-token cross-entropy |
| $\mathcal{L}(\theta)$ | average loss over the training set (the **empirical risk**) | $N\sim10^{13}$ tokens |

$\mathcal{L}$ is wildly non-convex. Two facts make the problem workable anyway:

1. **The gradient is cheap.** One backward pass costs about 2× a forward pass (Ch. 7).
2. **Bad local minima are rare.** In very high dimension most critical points are saddle points rather than bad local minima ([Dauphin et al., 2014](https://arxiv.org/abs/1406.2572)), so following the gradient keeps making progress.

---

## 5.2 Gradient descent and the descent lemma

**The idea:** step downhill, but not so far that you overshoot. How far is "too far" is set by the curvature.

**Gradient descent (GD):**

$$
\theta_{k+1} = \theta_k - \eta\,\nabla\mathcal{L}(\theta_k).
$$

By Theorem 2.3, $-\nabla\mathcal{L}$ is the direction of steepest local decrease. The **learning rate** $\eta$ is the step size. How big can it be?

### First, a parabola with numbers

Take $f(x) = x^2$, so $f'(x) = 2x$. One GD step is

$$
x_{k+1} = x_k - \eta\cdot2x_k = (1-2\eta)\,x_k .
$$

Each step multiplies $x$ by the same factor $1 - 2\eta$. Starting at $x_0 = 1$:

| $\eta$ | factor $1-2\eta$ | $x_0, x_1, x_2, x_3, \dots$ | Behavior |
|---|---|---|---|
| $0.1$ | $0.8$ | $1,\ 0.8,\ 0.64,\ 0.51$ | slow, steady convergence |
| $0.5$ | $0$ | $1,\ 0,\ 0,\ 0$ | lands on the minimum in one step |
| $0.9$ | $-0.8$ | $1,\ -0.8,\ 0.64,\ -0.51$ | **oscillates**, but converges |
| $1.0$ | $-1$ | $1,\ -1,\ 1,\ -1$ | bounces forever |
| $1.1$ | $-1.2$ | $1,\ -1.2,\ 1.44,\ -1.73$ | **diverges** |

**Converges exactly when $\lvert1-2\eta\rvert<1$, i.e. $\eta<1$.** Here the curvature is $f'' = 2$, so the limit is $\eta < 2/f''$. The rest of this section turns that picture into theorems for general losses.

### Smoothness: a bound on curvature

!!! recall "Refresher: Lipschitz"
    A function $g$ is **$L$-Lipschitz** if $\lVert g(\mathbf{x})-g(\mathbf{y})\rVert\le L\lVert\mathbf{x}-\mathbf{y}\rVert$: it can't change faster than rate $L$.

    Example: $g(x) = 2x$ is 2-Lipschitz, since $\lvert2x-2y\rvert = 2\lvert x-y\rvert$.

!!! definition "Definition 5.1 ($L$-smoothness)"
    $f$ is **$L$-smooth** if its gradient is $L$-Lipschitz:

    $$
    \lVert\nabla f(\mathbf{x})-\nabla f(\mathbf{y})\rVert\le L\lVert\mathbf{x}-\mathbf{y}\rVert .
    $$

    For twice-differentiable $f$ this is the same as: **every eigenvalue of the Hessian lies in $[-L, L]$**. $L$ bounds the curvature.

    Example: $f(x) = x^2$ has $f' = 2x$, which is 2-Lipschitz, so $L = 2$.

### The descent lemma

!!! theorem "Lemma 5.1 (Descent lemma)"
    If $f$ is $L$-smooth, then for all $\mathbf{x},\mathbf{y}$:

    $$
    f(\mathbf{y})\le f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}) + \frac L2\lVert\mathbf{y}-\mathbf{x}\rVert^2 .
    $$

    Consequently, one GD step with $\eta\le1/L$ satisfies

    $$
    f(\theta_{k+1})\le f(\theta_k) - \frac{\eta}{2}\lVert\nabla f(\theta_k)\rVert^2 .
    $$

**Read the first inequality as a picture.** The right-hand side is a parabola (a quadratic bowl) touching $f$ at $\mathbf{x}$: "tangent line + $\frac L2\cdot$distance²". Smoothness says $f$ never rises above this bowl. So if a step lowers the bowl, it lowers $f$ too.

???+ proof "Proof"
    **Part 1, Step 1 · Write the change in $f$ as an integral along the segment.** Let $\mathbf{d} = \mathbf{y}-\mathbf{x}$ and walk from $\mathbf{x}$ to $\mathbf{y}$ along $\mathbf{x}+t\mathbf{d}$, $t\in[0,1]$. By the fundamental theorem of calculus (§2.8), with the chain rule giving the slope $\nabla f(\mathbf{x}+t\mathbf{d})^\top\mathbf{d}$:

    $$
    f(\mathbf{y}) - f(\mathbf{x}) = \int_0^1\nabla f(\mathbf{x}+t\mathbf{d})^\top\mathbf{d}\,dt .
    $$

    **Part 1, Step 2 · Subtract the tangent-line term** $\nabla f(\mathbf{x})^\top\mathbf{d} = \int_0^1\nabla f(\mathbf{x})^\top\mathbf{d}\,dt$:

    $$
    f(\mathbf{y}) - f(\mathbf{x}) - \nabla f(\mathbf{x})^\top\mathbf{d} = \int_0^1\big(\nabla f(\mathbf{x}+t\mathbf{d})-\nabla f(\mathbf{x})\big)^\top\mathbf{d}\,dt .
    $$

    **Part 1, Step 3 · Bound the integrand.** By Cauchy–Schwarz and then smoothness:

    $$
    \big(\nabla f(\mathbf{x}+t\mathbf{d})-\nabla f(\mathbf{x})\big)^\top\mathbf{d}\le\lVert\nabla f(\mathbf{x}+t\mathbf{d})-\nabla f(\mathbf{x})\rVert\,\lVert\mathbf{d}\rVert\le Lt\lVert\mathbf{d}\rVert\cdot\lVert\mathbf{d}\rVert .
    $$

    **Part 1, Step 4 · Integrate:** $\int_0^1Lt\lVert\mathbf{d}\rVert^2dt = \frac L2\lVert\mathbf{d}\rVert^2$. That's the first inequality.

    **Part 2, Step 1 · Plug in a GD step.** Set $\mathbf{x} = \theta_k$ and $\mathbf{d} = -\eta\nabla f$, writing $\mathbf{g} = \nabla f(\theta_k)$:

    $$
    f(\theta_{k+1})\le f(\theta_k) - \eta\lVert\mathbf{g}\rVert^2 + \frac{L\eta^2}{2}\lVert\mathbf{g}\rVert^2 = f(\theta_k) - \eta\Big(1-\frac{L\eta}{2}\Big)\lVert\mathbf{g}\rVert^2 .
    $$

    **Part 2, Step 2 · Use $\eta\le1/L$.** Then $L\eta\le1$, so $1-\frac{L\eta}2\ge\frac12$. $\blacksquare$

!!! intuition "Intuition"
    - The gradient tells you the slope. $L$ tells you how fast the slope can change.
    - Step further than about $1/L$, and the ground may curve back up under you: the step can *increase* the loss.
    - The guaranteed decrease $\eta(1-\frac{L\eta}{2})$ is largest at $\eta = 1/L$ (set its derivative $1 - L\eta$ to zero). That's why $1/L$ is the "textbook" step size.

### Convergence on non-convex losses

The descent lemma says every step lowers the loss by at least $\frac{\eta}{2}\lVert\nabla f\rVert^2$. Add that up over many steps.

!!! recall "Refresher: telescoping sums"
    A sum of consecutive differences collapses to "first minus last":

    $$
    \sum_{k=0}^{K-1}(a_k - a_{k+1}) = (a_0 - a_1) + (a_1 - a_2) + \dots + (a_{K-1} - a_K) = a_0 - a_K .
    $$

!!! theorem "Theorem 5.2 (GD on non-convex functions)"
    If $f$ is $L$-smooth and bounded below by $f^*$, GD with $\eta = 1/L$ satisfies

    $$
    \min_{0\le k<K}\lVert\nabla f(\theta_k)\rVert^2\le\frac{2L\big(f(\theta_0)-f^*\big)}{K}.
    $$

    So GD finds an approximately flat point ($\lVert\nabla f\rVert\le\epsilon$) within $O(1/\epsilon^2)$ steps.

???+ proof "Proof"
    **Step 1 · Rearrange the descent lemma** with $\eta = 1/L$:

    $$
    \frac{1}{2L}\lVert\nabla f(\theta_k)\rVert^2\le f(\theta_k) - f(\theta_{k+1}).
    $$

    **Step 2 · Sum over $k = 0,\dots,K-1$.** The right side telescopes:

    $$
    \frac{1}{2L}\sum_{k<K}\lVert\nabla f(\theta_k)\rVert^2\le f(\theta_0)-f(\theta_K)\le f(\theta_0)-f^* .
    $$

    (The last step: $f(\theta_K)\ge f^*$ since $f^*$ is a lower bound.)

    **Step 3 · The smallest term is at most the average.** $\min_k\lVert\nabla f\rVert^2\le\frac1K\sum_k\lVert\nabla f\rVert^2\le\frac{2L(f(\theta_0)-f^*)}{K}$. $\blacksquare$

### Convergence on convex losses

With convexity we can say more: not just "flat", but "close to the best value".

!!! theorem "Theorem 5.3 (GD on convex and strongly convex functions)"
    Let $f$ be convex and $L$-smooth with minimizer $\theta^*$, and $\eta = 1/L$. Then

    $$
    f(\theta_K)-f(\theta^*)\le\frac{L\lVert\theta_0-\theta^*\rVert^2}{2K}.
    $$

    If additionally $f$ is **$\mu$-strongly convex** (all Hessian eigenvalues $\ge\mu>0$: curved up in *every* direction), the error shrinks geometrically:

    $$
    \lVert\theta_K-\theta^*\rVert^2\le(1-\mu/L)^K\lVert\theta_0-\theta^*\rVert^2 .
    $$

| Setting | Guarantee | Steps to accuracy $\epsilon$ |
|---|---|---|
| non-convex (Thm 5.2) | some gradient is small | $O(1/\epsilon^2)$ |
| convex | loss within $\epsilon$ of best | $O(1/\epsilon)$ |
| strongly convex | distance shrinks by $(1-\mu/L)$ each step | $O(\frac{L}{\mu}\log\frac1\epsilon)$ |

??? proof "Proof (convex case)"
    Write $\mathbf{g} = \nabla f(\theta_k)$ and $\mathbf{a} = \theta_k - \theta^*$ (how far we are from the optimum).

    **Step 1 · Convexity (tangent test, §2.6)** at $\theta_k$, evaluated at $\theta^*$, rearranged:

    $$
    f(\theta_k)\le f^* + \mathbf{g}^\top\mathbf{a} .
    $$

    **Step 2 · Combine with the descent lemma** ($\eta = 1/L$):

    $$
    f(\theta_{k+1})\le f(\theta_k) - \frac{1}{2L}\lVert\mathbf{g}\rVert^2\le f^* + \mathbf{g}^\top\mathbf{a} - \frac{1}{2L}\lVert\mathbf{g}\rVert^2 .
    $$

    **Step 3 · Complete the square.** For any vectors, $\lVert\mathbf{a}-\mathbf{b}\rVert^2 = \lVert\mathbf{a}\rVert^2 - 2\mathbf{a}^\top\mathbf{b} + \lVert\mathbf{b}\rVert^2$. With $\mathbf{b} = \mathbf{g}/L$, rearranged:

    $$
    \mathbf{g}^\top\mathbf{a} - \frac{1}{2L}\lVert\mathbf{g}\rVert^2 = \frac L2\Big(\lVert\mathbf{a}\rVert^2 - \big\lVert\mathbf{a}-\tfrac1L\mathbf{g}\big\rVert^2\Big).
    $$

    **Step 4 · Recognize the new distance.** $\mathbf{a}-\frac1L\mathbf{g} = \theta_k - \frac1L\mathbf{g} - \theta^* = \theta_{k+1}-\theta^*$. So

    $$
    f(\theta_{k+1}) - f^*\le\frac L2\Big(\lVert\theta_k-\theta^*\rVert^2 - \lVert\theta_{k+1}-\theta^*\rVert^2\Big).
    $$

    **Step 5 · Sum over $k = 0,\dots,K-1$.** The right side telescopes to at most $\frac L2\lVert\theta_0-\theta^*\rVert^2$:

    $$
    \sum_{k=1}^K(f(\theta_k)-f^*)\le\frac L2\lVert\theta_0-\theta^*\rVert^2 .
    $$

    **Step 6 · The last term is the smallest.** The descent lemma says $f(\theta_k)$ only goes down, so $f(\theta_K)-f^*$ is the smallest of the $K$ terms, and $K(f(\theta_K)-f^*)$ is at most their sum. Divide by $K$. $\blacksquare$

    *(Strongly convex case.)* For a quadratic this is the eigen-analysis in §5.3, with rate $\max_i\lvert1-\lambda_i/L\rvert\le1-\mu/L$. The general proof uses the inequality $(\nabla f(\mathbf{x})-\nabla f(\mathbf{y}))^\top(\mathbf{x}-\mathbf{y})\ge\mu\lVert\mathbf{x}-\mathbf{y}\rVert^2$; see Boyd & Vandenberghe §9.3.

---

## 5.3 Conditioning: why GD zig-zags

**The idea:** when the bowl is steep in one direction and shallow in another, a step size safe for the steep direction is painfully slow for the shallow one.

### Why quadratics tell the whole story

Near any minimum, a smooth loss looks like a quadratic bowl (Taylor, Thm 2.9), with the Hessian as its matrix. So study

$$
f(\theta) = \tfrac12\theta^\top\mathbf{A}\theta ,
$$

with $\mathbf{A}$ symmetric positive definite and eigen-decomposition $\mathbf{A} = \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$ (Thm 1.10).

### Decoupling into independent 1-D problems

Step by step:

1. **The gradient** is $\nabla f = \mathbf{A}\theta$ (Identity 2.5, with $\mathbf{A}$ symmetric and the $\frac12$ cancelling the 2).
2. **GD becomes** $\theta_{k+1} = \theta_k - \eta\mathbf{A}\theta_k = (\mathbf{I}-\eta\mathbf{A})\theta_k$.
3. **Change to eigen-coordinates** $\mathbf{c} = \mathbf{Q}^\top\theta$ (how much of $\theta$ lies along each eigenvector). Multiply step 2 by $\mathbf{Q}^\top$ and use $\mathbf{Q}^\top\mathbf{A} = \boldsymbol\Lambda\mathbf{Q}^\top$:

    $$
    \mathbf{c}_{k+1} = (\mathbf{I}-\eta\boldsymbol\Lambda)\,\mathbf{c}_k .
    $$

4. **$\boldsymbol\Lambda$ is diagonal, so each coordinate evolves on its own:**

$$
c_{i,k+1} = (1-\eta\lambda_i)\,c_{i,k}\quad\Longrightarrow\quad c_{i,k} = (1-\eta\lambda_i)^k\,c_{i,0}.
$$

Each eigen-direction is the parabola from §5.2 with its own curvature $\lambda_i$.

!!! theorem "Theorem 5.4 (GD on a quadratic)"
    GD converges if and only if $\lvert1-\eta\lambda_i\rvert<1$ for every $i$, i.e.

    $$
    \eta<\frac{2}{\lambda_\max}.
    $$

    The best fixed step is $\eta^* = \frac{2}{\lambda_\min+\lambda_\max}$. It gives the per-step contraction factor

    $$
    \rho = \frac{\kappa-1}{\kappa+1},\qquad \kappa = \frac{\lambda_\max}{\lambda_\min}\ \text{(the condition number)}.
    $$

    Reaching accuracy $\epsilon$ takes about $\frac{\kappa}{2}\log\frac1\epsilon$ steps.

???+ proof "Proof"
    **Step 1 · Convergence condition.** Each coordinate is multiplied by $(1-\eta\lambda_i)$ every step. It shrinks to 0 exactly when $\lvert1-\eta\lambda_i\rvert<1$, i.e. $0<\eta\lambda_i<2$. The tightest constraint comes from the largest $\lambda_i$.

    **Step 2 · The slowest coordinate sets the pace.** The overall rate is $\max_i\lvert1-\eta\lambda_i\rvert$. As a function of $\lambda$, $1-\eta\lambda$ is a straight line, so its largest absolute value over $[\lambda_\min,\lambda_\max]$ is at one of the two ends.

    **Step 3 · Balance the two ends.** Making one end smaller makes the other bigger, so the best $\eta$ makes them equal in size (opposite in sign):

    $$
    1-\eta\lambda_\min = -(1-\eta\lambda_\max)\;\Longrightarrow\;\eta^* = \frac{2}{\lambda_\min+\lambda_\max}.
    $$

    **Step 4 · The resulting rate:**

    $$
    \rho = 1 - \eta^*\lambda_\min = \frac{\lambda_\max-\lambda_\min}{\lambda_\max+\lambda_\min} = \frac{\kappa-1}{\kappa+1}
    $$

    (divide top and bottom by $\lambda_\min$).

    **Step 5 · Step count.** $\rho^k\le\epsilon$ needs $k\ge\log(1/\epsilon)/\log(1/\rho)$. For large $\kappa$, $\log\frac1\rho = \log\frac{1+1/\kappa}{1-1/\kappa}\approx\frac2\kappa$ (using $\log(1+x)\approx x$ for small $x$). So $k\approx\frac\kappa2\log\frac1\epsilon$. $\blacksquare$

!!! example "Worked example: a stretched bowl"
    $f(x,y) = \frac12(x^2 + 10y^2)$, so $\mathbf{A} = \diag(1, 10)$ and $\kappa = 10$.

    | Quantity | Value |
    |---|---|
    | stability limit $2/\lambda_\max$ | $0.2$ |
    | best step $\eta^* = 2/(1+10)$ | $0.18$ |
    | factor for $x$ (flat direction) $1 - 0.18\cdot1$ | $+0.82$: slow creep |
    | factor for $y$ (steep direction) $1 - 0.18\cdot10$ | $-0.82$: flips sign every step |
    | steps to shrink error by $10^{6}$ | about $70$ |

    The $y$ coordinate bounces from side to side of the valley while $x$ creeps along its floor. With $\kappa = 1000$ it would take about 7,000 steps.

The learning rate is capped by the **sharpest** direction ($\lambda_\max$), while the **flattest** direction ($\lambda_\min$) is what takes time. Deep-network Hessians are extremely ill-conditioned. That is the motivation for momentum and for adaptive methods.

<div class="widget" data-widget="gradient-descent"></div>

Try the ill-conditioned bowl with $\kappa = 30$: push SGD's $\eta$ until it oscillates, then compare momentum and Adam.

!!! llm "The edge of stability"
    Cohen et al., [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065) (2021), found that during full-batch training:

    - the sharpness $\lambda_\max$ rises until it hovers right at $2/\eta$, the stability boundary of Theorem 5.4;
    - the network adapts its curvature to the learning rate;
    - the loss keeps decreasing, but not monotonically.

    This is one reason learning-rate warmup and decay matter so much.

---

## 5.4 Stochastic gradient descent

**The idea:** use a cheap, noisy gradient from a small random batch instead of the exact one. It's right on average, but the noise sets a floor on how close you get.

The full gradient averages over $N\sim10^{13}$ examples: far too expensive per step. Instead sample a **mini-batch** $\mathcal{B}$ of $B$ examples and use

$$
\hat{\mathbf{g}} = \frac1B\sum_{i\in\mathcal{B}}\nabla\ell(\theta;\mathbf{x}^{(i)}).
$$

From Ch. 3 (§3.6):

| Property | Formula | Meaning |
|---|---|---|
| unbiased | $\E[\hat{\mathbf{g}}] = \nabla\mathcal{L}(\theta)$ | right on average |
| noise | $\Cov(\hat{\mathbf{g}}) = \boldsymbol\Sigma/B$ | shrinks like $1/B$ |

Here $\boldsymbol\Sigma$ is the covariance of a single example's gradient.

We need one identity: **a noisy vector's average squared length = (length of its mean)² + its noise.** Write $\hat{\mathbf{g}} = \nabla f + \boldsymbol\varepsilon$ with $\E\boldsymbol\varepsilon = \mathbf{0}$. Then

$$
\E\lVert\hat{\mathbf{g}}\rVert^2 = \lVert\nabla f\rVert^2 + 2\nabla f^\top\underbrace{\E\boldsymbol\varepsilon}_{=\mathbf{0}} + \E\lVert\boldsymbol\varepsilon\rVert^2 = \lVert\nabla f\rVert^2 + \E\lVert\hat{\mathbf{g}}-\nabla f\rVert^2 .
$$

!!! theorem "Theorem 5.5 (SGD convergence, non-convex)"
    Let $f$ be $L$-smooth and bounded below by $f^*$, and let the stochastic gradient have variance $\E\lVert\hat{\mathbf{g}}-\nabla f\rVert^2\le\sigma^2$. With a constant step $\eta\le1/L$,

    $$
    \frac1K\sum_{k<K}\E\lVert\nabla f(\theta_k)\rVert^2\le\underbrace{\frac{2(f(\theta_0)-f^*)}{\eta K}}_{\text{shrinks with more steps}} + \underbrace{L\eta\sigma^2}_{\text{noise floor}}.
    $$

???+ proof "Proof"
    **Step 1 · Descent lemma** (first part) with the step $-\eta\hat{\mathbf{g}}$:

    $$
    f(\theta_{k+1})\le f(\theta_k) - \eta\nabla f^\top\hat{\mathbf{g}} + \frac{L\eta^2}{2}\lVert\hat{\mathbf{g}}\rVert^2 .
    $$

    **Step 2 · Average over the batch randomness** (with $\theta_k$ fixed). Use $\E\hat{\mathbf{g}} = \nabla f$ in the middle term and the identity above in the last:

    $$
    \E f(\theta_{k+1})\le f(\theta_k) - \eta\lVert\nabla f\rVert^2 + \frac{L\eta^2}{2}\big(\lVert\nabla f\rVert^2+\sigma^2\big).
    $$

    **Step 3 · Use $\eta\le1/L$.** Then $-\eta + \frac{L\eta^2}{2}\le-\frac{\eta}{2}$, so

    $$
    \E f(\theta_{k+1})\le f(\theta_k)-\frac\eta2\lVert\nabla f\rVert^2 + \frac{L\eta^2\sigma^2}{2}.
    $$

    **Step 4 · Sum over $k$ and telescope** (exactly as in Thm 5.2), then divide by $\eta K/2$. $\blacksquare$

The bound has two terms, and they tell you how to train:

| Term | Behavior | What to do about it |
|---|---|---|
| $\frac{2(f_0-f^*)}{\eta K}$ | shrinks as you take more steps | train longer, or use a bigger $\eta$ |
| $L\eta\sigma^2$, the **noise floor** | does *not* shrink with more steps | eventually **decrease $\eta$**, or **increase $B$** to shrink $\sigma^2$ |

With a constant learning rate, SGD bounces around in a region whose size scales with $\eta$. To converge you must eventually lower $\eta$. (Classically, the Robbins–Monro conditions $\sum\eta_k = \infty$, $\sum\eta_k^2<\infty$: big enough to get anywhere, shrinking fast enough to settle.)

This is the theory behind learning-rate decay (§5.7), and why the loss drops sharply at the end of a cosine or WSD schedule.

!!! llm "Batch size and the critical batch size"
    Doubling $B$ halves the gradient variance but doubles the compute per step.

    - **While noise dominates**, a 2× bigger batch lets you take about 2× fewer steps (perfect scaling).
    - **Beyond the critical batch size**, the gradient is already accurate. Bigger batches waste compute.

    McCandlish et al., [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) (2018), estimate it with the **gradient noise scale**

    $$
    B_\text{noise} = \frac{\tr(\boldsymbol\Sigma)}{\lVert\nabla\mathcal{L}\rVert^2}\qquad\left(\frac{\text{total noise}}{\text{signal}^2}\right).
    $$

    It grows as the loss falls (the signal shrinks), which is why LLM runs often **ramp up the batch size** during training. GPT-3 175B used a batch of 3.2M tokens; Llama 3 405B grew from 4M to 16M tokens.

---

## 5.5 Momentum

**The idea:** keep a running sum of past gradients. Consistent directions build up speed; directions that flip back and forth cancel out.

**Heavy-ball momentum** (Polyak, 1964) keeps a velocity $\mathbf{v}$:

$$
\mathbf{v}_{k+1} = \mu\mathbf{v}_k + \nabla f(\theta_k),\qquad \theta_{k+1} = \theta_k - \eta\,\mathbf{v}_{k+1}.
$$

$\mu\in[0,1)$ (typically $0.9$) controls how much of the old velocity is kept.

!!! recall "Refresher: geometric series"
    $$
    1 + \mu + \mu^2 + \mu^3 + \dots = \frac{1}{1-\mu}\qquad(\lvert\mu\rvert<1).
    $$

    Why: call the sum $S$. Then $S = 1 + \mu(1 + \mu + \dots) = 1 + \mu S$, so $S(1-\mu) = 1$.

    The finite version: $\sum_{j=0}^{t-1}\mu^j = \dfrac{1-\mu^t}{1-\mu}$. Example: $\mu = 0.9$ gives $S = 10$.

**Unroll the recursion** (start from $\mathbf{v}_0 = \mathbf{0}$ and substitute repeatedly):

$$
\mathbf{v}_{k+1} = \nabla f(\theta_k) + \mu\nabla f(\theta_{k-1}) + \mu^2\nabla f(\theta_{k-2}) + \dots = \sum_{j\le k}\mu^{k-j}\nabla f(\theta_j).
$$

An exponentially weighted sum of past gradients: recent ones count most.

!!! example "Worked example: consistent vs. flip-flopping gradients ($\mu = 0.9$)"
    | Step | Constant gradient $+1$: velocity | Alternating gradient $+1, -1, \dots$: velocity |
    |---|---|---|
    | 1 | $1.00$ | $1.00$ |
    | 2 | $1.90$ | $-0.10$ |
    | 3 | $2.71$ | $0.91$ |
    | 4 | $3.44$ | $-0.18$ |
    | many | $\to 1/(1-\mu) = 10$ | swings of size $\approx 1/(1+\mu) = 0.53$ |

    - In a consistent direction the effective step grows to $\eta/(1-\mu)$: **10× for $\mu = 0.9$.**
    - Where the gradient flips sign every step, contributions cancel.

    So momentum **accelerates along the valley floor and damps the zig-zag across it** — exactly the two problems from §5.3.

!!! theorem "Theorem 5.6 (Momentum on a quadratic)"
    On $f = \frac12\theta^\top\mathbf{A}\theta$ with eigenvalues in $[\mu_A, L]$ (so $\kappa = L/\mu_A$), heavy-ball with tuned

    $$
    \eta = \frac{4}{(\sqrt L+\sqrt{\mu_A})^2},\qquad \mu = \left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^2
    $$

    converges at the rate

    $$
    \rho = \frac{\sqrt\kappa-1}{\sqrt\kappa+1}.
    $$

    So it needs about $\frac{\sqrt\kappa}{2}\log\frac1\epsilon$ steps instead of $\frac\kappa2\log\frac1\epsilon$.

| $\kappa$ | GD steps $\propto\kappa$ | Momentum steps $\propto\sqrt\kappa$ | Speed-up |
|---|---|---|---|
| $100$ | $\sim690$ | $\sim69$ | $10\times$ |
| $10^4$ | $\sim69{,}000$ | $\sim690$ | $100\times$ |

(For shrinking the error by $10^6$; see Exercise 2.)

??? proof "Proof sketch"
    **Step 1 · Decouple, as in §5.3.** In the eigenbasis, each coordinate obeys a recurrence involving the last *two* values (because velocity remembers the previous step):

    $$
    c_{k+1} = (1+\mu-\eta\lambda)\,c_k - \mu\,c_{k-1}.
    $$

    **Step 2 · Write it as a $2\times2$ linear system** acting on the pair (current, previous):

    $$
    \begin{pmatrix}c_{k+1}\\c_k\end{pmatrix} = \underbrace{\begin{pmatrix}1+\mu-\eta\lambda & -\mu\\ 1 & 0\end{pmatrix}}_{\mathbf{M}}\begin{pmatrix}c_k\\c_{k-1}\end{pmatrix}.
    $$

    **Step 3 · The rate is set by $\mathbf{M}$'s eigenvalues** $z$, which solve $z^2-(1+\mu-\eta\lambda)z+\mu = 0$.

    **Step 4 · When the roots are complex, they all have the same size.** If the discriminant is negative, the two roots are complex conjugates. Their product is $\mu$ (the constant term), and conjugates have equal size, so $\lvert z\rvert = \sqrt\mu$ — independent of $\lambda$!

    **Step 5 · Tune.** The given $\eta,\mu$ are the choice that makes the discriminant non-positive for *every* $\lambda\in[\mu_A,L]$ with the smallest possible $\mu$. That gives $\lvert z\rvert = \sqrt\mu = \frac{\sqrt\kappa-1}{\sqrt\kappa+1}$ for every eigen-direction at once. $\blacksquare$

    Goh's Distill article animates exactly this analysis.

PyTorch's `SGD(momentum=μ)` uses exactly this form ($\mathbf{v}\leftarrow\mu\mathbf{v}+\mathbf{g}$, $\theta\leftarrow\theta-\eta\mathbf{v}$). Our implementation matches it to 6 decimal places in the tests:

```python
--8<-- "code/optim.py:sgd"
```

**Nesterov momentum** evaluates the gradient at the look-ahead point $\theta_k-\eta\mu\mathbf{v}_k$ ("where momentum is about to take me"). It achieves the optimal $\sqrt\kappa$ rate for *all* smooth strongly convex functions, not just quadratics (Nesterov, 1983).

---

## 5.6 Adaptive methods: from AdaGrad to Adam

**The idea:** give every parameter its own step size, scaled down when its gradients are typically large and up when they're small.

Momentum only partly fixes conditioning. The adaptive methods evolved in three steps:

| Method | Per-parameter scale | Problem it fixed | Problem it had |
|---|---|---|---|
| **AdaGrad** (Duchi et al., 2011) | $\sqrt{\text{sum of all past } g^2}$ | one global LR for features of very different frequency | the sum only grows, so steps decay to zero |
| **RMSProp** (Hinton, 2012) | $\sqrt{\text{moving average of } g^2}$ | AdaGrad's freezing | no momentum; biased early estimates |
| **Adam** (Kingma & Ba, 2015) | RMSProp + momentum + bias correction | both | (weight decay interacts badly; see AdamW) |

### Exponential moving averages

Adam is built from **exponential moving averages** (EMAs). An EMA with decay $\beta$ updates a running estimate by mixing in each new value:

$$
m_t = \beta\,m_{t-1} + (1-\beta)\,g_t .
$$

- Each step keeps a fraction $\beta$ of the old estimate and adds $1-\beta$ of the new value.
- It effectively averages over the last $\approx1/(1-\beta)$ values: 10 for $\beta = 0.9$, 1,000 for $\beta = 0.999$.

### Adam, line by line

!!! definition "Algorithm 5.2 (Adam)"
    With $\mathbf{g}_t = \nabla\mathcal{L}(\theta_{t-1})$ (on a mini-batch), $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$, and all operations elementwise:

    $$
    \begin{aligned}
    \mathbf{m}_t &= \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t &&\text{(1st moment: average of } \mathbf{g})\\
    \mathbf{v}_t &= \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2 &&\text{(2nd moment: average of } \mathbf{g}^2)\\
    \hat{\mathbf{m}}_t &= \mathbf{m}_t/(1-\beta_1^t),\qquad \hat{\mathbf{v}}_t = \mathbf{v}_t/(1-\beta_2^t) &&\text{(bias correction)}\\
    \theta_t &= \theta_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon} &&\text{(the step)}
    \end{aligned}
    $$

| Line | What it does | Typical value |
|---|---|---|
| $\mathbf{m}_t$ | momentum: smoothed gradient direction | $\beta_1 = 0.9$ (~10 steps) |
| $\mathbf{v}_t$ | smoothed squared gradient: each parameter's typical gradient size² | $\beta_2 = 0.95$–$0.999$ |
| bias correction | fixes the start-from-zero underestimate (below) | — |
| step | move by $\eta\times$ (direction ÷ typical size) | $\epsilon = 10^{-8}$ prevents division by zero |

### Why bias correction

Because $\mathbf{m}$ and $\mathbf{v}$ start at zero, the first estimates are far too small.

!!! example "Worked example: a constant gradient of 1 with $\beta_1 = 0.9$"
    | $t$ | $m_t$ (raw) | $1-\beta_1^t$ | $\hat m_t$ (corrected) |
    |---|---|---|---|
    | 1 | $0.1$ | $0.1$ | $1.0$ |
    | 2 | $0.19$ | $0.19$ | $1.0$ |
    | 3 | $0.271$ | $0.271$ | $1.0$ |
    | 50 | $0.995$ | $0.995$ | $1.0$ |

    The raw average starts at 10% of the truth and crawls up. Dividing by $1-\beta_1^t$ makes it exactly right from step 1.

!!! theorem "Proposition 5.7 (Why bias correction)"
    If the gradients have a stationary distribution with $\E[\mathbf{g}_t] = \mathbf{g}$ and $\E[\mathbf{g}_t^2] = \mathbf{s}$, then

    $$
    \E[\mathbf{m}_t] = (1-\beta_1^t)\,\mathbf{g},\qquad \E[\mathbf{v}_t] = (1-\beta_2^t)\,\mathbf{s}.
    $$

    Dividing by $1-\beta^t$ makes the estimates unbiased.

???+ proof "Proof"
    **Step 1 · Unroll the recursion** from $\mathbf{m}_0 = \mathbf{0}$ (as for momentum):

    $$
    \mathbf{m}_t = (1-\beta_1)\sum_{j=1}^t\beta_1^{t-j}\mathbf{g}_j .
    $$

    **Step 2 · Take expectations.** Every $\E\mathbf{g}_j = \mathbf{g}$, so it factors out:

    $$
    \E\mathbf{m}_t = (1-\beta_1)\,\mathbf{g}\sum_{j=0}^{t-1}\beta_1^j .
    $$

    **Step 3 · Finite geometric series** (refresher in §5.5):

    $$
    = (1-\beta_1)\,\mathbf{g}\,\frac{1-\beta_1^t}{1-\beta_1} = (1-\beta_1^t)\,\mathbf{g}.
    $$

    **Step 4 · Same for $\mathbf{v}_t$**, with $\mathbf{g}_j^2$ and $\beta_2$. $\blacksquare$

**How wrong is step 1 without correction?** With $\beta_1 = 0.9$, $\beta_2 = 0.999$: $m_1 = 0.1g$ and $\sqrt{v_1} = \sqrt{0.001}\,\lvert g\rvert\approx0.032\lvert g\rvert$.

- Correct neither: the step is $0.1/0.032\approx3.2\times$ too large.
- Correct only $\mathbf{m}$: the step is $1/0.032\approx32\times$ too large.

### What Adam's update means

Ignoring $\epsilon$, parameter $i$ moves by $\eta\cdot\hat m_i/\sqrt{\hat v_i}$. The ratio $\hat m_i/\sqrt{\hat v_i}$ is like a **signal-to-noise ratio**:

| Gradient history for parameter $i$ | $\lvert\hat m_i\rvert$ vs $\sqrt{\hat v_i}$ | Step size |
|---|---|---|
| always the same sign and size | about equal | $\approx\eta$, *whatever the gradient's scale* |
| mostly noise, random signs | $\lvert\hat m_i\rvert\ll\sqrt{\hat v_i}$ | small |

Consequences:

- Adam is **invariant to rescaling** the loss or any single parameter's gradient.
- The learning rate becomes roughly "the maximum change per parameter per step".
- In the quadratic picture, dividing by $\sqrt{v_i}$ is a diagonal approximation to rescaling by the curvature, which undoes some of the ill-conditioning.

### AdamW: decoupled weight decay

Two ways to keep weights small, which sound identical:

| Name | What you do | Update for plain SGD |
|---|---|---|
| **L2 regularization** | add $\frac\lambda2\lVert\theta\rVert^2$ to the loss, i.e. $\lambda\theta$ to the gradient | $\theta - \eta(\mathbf{g}+\lambda\theta)$ |
| **weight decay** | shrink weights by a factor each step | $(1-\eta\lambda)\theta - \eta\mathbf{g}$ |

Expand the first row: $\theta-\eta\mathbf{g}-\eta\lambda\theta = (1-\eta\lambda)\theta-\eta\mathbf{g}$. **For SGD they are the same.**

**For Adam they differ.** With L2, the $\lambda\theta$ term goes into $\mathbf{g}$ and gets divided by $\sqrt{\hat{\mathbf{v}}}$. So weights with large gradient variance get *less* regularization — not what we want.

Loshchilov & Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (2019), proposed applying the decay directly, outside the normalization:

$$
\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon} + \lambda\,\theta_{t-1}\right).
$$

This is **AdamW**, the default optimizer for essentially every LLM. Our implementation matches `torch.optim.AdamW` exactly:

```python
--8<-- "code/optim.py:adamw"
```

!!! llm "Typical LLM optimizer settings"
    | | GPT-3 (2020) | Llama 2 / 3 | Notes |
    |---|---|---|---|
    | $\beta_1,\beta_2$ | 0.9, 0.95 | 0.9, 0.95 | $\beta_2 = 0.95$ (not 0.999) reacts faster to gradient spikes and improves stability |
    | $\epsilon$ | $10^{-8}$ | $10^{-5}$ to $10^{-8}$ | |
    | weight decay $\lambda$ | 0.1 | 0.1 | usually **not** applied to biases, norm gains or (often) embeddings |
    | peak LR | $6\times10^{-4}$ (125M) … $0.6\times10^{-4}$ (175B) | $3\times10^{-4}$ (7B) … $8\times10^{-5}$ (405B) | bigger models need smaller LR |
    | grad clip | 1.0 | 1.0 | global norm (§5.8) |

    **Memory cost.** AdamW stores $\mathbf{m}$ and $\mathbf{v}$: two extra numbers per parameter, usually in fp32. With fp32 master weights and bf16 gradients that's about 16 bytes per parameter (Ch. 16). A 70B model needs about 1.1 TB just for the weights, gradients and optimizer state, before activations.

---

## 5.7 Learning-rate schedules

**The idea:** start gently, run fast, finish slowly.

LLMs are never trained with a constant learning rate. The standard recipe has two parts.

### 1. Linear warmup

Ramp the LR from ~0 to its peak over the first ~0.1–2% of steps (GPT-3: 375M tokens; nanoGPT: 2,000 steps). Why the start is fragile:

- Adam's $\hat{\mathbf{v}}$ is estimated from very few gradients, so it is noisy.
- The freshly initialized network sits in a high-curvature region (large $\lambda_\max$, so the safe $\eta$ is small).
- Gradient norms are large.

Warmup keeps early updates small until the statistics settle and the curvature drops ([Liu et al., 2020](https://arxiv.org/abs/1908.03265); [Gilmer et al., 2021](https://arxiv.org/abs/2110.04369)).

### 2. Decay

Lower the LR toward a small final value. This removes the $\eta\sigma^2$ noise floor of Theorem 5.5. Two common shapes:

**Cosine decay** to ~10% of peak ([Loshchilov & Hutter, 2017](https://arxiv.org/abs/1608.03983)). With progress $p\in[0,1]$ through the decay phase and floor ratio $r$ (e.g. $0.1$):

$$
\eta(p) = \eta_\text{max}\Big(r + (1-r)\cdot\tfrac12\big(1+\cos(\pi p)\big)\Big).
$$

| $p$ | $\tfrac12(1+\cos\pi p)$ | $\eta/\eta_\text{max}$ with $r = 0.1$ |
|---|---|---|
| 0 | 1 | 1.00 |
| 0.25 | 0.85 | 0.87 |
| 0.5 | 0.5 | 0.55 |
| 0.75 | 0.15 | 0.23 |
| 1 | 0 | 0.10 |

Used by GPT-3, Llama, nanoGPT.

**Warmup–Stable–Decay (WSD).** Constant LR for most of training, then a short, fast decay (often 10–20% of steps) at the end ([Hu et al., MiniCPM, 2024](https://arxiv.org/abs/2404.06395); [Hägele et al., 2024](https://arxiv.org/abs/2405.18392)).

- It matches cosine in final loss.
- You don't need to fix the run length in advance: branch off a decay from any checkpoint.
- DeepSeek-V3 and many 2024–2025 models use WSD-style schedules.

<div class="widget" data-widget="lr-schedule"></div>

```python
--8<-- "code/optim.py:schedule"
```

---

## 5.8 Gradient clipping

**The idea:** if a gradient is suspiciously huge, shrink it to a maximum length — but keep its direction.

Occasional batches produce enormous gradients (loss spikes). One such step can throw the model into a bad region it never recovers from.

**Clipping by global norm** rescales the entire gradient (all parameters together) if its norm exceeds a threshold $c$:

$$
\mathbf{g}\leftarrow\mathbf{g}\cdot\min\Big(1,\ \frac{c}{\lVert\mathbf{g}\rVert}\Big),\qquad \lVert\mathbf{g}\rVert = \sqrt{\textstyle\sum_{\text{all params}}g^2}.
$$

!!! example "Worked example"
    $\mathbf{g} = (3, 4)$, threshold $c = 1$.

    - $\lVert\mathbf{g}\rVert = 5 > 1$, so scale by $1/5$.
    - Clipped: $(0.6, 0.8)$. Length 1, same direction.

    If $\lVert\mathbf{g}\rVert$ had been $0.5$, the factor would be $\min(1, 2) = 1$: unchanged.

In PyTorch: `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`, called after `backward()` and before `step()` (see the training loop in Ch. 14).

**Log $\lVert\mathbf{g}\rVert$ every step.** It's one of the best ways to monitor training health: a steadily rising gradient norm often comes before a divergence.

---

## 5.9 Beyond Adam

| Optimizer | Core idea | Notes |
|---|---|---|
| **Newton's method** | $\theta\leftarrow\theta-\mathbf{H}^{-1}\nabla f$: rescale every eigen-direction by its curvature | on a quadratic it converges in one step ($\kappa$ disappears); but $\mathbf{H}$ has $n^2$ entries, so with $n = 10^{10}$ it can't even be stored |
| **Shampoo / SOAP** ([Gupta et al., 2018](https://arxiv.org/abs/1802.09568); [Vyas et al., 2024](https://arxiv.org/abs/2409.11321)) | a Kronecker-factored curvature estimate per weight matrix | Shampoo won the external tuning track of the AlgoPerf benchmark |
| **Muon** ([K. Jordan, 2024](https://kellerjordan.github.io/posts/muon/)) | step along the **orthogonalized** momentum matrix | see below |
| **Lion** ([Chen et al., 2023](https://arxiv.org/abs/2302.06675)) | use only the *sign* of a momentum mix | found by program search; every update has magnitude exactly $\eta$; one state per parameter instead of two |
| **Schedule-free** ([Defazio et al., 2024](https://arxiv.org/abs/2405.15682)) | replace the LR schedule with iterate averaging | no need to choose a decay |

**Muon in more detail.** It treats each 2-D weight matrix as a whole, using the SVD from Ch. 1:

1. Take the momentum matrix $\mathbf{M} = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top$.
2. Replace it with its **orthogonalization** $\mathbf{U}\mathbf{V}^\top$: all singular values set to 1 (computed with a few Newton–Schulz iterations instead of a full SVD).
3. Step along that.

This is steepest descent under the spectral norm. Every direction of the update gets equal magnitude, so rare directions aren't drowned out. Muon set NanoGPT speed-run records and was scaled to a 1T-parameter MoE in Moonshot's Kimi K2 ([Moonshot AI, 2025](https://arxiv.org/abs/2507.20534)).

AdamW remains the safe default. When you read a new model's tech report, the optimizer section is now one of the places where real differences show up.

---

## Key takeaways

| Concept | One-line summary | Where it matters |
|---|---|---|
| Step-size limit | GD is stable only for $\eta<2/\lambda_\max$ | learning-rate choice |
| Descent lemma | with $\eta\le1/L$, every step lowers the loss by $\ge\frac\eta2\lVert\nabla f\rVert^2$ | convergence proofs |
| Conditioning | GD needs $\approx\frac\kappa2\log\frac1\epsilon$ steps; $\kappa\gg1$ is the core difficulty | why plain GD is slow |
| SGD | unbiased but noisy; the noise floor $\propto\eta\sigma^2$ | LR decay, batch size |
| Momentum | an EMA of gradients: fast along consistent directions, damped across oscillating ones; $\sqrt\kappa$ instead of $\kappa$ | SGD with momentum |
| Adam | momentum + per-parameter scaling by the RMS gradient + bias correction; steps of $\approx\eta$ | every LLM |
| **AdamW** | weight decay applied outside the adaptive scaling | the LLM default |
| Schedules | warmup (fragile start) + cosine/WSD decay (remove the noise floor) | training recipes |
| Clipping | cap the global gradient norm, keep the direction | stability |
| LLM defaults | $\beta = (0.9, 0.95)$, $\lambda = 0.1$, warmup + cosine/WSD, clip at 1.0 | Ch. 14–15 |

## Exercises

1. Show that for $f(x) = \frac\lambda2x^2$, GD diverges if and only if $\eta>2/\lambda$. What happens at exactly $\eta = 2/\lambda$?
2. For $\kappa = 100$, compute the GD rate $\frac{\kappa-1}{\kappa+1}$ and the momentum rate $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$. How many steps does each need to shrink the error by a factor of $10^{6}$?
3. Show that, with $\epsilon = 0$, Adam's first update is $\theta_1 = \theta_0-\eta\operatorname{sign}(\mathbf{g}_1)$.
4. Explain why L2 regularization and weight decay are identical for SGD but not for Adam. Give a two-parameter example where they produce different updates.
5. Implement the update of `AdamW` above without bias correction and train the GPT in Ch. 14 for 200 steps. Compare the first 20 losses with the corrected version.
6. **Code:** in the gradient-descent widget, set $\kappa = 50$ and find the largest SGD learning rate that still converges. Compare it with $2/\kappa$ (here $\lambda_\max = \kappa$).

??? success "Solutions"
    1. $f'(x) = \lambda x$, so $x_{k+1} = x_k - \eta\lambda x_k = (1-\eta\lambda)x_k$. It diverges if and only if $\lvert1-\eta\lambda\rvert>1$, i.e. $\eta>2/\lambda$. At $\eta = 2/\lambda$ the factor is $-1$, so $x_k = (-1)^kx_0$: it oscillates forever without converging.
    2. GD: $\rho = 99/101\approx0.980$, so about $\ln(10^6)/(-\ln0.980)\approx 690$ steps. Momentum: $\rho = 9/11\approx0.818$, so about $69$ steps.
    3. At $t = 1$: $\hat m_1 = \frac{(1-\beta_1)\mathbf{g}_1}{1-\beta_1} = \mathbf{g}_1$ and $\hat v_1 = \mathbf{g}_1^2$, so $\hat m_1/\sqrt{\hat v_1} = \mathbf{g}_1/\lvert\mathbf{g}_1\rvert = \operatorname{sign}(\mathbf{g}_1)$.
    4. For SGD, $\theta - \eta(\mathbf{g}+\lambda\theta) = (1-\eta\lambda)\theta-\eta\mathbf{g}$: the same thing. For Adam, the $\lambda\theta$ term is divided by $\sqrt{\hat v}$. Take two parameters with the same value $\theta_1 = \theta_2 = 1$ but gradient RMS 10 and 0.1: L2-in-Adam shrinks the second 100× more than the first, while AdamW shrinks both by the same factor $(1-\eta\lambda)$.
    5. Without correction the first numerator is scaled by $1-\beta_1$ and the denominator by $\sqrt{1-\beta_2}$. The ratio at step 1 is $(1-\beta_1)/\sqrt{1-\beta_2}$: $0.1/\sqrt{0.05}\approx 0.45$ with $\beta_2 = 0.95$ (steps too small), and $3.2$ with $\beta_2 = 0.999$ (steps too large). You'll see slower or more erratic early progress.
    6. The limit is $\eta<2/\lambda_\max = 2/50 = 0.04$. Just below it the iterates zig-zag in $y$ but converge; just above, they diverge.

## References

- Y. Nesterov, *Lectures on Convex Optimization*, 2nd ed., Springer (2018).
- S. Boyd & L. Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), Cambridge (2004), Ch. 9.
- L. Bottou, F. Curtis, J. Nocedal, [Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/abs/1606.04838) (SIAM Review, 2018).
- D. Kingma & J. Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (ICLR 2015).
- I. Loshchilov & F. Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (ICLR 2019).
- G. Goh, [Why Momentum Really Works](https://distill.pub/2017/momentum/) (Distill, 2017).
- S. McCandlish et al., [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) (2018).
- J. Cohen et al., [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065) (ICLR 2021).
- K. Jordan, [Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) (2024).
