# 5. Optimization

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 1](linear-algebra.md) (eigenvalues), [Ch. 2](calculus.md) (gradients, Taylor, Hessian), [Ch. 3](probability.md) (expectation, variance). **You will learn:** gradient descent and its convergence proofs (smooth, convex, strongly convex, non-convex), why ill-conditioning slows it down, stochastic gradient descent and batch size, momentum and its $\sqrt\kappa$ speed-up, Adam derived from scratch with bias correction, why AdamW decouples weight decay, warmup and cosine/WSD schedules, gradient clipping, and newer optimizers such as Muon.
    **Why it matters:** every LLM is trained by AdamW (or a close relative) with warmup, a decay schedule and gradient clipping. These choices decide whether a \$10M training run converges or diverges. You should know *why* each piece is there.

!!! watch "Watch"
    - 3Blue1Brown, [Gradient descent, how neural networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w).
    - Andrew Ng (DeepLearning.AI), [Gradient descent with momentum](https://www.youtube.com/watch?v=k8fTYJPd3_I) and [Adam optimization algorithm](https://www.youtube.com/watch?v=JXQT_vxqwIs). Short and clear.

    Reading: G. Goh, [Why Momentum Really Works](https://distill.pub/2017/momentum/) (Distill, 2017), an interactive masterpiece. S. Boyd & L. Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), Ch. 9. Goodfellow et al., [*Deep Learning*](https://www.deeplearningbook.org/), Ch. 8.

---

## 5.1 The problem

Training means solving

$$
\min_{\theta\in\R^n}\ \mathcal{L}(\theta) = \frac1N\sum_{i=1}^N\ell(\theta;\mathbf{x}^{(i)}),
$$

the average loss over the training set (**empirical risk**). For an LLM, $n\sim10^{9}$–$10^{12}$, $N\sim10^{13}$ tokens, and $\mathcal{L}$ is wildly non-convex. Two facts make it workable anyway: the gradient $\nabla\mathcal{L}$ is cheap (one backward pass, about 2× the cost of a forward pass, Ch. 7), and in very high dimension most critical points are saddle points rather than bad local minima ([Dauphin et al., 2014](https://arxiv.org/abs/1406.2572)), so following the gradient keeps making progress.

## 5.2 Gradient descent and the descent lemma

**Gradient descent (GD):** $\theta_{k+1} = \theta_k - \eta\nabla\mathcal{L}(\theta_k)$. By Theorem 2.3 this steps in the direction of steepest local decrease. But how big can $\eta$ be?

!!! definition "Definition 5.1 ($L$-smoothness)"
    $f$ is **$L$-smooth** if its gradient is $L$-Lipschitz: $\lVert\nabla f(\mathbf{x})-\nabla f(\mathbf{y})\rVert\le L\lVert\mathbf{x}-\mathbf{y}\rVert$. For twice-differentiable $f$ this is equivalent to every eigenvalue of the Hessian lying in $[-L, L]$. $L$ bounds the curvature.

!!! theorem "Lemma 5.1 (Descent lemma)"
    If $f$ is $L$-smooth, then $f(\mathbf{y})\le f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}) + \frac L2\lVert\mathbf{y}-\mathbf{x}\rVert^2$. Consequently one GD step with $\eta\le1/L$ satisfies

    $$
    f(\theta_{k+1})\le f(\theta_k) - \frac{\eta}{2}\lVert\nabla f(\theta_k)\rVert^2 .
    $$

??? proof "Proof"
    By the fundamental theorem of calculus along the segment $\mathbf{x}+t(\mathbf{y}-\mathbf{x})$:

    $$
    f(\mathbf{y}) - f(\mathbf{x}) - \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}) = \int_0^1\big(\nabla f(\mathbf{x}+t(\mathbf{y}-\mathbf{x}))-\nabla f(\mathbf{x})\big)^\top(\mathbf{y}-\mathbf{x})\,dt \le \int_0^1Lt\lVert\mathbf{y}-\mathbf{x}\rVert^2dt = \frac L2\lVert\mathbf{y}-\mathbf{x}\rVert^2,
    $$

    using Cauchy–Schwarz and smoothness. Now set $\mathbf{y}-\mathbf{x} = -\eta\nabla f$: $f(\theta_{k+1})\le f(\theta_k) - \eta\lVert\nabla f\rVert^2 + \frac{L\eta^2}{2}\lVert\nabla f\rVert^2 = f(\theta_k) - \eta(1-\tfrac{L\eta}{2})\lVert\nabla f\rVert^2$, and $1-\frac{L\eta}2\ge\frac12$ when $\eta\le1/L$. $\blacksquare$

!!! intuition "Intuition"
    The gradient tells you the slope; $L$ tells you how fast the slope can change. If you step further than about $1/L$, the ground may curve back up under you and the step can *increase* the loss. GD is guaranteed to go downhill only when the step size respects the curvature.

!!! theorem "Theorem 5.2 (GD on non-convex functions)"
    If $f$ is $L$-smooth and bounded below by $f^*$, GD with $\eta = 1/L$ satisfies

    $$
    \min_{0\le k<K}\lVert\nabla f(\theta_k)\rVert^2\le\frac{2L\big(f(\theta_0)-f^*\big)}{K}.
    $$

    So GD finds an approximately stationary point ($\lVert\nabla f\rVert\le\epsilon$) in $O(1/\epsilon^2)$ steps.

??? proof "Proof"
    Sum the descent lemma over $k=0..K-1$ (telescoping): $\frac{1}{2L}\sum_k\lVert\nabla f(\theta_k)\rVert^2\le f(\theta_0)-f(\theta_K)\le f(\theta_0)-f^*$. The minimum is at most the average. $\blacksquare$

!!! theorem "Theorem 5.3 (GD on convex and strongly convex functions)"
    Let $f$ be convex and $L$-smooth with minimizer $\theta^*$, and $\eta = 1/L$. Then

    $$
    f(\theta_K)-f(\theta^*)\le\frac{L\lVert\theta_0-\theta^*\rVert^2}{2K}.
    $$

    If additionally $f$ is **$\mu$-strongly convex** (all Hessian eigenvalues $\ge\mu>0$), the error shrinks geometrically: $\lVert\theta_K-\theta^*\rVert^2\le(1-\mu/L)^K\lVert\theta_0-\theta^*\rVert^2$.

??? proof "Proof (convex case)"
    Convexity gives $f(\theta_k)\le f^* + \nabla f(\theta_k)^\top(\theta_k-\theta^*)$. Combined with the descent lemma ($\eta = 1/L$, $\mathbf{g} = \nabla f(\theta_k)$):

    $$
    f(\theta_{k+1})\le f^* + \mathbf{g}^\top(\theta_k-\theta^*) - \frac{1}{2L}\lVert\mathbf{g}\rVert^2 = f^* + \frac L2\Big(\lVert\theta_k-\theta^*\rVert^2 - \lVert\theta_k-\theta^*-\tfrac1L\mathbf{g}\rVert^2\Big),
    $$

    by completing the square. The last vector is $\theta_{k+1}-\theta^*$. Summing over $k$ telescopes: $\sum_{k=1}^K(f(\theta_k)-f^*)\le\frac L2\lVert\theta_0-\theta^*\rVert^2$. Since $f(\theta_k)$ is non-increasing, $K(f(\theta_K)-f^*)$ is at most that sum. $\blacksquare$

    *(Strongly convex case.)* For a quadratic this is the eigen-analysis in §5.3 with rate $\max_i|1-\lambda_i/L|\le1-\mu/L$. The general proof uses the inequality $(\nabla f(\mathbf{x})-\nabla f(\mathbf{y}))^\top(\mathbf{x}-\mathbf{y})\ge\mu\lVert\mathbf{x}-\mathbf{y}\rVert^2$; see Boyd & Vandenberghe §9.3.

## 5.3 Conditioning: why GD zig-zags

All of the behavior of GD is visible on a quadratic, $f(\theta) = \frac12\theta^\top\mathbf{A}\theta$ with $\mathbf{A}$ symmetric positive definite and eigen-decomposition $\mathbf{A} = \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$ (Thm 1.10). Near any minimum, a smooth loss looks like this by Taylor's theorem (Thm 2.9), with $\mathbf{A}$ the Hessian.

The gradient is $\mathbf{A}\theta$, so $\theta_{k+1} = (\mathbf{I}-\eta\mathbf{A})\theta_k$. In the eigenbasis $\mathbf{c} = \mathbf{Q}^\top\theta$ the coordinates **decouple**:

$$
c_{i,k+1} = (1-\eta\lambda_i)\,c_{i,k}\quad\Longrightarrow\quad c_{i,k} = (1-\eta\lambda_i)^k\,c_{i,0}.
$$

!!! theorem "Theorem 5.4 (GD on a quadratic)"
    GD converges iff $|1-\eta\lambda_i|<1$ for all $i$, i.e. **$\eta<2/\lambda_\max$**. The best fixed step is $\eta^* = \frac{2}{\lambda_\min+\lambda_\max}$, giving the contraction factor

    $$
    \rho = \frac{\kappa-1}{\kappa+1},\qquad \kappa = \frac{\lambda_\max}{\lambda_\min}\ \text{(the condition number)}.
    $$

    Reaching accuracy $\epsilon$ takes about $\frac{\kappa}{2}\log\frac1\epsilon$ steps.

??? proof "Proof"
    The slowest coordinate decides the rate, so we want to minimize $\max_i|1-\eta\lambda_i|$. That maximum is attained at either $\lambda_\min$ or $\lambda_\max$, and it is minimized when the two are balanced: $1-\eta\lambda_\min = -(1-\eta\lambda_\max)$, giving $\eta^* = 2/(\lambda_\min+\lambda_\max)$ and $\rho = \frac{\lambda_\max-\lambda_\min}{\lambda_\max+\lambda_\min}$. For the step count, $\rho^k\le\epsilon$ needs $k\ge\log(1/\epsilon)/\log(1/\rho)$, and $\log\frac1\rho = \log\frac{1+1/\kappa}{1-1/\kappa}\approx\frac2\kappa$. $\blacksquare$

The learning rate is capped by the **sharpest** direction ($\lambda_\max$), while progress in the **flattest** direction ($\lambda_\min$) is what takes time. With $\kappa = 1000$ the step that is safe for the steep direction barely moves the shallow one. GD bounces across the steep valley walls and creeps along the valley floor. Deep-network Hessians are extremely ill-conditioned. That is the motivation for momentum and for adaptive methods.

<div class="widget" data-widget="gradient-descent"></div>

Try the ill-conditioned bowl with $\kappa = 30$: push SGD's $\eta$ until it oscillates, then compare momentum and Adam.

!!! llm "The edge of stability"
    Cohen et al., [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065) (2021), found that during full-batch training the sharpness $\lambda_\max$ rises until it hovers right at $2/\eta$, the stability boundary of Theorem 5.4. The network adapts its curvature to the learning rate. The loss keeps decreasing non-monotonically. This is one reason learning-rate warmup and decay matter so much.

## 5.4 Stochastic gradient descent

The full gradient averages over $N\sim10^{13}$ examples, which is too expensive per step. Instead sample a **mini-batch** $\mathcal{B}$ of size $B$ and use

$$
\hat{\mathbf{g}} = \frac1B\sum_{i\in\mathcal{B}}\nabla\ell(\theta;\mathbf{x}^{(i)}),\qquad \E[\hat{\mathbf{g}}] = \nabla\mathcal{L}(\theta),\qquad \operatorname{Cov}(\hat{\mathbf{g}}) = \frac{\boldsymbol\Sigma}{B},
$$

where $\boldsymbol\Sigma$ is the per-example gradient covariance. The estimate is unbiased, and its noise falls like $1/\sqrt B$ (§3.6).

!!! theorem "Theorem 5.5 (SGD convergence, non-convex)"
    Let $f$ be $L$-smooth, bounded below by $f^*$, and let the stochastic gradient have variance $\E\lVert\hat{\mathbf{g}}-\nabla f\rVert^2\le\sigma^2$. With constant step $\eta\le1/L$,

    $$
    \frac1K\sum_{k<K}\E\lVert\nabla f(\theta_k)\rVert^2\le\frac{2(f(\theta_0)-f^*)}{\eta K} + L\eta\sigma^2.
    $$

??? proof "Proof"
    Apply the smoothness inequality (Lemma 5.1, first part) with the step $-\eta\hat{\mathbf{g}}$ and take expectations conditioned on $\theta_k$. Since $\E\hat{\mathbf{g}} = \nabla f$ and $\E\lVert\hat{\mathbf{g}}\rVert^2 = \lVert\nabla f\rVert^2+\E\lVert\hat{\mathbf{g}}-\nabla f\rVert^2$:

    $$
    \E f(\theta_{k+1})\le f(\theta_k) - \eta\lVert\nabla f\rVert^2 + \frac{L\eta^2}{2}\big(\lVert\nabla f\rVert^2+\sigma^2\big)\le f(\theta_k)-\frac\eta2\lVert\nabla f\rVert^2 + \frac{L\eta^2\sigma^2}{2}.
    $$

    Sum over $k$, telescope, and divide by $\eta K/2$. $\blacksquare$

The bound has two terms. The first shrinks with more steps. The second is a **noise floor** proportional to $\eta\sigma^2$: with a constant learning rate, SGD bounces around in a region whose size scales with $\eta$. To converge you must eventually **decrease $\eta$** (classically, the Robbins–Monro conditions $\sum\eta_k = \infty$, $\sum\eta_k^2<\infty$), or **increase $B$** to shrink $\sigma^2$. This is the theory behind learning-rate decay (§5.7), and why the loss drops sharply at the end of a cosine or WSD schedule.

!!! llm "Batch size and the critical batch size"
    Doubling $B$ halves the gradient variance but doubles the compute per step. While noise dominates, a 2× bigger batch lets you take about 2× fewer steps (perfect scaling). Beyond the **critical batch size**, the gradient is already accurate and bigger batches waste compute. McCandlish et al., [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) (2018), estimate it with the **gradient noise scale** $B_\text{noise} = \tr(\boldsymbol\Sigma)/\lVert\nabla\mathcal{L}\rVert^2$. It grows as the loss falls, which is why LLM runs often **ramp up the batch size** during training. GPT-3 175B used a batch of 3.2M tokens; Llama 3 405B grew from 4M to 16M tokens.

## 5.5 Momentum

**Heavy-ball momentum** (Polyak, 1964) keeps a running velocity:

$$
\mathbf{v}_{k+1} = \mu\mathbf{v}_k + \nabla f(\theta_k),\qquad \theta_{k+1} = \theta_k - \eta\,\mathbf{v}_{k+1}.
$$

Unrolled, $\mathbf{v}_{k+1} = \sum_{j\le k}\mu^{k-j}\nabla f(\theta_j)$: an exponentially weighted sum of past gradients. In a consistent direction the effective step becomes $\eta/(1-\mu)$ (10× for $\mu = 0.9$). In directions where the gradient flips sign every step, the contributions cancel. So momentum **accelerates along the valley floor and damps the zig-zag across it**.

!!! theorem "Theorem 5.6 (Momentum on a quadratic)"
    On $f = \frac12\theta^\top\mathbf{A}\theta$ with eigenvalues in $[\mu_A, L]$, heavy-ball with tuned $\eta = \frac{4}{(\sqrt L+\sqrt{\mu_A})^2}$ and $\mu = \left(\frac{\sqrt\kappa-1}{\sqrt\kappa+1}\right)^2$ converges at the rate

    $$
    \rho = \frac{\sqrt\kappa-1}{\sqrt\kappa+1},
    $$

    so it needs about $\frac{\sqrt\kappa}{2}\log\frac1\epsilon$ steps instead of $\frac\kappa2\log\frac1\epsilon$. For $\kappa = 10^4$, that's 100× fewer steps.

??? proof "Proof sketch"
    In the eigenbasis each coordinate obeys the second-order recurrence $c_{k+1} = (1+\mu-\eta\lambda)c_k - \mu c_{k-1}$, i.e. the $2\times2$ linear system $\begin{psmallmatrix}c_{k+1}\\c_k\end{psmallmatrix} = \begin{psmallmatrix}1+\mu-\eta\lambda & -\mu\\ 1 & 0\end{psmallmatrix}\begin{psmallmatrix}c_k\\c_{k-1}\end{psmallmatrix}$. Its eigenvalues $z$ solve $z^2-(1+\mu-\eta\lambda)z+\mu = 0$. When the discriminant is negative, the roots are complex with $|z| = \sqrt\mu$ (product of roots $=\mu$). The given $\eta,\mu$ are the choice that makes the discriminant non-positive for *every* $\lambda\in[\mu_A,L]$ with the smallest possible $\mu$, which gives $|z| = \sqrt\mu = \frac{\sqrt\kappa-1}{\sqrt\kappa+1}$. Goh's Distill article animates exactly this analysis. $\blacksquare$

PyTorch's `SGD(momentum=μ)` uses exactly this form ($\mathbf{v}\leftarrow\mu\mathbf{v}+\mathbf{g}$, $\theta\leftarrow\theta-\eta\mathbf{v}$). Our implementation matches it to 6 decimal places in the tests:

```python
--8<-- "code/optim.py:sgd"
```

**Nesterov momentum** evaluates the gradient at the look-ahead point $\theta_k-\eta\mu\mathbf{v}_k$. It achieves the optimal $\sqrt\kappa$ rate for *all* smooth strongly convex functions, not just quadratics (Nesterov, 1983).

## 5.6 Adaptive methods: from AdaGrad to Adam

Momentum fixes conditioning only partly. A different idea is to **give every parameter its own learning rate**, scaled by how large its gradients typically are.

- **AdaGrad** (Duchi et al., 2011): divide by the root of the *sum* of all past squared gradients. Good for sparse features, but the effective step decays to zero.
- **RMSProp** (Hinton, 2012): use an exponential *moving average* of squared gradients instead, so the scale adapts and doesn't freeze.
- **Adam** (Kingma & Ba, 2015): RMSProp plus momentum plus bias correction.

!!! definition "Algorithm 5.2 (Adam)"
    With $\mathbf{g}_t = \nabla\mathcal{L}(\theta_{t-1})$ (on a mini-batch), $\mathbf{m}_0 = \mathbf{v}_0 = \mathbf{0}$, and all operations elementwise:

    $$
    \begin{aligned}
    \mathbf{m}_t &= \beta_1\mathbf{m}_{t-1} + (1-\beta_1)\,\mathbf{g}_t &&\text{(1st moment: mean of } \mathbf{g})\\
    \mathbf{v}_t &= \beta_2\mathbf{v}_{t-1} + (1-\beta_2)\,\mathbf{g}_t^2 &&\text{(2nd moment: mean of } \mathbf{g}^2)\\
    \hat{\mathbf{m}}_t &= \mathbf{m}_t/(1-\beta_1^t),\qquad \hat{\mathbf{v}}_t = \mathbf{v}_t/(1-\beta_2^t) &&\text{(bias correction)}\\
    \theta_t &= \theta_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t}+\epsilon}.
    \end{aligned}
    $$

!!! theorem "Proposition 5.7 (Why bias correction)"
    If the gradients have a stationary distribution with $\E[\mathbf{g}_t] = \mathbf{g}$ and $\E[\mathbf{g}_t^2] = \mathbf{s}$, then $\E[\mathbf{m}_t] = (1-\beta_1^t)\,\mathbf{g}$ and $\E[\mathbf{v}_t] = (1-\beta_2^t)\,\mathbf{s}$. Dividing by $1-\beta^t$ makes the estimates unbiased.

??? proof "Proof"
    Unrolling the recursion from $\mathbf{m}_0 = \mathbf{0}$: $\mathbf{m}_t = (1-\beta_1)\sum_{j=1}^t\beta_1^{t-j}\mathbf{g}_j$. Taking expectations, $\E\mathbf{m}_t = (1-\beta_1)\,\mathbf{g}\sum_{j=0}^{t-1}\beta_1^j = (1-\beta_1)\,\mathbf{g}\,\frac{1-\beta_1^t}{1-\beta_1} = (1-\beta_1^t)\,\mathbf{g}$. The same holds for $\mathbf{v}_t$. $\blacksquare$

    Without the correction, with $\beta_2 = 0.999$, $\mathbf{v}_1 = 0.001\,\mathbf{g}_1^2$, and the first step would be about $\sqrt{1000}\approx32$ times too large.

**What Adam's update means.** Ignoring $\epsilon$, the update for parameter $i$ is $\eta\cdot\hat m_i/\sqrt{\hat v_i}$. This ratio is like a **signal-to-noise ratio**: if every gradient has the same sign, $|\hat m_i|\approx\sqrt{\hat v_i}$ and the step is about $\eta$, *whatever the gradient's scale*. If the gradient is mostly noise, $|\hat m_i|\ll\sqrt{\hat v_i}$ and the step is small. Adam is **invariant to rescaling the loss or any single parameter's gradient**, and the learning rate becomes roughly "the maximum change per parameter per step". In the quadratic picture, dividing by $\sqrt{v_i}$ is a diagonal approximation to rescaling by the curvature, which undoes some of the ill-conditioning.

### AdamW: decoupled weight decay

L2 regularization adds $\frac\lambda2\lVert\theta\rVert^2$ to the loss, i.e. $\lambda\theta$ to the gradient. For plain SGD this is the same as multiplying the weights by $(1-\eta\lambda)$ each step, "weight decay". **For Adam they differ**: the $\lambda\theta$ term passes through the $1/\sqrt{\hat v}$ normalization, so weights with large gradient variance get *less* regularization. Loshchilov & Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (2019), proposed applying the decay directly:

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

    **Memory cost:** AdamW stores $\mathbf{m}$ and $\mathbf{v}$, two extra numbers per parameter, usually in fp32. With fp32 master weights and bf16 gradients that's about 16 bytes per parameter (Ch. 16): a 70B model needs about 1.1 TB just for the weights, gradients and optimizer state, before activations.

## 5.7 Learning-rate schedules

LLMs are never trained with a constant learning rate. The standard recipe has two parts.

1. **Linear warmup** over the first ~0.1–2% of steps (GPT-3: 375M tokens; nanoGPT: 2,000 steps). Early on, Adam's $\hat{\mathbf{v}}$ is estimated from very few gradients and is noisy, the network sits in a high-curvature region, and gradient norms are large. Warmup keeps early updates small until the statistics settle and the curvature drops ([Liu et al., 2020](https://arxiv.org/abs/1908.03265); [Gilmer et al., 2021](https://arxiv.org/abs/2110.04369)).
2. **Decay** to a small final value, which removes the $\eta\sigma^2$ noise floor of Theorem 5.5:
    - **Cosine** decay to ~10% of peak ([Loshchilov & Hutter, 2017](https://arxiv.org/abs/1608.03983)). Used by GPT-3, Llama, nanoGPT.
    - **Warmup–Stable–Decay (WSD)**: constant LR for most of training, then a short, fast decay (often 10–20% of steps) at the end ([Hu et al., MiniCPM, 2024](https://arxiv.org/abs/2404.06395); [Hägele et al., 2024](https://arxiv.org/abs/2405.18392)). It matches cosine, and you don't need to fix the run length in advance, since you can branch off a decay from any checkpoint. DeepSeek-V3 and many 2024–2025 models use WSD-style schedules.

<div class="widget" data-widget="lr-schedule"></div>

```python
--8<-- "code/optim.py:schedule"
```

## 5.8 Gradient clipping

Occasional batches produce huge gradients (loss spikes). One such step can throw the model into a bad region it never recovers from. **Clipping by global norm** rescales the entire gradient (all parameters together) if its norm exceeds a threshold $c$:

$$
\mathbf{g}\leftarrow\mathbf{g}\cdot\min\Big(1,\ \frac{c}{\lVert\mathbf{g}\rVert}\Big),\qquad \lVert\mathbf{g}\rVert = \sqrt{\textstyle\sum_{\text{all params}}g^2}.
$$

The direction is preserved; only the length is capped. In PyTorch it's `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`, called after `backward()` and before `step()` (see the training loop in Ch. 14). Logging $\lVert\mathbf{g}\rVert$ every step is one of the best ways to monitor training health: a steadily rising gradient norm often comes before a divergence.

## 5.9 Beyond Adam

- **Second-order methods.** Newton's method, $\theta\leftarrow\theta-\mathbf{H}^{-1}\nabla f$, rescales every eigen-direction perfectly (on a quadratic it converges in one step, $\kappa$ disappears). But $\mathbf{H}$ has $n^2$ entries; with $n = 10^{10}$ it can't even be stored. Practical methods approximate it with structure.
- **Shampoo / SOAP** ([Gupta et al., 2018](https://arxiv.org/abs/1802.09568); [Vyas et al., 2024](https://arxiv.org/abs/2409.11321)) keep a Kronecker-factored preconditioner per weight matrix. Shampoo won the external tuning track of the AlgoPerf benchmark.
- **Muon** ([K. Jordan, 2024](https://kellerjordan.github.io/posts/muon/)) treats each 2-D weight matrix as a whole. It takes the momentum matrix $\mathbf{M} = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top$ and steps along its **orthogonalization** $\mathbf{U}\mathbf{V}^\top$ (all singular values set to 1, via a few Newton–Schulz iterations instead of an SVD). That's steepest descent under the spectral norm: every direction of the update gets equal magnitude, so rare directions aren't drowned out. Muon set NanoGPT speed-run records and was scaled to a 1T-parameter MoE in Moonshot's Kimi K2 ([Moonshot AI, 2025](https://arxiv.org/abs/2507.20534)).
- **Lion** ([Chen et al., 2023](https://arxiv.org/abs/2302.06675)): found by program search; uses only the *sign* of a momentum mix, so each parameter's update has magnitude exactly $\eta$, and it stores one state per parameter instead of two.
- **Schedule-free** optimizers ([Defazio et al., 2024](https://arxiv.org/abs/2405.15682)) replace the schedule with iterate averaging.

AdamW remains the safe default. When you read a new model's tech report, the optimizer section is now one of the places where real differences show up.

## Key takeaways

- GD is safe for $\eta<2/\lambda_\max$, and converges in about $\kappa\log\frac1\epsilon$ steps on a quadratic. Ill-conditioning ($\kappa\gg1$) is the core difficulty.
- SGD's noise creates a floor $\propto\eta\sigma^2$. Decaying the learning rate (or growing the batch) removes it.
- Momentum averages gradients over time: faster along consistent directions, damped across oscillating ones; $\sqrt\kappa$ instead of $\kappa$.
- Adam = momentum + per-parameter scaling by the RMS gradient + bias correction. Its step is about $\eta$ in units of "signal-to-noise".
- **AdamW** decouples weight decay from the adaptive scaling. LLM defaults: $\beta = (0.9, 0.95)$, $\lambda = 0.1$, warmup + cosine/WSD decay, clip at 1.0.

## Exercises

1. Show that for $f(x) = \frac\lambda2x^2$, GD diverges iff $\eta>2/\lambda$. What happens at exactly $\eta = 2/\lambda$?
2. For $\kappa = 100$, compute the GD rate $\frac{\kappa-1}{\kappa+1}$ and the momentum rate $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$. How many steps does each need to shrink the error by $10^{-6}$?
3. Show that, with $\epsilon = 0$, Adam's first update is $\theta_1 = \theta_0-\eta\operatorname{sign}(\mathbf{g}_1)$.
4. Explain why L2 regularization and weight decay are identical for SGD but not for Adam. Give a two-parameter example where they produce different updates.
5. Implement the update of `AdamW` above without bias correction and train the GPT in Ch. 14 for 200 steps. Compare the first 20 losses with the corrected version.
6. **Code:** in the gradient-descent widget, set $\kappa = 50$ and find the largest SGD learning rate that still converges. Compare it with $2/\kappa$ (here $\lambda_\max = \kappa$).

??? success "Solutions"
    1. $x_{k+1} = (1-\eta\lambda)x_k$, which diverges iff $|1-\eta\lambda|>1$, i.e. $\eta>2/\lambda$. At $\eta = 2/\lambda$, $x_k = (-1)^kx_0$: it oscillates forever without converging.
    2. GD: $\rho = 99/101\approx0.980$, so about $\ln(10^6)/(-\ln0.980)\approx 690$ steps. Momentum: $\rho = 9/11\approx0.818$, so about $69$ steps.
    3. $\hat m_1 = \mathbf{g}_1$ and $\hat v_1 = \mathbf{g}_1^2$, so $\hat m_1/\sqrt{\hat v_1} = \mathbf{g}_1/|\mathbf{g}_1| = \operatorname{sign}(\mathbf{g}_1)$.
    4. For SGD, $\theta - \eta(\mathbf{g}+\lambda\theta) = (1-\eta\lambda)\theta-\eta\mathbf{g}$, the same thing. For Adam, the $\lambda\theta$ term is divided by $\sqrt{\hat v}$. Take two parameters with the same value $\theta_1 = \theta_2 = 1$ but gradient RMS 10 and 0.1: L2-in-Adam shrinks the second 100× more than the first, while AdamW shrinks both by the same factor $(1-\eta\lambda)$.
    5. Without correction the first steps are tiny (scaled by $1-\beta_1$) relative to the denominator, which is also tiny but scaled by $\sqrt{1-\beta_2}$. The ratio is $(1-\beta_1)/\sqrt{1-\beta_2}$ at step 1, which is $0.1/\sqrt{0.05}\approx 0.45$ with $\beta_2 = 0.95$ (and $3.2$ with $\beta_2 = 0.999$). You'll see slower or more erratic early progress.
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
