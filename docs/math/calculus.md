# 2. Calculus & Matrix Calculus

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 1](linear-algebra.md). **You will learn:** derivatives as linear approximations, the chain rule (with proof), gradients and why they point uphill, Jacobians, vector–Jacobian products, all the matrix-calculus identities a Transformer needs (linear layers, softmax, cross-entropy), Taylor expansions and the Hessian, numerical gradient checking, and the Gaussian integral.
    **Why it matters:** training = following gradients. Backpropagation (Ch. 7) is *nothing but* the chain rule applied to a computation graph, organized as vector–Jacobian products. If you know this chapter, you can derive the gradient of any layer by hand.

!!! watch "Watch"
    3Blue1Brown, *Essence of Calculus*. Start with [Visualizing the chain rule and product rule](https://www.youtube.com/watch?v=YG15m2VwSjA).

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/videoseries?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr" allowfullscreen loading="lazy"></iframe></div>

    Reading: T. Parr & J. Howard, [*The Matrix Calculus You Need For Deep Learning*](https://arxiv.org/abs/1802.01528) (2018); Deisenroth et al., [*MML*](https://mml-book.github.io/) Ch. 5.

---

## 2.1 The derivative is the best linear approximation

!!! definition "Definition 2.1 (Derivative)"
    $f:\R\to\R$ is differentiable at $x$ if the limit

    $$
    f'(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}
    $$

    exists. Equivalently, $f(x+h) = f(x) + f'(x)\,h + o(h)$, where $o(h)/h\to0$.

The second form is the one to remember: **near $x$, $f$ looks like a line with slope $f'(x)$**. "If I nudge the input by $h$, the output moves by about $f'(x)h$."

| $f(x)$ | $f'(x)$ | | $f(x)$ | $f'(x)$ |
|---|---|---|---|---|
| $x^n$ | $nx^{n-1}$ | | $\log x$ | $1/x$ |
| $e^x$ | $e^x$ | | $\sigma(x)=\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ |
| $\sin x$ | $\cos x$ | | $\tanh x$ | $1-\tanh^2x$ |
| $f+g$ | $f'+g'$ | | $fg$ | $f'g+fg'$ (product rule) |

??? proof "Proof of the product rule"
    $f(x+h)g(x+h) = \big(f + f'h + o(h)\big)\big(g + g'h + o(h)\big) = fg + (f'g+fg')h + o(h)$, so the linear coefficient is $f'g + fg'$. $\blacksquare$

## 2.2 The chain rule

!!! theorem "Theorem 2.1 (Chain rule)"
    If $g$ is differentiable at $x$ and $f$ at $g(x)$, then $(f\circ g)'(x) = f'(g(x))\,g'(x)$.

??? proof "Proof (via linear approximations)"
    Let $u = g(x)$ and $k = g(x+h)-g(x) = g'(x)h + o(h)$. Then

    $$
    f(g(x+h)) = f(u + k) = f(u) + f'(u)k + o(k) = f(u) + f'(u)g'(x)h + o(h),
    $$

    because $f'(u)\,o(h)$ and $o(k)$ are both $o(h)$, the latter since $k = O(h)$. The coefficient of $h$ is $f'(g(x))g'(x)$. $\blacksquare$

In words: **sensitivities multiply along a chain.** If $u$ changes 3× as fast as $x$, and $y$ changes 2× as fast as $u$, then $y$ changes 6× as fast as $x$.

!!! theorem "Example 2.2 (Sigmoid derivative)"
    $\sigma'(x) = \sigma(x)(1-\sigma(x))$.

??? proof "Proof"
    $\sigma(x) = (1+e^{-x})^{-1}$. By the chain rule, $\sigma'(x) = -(1+e^{-x})^{-2}\cdot(-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2} = \frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}} = \sigma(x)\,(1-\sigma(x))$. $\blacksquare$

Note that the derivative is expressed through the *output* $\sigma(x)$. Backprop implementations exploit this: they store the forward output and reuse it (see `tanh` in micrograd, Ch. 7).

## 2.3 Partial derivatives and the gradient

For $f:\R^n\to\R$, the **partial derivative** $\partial f/\partial x_i$ is the ordinary derivative with respect to $x_i$, holding the other coordinates fixed. The **gradient** collects them:

$$
\nabla f(\mathbf{x}) = \left(\frac{\partial f}{\partial x_1},\dots,\frac{\partial f}{\partial x_n}\right)^\top\in\R^n,
\qquad
f(\mathbf{x}+\mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top\mathbf{h} + o(\lVert\mathbf{h}\rVert).
$$

!!! theorem "Theorem 2.3 (The gradient is the direction of steepest ascent)"
    Among all unit directions $\mathbf{u}$, the directional derivative $D_\mathbf{u}f = \nabla f^\top\mathbf{u}$ is maximized by $\mathbf{u} = \nabla f/\lVert\nabla f\rVert$, with value $\lVert\nabla f\rVert$. It is minimized by the opposite direction, and it is zero for directions orthogonal to $\nabla f$ (along the level set).

??? proof "Proof"
    By Cauchy–Schwarz (Thm 1.2), $\nabla f^\top\mathbf{u}\le\lVert\nabla f\rVert\lVert\mathbf{u}\rVert = \lVert\nabla f\rVert$, with equality iff $\mathbf{u}$ is a positive multiple of $\nabla f$. $\blacksquare$

!!! intuition "Intuition"
    Stand on a hillside in fog. The gradient is the compass direction in which the ground rises fastest, and its length is how steep that is. **Gradient descent** steps the other way: $\mathbf{x}\leftarrow\mathbf{x}-\eta\nabla f(\mathbf{x})$. Contour lines are perpendicular to the gradient. You'll see this in the optimizer widget in Ch. 5.

## 2.4 Jacobians and the multivariate chain rule

For $\mathbf{f}:\R^n\to\R^m$, the **Jacobian** $\mathbf{J}\in\R^{m\times n}$ has $J_{ij} = \partial f_i/\partial x_j$. It is the matrix of the best linear approximation: $\mathbf{f}(\mathbf{x}+\mathbf{h}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J}\mathbf{h}$.

!!! theorem "Theorem 2.4 (Multivariate chain rule)"
    For $\mathbf{y} = \mathbf{f}(\mathbf{u})$ and $\mathbf{u} = \mathbf{g}(\mathbf{x})$:

    $$
    \frac{\partial\mathbf{y}}{\partial\mathbf{x}} = \frac{\partial\mathbf{y}}{\partial\mathbf{u}}\,\frac{\partial\mathbf{u}}{\partial\mathbf{x}}
    \qquad\text{i.e.}\qquad
    \frac{\partial y_i}{\partial x_j} = \sum_k\frac{\partial y_i}{\partial u_k}\frac{\partial u_k}{\partial x_j}.
    $$

    The influence of $x_j$ on $y_i$ is the **sum over all paths** through the intermediate variables $u_k$.

??? proof "Proof"
    Compose the linear approximations: $\mathbf{u}(\mathbf{x}+\mathbf{h}) \approx \mathbf{u} + \mathbf{J}_g\mathbf{h}$, then $\mathbf{y}\approx\mathbf{y}(\mathbf{u}) + \mathbf{J}_f\mathbf{J}_g\mathbf{h}$. The composition of linear maps is the matrix product (Thm 1.5), and the error terms vanish exactly as in Thm 2.1. $\blacksquare$

### The vector–Jacobian product (VJP): the heart of backprop

In deep learning the final output is a **scalar loss** $\mathcal{L}$. Suppose $\mathbf{y} = \mathbf{f}(\mathbf{x})$ is one step in the middle of the network and we already know $\mathbf{g}_\mathbf{y} = \nabla_\mathbf{y}\mathcal{L}$. The chain rule gives

$$
\boxed{\;\nabla_\mathbf{x}\mathcal{L} = \mathbf{J}^\top\,\nabla_\mathbf{y}\mathcal{L}\;}
\qquad\text{where } \mathbf{J} = \frac{\partial\mathbf{y}}{\partial\mathbf{x}}.
$$

(Take Theorem 2.4 with the scalar $\mathcal{L}$ as the output: $\frac{\partial\mathcal{L}}{\partial x_j} = \sum_i\frac{\partial\mathcal{L}}{\partial y_i}J_{ij}$, which is the $j$-th entry of $\mathbf{J}^\top\mathbf{g}_\mathbf{y}$.)

This is the most important equation in this chapter. **Backprop never forms Jacobians.** Each operation only needs to know how to multiply an incoming gradient by its transposed Jacobian, and that is usually far cheaper than building $\mathbf{J}$. For an elementwise function $y_i = \phi(x_i)$, $\mathbf{J} = \diag(\phi'(\mathbf{x}))$, so the VJP is just $\phi'(\mathbf{x})\odot\mathbf{g}_\mathbf{y}$, an $O(n)$ operation rather than an $n\times n$ matrix.

!!! llm "Why reverse mode?"
    Evaluating the chain $\mathbf{J}_L\cdots\mathbf{J}_2\mathbf{J}_1$ from the **output end** (vector times matrix, repeatedly) costs about as much as one forward pass when the output is a scalar. Evaluating from the input end would cost $n$ times more for $n$ parameters. With $n = 10^{11}$ parameters, this is the difference between possible and impossible. This is **reverse-mode automatic differentiation**, and Ch. 7 builds it.

## 2.5 Matrix calculus you will actually use

All results use the gradient convention from [Notation](../start/notation.md): $\nabla_\mathbf{W}\mathcal{L}$ has the same shape as $\mathbf{W}$. The universal technique is to **write the scalar in index notation, differentiate one entry, and re-assemble.**

!!! theorem "Identities 2.5"
    1. $\nabla_\mathbf{x}(\mathbf{a}^\top\mathbf{x}) = \mathbf{a}$
    2. $\nabla_\mathbf{x}(\mathbf{x}^\top\mathbf{A}\mathbf{x}) = (\mathbf{A}+\mathbf{A}^\top)\mathbf{x}$, which is $2\mathbf{A}\mathbf{x}$ if $\mathbf{A}$ is symmetric
    3. $\nabla_\mathbf{x}\lVert\mathbf{x}\rVert^2 = 2\mathbf{x}$
    4. $\nabla_\mathbf{x}\lVert\mathbf{x}\rVert = \mathbf{x}/\lVert\mathbf{x}\rVert$

??? proof "Proofs"
    1. $\partial(\sum_ia_ix_i)/\partial x_k = a_k$.
    2. $\partial(\sum_{i,j}x_iA_{ij}x_j)/\partial x_k = \sum_jA_{kj}x_j + \sum_ix_iA_{ik} = (\mathbf{A}\mathbf{x})_k + (\mathbf{A}^\top\mathbf{x})_k$.
    3. Identity 2 with $\mathbf{A}=\mathbf{I}$.
    4. Chain rule: $\nabla\sqrt{s} = \frac{1}{2\sqrt s}\nabla s$ with $s = \lVert\mathbf{x}\rVert^2$. $\blacksquare$

!!! theorem "Theorem 2.6 (Gradients of a linear layer)"
    Let $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ with $\mathbf{W}\in\R^{m\times n}$, and let $\mathbf{g} = \nabla_\mathbf{y}\mathcal{L}\in\R^m$. Then

    $$
    \nabla_\mathbf{x}\mathcal{L} = \mathbf{W}^\top\mathbf{g},\qquad
    \nabla_\mathbf{W}\mathcal{L} = \mathbf{g}\,\mathbf{x}^\top,\qquad
    \nabla_\mathbf{b}\mathcal{L} = \mathbf{g}.
    $$

    **Batched** (rows are examples/tokens, as in PyTorch): $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top + \mathbf{1}\mathbf{b}^\top$ with $\mathbf{X}\in\R^{N\times n}$ and $\mathbf{G} = \nabla_\mathbf{Y}\mathcal{L}\in\R^{N\times m}$:

    $$
    \nabla_\mathbf{X}\mathcal{L} = \mathbf{G}\mathbf{W},\qquad
    \nabla_\mathbf{W}\mathcal{L} = \mathbf{G}^\top\mathbf{X},\qquad
    \nabla_\mathbf{b}\mathcal{L} = \mathbf{G}^\top\mathbf{1} \;(\text{sum of rows}).
    $$

??? proof "Proof"
    $y_i = \sum_j W_{ij}x_j + b_i$, so $\partial y_i/\partial x_j = W_{ij}$, $\partial y_i/\partial W_{kl} = \delta_{ik}x_l$, and $\partial y_i/\partial b_k = \delta_{ik}$. Summing over paths through $\mathbf{y}$:

    - $\frac{\partial\mathcal{L}}{\partial x_j} = \sum_i g_iW_{ij} = (\mathbf{W}^\top\mathbf{g})_j$;
    - $\frac{\partial\mathcal{L}}{\partial W_{kl}} = \sum_ig_i\delta_{ik}x_l = g_kx_l = (\mathbf{g}\mathbf{x}^\top)_{kl}$;
    - $\frac{\partial\mathcal{L}}{\partial b_k} = g_k$.

    In the batched case each row $\mathbf{y}^{(r)} = \mathbf{W}\mathbf{x}^{(r)}+\mathbf{b}$ contributes independently. $\nabla_\mathbf{W}$ sums $\mathbf{g}^{(r)}\mathbf{x}^{(r)\top}$ over rows, which is $\mathbf{G}^\top\mathbf{X}$. Row $r$ of $\nabla_\mathbf{X}$ is $(\mathbf{W}^\top\mathbf{g}^{(r)})^\top = \mathbf{g}^{(r)\top}\mathbf{W}$, so $\nabla_\mathbf{X} = \mathbf{G}\mathbf{W}$. $\blacksquare$

!!! tip "The shape trick"
    If you forget the formula, **let the shapes tell you.** $\nabla_\mathbf{W}$ must be $m\times n$, and the only way to make that from $\mathbf{G}\in\R^{N\times m}$ and $\mathbf{X}\in\R^{N\times n}$ is $\mathbf{G}^\top\mathbf{X}$. Note also that the backward pass of a linear layer is **two matmuls** ($\mathbf{G}\mathbf{W}$ and $\mathbf{G}^\top\mathbf{X}$), each as expensive as the forward matmul. So backward ≈ 2× forward FLOPs, which gives the "6" in $6ND$ (Ch. 15).

**Least squares revisited.** $\mathcal{L}(\mathbf{w}) = \lVert\mathbf{X}\mathbf{w}-\mathbf{y}\rVert^2 = \mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w} - 2\mathbf{y}^\top\mathbf{X}\mathbf{w} + \mathbf{y}^\top\mathbf{y}$. By Identities 2.5, $\nabla\mathcal{L} = 2\mathbf{X}^\top\mathbf{X}\mathbf{w} - 2\mathbf{X}^\top\mathbf{y}$, and setting it to zero recovers the normal equations from Ch. 1.

### Softmax and cross-entropy: the gradient that trains every LLM

!!! definition "Definition 2.2 (Softmax, log-sum-exp)"
    $$
    \softmax(\mathbf{z})_i = p_i = \frac{e^{z_i}}{\sum_{j=1}^{V}e^{z_j}},
    \qquad
    \operatorname{LSE}(\mathbf{z}) = \log\sum_{j}e^{z_j}.
    $$

    Note $\log p_i = z_i - \operatorname{LSE}(\mathbf{z})$.

Type your own logits below and drag the temperature $T$ (the widget computes $\softmax(\mathbf{z}/T)$; temperature is covered in Ch. 18). The entropy readout is explained in Ch. 4.

<div class="widget" data-widget="softmax"></div>

!!! theorem "Theorem 2.7 (Softmax Jacobian)"
    $$
    \frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j),\qquad\text{i.e.}\qquad \mathbf{J}_{\softmax} = \diag(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top .
    $$

    Also $\nabla_\mathbf{z}\operatorname{LSE}(\mathbf{z}) = \mathbf{p}$.

??? proof "Proof"
    Let $S = \sum_k e^{z_k}$, so $\partial S/\partial z_j = e^{z_j}$. By the quotient rule,

    $$
    \frac{\partial}{\partial z_j}\frac{e^{z_i}}{S} = \frac{\delta_{ij}e^{z_i}S - e^{z_i}e^{z_j}}{S^2} = p_i\delta_{ij} - p_ip_j .
    $$

    For LSE: $\partial\log S/\partial z_j = e^{z_j}/S = p_j$. $\blacksquare$

!!! theorem "Theorem 2.8 (Softmax + cross-entropy gradient)"
    For logits $\mathbf{z}\in\R^V$, target class $y$ (one-hot $\mathbf{y}$), and loss $\mathcal{L} = -\log p_y = -\log\softmax(\mathbf{z})_y$:

    $$
    \boxed{\;\nabla_\mathbf{z}\mathcal{L} = \mathbf{p} - \mathbf{y}\;}
    $$

??? proof "Proof"
    $\mathcal{L} = -z_y + \operatorname{LSE}(\mathbf{z})$. The first term has gradient $-\mathbf{y}$ and the second has gradient $\mathbf{p}$ (Thm 2.7). $\blacksquare$

!!! intuition "Intuition"
    The gradient on each logit is "**predicted probability minus target probability**". The correct token's logit is pushed up by $1-p_y$, and every wrong token's logit is pushed down in proportion to how much probability it stole. When the model is confident and right, $\mathbf{p}\approx\mathbf{y}$ and the gradient vanishes. This simple, well-conditioned gradient is why softmax + cross-entropy is used for *every* language model.

!!! warning "Numerical stability"
    Never compute $e^{z_i}$ directly: $e^{1000}$ overflows. Since $\softmax(\mathbf{z}) = \softmax(\mathbf{z} - c\mathbf{1})$ for any $c$ (the $e^{-c}$ cancels), implementations subtract $c=\max_i z_i$, and compute $\log p_y = z_y - \max\mathbf{z} - \log\sum_j e^{z_j-\max\mathbf{z}}$. PyTorch's `F.cross_entropy` takes **logits** and does this for you. Never apply softmax and then log yourself.

```python
import torch, torch.nn.functional as F
z = torch.randn(5, requires_grad=True)
y = torch.tensor(2)
loss = F.cross_entropy(z[None], y[None])
loss.backward()
p = F.softmax(z.detach(), -1)
print(torch.allclose(z.grad, p - F.one_hot(y, 5).float()))  # True: grad = p - y
```

## 2.6 Second order: Taylor expansion, Hessian, convexity

!!! theorem "Theorem 2.9 (Second-order Taylor expansion)"
    For twice-differentiable $f:\R^n\to\R$ with Hessian $\mathbf{H} = \nabla^2f(\mathbf{x})$, $H_{ij} = \partial^2f/\partial x_i\partial x_j$:

    $$
    f(\mathbf{x}+\mathbf{h}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top\mathbf{h} + \tfrac12\mathbf{h}^\top\mathbf{H}\mathbf{h} + o(\lVert\mathbf{h}\rVert^2).
    $$

The Hessian is symmetric (for $C^2$ functions), so by the spectral theorem it has orthogonal eigen-directions with real curvatures $\lambda_i$.

- At a **critical point** ($\nabla f = \mathbf{0}$): if $\mathbf{H}\succ0$ it is a local minimum (every direction curves up), if $\mathbf{H}\prec0$ a local maximum, and if there are mixed signs a **saddle point**. In high-dimensional neural-network losses, saddles are far more common than bad local minima.
- $f$ is **convex** iff $\mathbf{H}\succeq0$ everywhere, iff $f(\mathbf{y})\ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$ (the function lies above its tangents). For convex $f$, every critical point is a global minimum. Neural-network losses are **not** convex, but convex analysis still explains *local* optimizer behavior (Ch. 5).

!!! llm "LLM connection"
    The largest Hessian eigenvalue $\lambda_\max$ (the "sharpness") limits the stable learning rate: gradient descent diverges if $\eta > 2/\lambda_\max$ (proved in Ch. 5). Warmup, Adam's per-parameter scaling, and normalization layers can all be understood as ways of managing curvature.

## 2.7 Numerical gradient checking

!!! theorem "Proposition 2.10 (Centered differences are second-order accurate)"
    $\dfrac{f(x+h) - f(x-h)}{2h} = f'(x) + O(h^2)$, whereas the forward difference $\dfrac{f(x+h)-f(x)}{h} = f'(x) + O(h)$.

??? proof "Proof"
    Taylor: $f(x\pm h) = f \pm f'h + \tfrac12f''h^2 \pm \tfrac16f'''h^3 + O(h^4)$. Subtracting, the even terms cancel: $f(x+h)-f(x-h) = 2f'h + \tfrac13f'''h^3 + O(h^5)$. Divide by $2h$. $\blacksquare$

```python
import torch

def numerical_grad(f, x, h=1e-5):
    g = torch.zeros_like(x)
    for i in range(x.numel()):
        e = torch.zeros_like(x).view(-1); e[i] = h; e = e.view_as(x)
        g.view(-1)[i] = (f(x + e) - f(x - e)) / (2 * h)
    return g

torch.manual_seed(0)
W = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
x = torch.randn(4, dtype=torch.float64)
f = lambda W: torch.tanh(W @ x).pow(2).sum()
f(W).backward()
print(torch.allclose(W.grad, numerical_grad(f, W.detach()), atol=1e-8))  # True
```

Use `float64` for gradient checks: in `float32`, rounding errors of size $\epsilon/h$ swamp the $O(h^2)$ accuracy. PyTorch ships `torch.autograd.gradcheck` for exactly this.

## 2.8 Integrals (just enough for probability)

The **integral** $\int_a^bf(x)\,dx$ is the signed area under $f$. The **fundamental theorem of calculus** says that differentiation and integration are inverses: $\frac{d}{dx}\int_a^xf(t)\,dt = f(x)$. For continuous random variables (Ch. 3), probabilities are integrals of densities and expectations are $\E[g(X)] = \int g(x)p(x)\,dx$.

!!! theorem "Theorem 2.11 (Gaussian integral)"
    $\displaystyle\int_{-\infty}^{\infty}e^{-x^2/2}\,dx = \sqrt{2\pi}$. Hence $p(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ is a valid probability density.

??? proof "Proof (Poisson's polar-coordinates trick)"
    Let $I = \int e^{-x^2/2}dx$. Then $I^2 = \int\!\!\int e^{-(x^2+y^2)/2}\,dx\,dy$. Switch to polar coordinates ($x^2+y^2=r^2$, $dx\,dy = r\,dr\,d\phi$):

    $$
    I^2 = \int_0^{2\pi}\!\!\int_0^\infty e^{-r^2/2}\,r\,dr\,d\phi = 2\pi\Big[-e^{-r^2/2}\Big]_0^\infty = 2\pi .
    $$

    Since $I > 0$, $I = \sqrt{2\pi}$. $\blacksquare$

!!! watch "Watch"
    3Blue1Brown: [Why π is in the normal distribution](https://www.youtube.com/watch?v=cy8r7WSuT1I).

---

## Key takeaways

- A derivative is the slope of the best linear approximation. The chain rule multiplies slopes along a path and sums over paths.
- The gradient points uphill and is orthogonal to level sets. Gradient descent steps along $-\nabla f$.
- Backprop is repeated **vector–Jacobian products**: $\nabla_\mathbf{x}\mathcal{L} = \mathbf{J}^\top\nabla_\mathbf{y}\mathcal{L}$. Jacobians are never formed explicitly.
- Linear layer: $\nabla_\mathbf{X} = \mathbf{G}\mathbf{W}$, $\nabla_\mathbf{W} = \mathbf{G}^\top\mathbf{X}$, $\nabla_\mathbf{b} = $ row-sum of $\mathbf{G}$. Backward costs ≈ 2× forward.
- Softmax + cross-entropy: $\nabla_\mathbf{z}\mathcal{L} = \mathbf{p}-\mathbf{y}$. Always compute it from logits with the max-subtraction trick.
- The Hessian measures curvature. Its eigenvalues govern optimization stability.

## Exercises

1. Derive $\frac{d}{dx}\tanh x = 1-\tanh^2 x$ from $\tanh x = \frac{e^x-e^{-x}}{e^x+e^{-x}}$.
2. Show that $\tanh(x) = 2\sigma(2x)-1$.
3. Let $\mathcal{L} = \lVert\mathbf{W}\mathbf{x}-\mathbf{y}\rVert^2$. Find $\nabla_\mathbf{W}\mathcal{L}$ using Theorem 2.6.
4. Show that the softmax Jacobian is singular and find its null space. (Hint: $\mathbf{J}\mathbf{1}$.) Interpret.
5. For the **binary** cross-entropy $\mathcal{L} = -[y\log\sigma(z) + (1-y)\log(1-\sigma(z))]$, show $\partial\mathcal{L}/\partial z = \sigma(z)-y$.
6. **Code:** using only `torch` tensor ops (no autograd), implement forward and backward of `y = softmax(x @ W.T)` followed by cross-entropy, and verify against autograd.

??? success "Solutions"
    1. Quotient rule: $\frac{(e^x+e^{-x})^2-(e^x-e^{-x})^2}{(e^x+e^{-x})^2} = 1-\tanh^2x$.
    2. $2\sigma(2x)-1 = \frac{2}{1+e^{-2x}}-1 = \frac{1-e^{-2x}}{1+e^{-2x}}$. Multiply numerator and denominator by $e^x$.
    3. $\mathbf{g} = \nabla_{\hat{\mathbf{y}}}\mathcal{L} = 2(\mathbf{W}\mathbf{x}-\mathbf{y})$, so $\nabla_\mathbf{W}\mathcal{L} = 2(\mathbf{W}\mathbf{x}-\mathbf{y})\mathbf{x}^\top$.
    4. $(\diag(\mathbf{p})-\mathbf{p}\mathbf{p}^\top)\mathbf{1} = \mathbf{p}-\mathbf{p}(\mathbf{p}^\top\mathbf{1}) = \mathbf{p}-\mathbf{p} = \mathbf{0}$. Adding a constant to all logits doesn't change the softmax, so that direction has zero derivative.
    5. $\partial\log\sigma(z)/\partial z = 1-\sigma(z)$ and $\partial\log(1-\sigma(z))/\partial z = -\sigma(z)$. Combine: $-[y(1-\sigma) - (1-y)\sigma] = \sigma - y$.
    6. Forward: `z = x @ W.T; p = softmax(z)`. Backward: `G = (p - onehot) / N; dW = G.T @ x; dx = G @ W`.

## References

- T. Parr, J. Howard, [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) (2018).
- M. P. Deisenroth et al., [*Mathematics for Machine Learning*](https://mml-book.github.io/), Ch. 5.
- A. G. Baydin et al., [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767), *JMLR* (2018).
- I. Goodfellow, Y. Bengio, A. Courville, [*Deep Learning*](https://www.deeplearningbook.org/), Ch. 4 & 6.5 (2016).
