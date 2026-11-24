# 7. Backpropagation & Autograd

!!! abstract "In this chapter"
    **Prerequisites:** [§2.4](../math/calculus.md#24-jacobians-and-the-multivariate-chain-rule) (VJPs), [Ch. 6](mlp.md). **You will learn:** computation graphs, forward vs reverse mode and why reverse mode wins, backprop on a worked example by hand, a complete scalar autograd engine (micrograd) line by line, the matrix form of backprop for an MLP with proof, the backward rule of every common op (matmul, broadcast, softmax, LayerNorm, embedding lookup), how PyTorch autograd works, why training costs about 3× a forward pass ($6N$ FLOPs per token), and activation memory and gradient checkpointing.
    **Why it matters:** backprop is the algorithm that makes it possible to train a trillion-parameter model. If you can derive the backward pass of a layer yourself, you'll be able to debug training, read kernels, and understand every memory and compute estimate in Part IV.

!!! watch "Watch (the core video of this site)"
    Andrej Karpathy, [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) (2.5 h). Build it along with him.

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/VMj-3S1tku0" allowfullscreen loading="lazy"></iframe></div>

    Then: Karpathy, [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI) (manual backprop through a whole MLP with BatchNorm). 3Blue1Brown, [Backpropagation, intuitively](https://www.youtube.com/watch?v=Ilg3gGewQ5U) and [Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8). Reading: Karpathy, [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b). Stanford [CS231n notes on backprop](https://cs231n.github.io/optimization-2/).

---

## 7.1 Computation graphs

Any computation built from primitive operations is a **directed acyclic graph (DAG)**: nodes are values, edges say "this value was used to compute that one". The loss is the final node. For example,

$$
e = a\cdot b,\qquad d = e + c,\qquad \mathcal{L} = \tanh(d).
$$

Each primitive knows its **local derivative**: $\partial e/\partial a = b$, $\partial d/\partial e = 1$, $\partial\mathcal{L}/\partial d = 1-\tanh^2d$. The chain rule (Thm 2.4) says the derivative of the output with respect to any node is the **sum over all paths** from that node to the output of the **product of local derivatives** along each path.

There are two ways to organize that sum.

- **Forward mode** fixes one *input* and pushes $\partial(\cdot)/\partial x$ forward through the graph. One pass gives the derivatives of every output with respect to *one* input.
- **Reverse mode** fixes the *output* and pulls $\partial\mathcal{L}/\partial(\cdot)$ backward. One pass gives the derivative of *one* output with respect to *every* input.

A neural network has one scalar output (the loss) and $10^9$+ inputs (the parameters), so reverse mode is the right choice by a factor of $10^9$. **Backpropagation is reverse-mode automatic differentiation** applied to a neural network ([Rumelhart, Hinton & Williams, 1986](https://www.nature.com/articles/323533a0); for the general theory see [Baydin et al., 2018](https://arxiv.org/abs/1502.05767)).

!!! theorem "Theorem 7.1 (Cheap gradient principle)"
    For any program computing a scalar $f(\mathbf{x})$ from primitives whose local derivatives cost $O(1)$ times the primitive itself, reverse mode computes the full gradient $\nabla f$ at cost at most a small constant (≈ 2–4) times the cost of evaluating $f$, **independent of the number of inputs**.

??? proof "Proof sketch"
    Reverse mode visits each node once in reverse topological order. At node $v$ with incoming gradient $\bar v = \partial\mathcal{L}/\partial v$, it adds $\bar v\cdot\partial v/\partial u$ into $\bar u$ for each parent $u$. That's one multiply-add per edge of the graph, and the forward pass already spent at least one operation per edge. So the backward pass is a constant multiple of the forward pass. The constant depends on the primitives: for a matmul it is exactly 2 (§7.6). $\blacksquare$

## 7.2 Backprop by hand

Take $a = 2$, $b = -3$, $c = 10$ in the graph above.

**Forward:** $e = -6$, $d = 4$, $\mathcal{L} = \tanh4\approx0.99933$.

**Backward**, from the output, writing $\bar v$ for $\partial\mathcal{L}/\partial v$:

| Node | Rule | Value |
|---|---|---|
| $\bar{\mathcal{L}}$ | seed | $1$ |
| $\bar d$ | $\bar{\mathcal{L}}\cdot(1-\tanh^2d)$ | $1-0.99933^2\approx0.001341$ |
| $\bar e,\ \bar c$ | $+$ passes the gradient through unchanged | $0.001341$ each |
| $\bar a$ | $\bar e\cdot b$ | $-0.004023$ |
| $\bar b$ | $\bar e\cdot a$ | $0.002682$ |

Check $\bar a$ numerically: $\tanh(2.0001\cdot(-3)+10) - \tanh(4)\approx -4.0\times10^{-7}$, divided by $10^{-4}$ that's $\approx-0.00402$. ✓.

Three patterns cover nearly everything:

- **Add distributes**: $+$ copies the incoming gradient to every input.
- **Multiply swaps**: $\overline{a}$ gets $\bar e$ times the *other* input. That's why the forward pass must **save its inputs** for the backward pass, and why activations dominate training memory (§7.7).
- **Branches accumulate**: if a value is used twice, its gradient is the **sum** of the gradients from both uses (the sum over paths). In code that's `+=`, never `=`.

## 7.3 A complete autograd engine in about 100 lines

Here is the whole engine. Each `Value` stores its data, its gradient, its parents, and a closure `_backward` that applies its **local** chain rule. `backward()` sorts the graph topologically and calls the closures in reverse.

```python
--8<-- "code/micrograd.py:value"
```

Things to notice:

1. **Every `_backward` uses `+=`.** That's the "branches accumulate" rule. It's also why PyTorch accumulates into `.grad` and you must call `zero_grad()` each step.
2. **Topological order** guarantees that when `v._backward()` runs, `v.grad` is already complete: every node that consumes `v` comes later in `topo`, so it was processed earlier in the reversed loop.
3. **Derived ops** (`-`, `/`) reuse primitives, so their gradients come for free.

A neuron, a layer and an MLP are a few lines on top:

```python
--8<-- "code/micrograd.py:nn"
```

And training is the three-step loop you'll see in every training script on this site: **zero grads, backward, step.**

```python
--8<-- "code/micrograd.py:train"
```

The test suite builds the same expression in micrograd and in PyTorch and checks that the forward values and all gradients agree to $10^{-9}$, and that this MLP drives its loss below 0.01.

## 7.4 Backprop in matrix form

Scalar graphs are great for understanding but far too slow (one Python object per number). Real networks run backprop on **tensors**, with one vector–Jacobian product (VJP) per layer. For the MLP of Definition 6.1 on one example, define the **error signal** $\boldsymbol\delta^{(\ell)} = \partial\mathcal{L}/\partial\mathbf{z}^{(\ell)}$, the gradient with respect to layer $\ell$'s pre-activations.

!!! theorem "Theorem 7.2 (Backpropagation equations)"
    With $\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)}\mathbf{h}^{(\ell-1)}+\mathbf{b}^{(\ell)}$, $\mathbf{h}^{(\ell)} = \phi(\mathbf{z}^{(\ell)})$, and softmax-cross-entropy on the output logits $\mathbf{z}^{(L)}$:

    $$
    \begin{aligned}
    \boldsymbol\delta^{(L)} &= \mathbf{p}-\mathbf{y}, \\
    \boldsymbol\delta^{(\ell)} &= \big(\mathbf{W}^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}\big)\odot\phi'(\mathbf{z}^{(\ell)}), \\
    \nabla_{\mathbf{W}^{(\ell)}}\mathcal{L} &= \boldsymbol\delta^{(\ell)}\,\mathbf{h}^{(\ell-1)\top},\qquad \nabla_{\mathbf{b}^{(\ell)}}\mathcal{L} = \boldsymbol\delta^{(\ell)}.
    \end{aligned}
    $$

??? proof "Proof"
    The first line is Theorem 2.8. For the recursion, $\mathbf{z}^{(\ell+1)}$ depends on $\mathbf{z}^{(\ell)}$ through $\mathbf{h}^{(\ell)} = \phi(\mathbf{z}^{(\ell)})$. The Jacobian of $\mathbf{z}^{(\ell+1)}$ with respect to $\mathbf{h}^{(\ell)}$ is $\mathbf{W}^{(\ell+1)}$, and that of the elementwise $\phi$ is $\diag(\phi'(\mathbf{z}^{(\ell)}))$. By the VJP rule $\nabla_\mathbf{x}\mathcal{L} = \mathbf{J}^\top\nabla_\mathbf{y}\mathcal{L}$ applied twice:

    $$
    \boldsymbol\delta^{(\ell)} = \diag(\phi'(\mathbf{z}^{(\ell)}))\,\mathbf{W}^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)} = \big(\mathbf{W}^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)}\big)\odot\phi'(\mathbf{z}^{(\ell)}).
    $$

    For the weights, $z_i^{(\ell)} = \sum_jW_{ij}h_j^{(\ell-1)}+b_i$, so $\partial\mathcal{L}/\partial W_{ij} = \delta_i^{(\ell)}h_j^{(\ell-1)}$: the outer product. And $\partial z_i/\partial b_i = 1$. (This is Theorem 2.6, the linear-layer gradients.) $\blacksquare$

**Read it as a reversed network.** The backward pass is another network running from output to input: multiply by $\mathbf{W}^\top$ instead of $\mathbf{W}$, and multiply elementwise by $\phi'$ instead of applying $\phi$. Every weight's gradient is "error at its output × activity at its input", a local, Hebbian-looking rule.

**Batched.** Stack $B$ examples as rows ($\mathbf{H}\in\R^{B\times n}$, $\boldsymbol\Delta\in\R^{B\times m}$). Then $\nabla_\mathbf{W}\mathcal{L} = \boldsymbol\Delta^\top\mathbf{H}$, which *sums* the per-example outer products, and $\nabla_\mathbf{b}\mathcal{L} = \boldsymbol\Delta^\top\mathbf{1}$, the column sums. Here is the whole backward pass written by hand, checked against autograd in the test suite:

```python
--8<-- "code/backprop.py:mlp"
```

!!! warning "Broadcast in forward ⇒ sum in backward"
    The bias $\mathbf{b}$ is *broadcast* (copied) across the batch in the forward pass, so in the backward pass its gradient is the *sum* over the batch. In general, **if a tensor was broadcast along an axis in the forward pass, sum the gradient along that axis in the backward pass.** "Copy" and "sum" are adjoint (transpose) operations. Forgetting this is the most common bug in hand-written backprop.

## 7.5 Backward rules for the ops in a Transformer

With upstream gradient $\bar{\mathbf{Y}} = \partial\mathcal{L}/\partial\mathbf{Y}$, always the same shape as $\mathbf{Y}$ (the gradient convention):

| Forward | Backward | Notes |
|---|---|---|
| $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$ | $\bar{\mathbf{X}} = \bar{\mathbf{Y}}\mathbf{W}$, $\ \bar{\mathbf{W}} = \bar{\mathbf{Y}}^\top\mathbf{X}$ | two matmuls, each as costly as the forward |
| $\mathbf{Y} = \mathbf{X} + \mathbf{1}\mathbf{b}^\top$ (broadcast) | $\bar{\mathbf{X}} = \bar{\mathbf{Y}}$, $\ \bar{\mathbf{b}} = \bar{\mathbf{Y}}^\top\mathbf{1}$ | sum over broadcast axis |
| $\mathbf{Y} = \phi(\mathbf{X})$ elementwise | $\bar{\mathbf{X}} = \bar{\mathbf{Y}}\odot\phi'(\mathbf{X})$ | needs $\mathbf{X}$ (or $\mathbf{Y}$) saved |
| $\mathbf{Y} = \mathbf{A}\odot\mathbf{B}$ | $\bar{\mathbf{A}} = \bar{\mathbf{Y}}\odot\mathbf{B}$, $\ \bar{\mathbf{B}} = \bar{\mathbf{Y}}\odot\mathbf{A}$ | "multiply swaps" |
| $\mathbf{y} = \softmax(\mathbf{x})$ | $\bar{\mathbf{x}} = \mathbf{y}\odot(\bar{\mathbf{y}} - (\bar{\mathbf{y}}^\top\mathbf{y})\mathbf{1})$ | from the softmax Jacobian $\diag(\mathbf{y})-\mathbf{y}\mathbf{y}^\top$ (Thm 2.7) |
| $\ell = -\log\softmax(\mathbf{z})_y$ | $\bar{\mathbf{z}} = \mathbf{p}-\mathbf{y}$ | fused: never backprop through softmax and log separately |
| $\mathbf{Y} = \mathbf{E}[\mathbf{x}]$ (embedding lookup) | $\bar{\mathbf{E}}[x_t] \mathrel{+}= \bar{\mathbf{Y}}_t$ for each $t$ | scatter-add; repeated tokens accumulate |
| $s = \sum_i x_i$, $\ m = \frac1n\sum_i x_i$ | $\bar{\mathbf{x}} = \bar s\,\mathbf{1}$, $\ \bar{\mathbf{x}} = \frac{\bar m}{n}\mathbf{1}$ | reductions broadcast back |
| $\mathbf{Y} = \text{reshape/transpose}(\mathbf{X})$ | inverse reshape/transpose of $\bar{\mathbf{Y}}$ | data movement only |

??? proof "Derivation of the softmax backward rule"
    $\bar{\mathbf{x}} = \mathbf{J}^\top\bar{\mathbf{y}}$ with $\mathbf{J} = \diag(\mathbf{y})-\mathbf{y}\mathbf{y}^\top$ (symmetric). So $\bar{\mathbf{x}} = \mathbf{y}\odot\bar{\mathbf{y}} - \mathbf{y}(\mathbf{y}^\top\bar{\mathbf{y}}) = \mathbf{y}\odot(\bar{\mathbf{y}}-(\mathbf{y}^\top\bar{\mathbf{y}})\mathbf{1})$. It costs $O(n)$, not $O(n^2)$, so the Jacobian is never formed. This is the rule FlashAttention's backward pass uses (Ch. 19). $\blacksquare$

### LayerNorm backward

LayerNorm (Ch. 8) computes, over the $d$ features of one vector, $\mu = \frac1d\sum x_i$, $\sigma^2 = \frac1d\sum(x_i-\mu)^2$, $\hat x_i = (x_i-\mu)/\sqrt{\sigma^2+\epsilon}$, $y_i = \gamma_i\hat x_i+\beta_i$.

!!! theorem "Proposition 7.3 (LayerNorm gradient)"
    With $\mathbf{g} = \bar{\mathbf{y}}\odot\boldsymbol\gamma$ and $r = (\sigma^2+\epsilon)^{-1/2}$,

    $$
    \bar{\mathbf{x}} = r\Big(\mathbf{g} - \overline{g}\,\mathbf{1} - \hat{\mathbf{x}}\,\overline{g\hat x}\Big),\qquad \overline{g} = \tfrac1d\textstyle\sum_ig_i,\quad \overline{g\hat x} = \tfrac1d\sum_ig_i\hat x_i,
    $$

    and $\bar{\boldsymbol\gamma} = \sum_\text{rows}\bar{\mathbf{y}}\odot\hat{\mathbf{x}}$, $\bar{\boldsymbol\beta} = \sum_\text{rows}\bar{\mathbf{y}}$.

??? proof "Proof"
    $\hat x_i = (x_i-\mu)r$. Differentiate: $\frac{\partial\mu}{\partial x_j} = \frac1d$, and $\frac{\partial\sigma^2}{\partial x_j} = \frac2d(x_j-\mu)$ (the terms from $\partial\mu/\partial x_j$ cancel because $\sum_i(x_i-\mu) = 0$). So $\frac{\partial r}{\partial x_j} = -\frac12r^3\cdot\frac2d(x_j-\mu) = -\frac rd\hat x_j\,r$. Then

    $$
    \frac{\partial\hat x_i}{\partial x_j} = r\Big(\delta_{ij}-\frac1d\Big) + (x_i-\mu)\frac{\partial r}{\partial x_j} = r\Big(\delta_{ij}-\frac1d-\frac1d\hat x_i\hat x_j\Big).
    $$

    Contract with $g_i = \partial\mathcal{L}/\partial\hat x_i$: $\bar x_j = r\big(g_j - \frac1d\sum_ig_i - \hat x_j\frac1d\sum_ig_i\hat x_i\big)$. $\blacksquare$

    Interpretation: the gradient is projected to remove its mean and its component along $\hat{\mathbf{x}}$. Those are exactly the two directions LayerNorm's output is invariant to (shifting $\mathbf{x}$, and scaling it).

```python
--8<-- "code/backprop.py:layernorm"
```

## 7.6 The cost of training: why "6N FLOPs per token"

Take one linear layer $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$ with $\mathbf{X}\in\R^{T\times n}$, $\mathbf{W}\in\R^{m\times n}$. The forward matmul costs $2Tmn$ FLOPs (Fact 1.6). The backward needs **two** matmuls of the same size: $\bar{\mathbf{X}} = \bar{\mathbf{Y}}\mathbf{W}$ (to keep backpropagating) and $\bar{\mathbf{W}} = \bar{\mathbf{Y}}^\top\mathbf{X}$ (the parameter gradient), so $4Tmn$ FLOPs.

!!! theorem "Proposition 7.4 (Training FLOPs)"
    For a network dominated by matmuls with $N$ parameters, a forward pass costs about $2N$ FLOPs per token and a forward + backward pass about $6N$ FLOPs per token. Training on $D$ tokens therefore costs

    $$
    C\approx 6ND.
    $$

??? proof "Proof"
    Each weight $W_{ij}$ takes part in one multiply-add per token in the forward pass: $2$ FLOPs per parameter per token, so $2N$ in total. By the argument above, the backward pass costs twice the forward, $4N$. The sum is $6N$. Attention's score computation adds $\approx 2\cdot2\cdot LTd$ forward FLOPs per token, which is small next to $2N$ when $T\ll 6d$ or so (Ch. 15 does this precisely). $\blacksquare$

This is **the** most important cost formula in Part IV. GPT-3 (175B parameters, 300B tokens) took $6\times175\times10^9\times300\times10^9\approx3.15\times10^{23}$ FLOPs.

## 7.7 Memory: activations and gradient checkpointing

The backward pass needs values from the forward pass (inputs to matmuls, pre-activations for $\phi'$, softmax outputs). These **activations** must be stored from the forward pass until the backward pass uses them. For a Transformer, activation memory grows with $B\times T\times d\times L$, and for long sequences it can exceed the memory for the weights themselves.

**Gradient (activation) checkpointing** ([Chen et al., 2016](https://arxiv.org/abs/1604.06174)) stores activations only at block boundaries and **recomputes** the inside of each block during the backward pass. Storing every $\sqrt L$-th layer gives $O(\sqrt L)$ memory for one extra forward pass, about **33% more compute** ($8N$ instead of $6N$ per token). In PyTorch: `torch.utils.checkpoint.checkpoint(block, x)`. Selective variants recompute only the cheap, memory-heavy ops such as attention softmax and GELU ([Korthikanti et al., 2022](https://arxiv.org/abs/2205.05198)).

## 7.8 How PyTorch autograd works

PyTorch is micrograd with tensors, a C++ engine and GPU kernels:

```python
import torch
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-3.0, requires_grad=True)
c = torch.tensor(10.0, requires_grad=True)
L = torch.tanh(a * b + c)
print(L.grad_fn)                     # <TanhBackward0>: the closure, like micrograd's _backward
print(L.grad_fn.next_functions)      # ((AddBackward0, 0),): edges to the parents
L.backward()                         # reverse topological traversal, accumulating .grad
print(a.grad, b.grad, c.grad)        # tensor(-0.0040) tensor(0.0027) tensor(0.0013), as in §7.2
```

| Concept | micrograd | PyTorch |
|---|---|---|
| node | `Value` | `Tensor` with `requires_grad=True` |
| local backward | `_backward` closure | `grad_fn` (a `Function` node) |
| parents | `_prev` | `grad_fn.next_functions` |
| run backprop | `loss.backward()` | `loss.backward()` (graph freed afterwards unless `retain_graph=True`) |
| reset | `p.grad = 0` | `optimizer.zero_grad(set_to_none=True)` |
| stop tracking | n/a | `with torch.no_grad():`, `tensor.detach()` |

**Custom backward rules.** When you know a better gradient formula (more stable, fused, or deliberately "wrong" such as a straight-through estimator), write a `torch.autograd.Function`:

```python
class StraightThroughRound(torch.autograd.Function):
    """Forward: round to integers. Backward: pretend it was the identity.
    The trick behind quantization-aware training (Ch. 19)."""
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out              # d round(x)/dx is 0 almost everywhere; use 1 instead
```

!!! tip "Debugging gradients"
    - Check a new layer with `torch.autograd.gradcheck` (it compares against centered finite differences in float64, §2.7).
    - Log the global gradient norm, and per-layer norms, every step.
    - A `None` gradient means the parameter isn't connected to the loss (often a `.detach()` or a non-differentiable op such as `argmax`). An all-zero gradient often means dead ReLUs or a saturated softmax.

## Key takeaways

- Backprop = reverse-mode autodiff = repeated vector–Jacobian products from the loss backward. The cost is a small constant times the forward pass, regardless of parameter count.
- Local rules: add distributes, multiply swaps, branches accumulate (`+=`), broadcast becomes sum.
- Matrix form: $\boldsymbol\delta^{(\ell)} = (\mathbf{W}^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)})\odot\phi'(\mathbf{z}^{(\ell)})$, $\nabla_\mathbf{W} = \boldsymbol\delta\,\mathbf{h}^\top$.
- Backward ≈ 2× forward for matmuls, so **training ≈ 6N FLOPs per token**, and $C\approx6ND$.
- Training memory = weights + gradients + optimizer state + **activations**. Checkpointing trades ~33% extra compute for much less activation memory.

## Exercises

1. Extend micrograd with `sigmoid()` and `__truediv__` for `Value / Value`, and check against PyTorch.
2. For $f(x) = x\cdot x$ (a node used twice), show that micrograd gives $f'(x) = 2x$ *only* because of `+=`. What would `=` give?
3. Derive the backward pass of $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$ in index notation and confirm the shapes.
4. Derive the backward pass of RMSNorm, $y_i = \gamma_ix_i/\sqrt{\frac1d\sum_jx_j^2+\epsilon}$ (Ch. 20).
5. A 7B-parameter model is trained on 2T tokens. How many FLOPs is that? At 40% utilization of GPUs delivering $10^{15}$ FLOP/s each, how many GPU-hours?
6. **Code:** watch Karpathy's "Backprop Ninja" and do exercise 1 of his notebook: backprop through cross-entropy, BatchNorm and tanh by hand, and match PyTorch.

??? success "Solutions"
    1. `sigmoid`: `s = 1/(1+exp(-x))`, backward `self.grad += s*(1-s)*out.grad`. Division is already derived as `self * other**-1`.
    2. `__mul__` with `self is other`: the closure runs `self.grad += other.data * out.grad` twice in effect (once for each role), giving $2x$. With `=`, the second assignment overwrites the first and gives $x$.
    3. $Y_{ti} = \sum_jX_{tj}W_{ij}$. $\partial\mathcal{L}/\partial X_{tj} = \sum_i\bar Y_{ti}W_{ij}$, so $\bar{\mathbf{X}} = \bar{\mathbf{Y}}\mathbf{W}$, shape $(T, n)$. $\partial\mathcal{L}/\partial W_{ij} = \sum_t\bar Y_{ti}X_{tj}$, so $\bar{\mathbf{W}} = \bar{\mathbf{Y}}^\top\mathbf{X}$, shape $(m, n)$.
    4. With $r = (\overline{x^2}+\epsilon)^{-1/2}$, $\hat{\mathbf{x}} = r\mathbf{x}$, $\mathbf{g} = \bar{\mathbf{y}}\odot\boldsymbol\gamma$: $\bar{\mathbf{x}} = r\big(\mathbf{g} - \hat{\mathbf{x}}\cdot\frac1d\sum_ig_i\hat x_i\big)$. It's LayerNorm's formula without the mean term.
    5. $6\cdot7\times10^9\cdot2\times10^{12} = 8.4\times10^{22}$ FLOPs. At $4\times10^{14}$ effective FLOP/s per GPU, that's $2.1\times10^8$ GPU-seconds ≈ 58,000 GPU-hours.
    6. See Karpathy's [notebook](https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part4_backprop.ipynb).

## References

- D. Rumelhart, G. Hinton, R. Williams, [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0), *Nature* 323 (1986).
- A. G. Baydin, B. Pearlmutter, A. Radul, J. Siskind, [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) (JMLR 2018).
- A. Griewank & A. Walther, *Evaluating Derivatives*, 2nd ed., SIAM (2008). The standard reference on AD and the cheap gradient principle.
- A. Karpathy, [micrograd](https://github.com/karpathy/micrograd) and [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero).
- T. Chen et al., [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) (2016).
- PyTorch docs, [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html).
