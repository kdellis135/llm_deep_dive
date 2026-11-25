---
seo_title: "Neurons & Multilayer Perceptrons (MLPs)"
description: "The artificial neuron, every activation function used in LLMs (ReLU, GELU, SiLU) and its derivative, MLP shapes and parameter counts, and a proof of universal approximation."
---

# 6. Neurons & MLPs

!!! abstract "In this chapter"
    **Prerequisites:** Part I, especially [Ch. 1](../math/linear-algebra.md) (matrices) and [§2.5](../math/calculus.md#25-matrix-calculus-you-will-actually-use) (softmax and cross-entropy gradients). **You will learn:** the artificial neuron, from linear and logistic regression, why stacking linear layers is pointless without a nonlinearity, every activation function used in LLMs (sigmoid, tanh, ReLU, GELU, SiLU) and its derivative, the multilayer perceptron (MLP) with exact shapes and parameter counts, a full proof of universal approximation in one dimension, how a network "folds space" to solve XOR, and where MLPs live inside a Transformer.
    **Why it matters:** about two-thirds of an LLM's parameters sit in MLP (feed-forward) layers. Everything in this chapter appears unchanged inside GPT-2 and Llama.

!!! watch "Watch"
    3Blue1Brown, [But what is a neural network?](https://www.youtube.com/watch?v=aircAruvnKk) (Ch. 1 of the *Neural Networks* series).

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/aircAruvnKk" allowfullscreen loading="lazy"></iframe></div>

    Then play with the [TensorFlow Playground](https://playground.tensorflow.org/): train tiny MLPs in your browser and watch each neuron's decision boundary form. M. Nielsen, [*Neural Networks and Deep Learning*, Ch. 4](http://neuralnetworksanddeeplearning.com/chap4.html), has an interactive visual proof of universality.

---

## 6.1 One neuron: linear and logistic regression

A **neuron** computes a weighted sum of its inputs plus a bias, then applies a fixed nonlinear function $\phi$:

$$
a = \phi(z),\qquad z = \mathbf{w}^\top\mathbf{x} + b = \sum_{j=1}^{n}w_jx_j + b.
$$

$z$ is the **pre-activation**, $a$ the **activation**. Geometrically, $\mathbf{w}^\top\mathbf{x}+b = 0$ is a hyperplane, and $z$ is proportional to the signed distance of $\mathbf{x}$ from it (the distance is $z/\lVert\mathbf{w}\rVert$). A neuron asks: *how far, and on which side of my hyperplane, is this input?*

Two special cases are classical statistics:

- **Linear regression**: $\phi(z) = z$, trained with mean-squared error. By Exercise 3.3 this is MLE under Gaussian noise, and the optimum solves the normal equations (Thm 1.13).
- **Logistic regression**: $\phi = \sigma$, the sigmoid, output read as $p(y=1\mid\mathbf{x})$ and trained with the binary cross-entropy $-y\log a - (1-y)\log(1-a)$, which is MLE under a Bernoulli model (§3.7).

!!! theorem "Proposition 6.1 (Logistic regression gradient)"
    For one example, $\dfrac{\partial\mathcal{L}}{\partial z} = a - y$, so $\nabla_\mathbf{w}\mathcal{L} = (a-y)\,\mathbf{x}$ and $\dfrac{\partial\mathcal{L}}{\partial b} = a-y$.

??? proof "Proof"
    $\frac{\partial\mathcal{L}}{\partial a} = -\frac ya + \frac{1-y}{1-a} = \frac{a-y}{a(1-a)}$, and $\frac{da}{dz} = a(1-a)$ (Example 2.2). Multiply them (chain rule): $\frac{\partial\mathcal{L}}{\partial z} = a-y$. Then $\frac{\partial z}{\partial w_j} = x_j$ and $\frac{\partial z}{\partial b} = 1$. $\blacksquare$

"Prediction minus target, times input." The same pattern, $\mathbf{p}-\mathbf{y}$, is the gradient of softmax cross-entropy (Thm 2.8), the loss that trains every LLM. Sigmoid + binary cross-entropy is the two-class special case of softmax + cross-entropy: $\sigma(z) = \softmax(0, z)_2$.

## 6.2 Layers, and why we need nonlinearity

Stack $m$ neurons that share the same input and you get a **layer**: $\mathbf{a} = \phi(\mathbf{W}\mathbf{x}+\mathbf{b})$ with $\mathbf{W}\in\R^{m\times n}$ (row $i$ is neuron $i$'s weights), $\mathbf{b}\in\R^m$, and $\phi$ applied elementwise.

!!! theorem "Proposition 6.2 (Linear layers collapse)"
    Without a nonlinearity, any stack of layers is a single affine map: $\mathbf{W}_2(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2 = (\mathbf{W}_2\mathbf{W}_1)\mathbf{x} + (\mathbf{W}_2\mathbf{b}_1+\mathbf{b}_2)$.

So a 100-layer linear network can only represent what one linear layer can: hyperplane decision boundaries. Depth only helps if something bends space between the layers. That bending is the job of the **activation function**.

## 6.3 Activation functions

<div class="widget" data-widget="activations"></div>

| Name | $\phi(z)$ | $\phi'(z)$ | Used in |
|---|---|---|---|
| Sigmoid | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))\le\frac14$ | gates (LSTM), binary outputs |
| Tanh | $\tanh z = 2\sigma(2z)-1$ | $1-\tanh^2z$ | RNNs, early MLPs, micrograd |
| ReLU | $\max(0,z)$ | $\mathbb{1}[z>0]$ | CNNs, original Transformer |
| GELU | $z\,\Phi(z)$ | $\Phi(z)+z\,\varphi(z)$ | BERT, GPT-2, GPT-3 |
| SiLU / Swish | $z\,\sigma(z)$ | $\sigma(z)\big(1+z(1-\sigma(z))\big)$ | inside SwiGLU: Llama, Mistral, Qwen, DeepSeek |

Here $\Phi$ and $\varphi$ are the standard normal CDF and density.

- **Sigmoid and tanh saturate.** For $|z|\gtrsim4$ the derivative is nearly zero, so gradients passing through vanish (Ch. 8). The sigmoid's derivative is at most $1/4$, so even at best a chain of $L$ sigmoids shrinks gradients by $4^{-L}$.
- **ReLU** doesn't saturate for $z>0$ and is very cheap. Its drawback is that a neuron whose pre-activation is negative for every input gets zero gradient forever (a "dead ReLU").
- **GELU** ([Hendrycks & Gimpel, 2016](https://arxiv.org/abs/1606.08415)) weights the input by the probability that a standard Gaussian is below it, $\Phi(z)$: a smooth, probabilistic ReLU. It's slightly negative for $z<0$ (minimum $\approx-0.17$ at $z\approx-0.75$) and smooth everywhere. GPT-2 uses the tanh approximation $\tfrac12z\big(1+\tanh\big[\sqrt{2/\pi}(z+0.044715z^3)\big]\big)$, which is why our code uses `nn.GELU(approximate="tanh")`: to match the pretrained weights exactly.
- **SiLU** ([Ramachandran et al., 2017](https://arxiv.org/abs/1710.05941); also called Swish) has almost the same shape as GELU. Its main use today is as the gate in **SwiGLU** (Ch. 13, 20).

??? proof "Derivation of the GELU and SiLU derivatives"
    Product rule: $\frac{d}{dz}[z\Phi(z)] = \Phi(z) + z\Phi'(z) = \Phi(z)+z\varphi(z)$. For SiLU, $\frac{d}{dz}[z\sigma(z)] = \sigma(z) + z\sigma(z)(1-\sigma(z))$, using Example 2.2. $\blacksquare$

## 6.4 The multilayer perceptron

!!! definition "Definition 6.1 (MLP)"
    An $L$-layer MLP with widths $n_0\to n_1\to\cdots\to n_L$ computes

    $$
    \mathbf{h}^{(0)} = \mathbf{x},\qquad \mathbf{h}^{(\ell)} = \phi\big(\mathbf{W}^{(\ell)}\mathbf{h}^{(\ell-1)}+\mathbf{b}^{(\ell)}\big)\ \ (\ell = 1,\dots,L-1),\qquad \mathbf{z} = \mathbf{W}^{(L)}\mathbf{h}^{(L-1)}+\mathbf{b}^{(L)},
    $$

    with $\mathbf{W}^{(\ell)}\in\R^{n_\ell\times n_{\ell-1}}$. The last layer has **no** activation; its outputs $\mathbf{z}$ are logits (classification) or predictions (regression). The intermediate $\mathbf{h}^{(\ell)}$ are **hidden** representations.

**Parameter count:** $\sum_{\ell=1}^L n_\ell(n_{\ell-1}+1)$. **Forward cost:** about $2\sum_\ell n_\ell n_{\ell-1}$ FLOPs per example (Fact 1.6: one multiply and one add per weight).

**Batched form.** With a batch stacked as rows, $\mathbf{X}\in\R^{B\times n_0}$, each layer is $\mathbf{H}^{(\ell)} = \phi(\mathbf{H}^{(\ell-1)}\mathbf{W}^{(\ell)\top}+\mathbf{1}\mathbf{b}^{(\ell)\top})$: one matrix multiplication per layer for the whole batch. This is exactly what `nn.Linear` computes (see the row/column convention in [Notation](../start/notation.md)).

```python
import torch, torch.nn as nn

mlp = nn.Sequential(          # 784 -> 256 -> 256 -> 10, e.g. MNIST digits
    nn.Linear(784, 256), nn.GELU(),
    nn.Linear(256, 256), nn.GELU(),
    nn.Linear(256, 10),       # logits, no activation
)
x = torch.randn(32, 784)      # (B, n_0)
logits = mlp(x)               # (B, 10)
loss = nn.functional.cross_entropy(logits, torch.randint(0, 10, (32,)))
print(sum(p.numel() for p in mlp.parameters()))   # 784*256+256 + 256*256+256 + 256*10+10 = 269,322
```

In Ch. 7 you will build the same thing from scalars with your own autograd engine (`code/micrograd.py`), and check that it agrees with PyTorch to 9 decimal places.

## 6.5 How depth bends space: XOR

XOR ($y = 1$ iff exactly one of $x_1,x_2\in\{0,1\}$ is 1) is the classic function a single neuron can't compute: no line separates $\{(0,1),(1,0)\}$ from $\{(0,0),(1,1)\}$. One hidden layer of two ReLUs solves it exactly:

$$
h_1 = \operatorname{ReLU}(x_1+x_2),\qquad h_2 = \operatorname{ReLU}(x_1+x_2-1),\qquad y = h_1-2h_2.
$$

Check: $(0,0)\to0$, $(1,0)\to1$, $(0,1)\to1$, $(1,1)\to2-2 = 0$. The hidden layer maps the four inputs to $(0,0), (1,0), (1,0), (2,1)$. The two "1" inputs land on the *same* point, and the new points are linearly separable. **Each layer re-represents the data so that the next layer's job is easier.** That's the whole idea of representation learning, and an LLM's residual stream (Ch. 13) is the same idea at a scale of 100 layers.

## 6.6 Universal approximation

How expressive is one hidden layer?

!!! theorem "Theorem 6.3 (Universal approximation)"
    Let $\phi$ be continuous and not a polynomial. For any continuous $f:[0,1]^n\to\R$ and any $\epsilon>0$ there is a one-hidden-layer network $g(\mathbf{x}) = \sum_{i=1}^m c_i\,\phi(\mathbf{w}_i^\top\mathbf{x}+b_i)$ with $\sup_\mathbf{x}|f(\mathbf{x})-g(\mathbf{x})|<\epsilon$.

    (Cybenko, 1989, for sigmoids; Hornik, 1991; Leshno et al., 1993, for this general form.)

We prove the one-dimensional ReLU case completely, because the proof shows *how* networks approximate.

??? proof "Proof for $n = 1$ and $\phi = \operatorname{ReLU}$"
    *Step 1: continuous functions on $[0,1]$ are uniformly close to piecewise-linear ones.* $f$ is uniformly continuous on the compact interval, so there's a $\delta$ with $|f(x)-f(y)|<\epsilon$ whenever $|x-y|<\delta$. Choose knots $0 = t_0<t_1<\dots<t_m = 1$ with spacing $<\delta$ and let $p$ be the linear interpolant of $f$ through the points $(t_k, f(t_k))$. On each piece $[t_{k-1},t_k]$, $p(x)$ lies between $f(t_{k-1})$ and $f(t_k)$, both within $\epsilon$ of $f(x)$, so $|f-p|<\epsilon$ everywhere.

    *Step 2: every piecewise-linear function is exactly a one-layer ReLU network.* Let $s_k$ be the slope of $p$ on $[t_{k-1},t_k]$. Then

    $$
    p(x) = f(0) + s_1\operatorname{ReLU}(x-t_0) + \sum_{k=2}^{m}(s_k-s_{k-1})\operatorname{ReLU}(x-t_{k-1}).
    $$

    Check: on $[t_{j-1},t_j]$ exactly the first $j$ ReLUs are active, so the slope there is $s_1+\sum_{k=2}^{j}(s_k-s_{k-1}) = s_j$, and at $x = 0$ the value is $f(0)$. Two piecewise-linear functions with the same starting value and the same slopes on every piece are equal. (The constant $f(0)$ is the output bias.) So $m$ hidden ReLUs approximate $f$ to within $\epsilon$. $\blacksquare$

    The $n$-dimensional case can be reduced to this one by approximating $f$ with sums of "ridge functions" $\psi(\mathbf{w}^\top\mathbf{x})$ (a consequence of the Stone–Weierstrass theorem), each of which is a 1-D problem.

!!! warning "What the theorem does *not* say"
    It says nothing about **how many** neurons you need (in general, exponentially many in $n$), whether gradient descent will **find** the weights, or whether the network will **generalize** beyond its training data. Depth matters for efficiency: Telgarsky ([2016](https://arxiv.org/abs/1602.04485)) constructed functions that a deep ReLU network computes with $O(k^3)$ neurons, but a network with only $O(k)$ layers needs $\Omega(2^k)$. Composing $k$ "tent" maps produces $2^k$ linear pieces, while a shallow network gets only about one piece per neuron.

## 6.7 Multi-class outputs: softmax regression

For $K$ classes, the last layer outputs logits $\mathbf{z}\in\R^K$, and the model's distribution is $\mathbf{p} = \softmax(\mathbf{z})$. Training minimizes the cross-entropy $-\log p_y$, which is MLE for a categorical model (§3.7) and equals forward KL from the data plus a constant (Thm 4.9). Its gradient with respect to the logits is $\mathbf{p}-\mathbf{y}$ (Thm 2.8).

**An LLM is exactly this**, with $K = V\approx 10^5$ classes (the vocabulary) and an extremely elaborate network computing the logits from the context. The rest of this site is about that network.

!!! llm "Where MLPs live in a Transformer"
    Every Transformer block contains a position-wise **feed-forward network** (FFN): a 2-layer MLP applied to each token's vector independently,

    $$
    \operatorname{FFN}(\mathbf{x}) = \mathbf{W}_2\,\phi(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2,\qquad \mathbf{W}_1\in\R^{4d\times d},\ \mathbf{W}_2\in\R^{d\times4d}.
    $$

    With hidden width $4d$, the FFN has $8d^2$ weights against attention's $4d^2$, so it holds about **2/3 of the block's parameters** (Ch. 13). Geva et al., [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) (2021), interpret it this way: each row of $\mathbf{W}_1$ is a "key" that detects a pattern in the residual stream, and the matching column of $\mathbf{W}_2$ is a "value" written back when the key fires. Factual associations ("Paris" → "France") appear to be stored largely in these layers ([Meng et al., ROME, 2022](https://arxiv.org/abs/2202.05262); see Ch. 24). 3Blue1Brown's [How might LLMs store facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8) animates this.

## Key takeaways

- A neuron is a hyperplane test followed by a nonlinearity. Logistic regression is one neuron with a sigmoid, and its gradient is (prediction − target) × input.
- Without nonlinearities, depth collapses to one linear map.
- Know the activations and their derivatives: sigmoid/tanh saturate, ReLU is piecewise linear, GELU (GPT-2) and SiLU (SwiGLU in Llama) are smooth ReLU-like functions.
- An MLP is alternating affine maps and elementwise nonlinearities. Each layer re-represents the data.
- One hidden layer is a universal approximator (proved in 1-D by piecewise-linear interpolation), but depth can be exponentially more efficient.
- The output layer of an LLM is softmax regression over the vocabulary. The FFN inside each block is a 2-layer MLP holding about 2/3 of the parameters.

## Exercises

1. Find a single neuron (weights, bias, and a threshold activation) that computes AND, and one that computes OR, on $\{0,1\}^2$.
2. Show that $\tanh(z) = 2\sigma(2z)-1$, and hence that a tanh network can be rewritten exactly as a sigmoid network (with rescaled weights and biases).
3. Count the parameters of an MLP with widths $768\to3072\to768$ (with biases). Compare it with GPT-2's per-block FFN.
4. Write $f(x) = |x|$ on $[-1,1]$ as an exact ReLU network. How many hidden units does it need?
5. Show that GELU's minimum is the solution of $\Phi(z)+z\varphi(z) = 0$, and find it numerically.
6. **Code:** train the PyTorch MLP above to fit $f(x) = \sin(3x)$ on $[-2,2]$ with 4, 16 and 64 ReLU hidden units. Plot the fits and point out the "kinks".

??? success "Solutions"
    1. AND: $\mathbb{1}[x_1+x_2-1.5>0]$. OR: $\mathbb{1}[x_1+x_2-0.5>0]$.
    2. $2\sigma(2z)-1 = \frac{2}{1+e^{-2z}}-1 = \frac{1-e^{-2z}}{1+e^{-2z}} = \tanh z$. A tanh unit $\tanh(\mathbf{w}^\top\mathbf{x}+b)$ equals $2\sigma(2\mathbf{w}^\top\mathbf{x}+2b)-1$; absorb the factor 2 and the offset into the next layer's weights and bias.
    3. $768\cdot3072+3072+3072\cdot768+768 = 4{,}722{,}432$. That's exactly GPT-2's per-block MLP (`c_fc` + `c_proj`).
    4. $|x| = \operatorname{ReLU}(x)+\operatorname{ReLU}(-x)$: two hidden units.
    5. Setting the derivative to zero gives $z\approx-0.7518$, where $\operatorname{GELU}(z)\approx-0.170$.
    6. Each ReLU unit adds one kink. With 4 units the fit is visibly polygonal; with 64 it's smooth to the eye.

## References

- G. Cybenko, Approximation by Superpositions of a Sigmoidal Function, *Math. Control Signals Systems* 2 (1989).
- M. Leshno, V. Lin, A. Pinkus, S. Schocken, Multilayer Feedforward Networks With a Nonpolynomial Activation Function Can Approximate Any Function, *Neural Networks* 6 (1993).
- M. Telgarsky, [Benefits of Depth in Neural Networks](https://arxiv.org/abs/1602.04485) (COLT 2016).
- D. Hendrycks & K. Gimpel, [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) (2016).
- M. Geva et al., [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) (EMNLP 2021).
- I. Goodfellow, Y. Bengio, A. Courville, [*Deep Learning*](https://www.deeplearningbook.org/), Ch. 6 (2016).
- S. Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/), Ch. 3–4 (2023). Free, with excellent figures.
