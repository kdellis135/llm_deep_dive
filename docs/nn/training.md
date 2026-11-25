---
seo_title: "Training Deep Networks: Initialization, Norms, Residuals"
description: "Why gradients vanish or explode, Xavier and Kaiming initialization derived, BatchNorm vs LayerNorm vs RMSNorm, residual connections, pre-norm vs post-norm, dropout and weight decay."
---

# 8. Training Deep Networks

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 3](../math/probability.md) (variance), [Ch. 5](../math/optimization.md), [Ch. 7](backprop.md). **You will learn:** why gradients vanish or explode in deep networks, how to derive Xavier and Kaiming initialization from a variance argument, GPT-2's residual-scaled initialization, BatchNorm vs LayerNorm vs RMSNorm, why residual connections make depth trainable (with proof), pre-norm vs post-norm, overfitting, dropout and weight decay, and a practical recipe for diagnosing training runs.
    **Why it matters:** a Transformer is a 100-layer-deep network. Every design choice in its block (residual streams, pre-norm, small initial residual weights) exists to keep signals and gradients at a healthy scale through that depth. This chapter derives each one.

!!! watch "Watch"
    Andrej Karpathy, [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc). The best practical treatment of initialization and of reading activation and gradient statistics.

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/P6sfmUTpUmc" allowfullscreen loading="lazy"></iframe></div>

    Reading: Karpathy, [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) (2019). S. Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/), Ch. 7 (initialization) and Ch. 11 (residual networks).

---

## 8.1 Vanishing and exploding gradients

By Theorem 7.2, the error signal at layer $\ell$ is the output error multiplied by a chain of Jacobians:

$$
\boldsymbol\delta^{(\ell)} = \diag(\phi'^{(\ell)})\,\mathbf{W}^{(\ell+1)\top}\diag(\phi'^{(\ell+1)})\,\mathbf{W}^{(\ell+2)\top}\cdots\,\boldsymbol\delta^{(L)}.
$$

A product of $L-\ell$ matrices behaves roughly like the $(L-\ell)$-th power of a typical factor. If each factor scales vectors by $\gamma$ on average, the gradient scales like $\gamma^{L-\ell}$: it **explodes** if $\gamma>1$ and **vanishes** if $\gamma<1$. With $\gamma = 0.9$ and 50 layers the first layers get $0.9^{50}\approx0.5\%$ of the signal. With $\gamma = 1.1$ they get $117\times$. Training only works if every layer keeps $\gamma\approx1$, in both the forward pass (activations) and the backward pass (gradients). The tools are **initialization**, **normalization** and **residual connections**.

## 8.2 Initialization from a variance argument

Consider one linear layer $z_i = \sum_{j=1}^{n_\text{in}}W_{ij}h_j$ with weights i.i.d. with mean 0 and variance $\sigma_w^2$, independent of the inputs.

!!! theorem "Lemma 8.1 (Variance of a linear layer's output)"
    $\Var(z_i) = n_\text{in}\,\sigma_w^2\,\E[h_j^2]$.

??? proof "Proof"
    The terms $W_{ij}h_j$ are independent with mean zero (because $\E W_{ij} = 0$ and $W$ is independent of $h$), so the variances add (§3.4): $\Var(z_i) = \sum_j\Var(W_{ij}h_j) = \sum_j\E[W_{ij}^2]\E[h_j^2] = n_\text{in}\sigma_w^2\E[h^2]$. $\blacksquare$

!!! theorem "Theorem 8.2 (Xavier and Kaiming initialization)"
    - **Linear / tanh (near 0):** if $h = z$ from the previous layer, keeping $\Var(z)$ constant across layers requires $\sigma_w^2 = 1/n_\text{in}$. Keeping the *backward* variance constant requires $\sigma_w^2 = 1/n_\text{out}$. **Xavier/Glorot** init compromises with $\sigma_w^2 = \dfrac{2}{n_\text{in}+n_\text{out}}$ ([Glorot & Bengio, 2010](https://proceedings.mlr.press/v9/glorot10a.html)).
    - **ReLU:** with $h = \operatorname{ReLU}(z)$ and $z$ symmetric about 0, $\E[h^2] = \frac12\Var(z)$, so constant variance requires $\sigma_w^2 = \dfrac{2}{n_\text{in}}$. This is **Kaiming/He** init ([He et al., 2015](https://arxiv.org/abs/1502.01852)).

??? proof "Proof"
    *Forward:* by Lemma 8.1, $\Var(z^{(\ell)}) = n_\text{in}\sigma_w^2\E[(h^{(\ell-1)})^2]$. For the identity activation, $\E[h^2] = \Var(z^{(\ell-1)})$, so the ratio between layers is $n_\text{in}\sigma_w^2$, which equals 1 iff $\sigma_w^2 = 1/n_\text{in}$.

    *Backward:* $\delta_j^{(\ell-1)} = \sum_iW_{ij}\delta^{(\ell)}_i\phi'(\cdot)$ is a sum over the $n_\text{out}$ outputs, so the same argument gives a ratio of $n_\text{out}\sigma_w^2$.

    *ReLU:* if $z$ is symmetric about 0, then $\E[\operatorname{ReLU}(z)^2] = \E[z^2\,\mathbb{1}_{z>0}] = \frac12\E[z^2]$. The half lost to the zeroed side is made up by doubling $\sigma_w^2$. $\blacksquare$

Karpathy's makemore Part 3 shows what goes wrong without this: with $\sigma_w = 1$, tanh pre-activations have standard deviation $\sim\sqrt{n_\text{in}}$, nearly every tanh saturates at ±1, $\phi'\approx0$, and gradients vanish. The fix is literally multiplying the initial weights by $1/\sqrt{n_\text{in}}$ (or $\frac53/\sqrt{n_\text{in}}$, the PyTorch "gain" for tanh).

**At the output layer**, you want the initial logits to be near zero so the initial prediction is near uniform and the initial loss is near $\ln V$ (§4.2). Large random logits make the model *confidently wrong* at step 0. Early training then wastes steps just shrinking the logits, which shows up as a "hockey stick" at the start of the loss curve.

### GPT-2's initialization

GPT-2 (and our `gpt.py`) uses $\mathcal{N}(0, 0.02^2)$ for all weights, which for $d = 768$ is close to $1/\sqrt d\approx0.036$. It adds one important twist: the output projections of each residual branch (`c_proj` in attention and MLP) use standard deviation $0.02/\sqrt{2L}$.

!!! theorem "Proposition 8.3 (Why scale residual branches by $1/\sqrt{2L}$)"
    In a residual network $\mathbf{x}_{k+1} = \mathbf{x}_k + \mathbf{f}_k(\mathbf{x}_k)$ where each branch output is roughly independent of $\mathbf{x}_k$ with variance $s^2$ per coordinate, $\Var(\mathbf{x}_K) = \Var(\mathbf{x}_0) + Ks^2$. A Transformer has $K = 2L$ residual additions (attention and MLP in each of $L$ blocks). Scaling each branch's output projection by $1/\sqrt{2L}$ makes $Ks^2$ independent of depth.

??? proof "Proof"
    Variances of independent sums add: $\Var(\mathbf{x}_{k+1}) = \Var(\mathbf{x}_k)+s^2$. Induct. Scaling the last weight matrix of a branch by $c$ scales $s^2$ by $c^2$, so $c = 1/\sqrt{2L}$ turns $2Ls^2$ into $s^2$. $\blacksquare$

```python
--8<-- "code/gpt.py:gpt"
```

Look at `_init_weights`: the `RESIDUAL_SCALE` flag marks exactly those projections.

## 8.3 Normalization layers

Initialization sets good statistics at step 0, but training changes the weights. **Normalization layers** re-standardize activations at every step.

!!! definition "Definition 8.1 (BatchNorm, LayerNorm, RMSNorm)"
    For activations $\mathbf{X}\in\R^{B\times d}$ (batch × features), with learned gain $\boldsymbol\gamma$ and bias $\boldsymbol\beta$:

    - **BatchNorm** ([Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)): normalize each **feature** over the **batch**, $\hat X_{bi} = (X_{bi}-\mu_i)/\sqrt{\sigma_i^2+\epsilon}$, where $\mu_i,\sigma_i^2$ are computed over $b$.
    - **LayerNorm** ([Ba et al., 2016](https://arxiv.org/abs/1607.06450)): normalize each **example** over its **features**, $\hat X_{bi} = (X_{bi}-\mu_b)/\sqrt{\sigma_b^2+\epsilon}$, where $\mu_b,\sigma_b^2$ are computed over $i$.
    - **RMSNorm** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)): like LayerNorm but without subtracting the mean, $\hat X_{bi} = X_{bi}/\sqrt{\frac1d\sum_jX_{bj}^2+\epsilon}$.

    Then $Y = \boldsymbol\gamma\odot\hat X+\boldsymbol\beta$ (RMSNorm usually has no $\boldsymbol\beta$).

**Why Transformers use LayerNorm/RMSNorm, not BatchNorm.** BatchNorm couples the examples in a batch: an example's output depends on what else is in the batch. That breaks autoregressive generation (batch size 1, one token at a time), behaves differently in training and inference (running averages), and mixes statistics across positions of variable-length sequences. LayerNorm treats each token vector on its own, so it behaves identically in training and at inference.

**RMSNorm** drops the mean subtraction and the bias. It's cheaper (one reduction instead of two) and works just as well in practice, so Llama, Mistral, Qwen, Gemma and DeepSeek all use it (Ch. 20). We derived LayerNorm's backward pass in §7.5.

!!! intuition "What normalization buys"
    After normalization, the scale of a layer's input no longer depends on the scale of the weights before it. So the loss becomes invariant to rescaling those weights, gradients are better conditioned, and higher learning rates become stable. The learned $\boldsymbol\gamma$ gives the network back the freedom to choose a scale.

## 8.4 Residual connections

A **residual connection** ([He et al., 2016](https://arxiv.org/abs/1512.03385)) computes $\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + \mathbf{f}_\ell(\mathbf{x}_\ell)$ instead of $\mathbf{x}_{\ell+1} = \mathbf{f}_\ell(\mathbf{x}_\ell)$.

!!! theorem "Proposition 8.4 (Residual connections are gradient highways)"
    For $\mathbf{x}_{\ell+1} = \mathbf{x}_\ell+\mathbf{f}_\ell(\mathbf{x}_\ell)$,

    $$
    \frac{\partial\mathcal{L}}{\partial\mathbf{x}_\ell} = \Big(\mathbf{I}+\frac{\partial\mathbf{f}_\ell}{\partial\mathbf{x}_\ell}\Big)^{\!\top}\frac{\partial\mathcal{L}}{\partial\mathbf{x}_{\ell+1}},\qquad\text{so}\qquad \frac{\partial\mathbf{x}_L}{\partial\mathbf{x}_\ell} = \prod_{k=\ell}^{L-1}\Big(\mathbf{I}+\frac{\partial\mathbf{f}_k}{\partial\mathbf{x}_k}\Big) = \mathbf{I} + \sum_k\frac{\partial\mathbf{f}_k}{\partial\mathbf{x}_k} + \cdots
    $$

    The expansion always contains the identity term, so the gradient at the output reaches **every** layer directly, however deep the network.

??? proof "Proof"
    The Jacobian of $\mathbf{x}\mapsto\mathbf{x}+\mathbf{f}(\mathbf{x})$ is $\mathbf{I}+\partial\mathbf{f}/\partial\mathbf{x}$. Apply the chain rule over the layers and expand the product. $\blacksquare$

Without residuals the product of Jacobians must stay well-conditioned on its own (§8.1). With residuals, if the branches start small (which is what GPT-2's init does), every block starts close to the identity and the network behaves like a *shallow* network early in training, getting deeper in effect as the branches learn ([Veit et al., 2016](https://arxiv.org/abs/1605.06431) view a ResNet as an ensemble of paths of many lengths).

!!! llm "The residual stream"
    In a Transformer, the running vector $\mathbf{x}_\ell$ is called the **residual stream** ([Elhage et al., 2021](https://transformer-circuits.pub/2021/framework/index.html)). Each attention and MLP block *reads* from it (through a LayerNorm) and *adds* its output back. Nothing ever overwrites it. That makes the stream a shared communication channel between layers, and it's the central object of interpretability research (Ch. 24).

### Pre-norm vs post-norm

The original Transformer was **post-norm**: $\mathbf{x}_{\ell+1} = \operatorname{LN}(\mathbf{x}_\ell+\mathbf{f}(\mathbf{x}_\ell))$. GPT-2 and every modern LLM use **pre-norm**: $\mathbf{x}_{\ell+1} = \mathbf{x}_\ell+\mathbf{f}(\operatorname{LN}(\mathbf{x}_\ell))$, plus a final LayerNorm before the output head.

In post-norm, the normalization sits *on* the residual path, so the clean identity term of Proposition 8.4 is lost: every layer's gradient passes through all the LayerNorm Jacobians above it. Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (2020), show that post-norm has large gradients near the output at initialization, so it needs a careful warmup. Pre-norm keeps the residual path an exact identity, trains stably without warmup tricks, and scales to hundreds of layers.

## 8.5 Generalization and regularization

We minimize the **training** loss, but what we care about is the loss on new data. The gap is **overfitting**. Always hold out a **validation set** and track both curves.

- **Weight decay** (L2, or AdamW's decoupled version, §5.6) pulls weights toward zero. It's MAP estimation under a Gaussian prior (Prop. 3.9).
- **Dropout** ([Srivastava et al., 2014](https://jmlr.org/papers/v15/srivastava14a.html)): during training, zero each activation independently with probability $p$ and scale the survivors by $1/(1-p)$ ("inverted dropout"). Then $\E[\tilde h] = (1-p)\cdot\frac{h}{1-p} = h$, so the expected activation is unchanged and no rescaling is needed at inference. It stops units from co-adapting and acts like an ensemble of thinned networks.
- **Early stopping**: stop when validation loss starts rising.
- **More data**, the best regularizer of all.

!!! llm "LLM pretraining is (mostly) not about overfitting"
    LLMs are usually pretrained for **less than one epoch** over most of their data: every token is seen about once. When each batch is new, the training loss is an unbiased estimate of the test loss, and the problem is *underfitting*, i.e. we'd like a bigger model or more compute. That's why modern pretraining uses **dropout = 0** (Llama, GPT-3 at scale, our `GPTConfig`). Overfitting comes back in **fine-tuning** on small datasets (Ch. 23), where you'll want dropout, low learning rates and few epochs. When high-quality data is limited, repeating it up to ~4 epochs costs almost nothing, but returns diminish quickly after that ([Muennighoff et al., 2023](https://arxiv.org/abs/2305.16264)).

**Double descent.** Classical theory predicts test error rises once a model can fit its training set exactly. In deep learning, past that interpolation threshold, test error often *falls again* as the model keeps growing ([Belkin et al., 2019](https://arxiv.org/abs/1812.11118); [Nakkiran et al., 2019](https://arxiv.org/abs/1912.02292)). Among the many solutions that fit the data, gradient descent from a small initialization tends to find "simple" ones (an *implicit bias*). This is part of why huge overparameterized models generalize.

## 8.6 A practical recipe for diagnosing training

Following Karpathy's [Recipe](http://karpathy.github.io/2019/04/25/recipe/) and the makemore lectures:

1. **Check the loss at initialization.** For a $V$-way classifier it should be about $\ln V$. If not, the output layer's init is wrong.
2. **Overfit one batch.** A healthy model and training loop drive the loss on a single repeated batch to near zero. If they can't, there's a bug.
3. **Watch activation statistics.** Histograms of each layer's activations: are tanh units saturated (most values near ±1)? Are ReLUs dead (always 0)? Is the standard deviation stable across layers?
4. **Watch gradient statistics.** Per-layer gradient norms should be within an order of magnitude of each other. The **update-to-data ratio** $\eta\lVert\nabla\mathbf{W}\rVert/\lVert\mathbf{W}\rVert$ around $10^{-3}$ per step is a healthy rule of thumb (Karpathy, makemore Part 3).
5. **Find the learning rate.** Sweep it on a log scale over a short run and pick the largest one that's stable, then back off a little.
6. **Log the global gradient norm.** Spikes come before loss spikes. Clip at 1.0 (§5.8).
7. **Only then scale up**, changing one thing at a time.

!!! llm "Stability tricks at scale"
    Large runs hit instabilities that small ones never show. Common fixes you'll see in Ch. 20, each studied in Wortsman et al., [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322) (2023):

    - **QK-norm**: normalize queries and keys before the dot product, which prevents attention logits from growing without bound.
    - **z-loss**: add $10^{-4}\cdot(\log\sum_je^{z_j})^2$ to the loss so the output logits stay near zero ([PaLM](https://arxiv.org/abs/2204.02311)).
    - **Logit soft-capping**: $z\leftarrow c\tanh(z/c)$ (Gemma 2).
    - Lower $\beta_2$ (0.95), smaller $\epsilon$, warmup, and no weight decay on gains and biases.

## Key takeaways

- Deep networks multiply Jacobians, so each layer must preserve scale or gradients vanish/explode.
- Initialization from variance: $\sigma_w^2 = 1/n_\text{in}$ (linear/tanh), $2/n_\text{in}$ (ReLU). Output logits should start near 0 (loss ≈ $\ln V$).
- GPT-2: $\mathcal{N}(0,0.02^2)$, with residual output projections scaled by $1/\sqrt{2L}$ so the residual stream's variance doesn't grow with depth.
- LayerNorm/RMSNorm normalize per token. BatchNorm doesn't suit autoregressive models.
- Residual connections give an identity path for gradients. Pre-norm keeps that path clean and is the modern default.
- LLM pretraining underfits (≈1 epoch, dropout 0). Regularization matters mostly in fine-tuning.

## Exercises

1. Simulate a 50-layer tanh MLP of width 512 with $\sigma_w = 1/\sqrt{512}$, $\sigma_w = 1$, and $\sigma_w = \frac53/\sqrt{512}$. Plot the standard deviation of activations at each layer.
2. Prove that for $z\sim\mathcal{N}(0,s^2)$, $\E[\operatorname{ReLU}(z)] = s/\sqrt{2\pi}$ and $\E[\operatorname{ReLU}(z)^2] = s^2/2$.
3. Show that LayerNorm's output is invariant to $\mathbf{x}\mapsto a\mathbf{x}+b\mathbf{1}$ for $a>0$ (ignoring $\epsilon$). What is RMSNorm invariant to?
4. With $L = 48$ (GPT-2 XL), by what factor does Proposition 8.3 shrink the residual projections' standard deviation?
5. Show that inverted dropout is unbiased, and compute the variance it adds to an activation $h$.
6. **Code:** in `gpt.py`, remove the `RESIDUAL_SCALE` init and set `n_layer = 24`. Compare the standard deviation of the residual stream after the last block at init, with and without the scaling.

??? success "Solutions"
    1. With $1/\sqrt{512}$ the std shrinks slowly (tanh contracts). With $\sigma_w = 1$ it saturates near 1 immediately (all units at ±1). With gain $5/3$ it stays roughly constant.
    2. $\E[\operatorname{ReLU}(z)] = \int_0^\infty z\frac{1}{s\sqrt{2\pi}}e^{-z^2/2s^2}dz = \frac{s}{\sqrt{2\pi}}$. By symmetry $\E[z^2\mathbb{1}_{z>0}] = \frac12\E[z^2] = s^2/2$.
    3. $\mu\mapsto a\mu+b$, $\sigma\mapsto a\sigma$, so $(x-\mu)/\sigma$ is unchanged. RMSNorm is invariant only to scaling ($a>0$, $b = 0$).
    4. $1/\sqrt{96}\approx0.102$, so std $0.02\to0.00204$.
    5. $\tilde h = h\,m/(1-p)$ with $m\sim\text{Bernoulli}(1-p)$: $\E\tilde h = h$, $\Var\tilde h = h^2\frac{p}{1-p}$.
    6. Without scaling the residual stream's std grows roughly like $\sqrt{1+2Ls^2/\sigma_0^2}$; with scaling it stays near its embedding-level value.

## References

- X. Glorot & Y. Bengio, [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html) (AISTATS 2010).
- K. He, X. Zhang, S. Ren, J. Sun, [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (ICCV 2015) and [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2016).
- S. Ioffe & C. Szegedy, [Batch Normalization](https://arxiv.org/abs/1502.03167) (ICML 2015).
- J. L. Ba, J. R. Kiros, G. Hinton, [Layer Normalization](https://arxiv.org/abs/1607.06450) (2016).
- B. Zhang & R. Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (NeurIPS 2019).
- R. Xiong et al., [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) (ICML 2020).
- N. Srivastava et al., [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) (JMLR 2014).
- P. Nakkiran et al., [Deep Double Descent](https://arxiv.org/abs/1912.02292) (ICLR 2020).
- M. Wortsman et al., [Small-scale proxies for large-scale Transformer training instabilities](https://arxiv.org/abs/2309.14322) (ICLR 2024).
