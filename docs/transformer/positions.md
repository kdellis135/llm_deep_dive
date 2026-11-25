---
seo_title: "Embeddings & Positional Encoding: RoPE, ALiBi"
description: "Token embeddings and weight tying, why attention needs positions, sinusoidal and learned encodings, ALiBi, rotary position embedding (RoPE) derived with proof, and context extension."
---

# 11. Embeddings & Positions

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 1](../math/linear-algebra.md) (dot products, rotations), [Ch. 9](../nn/language-modeling.md) (embeddings as lookup), [Ch. 10](tokenization.md). **You will learn:** token embeddings and their geometry, weight tying, why self-attention needs position information (permutation equivariance, with proof), sinusoidal and learned absolute positions, relative schemes (T5 bias, ALiBi), rotary position embedding (RoPE) derived from complex numbers with a proof of its relative-position property, and how context windows are extended after training (position interpolation, NTK scaling, YaRN).
    **Why it matters:** this chapter covers the input of the Transformer: how a list of integer ids becomes a $T\times d$ matrix the blocks can work on. RoPE in particular is used by nearly every modern LLM (Llama, Mistral, Qwen, DeepSeek, Gemma), and its frequency structure decides how far a model's context can be stretched.

!!! watch "Watch"
    3Blue1Brown, [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) (Deep Learning Ch. 5). Embeddings as directions in a high-dimensional space, beautifully animated.

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/wjZofJX0v4M" allowfullscreen loading="lazy"></iframe></div>

    Reading: EleutherAI, [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) (2021). J. Su et al., [RoFormer](https://arxiv.org/abs/2104.09864) (2021).

---

## 11.1 Token embeddings

The first layer maps each token id $x_t$ to a vector by looking up row $x_t$ of the **embedding matrix** $\mathbf{E}\in\R^{V\times d}$ (§9.3):

$$
\mathbf{h}^{(0)}_t = \mathbf{E}[x_t]\in\R^d,\qquad \mathbf{H}^{(0)} = \begin{pmatrix}\mathbf{E}[x_1]^\top\\ \vdots\\ \mathbf{E}[x_T]^\top\end{pmatrix}\in\R^{T\times d}.
$$

In code, `nn.Embedding(V, d)(idx)` maps a `(B, T)` tensor of ids to a `(B, T, d)` tensor. The embeddings are trained by backprop like any other weights. Only the rows of tokens present in the batch receive gradient.

**Geometry.** Training puts tokens used in similar contexts at nearby points, with similarity measured by the cosine $\cos\angle(\mathbf{u},\mathbf{v}) = \mathbf{u}^\top\mathbf{v}/(\lVert\mathbf{u}\rVert\lVert\mathbf{v}\rVert)$ (Thm 1.1). Directions carry meaning: the difference between "king" and "queen" points roughly the same way as the difference between "man" and "woman". In high dimensions there's room for a huge number of almost-orthogonal directions (Prop. 1.14), which is part of why one $d$-dimensional vector can encode many features at once (superposition, Ch. 24).

### Weight tying

The output layer computes logits $\mathbf{z}_t = \mathbf{W}_\text{out}\mathbf{h}_t\in\R^V$ with $\mathbf{W}_\text{out}\in\R^{V\times d}$, the same shape as $\mathbf{E}$. **Weight tying** ([Press & Wolf, 2017](https://arxiv.org/abs/1608.05859); [Inan et al., 2017](https://arxiv.org/abs/1611.01462)) sets $\mathbf{W}_\text{out} = \mathbf{E}$, so

$$
z_{t,v} = \mathbf{E}[v]^\top\mathbf{h}_t:
$$

the logit of token $v$ is the dot product of the final hidden state with $v$'s own embedding. The model predicts tokens whose input embedding points in the direction of its final state. This saves $Vd$ parameters (38.6M of GPT-2 small's 124M) and gives rare tokens' embeddings extra gradient from the output side. GPT-2 ties (`self.lm_head.weight = self.transformer.wte.weight` in `gpt.py`). Large modern models usually don't (Llama untie them) because at scale the embedding is a small share of the parameters and separate matrices work slightly better. Small models such as Gemma and Llama 3.2 1B/3B still tie.

## 11.2 Why positions are needed

Self-attention (Ch. 12) computes, for the whole sequence at once, $\operatorname{Attn}(\mathbf{X}) = \softmax\big(\mathbf{X}\mathbf{W}_Q\mathbf{W}_K^\top\mathbf{X}^\top/\sqrt{d_h}\big)\,\mathbf{X}\mathbf{W}_V$, where the softmax is applied to each row. The MLP and normalization layers act on each row separately.

!!! theorem "Proposition 11.1 (Attention without positions is permutation-equivariant)"
    For any permutation matrix $\mathbf{P}$ (which reorders the rows of $\mathbf{X}$), $\operatorname{Attn}(\mathbf{P}\mathbf{X}) = \mathbf{P}\operatorname{Attn}(\mathbf{X})$. Reordering the input tokens just reorders the outputs.

??? proof "Proof"
    Let $\mathbf{S} = \mathbf{X}\mathbf{W}_Q\mathbf{W}_K^\top\mathbf{X}^\top/\sqrt{d_h}$. For $\mathbf{P}\mathbf{X}$ the scores become $\mathbf{P}\mathbf{S}\mathbf{P}^\top$: rows and columns permuted the same way. A row-wise softmax commutes with permuting rows, and with permuting the columns inside each row, so $\softmax(\mathbf{P}\mathbf{S}\mathbf{P}^\top) = \mathbf{P}\softmax(\mathbf{S})\mathbf{P}^\top$. Multiplying by $\mathbf{P}\mathbf{X}\mathbf{W}_V$ and using $\mathbf{P}^\top\mathbf{P} = \mathbf{I}$ gives $\mathbf{P}\softmax(\mathbf{S})\mathbf{X}\mathbf{W}_V$. Row-wise layers trivially commute with $\mathbf{P}$, so the whole Transformer stack does too. $\blacksquare$

So without position information, "dog bites man" and "man bites dog" produce the same set of vectors, just reordered. The model needs to know *where* each token is. (A *causal* mask breaks the symmetry, since token $t$ sees exactly $t$ tokens, and decoder-only models with no position encoding at all, "NoPE", can learn positions from that ([Kazemnejad et al., 2023](https://arxiv.org/abs/2305.19466)). But explicit positions work better in practice.)

## 11.3 Absolute positions

**Sinusoidal** (the original Transformer, [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)): add a fixed vector $\mathbf{p}_t$ to each embedding, with

$$
p_{t,2i} = \sin(\omega_it),\qquad p_{t,2i+1} = \cos(\omega_it),\qquad \omega_i = 10000^{-2i/d},\quad i = 0,\dots,\tfrac d2-1.
$$

Each pair of coordinates is a clock ticking at its own frequency, from $\omega_0 = 1$ radian per token (period $2\pi\approx6$ tokens) down to $\omega_{d/2-1}\approx10^{-4}$ (period ≈ 60,000 tokens), like the hands of a clock or the digits of a binary counter.

!!! theorem "Proposition 11.2 (Shifting position is a rotation)"
    For every offset $k$ there is a fixed linear map, independent of $t$, taking $\mathbf{p}_t$ to $\mathbf{p}_{t+k}$. On each pair it's a rotation by angle $\omega_ik$.

??? proof "Proof"
    By the angle-addition formulas,

    $$
    \begin{pmatrix}\sin\omega(t+k)\\ \cos\omega(t+k)\end{pmatrix} = \begin{pmatrix}\cos\omega k & \sin\omega k\\ -\sin\omega k & \cos\omega k\end{pmatrix}\begin{pmatrix}\sin\omega t\\ \cos\omega t\end{pmatrix}.
    $$

    The matrix depends only on $k$. Stack one such block per pair. $\blacksquare$

The authors hoped this would let the model attend by *relative* position. RoPE (§11.5) turns that hope into an exact property.

**Learned absolute positions** (GPT-2, BERT): a second embedding table $\mathbf{P}\in\R^{T_\max\times d}$, trained like $\mathbf{E}$, so that $\mathbf{h}^{(0)}_t = \mathbf{E}[x_t]+\mathbf{P}[t]$. In `gpt.py` this is `wpe` with `block_size = 1024` rows. It's simple and works well, but positions beyond $T_\max$ have no embedding at all, so the context can't be extended.

## 11.4 Relative position biases

Instead of adding to the input, add a position-dependent term directly to the attention scores $s_{ij}$ between query $i$ and key $j$.

- **T5** ([Raffel et al., 2020](https://arxiv.org/abs/1910.10683)): a learned scalar bias $b_{\text{bucket}(i-j)}$ per head, with logarithmically sized buckets for distant offsets.
- **ALiBi** ([Press et al., 2022](https://arxiv.org/abs/2108.12409)): a fixed, linear penalty $s_{ij}\leftarrow s_{ij} - m_h\,(i-j)$, with head-specific slopes $m_h = 2^{-8h/H}$. Each head has a built-in recency bias at its own scale. ALiBi was designed to **extrapolate**: trained on 1k tokens, it degrades gracefully at 2k+. Used by BLOOM and MPT.

## 11.5 Rotary position embedding (RoPE)

RoPE ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) **rotates** the query and key vectors by an angle proportional to their position, just before the dot product. Values aren't rotated, and nothing is added to the residual stream.

### The 2-D case with complex numbers

Identify a 2-D vector $(a,b)$ with the complex number $a+ib$. Multiplying by $e^{i\phi}$ rotates it by $\phi$. The real dot product of two 2-D vectors is $\langle\mathbf{u},\mathbf{v}\rangle = \operatorname{Re}(u\,\bar v)$, where $\bar v$ is the complex conjugate.

Rotate a query at position $m$ by $m\theta$ and a key at position $n$ by $n\theta$:

$$
\langle R_m\mathbf{q},\ R_n\mathbf{k}\rangle = \operatorname{Re}\big(q e^{im\theta}\,\overline{ke^{in\theta}}\big) = \operatorname{Re}\big(q\bar k\,e^{i(m-n)\theta}\big).
$$

The absolute positions $m$ and $n$ have disappeared. **Only the offset $m-n$ remains.**

### General dimension

Split a $d_h$-dimensional vector into $d_h/2$ pairs and rotate pair $i$ by angle $m\theta_i$, with geometrically spaced frequencies $\theta_i = b^{-2i/d_h}$ (base $b = 10{,}000$ originally):

$$
R_m = \begin{pmatrix} R(m\theta_0) & & \\ & \ddots & \\ & & R(m\theta_{d_h/2-1})\end{pmatrix},\qquad R(\phi) = \begin{pmatrix}\cos\phi&-\sin\phi\\ \sin\phi&\cos\phi\end{pmatrix}.
$$

!!! theorem "Theorem 11.3 (RoPE scores depend only on relative position)"
    For all $\mathbf{q},\mathbf{k}\in\R^{d_h}$ and positions $m,n$:

    $$
    (R_m\mathbf{q})^\top(R_n\mathbf{k}) = \mathbf{q}^\top R_{n-m}\,\mathbf{k}.
    $$

    Moreover $\lVert R_m\mathbf{q}\rVert = \lVert\mathbf{q}\rVert$: rotation changes no lengths.

??? proof "Proof"
    Each $R(\phi)$ is orthogonal with $R(\phi)^\top = R(-\phi)$ and $R(\alpha)R(\beta) = R(\alpha+\beta)$ (Thm 1.9). Block-diagonal matrices multiply block by block, so $R_m^\top R_n = R_{-m}R_n = R_{n-m}$. Then $(R_m\mathbf{q})^\top(R_n\mathbf{k}) = \mathbf{q}^\top R_m^\top R_n\mathbf{k} = \mathbf{q}^\top R_{n-m}\mathbf{k}$. Orthogonal matrices preserve norms. $\blacksquare$

<div class="widget" data-widget="rope"></div>

The test suite checks both properties numerically: the score for positions (10, 3) equals the score for (110, 103), and rotation preserves the norm. Here is the implementation. The rotation is applied elementwise using precomputed $\cos$ and $\sin$ tables, never as a matrix:

```python
--8<-- "code/llama.py:rope"
```

!!! warning "Two layouts of the pairs"
    The RoFormer paper and Meta's reference Llama code rotate **interleaved** pairs $(x_0,x_1),(x_2,x_3),\dots$, as above. Hugging Face's Llama implementation uses a **half-split** layout, pairing $x_i$ with $x_{i+d_h/2}$ (the `rotate_half` function), and permutes the $\mathbf{W}_Q,\mathbf{W}_K$ weights when converting checkpoints. Both are valid RoPE. Mixing them silently ruins a model. If you load weights from one codebase into another and the outputs are garbage, check this first.

### What the frequencies do

The pair with frequency $\theta_i$ has wavelength $\lambda_i = 2\pi/\theta_i = 2\pi b^{2i/d_h}$ tokens. With $d_h = 128$ and $b = 10{,}000$, wavelengths range from about 6 tokens (fast pairs that encode fine local order) to about 54,000 tokens (slow pairs that barely rotate across a document and act almost like position-free "semantic" channels). Su et al. also show that, for typical vectors, the attainable attention score decays with distance $|m-n|$, a mild built-in recency bias.

**Why RoPE won.** It encodes relative position exactly; it adds no parameters; it only touches $\mathbf{q}$ and $\mathbf{k}$, so it works with KV caching (a cached key was already rotated by its own position, and that never changes); and it's compatible with fused attention kernels.

## 11.6 Extending the context window

A model trained with context $T_\text{train}$ has never seen rotation angles $m\theta_i$ beyond $T_\text{train}\theta_i$. For slow pairs whose wavelength exceeds $T_\text{train}$, larger positions produce angles never seen in training, and quality collapses past $T_\text{train}$. Fixes, usually followed by a short fine-tune on long documents:

- **Position interpolation** ([Chen et al., 2023](https://arxiv.org/abs/2306.15595)): to extend by a factor $s$, use position $m/s$ instead of $m$. All angles stay within the trained range, but nearby tokens become harder to tell apart.
- **NTK-aware scaling** ([bloc97, 2023](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)): increase the base $b$ instead. That stretches the slow frequencies a lot and the fast ones hardly at all, preserving local resolution.
- **YaRN** ([Peng et al., 2023](https://arxiv.org/abs/2309.00071)): interpolate per frequency band (don't touch high frequencies, fully interpolate low ones, blend in between) and slightly sharpen the attention softmax. Used by Qwen, DeepSeek and others.
- **Large base from the start**: Llama 3 trains with $b = 500{,}000$ (slowest wavelength ≈ 2.6M tokens), then extends from 8K to 128K tokens with a frequency-dependent rescaling in Llama 3.1.

!!! llm "Long context in practice"
    A long *window* isn't the same as long-context *ability*. Models can accept 128K–1M tokens but still miss information in the middle of long inputs ([Liu et al., Lost in the Middle, 2023](https://arxiv.org/abs/2307.03172)). Benchmarks such as needle-in-a-haystack and [RULER](https://arxiv.org/abs/2404.06654) test whether a model actually uses its window. Design prompts accordingly (Ch. 25).

## 11.7 Summary: the input to block 1

| Model | Input to block 1 | Position info |
|---|---|---|
| GPT-2 | $\mathbf{E}[x_t]+\mathbf{P}[t]$ | learned absolute, added once |
| Original Transformer | $\sqrt d\,\mathbf{E}[x_t]+\mathbf{p}_t$ | sinusoidal, added once |
| Llama / Mistral / Qwen / DeepSeek | $\mathbf{E}[x_t]$ | RoPE applied to $\mathbf{q},\mathbf{k}$ **in every attention layer** |

## Key takeaways

- An embedding is a learned lookup table $\mathbf{E}\in\R^{V\times d}$. With weight tying, the logits are dot products with the same embeddings.
- Without position information, attention is permutation-equivariant: word order is invisible.
- Sinusoidal and learned positions are absolute and added to the input. Learned positions can't go past $T_\max$.
- ALiBi and T5 add relative biases to the attention scores.
- **RoPE** rotates $\mathbf{q}$ and $\mathbf{k}$ by position-dependent angles, so $\langle R_m\mathbf{q},R_n\mathbf{k}\rangle$ depends only on $m-n$. It has no parameters, preserves norms, and works with KV caching.
- Context extension rescales RoPE's positions or frequencies (PI, NTK, YaRN, larger base) and then fine-tunes on long data.

## Exercises

1. Prove that the weight-tied logits of a token $v$ are maximized, for a fixed norm of $\mathbf{h}_t$, when $\mathbf{h}_t$ points along $\mathbf{E}[v]$.
2. Show that ALiBi is *not* permutation-equivariant, but *is* invariant to shifting all positions by a constant.
3. For RoPE with $d_h = 128$, $b = 10{,}000$, which pair index $i$ has a wavelength closest to 4,096 tokens?
4. Verify Theorem 11.3 numerically: take random $\mathbf{q},\mathbf{k}$ and check that the score is the same for $(m,n) = (5,2)$ and $(1005, 1002)$, using `rope_cache` and `apply_rope` from `llama.py`.
5. Position interpolation by $s = 4$ maps positions $0..32767$ into $0..8191$. What is the smallest *angle difference* between adjacent tokens on the fastest pair, before and after?
6. Why doesn't RoPE rotate the value vectors?

??? success "Solutions"
    1. $\mathbf{E}[v]^\top\mathbf{h}_t\le\lVert\mathbf{E}[v]\rVert\lVert\mathbf{h}_t\rVert$ by Cauchy–Schwarz, with equality iff $\mathbf{h}_t$ is a positive multiple of $\mathbf{E}[v]$.
    2. The bias $-m(i-j)$ depends on positions, so permuting tokens changes it. Shifting all positions by $c$ leaves every $i-j$ unchanged.
    3. $\lambda_i = 2\pi\cdot10000^{2i/128} = 4096$ gives $2i/128 = \log_{10}(4096/2\pi)/4\approx0.704$, so $i\approx45$.
    4. They match to float precision (that's the `RoPE score depends only on m - n` check in `test_all.py`).
    5. Before: $\theta_0 = 1$ radian per token. After: $1/4$ radian. Adjacent tokens are 4× harder to distinguish on the fastest pair, which is the main weakness of PI and what NTK/YaRN avoid.
    6. Only the *scores* need positional information, to decide where to attend. Rotating values would put position-dependent rotations into the content written to the residual stream, for no benefit.

## References

- A. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017), §3.5.
- O. Press & L. Wolf, [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) (EACL 2017).
- J. Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (2021).
- O. Press, N. Smith, M. Lewis, [Train Short, Test Long: Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409) (ICLR 2022).
- S. Chen et al., [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) (2023).
- B. Peng et al., [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) (ICLR 2024).
- A. Kazemnejad et al., [The Impact of Positional Encoding on Length Generalization in Transformers](https://arxiv.org/abs/2305.19466) (NeurIPS 2023).
