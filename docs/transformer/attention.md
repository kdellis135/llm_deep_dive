# 12. Attention

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 1](../math/linear-algebra.md) (dot products, high-dimensional geometry), [§2.5](../math/calculus.md#25-matrix-calculus-you-will-actually-use) (softmax Jacobian), [Ch. 7](../nn/backprop.md), [Ch. 11](positions.md). **You will learn:** attention as a soft, differentiable dictionary lookup, scaled dot-product attention with exact shapes, a proof of why scores are divided by $\sqrt{d_h}$, causal masking and why it lets one sequence train $T$ predictions at once, the full backward pass of attention, multi-head attention and its cost, what attention heads actually learn (previous-token and induction heads), and the variants that later chapters build on: cross-attention, linear attention, sliding windows.
    **Why it matters:** attention is the only place in a Transformer where tokens exchange information. Everything else operates on each position separately. Understanding attention means understanding how an LLM uses its context.

!!! watch "Watch"
    3Blue1Brown, [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) (Deep Learning Ch. 6).

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/eMlx5fFNoYc" allowfullscreen loading="lazy"></iframe></div>

    Then Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY): the section "the mathematical trick in self-attention" builds attention from a running average, exactly as in §12.1. Reading: J. Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/); L. Weng, [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/). Interactive: [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) and B. Bycroft's [LLM Visualization](https://bbycroft.net/llm).

---

## 12.1 From averaging to attention

Suppose each position $t$ wants to gather information from the previous positions $1..t$. The simplest choice is a plain **average**: $\mathbf{y}_t = \frac1t\sum_{s\le t}\mathbf{x}_s$. As a matrix, $\mathbf{Y} = \mathbf{A}\mathbf{X}$ with a lower-triangular $\mathbf{A}$ whose row $t$ is $(\frac1t,\dots,\frac1t,0,\dots,0)$. Karpathy's "mathematical trick": build $\mathbf{A}$ by filling the upper triangle of a zero matrix with $-\infty$ and applying a row-wise softmax.

A uniform average treats every past token as equally relevant. **Attention makes the weights data-dependent**: each position computes how relevant every earlier position is to it, and averages accordingly. It works like a soft dictionary lookup:

- every position emits a **query** $\mathbf{q}_t$ ("what am I looking for?"),
- a **key** $\mathbf{k}_s$ ("what do I contain?"),
- and a **value** $\mathbf{v}_s$ ("what do I pass on if selected?").

The output at $t$ is a weighted average of the values, weighted by how well the query matches each key. A hard dictionary returns the value of the single exactly matching key; attention returns a *blend* of all values, weighted by similarity, and it's differentiable, so the queries, keys and values can all be learned.

## 12.2 Scaled dot-product attention

!!! definition "Definition 12.1 (Scaled dot-product attention)"
    For input rows $\mathbf{X}\in\R^{T\times d}$, projections $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V\in\R^{d\times d_h}$ (the paper's convention; see [Notation](../start/notation.md)):

    $$
    \mathbf{Q} = \mathbf{X}\mathbf{W}_Q,\quad \mathbf{K} = \mathbf{X}\mathbf{W}_K,\quad \mathbf{V} = \mathbf{X}\mathbf{W}_V\qquad(\text{each }T\times d_h),
    $$

    $$
    \operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \underbrace{\softmax\!\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_h}} + \mathbf{M}\Big)}_{\mathbf{A}\ \in\ \R^{T\times T}}\mathbf{V}\ \in\R^{T\times d_h},
    $$

    with the softmax applied to each row and $\mathbf{M}$ the causal mask ($M_{ts} = 0$ for $s\le t$, $-\infty$ for $s>t$).

Per position, with $s_{ts} = \mathbf{q}_t^\top\mathbf{k}_s/\sqrt{d_h}$:

$$
a_{ts} = \frac{\exp(s_{ts})}{\sum_{s'\le t}\exp(s_{ts'})},\qquad \mathbf{y}_t = \sum_{s\le t}a_{ts}\,\mathbf{v}_s.
$$

Shapes to memorize: scores and weights are $T\times T$; each row of $\mathbf{A}$ is a probability distribution over the positions that row may see; the output has one $d_h$-vector per position.

```python
--8<-- "code/attention.py:sdpa"
```

The test suite checks this function against PyTorch's fused `F.scaled_dot_product_attention`, checks that rows sum to 1, and checks that the upper triangle of $\mathbf{A}$ is exactly zero.

!!! intuition "Attention is a kernel smoother"
    $\mathbf{y}_t = \sum_s\frac{\kappa(\mathbf{q}_t,\mathbf{k}_s)}{\sum_{s'}\kappa(\mathbf{q}_t,\mathbf{k}_{s'})}\mathbf{v}_s$ with the kernel $\kappa(\mathbf{q},\mathbf{k}) = \exp(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h})$. That's the Nadaraya–Watson estimator from 1960s statistics: predict at a query point by averaging the values at nearby keys, weighted by a similarity kernel. The difference is that a Transformer **learns** the space ($\mathbf{W}_Q,\mathbf{W}_K$) in which similarity is measured ([Tsai et al., 2019](https://arxiv.org/abs/1908.11775)).

## 12.3 Why divide by $\sqrt{d_h}$? { data-toc-label="12.3 Why divide by √dₕ?" }

!!! theorem "Proposition 12.1 (Dot products grow like $\sqrt{d_h}$)"
    If $\mathbf{q},\mathbf{k}\in\R^{d_h}$ have independent entries with mean 0 and variance 1, then $\E[\mathbf{q}^\top\mathbf{k}] = 0$ and $\Var(\mathbf{q}^\top\mathbf{k}) = d_h$. Hence $\Var(\mathbf{q}^\top\mathbf{k}/\sqrt{d_h}) = 1$ for every $d_h$.

??? proof "Proof"
    Each term $q_ik_i$ has mean $\E q_i\E k_i = 0$ and variance $\E[q_i^2]\E[k_i^2] = 1$, and the terms are independent, so their variances add: $\Var\sum_iq_ik_i = d_h$. Dividing by $\sqrt{d_h}$ divides the variance by $d_h$. $\blacksquare$

Why do large scores hurt? With unit-variance scores, the softmax is reasonably spread out. With $d_h = 128$ and no scaling, scores have standard deviation about 11, the largest score dominates, and the softmax becomes nearly one-hot. Its Jacobian $\diag(\mathbf{a})-\mathbf{a}\mathbf{a}^\top$ (Thm 2.7) then goes to **zero**: if $\mathbf{a}\approx\mathbf{e}_j$, every entry $a_i(\delta_{ik}-a_k)\approx0$. Gradients through the attention weights vanish and the queries and keys stop learning. Dividing by $\sqrt{d_h}$ keeps the softmax in its responsive regime at initialization, whatever the head size.

<div class="widget" data-widget="attention-scaling"></div>

## 12.4 Causal masking: $T$ predictions from one pass { data-toc-label="12.4 Causal masking: T predictions from one pass" }

A language model predicts $x_{t+1}$ from $x_{\le t}$. If position $t$ could attend to position $t+1$, it could just copy the answer. The **causal mask** sets $s_{ts} = -\infty$ for $s>t$, so $e^{-\infty} = 0$ and those weights vanish exactly.

!!! theorem "Proposition 12.2 (Causality)"
    With a causal mask in every attention layer, the output at position $t$ of a Transformer is a function of $x_1,\dots,x_t$ only.

??? proof "Proof"
    By induction on layers. Block inputs at position $t$ depend only on $x_t$ (embedding) and, by the induction hypothesis, layer outputs at $t$ depend only on $x_{\le t}$. Attention at $t$ combines values at positions $s\le t$ with weights computed from $\mathbf{q}_t$ and $\mathbf{k}_{s\le t}$, all functions of $x_{\le t}$. MLPs, norms and residual adds act position-wise. $\blacksquare$

    (The test suite checks this directly: perturbing the last token leaves every earlier output unchanged.)

This is what makes Transformer training efficient. One forward pass over a sequence of $T$ tokens produces **$T$ next-token predictions at once**, each using exactly the context it would have at generation time, and the loss averages all of them (teacher forcing, §9.9). An RNN needs $T$ sequential steps for the same thing.

## 12.5 The backward pass of attention

Let $\mathbf{S} = \mathbf{Q}\mathbf{K}^\top/\sqrt{d_h}$, $\mathbf{A} = \softmax(\mathbf{S})$ row-wise, $\mathbf{Y} = \mathbf{A}\mathbf{V}$, with upstream gradient $\bar{\mathbf{Y}}$. Applying the rules of §7.5 line by line in reverse:

$$
\bar{\mathbf{V}} = \mathbf{A}^\top\bar{\mathbf{Y}},\qquad \bar{\mathbf{A}} = \bar{\mathbf{Y}}\mathbf{V}^\top,\qquad \bar{\mathbf{S}}_{t,:} = \mathbf{A}_{t,:}\odot\big(\bar{\mathbf{A}}_{t,:}-(\bar{\mathbf{A}}_{t,:}\cdot\mathbf{A}_{t,:})\mathbf{1}\big),
$$

$$
\bar{\mathbf{Q}} = \frac{\bar{\mathbf{S}}\,\mathbf{K}}{\sqrt{d_h}},\qquad \bar{\mathbf{K}} = \frac{\bar{\mathbf{S}}^\top\mathbf{Q}}{\sqrt{d_h}}.
$$

Masked positions have $A_{ts} = 0$, so they get zero gradient automatically. Each line is checked against autograd in the tests:

```python
--8<-- "code/attention.py:backward"
```

Note that the backward pass needs $\mathbf{A}$, a $T\times T$ matrix per head. Storing it for every head and layer is what makes naive attention memory-hungry at long context. FlashAttention avoids storing it by *recomputing* it block by block during the backward pass (Ch. 19).

## 12.6 Multi-head attention

One attention head computes one weighted average per position, so it can follow one "kind" of relation at a time. **Multi-head attention** runs $H$ heads in parallel, each with its own $\mathbf{W}_Q^{(h)},\mathbf{W}_K^{(h)},\mathbf{W}_V^{(h)}\in\R^{d\times d_h}$, usually with $d_h = d/H$, then concatenates them and mixes them with an output projection $\mathbf{W}_O\in\R^{d\times d}$:

$$
\operatorname{MHA}(\mathbf{X}) = \big[\operatorname{head}_1;\dots;\operatorname{head}_H\big]\,\mathbf{W}_O,\qquad \operatorname{head}_h = \operatorname{Attention}(\mathbf{X}\mathbf{W}_Q^{(h)},\mathbf{X}\mathbf{W}_K^{(h)},\mathbf{X}\mathbf{W}_V^{(h)}).
$$

In code, all heads' projections are fused into one $d\times3d$ matrix multiplication, and the head dimension is split out with a reshape. No loop over heads:

```python
--8<-- "code/attention.py:mha"
```

Follow the shapes: `(B, T, C)` → fused projection `(B, T, 3C)` → split into q, k, v `(B, T, C)` → view as `(B, T, H, d_h)` → transpose to `(B, H, T, d_h)`, so batch and head are both "batch-like" dimensions → attention per head → transpose back and concatenate to `(B, T, C)` → $\mathbf{W}_O$.

<div class="widget" data-widget="attention"></div>

**Parameters and cost.** $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V,\mathbf{W}_O$ give $4d^2$ parameters (plus biases), independent of $H$. Per sequence, the projections cost $2\cdot4Td^2 = 8Td^2$ FLOPs, and the scores and weighted sum cost $2\cdot2T^2d$ FLOPs ($\mathbf{Q}\mathbf{K}^\top$ and $\mathbf{A}\mathbf{V}$, summed over heads), halved in principle by the causal mask. The $T^2$ term dominates once $T$ is larger than a few times $d$. Memory for the scores is $O(HT^2)$ per layer unless fused.

**The output projection is a sum over heads.** Split $\mathbf{W}_O$ into row blocks $\mathbf{W}_O^{(h)}\in\R^{d_h\times d}$. Then $\operatorname{MHA}(\mathbf{X}) = \sum_h\operatorname{head}_h\mathbf{W}_O^{(h)}$: **each head independently writes its result into the residual stream**. Elhage et al. ([2021](https://transformer-circuits.pub/2021/framework/index.html)) build their whole interpretability framework on this: each head is a pair of low-rank matrices, a "QK circuit" $\mathbf{W}_Q^{(h)}\mathbf{W}_K^{(h)\top}$ that decides *where* to read, and an "OV circuit" $\mathbf{W}_V^{(h)}\mathbf{W}_O^{(h)}$ that decides *what* to write.

## 12.7 What attention heads learn

Researchers have identified heads with clear, consistent jobs.

- **Previous-token heads** attend from position $t$ to $t-1$. They copy information about the previous token forward, which lets later layers see bigrams.
- **Induction heads** ([Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)) implement copying: if the context contains "... **A B** ... **A**", they predict **B**. The mechanism uses two heads in different layers. A previous-token head writes "the token before me was A" into position B's residual stream. Then, at the second A, an induction head's query ("I am A") matches the key at B ("the token before me was A"), and its OV circuit copies B to the output. Induction heads form abruptly at one point in training, at the same time as a sharp improvement in **in-context learning**: loss on later tokens of a document falls relative to earlier ones. They're the simplest example of a model using its context as working memory.
- **Other observed heads** track syntax (a verb attending to its subject), resolve coreference, and **attention sinks**: many heads put large weight on the first token (or a few fixed tokens) when they have "nothing to do", using it as a no-op ([Xiao et al., 2023](https://arxiv.org/abs/2309.17453)). Keeping these sink tokens is necessary when evicting old tokens from a KV cache (Ch. 19).

!!! llm "In-context learning, mechanically"
    "Few-shot prompting works" (Ch. 25) is, at its simplest, induction heads matching the pattern of your examples and copying the format to the new input. Larger models build far more abstract versions of the same matching, such as "a question in this style is followed by an answer in that style".

## 12.8 Variants you will meet

- **Cross-attention** (encoder–decoder models, e.g. the original Transformer, T5, Whisper): queries come from the decoder, keys and values from the encoder's output. No causal mask on the encoder side.
- **Multi-query and grouped-query attention (MQA/GQA)**: several query heads share one key/value head, cutting the KV cache by a factor $H/H_{kv}$ (Ch. 19–20).
- **Multi-head latent attention (MLA)** (DeepSeek-V2/V3): compress keys and values to a small latent vector per token (Ch. 20).
- **Sliding-window attention** (Mistral 7B, Gemma 2/3 local layers): each token attends only to the last $w$ tokens. Cost is $O(Tw)$ instead of $O(T^2)$. Information still travels further through depth, since $L$ layers reach $L\cdot w$ tokens back.
- **Linear attention** ([Katharopoulos et al., 2020](https://arxiv.org/abs/2006.16236)): replace $\exp(\mathbf{q}^\top\mathbf{k})$ with $\phi(\mathbf{q})^\top\phi(\mathbf{k})$ for a feature map $\phi$. Then

    $$
    \mathbf{y}_t = \frac{\phi(\mathbf{q}_t)^\top\sum_{s\le t}\phi(\mathbf{k}_s)\mathbf{v}_s^\top}{\phi(\mathbf{q}_t)^\top\sum_{s\le t}\phi(\mathbf{k}_s)},
    $$

    and the sums can be updated one token at a time: a **recurrent** state of fixed size. Cost is linear in $T$, generation cost per token is constant, but quality lags softmax attention, because a fixed-size state can't hold arbitrary context. State-space models like Mamba ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)) and gated linear attention refine this. Hybrid models interleave them with a few full-attention layers (Ch. 20).

## Key takeaways

- Attention: $\softmax(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_h}+\mathbf{M})\mathbf{V}$, a learned, data-dependent weighted average of values; each row of weights is a distribution.
- $\sqrt{d_h}$ keeps score variance at 1, so the softmax doesn't saturate and gradients flow.
- The causal mask makes position $t$ depend only on $x_{\le t}$, so one forward pass trains $T$ predictions.
- Backward: $\bar{\mathbf{V}} = \mathbf{A}^\top\bar{\mathbf{Y}}$, softmax VJP per row, then $\bar{\mathbf{Q}},\bar{\mathbf{K}}$ via $\bar{\mathbf{S}}$. It needs $\mathbf{A}$, which is $O(T^2)$ memory.
- Multi-head attention: $H$ heads of size $d/H$ in parallel, $4d^2$ parameters; each head reads with a QK circuit and writes with an OV circuit into the residual stream.
- Heads learn interpretable jobs. Previous-token + induction heads implement copying and underlie in-context learning.

## Exercises

1. Show that if all keys are equal, attention returns the average of the (visible) values regardless of the query.
2. Take $T = 3$, $d_h = 1$, $\mathbf{q} = \mathbf{k} = (1,2,3)^\top$, $\mathbf{v} = (10,20,30)^\top$, causal. Compute $\mathbf{A}$ and $\mathbf{y}$ by hand.
3. Prove that $\softmax(\mathbf{s}/\tau)$ converges to a one-hot vector on $\argmax\mathbf{s}$ as $\tau\to0$ (assume a unique maximum), and to uniform as $\tau\to\infty$.
4. Count FLOPs for one attention layer of GPT-2 small ($d = 768$, $H = 12$) at $T = 1024$: projections vs scores. Which dominates? At what $T$ do they match?
5. Why is the cost of MHA's projections independent of $H$, while the number of $T\times T$ score matrices grows with $H$?
6. **Code:** extract attention weights from `attention()` for a trained small model (e.g. the char-level GPT from Ch. 14) and plot them for a few heads. Can you find a previous-token head?

??? success "Solutions"
    1. Equal keys give equal scores in each row, so the softmax is uniform over the visible positions, and $\mathbf{y}_t$ is the plain average of $\mathbf{v}_{1..t}$.
    2. Scores $s_{ts} = q_tk_s$: row 1: $(1)$ → $a = (1)$, $y_1 = 10$. Row 2: $(2, 4)$ → $a = (0.119, 0.881)$, $y_2\approx18.8$. Row 3: $(3,6,9)$ → $a\approx(0.0025, 0.0473, 0.9502)$, $y_3\approx29.5$.
    3. $\frac{e^{s_i/\tau}}{\sum_je^{s_j/\tau}} = \frac{1}{\sum_je^{(s_j-s_i)/\tau}}$. For $i = \argmax$, every other term has $s_j-s_i<0$ and vanishes as $\tau\to0$, so the weight tends to 1. As $\tau\to\infty$ all exponents tend to 0 and the weights to $1/n$.
    4. Projections: $8Td^2 = 8\cdot1024\cdot768^2\approx4.8$ GFLOPs. Scores + weighted sum: $4T^2d = 4\cdot1024^2\cdot768\approx3.2$ GFLOPs (before exploiting the mask). Projections dominate slightly; they match at $T = 2d = 1536$.
    5. The projections are $d\times d$ matrices whatever the split into heads; splitting only reshapes them. But each head computes its own $T\times T$ matrix, of total size $HT^2$. (The FLOPs are the same, $\sum_h2T^2d_h = 2T^2d$, but memory isn't.)
    6. Look for a head whose weight matrix has a bright line just below the diagonal.

## References

- A. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017).
- D. Bahdanau, K. Cho, Y. Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (2015).
- N. Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Anthropic, 2021).
- C. Olsson et al., [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (Anthropic, 2022).
- A. Katharopoulos et al., [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) (ICML 2020).
- G. Xiao et al., [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) (ICLR 2024).
- A. Rush, [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
