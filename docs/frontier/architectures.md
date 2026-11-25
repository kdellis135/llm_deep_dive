---
seo_title: "Modern LLM Architectures: Llama, Mixture-of-Experts, MLA"
description: "Every change from GPT-2 to Llama 3 and why, a Llama-style model in code, real open-model configs, mixture-of-experts routing and load balancing, and DeepSeek's multi-head latent attention."
---

# 20. Modern Architectures

!!! abstract "In this chapter"
    **Prerequisites:** Part III, [Ch. 16](../scale/distributed.md) (expert parallelism), [Ch. 19](../inference/efficient.md) (KV cache). **You will learn:** the exact list of changes from GPT-2 to Llama 3 and why each one was made; a complete Llama-style model in code; real configurations of current open models; mixture-of-experts (routing, load balancing with proof, shared and fine-grained experts, auxiliary-loss-free balancing) with a working MoE layer; DeepSeek's multi-head latent attention and its KV-cache arithmetic; local/global attention patterns; state-space and hybrid models; multi-token prediction; and how text models become multimodal.
    **Why it matters:** this is what frontier and open models actually look like in 2025–2026. When you read a new model card or technical report, this chapter tells you what each line means.

!!! watch "Read & watch"
    - S. Raschka, [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) (2025), a clear side-by-side of DeepSeek-V3, Llama 4, Qwen3, Gemma 3, gpt-oss and others.
    - Meta, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024) and DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (2024), the two most detailed open technical reports.
    - Umar Jamil's and Karpathy's videos implement Llama-style models; our [`code/llama.py`](https://github.com/kdellis135/llm_deep_dive/blob/main/code/llama.py) is a compact, tested version.

---

## 20.1 From GPT-2 to Llama

The "Llama recipe" (Touvron et al., [2023a](https://arxiv.org/abs/2302.13971), [2023b](https://arxiv.org/abs/2307.09288); Meta, 2024) is a set of changes to the GPT-2 block, each one proven elsewhere and now nearly universal:

| Component | GPT-2 (2019) | Llama 3 (2024) | Why | Chapter |
|---|---|---|---|---|
| Normalization | LayerNorm | **RMSNorm** | as good, cheaper (no mean, no bias) | §8.3 |
| Positions | learned absolute, added at input | **RoPE** on q, k in every layer | relative positions, extendable context | §11.5 |
| FFN | GELU, $4d$ hidden | **SwiGLU**, $\approx3.5d$ hidden | better loss per parameter | §13.2 |
| Attention | MHA | **GQA** (8 KV heads) | 8× smaller KV cache | §19.2 |
| Biases | everywhere | **none** | no benefit; better stability | |
| Embeddings | tied | untied (large models) | small cost at scale, slightly better | §11.1 |
| Vocabulary | 50,257 | 128,256 | shorter sequences, multilingual | §10.8 |
| Context | 1,024 | 8K → 128K | RoPE base 500K + scaling | §11.6 |
| Data | 10B tokens | 15.6T tokens | overtraining for inference efficiency | §17.4 |

RMSNorm in code (note the computation in fp32, for stability under bf16):

```python
--8<-- "code/llama.py:rmsnorm"
```

The complete model, with the KV-cached generation loop of §19.2:

```python
--8<-- "code/llama.py:model"
```

Compare with `gpt.py` (Ch. 14): the structure is identical (embedding → blocks with residual pre-norm attention and FFN → final norm → output projection). Only the components inside changed.

### Real configurations

| Model | Params | $L$ | $d$ | $H$ / $H_{kv}$ | FFN hidden | Vocab | Context |
|---|---|---|---|---|---|---|---|
| GPT-2 small | 124M | 12 | 768 | 12 / 12 | 3,072 (GELU) | 50,257 | 1K |
| GPT-3 | 175B | 96 | 12,288 | 96 / 96 | 49,152 | 50,257 | 2K |
| Mistral 7B | 7.2B | 32 | 4,096 | 32 / 8 | 14,336 | 32,000 | 32K (sliding window 4K) |
| Llama 3 8B | 8.0B | 32 | 4,096 | 32 / 8 | 14,336 | 128,256 | 128K (3.1) |
| Llama 3 70B | 70.6B | 80 | 8,192 | 64 / 8 | 28,672 | 128,256 | 128K |
| Llama 3 405B | 405B | 126 | 16,384 | 128 / 8 | 53,248 | 128,256 | 128K |

## 20.2 Mixture of experts

A dense model uses every parameter for every token. A **mixture-of-experts (MoE)** model replaces each FFN with $E$ expert FFNs and a **router** that sends each token to only $k$ of them ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538); [Fedus et al., Switch Transformer, 2022](https://arxiv.org/abs/2101.03961)):

$$
\mathbf{p} = \softmax(\mathbf{W}_r\mathbf{x})\in\R^E,\qquad \mathcal{T} = \operatorname{top-}k(\mathbf{p}),\qquad \operatorname{MoE}(\mathbf{x}) = \sum_{e\in\mathcal{T}}\frac{p_e}{\sum_{e'\in\mathcal{T}}p_{e'}}\operatorname{FFN}_e(\mathbf{x}).
$$

This decouples **total parameters** (capacity: knowledge, skills) from **active parameters** (compute per token). A model can know much more for the same FLOPs per token.

```python
--8<-- "code/moe.py:moe"
```

The implementation loops over experts and gathers each expert's tokens. Production kernels do the same thing with grouped matmuls, and across GPUs with all-to-all communication (§16.10).

### Load balancing

Left alone, routers collapse: a few experts get most tokens, train faster, get chosen even more, and the rest are wasted. The Switch Transformer adds an auxiliary loss

$$
\mathcal{L}_\text{aux} = E\sum_{e=1}^Ef_e\,P_e,\qquad f_e = \text{fraction of tokens routed to }e,\quad P_e = \text{mean router probability of }e.
$$

!!! theorem "Proposition 20.1 (The balance loss is minimized by uniform routing)"
    If routing matches the router probabilities ($f_e = P_e$, as in expectation for $k = 1$), then $\mathcal{L}_\text{aux} = E\sum_eP_e^2\ge1$, with equality iff $P_e = 1/E$ for all $e$.

??? proof "Proof"
    By Cauchy–Schwarz, $1 = \big(\sum_eP_e\cdot1\big)^2\le\big(\sum_eP_e^2\big)\big(\sum_e1^2\big) = E\sum_eP_e^2$, with equality iff all $P_e$ are equal. $\blacksquare$

    The fractions $f_e$ come from a discrete top-k and aren't differentiable, so the gradient flows through $P_e$: experts that receive many tokens get their probability pushed down.

Other balancing tools: a **capacity factor** (each expert processes at most $c\cdot\frac{kN}{E}$ tokens per batch; overflow tokens skip the layer through the residual connection), and the **router z-loss** to keep router logits small (ST-MoE, [Zoph et al., 2022](https://arxiv.org/abs/2202.08906)).

### Modern MoE design

- **Fine-grained experts** ([DeepSeekMoE, Dai et al., 2024](https://arxiv.org/abs/2401.06066)): many small experts (e.g. 256) with more active per token (e.g. 8), instead of a few large ones. There are far more combinations of experts, which allows more specialization.
- **Shared experts**: one or more experts that *every* token uses, which handle common knowledge so the routed experts can specialize.
- **Auxiliary-loss-free balancing** (DeepSeek-V3): add a per-expert bias $b_e$ to the routing scores *only for choosing* the top-k (not for the mixing weights), and after each step nudge $b_e$ down for overloaded experts and up for underloaded ones. This balances the load without a loss term that conflicts with the language-modeling objective.

| Model | Total params | Active params | Experts (routed, active) |
|---|---|---|---|
| Mixtral 8×7B | 46.7B | 12.9B | 8, top-2 |
| DeepSeek-V3 / R1 | 671B | 37B | 256 + 1 shared, top-8 |
| Qwen3-235B-A22B | 235B | 22B | 128, top-8 |
| gpt-oss-120b | 117B | 5.1B | 128, top-4 |

MoE trade-offs: much better quality per training FLOP and per inference FLOP, but **memory** must hold all experts, serving needs expert parallelism and all-to-all communication, and small-batch decoding reads many different experts' weights. Most frontier models are believed to be MoE models; the open ones listed above certainly are.

## 20.3 Multi-head latent attention (MLA)

GQA shrinks the KV cache by sharing heads. DeepSeek-V2's **MLA** ([DeepSeek-AI, 2024](https://arxiv.org/abs/2405.04434)) compresses instead. For each token, compute a small latent vector

$$
\mathbf{c}^{KV}_t = \mathbf{W}^{DKV}\mathbf{h}_t\in\R^{d_c},\qquad \mathbf{k}_t^{(h)} = \mathbf{W}^{UK}_h\mathbf{c}^{KV}_t,\qquad \mathbf{v}_t^{(h)} = \mathbf{W}^{UV}_h\mathbf{c}^{KV}_t,
$$

and **cache only $\mathbf{c}^{KV}_t$**. At inference, the up-projection for keys can be **absorbed** into the query projection, since $\mathbf{q}^\top\mathbf{k} = \mathbf{q}^\top\mathbf{W}^{UK}_h\mathbf{c} = (\mathbf{W}^{UK\top}_h\mathbf{q})^\top\mathbf{c}$, so attention runs directly against the cached latents. (And $\mathbf{W}^{UV}_h$ can be absorbed into $\mathbf{W}_O$.)

RoPE complicates this, because a position-dependent rotation between $\mathbf{W}^{UK}$ and $\mathbf{q}$ blocks the absorption. MLA therefore uses a **decoupled RoPE key**: a small extra key component (64 dimensions) that carries the rotation and is cached separately, shared across heads.

**KV cache arithmetic** for DeepSeek-V3 (128 heads, $d_h = 128$, $d_c = 512$, RoPE part 64): MLA caches $512+64 = 576$ values per token per layer. Full MHA would cache $2\times128\times128 = 32{,}768$, which is **57× more**. GQA with 8 groups would cache $2\times8\times128 = 2{,}048$. DeepSeek reports MLA matching or beating MHA quality.

## 20.4 Attention patterns for long context

- **Interleaved local and global layers**: most layers use sliding-window attention over the last $w$ tokens (cheap, small cache), and every few layers one uses full attention. Gemma 3 uses 5 local layers (window 1,024) per global layer. gpt-oss alternates dense and banded-window layers.
- **Chunked attention and NoPE layers**: Llama 4 combines chunked local attention with RoPE in most layers and full attention without positional encoding in some layers ("iRoPE"), aiming at very long contexts.
- **Attention sinks as parameters**: gpt-oss adds a learned per-head "sink" logit to the softmax, formalizing the no-op behavior of §12.7.

## 20.5 Beyond attention: state-space and hybrid models

Attention's KV cache grows with context, and its prefill cost grows as $T^2$. Recurrent architectures have a fixed-size state.

- **Linear attention** (§12.8): $\mathbf{S}_t = \mathbf{S}_{t-1}+\phi(\mathbf{k}_t)\mathbf{v}_t^\top$, $\mathbf{y}_t = \mathbf{S}_t^\top\phi(\mathbf{q}_t)$.
- **State-space models (SSMs)**: from control theory, $\mathbf{h}_t = \bar{\mathbf{A}}\mathbf{h}_{t-1}+\bar{\mathbf{B}}x_t$, $y_t = \mathbf{C}\mathbf{h}_t$. With fixed matrices this is a linear time-invariant system that can be computed as a long convolution during training (S4, [Gu et al., 2022](https://arxiv.org/abs/2111.00396)).
- **Mamba** ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)) makes the SSM **selective**: $\bar{\mathbf{B}}_t$, $\mathbf{C}_t$ and the step size $\Delta_t$ depend on the current input, so the model can choose what to remember and what to forget, like an LSTM's gates. Training uses a parallel scan; inference is a constant-cost recurrence. Mamba-2 ([Dao & Gu, 2024](https://arxiv.org/abs/2405.21060)) shows that selective SSMs and linear attention are two views of the same structured matrix family.
- **Gated delta rules** (DeltaNet, [Yang et al., 2024](https://arxiv.org/abs/2406.06484)): update the linear-attention state with an error-correcting rule $\mathbf{S}_t = \mathbf{S}_{t-1}(\mathbf{I}-\beta_t\mathbf{k}_t\mathbf{k}_t^\top)+\beta_t\mathbf{k}_t\mathbf{v}_t^\top$ (plus a forget gate), which overwrites old associations instead of only adding.

Pure recurrent models struggle with exact recall from long contexts (copying, retrieving a fact from 50K tokens back), because a fixed-size state must compress everything. **Hybrids** keep a few full-attention layers among many recurrent ones and get most of both: Jamba (Mamba + attention + MoE), and Qwen3-Next and other 2025 models (gated DeltaNet + attention, roughly 3:1).

## 20.6 Multi-token prediction

DeepSeek-V3 trains with an extra **multi-token prediction (MTP)** module ([Gloeckle et al., 2024](https://arxiv.org/abs/2404.19737) proposed the idea): besides predicting $x_{t+1}$, a small additional block predicts $x_{t+2}$ from the main model's hidden state and the embedding of $x_{t+1}$. The extra loss densifies the training signal and encourages representations that plan ahead. At inference the MTP module can be dropped, or used as the draft model for speculative decoding (§19.6), with reported acceptance rates of 85–90% for the second token.

## 20.7 Multimodal models

Text models become multimodal by turning other inputs into sequences of vectors in the same residual stream:

- **Images**: a vision encoder (a Vision Transformer, ViT, [Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)) splits the image into patches (e.g. 14×14 pixels), embeds each patch as a vector, and runs Transformer layers over them. A small **projector** maps these patch vectors into the LLM's embedding space, and they're inserted into the token sequence like word embeddings (LLaVA, [Liu et al., 2023](https://arxiv.org/abs/2304.08485)). Training starts by aligning the projector, then fine-tunes on image-text instruction data.
- **Early fusion**: images are tokenized into discrete codes from the start and the model is trained on interleaved text and image tokens (Chameleon; Llama 4 uses early fusion).
- **Audio**: speech encoders (e.g. Whisper's) produce frame embeddings; audio tokenizers (neural codecs) allow speech *output*.

An image typically costs hundreds to thousands of tokens, which is why image inputs are billed as tokens and count against the context window.

!!! llm "What about Claude, GPT and Gemini?"
    Frontier labs don't publish architectures. Public information is limited to what model cards and documentation state: context window sizes, modalities, and capabilities such as tool use and extended thinking. It's widely believed that frontier models use MoE and the techniques in this chapter, but specific claims about any closed model's internals are speculation. What you *can* rely on is that everything they do is built from the components of Parts III–VI, trained with the methods of Ch. 15–17 and 21–22.

## Key takeaways

- Llama recipe = GPT-2 + RMSNorm + RoPE + SwiGLU + GQA − biases, with a bigger vocabulary and far more data.
- MoE decouples total parameters from active parameters. Top-k routing, load balancing ($E\sum f_eP_e\ge1$, equality at uniform), fine-grained and shared experts, auxiliary-loss-free bias balancing.
- MLA caches a compressed latent (576 values/token/layer in DeepSeek-V3, 57× less than MHA) and absorbs the up-projections into the query and output projections.
- Local/global interleaving and sliding windows cut long-context cost. SSMs (Mamba) and gated linear attention give constant-size state; hybrids keep a few attention layers for exact recall.
- MTP adds a lookahead training signal and a built-in speculative drafter. Multimodal models project patches or audio frames into the token stream.

## Exercises

1. Count the parameters of Mistral 7B from the configuration table (no biases, untied embeddings, RMSNorm).
2. For Mixtral 8×7B ($L = 32$, $d = 4096$, expert FFN hidden 14,336, 8 experts, top-2, GQA 32/8, $V = 32{,}000$), verify the total (46.7B) and active (12.9B) parameter counts.
3. Prove that with top-$k$ routing and renormalized weights, the MoE output is a convex combination of the chosen experts' outputs.
4. Compute the KV cache per token (all layers) for DeepSeek-V3 with MLA ($L = 61$, 576 values per layer, bf16), and compare it with Llama 3 405B with GQA.
5. Show that absorbing $\mathbf{W}^{UK}_h$ into the query makes the score computation cost $O(d_c)$ per (query, key) pair instead of $O(d_h)$ after reconstructing keys. When is that a win?
6. Why do pure SSMs struggle with tasks like "repeat the 3rd sentence of this 20-page document", and why does adding even a few attention layers fix it?

??? success "Solutions"
    1. Per layer: attention $2\cdot4096^2+2\cdot4096\cdot1024 = 41.9$M; FFN $3\cdot4096\cdot14336 = 176.2$M; norms 8K. ×32 = 6.98B. Embeddings + output: $2\cdot32000\cdot4096 = 262$M. Total ≈ 7.24B.
    2. Experts per layer: $8\cdot176.2$M $= 1.41$B; ×32 = 45.1B. Attention ×32 = 1.34B. Embeddings 0.26B. Total ≈ 46.7B. Active: $2\cdot176.2$M$\cdot32 = 11.3$B + 1.34B + 0.26B ≈ 12.9B.
    3. The weights $p_e/\sum_{e'\in\mathcal{T}}p_{e'}$ are positive and sum to 1 over $\mathcal{T}$.
    4. MLA: $61\cdot576\cdot2 = 70{,}272$ bytes ≈ 69 KB per token. Llama 3 405B: $2\cdot126\cdot8\cdot128\cdot2 = 516{,}096$ bytes ≈ 504 KB per token, over 7× more, for a model with fewer total parameters.
    5. Reconstructing all keys costs $O(d_cd_h)$ per cached token per head; absorption replaces it with a one-off $O(d_cd_h)$ per query and $O(d_c)$ per pair. It wins at long contexts where cached tokens far outnumber queries, i.e. decoding.
    6. A fixed-size state must compress 20 pages into a few MB of numbers, and exact text can't survive arbitrary compression. An attention layer can look up any cached position exactly, so a few such layers provide exact retrieval while the recurrent layers do cheap local processing.

## References

- H. Touvron et al., [LLaMA](https://arxiv.org/abs/2302.13971) (2023) and [Llama 2](https://arxiv.org/abs/2307.09288) (2023); Llama Team, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) (2024).
- A. Jiang et al., [Mistral 7B](https://arxiv.org/abs/2310.06825) (2023) and [Mixtral of Experts](https://arxiv.org/abs/2401.04088) (2024).
- W. Fedus, B. Zoph, N. Shazeer, [Switch Transformers](https://arxiv.org/abs/2101.03961) (JMLR 2022).
- DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434) (2024) and [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) (2024).
- A. Gu & T. Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (2023).
- Gemma Team, [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786) (2025).
- OpenAI, [gpt-oss-120b & gpt-oss-20b Model Card](https://arxiv.org/abs/2508.10925) (2025).
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) (2025).
