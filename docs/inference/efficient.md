# 19. Efficient Inference

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 12](../transformer/attention.md), [Ch. 16](../scale/distributed.md) (roofline, number formats), [Ch. 18](decoding.md). **You will learn:** why prefill is compute-bound and decoding is memory-bound, and the latency and throughput metrics that follow; the KV cache, its exact memory cost, and the techniques that shrink it; FlashAttention, with a full derivation and proof of the online softmax; continuous batching, PagedAttention and prefix caching; quantization (absmax, zero-point, group-wise, outliers, GPTQ/AWQ/SmoothQuant) with error bounds; speculative decoding, with a proof that it samples from exactly the target model's distribution; and the serving stacks used in practice.
    **Why it matters:** inference is where LLM money is spent once a model is deployed, and where latency is felt by users. Every technique here is in production at every major provider, and several (prompt caching, batching, quantized models) directly shape how you should build applications.

!!! watch "Read & watch"
    - T. Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) (2022), one of the most influential systems papers of the LLM era.
    - W. Kwon et al., [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM, 2023).
    - Google DeepMind, [How to Scale Your Model: Transformer inference](https://jax-ml.github.io/scaling-book/inference/).
    - L. Weng, [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/).

---

## 19.1 Prefill and decode are different workloads

**Prefill** processes the $T_p$ prompt tokens in one parallel pass. Every weight matrix multiplies a $T_p\times d$ activation matrix: arithmetic intensity is about $T_p$ FLOPs per byte (Exercise 16.3), so for prompts of hundreds of tokens or more it's **compute-bound**.

**Decode** produces one token per step per sequence. With batch size $B$, each step reads *all* the weights from HBM to do only $2B$ FLOPs per weight: intensity $\approx B$. For small $B$ it's heavily **memory-bandwidth-bound**, and the time per step is about

$$
t_\text{step}\approx\frac{\text{bytes of weights}+\text{bytes of KV cache read}}{\text{HBM bandwidth}}.
$$

Llama 3 8B in bf16 is 16 GB: on an H100 (3.35 TB/s) one decode step takes at least $16/3350\approx4.8$ ms, so at most about 200 tokens/s for a single sequence, *however fast the arithmetic units are*. Batching 64 sequences costs almost the same time per step (still memory-bound) but produces 64 tokens. This is the basic economics of LLM serving: **batch to amortize the weight reads**.

Metrics:

| Metric | Meaning | Dominated by |
|---|---|---|
| **TTFT** (time to first token) | latency until output starts | prefill (prompt length), queueing |
| **TPOT / ITL** (time per output token, inter-token latency) | streaming speed | decode: memory bandwidth, batch size |
| **Throughput** | total tokens/s across all users | batch size, efficiency |

Latency and throughput trade off: bigger batches give more throughput but slower tokens per user.

## 19.2 The KV cache

At decode step $t$, the new token's query must attend to the keys and values of all positions $1..t$. By causality (Prop. 12.2), the keys and values of earlier positions **never change** when later tokens are added. So compute them once and keep them: the **KV cache**. Each decode step then computes Q, K, V only for the newest token, appends K and V to the cache, and attends over the cache. Without the cache, step $t$ would recompute all $t$ positions, $O(T^2)$ total work instead of $O(T)$.

Our Llama-style attention keeps the cache, and `generate` runs one prefill and then single-token decode steps. The tests check that the output is **identical** to recomputing everything from scratch at every step:

```python
--8<-- "code/llama.py:gqa"
```

Note the mask `tril(diagonal=S - T)`: during decode the query is the *last* of $S$ positions, so it may see all $S$ keys.

!!! theorem "Proposition 19.1 (KV cache size)"
    $$
    \text{KV bytes} = 2\times L\times H_{kv}\times d_h\times\text{bytes per value}\times T\times B,
    $$

    (2 for keys and values) for $L$ layers, $H_{kv}$ key/value heads of size $d_h$, context $T$ and batch $B$.

For Llama 3 70B ($L = 80$, $H_{kv} = 8$, $d_h = 128$, bf16): $2\cdot80\cdot8\cdot128\cdot2 = 327{,}680$ bytes ≈ **320 KB per token**. A single 128K-token context needs 40 GB of KV cache, half an H100. With full multi-head attention ($H_{kv} = 64$) it would be 8× more. **The KV cache, not the weights, often limits batch size and context length.**

<div class="widget" data-widget="kv-cache"></div>

### Shrinking the KV cache

- **Fewer KV heads.** Multi-query attention (MQA, [Shazeer, 2019](https://arxiv.org/abs/1911.02150)) shares one K/V head among all query heads; grouped-query attention (GQA, [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)) shares each K/V head among a group (Llama 3 70B: 64 query heads, 8 KV heads, 8× smaller). Quality is close to full MHA.
- **Latent compression.** DeepSeek's multi-head latent attention (MLA) caches one small latent vector per token and layer, from which keys and values are reconstructed (Ch. 20).
- **Local attention.** Sliding-window layers cache only the last $w$ tokens (Mistral, Gemma 2/3 interleave local and global layers).
- **KV quantization** to fp8 or int4 halves or quarters the cache.
- **Eviction**: keep the attention-sink tokens plus a recent window ([StreamingLLM](https://arxiv.org/abs/2309.17453)), or the tokens that received the most attention ([H2O](https://arxiv.org/abs/2306.14048)). Lossy, but bounded memory.

## 19.3 FlashAttention

Standard attention writes the $T\times T$ score matrix $\mathbf{S}$ to HBM, reads it back for the softmax, writes $\mathbf{A}$, and reads it again to multiply by $\mathbf{V}$. For long sequences this memory traffic, not the FLOPs, dominates the runtime. A GPU has a small amount of very fast on-chip SRAM (about 228 KB per streaming multiprocessor on an H100, tens of MB in total) next to the large, slower HBM.

**FlashAttention** ([Dao et al., 2022](https://arxiv.org/abs/2205.14135)) computes exact attention **tile by tile in SRAM**, never writing $\mathbf{S}$ or $\mathbf{A}$ to HBM. Load a block of queries; stream blocks of keys and values past it; keep a running result. The obstacle is the softmax: its denominator needs *all* scores in a row before any output can be normalized. The solution is the **online softmax**.

### Online softmax

For one query row, process the scores in blocks. Keep three running quantities: the max so far $m$, the sum of exponentials relative to that max $\ell$, and the unnormalized output $\mathbf{o}$. When a new block of scores $\mathbf{s}^{(j)}$ with values $\mathbf{V}^{(j)}$ arrives:

$$
\begin{aligned}
m' &= \max\big(m,\ \max_i s^{(j)}_i\big)\\
\ell' &= e^{m-m'}\,\ell + \textstyle\sum_ie^{s^{(j)}_i-m'}\\
\mathbf{o}' &= e^{m-m'}\,\mathbf{o} + \textstyle\sum_ie^{s^{(j)}_i-m'}\,\mathbf{v}^{(j)}_i
\end{aligned}
$$

and at the end, output $\mathbf{o}/\ell$.

!!! theorem "Theorem 19.2 (The online softmax is exact)"
    After processing blocks $1..j$, the state satisfies $\ell = \sum_{i\in\text{seen}}e^{s_i-m}$ and $\mathbf{o} = \sum_{i\in\text{seen}}e^{s_i-m}\mathbf{v}_i$ with $m = \max_{i\in\text{seen}}s_i$. Hence the final $\mathbf{o}/\ell$ equals $\sum_i\softmax(\mathbf{s})_i\mathbf{v}_i$ exactly.

??? proof "Proof"
    By induction over blocks. Initially $m = -\infty$, $\ell = 0$, $\mathbf{o} = \mathbf{0}$. Assume the invariant before a block. Multiplying by $e^{m-m'}$ re-expresses the old sums relative to the new max: $e^{m-m'}\sum_\text{old}e^{s_i-m} = \sum_\text{old}e^{s_i-m'}$, and likewise for $\mathbf{o}$. Adding the new block's terms (already relative to $m'$) gives the invariant for the enlarged set. At the end, $\frac{\mathbf{o}}{\ell} = \frac{\sum_ie^{s_i-m}\mathbf{v}_i}{\sum_ie^{s_i-m}} = \sum_i\frac{e^{s_i}}{\sum_ke^{s_k}}\mathbf{v}_i$, since the factor $e^{-m}$ cancels. Subtracting the running max also keeps every exponent $\le0$, so nothing overflows. $\blacksquare$

The same merge rule combines partial results computed on *different GPUs*, which is how ring attention (§16.8) and flash-decoding work. Here it is in NumPy, tested against the naive computation:

```python
--8<-- "code/attention.py:online"
```

!!! theorem "Fact 19.3 (IO complexity, Dao et al. 2022, Thm 2)"
    With sequence length $T$, head dimension $d$ and SRAM size $M$ (with $d\le M\le Td$), standard attention needs $\Theta(Td+T^2)$ HBM accesses, while FlashAttention needs $\Theta(T^2d^2/M)$. For typical $d = 64$–$128$ and $M\approx100$ KB this is many times fewer. Memory use drops from $O(T^2)$ to $O(T)$.

**Backward pass.** FlashAttention saves only the output and the per-row statistics $(m,\ell)$, and **recomputes** the score tiles during the backward pass. That's more FLOPs but much less memory traffic, so it's faster overall. FlashAttention-2 ([Dao, 2023](https://arxiv.org/abs/2307.08691)) improves parallelism across the sequence; FlashAttention-3 ([Shah et al., 2024](https://arxiv.org/abs/2407.08608)) uses Hopper GPUs' asynchronous units and fp8. **Flash-decoding** splits the KV cache across thread blocks during decoding (when there's only one query) and merges the partial results with the same online-softmax rule. In PyTorch you get these kernels through `F.scaled_dot_product_attention`.

## 19.4 Batching and memory management

- **Static batching** waits until all sequences in a batch finish, so short responses sit idle while long ones complete.
- **Continuous (in-flight) batching** ([Yu et al., Orca, 2022](https://www.usenix.org/conference/osdi22/presentation/yu)) schedules at the level of single iterations: as soon as a sequence finishes, a waiting request takes its slot in the next decode step. Throughput gains are often several times.
- **PagedAttention** ([Kwon et al., vLLM, 2023](https://arxiv.org/abs/2309.06180)) stores each sequence's KV cache in fixed-size **blocks** (e.g. 16 tokens) allocated on demand, with a block table per sequence, exactly like virtual-memory pages in an operating system. This ends the fragmentation from pre-allocating the maximum length, and lets sequences **share** blocks.
- **Prefix caching**: requests that start with the same tokens (a long system prompt, the same document, earlier turns of a conversation) can reuse the KV blocks computed for that prefix instead of re-running prefill. SGLang's RadixAttention ([Zheng et al., 2023](https://arxiv.org/abs/2312.07104)) organizes cached prefixes in a radix tree.
- **Chunked prefill** splits a long prompt's prefill into chunks interleaved with other requests' decode steps, so one long prompt doesn't stall everyone's streaming.
- **Disaggregated serving** runs prefill and decode on separate GPU pools, since one is compute-bound and the other memory-bound ([DistServe](https://arxiv.org/abs/2401.09670), [Splitwise](https://arxiv.org/abs/2311.18677)).

!!! llm "Prompt caching in APIs is prefix caching"
    When an API offers **prompt caching** (e.g. Claude's `cache_control`, Ch. 28), it's selling you this mechanism: if your request starts with exactly the same tokens as a recent request, the provider reuses the stored KV cache, so you pay much less for those input tokens and the time to first token drops. The design consequence: **put stable content (instructions, tool definitions, documents) first and variable content last**, and keep the prefix byte-for-byte identical across calls.

## 19.5 Quantization

Decoding is memory-bound, so **storing weights in fewer bits makes decoding faster** almost proportionally, as well as letting bigger models fit. Quantization maps floats to a small set of integers.

```python
--8<-- "code/quant.py:quant"
```

!!! theorem "Proposition 19.4 (Round-to-nearest error)"
    With symmetric absmax quantization, scale $s = \max|w|/q_\max$ and $\hat w = s\cdot\operatorname{round}(w/s)$, every weight satisfies $|w-\hat w|\le s/2$.

??? proof "Proof"
    Since $|w/s|\le q_\max$, clamping never changes anything, and rounding a real number to the nearest integer moves it by at most $\frac12$. Multiply by $s$. $\blacksquare$

The error bound is proportional to $s$, which is set by the **largest** weight in the group. One outlier inflates $s$ for everyone else, and most weights then round to just a few levels. That's the central difficulty of LLM quantization:

- **Group-wise scales.** Use one scale per small group (e.g. 64 or 128 consecutive weights) instead of per tensor or per row, so an outlier only affects its own group. The tests verify that group-wise scales beat per-row scales on a matrix with one outlier. Standard for 4-bit formats (GPTQ, AWQ, GGUF's `Q4_K`, QLoRA's NF4).
- **Activation outliers.** Above about 6.7B parameters, a few feature dimensions of the activations develop huge magnitudes (up to 100× the rest), in every layer ([Dettmers et al., LLM.int8(), 2022](https://arxiv.org/abs/2208.07339)). Quantizing activations naively destroys these models. LLM.int8() keeps the outlier dimensions in fp16. **SmoothQuant** ([Xiao et al., 2023](https://arxiv.org/abs/2211.10438)) instead moves scale from activations into weights using the identity $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top = (\mathbf{X}\diag(\mathbf{s})^{-1})(\mathbf{W}\diag(\mathbf{s}))^\top$, which makes both easier to quantize.
- **Error-aware rounding.** **GPTQ** ([Frantar et al., 2023](https://arxiv.org/abs/2210.17323)) quantizes one column at a time and updates the not-yet-quantized columns to compensate for the error, using second-order (Hessian) information from calibration data. **AWQ** ([Lin et al., 2024](https://arxiv.org/abs/2306.00978)) protects the ~1% of weight channels that see the largest activations by scaling them up before quantization.
- **Formats.** Weight-only int8 and int4 (for memory-bound decode); fp8 weights *and* activations (W8A8) on H100-class GPUs, which also speeds up compute-bound prefill; KV-cache fp8. FP4 formats on Blackwell. **NF4** ([Dettmers et al., QLoRA, 2023](https://arxiv.org/abs/2305.14314)) places its 16 levels at quantiles of a normal distribution, matching the distribution of the weights (Ch. 23).

Rules of thumb: 8-bit is nearly lossless; good 4-bit methods lose a little (more for small models and for reasoning-heavy tasks); below 4 bits quality drops fast. **Quantization-aware training** (simulating quantization in the forward pass with a straight-through estimator, §7.8) recovers most of the loss when you control training.

## 19.6 Speculative decoding

Decoding one token at a time leaves the GPU underused (memory-bound). **Speculative decoding** ([Leviathan et al., 2023](https://arxiv.org/abs/2211.17192); [Chen et al., 2023](https://arxiv.org/abs/2302.01318)) uses a small, fast **draft model** $q$ to propose $\gamma$ tokens, then runs the large **target model** $p$ **once** on all of them in parallel (that's like a short prefill, nearly the cost of a single decode step), and accepts or rejects each proposal so that the result is distributed **exactly** as if sampled from $p$.

!!! definition "Algorithm 19.1 (Speculative sampling, one step)"
    1. Draft: sample $\tilde x_1,\dots,\tilde x_\gamma$ autoregressively from $q$.
    2. Verify: run $p$ on the prefix plus all drafts, obtaining $p(\cdot\mid\cdot)$ at each of the $\gamma+1$ positions.
    3. For $i = 1..\gamma$: accept $\tilde x_i$ with probability $\min\!\big(1,\ p(\tilde x_i)/q(\tilde x_i)\big)$. At the first rejection, sample a replacement from the **residual distribution** $r(x)\propto\max\big(0,\ p(x)-q(x)\big)$ and stop.
    4. If all $\gamma$ are accepted, sample one bonus token from $p$ at position $\gamma+1$.

!!! theorem "Theorem 19.5 (Speculative sampling is exact)"
    Each token produced by Algorithm 19.1 is distributed exactly according to the target distribution $p$.

??? proof "Proof"
    Consider one position with draft distribution $q$ and target $p$. The token $x$ is output either because the draft proposed $x$ and it was accepted, or because a draft was rejected and $x$ was sampled from the residual:

    $$
    \Pr[\text{out} = x] = q(x)\min\Big(1,\frac{p(x)}{q(x)}\Big) + \Pr[\text{reject}]\cdot\frac{\max(0,p(x)-q(x))}{Z},\qquad Z = \sum_y\max(0,p(y)-q(y)).
    $$

    The first term is $\min(q(x),p(x))$. The rejection probability is $1-\sum_y\min(p(y),q(y))$. Since $\sum_y p(y) = \sum_y\min(p,q)+\sum_y\max(0,p-q)$, it equals $Z$. So the second term is $\max(0,p(x)-q(x))$, and the total is $\min(p,q)+\max(0,p-q) = p(x)$. Later positions condition on accepted (correctly distributed) prefixes, so the argument applies position by position. $\blacksquare$

!!! theorem "Proposition 19.6 (Expected speed-up)"
    If each draft token is accepted independently with probability $\alpha$, the expected number of tokens produced per target-model call is

    $$
    \frac{1-\alpha^{\gamma+1}}{1-\alpha}.
    $$

??? proof "Proof"
    The step produces (number of accepted drafts) + 1 tokens: either the residual sample after a rejection, or the bonus token. At least $k$ drafts are accepted with probability $\alpha^k$, so the expected count is $\sum_{k=0}^{\gamma}\alpha^k = \frac{1-\alpha^{\gamma+1}}{1-\alpha}$. $\blacksquare$

With $\alpha = 0.8$ and $\gamma = 4$: $3.36$ tokens per target call. Real speed-ups are typically 2–3×, since the draft isn't free. Variants avoid a separate draft model: **Medusa** ([Cai et al., 2024](https://arxiv.org/abs/2401.10774)) adds extra prediction heads to the target model; **EAGLE** ([Li et al., 2024](https://arxiv.org/abs/2401.15077)) drafts at the feature level; DeepSeek-V3 trains **multi-token prediction** modules that double as drafters; **prompt-lookup decoding** drafts by copying n-grams from the prompt, which works very well for editing and RAG answers that quote their sources.

## 19.7 Serving stacks

| Stack | Strengths |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | PagedAttention, continuous batching, broad model support; the default open-source server |
| [SGLang](https://github.com/sgl-project/sglang) | RadixAttention prefix caching, fast structured output, strong for agent workloads |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA-optimized kernels, fp8/fp4 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / Ollama | CPU and laptop inference, GGUF quantized formats |
| Apple MLX | Apple Silicon unified memory |

## Key takeaways

- Prefill is compute-bound; decode is memory-bound, and time per step ≈ bytes read / bandwidth. Batching amortizes weight reads.
- The KV cache stores past keys and values: $2LH_{kv}d_h\cdot\text{bytes}$ per token (320 KB for Llama 3 70B). GQA, MLA, windows, quantization and eviction shrink it.
- FlashAttention tiles attention in SRAM using the exact online softmax (running max, rescaled sum), so it never stores the $T\times T$ matrix.
- Continuous batching, PagedAttention and prefix caching drive serving efficiency. API prompt caching is prefix caching: keep stable content first.
- Quantization error scales with the group's largest value; outliers are the core problem. Group-wise scales, SmoothQuant, GPTQ and AWQ address it.
- Speculative decoding is exact (accept with $\min(1,p/q)$, resample from $\max(0,p-q)$) and gives $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ tokens per target call.

## Exercises

1. Estimate the maximum single-sequence decode speed of a 70B model in bf16, in int4, on 4 H100s with tensor parallelism (ignoring the KV cache and communication).
2. Compute the KV cache per token for Llama 3 8B ($L = 32$, $H_{kv} = 8$, $d_h = 128$, bf16). How many 8K-token sequences fit in 40 GB?
3. Show that the online softmax state for two disjoint sets of scores, $(m_1,\ell_1,\mathbf{o}_1)$ and $(m_2,\ell_2,\mathbf{o}_2)$, can be merged into the state for their union. Write the formula.
4. For weights uniform on $[-1,1]$ plus one outlier at 20, compare the int8 absmax step size with and without the outlier.
5. Prove that if $q = p$, speculative decoding accepts every draft. What is the expected speed-up then?
6. Why does prefix caching need the prefix to match *exactly*, token for token?

??? success "Solutions"
    1. 140 GB across 4 GPUs' bandwidth ($4\times3.35 = 13.4$ TB/s) gives about 10.4 ms per token, ~96 tokens/s. In int4 (35 GB): ~2.6 ms, ~380 tokens/s (in practice less: kernels, communication).
    2. $2\cdot32\cdot8\cdot128\cdot2 = 131{,}072$ bytes = 128 KB per token. An 8K sequence takes 1 GB, so 40 sequences.
    3. $m = \max(m_1,m_2)$, $\ell = e^{m_1-m}\ell_1+e^{m_2-m}\ell_2$, $\mathbf{o} = e^{m_1-m}\mathbf{o}_1+e^{m_2-m}\mathbf{o}_2$. (This is how split-KV decoding and ring attention merge partial results.)
    4. Without the outlier $s = 1/127\approx0.0079$; with it $s = 20/127\approx0.157$, 20× coarser: the uniform weights now use only about 13 of the 255 levels.
    5. $\min(1,p/q) = 1$ always, so all $\gamma$ drafts are accepted plus a bonus: $\gamma+1$ tokens per call ($\alpha = 1$ in Prop. 19.6, taking the limit).
    6. Each cached key and value depends on the entire prefix before it (through all layers of attention). Change one earlier token and every later cached K/V is different.

## References

- T. Dao, D. Fu, S. Ermon, A. Rudra, C. Ré, [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (NeurIPS 2022); T. Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691) (2023); J. Shah et al., [FlashAttention-3](https://arxiv.org/abs/2407.08608) (2024).
- M. Milakov & N. Gimelshein, [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867) (2018).
- W. Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (SOSP 2023).
- J. Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) (EMNLP 2023).
- T. Dettmers et al., [LLM.int8()](https://arxiv.org/abs/2208.07339) (NeurIPS 2022); E. Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323) (ICLR 2023); J. Lin et al., [AWQ](https://arxiv.org/abs/2306.00978) (MLSys 2024); G. Xiao et al., [SmoothQuant](https://arxiv.org/abs/2211.10438) (ICML 2023).
- Y. Leviathan, M. Kalman, Y. Matias, [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (ICML 2023); C. Chen et al., [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) (2023).
