# 14. GPT from Scratch

!!! abstract "In this chapter"
    **Prerequisites:** Chapters 9–13. **You will learn:** how every piece built so far fits into one complete, ~200-line GPT-2 implementation; the shape of every tensor on the forward pass; initialization and the sanity checks that catch most bugs; a working training loop with AdamW, clipping and sampling; how to load OpenAI's pretrained GPT-2 weights into your own code; and what it takes to reproduce GPT-2 (124M) from scratch on a modern GPU node: gradient accumulation, mixed precision, `torch.compile`, FlashAttention, and the hyperparameters.
    **Why it matters:** this is the moment the theory turns into a working language model. Every frontier LLM is a scaled-up, modernized version of this file (Ch. 20 shows exactly which lines change).

!!! watch "Watch (two essential videos)"
    Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) (2 h), then [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) (4 h), which takes the same model to a real training run with every performance trick.

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/l8pRSuU81PU" allowfullscreen loading="lazy"></iframe></div>

    Code: [nanoGPT](https://github.com/karpathy/nanoGPT), [build-nanogpt](https://github.com/karpathy/build-nanogpt), [llm.c](https://github.com/karpathy/llm.c), [nanochat](https://github.com/karpathy/nanochat). Our [`code/gpt.py`](https://github.com/kdellis135/llm_deep_dive/blob/main/code/gpt.py) follows nanoGPT's structure and naming so you can read both side by side.

---

## 14.1 The forward pass, tensor by tensor

For GPT-2 small ($V = 50257$, $d = C = 768$, $L = 12$, $H = 12$, $d_h = 64$) on a batch of $B$ sequences of length $T$:

| Step | Operation | Output shape |
|---|---|---|
| 1 | token ids `idx` | `(B, T)` integers |
| 2 | token embedding `wte(idx)` | `(B, T, 768)` |
| 3 | + position embedding `wpe(arange(T))` (broadcast over B) | `(B, T, 768)` |
| 4 | **block ×12**: `x = x + attn(ln_1(x))` | `(B, T, 768)` |
| 4a | inside attn: `c_attn(x)` then split | 3 × `(B, T, 768)` |
| 4b | reshape to heads | `(B, 12, T, 64)` each |
| 4c | scores $\mathbf{Q}\mathbf{K}^\top/8$ + mask, softmax | `(B, 12, T, T)` |
| 4d | weights @ $\mathbf{V}$, merge heads, `c_proj` | `(B, T, 768)` |
| 4e | `x = x + mlp(ln_2(x))`: `c_fc` → GELU → `c_proj` | `(B, T, 3072)` → `(B, T, 768)` |
| 5 | final `ln_f` | `(B, T, 768)` |
| 6 | `lm_head` (tied to `wte`) | `(B, T, 50257)` logits |
| 7 | cross-entropy vs targets (shifted by one) | scalar |

## 14.2 The code

The configuration:

```python
--8<-- "code/gpt.py:config"
```

Causal self-attention. It's the multi-head attention of §12.6, using PyTorch's fused `scaled_dot_product_attention`, which runs a FlashAttention kernel when one is available (Ch. 19):

```python
--8<-- "code/gpt.py:attn"
```

The MLP and the block are as in Ch. 13:

```python
--8<-- "code/gpt.py:mlp"
```

```python
--8<-- "code/gpt.py:block"
```

The model: embeddings, blocks, final norm, tied output head, GPT-2 initialization, the loss, and generation.

```python
--8<-- "code/gpt.py:gpt"
```

Details worth noticing:

- **Weight tying** happens by assigning the *same* `Parameter` object to `lm_head.weight` and `wte.weight`. PyTorch then accumulates both uses' gradients into one tensor (branches accumulate, §7.2).
- **`_init_weights`**: $\mathcal{N}(0,0.02^2)$ everywhere, zeros for biases, and $0.02/\sqrt{2L}$ for the two residual output projections per block (Prop. 8.3).
- **The loss** flattens `(B, T, V)` logits to `(B·T, V)` and targets to `(B·T,)`: every position is one classification example (§9.9).
- **`generate`** crops the context to the last `block_size` tokens, uses only the logits at the last position, applies temperature and optional top-k, and samples (Ch. 18). It reruns the full context each step. The KV cache in Ch. 19–20 removes that waste.

## 14.3 Sanity checks before training

The test suite runs each of these, and you should run them on any new model:

1. **Parameter count.** `GPT(GPTConfig())` has 124,439,808 parameters (§13.3). A wrong count means a wrong architecture.
2. **Output shape.** Logits are `(B, T, V)`.
3. **Initial loss ≈ $\ln V$.** At initialization the logits are small, the prediction is close to uniform, and the loss should be $\approx\ln V$ (10.82 for GPT-2's vocabulary; the test uses $V = 50$, so $\ln50\approx3.91$). Much higher means overconfident initial logits (§8.2).
4. **Causality.** Changing a later token must not change earlier logits (Prop. 12.2).
5. **Overfit a tiny batch.** The loss should go to near zero. If it can't, there's a bug in the model or the loop.

## 14.4 The training loop

```python
--8<-- "code/gpt.py:train"
```

This trains a 2-layer, 64-wide model on a few lines of Shakespeare, character by character. The structure is that of every LLM training script:

1. **Sample a batch** of random windows; the targets are the inputs shifted by one.
2. **Forward** to get the loss.
3. **`zero_grad`** (set to `None`, which is cheaper than filling with zeros), then **`backward`**.
4. **Clip** the global gradient norm to 1.0 (§5.8).
5. **`step`**: AdamW with $\beta = (0.9, 0.95)$ and weight decay 0.1 (§5.6).

It starts at $\ln V\approx3.6$ and, in 300 steps, the loss falls below 1.0 (checked in the tests). It memorizes this tiny text; with real data you'd track a held-out validation loss.

!!! lab "Lab: Shakespeare"
    Download [tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (1 MB), set `TEXT` to its contents, use `n_layer=6, n_head=6, n_embd=384, block_size=256, dropout=0.2`, hold out 10% for validation, and train for 5,000 steps on a GPU (or a smaller model on a laptop). nanoGPT reaches a validation loss of about **1.47** and generates convincingly Shakespeare-shaped text.

## 14.5 Loading OpenAI's pretrained GPT-2 weights

Since our module names match GPT-2's checkpoint names, we can load the real 124M model from Hugging Face. One twist: OpenAI's checkpoint uses a `Conv1D` layer that stores weights **transposed** relative to `nn.Linear`, so four kinds of matrices must be transposed (this is nanoGPT's `from_pretrained`):

```python
from transformers import GPT2LMHeadModel       # pip install transformers
from gpt import GPT, GPTConfig
import torch

def load_gpt2(name="gpt2"):
    hf = GPT2LMHeadModel.from_pretrained(name).state_dict()
    model = GPT(GPTConfig())                    # 124M config: matches "gpt2"
    sd = model.state_dict()
    transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
    for k in sd:
        if k.endswith("attn.bias") or k not in hf:   # skip masks and the tied lm_head alias
            continue
        w = hf[k]
        with torch.no_grad():
            sd[k].copy_(w.t() if k.endswith(transposed) else w)
    return model
```

With the weights loaded, `model.generate` on the GPT-2 tokenizer's encoding of "Hello, I'm a language model," produces fluent (if rambling) English. (This snippet needs network access and `transformers`, so it isn't part of the offline test suite.)

## 14.6 Reproducing GPT-2 (124M) for real

Karpathy's [build-nanogpt](https://github.com/karpathy/build-nanogpt) trains this exact architecture from scratch and beats OpenAI's released GPT-2 124M on the HellaSwag benchmark. The recipe:

| Setting | Value | Why |
|---|---|---|
| Data | FineWeb-Edu, 10B-token sample | high-quality educational web text (Ch. 15) |
| Tokens per step | $2^{19} = 524{,}288$ (≈0.5M, GPT-3's choice for this size) | via **gradient accumulation** |
| Micro-batch | $B = 64$, $T = 1024$ per GPU | as large as memory allows |
| Optimizer | AdamW, $\beta = (0.9, 0.95)$, $\epsilon = 10^{-8}$, wd 0.1 on 2-D weights only | §5.6 |
| LR | peak $6\times10^{-4}$, linear warmup 715 steps, cosine to 10% | §5.7 |
| Steps | 19,073 (≈ 10B tokens, 1 epoch) | |
| Grad clip | 1.0 | |
| Result | val loss ≈ 3.0–3.3; HellaSwag ≈ 30% (GPT-2 124M: 29.4%) | |

**Gradient accumulation** gets a 0.5M-token batch from micro-batches that fit in memory. Gradients add up across `backward()` calls (`+=` again), so divide each micro-batch's loss by the number of micro-steps to get the *mean*:

```python
grad_accum_steps = total_batch_tokens // (B * T * world_size)   # e.g. 524288 // (64*1024*8) = 1
for step in range(max_steps):
    optimizer.zero_grad(set_to_none=True)
    for micro in range(grad_accum_steps):
        x, y = loader.next_batch()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):   # mixed precision (Ch. 16)
            _, loss = model(x, y)
        (loss / grad_accum_steps).backward()        # accumulate the MEAN gradient
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step, 6e-4, 715, max_steps)  # warmup + cosine (Ch. 5)
    optimizer.step()
```

**Performance tricks**, each explained in Karpathy's video and in Ch. 16 and 19, in order of impact on an A100/H100:

1. **TF32 matmuls** (`torch.set_float32_matmul_precision("high")`) and **bf16 autocast**: run on Tensor Cores at 8–16× fp32 throughput.
2. **`torch.compile(model)`**: fuses elementwise ops into fewer kernels, cutting memory traffic.
3. **FlashAttention** via `F.scaled_dot_product_attention(is_causal=True)`: never materializes the $T\times T$ matrix.
4. **"Nice numbers"**: pad the vocabulary from 50,257 to **50,304** ($= 2^7\cdot393$) so matmul dimensions divide evenly into GPU tiles. The padded tokens are never targets, so the network learns to give them near-zero probability. This alone gives several percent of speedup.
5. **Fused AdamW** (`torch.optim.AdamW(..., fused=True)`).
6. **Distributed data parallel** across 8 GPUs (Ch. 16).

In the video, these take one A100 from about 16K tokens/second (plain fp32) to about 170K, and 8 GPUs reach about 1.5M tokens/second, so the 10B-token run takes about 2 hours. In C/CUDA, [llm.c](https://github.com/karpathy/llm.c/discussions/481) reproduces GPT-2 124M in about 90 minutes on one 8×H100 node, for about \$20.

!!! llm "From GPT-2 to ChatGPT, for \$100"
    Karpathy's [nanochat](https://github.com/karpathy/nanochat) (2025) extends this into the full pipeline of a chat model in ~8,000 lines: train a tokenizer, pretrain a ~560M-parameter model, "midtrain" on conversations and multiple-choice data, SFT, optional RL on math problems, then serve it in a ChatGPT-style web UI, all in about 4 hours on one 8×H100 node. Parts IV–VI of this site explain each stage.

## Key takeaways

- A GPT is: embeddings (+ positions) → $L$ pre-norm blocks → final norm → tied output projection → cross-entropy on shifted targets.
- Shapes: `(B, T)` → `(B, T, C)` → heads `(B, H, T, d_h)` and scores `(B, H, T, T)` → `(B, T, C)` → `(B, T, V)`.
- Check before training: parameter count, logits shape, initial loss ≈ $\ln V$, causality, overfitting one batch.
- The training loop is: batch, forward, zero_grad, backward, clip, step, with AdamW, warmup + cosine.
- Real training uses gradient accumulation to reach large batches, bf16 autocast, `torch.compile`, FlashAttention, padded dimensions, and data parallelism.

## Exercises

1. Run `python code/gpt.py`. Then change `n_layer` to 1 and to 4. How do the final loss and the sample change?
2. Remove the causal mask (`is_causal=False`) and train the char-level model. Why does the training loss collapse toward 0, and why are the samples garbage?
3. Remove weight tying. How many parameters does GPT-2 small gain? Does the char-level training change?
4. Prove that `(loss / k).backward()` called on $k$ micro-batches of equal size gives the same gradient as one `backward()` on the concatenated batch with mean loss.
5. Why does padding the vocabulary to 50,304 not change what the model can learn?
6. **Code:** load the pretrained GPT-2 weights (§14.5), compute its loss on a paragraph of Wikipedia text, and convert it to perplexity and bits-per-byte (§4.7).

??? success "Solutions"
    1. More layers give a lower final loss and more coherent samples, up to the point where this tiny dataset is memorized.
    2. Without the mask, position $t$ can see $x_{t+1}$, the very token it should predict, and learns to copy it. At generation time the future isn't available, so the learned shortcut is useless.
    3. $50257\times768 = 38{,}597{,}376$ parameters. With a 40-character vocabulary the difference is negligible.
    4. Gradients are linear: $\nabla\frac1{kB}\sum_{i=1}^{kB}\ell_i = \sum_{j=1}^k\frac1k\nabla\Big(\frac1B\sum_{i\in\text{micro}_j}\ell_i\Big)$, and `.grad` accumulates the sum over the $k$ backward calls.
    5. The extra 47 tokens never appear in the data, so their only effect is 47 extra logits that training pushes down. The probabilities of real tokens are unaffected once those logits are very negative.
    6. Expect a loss around 3 nats/token on clean English for GPT-2 124M.

## References

- A. Radford et al., [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1, 2018) and [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019).
- T. Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (GPT-3, 2020), Table 2.1 (batch sizes and learning rates by model size).
- A. Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT), [build-nanogpt](https://github.com/karpathy/build-nanogpt), [llm.c](https://github.com/karpathy/llm.c), [nanochat](https://github.com/karpathy/nanochat).
- G. Penedo et al., [The FineWeb Datasets](https://arxiv.org/abs/2406.17557) (2024).
- R. Zellers et al., [HellaSwag](https://arxiv.org/abs/1905.07830) (ACL 2019).
