---
seo_title: "Distributed Training & Mixed Precision for LLMs"
description: "The memory budget of training, fp32 to fp8 and why bf16 won, mixed precision, the roofline model, ring all-reduce with proof, and data, ZeRO, tensor and pipeline parallelism."
---

# 16. Distributed Training & Precision

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 5](../math/optimization.md) (AdamW state), [Ch. 7](../nn/backprop.md) (activations, $6N$), [Ch. 13](../transformer/block.md), [Ch. 15](pretraining.md). **You will learn:** the exact memory budget of training (16 bytes per parameter plus activations), floating-point formats from fp32 to fp8 and why bf16 won, mixed precision and loss scaling, GPU basics and the roofline model (when you're compute-bound vs memory-bound), collective communication and the cost of ring all-reduce (with proof), and every axis of parallelism: data (DDP), sharded data (ZeRO/FSDP), tensor (Megatron, with proof that the split is exact), sequence/context, pipeline (with the bubble formula), and expert parallelism, ending with how a 405B model is laid out on 16,384 GPUs.
    **Why it matters:** no frontier model fits on one GPU, not even for one step. Parallelism and precision determine what can be trained at all, how fast, and at what cost.

!!! watch "Read & watch"
    - Hugging Face, [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (2025). The best single resource on this chapter, with interactive memory calculators.
    - Google DeepMind, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/) (2025). Rooflines and sharding from first principles.
    - H. He, [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html). Compute, memory bandwidth and overhead.
    - Karpathy's [Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU) covers mixed precision, `torch.compile` and DDP hands-on.

---

## 16.1 The memory budget

For a model with $\Psi$ parameters trained with AdamW in mixed precision (the standard setup; [Rajbhandari et al., ZeRO, 2020](https://arxiv.org/abs/1910.02054)):

| Item | Precision | Bytes per parameter |
|---|---|---|
| Weights (used in forward/backward) | bf16 | 2 |
| Gradients | bf16 | 2 |
| Master weights | fp32 | 4 |
| Adam first moment $\mathbf{m}$ | fp32 | 4 |
| Adam second moment $\mathbf{v}$ | fp32 | 4 |
| **Total "model states"** | | **16** |

So a 7B model needs 112 GB and a 70B model 1.12 TB of model states, against 80 GB on an H100. On top of that come **activations**, which grow with batch × sequence length × width × depth (§13.4), plus temporary buffers and fragmentation. Training even a 7B model on one 80 GB GPU requires sharding or offloading. Everything in this chapter is about splitting these bytes, and the computation, across many GPUs.

## 16.2 Number formats

A binary floating-point number with sign $s$, exponent $e$ and mantissa (fraction) $m$ represents $(-1)^s\cdot2^{e-\text{bias}}\cdot(1.m)_2$. The exponent bits set the **range**; the mantissa bits set the **precision**. The relative spacing between neighboring numbers (the machine epsilon) is $2^{-\#\text{mantissa bits}}$.

| Format | Sign / exponent / mantissa | Max value | Machine ε | Use |
|---|---|---|---|---|
| fp32 | 1 / 8 / 23 | $3.4\times10^{38}$ | $1.2\times10^{-7}$ | master weights, optimizer state, reductions |
| tf32 | 1 / 8 / 10 | (fp32 range) | $9.8\times10^{-4}$ | internal matmul format on Ampere+ GPUs |
| fp16 | 1 / 5 / 10 | 65,504 | $9.8\times10^{-4}$ | older mixed precision; needs loss scaling |
| **bf16** | 1 / 8 / 7 | $3.4\times10^{38}$ | $7.8\times10^{-3}$ | standard training and inference format |
| fp8 E4M3 | 1 / 4 / 3 | 448 | 0.125 | forward-pass matmuls (H100+) |
| fp8 E5M2 | 1 / 5 / 2 | 57,344 | 0.25 | gradients (more range, less precision) |

**Why bf16 won.** bf16 (brain float, from Google) keeps fp32's 8 exponent bits, so it has the same range: gradients of $10^{-10}$ and activations of $10^{10}$ are both representable. fp16 has more precision but a tiny range: values below about $6\times10^{-8}$ underflow to 0, and above 65,504 overflow to infinity. Deep-learning values span many orders of magnitude, and range matters more than precision.

### Mixed precision

[Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740) made half-precision training work:

1. Keep an **fp32 master copy** of the weights. The update $\eta\cdot\Delta$ is often smaller than bf16's spacing near the weight's value (for a weight of 1.0, bf16 can't represent changes smaller than about 0.004), so updates applied directly in bf16 would be lost to rounding.
2. Run forward and backward matmuls in **half precision** on Tensor Cores (much faster, half the memory traffic), but **accumulate** dot products in fp32 inside the matmul.
3. Keep numerically sensitive ops (softmax, norms, losses, reductions) in fp32. PyTorch's `torch.autocast` does this automatically.
4. With fp16 only: **loss scaling**. Multiply the loss by a factor $S$ (e.g. $2^{16}$) before `backward()` so small gradients don't underflow, then divide the gradients by $S$ before the update. Dynamic loss scaling halves $S$ whenever an overflow (inf/NaN) appears and skips that step, and doubles it after a run of clean steps. bf16 doesn't need it.

**FP8 training.** H100s run fp8 matmuls at twice the bf16 rate. Because fp8's range is tiny, each tensor (or each block of a tensor) carries its own scale factor. DeepSeek-V3 trained a 671B MoE model largely in fp8, with scales per $1\times128$ activation tile and per $128\times128$ weight block, and with accumulation promoted to fp32 at regular intervals ([DeepSeek-AI, 2024](https://arxiv.org/abs/2412.19437)). Blackwell GPUs add 4-bit floating-point (FP4) formats, used mainly for inference so far.

## 16.3 GPUs and the roofline model

A GPU has thousands of arithmetic units (grouped into streaming multiprocessors, with **Tensor Cores** for matrix multiply) and **HBM** memory. Roughly, for an NVIDIA H100 SXM:

| Resource | H100 SXM |
|---|---|
| bf16 dense matmul throughput | ≈ 989 TFLOP/s |
| fp8 dense matmul throughput | ≈ 1,979 TFLOP/s |
| HBM capacity / bandwidth | 80 GB / 3.35 TB/s |
| NVLink (GPU↔GPU within a node) | 900 GB/s |
| InfiniBand (between nodes) | ≈ 50 GB/s per 400 Gb/s NIC, usually 8 NICs per node |

An operation's **arithmetic intensity** is FLOPs per byte moved to/from memory. The **roofline model** says achievable throughput is $\min(\text{peak FLOP/s},\ \text{intensity}\times\text{bandwidth})$. The crossover ("ridge point") for an H100 is $989/3.35\approx295$ FLOPs per byte.

!!! theorem "Proposition 16.1 (Matmul intensity)"
    Multiplying $(m\times k)$ by $(k\times n)$ in bf16 costs $2mkn$ FLOPs and moves at least $2(mk+kn+mn)$ bytes. For square $n\times n$ matrices the intensity is $\frac{2n^3}{6n^2} = \frac n3$ FLOPs/byte, so matmuls are compute-bound for $n\gtrsim900$ on an H100.

**Elementwise ops** (adding a bias, GELU, dropout, residual adds) do about 1 FLOP per element while moving 4+ bytes: intensity below 1. They're entirely **memory-bound**. This is why **kernel fusion** matters: `torch.compile` fuses chains of elementwise ops into one kernel that reads its input once and writes its output once, and FlashAttention fuses the whole attention computation (Ch. 19). It's also why LLM *decoding* at small batch sizes is memory-bound: each step reads all the weights to do only $2$ FLOPs per weight per sequence (Ch. 19).

## 16.4 Collective communication

Distributed training is built on a few collective operations across $p$ GPUs:

- **all-reduce**: every GPU ends with the sum of all GPUs' tensors (used to average gradients);
- **reduce-scatter**: every GPU ends with one $1/p$ slice of the sum;
- **all-gather**: every GPU ends with the concatenation of all GPUs' slices;
- **all-to-all**: every GPU sends a different slice to every other GPU (used by MoE, §16.10).

An all-reduce is a reduce-scatter followed by an all-gather.

!!! theorem "Proposition 16.2 (Ring all-reduce is bandwidth-optimal)"
    Arrange $p$ GPUs in a ring. All-reducing $n$ bytes takes $2(p-1)$ steps, in each of which every GPU sends $n/p$ bytes to its neighbor. Each GPU sends $2\frac{p-1}{p}n$ bytes in total, **independent of $p$** for large $p$ (≈ $2n$), and no algorithm can do better than $\approx 2\frac{p-1}{p}n$ per GPU.

??? proof "Proof"
    Split the tensor into $p$ chunks. *Reduce-scatter:* in step $j$, GPU $i$ sends chunk $(i-j)\bmod p$ to GPU $i+1$, which adds it to its own copy. After $p-1$ steps, each GPU holds one chunk fully summed. *All-gather:* circulate the summed chunks around the ring for another $p-1$ steps. Each step moves $n/p$ bytes per GPU, so the total is $2(p-1)n/p$.

    *Lower bound:* each GPU must receive the contributions to the $(p-1)/p$ of the result it doesn't own, at least $\frac{p-1}{p}n$ bytes for the reduction, and then the reduced values of those chunks, another $\frac{p-1}{p}n$. $\blacksquare$

The time is about $2n/\text{bandwidth}$ plus a latency term per step. NCCL implements ring and tree variants. For a 7B model in bf16, an all-reduce of the gradients moves about $2\times14$ GB per GPU per step, which takes about 30 ms over NVLink but about 0.6 s over a single 50 GB/s link. Where GPUs are connected determines which parallelism you can afford.

## 16.5 Data parallelism (DDP)

Each of $p$ GPUs holds a **full replica** of the model, processes a different slice of the batch, and the gradients are **all-reduced** (averaged) before the identical optimizer step on every GPU. Mathematically it's the same as training with a $p$-times larger batch (§14.6, gradient accumulation across devices).

PyTorch's `DistributedDataParallel` overlaps communication with computation: gradients are grouped into buckets, and each bucket's all-reduce starts as soon as backward has produced it, while backward continues on earlier layers.

```python
# launch:  torchrun --nproc_per_node=8 train.py
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank % torch.cuda.device_count())
model = DDP(GPT(cfg).cuda())           # same init on every rank (broadcast from rank 0)
# each rank reads a DIFFERENT shard of data; loss.backward() now all-reduces gradients
```

**Limitation:** every GPU still holds all 16 bytes per parameter. DDP alone can't train anything over about 5B parameters on 80 GB GPUs.

## 16.6 Sharded data parallelism: ZeRO and FSDP

ZeRO ([Rajbhandari et al., 2020](https://arxiv.org/abs/1910.02054)) removes the redundancy: with $N_d$ data-parallel GPUs, each keeps only $1/N_d$ of the model states.

| Stage | Sharded across GPUs | Memory per GPU ($\Psi$ params) | Communication vs DDP |
|---|---|---|---|
| DDP | nothing | $16\Psi$ | 1× |
| ZeRO-1 | optimizer states | $4\Psi + 12\Psi/N_d$ | 1× |
| ZeRO-2 | + gradients | $2\Psi + 14\Psi/N_d$ | 1× |
| ZeRO-3 / **FSDP** | + weights | $16\Psi/N_d$ | 1.5× |

In ZeRO-3 / FSDP, each layer's weights are **all-gathered** just before use (forward and again in backward) and freed afterwards, and gradients are **reduce-scattered** so that each GPU receives only its own slice for its optimizer step. The extra all-gather is the 1.5× communication; it's overlapped with compute by prefetching the next layer. With 64 GPUs, a 70B model's 1.12 TB of states becomes 17.5 GB per GPU. PyTorch implements this as FSDP (`torch.distributed.fsdp.fully_shard`).

## 16.7 Tensor parallelism

When even one layer's activations and weights are too big, or the batch per GPU gets too small, split **individual matrix multiplications** across GPUs. Megatron-LM ([Shoeybi et al., 2019](https://arxiv.org/abs/1909.08053)) does this for the MLP with no communication in the middle.

!!! theorem "Proposition 16.3 (Column-then-row parallel MLP)"
    Split $\mathbf{W}_1\in\R^{d_\text{ff}\times d}$ by rows (output features) into $\mathbf{W}_1^{(1)},\dots,\mathbf{W}_1^{(p)}$, and $\mathbf{W}_2\in\R^{d\times d_\text{ff}}$ by the matching columns into $\mathbf{W}_2^{(1)},\dots,\mathbf{W}_2^{(p)}$. Then

    $$
    \mathbf{W}_2\,\phi(\mathbf{W}_1\mathbf{x}) = \sum_{i=1}^p\mathbf{W}_2^{(i)}\,\phi(\mathbf{W}_1^{(i)}\mathbf{x}).
    $$

    GPU $i$ computes one term with only its own slices. A single **all-reduce** at the end sums them.

??? proof "Proof"
    $\mathbf{W}_1\mathbf{x}$ is the stack of the blocks $\mathbf{W}_1^{(i)}\mathbf{x}$. Because $\phi$ acts elementwise, $\phi(\mathbf{W}_1\mathbf{x})$ is the stack of the blocks $\phi(\mathbf{W}_1^{(i)}\mathbf{x})$. Multiplying a column-blocked matrix by a row-blocked vector gives the sum of the block products. $\blacksquare$

    (This is why the nonlinearity must be elementwise. Splitting $\mathbf{W}_1$ the other way would need an all-reduce *before* $\phi$, since $\phi(a+b)\ne\phi(a)+\phi(b)$.)

Attention splits the same way: **assign heads to GPUs** (column-split $\mathbf{W}_Q,\mathbf{W}_K,\mathbf{W}_V$ by head, row-split $\mathbf{W}_O$), so attention needs one all-reduce too. Each block costs two all-reduces in the forward pass and two in the backward pass, on activations of size $B\times T\times d$. That's frequent, latency-sensitive communication, so tensor parallelism (TP) is used **within a node** over NVLink, typically with TP = 8.

**Sequence parallelism** ([Korthikanti et al., 2022](https://arxiv.org/abs/2205.05198)) additionally splits the LayerNorm and dropout regions along the sequence dimension, turning each all-reduce into a reduce-scatter plus an all-gather and removing duplicated activation memory.

## 16.8 Context parallelism

For very long sequences (128K+ tokens), even one sequence's activations don't fit on a GPU. **Context parallelism** splits the sequence across GPUs. Each GPU holds a chunk of the queries, keys and values. Attention needs every query to see every (earlier) key, so the K/V chunks are passed around a ring of GPUs (**ring attention**, [Liu et al., 2023](https://arxiv.org/abs/2310.01889)), combined with the online-softmax trick (Ch. 19) so that partial results from each chunk can be merged exactly. Causal masking makes the work uneven, which is fixed by giving each GPU one chunk from the start and one from the end of the sequence.

## 16.9 Pipeline parallelism

Split the model **by layers** into $p$ stages on different GPUs (or nodes). Activations flow forward from stage to stage, and gradients flow back. Only activations at stage boundaries cross the network, which is cheap enough for slower inter-node links.

The problem is idle time. A naive schedule keeps only one stage busy at a time. Splitting the batch into $m$ **micro-batches** lets stages work on different micro-batches at once (GPipe, [Huang et al., 2019](https://arxiv.org/abs/1811.06965)).

!!! theorem "Proposition 16.4 (Pipeline bubble)"
    With $p$ stages and $m$ micro-batches (GPipe schedule, all forwards then all backwards), the fraction of idle time is

    $$
    \text{bubble} = \frac{p-1}{m+p-1}.
    $$

??? proof "Proof"
    Let each stage take one time unit per micro-batch in the forward pass. The last stage starts the first micro-batch at time $p-1$ and finishes the last at $m+p-1$, so the forward phase takes $m+p-1$ units while each stage does only $m$ units of work. The backward phase is symmetric. Idle fraction: $\frac{(m+p-1)-m}{m+p-1}$. $\blacksquare$

With $p = 16$ stages you need $m\gg16$ micro-batches to make the bubble small (e.g. $m = 128$ gives 10.5%). **1F1B** schedules ([PipeDream](https://arxiv.org/abs/1806.03377), Megatron) interleave one forward and one backward per micro-batch: same bubble, but far less activation memory, since only $p$ micro-batches are in flight. **Interleaved** schedules give each GPU several non-adjacent chunks of layers, shrinking the bubble by that factor at the cost of more communication. DeepSeek-V3's **DualPipe** feeds micro-batches from both ends of the pipeline and overlaps computation with communication.

## 16.10 Expert parallelism

In a mixture-of-experts model (Ch. 20), each token is routed to a few of many experts. **Expert parallelism** places different experts on different GPUs. Each MoE layer then needs two **all-to-all** exchanges: dispatch each token to the GPUs holding its experts, and combine the results back. All-to-all traffic across nodes is the main bottleneck of MoE training, which is why DeepSeek-V3 restricts each token to experts on at most 4 nodes and wrote custom communication kernels.

## 16.11 Putting it together: 4-D parallelism

Real runs combine everything. A GPU's "coordinates" are (data-parallel rank, pipeline stage, tensor-parallel rank, context-parallel rank), with the most communication-hungry axis on the fastest links:

| Axis | Splits | Communication | Placed on |
|---|---|---|---|
| Tensor (TP) | each matmul | 2 all-reduces per block per pass, on activations | within a node (NVLink) |
| Context (CP) | the sequence | ring exchange of K/V | within or across nodes |
| Pipeline (PP) | layers | point-to-point activations between stages | across nodes |
| Data (DP, FSDP) | the batch | gradient reduce-scatter / weight all-gather, once per step | across nodes, outermost |

Llama 3 405B pretraining on 16,384 H100s used **TP = 8, PP = 16, DP = 128** at 8K context, and **TP = 8, CP = 16, PP = 16, DP = 8** for the long-context stage at 128K ([Meta, 2024](https://arxiv.org/abs/2407.21783), Table 4), with the data-parallel axis using FSDP-style sharding. That configuration reached 38–43% MFU.

!!! tip "Rules of thumb"
    1. Use FSDP/ZeRO alone if the model's states fit when sharded across your GPUs and the per-GPU batch is reasonable. It's the simplest option.
    2. Add tensor parallelism (≤ 8, within a node) when per-GPU activations or memory are still too large.
    3. Add pipeline parallelism to span many nodes with large models.
    4. Add context parallelism for long sequences, and expert parallelism for MoE.
    5. Use activation checkpointing (§7.7) to trade about 33% more compute for much less activation memory.

## Key takeaways

- Mixed-precision AdamW costs **16 bytes per parameter** (bf16 weights and gradients, fp32 master weights and two moments), plus activations.
- bf16 has fp32's range with less precision; it doesn't need loss scaling. Keep fp32 master weights and accumulations. FP8 needs fine-grained scaling.
- Roofline: matmuls are compute-bound for large dimensions; elementwise ops are memory-bound, so fuse kernels.
- Ring all-reduce moves about $2n$ bytes per GPU regardless of GPU count.
- DDP replicates; ZeRO/FSDP shards states ($16\Psi/N_d$); TP splits matmuls (column then row, one all-reduce); PP splits layers (bubble $\frac{p-1}{m+p-1}$); CP splits sequences; EP splits experts (all-to-all).

## Exercises

1. How much memory do the model states of a 13B model take with DDP, ZeRO-1, ZeRO-2 and ZeRO-3 on 32 GPUs?
2. Find the largest number below 1.0 and the smallest number above 1.0 in bf16. What does this mean for an update of $10^{-4}$ to a weight of 1.0?
3. For a matmul of $(4096\times4096)$ by $(4096\times B)$ (a linear layer applied to $B$ tokens), compute the arithmetic intensity as a function of $B$. At what $B$ does it become compute-bound on an H100?
4. Show that tensor-parallel attention (heads split across GPUs) needs only one all-reduce in the forward pass.
5. With $p = 8$ pipeline stages, how many micro-batches are needed to keep the bubble under 5%?
6. A 70B model trains with TP = 8, PP = 4, DP = 32. How many GPUs is that, and how many bytes of model states does each GPU hold if DP also uses ZeRO-1?

??? success "Solutions"
    1. $\Psi = 13\times10^9$: DDP $208$ GB; ZeRO-1 $52+156/32 = 56.9$ GB; ZeRO-2 $26+182/32 = 31.7$ GB; ZeRO-3 $208/32 = 6.5$ GB.
    2. bf16 has 7 mantissa bits, so just above 1.0 the spacing is $2^{-7} = 0.0078$, giving 1.0078; just below 1.0 the spacing is $2^{-8}$, giving 0.9961. An update of $10^{-4}$ rounds away to nothing, which is why fp32 master weights are needed.
    3. FLOPs $2\cdot4096^2B$; bytes $\approx2(4096^2+4096B+4096B)$. For $B\ll4096$, intensity $\approx B$ FLOPs/byte, so compute-bound needs $B\gtrsim300$. That's why decoding with small batches is memory-bound (Ch. 19).
    4. Each GPU computes its heads' outputs $\operatorname{head}_h\mathbf{W}_O^{(h)}$ entirely locally (the row-split of $\mathbf{W}_O$ matches the head split), and the full output is the sum over heads (§12.6). One all-reduce computes the sum.
    5. $\frac{7}{m+7}<0.05$ means $m>133$, so $m\ge134$.
    6. $8\cdot4\cdot32 = 1024$ GPUs. TP × PP split the parameters 32 ways: $70\text{B}/32 = 2.19$B parameters per GPU. With ZeRO-1 across the 32 DP ranks: $4\cdot2.19 + 12\cdot2.19/32\approx8.75+0.82\approx9.6$ GB.

## References

- S. Rajbhandari et al., [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (SC 2020).
- P. Micikevicius et al., [Mixed Precision Training](https://arxiv.org/abs/1710.03740) (ICLR 2018); P. Micikevicius et al., [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) (2022).
- M. Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053) (2019); D. Narayanan et al., [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) (SC 2021).
- Y. Huang et al., [GPipe](https://arxiv.org/abs/1811.06965) (NeurIPS 2019).
- V. Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) (2022).
- H. Liu, M. Zaharia, P. Abbeel, [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) (2023).
- Y. Zhao et al., [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) (VLDB 2023).
- Hugging Face, [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) (2025).
