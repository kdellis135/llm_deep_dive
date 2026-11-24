# 23. Fine-Tuning in Practice

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 1](../math/linear-algebra.md) (rank, SVD), [Ch. 16](../scale/distributed.md) (memory), [Ch. 19](../inference/efficient.md) (quantization), [Ch. 21](post-training.md) (SFT, DPO). **You will learn:** when fine-tuning is the right tool and when prompting or retrieval is better; the memory cost of full fine-tuning; parameter-efficient fine-tuning, and LoRA in full detail (the math, initialization, gradients, scaling, merging) with a tested implementation; QLoRA and the NF4 data type; data preparation; hyperparameters that work; evaluation and forgetting; distillation; and how fine-tuned models are served.
    **Why it matters:** fine-tuning is how you adapt a general model to your domain, format, style or latency budget. It's also widely misused: most "the model doesn't know our product" problems are retrieval problems, not fine-tuning problems. This chapter gives you the judgment and the mechanics.

!!! watch "Read"
    - E. Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (2021) and T. Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (2023).
    - S. Raschka, [Practical Tips for Finetuning LLMs Using LoRA](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms) (2023), with hundreds of controlled experiments.
    - Hugging Face [PEFT](https://huggingface.co/docs/peft) and [TRL](https://huggingface.co/docs/trl) documentation.

---

## 23.1 Should you fine-tune?

Work up this ladder, and stop at the first rung that meets your quality, cost and latency targets:

1. **Prompting** (Ch. 25): clear instructions, examples, structured output. Minutes to iterate.
2. **Retrieval-augmented generation** (Ch. 26): give the model the facts it needs at query time.
3. **Fine-tuning**: change the model's behavior by training.

| Goal | Best tool | Why |
|---|---|---|
| Model must know facts that change, or many private documents | **RAG** | facts stay current and citable; fine-tuning teaches facts unreliably and can increase hallucination ([Gekhman et al., 2024](https://arxiv.org/abs/2405.05904)) |
| Consistent format, style, tone or persona | fine-tuning (or a good prompt) | behaviors are what SFT teaches well (§21.2) |
| A narrow task done by a much smaller, cheaper, faster model | fine-tuning / **distillation** | a small tuned model can match a large general one on one task |
| A specialized skill or domain language (legal drafting, a DSL, a rare language) | fine-tuning, often with continued pretraining | needs many examples |
| Better judgment from preference data | preference fine-tuning (DPO) or RL fine-tuning | §21.7, §22.3 |

Fine-tuning has real costs: data collection, training infrastructure, an evaluation suite, model hosting, and a model that's frozen at a base version while the providers' general models keep improving. A fine-tuned small model can be overtaken by the next general model release.

## 23.2 Full fine-tuning

Updating every weight needs the full training memory of §16.1: about **16 bytes per parameter** plus activations, so about 112 GB for a 7B model. That's multi-GPU territory with FSDP. It gives the most capacity to change the model, and also the most **catastrophic forgetting**: a model fine-tuned hard on one task can lose general abilities it had before. Use low learning rates (around $10^{-5}$), few epochs, and mix in general data if forgetting matters.

## 23.3 Parameter-efficient fine-tuning (PEFT)

Freeze the pretrained weights and train a small number of new parameters:

- **Adapters** ([Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)): small bottleneck MLPs inserted in each block. Add inference latency.
- **Prompt / prefix tuning** ([Lester et al., 2021](https://arxiv.org/abs/2104.08691); [Li & Liang, 2021](https://arxiv.org/abs/2101.00190)): learn "virtual token" embeddings prepended to the input (or to each layer's keys and values). Very few parameters, but weaker.
- **LoRA**: learn a low-rank update to existing weight matrices. No added latency after merging. The standard choice.

## 23.4 LoRA

For a frozen pretrained weight $\mathbf{W}_0\in\R^{d_\text{out}\times d_\text{in}}$, LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)) learns

$$
\mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r}\,\mathbf{B}\mathbf{A},\qquad \mathbf{B}\in\R^{d_\text{out}\times r},\ \mathbf{A}\in\R^{r\times d_\text{in}},\ r\ll\min(d_\text{in},d_\text{out}),
$$

so the forward pass is $\mathbf{h} = \mathbf{W}_0\mathbf{x}+\frac\alpha r\mathbf{B}(\mathbf{A}\mathbf{x})$. The update $\mathbf{B}\mathbf{A}$ has rank at most $r$ (Thm 1.8).

```python
--8<-- "code/lora.py:lora"
```

<div class="widget" data-widget="lora"></div>

**Parameter count.** $r(d_\text{in}+d_\text{out})$ instead of $d_\text{in}d_\text{out}$. For a $4096\times4096$ matrix with $r = 16$: 131,072 vs 16.8M, **0.78%**. The paper fine-tuned GPT-3 175B with 10,000× fewer trainable parameters and 3× less GPU memory, matching full fine-tuning quality. The memory saving comes from not storing gradients and Adam states for the frozen weights. Only the forward/backward activations and the frozen bf16 weights remain.

**Initialization.** $\mathbf{A}$ is random (Kaiming-uniform, like `nn.Linear`) and $\mathbf{B} = \mathbf{0}$. So at step 0, $\mathbf{B}\mathbf{A} = \mathbf{0}$ and the model is **exactly** the pretrained model (checked in the tests). Training starts from the pretrained function, not from a random perturbation of it.

**Gradients.** With $\mathbf{g} = \partial\mathcal{L}/\partial\mathbf{h}$ and $s = \alpha/r$, by Theorem 2.6:

$$
\nabla_\mathbf{B}\mathcal{L} = s\,\mathbf{g}\,(\mathbf{A}\mathbf{x})^\top,\qquad \nabla_\mathbf{A}\mathcal{L} = s\,\mathbf{B}^\top\mathbf{g}\,\mathbf{x}^\top.
$$

At initialization $\mathbf{B} = \mathbf{0}$, so $\nabla_\mathbf{A}\mathcal{L} = \mathbf{0}$: in the first step only $\mathbf{B}$ moves, and $\mathbf{A}$ starts learning once $\mathbf{B}\ne\mathbf{0}$. (If both were zero, neither would ever move; if both were random, the model would start perturbed.)

**Why low rank suffices.** Fine-tuning updates appear to have low **intrinsic dimension**: Aghajanyan et al. ([2021](https://arxiv.org/abs/2012.13255)) showed that optimizing a random projection of just a few hundred parameters can reach 90% of full fine-tuning performance on many tasks, and larger pretrained models have *lower* intrinsic dimension. Hu et al. found that the learned $\Delta\mathbf{W}$ has a few dominant singular values, and that it amplifies directions already present in $\mathbf{W}_0$ but not emphasized.

**Scaling $\alpha/r$.** The factor makes the update's scale less sensitive to $r$, so a learning rate tuned for one rank roughly carries over to another. Common practice is $\alpha = 2r$ or $\alpha = r$. **rsLoRA** ([Kalajdzievski, 2023](https://arxiv.org/abs/2312.03732)) argues for $\alpha/\sqrt r$ so that large ranks keep learning.

**Merging.** After training, fold the update into the weight: $\mathbf{W}\leftarrow\mathbf{W}_0+\frac\alpha r\mathbf{B}\mathbf{A}$. Inference then has **zero extra cost** (tested: merged and unmerged outputs agree). Alternatively, keep adapters separate to serve many fine-tunes over one base model (§23.10).

**Where to apply LoRA.** The original paper used only $\mathbf{W}_Q$ and $\mathbf{W}_V$. QLoRA and later studies found that applying it to **all linear layers** (attention and MLP) matters more than the rank. `add_lora` in our code wraps the layers you name:

```python
from llama import Llama, LlamaConfig
from lora import add_lora

model = Llama(LlamaConfig(vocab_size=1000, dim=256, n_layers=4, n_heads=8, n_kv_heads=2, ffn_dim=688))
add_lora(model, target_names=("wq", "wk", "wv", "wo", "w1", "w2", "w3"), r=16, alpha=32)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
```

(Make sure to freeze the embeddings and norms too, or train them deliberately.)

**What LoRA trades off.** Biderman et al., [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (2024): on tasks far from pretraining (e.g. continued pretraining on code), LoRA underperforms full fine-tuning, but it also preserves the base model's other abilities better. For instruction and style tuning, LoRA is usually on par.

## 23.5 QLoRA

QLoRA ([Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)) stores the frozen base model in **4 bits** and trains LoRA adapters in bf16 on top. Gradients flow *through* the dequantized 4-bit weights into the adapters. It made fine-tuning a 65B model possible on **one 48 GB GPU**, matching 16-bit LoRA quality. Three ingredients:

1. **NF4 (4-bit NormalFloat).** Pretrained weights are roughly normally distributed. NF4 places its 16 levels at the **quantiles** of $\mathcal{N}(0,1)$ (rescaled to $[-1,1]$, with an exact zero), so each level is used about equally often. That's information-theoretically optimal for normal data. Each block of 64 weights has its own absmax scale.
2. **Double quantization.** The per-block fp32 scales themselves cost $32/64 = 0.5$ bits per parameter. Quantizing them to 8 bits (in blocks of 256) cuts that to about $0.127$ bits per parameter.
3. **Paged optimizers.** Optimizer states move to CPU memory automatically during memory spikes (long sequences), using unified memory, to avoid out-of-memory crashes.

Memory for a 7B model: about 3.5–4 GB for the 4-bit weights, plus adapters, their optimizer states and activations, which fits a 16–24 GB consumer GPU.

## 23.6 Preparing data

Data quality decides the outcome far more than hyperparameters.

- **Format.** Chat-formatted JSONL, one conversation per line, matching the chat template you'll use at inference:

    ```json
    {"messages": [{"role": "system", "content": "You are a support agent for Acme..."},
                  {"role": "user", "content": "How do I reset my router?"},
                  {"role": "assistant", "content": "Here are the steps: ..."}]}
    ```

- **Size.** For style and format: a few hundred to a few thousand high-quality examples. For a new skill: thousands to tens of thousands.
- **Quality.** Every response should be one you'd be happy for the model to give. Inconsistent or wrong examples teach inconsistency. Review a random sample by hand.
- **Coverage.** Include edge cases, refusals or "I don't know" answers where appropriate, multi-turn examples if you need multi-turn behavior, and the real distribution of inputs (length, language, typos).
- **Splits.** Hold out a validation and a test set, deduplicated against training, before you start. Decontaminate any benchmark you plan to report.
- **Synthetic data.** Generate candidates with a strong model, then filter (by rules, a grader model, or humans). Check the model provider's terms of use before training on its outputs.

## 23.7 Hyperparameters that work

| Setting | Full fine-tuning | LoRA / QLoRA |
|---|---|---|
| Learning rate | $5\times10^{-6}$ – $2\times10^{-5}$ | $1\times10^{-4}$ – $3\times10^{-4}$ |
| Schedule | warmup 3–10%, cosine or linear decay | same |
| Epochs | 1–3 | 1–3 (watch validation loss) |
| Rank $r$, $\alpha$ | n/a | $r = 8$–64, $\alpha = r$ or $2r$ |
| Target modules | all | all linear layers |
| Batch size | 32–128 sequences (with accumulation) | same |
| Loss | response tokens only (§21.2) | same |

Signs of overfitting: validation loss rising while training loss falls; outputs that copy phrases from training examples; loss of general abilities. Fix with fewer epochs, a lower learning rate, more diverse data, or a smaller rank.

A complete supervised fine-tuning script with Hugging Face TRL and PEFT (illustrative; library APIs change, so check the current docs):

```python
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})
peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                         target_modules="all-linear", task_type="CAUSAL_LM")
args = SFTConfig(
    output_dir="out", num_train_epochs=2, learning_rate=2e-4, lr_scheduler_type="cosine",
    warmup_ratio=0.05, per_device_train_batch_size=4, gradient_accumulation_steps=8,
    bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=100,
    assistant_only_loss=True,          # loss on assistant turns only
)
trainer = SFTTrainer(model="Qwen/Qwen2.5-1.5B-Instruct", args=args, peft_config=peft_config,
                     train_dataset=dataset["train"], eval_dataset=dataset["validation"])
trainer.train()
```

## 23.8 Evaluating a fine-tune

1. **Task metric on the held-out test set**: exact match, F1, pass rate of unit tests, or a rubric scored by humans or an LLM judge (Ch. 29). Compare against the base model with your best prompt, which is the baseline fine-tuning must beat.
2. **Regression suite**: a sample of general capabilities (instruction following, safety, reasoning) to detect forgetting.
3. **Qualitative review** of outputs on real inputs, especially failures.

## 23.9 Distillation

Train a small **student** to imitate a large **teacher**:

- **Sequence-level distillation** (most common with API models): generate teacher responses for many prompts and run SFT on them. This is how DeepSeek-R1's small reasoning models were made (§22.5).
- **Logit distillation**: minimize $\KL(p_\text{teacher}\Vert p_\text{student})$ over the full next-token distribution at each position, often with a temperature ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)). This carries much more information per example ("dark knowledge": which wrong answers are nearly right), but requires the teacher's logits, so it's mostly done in-house (e.g. Gemma and Llama 3.2's small models were distilled from larger ones).
- **On-policy distillation**: the student generates and the teacher scores the student's own tokens (reverse KL), which fixes the train/test mismatch of imitating teacher-written text ([Agarwal et al., 2024](https://arxiv.org/abs/2306.13649)).

## 23.10 Serving fine-tuned models

- **Merged weights**: simplest; one model per fine-tune.
- **Multi-LoRA serving**: keep one base model in GPU memory and many small adapters, batching requests for different adapters together ([S-LoRA](https://arxiv.org/abs/2311.03285), [Punica](https://arxiv.org/abs/2310.18547)). vLLM and SGLang support this. A 16-rank adapter for a 7B model is only tens of MB.
- **Quantize after merging** (GPTQ/AWQ, §19.5) for cheaper deployment, and re-evaluate afterwards.
- **Hosted fine-tuning**: several API providers offer fine-tuning of their models (supervised, preference, or RL fine-tuning with graders). Availability and supported models change frequently, so check the current documentation.

## Key takeaways

- Prompt first, retrieve for knowledge, fine-tune for behavior, format, style, narrow tasks and smaller models.
- Full fine-tuning: ~16 bytes/parameter, most capacity, most forgetting.
- LoRA: $\mathbf{W}_0+\frac\alpha r\mathbf{B}\mathbf{A}$ with $\mathbf{B} = 0$ at init (exact identity at start), $r(d_\text{in}+d_\text{out})$ parameters, mergeable for zero inference overhead. Apply to all linear layers; learning rate around $2\times10^{-4}$.
- QLoRA: frozen NF4 4-bit base + bf16 LoRA, double quantization and paged optimizers. Fine-tune big models on one GPU.
- Data quality and a held-out evaluation decide success. Distillation turns a large model's outputs into a small model's skills.

## Exercises

1. Compute LoRA's trainable parameters for Llama 3 8B with $r = 16$ applied to all seven linear layers in every block ($d = 4096$, KV dim 1024, FFN 14336, 32 layers).
2. Show that if $\mathbf{A}$ were initialized to zero (and $\mathbf{B}$ randomly), training would also work. Why do implementations zero $\mathbf{B}$ instead? (Hint: think about which matrix receives the input $\mathbf{x}$.)
3. Prove that the LoRA update $\mathbf{B}\mathbf{A}$ can represent any matrix of rank $\le r$, and that its column space is spanned by the columns of $\mathbf{B}$.
4. Estimate the memory for QLoRA fine-tuning of a 70B model: 4-bit weights with double quantization, LoRA with $r = 16$ on all linear layers, Adam states in fp32 for the adapters (ignore activations).
5. A team wants their support bot to know their product documentation, which changes weekly. Should they fine-tune? What would you build instead?
6. **Code:** fine-tune a 0.5–1.5B instruct model with LoRA on 500 examples of a narrow task (e.g. extracting fields from emails into JSON). Compare against the base model with a few-shot prompt.

??? success "Solutions"
    1. Per layer: $\mathbf{W}_Q$ and $\mathbf{W}_O$: $16(4096+4096)$ each; $\mathbf{W}_K,\mathbf{W}_V$: $16(4096+1024)$ each; $\mathbf{W}_1,\mathbf{W}_3$: $16(4096+14336)$ each; $\mathbf{W}_2$: $16(14336+4096)$. Sum $= 16\cdot(16384+10240+55296) = 1{,}310{,}720$; ×32 ≈ 41.9M (0.52% of 8B).
    2. With $\mathbf{A} = \mathbf{0}$ the product is also 0 at init, and $\nabla_\mathbf{A} = s\mathbf{B}^\top\mathbf{g}\mathbf{x}^\top\ne0$, so it trains. Convention aside, zeroing $\mathbf{B}$ lets $\mathbf{A}$ act as a fixed random projection of the input at first while $\mathbf{B}$ learns the output side; empirically both work, and the original paper chose $\mathbf{B} = 0$.
    3. Any rank-$\le r$ matrix has a factorization $\mathbf{U}_r\boldsymbol\Sigma_r\mathbf{V}_r^\top$ (truncated SVD, Thm 1.11): set $\mathbf{B} = \mathbf{U}_r\boldsymbol\Sigma_r$, $\mathbf{A} = \mathbf{V}_r^\top$. Every column of $\mathbf{B}\mathbf{A}$ is $\mathbf{B}$ times a column of $\mathbf{A}$, so it lies in the span of $\mathbf{B}$'s columns.
    4. Weights: $70\times10^9\times4.127/8\approx36$ GB. LoRA on all linear layers at $r = 16$ is roughly 0.3% of parameters ≈ 0.2B: bf16 weights 0.4 GB + fp32 Adam states (8 bytes) 1.6 GB + gradients 0.4 GB. Total ≈ 39 GB before activations, so it fits a 48 GB GPU with modest batch sizes.
    5. No. Build RAG over the documentation (Ch. 26), re-indexed on each change; optionally fine-tune for tone and format later.
    6. Typically the fine-tuned small model matches or beats few-shot prompting of the same model in accuracy and format reliability, and needs much shorter prompts.

## References

- E. Hu et al., [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (ICLR 2022).
- T. Dettmers, A. Pagnoni, A. Holtzman, L. Zettlemoyer, [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) (NeurIPS 2023).
- A. Aghajanyan, L. Zettlemoyer, S. Gupta, [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) (ACL 2021).
- D. Biderman et al., [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (TMLR 2024).
- S.-Y. Liu et al., [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353) (ICML 2024).
- Z. Gekhman et al., [Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?](https://arxiv.org/abs/2405.05904) (EMNLP 2024).
- C. Zhou et al., [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) (NeurIPS 2023).
