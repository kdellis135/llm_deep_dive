# Glossary

!!! abstract "How to use this page"
    One definition per term, used consistently across the site. The chapter in brackets is where the term is introduced or explained in depth. Notation is on the [Notation](../start/notation.md) page.

## A

**Activation** — the output of a neuron or layer after its nonlinearity; more loosely, any intermediate tensor of the forward pass, which must be stored for the backward pass. [6, 7]

**Activation checkpointing** — storing only some activations and recomputing the rest during backward, trading ~33% more compute for less memory. [7.7]

**Activation patching** — replacing one component's activation with its value from another input to measure its causal effect. [24.5]

**AdamW** — Adam with weight decay applied directly to the weights instead of through the gradient; the default LLM optimizer. [5.6]

**Advantage** — how much better an action or response was than expected, $A = R-b$; the weight in policy-gradient updates. [21.5]

**Agent** — an LLM in a loop that chooses its own actions (tool calls) based on observations, until a task is done. [27]

**ALiBi** — attention with linear biases: a fixed penalty proportional to distance added to attention scores. [11.4]

**All-reduce** — a collective operation leaving every GPU with the sum of all GPUs' tensors; used to average gradients. [16.4]

**Arithmetic intensity** — FLOPs performed per byte moved from memory; decides whether an operation is compute- or memory-bound. [16.3]

**Attention** — computing each position's output as a weighted average of value vectors, with weights from query–key similarity. [12]

**Attention sink** — a token (often the first) that receives large attention weight when a head has nothing to attend to. [12.7]

**Autoregressive** — generating one token at a time, each conditioned on all previous ones. [9.1]

## B

**Backpropagation** — reverse-mode automatic differentiation applied to a neural network. [7]

**Base model** — a pretrained model before post-training; it continues documents rather than following instructions. [15.7]

**Batch size** — number of sequences (or tokens) per optimizer step. [5.4]

**Beam search** — decoding that keeps the $k$ most probable partial sequences. [18.2]

**bf16 (bfloat16)** — 16-bit floating-point format with fp32's exponent range and 7 mantissa bits; the standard training format. [16.2]

**BM25** — the standard lexical retrieval score: IDF × saturating term frequency × length normalization. [26.3]

**BPE (byte-pair encoding)** — tokenizer training by repeatedly merging the most frequent adjacent pair of symbols. [10.3]

**Bradley–Terry model** — $\Pr(i\succ j) = \sigma(r_i-r_j)$; the model behind reward models and arena rankings. [21.3, 29.5]

## C

**Causal mask** — setting attention scores to $-\infty$ for future positions so position $t$ sees only $x_{\le t}$. [12.4]

**Chain of thought (CoT)** — intermediate reasoning written before the final answer. [22.1]

**Chat template** — the format that serializes a conversation (roles, special tokens) into one token sequence. [10.6]

**Chinchilla-optimal** — the compute-optimal allocation of parameters and tokens, about 20 tokens per parameter. [17.3]

**Circuit** — a subgraph of model components that implements a specific behavior. [24.5]

**Compaction** — summarizing earlier context so a conversation or agent can continue past the context limit. [27.8, 28.10]

**Constitutional AI** — training with model critiques and AI preference labels guided by written principles. [21.8]

**Constrained decoding** — masking invalid tokens at each step so output follows a grammar or schema. [18.7]

**Context engineering** — choosing what information goes into the context window at each step of an LLM application. [25.4]

**Context window** — maximum number of tokens the model can attend over in one sequence. [Notation]

**Continuous batching** — scheduling requests at the level of single decode steps, so finished sequences are replaced immediately. [19.4]

**Cross-entropy** — $H(p,q) = -\E_p\log q$; the LLM training loss with $p$ the data and $q$ the model. [4.4]

## D

**Data parallelism (DDP)** — replicating the model on each GPU and averaging gradients. [16.5]

**Decode** — the generation phase, producing one token per step per sequence; memory-bandwidth-bound. [19.1]

**Distillation** — training a smaller student model to imitate a larger teacher. [23.9]

**DPO (direct preference optimization)** — a supervised loss on preference pairs that solves the KL-regularized RLHF objective without RL. [21.7]

## E

**Effort** — an API control over how much a model reasons and how many tokens it spends. [28.5]

**Embedding** — a learned vector representing a token (input embedding) or a whole text (embedding model). [11.1, 26.2]

**Entropy** — expected information content, $H(p) = -\sum p\log p$. [4.2]

**Epoch** — one pass over the training data.

**Evaluation (eval)** — a test set plus grader used to measure a model or application. [29]

## F

**Fine-tuning** — any further training of a pretrained model (SFT, LoRA, DPO, RL are kinds of fine-tuning). [23]

**FlashAttention** — exact attention computed tile by tile in on-chip memory with an online softmax, never materializing the $T\times T$ matrix. [19.3]

**FLOP** — one floating-point operation; a multiply-add counts as 2. [Notation]

**FSDP / ZeRO** — sharding parameters, gradients and optimizer states across data-parallel GPUs. [16.6]

**Function calling / tool use** — the model emitting structured calls that application code executes. [27.1]

## G

**GELU** — $z\Phi(z)$, the activation in GPT-2 and BERT. [6.3]

**Gradient** — the vector of partial derivatives of a scalar, with the same shape as the variable. [2.3]

**Gradient accumulation** — summing gradients over several micro-batches before one optimizer step. [14.6]

**Gradient clipping** — rescaling the gradient when its global norm exceeds a threshold. [5.8]

**GQA (grouped-query attention)** — several query heads share one key/value head, shrinking the KV cache. [19.2]

**Greedy decoding** — always choosing the most probable next token. [18.2]

**GRPO** — PPO-style RL with group-normalized rewards as advantages and no value model. [22.4]

## H

**Hallucination** — fluent output that is false or unsupported by the provided sources.

**Hidden state** — the vector at a position in an intermediate layer; in a Transformer, the residual stream. [13.5]

## I

**Induction head** — an attention head that completes patterns "…A B … A → B" by copying. [12.7]

**Inference** — running a trained model to produce outputs. [Notation]

**In-context learning** — learning a task from examples in the prompt, without weight updates. [12.7, 25.1]

## K

**KL divergence** — $\KL(p\Vert q) = \E_p\log\frac pq\ge0$; extra cost of using $q$ when the truth is $p$. [4.5]

**KV cache** — stored keys and values of previous tokens, reused at every decode step. [19.2]

## L

**LayerNorm** — normalizing each vector to zero mean and unit variance over its features, with learned gain and bias. [8.3]

**Learning rate** — the step size $\eta$ of the optimizer. [5]

**LLM-as-judge** — using a language model to grade outputs against a rubric. [29.3]

**Logit lens** — decoding an intermediate residual stream with the final norm and unembedding. [13.5]

**Logits** — the model's raw output scores $\mathbf{z}\in\R^V$ before softmax. [Notation]

**LoRA** — fine-tuning a frozen weight with a trainable low-rank update $\frac\alpha r\mathbf{B}\mathbf{A}$. [23.4]

**Loss** — the scalar objective being minimized, here mostly next-token cross-entropy. [4.6]

## M

**MCP (Model Context Protocol)** — an open protocol for exposing tools, resources and prompts to LLM applications. [27.5]

**MFU (model FLOPs utilization)** — useful model FLOPs per second divided by the hardware's peak. [15.5]

**Mixed precision** — computing in bf16/fp16 while keeping fp32 master weights and accumulations. [16.2]

**MLA (multi-head latent attention)** — caching a compressed latent per token instead of full keys and values. [20.3]

**MLE (maximum likelihood estimation)** — choosing parameters that maximize the probability of the data. [3.7]

**MLP / FFN** — a stack of linear layers and nonlinearities; in a Transformer, the per-position feed-forward network. [6, 13.2]

**MoE (mixture of experts)** — a layer of many expert FFNs, of which a router selects a few per token. [20.2]

## N

**Nucleus (top-p) sampling** — sampling from the smallest set of tokens with total probability ≥ $p$. [18.4]

## O

**Online softmax** — computing a softmax-weighted sum in blocks with a running max and rescaled running sum. [19.3]

## P

**Perplexity** — $e^{\text{loss}}$; the effective number of equally likely choices per token. [4.7]

**Pipeline parallelism** — splitting the model by layers across devices, with micro-batches flowing through stages. [16.9]

**Policy** — in RL, the model viewed as a distribution over actions (tokens or responses), $\pi_\theta(y\mid x)$. [21.4]

**Post-training** — everything after pretraining: SFT, preference optimization, RL. [21]

**PPO** — policy optimization with a clipped probability ratio that keeps updates near the sampling policy. [21.6]

**Prefill** — processing the prompt's tokens in parallel to build the KV cache; compute-bound. [19.1]

**Pre-norm** — placing normalization at the input of each sub-layer, keeping the residual path an identity. [8.4]

**Pretraining** — next-token prediction on a large unlabeled corpus. [15]

**Prompt caching** — reusing the KV cache of a prompt prefix shared with a recent request. [19.4, 28.8]

**Prompt injection** — instructions embedded in untrusted content that try to hijack the model. [25.5]

## Q

**Quantization** — storing weights (or activations, KV) in fewer bits. [19.5]

## R

**RAG (retrieval-augmented generation)** — retrieving relevant documents at query time and answering from them. [26]

**Reasoning model** — a model trained (usually with RL) to produce long chains of thought before answering. [22]

**Reranker** — a cross-encoder that scores query–document pairs jointly to reorder retrieval candidates. [26.5]

**Residual stream** — the $d$-dimensional vector per position that every block reads from and adds to. [8.4, 13.5]

**Reward model** — a model trained on human or AI comparisons to score responses. [21.3]

**Reward hacking** — exploiting flaws in a reward signal to score well without doing the intended task. [21.9]

**RLHF** — reinforcement learning from human feedback: optimize a reward model's score with a KL penalty to a reference. [21.4]

**RLVR** — RL with verifiable rewards from automatic checkers (answers, unit tests). [22.3]

**RMSNorm** — normalization by the root mean square only, without centering or bias. [8.3]

**RoPE** — rotary position embedding: rotating query and key pairs by position-dependent angles. [11.5]

## S

**Scaling laws** — empirical power laws relating loss to parameters, data and compute. [17]

**Self-attention** — attention where queries, keys and values all come from the same sequence. [12]

**Self-consistency** — sampling several reasoning chains and taking a majority vote on the answer. [18.8]

**SFT (supervised fine-tuning)** — training on (prompt, ideal response) pairs with loss on the response tokens. [21.2]

**Softmax** — $\softmax(\mathbf{z})_i = e^{z_i}/\sum_je^{z_j}$; turns logits into probabilities. [2.5]

**Sparse autoencoder (SAE)** — a wide, sparsity-penalized autoencoder that extracts interpretable features from activations. [24.4]

**Speculative decoding** — drafting tokens with a fast model and verifying them in parallel with the target model, exactly preserving its distribution. [19.6]

**Structured outputs** — responses guaranteed to match a JSON schema through constrained decoding. [18.7, 28.7]

**Superposition** — representing more features than dimensions as nearly orthogonal directions. [24.3]

**SwiGLU** — a gated FFN, $\mathbf{W}_2(\operatorname{SiLU}(\mathbf{W}_1\mathbf{x})\odot\mathbf{W}_3\mathbf{x})$. [13.2]

**System prompt** — the operator's standing instructions at the start of a conversation. [25.1]

## T

**Teacher forcing** — training on the true prefix at every position rather than on the model's own outputs. [9.9]

**Temperature** — dividing logits by $T$ before softmax: lower is sharper, higher is flatter. [18.3]

**Tensor parallelism** — splitting individual weight matrices across GPUs. [16.7]

**Test-time compute** — spending more inference computation (longer reasoning, more samples) to improve answers. [22.2]

**Token** — an element of the vocabulary; the unit the model reads and writes. [10]

**Tokenizer** — the algorithm mapping text to token IDs and back. [10]

**Top-k sampling** — sampling only among the $k$ most probable tokens. [18.4]

**Transformer** — the architecture of stacked attention and FFN blocks with residual connections and normalization. [13]

## V

**Vocabulary** — the set of tokens $\mathcal{V}$, of size $V$. [10.8]

## W

**Weight decay** — shrinking weights toward zero each step; a regularizer. [5.6, 8.5]

**Weight tying** — using the token embedding matrix as the output projection. [11.1]
