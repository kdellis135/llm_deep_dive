---
seo_title: "Notation for Machine Learning & Transformers"
description: "One consistent set of symbols for linear algebra, calculus, probability, learning, Transformers, scaling and RL, plus the terminology used throughout LLM Deep Dive."
---

# Notation & Conventions

!!! abstract "In this page"
    One set of symbols for the whole site. When a chapter introduces a symbol, it uses the meaning given here. Bookmark this page.

## 1. Typography

| Object | Style | Examples |
|---|---|---|
| Scalar | italic lowercase / Greek | $a,\ x_i,\ \eta,\ \beta$ |
| Vector | **bold lowercase**, column by default | $\mathbf{x} \in \R^{n}$, $\mathbf{x} = (x_1,\dots,x_n)^\top$ |
| Matrix | **bold uppercase** | $\mathbf{W} \in \R^{m \times n}$, entry $W_{ij}$ |
| Set, space | calligraphic / blackboard | $\mathcal{V}$ (vocabulary), $\mathcal{D}$ (dataset), $\R$ |
| Random variable | uppercase italic | $X,\ Y$ (with values $x,\ y$) |
| Tensor in code | shape tuple in comments | `(B, T, C)` |

## 2. Linear algebra

| Symbol | Meaning |
|---|---|
| $\mathbf{x}^\top$ | transpose |
| $\mathbf{x}^\top\mathbf{y} = \langle \mathbf{x}, \mathbf{y}\rangle = \sum_i x_i y_i$ | dot (inner) product |
| $\lVert \mathbf{x} \rVert_2 = \sqrt{\mathbf{x}^\top\mathbf{x}}$ | Euclidean norm (we write $\lVert\cdot\rVert$ when the norm is $\ell_2$) |
| $\mathbf{A}\mathbf{B}$ | matrix product |
| $\odot$ | elementwise (Hadamard) product |
| $\mathbf{I}_n,\ \mathbf{0},\ \mathbf{1}$ | identity, zero vector/matrix, all-ones vector |
| $\tr(\mathbf{A}),\ \det(\mathbf{A}),\ \rank(\mathbf{A})$ | trace, determinant, rank |
| $\diag(\mathbf{x})$ | diagonal matrix with $\mathbf{x}$ on the diagonal |

!!! warning "Row vs column convention (important)"
    - **Single vectors are columns.** A linear layer on one input is $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ with $\mathbf{W} \in \R^{d_\text{out}\times d_\text{in}}$.
    - **Stacked sequences are rows.** A sequence of $T$ token vectors is a matrix $\mathbf{X} \in \R^{T \times d}$ whose row $t$ is $\mathbf{x}_t^\top$. A linear layer applied to every row is $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top + \mathbf{1}\mathbf{b}^\top$.
    - This matches PyTorch exactly: `nn.Linear(d_in, d_out).weight` has shape `(d_out, d_in)` and computes `x @ W.T + b`.
    - In the Transformer chapters we follow the original paper and write projections as $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$ with $\mathbf{W}_Q \in \R^{d\times d_h}$. That is the same thing with $\mathbf{W}_Q$ standing for the *transpose* of PyTorch's stored weight. We point this out wherever it matters.

## 3. Calculus

| Symbol | Meaning |
|---|---|
| $\dfrac{df}{dx},\ f'(x)$ | derivative of a scalar function of a scalar |
| $\dfrac{\partial f}{\partial x_i}$ | partial derivative |
| $\nabla_{\mathbf{x}} f \in \R^n$ | **gradient** of scalar $f$ w.r.t. $\mathbf{x}\in\R^n$, **same shape as $\mathbf{x}$** |
| $\mathbf{J} = \dfrac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \R^{m \times n}$ | **Jacobian** of $\mathbf{y}\in\R^m$ w.r.t. $\mathbf{x}\in\R^n$, $J_{ij} = \partial y_i/\partial x_j$ |
| $\mathbf{H} = \nabla^2 f$ | Hessian, $H_{ij} = \partial^2 f/\partial x_i \partial x_j$ |

The gradient of a scalar loss with respect to *any* tensor (vector, matrix, 4-D weight) always has **the same shape as that tensor**. This is the convention PyTorch's `.grad` follows. We call it the *gradient convention* and never deviate from it.

## 4. Probability & information

| Symbol | Meaning |
|---|---|
| $p(x)$, $p(x \mid y)$ | probability (mass or density), conditional probability |
| $X \sim p$ | $X$ is distributed according to $p$ |
| $\E_{x\sim p}[f(x)]$ | expectation |
| $\Var(X),\ \Cov(X,Y)$ | variance, covariance |
| $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | Gaussian with mean $\boldsymbol\mu$ and covariance $\boldsymbol\Sigma$ |
| $H(p)$ | entropy |
| $H(p, q)$ | cross-entropy |
| $\KL(p \,\Vert\, q)$ | Kullback–Leibler divergence |
| $\log$ | **natural log** (information in **nats**) unless written $\log_2$ (bits) |

## 5. Learning

| Symbol | Meaning |
|---|---|
| $\theta$ | all trainable parameters, flattened into one vector when convenient |
| $\mathcal{L}(\theta)$ | loss (to be minimized) |
| $\eta$ | learning rate |
| $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{n}$ | dataset; superscript $(i)$ indexes **examples** |
| $\sigma(z) = 1/(1+e^{-z})$ | logistic sigmoid |
| $\softmax(\mathbf{z})_i = e^{z_i}/\sum_j e^{z_j}$ | softmax |
| $\mathbf{z}$ | **logits** (unnormalized log-probabilities) |
| $\mathbf{p} = \softmax(\mathbf{z})$ | predicted probabilities |
| $\mathbf{y}$ | one-hot target vector (or $y$ = target class index) |

## 6. Language models & Transformers

| Symbol | Code name | Meaning | GPT-2 small |
|---|---|---|---|
| $\mathcal{V},\ V = \lvert\mathcal{V}\rvert$ | `vocab_size` | vocabulary, vocabulary size | 50,257 |
| $x_t \in \{0,\dots,V-1\}$ | `idx` | token ID at position $t$ | |
| $x_{<t} = (x_1,\dots,x_{t-1})$ | | prefix (context) before position $t$ | |
| $p_\theta(x_t \mid x_{<t})$ | | the model's next-token distribution | |
| $B$ | `B` | batch size (sequences per step) | |
| $T$ | `T`, `block_size` | sequence length / context window | 1,024 |
| $d$ (or $d_\text{model}$) | `C`, `n_embd`, `dim` | model width (residual stream size) | 768 |
| $L$ | `n_layer` | number of Transformer blocks | 12 |
| $H$ | `n_head` | number of attention heads | 12 |
| $d_h = d/H$ | `head_dim` | per-head dimension | 64 |
| $H_{kv}$ | `n_kv_heads` | key/value heads (GQA) | 12 |
| $d_\text{ff}$ | `ffn_dim` | feed-forward hidden size | 3,072 |
| $\mathbf{E} \in \R^{V\times d}$ | `wte` | token embedding matrix | |
| $\mathbf{h}_t^{(\ell)} \in \R^{d}$ | `x` | residual stream at position $t$ after block $\ell$ | |
| $\mathbf{Q},\mathbf{K},\mathbf{V} \in \R^{T\times d_h}$ | `q, k, v` | queries, keys, values (bold $\mathbf{V}$ is values; plain $V$ is vocabulary size) | |
| $\mathbf{A} \in \R^{T\times T}$ | `att` | attention weights, rows sum to 1 | |

## 7. Scale, RL and alignment

| Symbol | Meaning |
|---|---|
| $N$ | number of (non-embedding) parameters |
| $D$ | number of training tokens |
| $C \approx 6ND$ | training compute in FLOPs |
| $\pi_\theta(y \mid x)$ | the LM viewed as a **policy** generating response $y$ to prompt $x$ |
| $\pi_\text{ref}$ | frozen reference policy (usually the SFT model) |
| $r(x, y)$, $r_\phi$ | reward; learned reward model with parameters $\phi$ |
| $\beta$ | KL-penalty strength (RLHF, DPO) |
| $A_t$ | advantage |

## 8. Terminology used consistently

| Term | Meaning on this site |
|---|---|
| **token** | an element of the vocabulary; the unit the model reads and writes |
| **logits** | the model's raw output scores $\mathbf{z}\in\R^V$, before softmax |
| **context window** | maximum number of tokens $T$ the model can attend over |
| **residual stream** | the $d$-dimensional vector per position that every block reads from and adds to |
| **pretraining** | next-token prediction on a large unlabeled corpus |
| **post-training** | everything after pretraining: SFT, preference optimization, RL |
| **SFT** | supervised fine-tuning on (prompt, ideal response) pairs |
| **fine-tuning** | any further training of a pretrained model (SFT and LoRA are kinds of fine-tuning) |
| **inference** | running a trained model to produce outputs |
| **prefill / decode** | processing the prompt in parallel / generating one token per step |
| **FLOP** | one floating-point operation (multiply *or* add); a multiply-add is 2 FLOPs |

## 9. MathJax macros used on this site

For readability we define: `\R` → $\R$, `\E` → $\E$, `\softmax` → $\softmax$, `\argmax` → $\argmax$, `\KL` → $\KL$, `\tr`, `\diag`, `\rank`, `\Var`, `\Cov`.
