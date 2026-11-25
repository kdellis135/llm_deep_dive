---
seo_title: "LLM Formula Sheet: Every Key Equation"
description: "Every key formula for large language models on one page, from softmax gradients and attention to Chinchilla, DPO and GRPO, each linked to its step-by-step derivation."
---

# Formula Sheet

!!! abstract "How to use this page"
    Every key result on the site, on one page, with a link to where it's derived. Review it after 1 day, 1 week and 1 month. For each line, try to re-derive it before clicking through.

## Part I · Math foundations

| Result | Formula | Where |
|---|---|---|
| Dot product, angle | $\mathbf{x}^\top\mathbf{y} = \lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert\cos\theta$ | [§1.2](../math/linear-algebra.md#12-the-dot-product-angles-and-similarity) |
| Cauchy–Schwarz | $\lvert\mathbf{x}^\top\mathbf{y}\rvert\le\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert$ | [Thm 1.2](../math/linear-algebra.md#12-the-dot-product-angles-and-similarity) |
| Matmul cost | $(m\times k)(k\times n)$: $2mkn$ FLOPs | [Fact 1.6](../math/linear-algebra.md#14-matrix-multiplication-is-composition) |
| Spectral theorem | $\mathbf{A} = \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$ for symmetric $\mathbf{A}$ | [Thm 1.10](../math/linear-algebra.md#18-eigenvalues-eigenvectors-and-the-spectral-theorem) |
| SVD, best rank-$k$ | $\mathbf{A} = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top$; $\mathbf{A}_k = \sum_{i\le k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ | [Thm 1.11–1.12](../math/linear-algebra.md#19-the-singular-value-decomposition-svd) |
| Random vectors near-orthogonal | $\cos\angle(\mathbf{x},\mathbf{y})\sim\mathcal{N}(0, 1/d)$ approx. | [Prop 1.14](../math/linear-algebra.md#112-high-dimensional-geometry-why-everything-is-nearly-orthogonal) |
| Chain rule (VJP) | $\nabla_\mathbf{x}\mathcal{L} = \mathbf{J}^\top\nabla_\mathbf{y}\mathcal{L}$ | [§2.4](../math/calculus.md#24-jacobians-and-the-multivariate-chain-rule) |
| Linear layer gradients | $\mathbf{Y} = \mathbf{X}\mathbf{W}^\top$: $\bar{\mathbf{X}} = \bar{\mathbf{Y}}\mathbf{W}$, $\bar{\mathbf{W}} = \bar{\mathbf{Y}}^\top\mathbf{X}$ | [Thm 2.6](../math/calculus.md#25-matrix-calculus-you-will-actually-use) |
| Softmax Jacobian | $\partial\mathbf{p}/\partial\mathbf{z} = \diag(\mathbf{p})-\mathbf{p}\mathbf{p}^\top$ | [Thm 2.7](../math/calculus.md#25-matrix-calculus-you-will-actually-use) |
| Softmax + CE gradient | $\nabla_\mathbf{z}(-\log p_y) = \mathbf{p}-\mathbf{y}$ | [Thm 2.8](../math/calculus.md#25-matrix-calculus-you-will-actually-use) |
| Sigmoid derivative | $\sigma' = \sigma(1-\sigma)$ | [Ex 2.2](../math/calculus.md#22-the-chain-rule) |
| Bayes | $p(\theta\mid\mathcal{D})\propto p(\mathcal{D}\mid\theta)\,p(\theta)$ | [§3.3](../math/probability.md#33-bayes-theorem) |
| Variance of a mean | $\Var(\bar X) = \sigma^2/n$ | [§3.6](../math/probability.md#36-law-of-large-numbers-clt-and-monte-carlo) |
| MLE = min NLL | $\hat\theta = \argmin_\theta-\frac1n\sum_i\log p_\theta(x^{(i)})$ | [§3.7](../math/probability.md#37-maximum-likelihood-estimation-mle) |
| Gaussian prior = L2 | MAP adds $\frac{\lambda}{2}\lVert\theta\rVert^2$ | [Prop 3.9](../math/probability.md#38-map-estimation-and-regularization) |
| Gumbel-max | $\argmax_i(z_i+g_i)\sim\softmax(\mathbf{z})$ | [Thm 3.10](../math/probability.md#39-sampling-from-a-categorical-and-the-gumbel-max-trick) |
| Entropy, bounds | $H(p) = -\sum p\log p$, $0\le H\le\log n$ | [§4.2](../math/information-theory.md#42-entropy) |
| Cross-entropy, KL | $H(p,q) = H(p)+\KL(p\Vert q)$, $\KL\ge0$ | [§4.4–4.5](../math/information-theory.md#45-kl-divergence) |
| CE loss = MLE = forward KL | $-\frac1n\sum\log q_\theta = H(\hat p)+\KL(\hat p\Vert q_\theta)$ | [Thm 4.9](../math/information-theory.md#46-cross-entropy-loss-maximum-likelihood-minimum-kl) |
| Perplexity, bits per byte | $\mathrm{PPL} = e^{\mathcal{L}}$; bpb $=\frac{\mathcal{L}}{\ln2}\cdot\frac{\#\text{tokens}}{\#\text{bytes}}$ | [§4.7](../math/information-theory.md#47-perplexity-and-bits-per-byte) |
| $k_3$ KL estimator | $(r-1)-\log r$, $r = \pi_\text{ref}/\pi_\theta$ | [Prop 4.10](../math/information-theory.md#49-kl-estimators-used-in-rlhf-and-grpo) |
| Descent lemma | $f(\theta-\eta\nabla f)\le f(\theta)-\frac\eta2\lVert\nabla f\rVert^2$ for $\eta\le1/L$ | [Lemma 5.1](../math/optimization.md#52-gradient-descent-and-the-descent-lemma) |
| GD on a quadratic | stable iff $\eta<2/\lambda_\max$; rate $\frac{\kappa-1}{\kappa+1}$ | [Thm 5.4](../math/optimization.md#53-conditioning-why-gd-zig-zags) |
| Momentum rate | $\frac{\sqrt\kappa-1}{\sqrt\kappa+1}$ | [Thm 5.6](../math/optimization.md#55-momentum) |
| Adam | $\mathbf{m}_t = \beta_1\mathbf{m}+(1-\beta_1)\mathbf{g}$, $\mathbf{v}_t = \beta_2\mathbf{v}+(1-\beta_2)\mathbf{g}^2$, $\theta\mathrel{-}=\eta\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\epsilon}$ | [§5.6](../math/optimization.md#56-adaptive-methods-from-adagrad-to-adam) |
| AdamW | $\theta\mathrel{-}=\eta\big(\frac{\hat{\mathbf{m}}}{\sqrt{\hat{\mathbf{v}}}+\epsilon}+\lambda\theta\big)$ | [§5.6](../math/optimization.md#adamw-decoupled-weight-decay) |

## Part II · Neural networks

| Result | Formula | Where |
|---|---|---|
| Logistic regression gradient | $\partial\mathcal{L}/\partial z = a-y$ | [Prop 6.1](../nn/mlp.md#61-one-neuron-linear-and-logistic-regression) |
| GELU, SiLU | $z\Phi(z)$; $z\sigma(z)$ | [§6.3](../nn/mlp.md#63-activation-functions) |
| Backprop | $\boldsymbol\delta^{(\ell)} = (\mathbf{W}^{(\ell+1)\top}\boldsymbol\delta^{(\ell+1)})\odot\phi'(\mathbf{z}^{(\ell)})$, $\nabla_{\mathbf{W}^{(\ell)}} = \boldsymbol\delta^{(\ell)}\mathbf{h}^{(\ell-1)\top}$ | [Thm 7.2](../nn/backprop.md#74-backprop-in-matrix-form) |
| Training FLOPs | $C\approx6ND$ (forward $2N$, backward $4N$ per token) | [Prop 7.4](../nn/backprop.md#76-the-cost-of-training-why-6n-flops-per-token) |
| LayerNorm backward | $\bar{\mathbf{x}} = r(\mathbf{g}-\bar g\mathbf{1}-\hat{\mathbf{x}}\,\overline{g\hat x})$ | [Prop 7.3](../nn/backprop.md#layernorm-backward) |
| Xavier / Kaiming init | $\sigma_w^2 = \frac{2}{n_\text{in}+n_\text{out}}$ / $\frac{2}{n_\text{in}}$ | [Thm 8.2](../nn/training.md#82-initialization-from-a-variance-argument) |
| Residual scaling | output projections $\times1/\sqrt{2L}$ | [Prop 8.3](../nn/training.md#gpt-2s-initialization) |
| RMSNorm | $\mathbf{y} = \boldsymbol\gamma\odot\mathbf{x}/\sqrt{\frac1d\sum x_i^2+\epsilon}$ | [Def 8.1](../nn/training.md#83-normalization-layers) |
| Inverted dropout | $\tilde h = h\cdot m/(1-p)$, $\E\tilde h = h$ | [§8.5](../nn/training.md#85-generalization-and-regularization) |
| Autoregressive factorization | $p(\mathbf{x}) = \prod_tp(x_t\mid x_{<t})$ | [Def 9.1](../nn/language-modeling.md#91-what-a-language-model-is) |
| Bigram MLE | $\hat P_{ab} = N_{ab}/\sum_{b'}N_{ab'}$ | [Prop 9.1](../nn/language-modeling.md#92-n-gram-models-counting-is-maximum-likelihood) |

## Part III · The Transformer

| Result | Formula | Where |
|---|---|---|
| UTF-8 | 1–4 bytes per code point | [§10.2](../transformer/tokenization.md#102-unicode-and-utf-8) |
| Sinusoidal positions | $p_{t,2i} = \sin(t\,10000^{-2i/d})$, $p_{t,2i+1} = \cos(\cdot)$ | [§11.3](../transformer/positions.md#113-absolute-positions) |
| RoPE | $(R_m\mathbf{q})^\top(R_n\mathbf{k}) = \mathbf{q}^\top R_{n-m}\mathbf{k}$, $\theta_i = b^{-2i/d_h}$ | [Thm 11.3](../transformer/positions.md#115-rotary-position-embedding-rope) |
| Attention | $\softmax(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_h}+\mathbf{M})\mathbf{V}$ | [Def 12.1](../transformer/attention.md#122-scaled-dot-product-attention) |
| Why $\sqrt{d_h}$ | $\Var(\mathbf{q}^\top\mathbf{k}) = d_h$ | [Prop 12.1](../transformer/attention.md#123-why-divide-by-sqrtd_h) |
| Attention backward | $\bar{\mathbf{V}} = \mathbf{A}^\top\bar{\mathbf{Y}}$, $\bar{\mathbf{S}} = \mathbf{A}\odot(\bar{\mathbf{A}}-\text{rowsum}(\bar{\mathbf{A}}\odot\mathbf{A}))$, $\bar{\mathbf{Q}} = \bar{\mathbf{S}}\mathbf{K}/\sqrt{d_h}$ | [§12.5](../transformer/attention.md#125-the-backward-pass-of-attention) |
| MHA parameters | $4d^2$ | [§12.6](../transformer/attention.md#126-multi-head-attention) |
| Block parameters | $12d^2+13d$; model $N\approx12Ld^2$ | [Prop 13.2](../transformer/block.md#133-counting-every-parameter-of-gpt-2-small) |
| SwiGLU | $\mathbf{W}_2(\operatorname{SiLU}(\mathbf{W}_1\mathbf{x})\odot\mathbf{W}_3\mathbf{x})$, $d_\text{ff} = \frac83d$ | [Prop 13.1](../transformer/block.md#gated-ffns-swiglu) |
| Forward FLOPs per token | $\approx2N+4LTd$ | [§13.4](../transformer/block.md#134-compute-and-memory-of-a-block) |
| GPT-2 small | 124,439,808 parameters | [§13.3](../transformer/block.md#133-counting-every-parameter-of-gpt-2-small) |

## Part IV · Training at scale

| Result | Formula | Where |
|---|---|---|
| MinHash | $\Pr[h(A) = h(B)] = J(A,B)$ | [Thm 15.1](../scale/pretraining.md#deduplication-with-minhash) |
| LSH banding | $\Pr[\text{candidate}] = 1-(1-s^r)^b$ | [§15.3](../scale/pretraining.md#deduplication-with-minhash) |
| MFU | $\frac{6N\cdot\text{tokens/s}}{\#\text{GPUs}\cdot\text{peak FLOP/s}}$ | [Def 15.1](../scale/pretraining.md#155-compute-accounting-and-mfu) |
| Training memory | 16 bytes/parameter (mixed-precision AdamW) + activations | [§16.1](../scale/distributed.md#161-the-memory-budget) |
| Matmul intensity | $n/3$ FLOPs/byte for $n\times n$ bf16 | [Prop 16.1](../scale/distributed.md#163-gpus-and-the-roofline-model) |
| Ring all-reduce | $2\frac{p-1}{p}n$ bytes per GPU | [Prop 16.2](../scale/distributed.md#164-collective-communication) |
| ZeRO-1/2/3 | $4\Psi+\frac{12\Psi}{N_d}$, $2\Psi+\frac{14\Psi}{N_d}$, $\frac{16\Psi}{N_d}$ | [§16.6](../scale/distributed.md#166-sharded-data-parallelism-zero-and-fsdp) |
| Pipeline bubble | $\frac{p-1}{m+p-1}$ | [Prop 16.4](../scale/distributed.md#169-pipeline-parallelism) |
| Chinchilla loss | $L = E+A/N^\alpha+B/D^\beta$ | [§17.3](../scale/scaling-laws.md#173-chinchilla-compute-optimal-training) |
| Compute-optimal | $N_\text{opt} = G(C/6)^{\frac{\beta}{\alpha+\beta}}$, $D_\text{opt} = G^{-1}(C/6)^{\frac{\alpha}{\alpha+\beta}}$, $G = (\frac{\alpha A}{\beta B})^{\frac1{\alpha+\beta}}$; ≈20 tokens/param | [Thm 17.1](../scale/scaling-laws.md#the-parametric-model-and-its-optimum) |
| Effective repeated data | $D' = U+UR^*(1-e^{-R/R^*})$, $R^*\approx15.4$ | [§17.5](../scale/scaling-laws.md#175-running-out-of-unique-data) |

## Part V · Inference

| Result | Formula | Where |
|---|---|---|
| Temperature | $p_i\propto e^{z_i/T}$ | [§18.3](../inference/decoding.md#183-temperature) |
| Typical set | $-\frac1n\log p(X_{1:n})\to H$ | [Thm 18.1](../inference/decoding.md#why-the-most-likely-text-is-atypical) |
| Decode step time | $\approx\frac{\text{bytes(weights + KV)}}{\text{HBM bandwidth}}$ | [§19.1](../inference/efficient.md#191-prefill-and-decode-are-different-workloads) |
| KV cache | $2LH_{kv}d_h\cdot\text{bytes}\cdot T\cdot B$ | [Prop 19.1](../inference/efficient.md#192-the-kv-cache) |
| Online softmax | $m' = \max(m,\max s)$, $\ell' = e^{m-m'}\ell+\sum e^{s-m'}$, $\mathbf{o}' = e^{m-m'}\mathbf{o}+\sum e^{s-m'}\mathbf{v}$ | [Thm 19.2](../inference/efficient.md#online-softmax) |
| Quantization error | $\lvert w-\hat w\rvert\le s/2$ | [Prop 19.4](../inference/efficient.md#195-quantization) |
| Speculative decoding | accept $\min(1,p/q)$, else sample $\propto\max(0,p-q)$; $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ tokens per call | [Thm 19.5, Prop 19.6](../inference/efficient.md#196-speculative-decoding) |

## Part VI · Frontier models

| Result | Formula | Where |
|---|---|---|
| MoE | $\sum_{e\in\text{top-}k}\frac{p_e}{\sum p}\operatorname{FFN}_e(\mathbf{x})$; balance loss $E\sum_ef_eP_e\ge1$ | [§20.2](../frontier/architectures.md#202-mixture-of-experts) |
| Bradley–Terry | $\Pr(y_w\succ y_l) = \sigma(r_w-r_l)$ | [§21.3](../frontier/post-training.md#213-reward-models-and-the-bradleyterry-model) |
| Optimal RLHF policy | $\pi^*\propto\pi_\text{ref}\exp(r/\beta)$ | [Thm 21.1](../frontier/post-training.md#214-the-rlhf-objective-and-its-optimal-policy) |
| Policy gradient | $\nabla\E[R] = \E[(R-b)\nabla\log\pi_\theta]$ | [Thm 21.2](../frontier/post-training.md#215-policy-gradients) |
| GAE | $A_t = \sum_l(\gamma\lambda)^l\delta_{t+l}$, $\delta_t = r_t+\gamma V_{t+1}-V_t$ | [§21.5](../frontier/post-training.md#from-sequences-to-tokens-values-and-gae) |
| PPO-clip | $\min(\rho A,\operatorname{clip}(\rho,1\pm\epsilon)A)$ | [§21.6](../frontier/post-training.md#216-ppo) |
| DPO | $-\log\sigma\big(\beta\log\frac{\pi_\theta(y_w)}{\pi_\text{ref}(y_w)}-\beta\log\frac{\pi_\theta(y_l)}{\pi_\text{ref}(y_l)}\big)$ | [§21.7](../frontier/post-training.md#217-direct-preference-optimization-dpo) |
| GRPO advantage | $\hat A_i = (r_i-\operatorname{mean}(\mathbf{r}))/\operatorname{std}(\mathbf{r})$ | [§22.4](../frontier/reasoning.md#224-grpo) |
| pass@k | $1-\binom{n-c}{k}/\binom nk$ | [§22.6](../frontier/reasoning.md#226-measuring-reasoning-passk) |
| LoRA | $\mathbf{W}_0+\frac\alpha r\mathbf{B}\mathbf{A}$, $r(d_\text{in}+d_\text{out})$ params | [§23.4](../frontier/fine-tuning.md#234-lora) |
| SAE | $\lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2+\lambda\sum_if_i\lVert\mathbf{d}_i\rVert$ | [§24.4](../frontier/interpretability.md#244-sparse-autoencoders-extracting-features) |
| Attribution patching | $\text{effect}\approx(h_\text{clean}-h_\text{corr})^\top\nabla_hM$ | [§24.5](../frontier/interpretability.md#245-circuits-and-causal-interventions) |

## Part VII · Applying LLMs

| Result | Formula | Where |
|---|---|---|
| InfoNCE | $-\log\frac{e^{s(q,d^+)/\tau}}{\sum_je^{s(q,d_j)/\tau}}$; $I\ge\log N-\mathcal{L}$ | [§26.2](../applied/rag.md#contrastive-training-the-infonce-loss) |
| BM25 | $\sum_t\operatorname{IDF}(t)\frac{f(k_1+1)}{f+k_1(1-b+b|D|/\text{avgdl})}$ | [§26.3](../applied/rag.md#263-lexical-retrieval-bm25) |
| RRF | $\sum_r\frac{1}{60+\operatorname{rank}_r(d)}$ | [§26.5](../applied/rag.md#265-hybrid-search-fusion-and-reranking) |
| nDCG | $\frac{1}{\text{IDCG}}\sum_i\frac{2^{\text{rel}_i}-1}{\log_2(i+1)}$ | [§26.8](../applied/rag.md#268-evaluating-rag) |
| Multi-step success | $p^n$; with retries $1-(1-p)^{r+1}$ per step | [Prop 27.1](../applied/agents.md#277-the-reliability-math-of-multi-step-tasks) |
| Accuracy SE | $\sqrt{p(1-p)/n}$ | [Prop 29.1](../applied/production.md#294-the-statistics-of-evals) |
| McNemar | $z = (b-c)/\sqrt{b+c}$ | [§29.4](../applied/production.md#294-the-statistics-of-evals) |
| Cohen's kappa | $\kappa = \frac{p_o-p_e}{1-p_e}$ | [§29.3](../applied/production.md#llm-as-judge) |
