# 17. Scaling Laws

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 4](../math/information-theory.md) (loss as cross-entropy), [Ch. 7](../nn/backprop.md) ($C\approx6ND$), [Ch. 15](pretraining.md). **You will learn:** power laws and why they're straight lines on log-log plots; Kaplan et al.'s scaling laws; the Chinchilla result with a complete derivation of the compute-optimal model size and token count; why the two disagreed; the replication controversy around Chinchilla's fit; inference-aware "overtraining"; scaling with repeated data; emergent abilities and the debate about them; hyperparameter transfer with μP; and how labs use small runs to predict big ones.
    **Why it matters:** scaling laws turned LLM development from guesswork into engineering. They decide how big a model to train for a given budget, and how much data it needs. They explain why GPT-3 was "undertrained", and why Llama 3 8B was trained on 15T tokens on purpose.

!!! watch "Read"
    J. Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020) and J. Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Chinchilla, 2022), the two foundational papers. Karpathy's [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (1 h) gives the big-picture view of scaling. Stanford [CS336](https://stanford-cs336.github.io/) has two full lectures on scaling laws.

---

## 17.1 Power laws

A **power law** is $y = a x^{-\alpha}$. Taking logs, $\log y = \log a - \alpha\log x$: a straight line on log-log axes with slope $-\alpha$. Power laws are scale-free: multiplying $x$ by 10 always multiplies $y$ by the same factor $10^{-\alpha}$. They describe phenomena with no characteristic scale, and, empirically, the loss of neural language models.

## 17.2 Kaplan et al. (2020)

Training hundreds of Transformers from $10^3$ to $10^9$ non-embedding parameters, Kaplan et al. found that the test loss follows power laws in each resource when the others aren't the bottleneck:

$$
L(N) = \Big(\frac{N_c}{N}\Big)^{\alpha_N},\qquad L(D) = \Big(\frac{D_c}{D}\Big)^{\alpha_D},\qquad L(C_\text{min}) = \Big(\frac{C_c}{C_\text{min}}\Big)^{\alpha_C},
$$

with $\alpha_N\approx0.076$, $\alpha_D\approx0.095$, $\alpha_C\approx0.050$, holding smoothly over 6–7 orders of magnitude. Other findings that held up:

- **Shape matters little.** At fixed $N$, the loss barely depends on depth vs width or on the number of heads, over a wide range.
- **Bigger models are more sample-efficient**: they reach a given loss with fewer tokens.
- **Smooth, predictable improvement.** A doubling of compute gives a predictable reduction of loss. You can predict a large run from small ones.

Kaplan et al. concluded that, as compute grows, most of it should go into **parameters**: $N_\text{opt}\propto C^{0.73}$, with data growing only as $C^{0.27}$. GPT-3 (175B parameters, 300B tokens, 2020) followed this advice.

## 17.3 Chinchilla: compute-optimal training

Hoffmann et al. (2022) revisited the question with over 400 models from 70M to 16B parameters, trained on 5B to 500B tokens, **with the learning-rate schedule matched to each run's length**. Three different methods (minimum loss over training curves; IsoFLOP profiles; a parametric fit) all said the opposite of Kaplan: **parameters and data should grow equally**, $N_\text{opt}\propto C^{0.5}$ and $D_\text{opt}\propto C^{0.5}$, at about **20 tokens per parameter**.

They tested it directly: **Chinchilla** (70B parameters, 1.4T tokens) used the same compute as **Gopher** (280B, 300B tokens) and beat it on nearly every benchmark, while being 4× cheaper to run.

### The parametric model and its optimum

Chinchilla's third approach fits

$$
L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}.
$$

The terms have clear meanings: $E$ is the **irreducible loss**, the entropy of natural text (§4.6); $A/N^\alpha$ is the cost of a finite model, the error of the best model of size $N$ trained on infinite data; $B/D^\beta$ is the cost of finite data, the error of an optimizer that sees only $D$ tokens.

!!! theorem "Theorem 17.1 (Compute-optimal allocation)"
    Minimizing $L(N,D)$ subject to $C = 6ND$ gives

    $$
    N_\text{opt}(C) = G\left(\frac C6\right)^{\frac{\beta}{\alpha+\beta}},\qquad D_\text{opt}(C) = G^{-1}\left(\frac C6\right)^{\frac{\alpha}{\alpha+\beta}},\qquad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac1{\alpha+\beta}}.
    $$

    At the optimum the two reducible terms are in the fixed ratio $\dfrac{A/N^\alpha}{B/D^\beta} = \dfrac\beta\alpha$.

??? proof "Proof"
    Substitute $D = C/(6N)$, so we minimize $f(N) = A N^{-\alpha} + B(C/6)^{-\beta}N^{\beta}$ over $N>0$. It's convex in $\log N$ (a sum of exponentials of $\log N$), so the stationary point is the minimum:

    $$
    f'(N) = -\alpha AN^{-\alpha-1} + \beta B(C/6)^{-\beta}N^{\beta-1} = 0\iff \alpha AN^{-\alpha} = \beta B\Big(\frac{C}{6N}\Big)^{-\beta} = \beta BD^{-\beta}.
    $$

    That's the ratio statement. Solving for $N$: $N^{\alpha+\beta} = \frac{\alpha A}{\beta B}(C/6)^\beta$, so $N_\text{opt} = G(C/6)^{\beta/(\alpha+\beta)}$, and $D_\text{opt} = (C/6)/N_\text{opt} = G^{-1}(C/6)^{1-\beta/(\alpha+\beta)} = G^{-1}(C/6)^{\alpha/(\alpha+\beta)}$. $\blacksquare$

**Consequences.** If $\alpha\approx\beta$, both exponents are about $\frac12$: double the compute, and multiply both $N$ and $D$ by $\sqrt2$. The tokens-per-parameter ratio $D_\text{opt}/N_\text{opt} = G^{-2}(C/6)^{(\alpha-\beta)/(\alpha+\beta)}$ is then nearly constant across budgets.

<div class="widget" data-widget="chinchilla"></div>

!!! warning "The fit everyone quotes is inconsistent with the paper's headline"
    Hoffmann et al. report $E = 1.69$, $A = 406.4$, $B = 410.7$, $\alpha = 0.34$, $\beta = 0.28$. Plug these into Theorem 17.1 at Chinchilla's own budget ($C = 5.76\times10^{23}$) and you get $N_\text{opt}\approx32$B, $D_\text{opt}\approx3.0$T, about **93 tokens per parameter**, not 20. Besiroglu et al., [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102) (2024), reconstructed the data from the paper's figures and found the published fit was poorly converged. Their refit, $E = 1.82$, $A = 482$, $B = 2085$, $\alpha = 0.348$, $\beta = 0.366$, gives about **72B parameters and 1.33T tokens (≈18 tokens/parameter)** at that budget, consistent with Approaches 1 and 2 and with Chinchilla itself. Switch between the two fits in the widget above to see the difference. The "≈20 tokens per parameter" rule is right; the famous constants aren't.

### Why Kaplan and Chinchilla disagreed

Later analyses ([Pearce & Song, 2024](https://arxiv.org/abs/2406.12907); [Porian et al., 2024](https://arxiv.org/abs/2406.19146)) traced the discrepancy to:

1. **Counting parameters differently**: Kaplan excluded embedding parameters, which are a large share of small models.
2. **Learning-rate schedules**: Kaplan used one schedule length for runs of different lengths, so shorter runs were stopped before their learning rate had decayed, understating what small models could do with more data.
3. **Smaller scale**: Kaplan's models were mostly below 1B parameters, where these effects are largest.

## 17.4 Beyond compute-optimal: training for inference

Chinchilla-optimal minimizes **training** compute for a given loss. But a deployed model is used for inference many times: each generated token costs about $2N$ FLOPs (§7.6). If a model will serve $D_\text{inf}$ tokens over its lifetime, the total cost is about

$$
C_\text{total}\approx6ND_\text{train} + 2ND_\text{inf}.
$$

For a heavily used model, it pays to choose a **smaller $N$ trained on more tokens** than Chinchilla suggests. The loss is slightly worse for the training compute spent, but every inference call is cheaper ([Sardana et al., Beyond Chinchilla-Optimal, 2023](https://arxiv.org/abs/2401.00448)). This is why modern open models are heavily "overtrained":

| Model | Parameters | Tokens | Tokens / parameter |
|---|---|---|---|
| Chinchilla | 70B | 1.4T | 20 |
| Llama 2 7B | 7B | 2T | 286 |
| Llama 3 8B | 8B | 15T | ≈1,900 |
| Qwen3 0.6B | 0.6B | 36T | ≈60,000 |

The loss keeps improving with more tokens, just more slowly. Under the replication fit, an 8B model trained on 15T tokens is within a few hundredths of a nat of the compute-optimal model for the same training budget (about 36B on 3.4T), while being 4–5× cheaper to serve.

## 17.5 Running out of unique data

When unique data is limited, how valuable are repeated epochs? Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) (2023), fit an **effective data** model: with $U$ unique tokens repeated $R$ extra times,

$$
D' = U + U R^*\left(1-e^{-R/R^*}\right),\qquad R^*\approx15.4.
$$

For small $R$, $1-e^{-R/R^*}\approx R/R^*$, so $D'\approx U(1+R)$: repeats are nearly as good as new data. As $R$ grows, the value saturates at $U(1+R^*)$. Practically, **up to about 4 epochs is almost free; beyond about 16 epochs, extra repeats are worth very little.**

## 17.6 Emergent abilities, and whether they are real

Wei et al., [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) (2022), documented tasks where performance is near random for small models, then jumps sharply above some scale: multi-digit arithmetic, some reasoning benchmarks, in-context learning of new tasks.

Schaeffer et al., [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) (NeurIPS 2023), argued that many jumps are artifacts of **discontinuous metrics**. If the per-token accuracy $p$ improves smoothly with scale, the exact-match accuracy on a 10-token answer is $p^{10}$, which stays near 0 until $p$ is close to 1 and then shoots up. Measured with a continuous metric (per-token log-likelihood, edit distance), the same capability improves smoothly and predictably.

The balanced view: the underlying loss improves smoothly, but *usefulness* often has thresholds (an answer that's 90% correct may be worthless), so abilities can appear abruptly in practice even when they're predictable in principle.

## 17.7 Scaling hyperparameters: μP

The best learning rate depends on model width. With standard parameterization, the optimal LR shifts as you scale up, so it must be re-tuned at every size. **Maximal update parameterization (μP)** ([Yang et al., Tensor Programs V, 2022](https://arxiv.org/abs/2203.03466)) rescales initialization and per-layer learning rates with width so that each layer's update stays the same size as width grows. Then the optimal hyperparameters **transfer**: tune on a small proxy model, and reuse them on the large one ("μTransfer"). They tuned a 40M-parameter proxy and transferred its hyperparameters to a 6.7B model, which outperformed the published GPT-3 6.7B results, with tuning costing only about 7% of one pretraining run. Other labs fit power laws for the optimal learning rate and batch size as functions of compute (e.g. [DeepSeek LLM, 2024](https://arxiv.org/abs/2401.02954)).

## 17.8 Scaling laws in practice

How a lab plans a large run:

1. **Train a ladder of small models** across 2–3 orders of magnitude of compute (e.g. $10^{18}$–$10^{21}$ FLOPs), each with tuned hyperparameters and a schedule matched to its length.
2. **IsoFLOP analysis**: for several fixed budgets, vary $N$ (and hence $D = C/6N$) and find the loss-minimizing size. The minima trace out $N_\text{opt}(C)$.
3. **Fit** $L(N,D)$ and extrapolate to the target budget. Llama 3 used this to choose 405B parameters for $3.8\times10^{25}$ FLOPs, and also predicted downstream benchmark accuracy from loss with a second fit.
4. **Validate** with a medium-sized run before committing the full budget.

!!! llm "The new axis: test-time compute"
    Since 2024, a second scaling axis has opened. Reasoning models (OpenAI o1, DeepSeek-R1, Claude with extended thinking) improve smoothly as they're allowed to generate more tokens of reasoning before answering, and as more RL compute is spent teaching them to reason ([OpenAI, 2024](https://openai.com/index/learning-to-reason-with-llms/); [Snell et al., 2024](https://arxiv.org/abs/2408.03314)). Accuracy grows roughly log-linearly with thinking tokens. Ch. 22 covers this.

## Key takeaways

- LLM loss follows power laws in parameters, data and compute: straight lines on log-log plots, smooth over many orders of magnitude.
- Chinchilla: $L = E + A/N^\alpha + B/D^\beta$. Minimizing it under $C = 6ND$ gives $N_\text{opt}\propto C^{\beta/(\alpha+\beta)}$, $D_\text{opt}\propto C^{\alpha/(\alpha+\beta)}$, roughly $C^{0.5}$ each, about **20 tokens per parameter**.
- The published Chinchilla constants are inconsistent with that headline; the 2024 replication fit is consistent.
- Kaplan's "favor parameters" advice came from untuned schedules and parameter-counting choices.
- Models that will serve many tokens are deliberately overtrained (Llama 3 8B: ~1,900 tokens/parameter).
- Repeated data is nearly free up to ~4 epochs. "Emergence" is often a metric artifact. μP transfers hyperparameters across widths.

## Exercises

1. Under $L(N) = (N_c/N)^{0.076}$, by what factor must $N$ grow to cut the reducible loss in half?
2. Derive Theorem 17.1 with a Lagrange multiplier instead of substitution.
3. Using the replication fit, compute $N_\text{opt}$, $D_\text{opt}$ and the predicted loss for $C = 10^{24}$ FLOPs.
4. A model will be trained once and then serve $10^{13}$ tokens. Compare the total compute (training + inference) of a 70B model trained on 1.4T tokens and an 8B model trained on 15T tokens.
5. Using the data-constrained formula, what is the effective data for $U = 1$T unique tokens repeated for 4 epochs in total ($R = 3$)? For 40 epochs ($R = 39$)?
6. Show that if per-token accuracy improves smoothly as $p(N) = 1-cN^{-\gamma}$, then exact-match accuracy on a $k$-token answer, $p^k$, can look like a sudden jump when plotted against $\log N$.

??? success "Solutions"
    1. $(1/2) = (N_1/N_2)^{0.076}$, so $N_2/N_1 = 2^{1/0.076}\approx9{,}100$.
    2. Minimize $A N^{-\alpha}+BD^{-\beta}+\lambda(6ND-C)$: $\alpha AN^{-\alpha-1} = 6\lambda D$ and $\beta BD^{-\beta-1} = 6\lambda N$. Dividing, $\frac{\alpha AN^{-\alpha}}{\beta BD^{-\beta}} = 1$, the same ratio condition. Then substitute $D = C/(6N)$.
    3. $N\approx96$B, $D\approx1.74$T, $L\approx1.96$.
    4. 70B: $6\cdot70\text{e}9\cdot1.4\text{e}12+2\cdot70\text{e}9\cdot10^{13} = 5.9\text{e}23+1.4\text{e}24\approx2.0\times10^{24}$. 8B: $6\cdot8\text{e}9\cdot15\text{e}12+2\cdot8\text{e}9\cdot10^{13} = 7.2\text{e}23+1.6\text{e}23\approx8.8\times10^{23}$. The overtrained small model costs less than half in total (with somewhat lower quality).
    5. $R = 3$: $D' = 1+15.4(1-e^{-3/15.4})\approx1+2.73 = 3.73$T (vs 4T if all were new). $R = 39$: $1+15.4(1-e^{-2.53})\approx1+14.2 = 15.2$T (vs 40T).
    6. $p^k = (1-cN^{-\gamma})^k\approx e^{-kcN^{-\gamma}}$. As a function of $\log N$ it's a double exponential: near 0 for small $N$, then rising steeply over a narrow range of $\log N$, more sharply the larger $k$ is.

## References

- J. Kaplan et al., [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (2020).
- J. Hoffmann et al., [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (NeurIPS 2022).
- T. Besiroglu, E. Erdil, M. Barnett, J. You, [Chinchilla Scaling: A replication attempt](https://arxiv.org/abs/2404.10102) (2024).
- T. Porian et al., [Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://arxiv.org/abs/2406.19146) (NeurIPS 2024).
- N. Sardana et al., [Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) (ICML 2024).
- N. Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) (NeurIPS 2023).
- J. Wei et al., [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) (TMLR 2022); R. Schaeffer, B. Miranda, S. Koyejo, [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) (NeurIPS 2023).
- G. Yang et al., [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) (2022).
