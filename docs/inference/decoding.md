---
seo_title: "LLM Decoding & Sampling: Temperature, Top-k, Top-p"
description: "Greedy and beam search, why the most likely text is bad text, temperature, top-k, top-p and min-p sampling with code, repetition penalties, constrained decoding for JSON, and best-of-n."
---

# 18. Decoding & Sampling

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 3](../math/probability.md) (sampling, Gumbel-max, law of large numbers), [Ch. 4](../math/information-theory.md) (entropy), [Ch. 14](../transformer/gpt.md). **You will learn:** the generation loop; greedy and beam search, and why the most likely text is bad text (with an information-theoretic explanation via the typical set); temperature, top-k, top-p (nucleus) and min-p sampling, implemented and tested; repetition and frequency penalties; why "temperature 0" is often not deterministic; constrained decoding for JSON and grammars; best-of-n and self-consistency; and practical settings.
    **Why it matters:** the same model can produce brilliant or degenerate text depending on how you pick tokens from its distribution. Decoding is the one part of an LLM you control at inference time without retraining.

!!! watch "Read & watch"
    A. Holtzman et al., [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (ICLR 2020), the paper behind nucleus sampling. Hugging Face, [How to generate text](https://huggingface.co/blog/how-to-generate). Karpathy's [Deep Dive into LLMs](https://www.youtube.com/watch?v=7xTGNNLPyMI) covers inference as sampling ("the model is a token tumbler").

---

## 18.1 The generation loop

Given a prompt $x_{1:t}$, generation repeats:

1. Run the model on the current sequence to get logits $\mathbf{z}\in\R^V$ for the next position.
2. Turn the logits into a choice of next token (this chapter).
3. Append it; stop if it is an end-of-sequence token, a stop string, or the length limit.

The first model call processes the whole prompt in parallel (**prefill**); each later call processes one new token (**decode**). Ch. 19 makes this efficient with a KV cache. Here we focus on step 2.

## 18.2 Greedy and beam search

**Greedy decoding** picks $\argmax_i z_i$ at each step. It's deterministic and cheap. **Beam search** keeps the $k$ highest-probability partial sequences ("beams"), extends each by every token, and keeps the best $k$ of the $kV$ candidates, by total log-probability $\sum_t\log p(x_t\mid x_{<t})$ (usually divided by length$^\alpha$, since otherwise short outputs win). It approximates the most likely *sequence*, which greedy doesn't find: a high-probability first token can lead to a low-probability continuation.

Beam search works well when there's essentially one right answer (translation, speech recognition). For open-ended generation, both greedy and beam search produce text that's **bland and repetitive**, and often falls into loops ("I don't know. I don't know. I don't know.").

### Why the most likely text is atypical

!!! theorem "Theorem 18.1 (Asymptotic equipartition, i.i.d. case)"
    Let $X_1,\dots,X_n$ be i.i.d. from $p$ with entropy $H$. Then $-\frac1n\log p(X_1,\dots,X_n)\to H$ in probability. So, with high probability, a sample has probability close to $e^{-nH}$.

??? proof "Proof"
    $-\frac1n\log p(X_{1:n}) = \frac1n\sum_{i=1}^n\big(-\log p(X_i)\big)$, an average of $n$ i.i.d. variables with mean $\E[-\log p(X)] = H$. Apply the weak law of large numbers (Thm 3.6). $\blacksquare$

    The same holds for stationary ergodic sources such as language, where $H$ becomes the entropy rate (Shannon–McMillan–Breiman theorem).

So nearly all the probability mass of long texts sits on a huge **typical set** of sequences, each with probability about $e^{-nH}$. The single most likely sequence has much higher probability than that, but it's one sequence, and it's not typical: it consists of the most predictable token at every step, which is exactly what makes it dull. Human text is typical: its per-token surprise hovers around the entropy, with occasional surprising words. Holtzman et al. showed that human text has much lower per-token probability under a language model than beam-search text does. Good decoding should sample from the typical region, not climb to the mode. ([Meister et al., 2023](https://arxiv.org/abs/2202.00666) build a "typical sampling" method directly on this idea.)

## 18.3 Temperature

**Sampling** draws $x\sim\softmax(\mathbf{z}/T)$. The temperature $T$ rescales the logits:

$$
p_i(T) = \frac{e^{z_i/T}}{\sum_je^{z_j/T}}.
$$

- $T = 1$: the model's own distribution.
- $T<1$: sharper. Probability concentrates on the top tokens; $T\to0$ gives greedy decoding.
- $T>1$: flatter; $T\to\infty$ gives uniform sampling.

Entropy is monotone in $T$ (Exercise 4.5). Ratios of probabilities are powers: $p_i/p_j = e^{(z_i-z_j)/T}$. Halving the temperature squares every probability ratio.

## 18.4 Truncation: top-k, top-p, min-p

Even at $T = 1$, pure sampling has a problem: the **long tail**. Each of the 100,000 unlikely tokens has a tiny probability, but together they may hold 5–10% of the mass, so every few dozen tokens you sample something bizarre, and the error compounds because the model then conditions on it. Truncation methods zero out the tail and renormalize.

- **Top-k** ([Fan et al., 2018](https://arxiv.org/abs/1805.04833)): keep the $k$ most likely tokens. The problem is that the right $k$ varies: after "The capital of France is", one token deserves the mass; after "My favorite food is", hundreds do.
- **Top-p / nucleus** ([Holtzman et al., 2020](https://arxiv.org/abs/1904.09751)): keep the smallest set of top tokens whose total probability is at least $p$ (e.g. 0.9). The set adapts: small when the model is confident, large when it isn't.
- **Min-p** ([Nguyen et al., 2024](https://arxiv.org/abs/2407.01082)): keep tokens with probability at least $p_\text{min}\cdot\max_jp_j$. The threshold scales with the model's confidence, which works well with high temperatures.

They're applied in sequence, as in this implementation (each function is tested in `test_all.py`):

```python
--8<-- "code/sampling.py:sampling"
```

Note the subtle details in `top_p_filter`: the cumulative sum is shifted right by one so that the token that *crosses* $p$ is kept, and the top token is always kept so that the set is never empty.

<div class="widget" data-widget="sampling"></div>

!!! warning "Order matters"
    Temperature applied *before* top-p changes which tokens are in the nucleus, since a higher $T$ spreads the mass and enlarges the set. Most libraries (Hugging Face, vLLM) apply temperature first, then the truncations, but some providers differ. Check before porting settings between systems.

## 18.5 Repetition penalties

Models, especially small ones at low temperature, fall into repetition loops. Common fixes, applied to the logits before sampling:

- **Frequency and presence penalties** (OpenAI style): with $c_i$ the number of times token $i$ has appeared so far, $z_i\leftarrow z_i-\alpha_\text{freq}\,c_i-\alpha_\text{pres}\,\mathbb{1}[c_i>0]$.
- **Repetition penalty** ([Keskar et al., CTRL, 2019](https://arxiv.org/abs/1909.05858)): for tokens already generated, divide a positive logit by $\theta>1$ (multiply a negative one by $\theta$).
- **No-repeat n-gram**: forbid any n-gram from appearing twice.

These are blunt instruments. They also penalize legitimate repetition (a variable name in code, a person's name in a story). Modern, well-trained chat models rarely need them.

## 18.6 Randomness, seeds, and why "temperature 0" isn't deterministic

Sampling with a fixed random seed is reproducible on one machine. The Gumbel-max trick (Thm 3.10) shows another way to see it: sampling is $\argmax_i(z_i/T+g_i)$ with fixed Gumbel noise $g_i$, so fixing the noise makes the choice a deterministic function of the logits.

But even greedy decoding in a production API is often **not** bit-for-bit reproducible. Floating-point addition isn't associative: $(a+b)+c\ne a+(b+c)$ in finite precision. Kernels for matmuls and attention use different reduction orders depending on the **batch size** and the other requests batched with yours, so the logits differ in the last bits, and when two tokens are nearly tied, the argmax can flip. After one flip, the continuations diverge. [He et al. (Thinking Machines Lab), 2025, "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) traces this to a lack of *batch invariance* in kernels and shows how to fix it, at some cost in speed. For applications: don't rely on exact reproducibility; design evaluations that tolerate variation (Ch. 29).

## 18.7 Constrained decoding and structured output

Often the output must follow a format: valid JSON matching a schema, one of a fixed set of labels, a function call, SQL. **Constrained decoding** enforces this exactly by **masking** the logits of every token that would make the output invalid, at every step.

- A regular expression or JSON schema compiles to a **finite-state machine** over characters. Precompute, for each FSM state, which vocabulary tokens are allowed (a token is allowed if its characters can be consumed from that state). Decoding then costs one table lookup per step ([Willard & Louf, Outlines, 2023](https://arxiv.org/abs/2307.09702)).
- Context-free grammars (nested JSON, code) need a pushdown automaton. Libraries such as [XGrammar](https://arxiv.org/abs/2411.15100) and llama.cpp's GBNF grammars make this fast.
- **Logit bias**: add a fixed offset to chosen tokens' logits (e.g. +100 to force, −100 to forbid). It's the simplest form of constraint.

Hosted APIs expose this as "structured outputs" or JSON-schema modes, and as tool/function calling (Ch. 27–28). A caveat: forcing a format can hurt quality if the model "wanted" to say something the format doesn't allow, so give it a place in the schema for reasoning or uncertainty when that matters.

## 18.8 Spending more samples: best-of-n and self-consistency

Sampling gives a *distribution* of answers, and you can use more than one:

- **Best-of-n**: sample $n$ answers and pick the best by a scorer: a reward model, unit tests for code, a verifier model, or the model's own log-probability.
- **Self-consistency** ([Wang et al., 2023](https://arxiv.org/abs/2203.11171)): for problems with a short final answer (math), sample $n$ reasoning chains at $T\approx0.7$–1 and take the **majority vote** of the final answers. If each chain is right with probability $q>\frac12$ and errors are spread over many wrong answers, the majority is right with probability approaching 1 as $n$ grows. On GSM8K it improved accuracy by double digits over greedy decoding.

These are the simplest forms of **test-time compute scaling** (Ch. 22): spend more inference compute to get better answers from the same model.

## 18.9 Log-probabilities as a tool

Many APIs can return the log-probability of each generated token (and the top alternatives). Uses:

- **Classification with calibrated confidence**: ask for a one-token label and read $p(\text{"yes"})$ vs $p(\text{"no"})$ instead of the sampled word.
- **Uncertainty detection**: low-probability tokens in factual answers often mark guesses, which can be used to trigger retrieval or a disclaimer.
- **Scoring** candidate texts (e.g. multiple-choice evaluation, §15.8).

Note that post-trained models are less calibrated than base models (§4.10), so treat these probabilities as signals rather than exact confidences.

## 18.10 Practical settings

| Task | Typical settings |
|---|---|
| Extraction, classification, code edits, tool calls | low temperature (0–0.3), or structured output |
| General chat and writing | $T\approx0.7$–1.0 with top-p 0.9–0.95 (or provider defaults) |
| Brainstorming, fiction, diverse synthetic data | $T\approx1.0$–1.2, maybe with min-p |
| Reasoning models | usually the provider's fixed or recommended settings, often $T = 1$; they're trained to reason with sampling on |
| Self-consistency | $T\approx0.7$–1.0, $n = 5$–40 samples, majority vote |

Change one setting at a time, and evaluate on a fixed test set (Ch. 29), not on a single impressive example.

!!! llm "Not every API exposes these knobs"
    The settings above apply to open models (Hugging Face, vLLM, llama.cpp) and to APIs that expose sampling parameters. Some hosted models don't: current Claude models, for example, reject `temperature`, `top_p` and `top_k`, and you steer them with the prompt, an effort level and structured outputs instead (Ch. 28). The theory in this chapter still explains what happens inside: the provider chooses the decoding strategy for you.

## Key takeaways

- Greedy and beam search approximate the most likely sequence, which is atypical, so open-ended text becomes bland and repetitive. The typical set explains why.
- Temperature rescales logits: $T\to0$ greedy, $T = 1$ the model's distribution, larger $T$ flatter.
- Truncate the long tail: top-k (fixed size), top-p (fixed mass, adaptive size), min-p (threshold relative to the top token).
- Repetition penalties are blunt; strong models rarely need them.
- Floating-point non-associativity plus batch-dependent kernels make even greedy decoding non-deterministic in production.
- Constrained decoding masks invalid tokens with an automaton, guaranteeing valid JSON or grammar output. Best-of-n and self-consistency turn extra samples into accuracy.

## Exercises

1. For logits $(2, 1, 0)$, compute the probabilities at $T = 0.5$, $1$, $2$.
2. Give an example where greedy decoding doesn't find the most likely two-token sequence.
3. With probabilities $(0.5, 0.2, 0.1, 0.1, 0.05, 0.05)$, which tokens are kept by top-$k = 3$, top-$p = 0.75$, and min-$p = 0.15$?
4. Each self-consistency sample is independently correct with probability 0.6, and wrong answers never coincide (every wrong sample gives a different answer). What's the probability that the most common of 5 answers is the correct one? (Count a tie as a failure.)
5. Why can't you get valid JSON reliably just by instructing a model, and why does constrained decoding guarantee it?
6. **Code:** using `generate` from `gpt.py` with your trained Shakespeare model, generate 200 characters at $T = 0.2, 0.8, 1.5$ and with top-k = 5. Describe the differences.

??? success "Solutions"
    1. $T = 0.5$: logits $(4,2,0)$ → $(0.867, 0.117, 0.016)$. $T = 1$: $(0.665, 0.245, 0.090)$. $T = 2$: logits $(1, 0.5, 0)$ → $(0.506, 0.307, 0.186)$.
    2. $p(A) = 0.6$, $p(B) = 0.4$; after A, the best next token has probability 0.3; after B, 0.9. Greedy picks A then gets $0.6\cdot0.3 = 0.18$, but B followed by its best has $0.4\cdot0.9 = 0.36$.
    3. Top-3: the first three. Top-0.75: cumulative $0.5, 0.7, 0.8$, so the first three (the third crosses 0.75). Min-p 0.15: threshold $0.15\cdot0.5 = 0.075$, so the first four (0.1 ≥ 0.075).
    4. Under these assumptions, the vote is right whenever at least 2 of 5 samples are correct: $1-0.4^5-5\cdot0.6\cdot0.4^4 = 1-0.0102-0.0768\approx0.913$.
    5. Instructions only make valid tokens *more likely*; any sampled step can still produce an invalid character. With masking, invalid tokens have probability exactly 0, so every completed output is valid by construction.
    6. Low $T$ gives repetitive loops of common words; $T = 0.8$ gives the most Shakespeare-like text; $T = 1.5$ gives misspelled nonsense; top-k = 5 removes the worst errors even at higher $T$.

## References

- A. Holtzman, J. Buys, L. Du, M. Forbes, Y. Choi, [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) (ICLR 2020).
- A. Fan, M. Lewis, Y. Dauphin, [Hierarchical Neural Story Generation](https://arxiv.org/abs/1805.04833) (ACL 2018).
- M. Nguyen et al., [Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs](https://arxiv.org/abs/2407.01082) (2024).
- C. Meister, T. Pimentel, G. Wiher, R. Cotterell, [Locally Typical Sampling](https://arxiv.org/abs/2202.00666) (TACL 2023).
- X. Wang et al., [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) (ICLR 2023).
- B. Willard & R. Louf, [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) (2023).
- H. He et al., [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) (Thinking Machines Lab, 2025).
- T. Cover & J. Thomas, *Elements of Information Theory*, Ch. 3 (AEP) and §16.8 (Shannon–McMillan–Breiman).
