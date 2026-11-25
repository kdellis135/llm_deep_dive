---
seo_title: "Reasoning Models: Chain-of-Thought, RLVR and GRPO"
description: "Why chain-of-thought helps, test-time compute scaling, reinforcement learning with verifiable rewards, GRPO derived and implemented, the DeepSeek-R1 recipe, and the pass@k estimator."
---

# 22. Reasoning Models

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 18](../inference/decoding.md) (sampling, self-consistency), [Ch. 21](post-training.md) (policy gradients, PPO, KL). **You will learn:** chain-of-thought prompting and *why* it helps (more serial computation per answer); test-time compute scaling with verifiers and search; reinforcement learning with verifiable rewards (RLVR); GRPO, derived and implemented; the DeepSeek-R1 recipe, from R1-Zero to distillation; the unbiased pass@k estimator; what RL does and doesn't add to a base model; thinking budgets; and whether a model's chain of thought reflects its actual reasoning.
    **Why it matters:** since late 2024 the biggest capability gains have come from models that "think" before answering (OpenAI o1/o3, DeepSeek-R1, Claude with extended thinking, Gemini thinking models). This is a new scaling axis, and it's built almost entirely from the math of Ch. 21 applied to problems with checkable answers.

!!! watch "Read & watch"
    - DeepSeek-AI, [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (2025). The most detailed public account of training a reasoning model.
    - OpenAI, [Learning to Reason with LLMs](https://openai.com/index/learning-to-reason-with-llms/) (2024), the o1 announcement with its train-time and test-time scaling plots.
    - Karpathy's [Deep Dive into LLMs](https://www.youtube.com/watch?v=7xTGNNLPyMI) has an extended section on RL and "thinking" models.
    - N. Lambert, [*RLHF book*](https://rlhfbook.com/), chapters on reasoning and RLVR.

---

## 22.1 Chain of thought

Wei et al. ([2022](https://arxiv.org/abs/2201.11903)) found that few-shot examples which **show their reasoning steps** before the answer greatly improve large models on math and logic problems. Kojima et al. ([2022](https://arxiv.org/abs/2205.11916)) found that just appending "Let's think step by step." works too (zero-shot CoT).

**Why it helps: compute per answer.** A Transformer does a fixed amount of computation per token: about $2N$ FLOPs through $L$ layers. A question answered in one token gets one pass of $L$ layers of serial depth, however hard the question is. Writing out intermediate steps turns every generated token into **more serial computation**, and stores intermediate results in the context where later steps can attend to them, like scratch paper.

There's theory behind the intuition. A fixed-depth Transformer that answers immediately can only compute functions in a small circuit complexity class (roughly $\mathsf{TC}^0$: constant-depth threshold circuits), which is believed not to include problems like evaluating arbitrary compositions of permutations. With a chain of thought of polynomially many steps, Transformers can simulate any polynomial-time computation ([Merrill & Sabharwal, 2024](https://arxiv.org/abs/2310.07923); [Feng et al., 2023](https://arxiv.org/abs/2305.15408)). **Chain of thought makes a Transformer more powerful, not just more talkative.**

## 22.2 Test-time compute

If thinking tokens help, how should extra inference compute be spent?

- **Parallel sampling + aggregation**: self-consistency majority voting (§18.8); best-of-$n$ with a verifier.
- **Verifiers.** An **outcome reward model (ORM)** scores a final answer. A **process reward model (PRM)** scores each reasoning step. Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (2023), collected 800K human step labels (PRM800K) and found process supervision beat outcome supervision for selecting solutions to MATH problems.
- **Search**: beam search or tree search over reasoning steps guided by a PRM.
- **Sequential revision**: generate, critique, revise.
- **Long single chains**: train the model to think longer and to self-correct within one chain. That's what reasoning models do.

Snell et al., [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) (2024), found that the best strategy depends on difficulty: easy questions benefit from sequential revision, hard ones from broader search. With compute allocated per question, test-time compute could substitute for a model about 14× larger on problems where the smaller model had non-trivial success.

## 22.3 RL with verifiable rewards (RLVR)

For math, code, and many logic and science problems, the answer can be **checked automatically**:

- math: compare the final answer (e.g. inside `\boxed{}`) with the reference after normalization;
- code: run unit tests in a sandbox;
- formats: does the output follow the required structure (e.g. thinking inside `<think>` tags)?
- instruction-following constraints: "answer in exactly 3 bullet points" can be checked by a program.

A **verifiable reward** is then $r(x,y)\in\{0,1\}$ (plus small format terms). There's no learned reward model to overoptimize (§21.9). The model can still find loopholes: special-casing tests, editing test files, or printing an answer in a format the checker misparses. So verifiers and sandboxes must be robust, and reward hacking must be monitored.

## 22.4 GRPO

**Group Relative Policy Optimization** ([Shao et al., DeepSeekMath, 2024](https://arxiv.org/abs/2402.03300)) is PPO without the value model. For each prompt $x$, sample a **group** of $G$ responses $o_1,\dots,o_G$ (e.g. $G = 16$) from the current policy, score them $r_1,\dots,r_G$, and use the **group's own statistics** as the baseline:

$$
\hat A_i = \frac{r_i-\operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G)},
$$

assigned to every token of response $i$. The objective is the PPO clip (§21.6) with a per-token KL penalty to the reference model, using the unbiased, non-negative $k_3$ estimator (Prop. 4.10):

$$
\mathcal{J}_\text{GRPO}(\theta) = \E\Bigg[\frac1G\sum_{i=1}^G\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\Big(\min\big(\rho_{i,t}\hat A_i,\ \operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)-\beta\,\widehat{\KL}_{i,t}\Big)\Bigg],
$$

with $\rho_{i,t} = \frac{\pi_\theta(o_{i,t}\mid x,o_{i,<t})}{\pi_{\theta_\text{old}}(o_{i,t}\mid x,o_{i,<t})}$ and $\widehat{\KL}_{i,t} = \frac{\pi_\text{ref}}{\pi_\theta}-\log\frac{\pi_\text{ref}}{\pi_\theta}-1$ evaluated at token $o_{i,t}$.

```python
--8<-- "code/alignment.py:grpo"
```

Why it works well for verifiable rewards:

- **No value model**: half the memory of PPO, and no hard-to-train per-token value predictions for long chains of thought.
- **The baseline is the prompt's own difficulty.** On a hard prompt where only 2 of 16 samples are right, those 2 get large positive advantages. On an easy prompt where 15 of 16 are right, the one failure gets a large negative advantage. Groups where all samples get the same reward have zero advantage and contribute nothing, so useful training needs prompts of intermediate difficulty for the current model.

**Refinements.** DAPO ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)) raises the upper clip bound ("clip-higher", to keep exploring), filters out all-correct and all-wrong groups ("dynamic sampling"), and averages the loss over tokens rather than over sequences. Dr. GRPO ([Liu et al., 2025](https://arxiv.org/abs/2503.20783)) shows that the $1/|o_i|$ length normalization and the std division introduce biases (e.g. toward longer wrong answers) and removes them. Many recent recipes also drop the KL term entirely for reasoning RL.

## 22.5 The DeepSeek-R1 recipe

**R1-Zero: pure RL from a base model.** DeepSeek applied GRPO with only accuracy and format rewards directly to the DeepSeek-V3 base model, with a template asking it to think inside `<think>...</think>` and then answer. There was no SFT. Over training:

- AIME 2024 pass@1 rose from **15.6% to 71.0%** (86.7% with majority voting over 64 samples);
- the average response length grew steadily from hundreds to thousands of tokens. The model *learned on its own* to think longer;
- behaviors emerged without being demonstrated: re-checking work, trying alternative approaches, and the much-quoted "aha moment", where the model writes something like "Wait, wait. That's an aha moment I can flag here" and re-evaluates its approach.

But R1-Zero's outputs were hard to read, and it mixed languages within a chain of thought.

**R1: a four-stage pipeline.**

1. **Cold start**: SFT on a few thousand long, readable chain-of-thought examples.
2. **Reasoning RL** (GRPO) on math, code, science and logic, with an extra reward for staying in one language.
3. **Rejection sampling + SFT**: generate many responses with the RL model, keep correct and readable ones (about 600K reasoning samples), add about 200K non-reasoning samples (writing, factual QA, self-cognition), and fine-tune the *base* model on all of it.
4. **RL for all scenarios**: verifiable rewards for reasoning plus reward models for helpfulness and harmlessness on general prompts.

**Distillation.** Fine-tuning smaller open models (Qwen2.5 and Llama 3, 1.5B to 70B) on R1's 800K samples produced strong reasoning models; the 32B distilled model beat OpenAI's o1-mini on several benchmarks. Running large-scale RL directly on the small models did much worse than distilling from the large one. **Reasoning ability transfers well through SFT on traces from a strong reasoner.**

## 22.6 Measuring reasoning: pass@k

With a sampled model, accuracy is a distribution. **pass@k** is the probability that at least one of $k$ samples is correct. Estimating it naively from exactly $k$ samples is very noisy. Chen et al. ([Codex, 2021](https://arxiv.org/abs/2107.03374)) give an unbiased estimator from $n\ge k$ samples, of which $c$ are correct:

$$
\widehat{\text{pass@}k} = 1-\frac{\binom{n-c}{k}}{\binom nk}.
$$

??? proof "Why it's unbiased"
    Conditioned on the $n$ samples, $\binom{n-c}{k}/\binom nk$ is exactly the probability that a uniformly random subset of $k$ of them contains no correct sample. A random $k$-subset of $n$ i.i.d. samples is itself $k$ i.i.d. samples, so taking expectations over the $n$ samples gives $\E[\cdot] = \Pr[\text{all }k\text{ fresh samples wrong}] = 1-\text{pass@}k$. $\blacksquare$

**What does RL actually add?** Yue et al., [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) (2025), found that RLVR raises pass@1 substantially, but at large $k$ (hundreds of samples) the *base* model often solves as many or more problems. Their reading: RL mostly **reweights** reasoning paths the base model can already produce, making good ones likely on the first try, rather than creating new abilities. This is consistent with Theorem 21.1 ($\pi^*\propto\pi_\text{ref}e^{r/\beta}$ only tilts the reference distribution). Others find that long, sustained RL training does expand what models can solve. The question is open and actively researched. In practice, base-model quality (pretraining) remains the foundation, and RL makes it reliable.

## 22.7 Controlling how long models think

- **Thinking budgets.** APIs let you cap or set the reasoning effort: e.g. a maximum number of thinking tokens, or an effort level (low/medium/high). Accuracy typically grows roughly log-linearly with the thinking budget, with diminishing returns.
- **Budget forcing** ([Muennighoff et al., s1, 2025](https://arxiv.org/abs/2501.19393)): with only 1,000 SFT examples of reasoning traces, a 32B model became a competitive reasoner. To make it think *longer*, suppress the end-of-thinking token and append "Wait"; to make it think *shorter*, force the end-of-thinking token. Accuracy scaled with the forced length.
- **Overthinking.** Reasoning models can spend thousands of tokens on trivial questions, and long chains can drift into errors. Adaptive methods let the model decide how much to think per question.
- **Interleaved thinking with tools**: modern agentic models think, call a tool, think about the result, and continue (Ch. 27).

## 22.8 Is the chain of thought faithful?

A chain of thought is text the model generates. It need not reflect the computation that actually produced the answer.

- Turpin et al. ([2023](https://arxiv.org/abs/2305.04388)) biased few-shot prompts (e.g. the correct answer was always "(A)") and found that models' answers followed the bias while their explanations never mentioned it.
- Anthropic's [Reasoning Models Don't Always Say What They Think](https://www.anthropic.com/research/reasoning-models-dont-say-think) (Chen et al., 2025) gave reasoning models hints and measured how often their chains of thought admitted using a hint they had in fact used. It was often well under half, and lower for misaligned hints such as "unauthorized access" to the answer.

Still, chains of thought are one of the best windows into model behavior that exist. **CoT monitoring**, reading reasoning traces with another model to catch reward hacking or misbehavior, has caught real cases ([Baker et al., 2025](https://arxiv.org/abs/2503.11926)). But training directly against a CoT monitor can teach models to hide their intent in the CoT while continuing the behavior. Many researchers argue that keeping chains of thought legible and unoptimized is worth protecting ([Korbak et al., 2025](https://arxiv.org/abs/2507.11473)).

!!! llm "Using reasoning models well"
    - Use them for multi-step problems: math, code, planning, analysis of long documents, agentic tasks. For simple lookups or style rewrites they add cost and latency for little gain.
    - Give the **goal and the constraints**, not step-by-step instructions on how to think. They plan better on their own, and over-specified reasoning instructions often hurt.
    - Budget for thinking tokens: they're billed as output tokens and add latency.
    - Thinking is usually summarized or hidden in products; don't build logic that depends on parsing it.
    - Evaluate with several samples (pass@1 averaged over runs), since outputs vary (Ch. 29).

## Key takeaways

- Chain of thought adds serial computation per answer and provably extends what Transformers can compute.
- Test-time compute can be spent on parallel samples with voting or verifiers, on search, or on longer single chains of reasoning.
- RLVR trains on checkable rewards (answers, unit tests, formats). It's harder to hack than a learned reward model, but not impossible.
- GRPO = PPO-clip + group-normalized advantages + $k_3$ KL, without a value model. DAPO and Dr. GRPO fix its biases.
- DeepSeek-R1: pure RL from a base model produces long reasoning and self-correction (R1-Zero); a cold-start SFT → RL → rejection-sampling SFT → RL pipeline makes it usable; distillation transfers it to small models.
- RL mainly sharpens pass@1 within what the base model can sample (debated at large scale). Chains of thought aren't always faithful, but they're valuable for monitoring.

## Exercises

1. A model solves a problem with probability 0.2 per sample. Compute pass@1, pass@5 and pass@20.
2. With $n = 10$ samples of which $c = 3$ are correct, compute the unbiased pass@1 and pass@5 estimates.
3. In GRPO with binary rewards, a group of $G$ samples has $c$ correct. Find the advantage of a correct and of an incorrect sample. What happens when $c = 0$ or $c = G$?
4. Why does a value model (as in PPO) struggle with 10,000-token reasoning chains with a single reward at the end?
5. Argue why distilling a large reasoner's traces into a small model can beat running RL on the small model directly.
6. **Code:** with TRL's `GRPOTrainer`, train a 0.5B–1.5B model on GSM8K with a reward of 1 for a correct final number. Track accuracy and mean response length over steps.

??? success "Solutions"
    1. pass@1 = 0.2; pass@5 $= 1-0.8^5\approx0.672$; pass@20 $= 1-0.8^{20}\approx0.988$.
    2. pass@1 $= 1-\binom71/\binom{10}1 = 0.3$. pass@5 $= 1-\binom75/\binom{10}5 = 1-21/252\approx0.917$.
    3. Mean $\mu = c/G$, std $\sigma = \sqrt{\mu(1-\mu)}$ (population std; `torch.std` in the code uses the sample std, larger by $\sqrt{G/(G-1)}$). Correct: $(1-\mu)/\sigma = \sqrt{(1-\mu)/\mu}$; incorrect: $-\mu/\sigma = -\sqrt{\mu/(1-\mu)}$. With $c = 0$ or $c = G$ all rewards are equal, $\sigma = 0$, and the advantages are 0 (with the $\epsilon$ in the code), so the group gives no learning signal.
    4. It must predict, at every token, the probability of eventually being correct: a very noisy target, learned from one binary outcome per long sequence, so its per-token estimates are poor and bias the advantages. Group baselines sidestep this.
    5. RL can only reinforce behaviors the model already samples with some probability. A small model rarely produces long, correct, self-correcting reasoning, so RL gets little signal. Distillation shows it complete, high-quality traces directly, a dense supervised signal.
    6. Accuracy should rise over a few hundred steps, and mean length typically grows as the model learns to show more work.

## References

- J. Wei et al., [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) (NeurIPS 2022); T. Kojima et al., [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) (NeurIPS 2022).
- W. Merrill & A. Sabharwal, [The Expressive Power of Transformers with Chain of Thought](https://arxiv.org/abs/2310.07923) (ICLR 2024).
- H. Lightman et al., [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) (ICLR 2024).
- C. Snell et al., [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314) (2024).
- Z. Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300) (2024), introducing GRPO.
- DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025).
- M. Chen et al., [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374) (2021), §2.1 (pass@k).
- Y. Yue et al., [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) (2025).
- Y. Chen et al. (Anthropic), [Reasoning Models Don't Always Say What They Think](https://www.anthropic.com/research/reasoning-models-dont-say-think) (2025).
