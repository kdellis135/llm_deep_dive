# 4. Information Theory

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 3](probability.md) (distributions, expectation, MLE). **You will learn:** why information is $-\log p$, entropy and its maximum, Jensen's inequality, the source-coding theorem (compression = prediction), cross-entropy, KL divergence and why it is non-negative, forward vs reverse KL, perplexity and bits-per-byte, mutual information, and the KL estimators used in RLHF.
    **Why it matters:** the LLM training loss *is* a cross-entropy. Perplexity, the headline number for language models, is an exponentiated entropy. KL divergence appears in distillation, in RLHF's penalty, in DPO's derivation, and in variational methods. And the deepest reason next-token prediction works so well is information-theoretic: **a model that predicts well compresses well**, and compressing human text requires modeling the world that produced it.

!!! watch "Watch"
    - Aurélien Géron, [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) (11 min). The clearest short treatment.
    - 3Blue1Brown, [Solving Wordle using information theory](https://www.youtube.com/watch?v=v68zYyaEmEA). Entropy as "expected bits of information", made concrete.

    Reading: T. Cover & J. Thomas, *Elements of Information Theory*, Ch. 2 and 5 (the standard text). D. MacKay, [*Information Theory, Inference, and Learning Algorithms*](https://www.inference.org.uk/itprnn/book.html) (free), Ch. 2, 4–6. MacKay's [lecture videos](https://www.inference.org.uk/itprnn_lectures/) are excellent.

---

## 4.1 Information content: why $-\log p$?

We want a number $I(x)$ that measures how *surprised* we are to see an outcome $x$ of probability $p(x)$. Three reasonable requirements:

1. $I$ depends only on the probability: $I(x) = f(p(x))$.
2. Certain events carry no information: $f(1) = 0$, and rarer events are more surprising: $f$ is decreasing.
3. Information from **independent** events adds: $f(pq) = f(p) + f(q)$.

!!! theorem "Theorem 4.1 (Uniqueness of the logarithm)"
    The only continuous functions satisfying 1–3 are $f(p) = -c\log p$ with $c>0$.

??? proof "Proof"
    Let $g(t) = f(e^{-t})$ for $t\ge0$. Requirement 3 becomes $g(s+t) = g(s)+g(t)$ (Cauchy's functional equation). By induction $g(nt) = n\,g(t)$ for integers $n$, so $g(m/n) = \tfrac{m}{n}g(1)$ for every positive rational. By continuity $g(t) = t\,g(1)$ for all real $t\ge0$. Setting $c = g(1)>0$ (positive because $f$ decreases) gives $f(e^{-t}) = ct$, i.e. $f(p) = -c\log p$. $\blacksquare$

The constant $c$ only picks the unit. With $\log_2$ the unit is **bits**; with $\ln$ it is **nats** ($1\text{ nat} = 1/\ln 2 \approx 1.443$ bits). This site uses nats unless stated otherwise, because PyTorch's `cross_entropy` uses $\ln$.

!!! intuition "Intuition"
    A fair coin flip carries $-\log_2\frac12 = 1$ bit. An event with probability $1/1024$ carries 10 bits: it is as surprising as ten heads in a row. Each extra bit halves the probability.

## 4.2 Entropy

!!! definition "Definition 4.1 (Entropy)"
    The **entropy** of a discrete distribution $p$ is the expected information content:

    $$
    H(p) = \E_{x\sim p}[-\log p(x)] = -\sum_x p(x)\log p(x),
    $$

    with the convention $0\log 0 = 0$ (the limit of $t\log t$ as $t\to0^+$).

Entropy measures **uncertainty**: how many bits, on average, you need to learn the outcome. A Bernoulli($q$) variable has $H = -q\log q - (1-q)\log(1-q)$, which is 0 at $q\in\{0,1\}$ and peaks at 1 bit when $q = \frac12$.

To prove facts about entropy we need one inequality, used constantly in ML.

!!! theorem "Theorem 4.2 (Jensen's inequality)"
    If $\varphi$ is convex, then $\varphi(\E X) \le \E[\varphi(X)]$. If $\varphi$ is concave the inequality reverses. For strictly convex $\varphi$, equality holds iff $X$ is constant (almost surely).

??? proof "Proof (finite case, by induction)"
    For two points convexity *is* the statement: $\varphi(\lambda a + (1-\lambda)b)\le\lambda\varphi(a)+(1-\lambda)\varphi(b)$. Suppose it holds for $n-1$ points. For weights $\lambda_1,\dots,\lambda_n$ summing to 1 with $\lambda_n<1$, write $\sum_i\lambda_ix_i = \lambda_nx_n + (1-\lambda_n)\sum_{i<n}\frac{\lambda_i}{1-\lambda_n}x_i$. Apply the two-point inequality, then the induction hypothesis to the inner average:

    $$
    \varphi\Big(\sum_i\lambda_ix_i\Big)\le\lambda_n\varphi(x_n) + (1-\lambda_n)\sum_{i<n}\frac{\lambda_i}{1-\lambda_n}\varphi(x_i) = \sum_i\lambda_i\varphi(x_i).\ \blacksquare
    $$

    (The general case follows from the existence of a supporting line $\varphi(x)\ge\varphi(\mu) + s(x-\mu)$ at $\mu = \E X$: take expectations of both sides.)

!!! theorem "Theorem 4.3 (Bounds on entropy)"
    For a distribution on $n$ outcomes, $0 \le H(p) \le \log n$. The lower bound is attained only by a point mass, the upper bound only by the uniform distribution.

??? proof "Proof"
    *Lower:* each term $-p\log p\ge0$ for $p\in[0,1]$, and it is zero only at $p\in\{0,1\}$.

    *Upper:* $\log$ is concave, so by Jensen,

    $$
    H(p) = \sum_x p(x)\log\frac{1}{p(x)} \le \log\sum_x p(x)\cdot\frac{1}{p(x)} = \log\big|\{x: p(x)>0\}\big| \le \log n,
    $$

    with equality iff $1/p(x)$ is constant on the support and the support is everything, i.e. $p$ is uniform. $\blacksquare$

!!! llm "LLM connection"
    At initialization, a well-initialized LLM outputs nearly uniform probabilities over $V$ tokens, so its loss starts near the maximum entropy $\ln V$. For GPT-2's $V = 50{,}257$ that's $\ln 50257\approx10.82$ nats. If your first training step reports a loss far above that, your initialization is broken (Ch. 14 checks this). The softmax widget in Ch. 2 reports the entropy of the distribution as you change the temperature: $T\to0$ drives it to 0 and $T\to\infty$ drives it to $\log n$.

**Joint and conditional entropy.** For two variables, $H(X,Y) = -\sum p(x,y)\log p(x,y)$, and the **conditional entropy** is $H(Y\mid X) = \E_{x}\big[H(Y\mid X=x)\big] = -\sum_{x,y}p(x,y)\log p(y\mid x)$.

!!! theorem "Theorem 4.4 (Chain rule for entropy)"
    $H(X,Y) = H(X) + H(Y\mid X)$. More generally, $H(X_1,\dots,X_T) = \sum_{t=1}^T H(X_t\mid X_{<t})$.

??? proof "Proof"
    Take $-\log$ of the probability chain rule $p(x,y) = p(x)\,p(y\mid x)$ (Thm 3.1) and take expectations under $p(x,y)$. The $T$-variable version is the same argument applied to $p(x_{1:T}) = \prod_tp(x_t\mid x_{<t})$. $\blacksquare$

This is exactly how a language model factorizes text: the uncertainty of a whole document equals the sum of the per-token uncertainties given the prefix. **Next-token prediction loses nothing** compared with modeling whole sequences at once.

## 4.3 Source coding: compression *is* prediction

Suppose we want to encode outcomes of $X\sim p$ as binary strings, with no codeword a prefix of another (so a stream can be decoded unambiguously). Let $\ell(x)$ be the length of $x$'s codeword.

!!! theorem "Theorem 4.5 (Kraft inequality)"
    A prefix code with lengths $\ell(x)$ exists if and only if $\sum_x 2^{-\ell(x)}\le1$.

??? proof "Proof sketch"
    Picture all binary strings as a binary tree. A codeword of length $\ell$ is a node at depth $\ell$, and the prefix property means no codeword lies below another. A node at depth $\ell$ "owns" a fraction $2^{-\ell}$ of the leaves at any deeper level, and these sets are disjoint, so the fractions sum to at most 1. Conversely, given lengths satisfying the inequality, assign codewords greedily in order of increasing length: each choice uses up a fraction $2^{-\ell}$ of the tree, and the inequality guarantees we never run out. $\blacksquare$

!!! theorem "Theorem 4.6 (Shannon's source-coding theorem)"
    Every prefix code has expected length $\E[\ell(X)]\ge H_2(p)$ bits (entropy in base 2). Conversely there is a code with $\E[\ell(X)] < H_2(p)+1$, namely $\ell(x) = \lceil-\log_2p(x)\rceil$.

??? proof "Proof"
    *Lower bound.* Let $Z = \sum_x2^{-\ell(x)}\le1$ (Kraft) and define the distribution $q(x) = 2^{-\ell(x)}/Z$. Then

    $$
    \E[\ell(X)] - H_2(p) = \sum_xp(x)\log_2\frac{p(x)}{2^{-\ell(x)}} = \sum_xp(x)\log_2\frac{p(x)}{q(x)} - \log_2Z \ge 0,
    $$

    because the first sum is a KL divergence, which is $\ge0$ (Thm 4.8 below), and $-\log_2 Z\ge0$.

    *Achievability.* $\ell(x) = \lceil-\log_2p(x)\rceil$ satisfies Kraft because $2^{-\ell(x)}\le p(x)$, and $\ell(x)<-\log_2p(x)+1$. Take expectations. $\blacksquare$

Coding blocks of many symbols at once makes the "+1" negligible per symbol, and **arithmetic coding** achieves this in practice: given *any* predictive model $q(x_t\mid x_{<t})$, it encodes a sequence in about $\sum_t-\log_2q(x_t\mid x_{<t})$ bits, i.e. the model's cross-entropy on that sequence.

!!! llm "Language modeling is compression"
    Combine the chain rule (Thm 4.4) with arithmetic coding: **an LLM with average loss $\mathcal{L}$ nats/token is a compressor using $\mathcal{L}/\ln2$ bits per token.** Lower loss means better compression, and nothing more or less. Delétang et al., [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) (2023), show that a 70B LLM used this way compresses text better than gzip or even specialized compressors, and compresses images and audio surprisingly well too. To compress text from people well, you must predict what those people will write next, which requires modeling their knowledge and reasoning. That is the information-theoretic argument for why scaling next-token prediction produces capability.

## 4.4 Cross-entropy

What if we encode data from $p$ using a code optimized for a *different* distribution $q$, i.e. lengths $-\log q(x)$?

!!! definition "Definition 4.2 (Cross-entropy)"
    $$
    H(p,q) = \E_{x\sim p}[-\log q(x)] = -\sum_xp(x)\log q(x).
    $$

    It is the expected surprise of a model $q$ when the data actually comes from $p$.

!!! theorem "Theorem 4.7 (Gibbs' inequality)"
    $H(p,q)\ge H(p)$, with equality iff $q=p$.

This is the next theorem in disguise, since $H(p,q) - H(p) = \KL(p\Vert q)$.

## 4.5 KL divergence

!!! definition "Definition 4.3 (Kullback–Leibler divergence)"
    $$
    \KL(p\,\Vert\,q) = \sum_xp(x)\log\frac{p(x)}{q(x)} = \E_{x\sim p}\left[\log\frac{p(x)}{q(x)}\right] = H(p,q) - H(p).
    $$

    It is the **extra** nats per sample you pay for using $q$ when the truth is $p$. If $q(x)=0$ somewhere $p(x)>0$, it is $+\infty$.

!!! theorem "Theorem 4.8 (KL is non-negative)"
    $\KL(p\Vert q)\ge0$, with equality iff $p=q$.

??? proof "Proof (Jensen)"
    $-\log$ is strictly convex. Summing over the support of $p$:

    $$
    \KL(p\Vert q) = \E_{x\sim p}\left[-\log\frac{q(x)}{p(x)}\right]\ \ge\ -\log\E_{x\sim p}\left[\frac{q(x)}{p(x)}\right] = -\log\sum_{x:\,p(x)>0}q(x)\ \ge\ -\log1 = 0.
    $$

    Equality in Jensen needs $q(x)/p(x)$ constant on the support, and equality in the last step needs $q$ to put all its mass there. Together that forces $q = p$. $\blacksquare$

??? proof "Alternative proof (the inequality $\log t\le t-1$)"
    $\log t\le t-1$ for $t>0$, with equality only at $t=1$ (the tangent line at 1 lies above the concave $\log$). So $-\KL(p\Vert q) = \sum p\log\frac qp\le\sum p\left(\frac qp-1\right) = \sum q - \sum p\le 0$. $\blacksquare$

KL is **not a distance**. It is not symmetric, and it doesn't satisfy the triangle inequality. The asymmetry matters a lot in practice.

<div class="widget" data-widget="kl"></div>

### Forward vs reverse KL

Suppose the truth $p$ is bimodal and we fit a unimodal $q_\theta$.

- **Forward KL**, $\min_\theta\KL(p\Vert q_\theta)$, averages over samples from $p$. The term $p\log(p/q)$ blows up wherever $p>0$ but $q\approx0$, so $q$ must **cover every mode** of $p$. It is "mean-seeking" or "mass-covering", and the fit ends up smeared between the modes. **Maximum likelihood minimizes forward KL** (Thm 4.9), which is why pretrained LMs put probability on everything people might write, including the bad stuff.
- **Reverse KL**, $\min_\theta\KL(q_\theta\Vert p)$, averages over samples from $q$. It only penalizes $q$ for putting mass where $p$ is small, so $q$ can **lock onto a single mode** and ignore the others. It is "mode-seeking".

!!! llm "Where each KL appears"
    - **Pretraining / SFT**: forward KL from the data distribution (via cross-entropy).
    - **Knowledge distillation**: forward KL from teacher to student, $\KL(p_\text{teacher}\Vert p_\text{student})$, on the teacher's soft probabilities ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)). Recent on-policy distillation uses reverse KL ([Agarwal et al., 2024](https://arxiv.org/abs/2306.13649)).
    - **RLHF**: maximize reward minus $\beta\,\KL(\pi_\theta\Vert\pi_\text{ref})$, a **reverse** KL evaluated on the policy's own samples. It keeps the policy close to the SFT model, and it's part of why RL-tuned models are less diverse ("mode collapse") than base models (Ch. 21).

## 4.6 Cross-entropy loss = maximum likelihood = minimum KL

!!! theorem "Theorem 4.9 (The three views of the LLM loss coincide)"
    Let $\hat p$ be the empirical distribution of a dataset $\{x^{(i)}\}_{i=1}^n$. For any model $q_\theta$,

    $$
    \underbrace{-\frac1n\sum_{i=1}^n\log q_\theta(x^{(i)})}_{\text{average NLL (MLE)}} \;=\; \underbrace{H(\hat p, q_\theta)}_{\text{cross-entropy}} \;=\; H(\hat p) + \underbrace{\KL(\hat p\,\Vert\,q_\theta)}_{\text{forward KL}}.
    $$

    Since $H(\hat p)$ doesn't depend on $\theta$, **minimizing the NLL, the cross-entropy, and the forward KL are the same optimization problem.**

??? proof "Proof"
    $\hat p(x) = \frac1n\#\{i: x^{(i)} = x\}$, so $H(\hat p, q) = -\sum_x\hat p(x)\log q(x) = -\frac1n\sum_i\log q(x^{(i)})$. The second equality is Definition 4.3. $\blacksquare$

For a language model, the "outcome" at each position is the next token, and the loss averages over all $B\cdot T$ positions in a batch:

$$
\mathcal{L}(\theta) = -\frac{1}{BT}\sum_{b=1}^B\sum_{t=1}^T\log p_\theta\big(x^{(b)}_{t+1}\mid x^{(b)}_{\le t}\big).
$$

With a one-hot target $\mathbf{y}$ and predicted $\mathbf{p} = \softmax(\mathbf{z})$, the cross-entropy for one position is $-\sum_iy_i\log p_i = -\log p_y = -z_y + \log\sum_je^{z_j}$, with gradient $\mathbf{p}-\mathbf{y}$ (Thm 2.8).

!!! warning "The loss can't go to zero"
    The irreducible part is $H(\hat p)$, or for the true distribution of text, its **entropy rate**. Natural language is genuinely uncertain ("The weather today is ___"), so even a perfect model has loss well above 0. Scaling laws (Ch. 17) model this explicitly as the constant $E$ in $L(N,D) = E + A/N^\alpha + B/D^\beta$.

## 4.7 Perplexity and bits-per-byte

!!! definition "Definition 4.4 (Perplexity)"
    The **perplexity** of a model on a token sequence is the exponentiated average NLL:

    $$
    \mathrm{PPL} = \exp\Big(-\frac1T\sum_{t=1}^T\log p_\theta(x_t\mid x_{<t})\Big) = e^{\mathcal{L}}.
    $$

**Interpretation.** A uniform distribution over $k$ choices has perplexity exactly $k$ (its entropy is $\log k$). So perplexity is the **effective branching factor**: a model with PPL 20 is, on average, as uncertain as if it were choosing uniformly among 20 tokens. GPT-2 small reaches a validation loss around 3.0–3.3 nats on web text, so PPL $\approx 20$–$27$. Frontier models are far lower.

!!! warning "Perplexity depends on the tokenizer"
    Loss per *token* isn't comparable across models with different tokenizers: a tokenizer with longer tokens has fewer, harder predictions. To compare fairly, normalize by a tokenizer-independent unit. **Bits per byte** (bpb) is the standard:

    $$
    \text{bpb} = \frac{\text{total NLL in nats}}{\ln 2\times\text{number of UTF-8 bytes}} = \frac{\mathcal{L}_\text{token}}{\ln 2}\cdot\frac{\#\text{tokens}}{\#\text{bytes}}.
    $$

    A typical English BPE tokenizer averages about 4 bytes per token. Karpathy's [nanochat](https://github.com/karpathy/nanochat) reports bpb for exactly this reason.

## 4.8 Mutual information

!!! definition "Definition 4.5 (Mutual information)"
    $$
    I(X;Y) = \KL\big(p(x,y)\,\Vert\,p(x)p(y)\big) = H(Y) - H(Y\mid X) = H(X) - H(X\mid Y).
    $$

    It is how much knowing $X$ reduces uncertainty about $Y$. It is symmetric, $\ge0$ (Thm 4.8), and zero iff $X$ and $Y$ are independent.

??? proof "Proof that the forms agree"
    $\KL(p(x,y)\Vert p(x)p(y)) = \sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)} = \sum p(x,y)\log\frac{p(y\mid x)}{p(y)} = -H(Y\mid X) + H(Y)$. Swap the roles for the other form. $\blacksquare$

Two consequences:

- **Conditioning reduces entropy on average:** $H(Y\mid X)\le H(Y)$, since $I\ge0$. More context can only help a language model in expectation, which is why longer contexts lower the loss (and why loss *per position* falls with position in a sequence).
- **Data-processing inequality.** If $X\to Y\to Z$ is a Markov chain (Z depends on X only through Y), then $I(X;Z)\le I(X;Y)$: no processing of $Y$ can create information about $X$ that $Y$ didn't contain. Each Transformer layer is a deterministic function of the previous one, so the residual stream can never gain information about the input, only make it more *usable*.

## 4.9 KL estimators (used in RLHF and GRPO)

In RL fine-tuning we need $\KL(\pi_\theta\Vert\pi_\text{ref})$ for a policy over *sequences*, so we can't sum over all outcomes. We estimate it from samples $x\sim\pi_\theta$. Let $r(x) = \pi_\text{ref}(x)/\pi_\theta(x)$, so $\E_{x\sim\pi_\theta}[r] = 1$. Following [J. Schulman's note on KL approximation](http://joschu.net/blog/kl-approx.html):

| Estimator | Formula | Unbiased? | Always $\ge 0$? |
|---|---|---|---|
| $k_1$ | $-\log r$ | yes | no (high variance) |
| $k_2$ | $\tfrac12(\log r)^2$ | no (low bias) | yes |
| $k_3$ | $(r-1)-\log r$ | **yes** | **yes** |

!!! theorem "Proposition 4.10 (The $k_3$ estimator)"
    $k_3 = (r-1)-\log r$ is an unbiased, non-negative estimator of $\KL(\pi_\theta\Vert\pi_\text{ref})$.

??? proof "Proof"
    *Unbiased:* $\E[k_3] = \E[r]-1 + \E[-\log r] = 1 - 1 + \KL(\pi_\theta\Vert\pi_\text{ref}) = \KL$. *Non-negative:* $\log r\le r-1$ for all $r>0$ (the tangent-line bound used in the second proof of Thm 4.8). Adding the zero-mean term $r-1$ to $k_1$ is a **control variate**: it cancels much of $k_1$'s noise. $\blacksquare$

This is exactly `kl_k3` in [`code/alignment.py`](https://github.com/kdellis135/llm_deep_dive/blob/main/code/alignment.py), the per-token KL penalty in GRPO (Ch. 22):

```python
--8<-- "code/alignment.py:grpo"
```

## 4.10 Temperature, label smoothing, and entropy in practice

- **Temperature** divides logits by $T$. As $T$ grows the entropy of $\softmax(\mathbf{z}/T)$ increases monotonically toward $\log V$ (Exercise 5). Sampling at low $T$ trades diversity for likelihood (Ch. 18).
- **Label smoothing** replaces the one-hot target with $(1-\epsilon)\mathbf{y}+\epsilon/V$. The loss becomes $(1-\epsilon)H(\mathbf{y},\mathbf{p}) + \epsilon H(\mathbf{u},\mathbf{p})$, where $\mathbf{u}$ is uniform. The second term penalizes overconfident predictions. It was used in the original Transformer, but it's rarely used for LLM pretraining, because it biases the model away from calibrated probabilities.
- **Calibration.** A model is calibrated if, among predictions made with confidence 0.8, 80% are correct. Minimizing cross-entropy, a *proper scoring rule* (its expected value is minimized only by reporting the true distribution, as Gibbs' inequality shows), makes pretrained LMs well calibrated. RLHF tends to degrade calibration ([GPT-4 technical report](https://arxiv.org/abs/2303.08774), Fig. 8).

## Key takeaways

- Information is $-\log p$; entropy is expected information, between $0$ and $\log n$.
- Jensen's inequality ⇒ KL $\ge0$ ⇒ cross-entropy $\ge$ entropy.
- **Cross-entropy loss = average NLL = MLE = forward KL from the data + a constant.** That is the LLM pretraining objective.
- A model's loss in bits is its compression rate. Good prediction is good compression.
- Perplexity $= e^{\text{loss}}$ is the effective branching factor. Compare across tokenizers with bits-per-byte.
- Forward KL covers modes (MLE); reverse KL seeks modes (the RLHF penalty). The $k_3$ estimator gives an unbiased, positive KL estimate from samples.

## Exercises

1. Compute the entropy in bits of a fair die, and of a die that lands on 6 with probability 1/2 and on the other faces uniformly.
2. Show $H(X,Y)\le H(X)+H(Y)$ with equality iff $X$ and $Y$ are independent.
3. A model's validation loss is 2.30 nats/token and its tokenizer averages 4.2 bytes per token. What are its perplexity and bits-per-byte?
4. Compute $\KL(p\Vert q)$ and $\KL(q\Vert p)$ for $p=(0.5,0.5)$, $q=(0.9,0.1)$. Which is larger and why?
5. Prove that the entropy of $\softmax(\mathbf{z}/T)$ is non-decreasing in $T$. (Hint: differentiate with respect to $\beta = 1/T$ and show the derivative is $-\beta\Var_{p}(z)$.)
6. **Code:** train a character-level bigram model (Ch. 9) on any text, convert its loss to bits-per-character, and compare with the size of `gzip -9` on the same text. Which compresses better?

??? success "Solutions"
    1. $\log_26\approx2.585$ bits. For the biased die: $-\frac12\log_2\frac12 - 5\cdot\frac1{10}\log_2\frac1{10} = 0.5 + 0.5\log_210\approx2.161$ bits.
    2. $H(X)+H(Y)-H(X,Y) = I(X;Y)\ge0$, and $I = 0$ iff $p(x,y) = p(x)p(y)$ (Thm 4.8).
    3. PPL $=e^{2.30}\approx9.97$. bpb $=2.30/(\ln2\cdot4.2)\approx0.79$.
    4. $\KL(p\Vert q) = 0.5\ln\frac{0.5}{0.9}+0.5\ln\frac{0.5}{0.1}\approx0.511$ nats. $\KL(q\Vert p) = 0.9\ln1.8+0.1\ln0.2\approx0.368$ nats. The first is larger because $p$ puts substantial mass (0.5) where $q$ is small (0.1), which forward KL punishes heavily.
    5. With $p_i\propto e^{\beta z_i}$, $H = -\sum p_i(\beta z_i - \log Z) = \log Z - \beta\,\E_p[z]$. Using $\frac{d}{d\beta}\log Z = \E_p[z]$ and $\frac{d}{d\beta}\E_p[z] = \Var_p(z)$ gives $\frac{dH}{d\beta} = -\beta\Var_p(z)\le0$. Since $\beta = 1/T$ decreases in $T$, $H$ increases with $T$.
    6. A character bigram model gets roughly 3.3–3.6 bits/char on English (close to Shannon's second-order estimate of 3.3), which is *worse* than `gzip -9` (about 2.5–3 bits/char), because gzip exploits long repeated strings. A small character Transformer beats both: nanoGPT's Shakespeare-char model reaches about 1.47 nats ≈ 2.1 bits/char on validation, and large models go far lower. Better prediction means better compression.

## References

- C. E. Shannon, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) (1948). The founding paper; Section 7 discusses the entropy of English.
- T. M. Cover & J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley (2006), Ch. 2, 5.
- D. J. C. MacKay, [*Information Theory, Inference, and Learning Algorithms*](https://www.inference.org.uk/itprnn/book.html) (2003).
- G. Delétang et al., [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) (ICLR 2024).
- G. Hinton, O. Vinyals, J. Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015).
- J. Schulman, [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) (2020).
- C. Olah, [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) (2015). A beautiful visual introduction.
