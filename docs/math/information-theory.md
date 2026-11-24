# 4. Information Theory

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 3](probability.md) (distributions, expectation, MLE). Logarithms are refreshed below.

    **You will learn:**

    - why the information in an outcome is $-\log p$
    - entropy and its maximum, and Jensen's inequality
    - the source-coding theorem: compression = prediction
    - cross-entropy and KL divergence, and why KL is never negative
    - forward vs reverse KL
    - perplexity and bits-per-byte
    - mutual information
    - the KL estimators used in RLHF

    **Why it matters:**

    - The LLM training loss *is* a cross-entropy.
    - Perplexity, the headline number for language models, is an exponentiated entropy.
    - KL divergence appears in distillation, in RLHF's penalty, in DPO's derivation, and in variational methods.
    - The deepest reason next-token prediction works is information-theoretic: **a model that predicts well compresses well**, and compressing human text requires modeling the world that produced it.

!!! watch "Watch"
    - Aurélien Géron, [A Short Introduction to Entropy, Cross-Entropy and KL-Divergence](https://www.youtube.com/watch?v=ErfnhcEV1O8) (11 min). The clearest short treatment.
    - 3Blue1Brown, [Solving Wordle using information theory](https://www.youtube.com/watch?v=v68zYyaEmEA). Entropy as "expected bits of information", made concrete.

    Reading: T. Cover & J. Thomas, *Elements of Information Theory*, Ch. 2 and 5 (the standard text). D. MacKay, [*Information Theory, Inference, and Learning Algorithms*](https://www.inference.org.uk/itprnn/book.html) (free), Ch. 2, 4–6. MacKay's [lecture videos](https://www.inference.org.uk/itprnn_lectures/) are excellent.

---

## Roadmap

| § | Question it answers | Used later in |
|---|---|---|
| Warm-up | How do logarithms behave? | every formula here |
| 4.1 | How much information is in one outcome? | everything below |
| 4.2 | How uncertain is a distribution? | loss at initialization, temperature |
| 4.3 | Why is prediction the same as compression? | why LLMs work |
| 4.4 | What does it cost to use the wrong model? | the training loss |
| 4.5 | How far apart are two distributions? | distillation, RLHF (Ch. 21) |
| 4.6 | Why are cross-entropy, MLE and KL the same thing? | pretraining (Ch. 15) |
| 4.7 | What is perplexity? | evaluation |
| 4.8 | How much does one variable tell us about another? | context length, interpretability |
| 4.9 | How do we estimate KL from samples? | GRPO (Ch. 22) |
| 4.10 | What do temperature, label smoothing and calibration mean here? | decoding, training |

---

## Warm-up: logarithms

Information theory is written in logarithms. Here is everything you need.

$\log_b x$ answers: "**$b$ to what power gives $x$?**" So $\log_2 8 = 3$ because $2^3 = 8$.

| Rule | Formula | Example ($\log_2$) |
|---|---|---|
| product → sum | $\log(ab) = \log a + \log b$ | $\log_2(4\cdot8) = 2 + 3 = 5$ |
| quotient → difference | $\log(a/b) = \log a - \log b$ | $\log_2(8/2) = 3 - 1 = 2$ |
| power → multiple | $\log(a^k) = k\log a$ | $\log_2(2^{10}) = 10$ |
| one | $\log 1 = 0$ | |
| fractions are negative | $\log p < 0$ for $0<p<1$ | $\log_2\frac14 = -2$ |
| flip a fraction | $\log\frac1p = -\log p$ | $\log_2 4 = -\log_2\frac14 = 2$ |
| change of base | $\log_2 x = \dfrac{\ln x}{\ln 2}$ | $\log_2 10 = \frac{2.303}{0.693} = 3.32$ |

Here $\ln$ is the natural log (base $e\approx2.718$).

**Units.** The base only changes the unit:

| Base | Unit | Convert |
|---|---|---|
| $2$ | **bits** | |
| $e$ | **nats** | $1\text{ nat} = 1/\ln 2 \approx 1.443$ bits |

This site uses nats unless stated otherwise, because PyTorch's `cross_entropy` uses $\ln$.

---

## 4.1 Information content: why $-\log p$? { data-toc-label="4.1 Information content: why −log p?" }

**The idea:** rare events are surprising. Surprise should add up for independent events. Only one function does both: $-\log p$.

We want a number $I(x)$ that measures how *surprised* we are to see an outcome $x$ of probability $p(x)$. Three reasonable requirements:

1. **It depends only on the probability:** $I(x) = f(p(x))$.
2. **Certain events carry no information, rarer events carry more:** $f(1) = 0$, and $f$ is decreasing.
3. **Information from independent events adds:** two independent events with probabilities $p$ and $q$ happen together with probability $pq$, and should carry $f(pq) = f(p) + f(q)$.

Requirement 3 is the key. Logs turn products into sums, so $-\log p$ fits. The surprise is: nothing else does.

!!! theorem "Theorem 4.1 (Uniqueness of the logarithm)"
    The only continuous functions satisfying 1–3 are $f(p) = -c\log p$ with $c>0$.

??? proof "Proof"
    **Step 1 · Change variables to turn multiplication into addition.** Write $p = e^{-t}$ (so $t = -\ln p\ge0$), and define $g(t) = f(e^{-t})$. Since $e^{-s}e^{-t} = e^{-(s+t)}$, requirement 3 becomes

    $$
    g(s+t) = g(s)+g(t).
    $$

    **Step 2 · Whole multiples.** Apply it repeatedly: $g(2t) = g(t) + g(t) = 2g(t)$, and in general $g(nt) = n\,g(t)$ for every positive integer $n$.

    **Step 3 · Fractions.** Put $t = 1/n$ in Step 2: $g(1) = n\,g(1/n)$, so $g(1/n) = g(1)/n$. Then $g(m/n) = m\,g(1/n) = \frac{m}{n}g(1)$. So $g(t) = t\,g(1)$ for every positive fraction $t$.

    **Step 4 · All real numbers.** Every real $t\ge0$ is a limit of fractions, and $g$ is continuous, so $g(t) = t\,g(1)$ for all $t\ge0$.

    **Step 5 · Translate back.** Let $c = g(1)$. It is positive, because $f$ decreases ($g(1) = f(e^{-1}) > f(1) = 0$). Then $f(e^{-t}) = ct$, i.e. $f(p) = -c\ln p$. $\blacksquare$

The constant $c$ just picks the unit (bits or nats).

!!! example "Worked example: how surprising is it?"
    | Event | $p$ | Information $-\log_2 p$ |
    |---|---|---|
    | sun rises | $1$ | $0$ bits |
    | likely event | $0.9$ | $0.15$ bits |
    | fair coin lands heads | $1/2$ | $1$ bit |
    | fair die shows 6 | $1/6$ | $2.58$ bits |
    | ten heads in a row | $1/1024$ | $10$ bits |

    **Each extra bit halves the probability.** And independent events add: ten coin flips, 1 bit each, 10 bits total.

---

## 4.2 Entropy

**The idea:** entropy is the *average* surprise. It measures how uncertain a distribution is.

!!! definition "Definition 4.1 (Entropy)"
    The **entropy** of a discrete distribution $p$ is its expected information content:

    $$
    H(p) = \E_{x\sim p}[-\log p(x)] = -\sum_x p(x)\log p(x).
    $$

    Convention: $0\log 0 = 0$. (It is the limit of $t\log t$ as $t\to0^+$, and it means impossible outcomes contribute nothing.)

!!! example "Worked examples"
    | Distribution | Calculation | $H$ (bits) |
    |---|---|---|
    | certain: $(1, 0)$ | $-1\log_21 - 0$ | $0$ |
    | biased coin: $(0.9, 0.1)$ | $0.9(0.152) + 0.1(3.32)$ | $0.469$ |
    | fair coin: $(0.5, 0.5)$ | $0.5(1) + 0.5(1)$ | $1$ |
    | fair 4-sided die | $4\times\frac14(2)$ | $2$ |
    | $(\frac12,\frac14,\frac18,\frac18)$ | $\frac12(1)+\frac14(2)+\frac18(3)+\frac18(3)$ | $1.75$ |
    | fair 6-sided die | $\log_26$ | $2.585$ |

    Pattern: **the more spread out, the higher the entropy.** A fair coin is maximally uncertain among 2-outcome distributions.

Entropy is how many bits, on average, you need to learn the outcome. For a coin with $P(\text{heads}) = q$, $H(q) = -q\log q - (1-q)\log(1-q)$. It is 0 at $q\in\{0,1\}$ and peaks at 1 bit when $q = \frac12$.

### Convexity and Jensen's inequality

To *prove* things about entropy we need one inequality, used constantly in ML. It's about convex functions (Ch. 2, §2.6).

!!! recall "Refresher: convex and concave"
    - **Convex** (cup-shaped, like $x^2$, $e^x$, $-\log x$): every chord lies **above** the graph. Test: $f''\ge0$.
    - **Concave** (cap-shaped, like $\log x$, $\sqrt x$): every chord lies **below** the graph. Test: $f''\le0$.

    For $\log$: $(\log x)'' = -1/x^2 < 0$, so $\log$ is concave and $-\log$ is convex.

!!! theorem "Theorem 4.2 (Jensen's inequality)"
    If $\varphi$ is convex, then

    $$
    \varphi(\E X) \le \E[\varphi(X)].
    $$

    "The function of the average is at most the average of the function." If $\varphi$ is concave, the inequality reverses. For *strictly* convex $\varphi$, equality holds only if $X$ is constant.

!!! example "Worked example"
    $X$ is $1$ or $3$ with equal probability. $\varphi(x) = x^2$ (convex).

    - $\varphi(\E X) = \varphi(2) = 4$.
    - $\E[\varphi(X)] = \frac12(1) + \frac12(9) = 5$.
    - $4\le5$. ✓

    Geometrically: the midpoint of the chord from $(1,1)$ to $(3,9)$ is $(2, 5)$, which lies above the curve's point $(2, 4)$.

??? proof "Proof (finite case, by induction)"
    **Step 1 · Two points.** For two values with weights $\lambda$ and $1-\lambda$, Jensen *is* the definition of convexity:

    $$
    \varphi(\lambda a + (1-\lambda)b)\le\lambda\varphi(a)+(1-\lambda)\varphi(b).
    $$

    **Step 2 · Assume it holds for $n-1$ points.** Take $n$ points $x_i$ with weights $\lambda_i\ge0$ summing to 1 (and $\lambda_n<1$).

    **Step 3 · Split off the last point.** Write the average as a two-point mix of $x_n$ and the average of the others:

    $$
    \sum_i\lambda_ix_i = \lambda_nx_n + (1-\lambda_n)\underbrace{\sum_{i<n}\frac{\lambda_i}{1-\lambda_n}x_i}_{\text{an average of } n-1 \text{ points}} .
    $$

    (The inner weights $\frac{\lambda_i}{1-\lambda_n}$ sum to 1.)

    **Step 4 · Apply the two-point inequality, then the assumption** to the inner average:

    $$
    \varphi\Big(\sum_i\lambda_ix_i\Big)\le\lambda_n\varphi(x_n) + (1-\lambda_n)\,\varphi\Big(\sum_{i<n}\tfrac{\lambda_i}{1-\lambda_n}x_i\Big)\le\lambda_n\varphi(x_n) + (1-\lambda_n)\sum_{i<n}\frac{\lambda_i}{1-\lambda_n}\varphi(x_i) = \sum_i\lambda_i\varphi(x_i).\ \blacksquare
    $$

    (The general case uses a supporting line: convexity gives $\varphi(x)\ge\varphi(\mu) + s(x-\mu)$ for some slope $s$, at $\mu = \E X$. Take expectations of both sides; the $s(x-\mu)$ term averages to 0.)

### How big can entropy be?

!!! theorem "Theorem 4.3 (Bounds on entropy)"
    For a distribution on $n$ outcomes:

    $$
    0 \le H(p) \le \log n .
    $$

    - $H = 0$ only for a point mass (one outcome is certain).
    - $H = \log n$ only for the uniform distribution.

???+ proof "Proof"
    **Lower bound.** Each term $-p\log p$ is $\ge0$ for $p\in[0,1]$ (a non-negative number times a non-positive log, negated). It is zero only at $p\in\{0,1\}$. So $H\ge0$, and $H = 0$ only if every probability is 0 or 1.

    **Upper bound, Step 1 · Rewrite** with $\log\frac1p = -\log p$:

    $$
    H(p) = \sum_x p(x)\log\frac{1}{p(x)} = \E_{x\sim p}\Big[\log\frac{1}{p(x)}\Big].
    $$

    **Upper bound, Step 2 · Jensen for the concave $\log$** (sum over outcomes with $p(x)>0$):

    $$
    \E\Big[\log\frac{1}{p(x)}\Big]\le\log\E\Big[\frac{1}{p(x)}\Big] = \log\sum_x p(x)\cdot\frac{1}{p(x)} = \log\big|\{x: p(x)>0\}\big| \le \log n .
    $$

    **Upper bound, Step 3 · Equality.** Jensen is tight only when $1/p(x)$ is the same for every outcome in the support, and the last step is tight only if all $n$ outcomes are in the support. Together: $p$ is uniform. $\blacksquare$

!!! llm "LLM connection: the loss at step 0"
    A well-initialized LLM outputs nearly uniform probabilities over its $V$ tokens. So its loss starts near the maximum entropy, $\ln V$.

    - For GPT-2's $V = 50{,}257$: $\ln 50257\approx10.82$ nats.
    - If your first training step reports a loss far above that, your initialization is broken (Ch. 14 checks this).

    The softmax widget in Ch. 2 reports the entropy as you change the temperature: $T\to0$ drives it to 0, $T\to\infty$ drives it to $\log n$.

### Joint and conditional entropy

For two variables:

| Quantity | Formula | Meaning |
|---|---|---|
| joint entropy $H(X,Y)$ | $-\sum p(x,y)\log p(x,y)$ | uncertainty about the pair |
| conditional entropy $H(Y\mid X)$ | $\E_{x}\big[H(Y\mid X=x)\big] = -\sum_{x,y}p(x,y)\log p(y\mid x)$ | uncertainty left about $Y$ once you know $X$, averaged over $x$ |

!!! theorem "Theorem 4.4 (Chain rule for entropy)"
    $H(X,Y) = H(X) + H(Y\mid X)$. More generally,

    $$
    H(X_1,\dots,X_T) = \sum_{t=1}^T H(X_t\mid X_{<t}).
    $$

???+ proof "Proof"
    **Step 1 · Start from the probability chain rule** (Thm 3.1): $p(x,y) = p(x)\,p(y\mid x)$.

    **Step 2 · Take $-\log$.** The product becomes a sum:

    $$
    -\log p(x,y) = -\log p(x) - \log p(y\mid x).
    $$

    **Step 3 · Take expectations under $p(x,y)$.** The left side is $H(X,Y)$. On the right, the first term averages to $H(X)$ (it depends only on $x$), and the second is $H(Y\mid X)$ by definition.

    **Step 4 · $T$ variables.** Same argument, starting from $p(x_{1:T}) = \prod_tp(x_t\mid x_{<t})$. $\blacksquare$

This is exactly how a language model factorizes text: **the uncertainty of a whole document equals the sum of the per-token uncertainties given the prefix.** Next-token prediction loses nothing compared with modeling whole sequences at once.

---

## 4.3 Source coding: compression *is* prediction

**The idea:** the best possible compression of data from $p$ uses about $-\log_2 p(x)$ bits for outcome $x$. So the average length is the entropy.

We want to encode outcomes as binary strings ("codewords") so that a stream of them can be decoded without separators.

**Prefix codes** guarantee this: no codeword is the beginning of another.

!!! example "Worked example: a good code"
    Outcomes A, B, C, D with probabilities $\frac12, \frac14, \frac18, \frac18$.

    | Outcome | $p$ | $-\log_2 p$ | Codeword | Length $\ell$ |
    |---|---|---|---|---|
    | A | 1/2 | 1 | `0` | 1 |
    | B | 1/4 | 2 | `10` | 2 |
    | C | 1/8 | 3 | `110` | 3 |
    | D | 1/8 | 3 | `111` | 3 |

    - The stream `0110100` decodes uniquely as `0 | 110 | 10 | 0` = A C B A.
    - Expected length: $\frac12(1) + \frac14(2) + \frac18(3) + \frac18(3) = 1.75$ bits.
    - That's exactly the entropy (see §4.2's table). Frequent outcomes got short codewords.

    **A bad (non-prefix) code:** A = `0`, B = `01`. Is `01` "A then something" or "B"? You can't tell.

How short can codewords be? Short codewords are a limited resource:

!!! theorem "Theorem 4.5 (Kraft inequality)"
    A prefix code with lengths $\ell(x)$ exists if and only if

    $$
    \sum_x 2^{-\ell(x)}\le1 .
    $$

    Check the example: $2^{-1} + 2^{-2} + 2^{-3} + 2^{-3} = \frac12+\frac14+\frac18+\frac18 = 1$. ✓

??? proof "Proof sketch"
    **Step 1 · Picture all binary strings as a tree.** From each node, `0` goes left and `1` goes right. A codeword of length $\ell$ is a node at depth $\ell$.

    **Step 2 · Prefix property = no codeword below another.** So each codeword "owns" the whole subtree beneath it, and these subtrees don't overlap.

    **Step 3 · Count the share each one owns.** Go down to some very deep level $D$. A node at depth $\ell$ owns $2^{D-\ell}$ of the $2^D$ nodes there: a fraction $2^{-\ell}$.

    **Step 4 · Fractions of a whole sum to at most 1.** Since the owned sets don't overlap, $\sum_x2^{-\ell(x)}\le1$.

    **Step 5 · Converse.** Given lengths satisfying the inequality, assign codewords greedily, shortest first. Each choice uses up a fraction $2^{-\ell}$ of the tree, and the inequality guarantees we never run out. $\blacksquare$

!!! theorem "Theorem 4.6 (Shannon's source-coding theorem)"
    1. **No code beats entropy:** every prefix code has expected length $\E[\ell(X)]\ge H_2(p)$ bits.
    2. **Entropy is nearly achievable:** the lengths $\ell(x) = \lceil-\log_2p(x)\rceil$ give a code with $\E[\ell(X)] < H_2(p)+1$.

    ($\lceil\cdot\rceil$ rounds up to a whole number, since codewords have whole-number lengths.)

??? proof "Proof"
    **Part 1, Step 1 · Turn the code lengths into a distribution.** Let $Z = \sum_x2^{-\ell(x)}$, which is $\le1$ by Kraft. Define $q(x) = 2^{-\ell(x)}/Z$, so $\ell(x) = -\log_2 q(x) - \log_2 Z$.

    **Part 1, Step 2 · Compute the gap:**

    $$
    \E[\ell(X)] - H_2(p) = \sum_xp(x)\big(\ell(x) + \log_2p(x)\big) = \sum_xp(x)\log_2\frac{p(x)}{q(x)} - \log_2Z .
    $$

    **Part 1, Step 3 · Both pieces are $\ge0$.** The sum is a KL divergence, which is never negative (Thm 4.8 below). And $Z\le1$ means $-\log_2Z\ge0$.

    **Part 2, Step 1 · The rounded lengths satisfy Kraft.** $\ell(x)\ge-\log_2p(x)$, so $2^{-\ell(x)}\le p(x)$, and summing gives $\le1$. So the code exists.

    **Part 2, Step 2 · Rounding up adds less than 1 bit:** $\ell(x)<-\log_2p(x)+1$. Take expectations: $\E[\ell]<H_2(p)+1$. $\blacksquare$

**Arithmetic coding** removes the "+1" in practice. Given *any* predictive model $q(x_t\mid x_{<t})$, it encodes a whole sequence in about

$$
\sum_t-\log_2q(x_t\mid x_{<t})\ \text{bits},
$$

which is the model's loss on that sequence, in bits.

!!! llm "Language modeling is compression"
    Combine the chain rule (Thm 4.4) with arithmetic coding:

    > **An LLM with average loss $\mathcal{L}$ nats/token is a compressor using $\mathcal{L}/\ln2$ bits per token.**

    - Lower loss means better compression — nothing more, nothing less.
    - Delétang et al., [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) (2023), show a 70B LLM used this way compresses text better than gzip or specialized compressors, and compresses images and audio surprisingly well too.
    - To compress text from people well, you must predict what those people will write next. That requires modeling their knowledge and reasoning.

    This is the information-theoretic argument for why scaling next-token prediction produces capability.

---

## 4.4 Cross-entropy

**The idea:** if you encode data from $p$ using a code built for a *different* distribution $q$, you pay $-\log q(x)$ bits per outcome. The average is the cross-entropy.

!!! definition "Definition 4.2 (Cross-entropy)"
    $$
    H(p,q) = \E_{x\sim p}[-\log q(x)] = -\sum_xp(x)\log q(x).
    $$

    The expected surprise of a **model $q$** when the data actually comes from **$p$**.

    Note the roles: the average is over the truth $p$; the log is of the model $q$.

!!! example "Worked example"
    Truth $p = (0.5, 0.5)$ (a fair coin). Model $q = (0.9, 0.1)$ (thinks heads is very likely).

    | | heads | tails |
    |---|---|---|
    | $p$ (truth) | 0.5 | 0.5 |
    | $q$ (model) | 0.9 | 0.1 |
    | surprise $-\ln q$ | 0.105 | 2.303 |

    - $H(p,q) = 0.5(0.105) + 0.5(2.303) = 1.204$ nats.
    - $H(p) = \ln 2 = 0.693$ nats.

    The wrong model costs an extra $0.511$ nats per flip, mostly from being very surprised by tails.

Using the wrong model always costs something:

!!! theorem "Theorem 4.7 (Gibbs' inequality)"
    $H(p,q)\ge H(p)$, with equality if and only if $q=p$.

This is the next theorem in disguise, since $H(p,q) - H(p) = \KL(p\Vert q)$.

---

## 4.5 KL divergence

**The idea:** KL divergence is the *extra* cost of using model $q$ when the truth is $p$. It is zero only when they match.

!!! definition "Definition 4.3 (Kullback–Leibler divergence)"
    $$
    \KL(p\,\Vert\,q) = \sum_xp(x)\log\frac{p(x)}{q(x)} = \E_{x\sim p}\left[\log\frac{p(x)}{q(x)}\right] = H(p,q) - H(p).
    $$

    - It is the **extra** nats per sample you pay for using $q$ when the truth is $p$.
    - If $q(x)=0$ somewhere that $p(x)>0$, it is $+\infty$: the model says "impossible" about something that happens.

    The last equality: $\log\frac{p}{q} = \log p - \log q$, so the average splits into $-H(p) + H(p,q)$.

For the coin example: $\KL(p\Vert q) = 1.204 - 0.693 = 0.511$ nats.

!!! theorem "Theorem 4.8 (KL is non-negative)"
    $\KL(p\Vert q)\ge0$, with equality if and only if $p=q$.

???+ proof "Proof (Jensen)"
    **Step 1 · Flip the fraction** so the log is of $q/p$: $\KL(p\Vert q) = \E_{x\sim p}\left[-\log\frac{q(x)}{p(x)}\right]$.

    **Step 2 · Jensen** for the convex function $-\log$ (sum over $x$ with $p(x)>0$): the average of $-\log$ is at least $-\log$ of the average:

    $$
    \E_{x\sim p}\left[-\log\frac{q(x)}{p(x)}\right]\ \ge\ -\log\E_{x\sim p}\left[\frac{q(x)}{p(x)}\right].
    $$

    **Step 3 · The $p$'s cancel inside:**

    $$
    \E_{x\sim p}\left[\frac{q(x)}{p(x)}\right] = \sum_{x:\,p(x)>0}p(x)\frac{q(x)}{p(x)} = \sum_{x:\,p(x)>0}q(x)\ \le\ 1 .
    $$

    **Step 4 · Finish.** $-\log$ is decreasing, so $-\log(\text{something}\le1)\ge-\log1 = 0$.

    **Step 5 · Equality.** Jensen is tight only if $q(x)/p(x)$ is the same constant everywhere $p>0$, and Step 3 is tight only if $q$ puts all its mass there. Together that forces $q = p$. $\blacksquare$

??? proof "Alternative proof (the inequality $\log t\le t-1$)"
    **Step 1 · A tangent-line bound.** $\log t\le t-1$ for all $t>0$, with equality only at $t=1$. (The line $t-1$ is the tangent to the concave $\log$ at $t=1$, and a concave function lies below its tangents.)

    **Step 2 · Apply it with $t = q/p$:**

    $$
    -\KL(p\Vert q) = \sum p\log\frac qp\le\sum p\left(\frac qp-1\right) = \sum q - \sum p\le 1 - 1 = 0 .\ \blacksquare
    $$

**KL is not a distance.** It is not symmetric, and it doesn't satisfy the triangle inequality. Reverse the coin example:

| | Formula | Value |
|---|---|---|
| $\KL(p\Vert q)$ | $0.5\ln\frac{0.5}{0.9}+0.5\ln\frac{0.5}{0.1}$ | $0.511$ nats |
| $\KL(q\Vert p)$ | $0.9\ln\frac{0.9}{0.5}+0.1\ln\frac{0.1}{0.5}$ | $0.368$ nats |

The asymmetry matters a lot in practice.

<div class="widget" data-widget="kl"></div>

### Forward vs reverse KL

Suppose the truth $p$ has two peaks (bimodal), and we fit a $q_\theta$ that can only have one peak.

| | **Forward KL** $\min_\theta\KL(p\Vert q_\theta)$ | **Reverse KL** $\min_\theta\KL(q_\theta\Vert p)$ |
|---|---|---|
| averages over samples from | the truth $p$ | the model $q$ |
| big penalty when | $p>0$ but $q\approx0$ (model misses something real) | $q>0$ but $p\approx0$ (model invents something) |
| resulting behavior | **covers every mode**, smeared between them | **locks onto one mode**, ignores the rest |
| nickname | "mean-seeking", "mass-covering" | "mode-seeking" |
| who uses it | **maximum likelihood** (Thm 4.9) | the RLHF penalty |

Forward KL is why pretrained LMs put probability on everything people might write, including the bad stuff.

!!! llm "Where each KL appears"
    - **Pretraining / SFT:** forward KL from the data distribution (via cross-entropy).
    - **Knowledge distillation:** forward KL from teacher to student, $\KL(p_\text{teacher}\Vert p_\text{student})$, on the teacher's soft probabilities ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)). Recent on-policy distillation uses reverse KL ([Agarwal et al., 2024](https://arxiv.org/abs/2306.13649)).
    - **RLHF:** maximize reward minus $\beta\,\KL(\pi_\theta\Vert\pi_\text{ref})$. This is a **reverse** KL, evaluated on the policy's own samples. It keeps the policy close to the SFT model, and it's part of why RL-tuned models are less diverse ("mode collapse") than base models (Ch. 21).

---

## 4.6 Cross-entropy loss = maximum likelihood = minimum KL

**The idea:** three different-sounding objectives — "maximize likelihood", "minimize cross-entropy", "minimize KL from the data" — are the same optimization problem.

The bridge is the **empirical distribution** $\hat p$ of a dataset: the fraction of the data equal to each $x$,

$$
\hat p(x) = \frac{\#\{i: x^{(i)} = x\}}{n}.
$$

!!! theorem "Theorem 4.9 (The three views of the LLM loss coincide)"
    For any model $q_\theta$,

    $$
    \underbrace{-\frac1n\sum_{i=1}^n\log q_\theta(x^{(i)})}_{\text{average NLL (MLE)}} \;=\; \underbrace{H(\hat p, q_\theta)}_{\text{cross-entropy}} \;=\; H(\hat p) + \underbrace{\KL(\hat p\,\Vert\,q_\theta)}_{\text{forward KL}}.
    $$

    $H(\hat p)$ doesn't depend on $\theta$. So **minimizing the NLL, the cross-entropy, and the forward KL are the same optimization problem.**

???+ proof "Proof"
    **Step 1 · Group the NLL by value.** In the average over data points, each distinct value $x$ appears $n\hat p(x)$ times:

    $$
    -\frac1n\sum_i\log q(x^{(i)}) = -\frac1n\sum_x n\,\hat p(x)\log q(x) = -\sum_x\hat p(x)\log q(x) = H(\hat p, q).
    $$

    **Step 2 · Split the cross-entropy** using Definition 4.3: $H(\hat p, q) = H(\hat p) + \KL(\hat p\Vert q)$. $\blacksquare$

!!! example "Worked example: a three-point dataset"
    Data: `a, a, b`, so $\hat p = (\frac23, \frac13)$. Model: $q = (\frac12,\frac12)$.

    | View | Calculation | Value (nats) |
    |---|---|---|
    | average NLL | $-\frac13(\ln\frac12+\ln\frac12+\ln\frac12)$ | $0.693$ |
    | $H(\hat p)$ | $-\frac23\ln\frac23-\frac13\ln\frac13$ | $0.637$ |
    | $\KL(\hat p\Vert q)$ | $\frac23\ln\frac{2/3}{1/2}+\frac13\ln\frac{1/3}{1/2}$ | $0.057$ |
    | $H(\hat p) + \KL$ | | $0.693$ ✓ |

    The best model, $q = \hat p$, would make the KL zero and the loss $0.637$.

For a language model, the "outcome" at each position is the next token, and the loss averages over all $B\cdot T$ positions in a batch:

$$
\mathcal{L}(\theta) = -\frac{1}{BT}\sum_{b=1}^B\sum_{t=1}^T\log p_\theta\big(x^{(b)}_{t+1}\mid x^{(b)}_{\le t}\big).
$$

For one position, with one-hot target $\mathbf{y}$ and prediction $\mathbf{p} = \softmax(\mathbf{z})$, the cross-entropy is

$$
-\sum_iy_i\log p_i = -\log p_y = -z_y + \log\sum_je^{z_j},
$$

with gradient $\mathbf{p}-\mathbf{y}$ (Thm 2.8).

!!! warning "The loss can't go to zero"
    The irreducible part is $H(\hat p)$ — or, for the true distribution of text, its **entropy rate**.

    - Natural language is genuinely uncertain ("The weather today is ___"), so even a perfect model has loss well above 0.
    - Scaling laws (Ch. 17) model this explicitly as the constant $E$ in $L(N,D) = E + A/N^\alpha + B/D^\beta$.

---

## 4.7 Perplexity and bits-per-byte

**The idea:** perplexity converts a loss into "how many equally likely choices is the model torn between?"

!!! definition "Definition 4.4 (Perplexity)"
    The **perplexity** of a model on a token sequence is its exponentiated average NLL:

    $$
    \mathrm{PPL} = \exp\Big(-\frac1T\sum_{t=1}^T\log p_\theta(x_t\mid x_{<t})\Big) = e^{\mathcal{L}}.
    $$

**Interpretation.** A uniform distribution over $k$ choices has loss $-\log\frac1k = \log k$, so its perplexity is $e^{\log k} = k$ exactly. So perplexity is the **effective branching factor**.

| Loss $\mathcal{L}$ (nats) | Perplexity $e^{\mathcal{L}}$ | Meaning |
|---|---|---|
| $10.82$ | $50{,}257$ | uniform over GPT-2's vocabulary (untrained) |
| $3.3$ | $27$ | GPT-2 small on web text, roughly |
| $3.0$ | $20$ | like choosing among 20 equally likely tokens |
| $2.0$ | $7.4$ | |
| $0.69$ | $2$ | a fair coin flip per token |
| $0$ | $1$ | perfect certainty (impossible for real text) |

Frontier models reach far lower perplexities than GPT-2.

!!! warning "Perplexity depends on the tokenizer"
    Loss per *token* isn't comparable across models with different tokenizers: a tokenizer with longer tokens makes fewer, harder predictions.

    To compare fairly, normalize by a tokenizer-independent unit. **Bits per byte** (bpb) is the standard:

    $$
    \text{bpb} = \frac{\text{total NLL in nats}}{\ln 2\times\text{number of UTF-8 bytes}} = \frac{\mathcal{L}_\text{token}}{\ln 2}\cdot\frac{\#\text{tokens}}{\#\text{bytes}}.
    $$

    **Example:** loss 3.0 nats/token, with about 4 bytes per token (typical for English BPE): $\frac{3.0}{0.693}\cdot\frac14\approx1.08$ bpb.

    Karpathy's [nanochat](https://github.com/karpathy/nanochat) reports bpb for exactly this reason.

---

## 4.8 Mutual information

**The idea:** mutual information is how many nats of uncertainty about $Y$ disappear once you learn $X$.

!!! definition "Definition 4.5 (Mutual information)"
    $$
    I(X;Y) = \KL\big(p(x,y)\,\Vert\,p(x)p(y)\big) = H(Y) - H(Y\mid X) = H(X) - H(X\mid Y).
    $$

    - The KL form: how far the joint is from "independent".
    - The entropy form: uncertainty about $Y$ before, minus uncertainty after learning $X$.

    It is symmetric, $\ge0$ (Thm 4.8), and zero if and only if $X$ and $Y$ are independent.

??? proof "Proof that the forms agree"
    **Step 1 · Write the KL form out:**

    $$
    \KL(p(x,y)\Vert p(x)p(y)) = \sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)} .
    $$

    **Step 2 · Use $p(x,y) = p(x)p(y\mid x)$** to cancel $p(x)$ inside the log:

    $$
    = \sum p(x,y)\log\frac{p(y\mid x)}{p(y)} = \sum p(x,y)\log p(y\mid x) - \sum p(x,y)\log p(y).
    $$

    **Step 3 · Recognize the pieces.** The first sum is $-H(Y\mid X)$. In the second, summing over $x$ first leaves $\sum_yp(y)\log p(y) = -H(Y)$. So the total is $H(Y) - H(Y\mid X)$.

    **Step 4 · Swap the roles** of $X$ and $Y$ for the other form. $\blacksquare$

!!! example "Worked example: weather and umbrellas (from Ch. 3)"
    Joint table: sun/umbrella 0.05, sun/none 0.55, rain/umbrella 0.30, rain/none 0.10.

    | Quantity | Calculation | Value (nats) |
    |---|---|---|
    | $H(Y)$, umbrella $(0.35, 0.65)$ | $-0.35\ln0.35 - 0.65\ln0.65$ | $0.647$ |
    | $H(Y\mid\text{sun})$, $(0.083, 0.917)$ | | $0.287$ |
    | $H(Y\mid\text{rain})$, $(0.75, 0.25)$ | | $0.562$ |
    | $H(Y\mid X)$ | $0.6(0.287) + 0.4(0.562)$ | $0.397$ |
    | $I(X;Y)$ | $0.647 - 0.397$ | $0.250$ |

    Knowing the weather removes about 39% of the uncertainty about umbrellas.

Two consequences:

1. **Conditioning reduces entropy on average:** $H(Y\mid X)\le H(Y)$, because $I\ge0$.
    - More context can only help a language model, in expectation.
    - That's why longer contexts lower the loss, and why the loss *per position* falls with position in a sequence.
2. **Data-processing inequality.** If $X\to Y\to Z$ is a Markov chain ($Z$ depends on $X$ only through $Y$), then $I(X;Z)\le I(X;Y)$.
    - No processing of $Y$ can create information about $X$ that $Y$ didn't already contain.
    - Each Transformer layer is a deterministic function of the previous one. So the residual stream can never *gain* information about the input — only make it more *usable*.

---

## 4.9 KL estimators (used in RLHF and GRPO)

**The idea:** when we can't sum over every possible outcome, we estimate KL by averaging a formula over samples. Some formulas are much better than others.

In RL fine-tuning we need $\KL(\pi_\theta\Vert\pi_\text{ref})$ for a policy over *sequences*. There are astronomically many sequences, so we can't sum over them. We estimate it from samples $x\sim\pi_\theta$.

!!! recall "What makes an estimator good?"
    An **estimator** is a formula $k(x)$ whose average over samples approximates the quantity we want.

    | Property | Meaning |
    |---|---|
    | **unbiased** | $\E[k(x)]$ equals the true value exactly; errors cancel on average |
    | **low variance** | individual samples don't swing wildly; fewer samples needed |
    | **non-negative** | each sample respects "KL $\ge 0$", which keeps per-token penalties sensible |

Let $r(x) = \pi_\text{ref}(x)/\pi_\theta(x)$ be the probability ratio. A useful fact:

$$
\E_{x\sim\pi_\theta}[r(x)] = \sum_x\pi_\theta(x)\frac{\pi_\text{ref}(x)}{\pi_\theta(x)} = \sum_x\pi_\text{ref}(x) = 1 .
$$

Following [J. Schulman's note on KL approximation](http://joschu.net/blog/kl-approx.html):

| Estimator | Formula | Unbiased? | Always $\ge 0$? |
|---|---|---|---|
| $k_1$ | $-\log r$ | yes (it is the definition of KL) | no (high variance) |
| $k_2$ | $\tfrac12(\log r)^2$ | no (low bias) | yes |
| $k_3$ | $(r-1)-\log r$ | **yes** | **yes** |

!!! theorem "Proposition 4.10 (The $k_3$ estimator)"
    $k_3 = (r-1)-\log r$ is an unbiased, non-negative estimator of $\KL(\pi_\theta\Vert\pi_\text{ref})$.

???+ proof "Proof"
    **Unbiased.** Use linearity, $\E[r] = 1$, and the definition $\KL = \E_{\pi_\theta}[\log\frac{\pi_\theta}{\pi_\text{ref}}] = \E[-\log r]$:

    $$
    \E[k_3] = \E[r]-1 + \E[-\log r] = 1 - 1 + \KL = \KL .
    $$

    **Non-negative.** $\log r\le r-1$ for all $r>0$ (the tangent-line bound from the second proof of Thm 4.8). Rearranged: $(r-1) - \log r\ge0$.

    **Why lower variance.** $k_3$ is $k_1$ plus the term $r - 1$, which has mean 0. Adding a zero-mean term doesn't change the average, but this one is correlated with $k_1$'s noise and cancels much of it. Such a term is called a **control variate**. $\blacksquare$

This is exactly `kl_k3` in [`code/alignment.py`](https://github.com/kdellis135/llm_deep_dive/blob/main/code/alignment.py), the per-token KL penalty in GRPO (Ch. 22):

```python
--8<-- "code/alignment.py:grpo"
```

---

## 4.10 Temperature, label smoothing, and entropy in practice

**Temperature.**

- Divides the logits by $T$ before softmax.
- As $T$ grows, the entropy of $\softmax(\mathbf{z}/T)$ increases steadily toward $\log V$ (Exercise 5).
- Sampling at low $T$ trades diversity for likelihood (Ch. 18).

**Label smoothing.**

- Replaces the one-hot target $\mathbf{y}$ with $(1-\epsilon)\mathbf{y}+\epsilon/V$.
- The loss becomes $(1-\epsilon)H(\mathbf{y},\mathbf{p}) + \epsilon H(\mathbf{u},\mathbf{p})$, where $\mathbf{u}$ is uniform. The second term penalizes overconfident predictions.
- It was used in the original Transformer. It's rarely used for LLM pretraining, because it biases the model away from calibrated probabilities.

**Calibration.**

- A model is **calibrated** if, among predictions made with confidence 0.8, 80% are correct.
- Cross-entropy is a *proper scoring rule*: its expected value is minimized only by reporting the true distribution (that's Gibbs' inequality).
- So minimizing it makes pretrained LMs well calibrated. RLHF tends to degrade calibration ([GPT-4 technical report](https://arxiv.org/abs/2303.08774), Fig. 8).

---

## Key takeaways

| Concept | One-line summary | Where it matters |
|---|---|---|
| Information | $-\log p$: the only measure that adds for independent events | everything here |
| Entropy | expected surprise, between $0$ and $\log n$ | loss at init, temperature |
| Jensen | $\varphi(\E X)\le\E\varphi(X)$ for convex $\varphi$ | proves every inequality here |
| Source coding | best average code length ≈ entropy; loss in bits = compression rate | why LLMs work |
| Cross-entropy | $H(p,q)\ge H(p)$; the extra is $\KL(p\Vert q)\ge0$ | the training loss |
| **Three views** | cross-entropy loss = average NLL = MLE = forward KL + constant | pretraining |
| Perplexity | $e^{\text{loss}}$, the effective branching factor; compare across tokenizers with bpb | evaluation |
| Forward vs reverse KL | forward covers modes (MLE); reverse seeks modes (RLHF) | post-training |
| Mutual information | $H(Y) - H(Y\mid X)$; context never hurts on average | context length |
| $k_3$ | $(r-1)-\log r$: unbiased and non-negative | GRPO |

## Exercises

1. Compute the entropy in bits of a fair die, and of a die that lands on 6 with probability 1/2 and on the other faces uniformly.
2. Show $H(X,Y)\le H(X)+H(Y)$ with equality if and only if $X$ and $Y$ are independent.
3. A model's validation loss is 2.30 nats/token and its tokenizer averages 4.2 bytes per token. What are its perplexity and bits-per-byte?
4. Compute $\KL(p\Vert q)$ and $\KL(q\Vert p)$ for $p=(0.5,0.5)$, $q=(0.9,0.1)$. Which is larger and why?
5. Prove that the entropy of $\softmax(\mathbf{z}/T)$ is non-decreasing in $T$. (Hint: differentiate with respect to $\beta = 1/T$ and show the derivative is $-\beta\Var_{p}(z)$.)
6. **Code:** train a character-level bigram model (Ch. 9) on any text, convert its loss to bits-per-character, and compare with the size of `gzip -9` on the same text. Which compresses better?

??? success "Solutions"
    1. Fair die: $\log_26\approx2.585$ bits. Biased die: $-\frac12\log_2\frac12 - 5\cdot\frac1{10}\log_2\frac1{10} = 0.5 + 0.5\log_210\approx2.161$ bits.
    2. $H(X)+H(Y)-H(X,Y) = H(Y) - H(Y\mid X) = I(X;Y)\ge0$ (chain rule, then Definition 4.5). And $I = 0$ if and only if $p(x,y) = p(x)p(y)$ (Thm 4.8).
    3. PPL $=e^{2.30}\approx9.97$. bpb $=2.30/(\ln2\cdot4.2)\approx0.79$.
    4. $\KL(p\Vert q) = 0.5\ln\frac{0.5}{0.9}+0.5\ln\frac{0.5}{0.1}\approx0.511$ nats. $\KL(q\Vert p) = 0.9\ln1.8+0.1\ln0.2\approx0.368$ nats. The first is larger because $p$ puts substantial mass (0.5) where $q$ is small (0.1), which forward KL punishes heavily.
    5. With $p_i\propto e^{\beta z_i}$ and $Z = \sum_i e^{\beta z_i}$: $H = -\sum p_i(\beta z_i - \log Z) = \log Z - \beta\,\E_p[z]$. Using $\frac{d}{d\beta}\log Z = \E_p[z]$ and $\frac{d}{d\beta}\E_p[z] = \Var_p(z)$ gives $\frac{dH}{d\beta} = \E_p[z] - \E_p[z] - \beta\Var_p(z) = -\beta\Var_p(z)\le0$. Since $\beta = 1/T$ decreases as $T$ grows, $H$ increases with $T$.
    6. A character bigram model gets roughly 3.3–3.6 bits/char on English (close to Shannon's second-order estimate of 3.3). That is *worse* than `gzip -9` (about 2.5–3 bits/char), because gzip exploits long repeated strings. A small character Transformer beats both: nanoGPT's Shakespeare-char model reaches about 1.47 nats ≈ 2.1 bits/char on validation, and large models go far lower. Better prediction means better compression.

## References

- C. E. Shannon, [A Mathematical Theory of Communication](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) (1948). The founding paper; Section 7 discusses the entropy of English.
- T. M. Cover & J. A. Thomas, *Elements of Information Theory*, 2nd ed., Wiley (2006), Ch. 2, 5.
- D. J. C. MacKay, [*Information Theory, Inference, and Learning Algorithms*](https://www.inference.org.uk/itprnn/book.html) (2003).
- G. Delétang et al., [Language Modeling Is Compression](https://arxiv.org/abs/2309.10668) (ICLR 2024).
- G. Hinton, O. Vinyals, J. Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (2015).
- J. Schulman, [Approximating KL Divergence](http://joschu.net/blog/kl-approx.html) (2020).
- C. Olah, [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/) (2015). A beautiful visual introduction.
