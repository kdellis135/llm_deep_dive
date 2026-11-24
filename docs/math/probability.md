# 3. Probability & Statistics

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 1](linear-algebra.md), [Ch. 2](calculus.md) (derivatives, integrals). No prior probability is assumed.

    **You will learn:**

    - random variables and distributions, from coin flips up
    - joint, marginal and conditional probability, and the **chain rule of probability** (the factorization every LLM is built on)
    - Bayes' theorem
    - expectation and variance (with proofs)
    - the categorical and Gaussian distributions
    - the law of large numbers, the central limit theorem, and Monte Carlo
    - **maximum likelihood estimation**, MAP estimation and why it equals weight decay
    - how to sample from a softmax (including the Gumbel-max trick)

    **Why it matters:** an LLM *is* a probability distribution over text.

    - Training is maximum likelihood.
    - Sampling is drawing from a categorical distribution.
    - SGD is a Monte Carlo estimate of a gradient.

!!! watch "Watch"
    - 3Blue1Brown: [Bayes' theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM) · [But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo)
    - Joe Blitzstein, [Harvard Stat 110: Probability](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo), the gold-standard full course ([course site & free book](https://projects.iq.harvard.edu/stat110)).

---

## Roadmap

| § | Question it answers | Used later in |
|---|---|---|
| Warm-up | What is a probability? | everything |
| 3.1 | What is a random variable? | everything |
| 3.2 | How do probabilities of several things combine? | the definition of an LLM (Ch. 9) |
| 3.3 | How do we reverse a conditional? | evals, guardrails (Ch. 29) |
| 3.4 | What is an average, and how spread out is it? | initialization (Ch. 8), attention scaling |
| 3.5 | Which distributions matter? | LLM outputs, initialization |
| 3.6 | Why do averages become reliable? | SGD, error bars |
| 3.7 | How do we fit a model to data? | pretraining loss |
| 3.8 | How do priors become regularizers? | weight decay, smoothing |
| 3.9 | How do we sample a token? | decoding (Ch. 18) |

---

## Warm-up: what a probability is

Probability starts with three ingredients. Use a six-sided die as the running example.

| Ingredient | Meaning | Die example |
|---|---|---|
| **sample space** $\Omega$ | every possible outcome | $\{1,2,3,4,5,6\}$ |
| **event** | a set of outcomes | "even" $=\{2,4,6\}$ |
| **probability** $P$ | a number in $[0,1]$ for each event | $P(\text{even}) = 3/6 = 1/2$ |

Every probability obeys three rules (the **axioms**). Everything in this chapter is built from them.

1. **Never negative:** $P(A)\ge0$.
2. **Something happens:** $P(\Omega) = 1$.
3. **Non-overlapping events add:** if $A$ and $B$ can't both happen, $P(A\text{ or }B) = P(A) + P(B)$.

One consequence is used constantly, the **complement rule**: $P(\text{not }A) = 1 - P(A)$. (Because "$A$" and "not $A$" don't overlap and together make up everything.)

---

## 3.1 Random variables and distributions

**The idea:** a random variable is a number whose value depends on chance. Its distribution lists how likely each value is.

A **random variable** $X$ is a quantity whose value is uncertain. Convention: uppercase $X$ for the variable, lowercase $x$ for a particular value it can take.

### Discrete random variables

A **discrete** variable takes values you can list (e.g. the next token). Its distribution is a **probability mass function** (pmf):

$$
p(x) = P(X=x),\qquad p(x)\ge0,\qquad \sum_xp(x)=1.
$$

!!! example "Worked example: two variables from one die"
    Let $X$ be the roll, and $Y = 1$ if the roll is even, $0$ if odd.

    | $x$ | 1 | 2 | 3 | 4 | 5 | 6 |
    |---|---|---|---|---|---|---|
    | $p(x)$ | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |

    | $y$ | 0 | 1 |
    |---|---|---|
    | $p(y)$ | 1/2 | 1/2 |

    Both columns sum to 1. ✓

### Continuous random variables

A **continuous** variable can take any value in a range (e.g. a weight at initialization). Now the probability of any *exact* value is 0. Instead there is a **density** $p(x)\ge0$, and probabilities are **areas** under it (Ch. 2, §2.8):

$$
P(a\le X\le b) = \int_a^bp(x)\,dx,\qquad \int_{-\infty}^{\infty} p(x)\,dx = 1.
$$

!!! example "Worked example: a density can exceed 1"
    $X$ uniform on $[0, 0.5]$: every value in that interval is equally likely.

    - The density must be flat, and its total area must be 1.
    - The interval has width 0.5, so the height is $1/0.5 = 2$.
    - $P(0.1\le X\le 0.2) = 2\times0.1 = 0.2$.

    A density of 2 is fine. Only *areas* under the density have to be $\le 1$.

The **cumulative distribution function** (CDF) is $F(x) = P(X\le x)$: the running total of probability up to $x$. For the die, $F(2) = 2/6$ and $F(6) = 1$.

---

## 3.2 Joint, marginal, conditional

**The idea:** when there are two random things, a table of "both at once" contains all the information. Everything else is read off that table.

!!! example "The running example: weather and umbrellas"
    One random day. $X$ = weather, $Y$ = whether someone carries an umbrella.

    | | umbrella | no umbrella | **row total** |
    |---|---|---|---|
    | **sun** | 0.05 | 0.55 | **0.60** |
    | **rain** | 0.30 | 0.10 | **0.40** |
    | **column total** | **0.35** | **0.65** | **1.00** |

    The four inner numbers are the **joint** distribution $p(x, y)$. The totals are **marginals**.

The three ways to read the table:

| Concept | Formula | In the table | Example |
|---|---|---|---|
| **joint** $p(x,y)$ | probability of both | one inner cell | $p(\text{rain},\text{umbrella}) = 0.30$ |
| **marginal** $p(x)$ | $\sum_yp(x,y)$ (the **sum rule**) | a row or column total | $p(\text{rain}) = 0.30 + 0.10 = 0.40$ |
| **conditional** $p(y\mid x)$ | $\dfrac{p(x,y)}{p(x)}$ | one row, rescaled to sum to 1 | $p(\text{umbrella}\mid\text{rain}) = \frac{0.30}{0.40} = 0.75$ |

The conditional formula says: **once you know it's raining, only the rain row matters.** Divide by its total so the row sums to 1 again.

Rearranging the conditional gives the **product rule**:

$$
p(x,y) = p(x)\,p(y\mid x).
$$

"The probability of both = probability of the first × probability of the second given the first." Check: $0.40\times0.75 = 0.30$. ✓

### The chain rule of probability

Apply the product rule over and over, and you can factor *any* joint distribution into a chain of conditionals.

!!! theorem "Theorem 3.1 (Chain rule of probability)"
    For any random variables $X_1,\dots,X_T$:

    $$
    p(x_1,\dots,x_T) = \prod_{t=1}^{T}p(x_t\mid x_1,\dots,x_{t-1}) = \prod_{t=1}^{T}p(x_t\mid x_{<t}).
    $$

    ($\prod$ is like $\sum$ but multiplies. $x_{<t}$ is shorthand for $x_1,\dots,x_{t-1}$.)

    Written out for $T = 3$:

    $$
    p(x_1,x_2,x_3) = p(x_1)\;p(x_2\mid x_1)\;p(x_3\mid x_1,x_2).
    $$

???+ proof "Proof (by induction)"
    Induction proves a statement for every $T$ in two moves: show it for $T=1$, then show "true for $T-1$" implies "true for $T$".

    **Step 1 · Base case $T=1$.** The claim is $p(x_1) = p(x_1)$. True.

    **Step 2 · Assume it holds for $T-1$ variables:** $p(x_{<T}) = \prod_{t=1}^{T-1}p(x_t\mid x_{<t})$.

    **Step 3 · Product rule** with "first thing" $= x_{<T}$ (all earlier variables, treated as one) and "second thing" $= x_T$:

    $$
    p(x_{1:T}) = p(x_{<T})\;p(x_T\mid x_{<T}).
    $$

    **Step 4 · Substitute the assumption** from Step 2 for $p(x_{<T})$. The result is the product of all $T$ factors. $\blacksquare$

!!! llm "This is the definition of an autoregressive language model"
    Theorem 3.1 holds for **any** distribution, with no assumptions at all. So a model that outputs $p_\theta(x_t\mid x_{<t})$ — "the next token given everything so far" — automatically defines a full distribution over whole sequences.

    | Factor | Example probability | $\log$ (nats) |
    |---|---|---|
    | $p_\theta(\text{The})$ | $0.05$ | $-3.00$ |
    | $p_\theta(\text{cat}\mid\text{The})$ | $0.01$ | $-4.61$ |
    | $p_\theta(\text{sat}\mid\text{The cat})$ | $0.20$ | $-1.61$ |
    | **sequence** $p_\theta(\text{"The cat sat"})$ | $0.05\times0.01\times0.20 = 10^{-4}$ | $-9.21$ |

    - GPT, Llama and Claude are all models of the right-hand side.
    - **Generation** samples one factor at a time (Ch. 18).
    - **Training** maximizes the product on real text (§3.7). Note the logs add — we'll use that.
    - The only modeling question is *how to compute $p_\theta(x_t\mid x_{<t})$ from the prefix*. The answer evolved from n-gram counts (Ch. 9) to Transformers (Ch. 12–14).

### Independence

$X$ and $Y$ are **independent** if knowing one tells you nothing about the other:

$$
p(y\mid x) = p(y)\quad\Longleftrightarrow\quad p(x,y) = p(x)\,p(y).
$$

!!! example "Worked example: are weather and umbrellas independent?"
    Test one cell: $p(\text{rain})\,p(\text{umbrella}) = 0.40\times0.35 = 0.14$. But the table says $p(\text{rain},\text{umbrella}) = 0.30$.

    Not equal, so **dependent**. Of course: people carry umbrellas when it rains.

A **Markov assumption** is a deliberate simplification of the chain rule. An order-$(n-1)$ Markov model assumes only the last $n-1$ items matter:

$$
p(x_t\mid x_{<t}) = p(x_t\mid x_{t-n+1},\dots,x_{t-1}).
$$

That is the n-gram model of Ch. 9. Transformers drop this assumption (up to the length of their context window).

---

## 3.3 Bayes' theorem

**The idea:** Bayes' theorem turns "$P(\text{evidence}\mid\text{cause})$" into "$P(\text{cause}\mid\text{evidence})$".

!!! theorem "Theorem 3.2 (Bayes)"
    $$
    p(\theta\mid\mathcal{D}) = \frac{p(\mathcal{D}\mid\theta)\,p(\theta)}{p(\mathcal{D})},\qquad p(\mathcal{D}) = \sum_\theta p(\mathcal{D}\mid\theta)p(\theta).
    $$

    | Term | Name | Meaning |
    |---|---|---|
    | $p(\theta)$ | **prior** | belief about $\theta$ before seeing data |
    | $p(\mathcal{D}\mid\theta)$ | **likelihood** | how well $\theta$ explains the data |
    | $p(\mathcal{D})$ | **evidence** | total probability of the data (a normalizer) |
    | $p(\theta\mid\mathcal{D})$ | **posterior** | belief after seeing data |

???+ proof "Proof"
    **Step 1 · Write the joint two ways** with the product rule (order doesn't matter):

    $$
    p(\theta,\mathcal{D}) = p(\theta\mid\mathcal{D})\,p(\mathcal{D}) = p(\mathcal{D}\mid\theta)\,p(\theta).
    $$

    **Step 2 · Divide** both sides by $p(\mathcal{D})$.

    **Step 3 · The evidence** is a marginal: sum the joint $p(\mathcal{D}\mid\theta)p(\theta)$ over all $\theta$ (sum rule). $\blacksquare$

!!! theorem "Example 3.3 (Base rates)"
    A toxicity classifier:

    - catches 99% of toxic texts (sensitivity),
    - wrongly flags 2% of clean texts (false-positive rate),
    - and 1% of all texts are toxic.

    **A text is flagged. How likely is it actually toxic?**

The easiest way to see the answer is to imagine 10,000 texts:

| | flagged | not flagged | total |
|---|---|---|---|
| **toxic** (1%) | $99$ (99% of 100) | $1$ | $100$ |
| **clean** (99%) | $198$ (2% of 9,900) | $9{,}702$ | $9{,}900$ |
| **total** | $297$ | $9{,}703$ | $10{,}000$ |

Among the 297 flagged texts, only 99 are toxic:

$$
P(\text{toxic}\mid\text{flag}) = \frac{99}{297} = \frac{0.99\cdot0.01}{0.99\cdot0.01 + 0.02\cdot0.99} = \frac13 .
$$

**Two thirds of flags are false alarms**, even with a "99% accurate" classifier. The rare class is swamped by mistakes on the common class. This matters whenever you build LLM guardrails and evals (Ch. 29).

---

## 3.4 Expectation and variance

**The idea:** the expectation is the long-run average. The variance measures how far values typically stray from it.

!!! definition "Definition 3.1 (Expectation, variance, covariance)"
    | Quantity | Formula | Meaning |
    |---|---|---|
    | **expectation** (mean) | $\E[X] = \sum_xx\,p(x)$, or $\int x\,p(x)\,dx$ | probability-weighted average |
    | expectation of a function | $\E[g(X)] = \sum_xg(x)\,p(x)$ | average of $g$ of the outcome |
    | **variance** | $\Var(X) = \E[(X-\E X)^2]$ | average squared distance from the mean |
    | shortcut | $\Var(X) = \E[X^2]-(\E X)^2$ | (Exercise 1) |
    | **standard deviation** | $\sqrt{\Var X}$ | spread, in the original units |
    | **covariance** | $\Cov(X,Y) = \E[(X-\E X)(Y-\E Y)]$ | do $X$ and $Y$ move together? |

!!! example "Worked example: one die roll"
    | $x$ | 1 | 2 | 3 | 4 | 5 | 6 | |
    |---|---|---|---|---|---|---|---|
    | $p(x)$ | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | |
    | $x\,p(x)$ | 1/6 | 2/6 | 3/6 | 4/6 | 5/6 | 6/6 | sum $= 21/6 = 3.5$ |
    | $x^2p(x)$ | 1/6 | 4/6 | 9/6 | 16/6 | 25/6 | 36/6 | sum $= 91/6\approx15.17$ |

    - $\E X = 3.5$ (never actually rolled — it's an average).
    - $\Var X = \E[X^2] - (\E X)^2 = 15.17 - 12.25 = 2.92$.
    - Standard deviation $\approx1.71$.

**Reading covariance:** positive means "when $X$ is above its mean, $Y$ tends to be too"; negative means the opposite; zero means no linear relationship.

!!! theorem "Theorem 3.4 (Properties)"
    1. **Linearity:** $\E[aX+bY] = a\E X + b\E Y$, *always* — no independence needed.
    2. **Scaling variance:** $\Var(aX+b) = a^2\Var X$. (Shifting doesn't change spread; scaling by $a$ scales squared distances by $a^2$.)
    3. **Variance of a sum:** $\Var(X+Y) = \Var X + \Var Y + 2\Cov(X,Y)$. If $X,Y$ are independent, $\Cov = 0$ and **the variances add**.
    4. **Averaging shrinks variance:** if $X_1,\dots,X_n$ are independent, each with variance $\sigma^2$, their mean $\bar X = \frac1n\sum X_i$ has $\Var\bar X = \sigma^2/n$.

???+ proof "Proofs"
    **Property 1.**

    *Step 1.* Write the expectation over the joint distribution: $\E[aX+bY] = \sum_{x,y}(ax+by)\,p(x,y)$.

    *Step 2.* Split the sum and pull out constants: $a\sum_{x}x\sum_yp(x,y) + b\sum_{y}y\sum_xp(x,y)$.

    *Step 3.* The inner sums are marginals (sum rule): $\sum_yp(x,y) = p(x)$. So this is $a\sum_x x\,p(x) + b\sum_y y\,p(y) = a\E X + b\E Y$.

    **Property 2.** By linearity, $\E[aX+b] = a\E X + b$. So $(aX+b) - \E[aX+b] = a(X-\E X)$. Square it and take the expectation: $a^2\E[(X-\E X)^2] = a^2\Var X$.

    **Property 3.**

    *Step 1.* Write $\tilde X = X-\E X$, $\tilde Y = Y-\E Y$ (both have mean 0). Then $\Var(X+Y) = \E[(\tilde X+\tilde Y)^2]$.

    *Step 2.* Expand the square and use linearity: $\E[\tilde X^2] + \E[\tilde Y^2] + 2\E[\tilde X\tilde Y] = \Var X + \Var Y + 2\Cov(X,Y)$.

    *Step 3.* Under independence, the expectation of a product factorizes: $\E[\tilde X\tilde Y] = \E[\tilde X]\,\E[\tilde Y] = 0\cdot0 = 0$.

    **Property 4.**

    *Step 1.* By Property 2 with $a = 1/n$: $\Var(\frac1n\sum X_i) = \frac{1}{n^2}\Var(\sum X_i)$.

    *Step 2.* By Property 3 (repeatedly, using independence): $\Var(\sum X_i) = \sum\Var X_i = n\sigma^2$.

    *Step 3.* Combine: $\frac{1}{n^2}\cdot n\sigma^2 = \frac{\sigma^2}{n}$. $\blacksquare$

!!! llm "LLM connection: variance propagation"
    A neuron's pre-activation is $s = \sum_{j=1}^{n}W_{j}x_j$. Assume the weights are independent, with mean 0 and variance $\Var(W)$, independent of the inputs. Then, step by step:

    1. **One term.** Mean 0, so its variance is $\E[W_j^2x_j^2] = \E[W_j^2]\,\E[x_j^2] = \Var(W)\,\E[x^2]$.
    2. **Different terms are uncorrelated** (the zero-mean weight factors out), so the variances add (Property 3).
    3. **Total:** $\Var(s) = n\,\Var(W)\,\E[x^2]$.

    So each layer multiplies the signal's variance by $n\Var(W)$. Over 100 layers, anything other than 1 explodes or vanishes. Hence we choose **$\Var(W)\propto 1/n$** (Ch. 8).

    The same calculation is why attention divides by $\sqrt{d_h}$ (Prop. 1.14).

---

## 3.5 The distributions you need

**The idea:** a handful of standard distributions cover almost everything in deep learning.

### The summary table

| Distribution | Values | Parameters | Mass / density | Mean | Variance |
|---|---|---|---|---|---|
| Bernoulli | $\{0,1\}$ | $\pi$ | $\pi^x(1-\pi)^{1-x}$ | $\pi$ | $\pi(1-\pi)$ |
| **Categorical** | $\{1,\dots,V\}$ | $\boldsymbol\pi\in\Delta^{V-1}$ | $\pi_x$ | — | — |
| Uniform | $[a,b]$ | $a,b$ | $\frac1{b-a}$ | $\frac{a+b}2$ | $\frac{(b-a)^2}{12}$ |
| Gaussian | $\R$ | $\mu,\sigma^2$ | $\frac1{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| Multivariate Gaussian | $\R^n$ | $\boldsymbol\mu,\boldsymbol\Sigma$ | $\frac{\exp(-\frac12(\mathbf{x}-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu))}{\sqrt{(2\pi)^n\det\boldsymbol\Sigma}}$ | $\boldsymbol\mu$ | $\boldsymbol\Sigma$ |

Each one in plain words:

### Bernoulli: one coin flip

$X = 1$ ("success") with probability $\pi$, else $X = 0$. The formula $\pi^x(1-\pi)^{1-x}$ is just a compact way of writing "$\pi$ if $x=1$, $1-\pi$ if $x=0$".

- Mean: $0\cdot(1-\pi) + 1\cdot\pi = \pi$.
- Example: "is this eval answer correct?" with $\pi = 0.8$.

### Categorical: one roll of a $V$-sided, possibly unfair die

$X$ takes one of $V$ values, value $k$ with probability $\pi_k$.

The vector $\boldsymbol\pi$ must live in the **probability simplex**:

$$
\Delta^{V-1} = \{\boldsymbol\pi\in\R^V:\ \pi_i\ge0,\ \textstyle\sum_i\pi_i=1\}.
$$

!!! llm "The output of every LLM step is a categorical distribution"
    - $V$ = vocabulary size (e.g. 50,257 for GPT-2).
    - The network outputs unconstrained logits $\mathbf{z}\in\R^V$.
    - Softmax (Ch. 2) maps them to $\boldsymbol\pi = \softmax(\mathbf{z})$, always inside the simplex.

    That's how an unconstrained network produces valid probabilities.

### Gaussian (normal): the bell curve

Centered at $\mu$, with width set by $\sigma$. The $\frac{1}{\sqrt{2\pi\sigma^2}}$ in front makes the total area 1 (Thm 2.11). The rule of thumb for how much probability lies within $k$ standard deviations of the mean:

| Range | Probability |
|---|---|
| $\mu\pm1\sigma$ | $\approx 68\%$ |
| $\mu\pm2\sigma$ | $\approx 95\%$ |
| $\mu\pm3\sigma$ | $\approx 99.7\%$ |

The Gaussian is everywhere for three reasons:

1. **The central limit theorem** (§3.6): averages of many independent things are approximately Gaussian.
2. **Maximum entropy:** it is the "least informative" distribution with a given variance (Ch. 4).
3. **Closed under linear maps:** if $\mathbf{x}\sim\mathcal{N}(\boldsymbol\mu,\boldsymbol\Sigma)$ then $\mathbf{A}\mathbf{x}+\mathbf{b}\sim\mathcal{N}(\mathbf{A}\boldsymbol\mu+\mathbf{b},\ \mathbf{A}\boldsymbol\Sigma\mathbf{A}^\top)$. A linear layer applied to Gaussian inputs gives Gaussian outputs.

### Multivariate Gaussian

The vector version: $\boldsymbol\mu$ is the center, and the **covariance matrix** $\boldsymbol\Sigma$ (with $\Sigma_{ij} = \Cov(x_i, x_j)$) is symmetric and PSD (Ch. 1, §1.8). Its eigenvectors are the axes of the elliptical "bell", and its eigenvalues are the variances along those axes.

---

## 3.6 Law of large numbers, CLT, and Monte Carlo

**The idea:** the average of many independent samples gets reliably close to the true mean, with error shrinking like $1/\sqrt n$.

### Two inequalities

We need two simple bounds first. They say "a random variable can't be far from its mean very often".

!!! recall "The indicator $\mathbb{1}\{\cdot\}$"
    $\mathbb{1}\{\text{condition}\}$ is $1$ if the condition is true and $0$ if false. Its expectation is a probability:

    $$
    \E[\mathbb{1}\{X\ge a\}] = 1\cdot P(X\ge a) + 0\cdot P(X<a) = P(X\ge a).
    $$

!!! theorem "Lemma 3.5 (Markov and Chebyshev inequalities)"
    - **Markov:** for $X\ge0$ and $a>0$, $\;P(X\ge a)\le\dfrac{\E X}{a}$.
    - **Chebyshev:** for any $X$ with mean $\mu$ and variance $\sigma^2$, $\;P(\lvert X-\mu\rvert\ge\epsilon)\le\dfrac{\sigma^2}{\epsilon^2}$.

    **Examples.** If an average loss is 2, at most 20% of examples can have loss $\ge 10$ (Markov). And at most 25% of any distribution lies 2 or more standard deviations from its mean (Chebyshev with $\epsilon = 2\sigma$).

???+ proof "Proof"
    **Markov, Step 1 · Compare $X$ with a smaller variable.** Since $X\ge0$, we have $X\ge X\cdot\mathbb{1}\{X\ge a\}$. (When the indicator is 1, both sides equal $X$; when it's 0, the right side is 0.)

    **Markov, Step 2 · And that is at least $a$ times the indicator:** when the indicator is 1, $X\ge a$. So $X\cdot\mathbb{1}\{X\ge a\}\ge a\cdot\mathbb{1}\{X\ge a\}$.

    **Markov, Step 3 · Take expectations** of $X\ge a\cdot\mathbb{1}\{X\ge a\}$: $\E X\ge a\,P(X\ge a)$. Divide by $a$.

    **Chebyshev · Apply Markov** to the non-negative variable $(X-\mu)^2$ with threshold $\epsilon^2$:

    $$
    P(\lvert X-\mu\rvert\ge\epsilon) = P\big((X-\mu)^2\ge\epsilon^2\big)\le\frac{\E[(X-\mu)^2]}{\epsilon^2} = \frac{\sigma^2}{\epsilon^2}. \qquad\blacksquare
    $$

### Law of large numbers

"i.i.d." means **independent and identically distributed**: separate draws from the same distribution, like repeated die rolls.

!!! theorem "Theorem 3.6 (Weak law of large numbers)"
    If $X_1,X_2,\dots$ are i.i.d. with mean $\mu$ and finite variance $\sigma^2$, then for every $\epsilon>0$,

    $$
    P(\lvert\bar X_n-\mu\rvert\ge\epsilon)\to0 \quad\text{as } n\to\infty.
    $$

    The sample average gets arbitrarily close to the true mean.

???+ proof "Proof"
    **Step 1 · Variance of the average.** By Thm 3.4(4), $\Var\bar X_n = \sigma^2/n$. (And $\E\bar X_n = \mu$ by linearity.)

    **Step 2 · Chebyshev** applied to $\bar X_n$:

    $$
    P(\lvert\bar X_n-\mu\rvert\ge\epsilon)\le\frac{\sigma^2}{n\epsilon^2}.
    $$

    **Step 3 · Let $n\to\infty$.** The right side goes to 0. $\blacksquare$

### Central limit theorem

The LLN says the average converges. The CLT says **what shape** its fluctuations have.

!!! theorem "Theorem 3.7 (Central limit theorem, statement)"
    Under the same assumptions, $\dfrac{\sqrt n(\bar X_n-\mu)}{\sigma}$ converges in distribution to $\mathcal{N}(0,1)$.

    In words: averages of many independent things are approximately **Gaussian**, centered at $\mu$, with standard deviation $\sigma/\sqrt n$.

The $1/\sqrt n$ is the most useful number in statistics:

| Samples $n$ | Standard deviation of the average | Relative to $n=100$ |
|---|---|---|
| 100 | $\sigma/10$ | 1× |
| 400 | $\sigma/20$ | 2× more precise |
| 10,000 | $\sigma/100$ | 10× more precise |

**To halve the error you need 4× the samples.**

### Monte Carlo estimation

To approximate an expectation, draw samples and average:

$$
\E[g(X)]\approx\frac1n\sum_{i=1}^n g(x^{(i)}),\qquad x^{(i)}\sim p .
$$

- The estimate is **unbiased**: its own expectation is exactly $\E[g(X)]$ (linearity).
- Its error shrinks like $\sigma/\sqrt n$ (CLT).

!!! llm "LLM connection: SGD is Monte Carlo"
    The true training objective is an expectation over the data distribution:

    $$
    \mathcal{L}(\theta) = \E_{x\sim p_\text{data}}[\ell(x;\theta)].
    $$

    A **minibatch** of $B$ examples gives the Monte Carlo gradient estimate

    $$
    \hat{\mathbf{g}} = \frac1B\sum_{i=1}^B\nabla\ell(x^{(i)};\theta).
    $$

    | Property | Value | Consequence |
    |---|---|---|
    | mean | $\E\hat{\mathbf{g}} = \nabla\mathcal{L}$ | unbiased: right on average |
    | covariance | $\boldsymbol\Sigma/B$ | doubling the batch halves the noise variance |

    That trade-off underlies the "critical batch size" of [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) and the batch-size schedules in Ch. 15.

!!! llm "LLM connection: always report error bars"
    An eval accuracy is a Monte Carlo estimate too: the average of $n$ Bernoulli(correct) outcomes.

    - Its standard error is $\sqrt{p(1-p)/n}$.
    - Example: $p = 0.80$ on $n = 400$ questions gives $\sqrt{0.16/400} = 0.02$.
    - The 95% interval is about $\pm 2\times0.02 = \pm 4$ points: $[0.76, 0.84]$.

    A "2-point improvement" on this benchmark is noise (Ch. 29).

---

## 3.7 Maximum likelihood estimation (MLE)

**The idea:** choose the model parameters that make the data you actually observed as probable as possible.

### A coin first

You flip a coin 10 times and see **7 heads, 3 tails**. What's your best estimate of $\pi = P(\text{heads})$?

**The likelihood** is the probability of exactly this data under each candidate $\pi$. Flips are independent, so multiply:

$$
L(\pi) = \pi^7(1-\pi)^3 .
$$

| Candidate $\pi$ | $L(\pi) = \pi^7(1-\pi)^3$ |
|---|---|
| 0.5 | 0.00098 |
| 0.6 | 0.00179 |
| **0.7** | **0.00222** (largest) |
| 0.8 | 0.00168 |

The data are most probable at $\pi = 0.7$. Now find the exact maximum with calculus:

1. **Take the log** (it turns products into sums, and doesn't move the maximum because $\log$ is increasing):

    $$
    \log L(\pi) = 7\log\pi + 3\log(1-\pi).
    $$

2. **Differentiate** (Ch. 2: $(\log x)' = 1/x$, plus the chain rule for $\log(1-\pi)$):

    $$
    \frac{d}{d\pi}\log L = \frac{7}{\pi} - \frac{3}{1-\pi}.
    $$

3. **Set to zero and solve:** $7(1-\pi) = 3\pi$, so $7 = 10\pi$, so $\hat\pi = 0.7$.

The answer is the obvious one — the observed frequency. The machinery is what generalizes.

### The general recipe

Given i.i.d. data $\mathcal{D} = \{x^{(1)},\dots,x^{(n)}\}$ and a model family $p_\theta$:

$$
\hat\theta_\text{MLE} = \argmax_\theta\prod_{i=1}^np_\theta(x^{(i)}) = \argmin_\theta\underbrace{\left(-\frac1n\sum_{i=1}^n\log p_\theta(x^{(i)})\right)}_{\text{negative log-likelihood (NLL)}}.
$$

Why the second form is the one everyone uses:

| Change | Why it's allowed | Why it helps |
|---|---|---|
| take $\log$ | $\log$ is increasing, so the best $\theta$ doesn't move | a product of millions of tiny numbers underflows to 0; a sum of logs doesn't |
| products → sums | $\log(ab) = \log a + \log b$ | the gradient of a sum is the sum of gradients |
| negate, $\max\to\min$ | maximizing $f$ = minimizing $-f$ | optimizers minimize "losses" |
| divide by $n$ | a positive constant doesn't move the best $\theta$ | the loss is an *average*, comparable across dataset sizes |

### MLE for a categorical distribution

The coin generalizes to a $V$-sided die. The constraint "$\pi$'s sum to 1" needs one extra tool.

!!! recall "Refresher: Lagrange multipliers"
    To maximize $f(\mathbf{x})$ subject to a constraint $g(\mathbf{x}) = c$:

    - At the best point, you can't improve $f$ by moving *along* the constraint.
    - That happens exactly when the gradients are parallel: $\nabla f = \lambda\nabla g$ for some number $\lambda$.

    **Example:** maximize $xy$ subject to $x + y = 10$. Here $\nabla(xy) = (y, x)$ and $\nabla(x+y) = (1,1)$. Parallel means $y = \lambda$ and $x = \lambda$, so $x = y$. With the constraint: $x = y = 5$.

!!! theorem "Theorem 3.8 (MLE of a categorical distribution is the empirical frequency)"
    If outcome $k$ appears $n_k$ times among $n$ observations, the MLE is

    $$
    \hat\pi_k = \frac{n_k}{n}.
    $$

???+ proof "Proof (Lagrange multipliers)"
    **Step 1 · The log-likelihood.** Each observation of outcome $k$ contributes a factor $\pi_k$, so $L = \prod_k\pi_k^{n_k}$ and

    $$
    \ell(\boldsymbol\pi) = \sum_kn_k\log\pi_k ,\qquad\text{subject to } \sum_k\pi_k = 1 .
    $$

    **Step 2 · Parallel gradients.** $\partial\ell/\partial\pi_k = n_k/\pi_k$, and the constraint's gradient is $(1,\dots,1)$. So

    $$
    \frac{n_k}{\pi_k} = \lambda\quad\text{for every } k,\qquad\text{i.e.}\qquad \pi_k = \frac{n_k}{\lambda}.
    $$

    **Step 3 · Use the constraint to find $\lambda$.** $\sum_k\pi_k = \sum_k n_k/\lambda = n/\lambda = 1$, so $\lambda = n$.

    **Step 4 · Conclude** $\pi_k = n_k/n$. The objective is concave (a sum of $\log$s), so this critical point is the maximum. $\blacksquare$

    (A second proof, via Gibbs' inequality, is in Ch. 4.)

!!! llm "LLM connection: pretraining is MLE"
    Combine the chain rule (Thm 3.1) with MLE. The log of a sequence's probability is a sum of per-token log-probabilities, so the pretraining loss is

    $$
    \mathcal{L}(\theta) = -\frac{1}{\#\text{tokens}}\sum_{\text{sequences}}\sum_{t}\log p_\theta(x_t\mid x_{<t}),
    $$

    **the average negative log-probability of each actual next token.**

    - That is exactly `F.cross_entropy(logits, targets)` in `gpt.py`.
    - For a bigram model (Ch. 9), Theorem 3.8 says the MLE is simply **count and normalize**.
    - Transformers replace the count table with a neural network that *generalizes* to contexts it has never seen.

---

## 3.8 MAP estimation and regularization

**The idea:** add a prior belief to MLE. A prior that "weights are probably small" turns into weight decay.

**Maximum a posteriori (MAP)** picks the most probable $\theta$ *after* seeing the data — the peak of the posterior. Derivation:

1. **Bayes:** $p(\theta\mid\mathcal{D})\propto p(\mathcal{D}\mid\theta)\,p(\theta)$. (The evidence $p(\mathcal{D})$ doesn't depend on $\theta$, so it doesn't move the peak.)
2. **Take $-\log$**, turning the product into a sum and max into min:

$$
\hat\theta_\text{MAP} = \argmax_\theta\,p(\theta\mid\mathcal{D}) = \argmin_\theta\Big[\underbrace{-\log p(\mathcal{D}\mid\theta)}_{\text{NLL}}\;\underbrace{-\log p(\theta)}_{\text{penalty}}\Big].
$$

The prior becomes an extra **penalty term** in the loss.

!!! theorem "Proposition 3.9 (Gaussian prior = L2 regularization)"
    With prior $\theta_j\sim\mathcal{N}(0,\tau^2)$ independently for each parameter, MAP minimizes

    $$
    \mathrm{NLL}(\theta) + \frac{1}{2\tau^2}\lVert\theta\rVert^2 .
    $$

???+ proof "Proof"
    **Step 1 · Write the prior.** Independent parameters multiply: $p(\theta) = \prod_j\frac{1}{\sqrt{2\pi\tau^2}}e^{-\theta_j^2/(2\tau^2)}$.

    **Step 2 · Take $-\log$.** The product becomes a sum, and $-\log e^{-a} = a$:

    $$
    -\log p(\theta) = \sum_j\Big[\frac{\theta_j^2}{2\tau^2} + \tfrac12\log(2\pi\tau^2)\Big].
    $$

    **Step 3 · Drop the constant.** The $\log(2\pi\tau^2)$ terms don't depend on $\theta$. What's left is $\frac{1}{2\tau^2}\sum_j\theta_j^2 = \frac{\lVert\theta\rVert^2}{2\tau^2}$. $\blacksquare$

So **weight decay** has a Bayesian reading: "believe weights are small unless the data says otherwise". A smaller $\tau$ (stronger belief) means a bigger penalty.

### Smoothing counts

A different prior (the *Dirichlet*) on a categorical distribution gives **add-$\alpha$ (Laplace) smoothing**:

$$
\hat\pi_k = \frac{n_k+\alpha}{n+V\alpha}.
$$

It is "pretend you saw every outcome $\alpha$ extra times".

!!! example "Worked example"
    Counts over $V = 3$ tokens: $(3, 0, 1)$, so $n = 4$.

    | | token 1 | token 2 | token 3 |
    |---|---|---|---|
    | MLE $n_k/n$ | $3/4$ | $\mathbf{0}$ | $1/4$ |
    | add-1 $(n_k+1)/(4+3)$ | $4/7$ | $1/7$ | $2/7$ |

    MLE says token 2 is *impossible* — one unseen example and the model's loss is infinite. Smoothing fixes that. This is how n-gram models (Ch. 9) avoid assigning probability zero.

---

## 3.9 Sampling from a categorical (and the Gumbel-max trick)

**The idea:** to sample a token, lay the probabilities end to end on $[0,1]$, throw a dart, and see where it lands.

### Inverse-CDF sampling

To sample $k\sim\text{Categorical}(\boldsymbol\pi)$:

1. Draw $u\sim\text{Uniform}[0,1]$.
2. Return the first $k$ whose running total $\sum_{j\le k}\pi_j$ exceeds $u$.

!!! example "Worked example"
    $\boldsymbol\pi = (0.5, 0.3, 0.2)$.

    | token | $\pi_k$ | running total | interval of $u$ that picks it |
    |---|---|---|---|
    | 1 | 0.5 | 0.5 | $[0, 0.5)$ |
    | 2 | 0.3 | 0.8 | $[0.5, 0.8)$ |
    | 3 | 0.2 | 1.0 | $[0.8, 1]$ |

    Draw $u = 0.65$: it falls in token 2's interval. Each interval's length equals the token's probability, so each token is picked with exactly the right probability.

That's what `torch.multinomial` does conceptually, and what the sampling widget in Ch. 18 does.

### The Gumbel-max trick

A surprising alternative: add special random noise to each logit and take the **argmax**.

!!! theorem "Theorem 3.10 (Gumbel-max trick)"
    Let $g_1,\dots,g_V$ be i.i.d. standard Gumbel noise: $g = -\log(-\log u)$ with $u\sim\text{Uniform}(0,1)$. Then

    $$
    \argmax_k\,(z_k + g_k)\sim\softmax(\mathbf{z}).
    $$

??? proof "Proof"
    **Step 1 · Facts about the Gumbel distribution.** Its CDF is $F(g) = \exp(-e^{-g})$, and its density (the derivative) is $f(g) = e^{-g}\exp(-e^{-g})$.

    **Step 2 · Set up "token $k$ wins".** Let $G_i = z_i + g_i$ and $S = \sum_ie^{z_i}$. Token $k$ wins if every other $G_i$ is smaller than $G_k$. Condition on the value $x$ of $G_k$:

    - the density of $G_k$ at $x$ is $f(x - z_k)$;
    - each other $G_i < x$ with probability $F(x - z_i)$, independently.

    So

    $$
    P(k\text{ wins}) = \int_{-\infty}^{\infty}f(x-z_k)\prod_{i\ne k}F(x-z_i)\,dx .
    $$

    **Step 3 · Plug in the formulas.** $f(x - z_k) = e^{-(x-z_k)}F(x - z_k)$, so the $F$ for $k$ joins the product:

    $$
    = \int e^{-(x-z_k)}\prod_{i}\exp\!\big(-e^{-(x-z_i)}\big)\,dx .
    $$

    **Step 4 · Simplify the product.** $\prod_i\exp(a_i) = \exp(\sum_i a_i)$, and $\sum_i e^{-(x-z_i)} = e^{-x}\sum_i e^{z_i} = e^{-x}S$. Pull $e^{z_k}$ out front:

    $$
    = e^{z_k}\int e^{-x}\exp(-e^{-x}S)\,dx .
    $$

    **Step 5 · Substitute** $u = e^{-x}$, so $du = -e^{-x}\,dx$. As $x$ goes from $-\infty$ to $\infty$, $u$ goes from $\infty$ to $0$:

    $$
    = e^{z_k}\int_0^\infty e^{-uS}\,du = e^{z_k}\cdot\frac1S = \softmax(\mathbf{z})_k . \qquad\blacksquare
    $$

The trick turns sampling into an argmax. That is handy for:

- parallel samplers,
- sampling *without* replacement (add fresh noise once, take the top-$k$),
- the Gumbel-softmax relaxation (a differentiable approximation to sampling).

```python
import torch
z = torch.tensor([2.0, 1.0, 0.0, -1.0])
p = torch.softmax(z, 0)
n = 200_000
gumbel = -torch.log(-torch.log(torch.rand(n, 4)))
counts = torch.bincount((z + gumbel).argmax(1), minlength=4) / n
print(p, counts)   # agree to ~3 decimals
```

---

## Key takeaways

| Concept | One-line summary | Where it matters |
|---|---|---|
| Joint / marginal / conditional | read a table: a cell, a row total, a rescaled row | every probabilistic model |
| **Chain rule of probability** | $p(x_{1:T}) = \prod_tp(x_t\mid x_{<t})$, exact for any distribution | the blueprint of autoregressive LLMs |
| Bayes | posterior ∝ likelihood × prior; mind the base rate | evals, guardrails |
| Expectation | linear, always | everywhere |
| Variance | independent variances add; averages of $n$ shrink it by $n$ | initialization, attention scaling |
| Categorical | every LLM step outputs one, via softmax | sampling, loss |
| LLN / CLT | averages converge, with Gaussian error $\propto1/\sqrt n$ | SGD noise, error bars |
| **MLE** | minimize the average negative log-likelihood | next-token cross-entropy loss |
| Categorical MLE | normalized counts | n-gram models |
| MAP | Gaussian prior = L2 weight decay; Dirichlet prior = add-α smoothing | regularization |
| Sampling | inverse CDF, or Gumbel-max | decoding |

## Exercises

1. Prove $\Var X = \E[X^2]-(\E X)^2$.
2. Derive the MLE of $\mu$ and $\sigma^2$ for i.i.d. Gaussian data.
3. Show that the NLL of a Gaussian model with fixed $\sigma$ is the mean-squared error up to constants. Why does that justify MSE for regression?
4. A model is evaluated on 400 questions and gets 82%. Give a 95% confidence interval. Is it significantly better than a model that got 79%?
5. Prove $\E[\E[X\mid Y]] = \E X$ (the law of total expectation).
6. **Code:** estimate $\pi$ by Monte Carlo (the fraction of random points in the unit square that fall inside the quarter circle, times 4). Plot the error versus $n$ on log-log axes and confirm the slope is $-1/2$.

??? success "Solutions"
    1. Let $\mu = \E X$. Expand the square and use linearity: $\E[(X-\mu)^2] = \E[X^2] - 2\mu\E X + \mu^2 = \E[X^2] - 2\mu^2 + \mu^2 = \E[X^2]-\mu^2$.
    2. The log-likelihood is $-\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum(x_i-\mu)^2$. Setting $\partial/\partial\mu = 0$ gives $\hat\mu = \frac1n\sum x_i$. Setting $\partial/\partial\sigma^2 = 0$ gives $\hat\sigma^2 = \frac1n\sum(x_i-\hat\mu)^2$.
    3. $-\log\mathcal{N}(y;f(x),\sigma^2) = \frac{(y-f(x))^2}{2\sigma^2} + \frac12\log(2\pi\sigma^2)$. The second term is constant, so minimizing the NLL is minimizing the squared error. MSE is MLE under Gaussian noise.
    4. SE $=\sqrt{0.82\cdot0.18/400}\approx0.019$, so the 95% CI is $\approx0.82\pm0.038 = [0.78, 0.86]$. The difference of 3 points has SE $\approx\sqrt{0.019^2+0.020^2}\approx0.028$, giving $z\approx1.1$: not significant. (If both models answered the *same* questions, a paired test is more powerful; see Ch. 29.)
    5. $\E[\E[X\mid Y]] = \sum_yp(y)\sum_xx\,p(x\mid y) = \sum_x x\sum_yp(y)p(x\mid y) = \sum_x x\sum_yp(x,y) = \sum_xx\,p(x) = \E X$.
    6. The error is $\propto n^{-1/2}$ (CLT), so the log-log slope is $-0.5$.

## References

- J. Blitzstein, J. Hwang, [*Introduction to Probability*](https://projects.iq.harvard.edu/stat110), 2nd ed. (free online).
- K. P. Murphy, [*Probabilistic Machine Learning: An Introduction*](https://probml.github.io/pml-book/book1.html), Ch. 2–4 (2022).
- C. M. Bishop, [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), Ch. 1–2 (2006).
- S. McCandlish et al., [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) (2018).
- C. J. Maddison, D. Tarlow, T. Minka, [A* Sampling](https://arxiv.org/abs/1411.0030) (2014), on the Gumbel-max trick.
