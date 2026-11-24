# 3. Probability & Statistics

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 1](linear-algebra.md), [Ch. 2](calculus.md). **You will learn:** random variables and distributions, conditional probability and the **chain rule of probability** (the factorization every LLM is built on), Bayes' theorem, expectation and variance (with proofs), the categorical and Gaussian distributions, the law of large numbers and the central limit theorem, **maximum likelihood estimation**, MAP estimation and why it equals weight decay, and how to sample from a softmax (including the Gumbel-max trick).
    **Why it matters:** an LLM *is* a probability distribution over text. Training is maximum likelihood, sampling is drawing from a categorical distribution, and SGD is a Monte Carlo estimate of a gradient.

!!! watch "Watch"
    - 3Blue1Brown: [Bayes' theorem, the geometry of changing beliefs](https://www.youtube.com/watch?v=HZGCoVF3YvM) · [But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo)
    - Joe Blitzstein, [Harvard Stat 110: Probability](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo), the gold-standard full course ([course site & free book](https://projects.iq.harvard.edu/stat110)).

---

## 3.1 Random variables and distributions

A **random variable** $X$ is a quantity whose value is uncertain. Its **distribution** says how likely each value is:

- **Discrete** (e.g. the next token): a **probability mass function** $p(x) = P(X=x)$ with $p(x)\ge0$ and $\sum_xp(x)=1$.
- **Continuous** (e.g. a weight at initialization): a **density** $p(x)\ge0$ with $\int p(x)\,dx = 1$, where $P(a\le X\le b) = \int_a^bp(x)\,dx$. A density can exceed 1. Only its integrals are probabilities.

The **cumulative distribution function** is $F(x) = P(X\le x)$.

## 3.2 Joint, marginal, conditional

For two random variables, the **joint** $p(x,y)$ gives the probability of both. Summing out one variable gives the **marginal**, $p(x) = \sum_yp(x,y)$ (the *sum rule*). The **conditional** is

$$
p(y\mid x) = \frac{p(x,y)}{p(x)}\qquad(p(x)>0),
$$

which rearranges to the **product rule** $p(x,y) = p(x)\,p(y\mid x)$.

!!! theorem "Theorem 3.1 (Chain rule of probability)"
    For any random variables $X_1,\dots,X_T$:

    $$
    p(x_1,\dots,x_T) = \prod_{t=1}^{T}p(x_t\mid x_1,\dots,x_{t-1}) = \prod_{t=1}^{T}p(x_t\mid x_{<t}).
    $$

??? proof "Proof"
    By induction on $T$. $T=1$ is trivial. Assume the claim for $T-1$. By the product rule with $x = x_{<T}$ and $y = x_T$: $p(x_{1:T}) = p(x_{<T})\,p(x_T\mid x_{<T})$. Apply the hypothesis to $p(x_{<T})$. $\blacksquare$

!!! llm "This is the definition of an autoregressive language model"
    Theorem 3.1 holds for **any** distribution, with no assumptions. So a model that outputs the conditional $p_\theta(x_t\mid x_{<t})$ for every prefix automatically defines a full distribution over sequences:

    $$
    p_\theta(\text{"The cat sat"}) = p_\theta(\text{The})\;p_\theta(\text{cat}\mid\text{The})\;p_\theta(\text{sat}\mid\text{The cat}).
    $$

    GPT, Llama and Claude are all models of the right-hand side. Generation samples one factor at a time (Ch. 18). Training maximizes the product on real text (§3.7). The only modeling question is *how to compute $p_\theta(x_t\mid x_{<t})$ from the prefix*, and the answer has evolved from n-gram counts (Ch. 9) to Transformers (Ch. 12–14).

**Independence.** $X$ and $Y$ are independent if $p(x,y) = p(x)p(y)$, equivalently $p(y\mid x) = p(y)$. A **Markov assumption** of order $n-1$ says $p(x_t\mid x_{<t}) = p(x_t\mid x_{t-n+1},\dots,x_{t-1})$: only the last $n-1$ tokens matter. That is the n-gram model of Ch. 9. Transformers drop this assumption (up to the context window).

## 3.3 Bayes' theorem

!!! theorem "Theorem 3.2 (Bayes)"
    $$
    p(\theta\mid\mathcal{D}) = \frac{p(\mathcal{D}\mid\theta)\,p(\theta)}{p(\mathcal{D})},\qquad p(\mathcal{D}) = \sum_\theta p(\mathcal{D}\mid\theta)p(\theta).
    $$

    That is, posterior = likelihood × prior / evidence.

??? proof "Proof"
    Write the joint both ways with the product rule, $p(\theta,\mathcal{D}) = p(\theta\mid\mathcal{D})p(\mathcal{D}) = p(\mathcal{D}\mid\theta)p(\theta)$, and divide by $p(\mathcal{D})$. $\blacksquare$

!!! theorem "Example 3.3 (Base rates)"
    A classifier flags toxic text with 99% sensitivity and a 2% false-positive rate, and 1% of texts are toxic. If a text is flagged, $P(\text{toxic}\mid\text{flag}) = \frac{0.99\cdot0.01}{0.99\cdot0.01 + 0.02\cdot0.99} = \frac{1}{3}$. Two thirds of flags are false alarms. This matters when you build LLM guardrails and evals (Ch. 29).

## 3.4 Expectation and variance

!!! definition "Definition 3.1"
    $\E[X] = \sum_xx\,p(x)$ (or $\int x\,p(x)\,dx$), and $\E[g(X)] = \sum_xg(x)p(x)$. The **variance** is $\Var(X) = \E[(X-\E X)^2] = \E[X^2]-(\E X)^2$, and the **standard deviation** is $\sqrt{\Var X}$. $\Cov(X,Y) = \E[(X-\E X)(Y-\E Y)]$.

!!! theorem "Theorem 3.4 (Properties)"
    1. **Linearity:** $\E[aX+bY] = a\E X + b\E Y$, *always* (no independence needed).
    2. $\Var(aX+b) = a^2\Var X$.
    3. $\Var(X+Y) = \Var X + \Var Y + 2\Cov(X,Y)$. If $X,Y$ are independent, $\Cov = 0$ and the variances add.
    4. If $X_1,\dots,X_n$ are independent with variance $\sigma^2$, the **mean** $\bar X = \frac1n\sum X_i$ has $\Var\bar X = \sigma^2/n$.

??? proof "Proofs"
    1. $\E[aX+bY] = \sum_{x,y}(ax+by)p(x,y) = a\sum_xx\sum_yp(x,y) + b\sum_yy\sum_xp(x,y) = a\E X + b\E Y$ by the sum rule.
    2. $\Var(aX+b) = \E[(aX+b - a\E X - b)^2] = a^2\E[(X-\E X)^2]$.
    3. Expand $\E[(X-\E X + Y-\E Y)^2]$. Under independence, $\E[(X-\E X)(Y-\E Y)] = \E[X-\E X]\,\E[Y-\E Y] = 0$, because the expectation of a product of independent variables factorizes.
    4. By 3 (repeatedly) and 2: $\Var(\frac1n\sum X_i) = \frac1{n^2}\sum\Var X_i = \frac{\sigma^2}{n}$. $\blacksquare$

!!! llm "LLM connection: variance propagation"
    Property 3 drives initialization (Ch. 8): a pre-activation $\sum_{j=1}^{n}W_{ij}x_j$ with independent zero-mean terms has variance $n\Var(W)\Var(x)$. To keep activations from exploding or vanishing over 100 layers, we choose $\Var(W)\propto 1/n$. It is also why attention divides by $\sqrt{d_h}$ (Prop. 1.14).

## 3.5 The distributions you need

| Distribution | Support | Parameters | Mass/density | Mean | Variance |
|---|---|---|---|---|---|
| Bernoulli | $\{0,1\}$ | $\pi$ | $\pi^x(1-\pi)^{1-x}$ | $\pi$ | $\pi(1-\pi)$ |
| **Categorical** | $\{1,\dots,V\}$ | $\boldsymbol\pi\in\Delta^{V-1}$ | $\pi_x$ | | |
| Uniform | $[a,b]$ | $a,b$ | $\frac1{b-a}$ | $\frac{a+b}2$ | $\frac{(b-a)^2}{12}$ |
| Gaussian | $\R$ | $\mu,\sigma^2$ | $\frac1{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| Multivariate Gaussian | $\R^n$ | $\boldsymbol\mu,\boldsymbol\Sigma$ | $\frac{\exp(-\frac12(\mathbf{x}-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu))}{\sqrt{(2\pi)^n\det\boldsymbol\Sigma}}$ | $\boldsymbol\mu$ | $\boldsymbol\Sigma$ |

$\Delta^{V-1} = \{\boldsymbol\pi\in\R^V:\pi_i\ge0,\ \sum_i\pi_i=1\}$ is the **probability simplex**. **The output of every LLM step is a categorical distribution over the vocabulary**, parameterized by logits through softmax, $\boldsymbol\pi = \softmax(\mathbf{z})$. Softmax maps all of $\R^V$ onto the interior of the simplex, which lets an unconstrained network output valid probabilities.

The Gaussian is everywhere because of the central limit theorem (below), because it is the maximum-entropy distribution for a given variance (Ch. 4), and because it is closed under linear maps: if $\mathbf{x}\sim\mathcal{N}(\boldsymbol\mu,\boldsymbol\Sigma)$ then $\mathbf{A}\mathbf{x}+\mathbf{b}\sim\mathcal{N}(\mathbf{A}\boldsymbol\mu+\mathbf{b},\mathbf{A}\boldsymbol\Sigma\mathbf{A}^\top)$.

## 3.6 Law of large numbers, CLT, and Monte Carlo

!!! theorem "Lemma 3.5 (Markov and Chebyshev inequalities)"
    For $X\ge0$ and $a>0$: $P(X\ge a)\le\E X/a$. For any $X$ with mean $\mu$ and variance $\sigma^2$: $P(\lvert X-\mu\rvert\ge\epsilon)\le\sigma^2/\epsilon^2$.

??? proof "Proof"
    Markov: $\E X\ge\E[X\cdot\mathbb{1}\{X\ge a\}]\ge a\,P(X\ge a)$. Chebyshev: apply Markov to $(X-\mu)^2$ with $a=\epsilon^2$. $\blacksquare$

!!! theorem "Theorem 3.6 (Weak law of large numbers)"
    If $X_1,X_2,\dots$ are i.i.d. with mean $\mu$ and finite variance $\sigma^2$, then for every $\epsilon>0$, $P(\lvert\bar X_n-\mu\rvert\ge\epsilon)\to0$.

??? proof "Proof"
    By Thm 3.4(4), $\Var\bar X_n = \sigma^2/n$. Chebyshev gives $P(\lvert\bar X_n-\mu\rvert\ge\epsilon)\le\sigma^2/(n\epsilon^2)\to0$. $\blacksquare$

!!! theorem "Theorem 3.7 (Central limit theorem, statement)"
    Under the same assumptions, $\sqrt n(\bar X_n-\mu)/\sigma$ converges in distribution to $\mathcal{N}(0,1)$. Averages of many independent things are approximately Gaussian, with standard deviation shrinking like $1/\sqrt n$.

**Monte Carlo estimation** approximates an expectation by an average of samples: $\E[g(X)]\approx\frac1n\sum_ig(x^{(i)})$. The estimate is unbiased, and its error shrinks like $\sigma/\sqrt n$.

!!! llm "LLM connection: SGD is Monte Carlo"
    The true training objective is an expectation over the data distribution, $\mathcal{L}(\theta) = \E_{x\sim p_\text{data}}[\ell(x;\theta)]$. A **minibatch** of $B$ examples gives the Monte Carlo gradient estimate $\hat{\mathbf{g}} = \frac1B\sum_{i=1}^B\nabla\ell(x^{(i)};\theta)$. It is unbiased, $\E\hat{\mathbf{g}} = \nabla\mathcal{L}$, with covariance $\boldsymbol\Sigma/B$. Doubling the batch halves the gradient-noise variance, and that trade-off underlies the "critical batch size" of [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) and the batch-size schedules in Ch. 15. Evaluation metrics are Monte Carlo estimates too, so **always report error bars** (Ch. 29): an accuracy measured on $n$ items has standard error $\sqrt{p(1-p)/n}$.

## 3.7 Maximum likelihood estimation (MLE)

Given i.i.d. data $\mathcal{D} = \{x^{(1)},\dots,x^{(n)}\}$ and a model family $p_\theta$, the **likelihood** is $p(\mathcal{D}\mid\theta) = \prod_ip_\theta(x^{(i)})$. MLE picks

$$
\hat\theta_\text{MLE} = \argmax_\theta\prod_{i=1}^np_\theta(x^{(i)}) = \argmin_\theta\underbrace{\left(-\frac1n\sum_{i=1}^n\log p_\theta(x^{(i)})\right)}_{\text{negative log-likelihood (NLL)}}.
$$

The log turns products into sums: it is numerically stable, and the gradient of a sum is the sum of gradients. Since $\log$ is increasing, the argmax doesn't change.

!!! theorem "Theorem 3.8 (MLE of a categorical distribution is the empirical frequency)"
    If outcome $k$ appears $n_k$ times among $n$ observations, the MLE is $\hat\pi_k = n_k/n$.

??? proof "Proof (Lagrange multipliers)"
    Maximize $\ell(\boldsymbol\pi) = \sum_kn_k\log\pi_k$ subject to $\sum_k\pi_k = 1$. At a constrained optimum, the gradient of the objective is parallel to the gradient of the constraint: $\nabla\ell = \lambda\nabla(\sum_k\pi_k)$, i.e. $n_k/\pi_k = \lambda$ for all $k$. So $\pi_k = n_k/\lambda$, and summing gives $\lambda = n$. The objective is concave, so this is the maximum. (A second proof, via Gibbs' inequality, is in Ch. 4.) $\blacksquare$

!!! llm "LLM connection: pretraining is MLE"
    Combining Theorem 3.1 with MLE, the pretraining loss on a corpus of token sequences is

    $$
    \mathcal{L}(\theta) = -\frac{1}{\#\text{tokens}}\sum_{\text{sequences}}\sum_{t}\log p_\theta(x_t\mid x_{<t}),
    $$

    the average negative log-probability of each actual next token. That is exactly `F.cross_entropy(logits, targets)` in `gpt.py`. For a bigram model (Ch. 9), Theorem 3.8 says the MLE is simply **count and normalize**. Transformers replace the count table with a neural network that *generalizes* to contexts it has never seen.

## 3.8 MAP estimation and regularization

**Maximum a posteriori** estimation adds a prior: $\hat\theta_\text{MAP} = \argmax_\theta\,p(\theta\mid\mathcal{D}) = \argmin_\theta[-\log p(\mathcal{D}\mid\theta) - \log p(\theta)]$.

!!! theorem "Proposition 3.9 (Gaussian prior = L2 regularization)"
    With prior $\theta_j\sim\mathcal{N}(0,\tau^2)$ i.i.d., MAP minimizes $\mathrm{NLL}(\theta) + \frac{1}{2\tau^2}\lVert\theta\rVert^2$.

??? proof "Proof"
    $-\log p(\theta) = \sum_j\big[\frac{\theta_j^2}{2\tau^2} + \tfrac12\log(2\pi\tau^2)\big] = \frac{\lVert\theta\rVert^2}{2\tau^2} + \text{const}$. $\blacksquare$

So **weight decay** has a Bayesian reading: "believe weights are small unless the data says otherwise." A Dirichlet prior on a categorical gives **add-$\alpha$ (Laplace) smoothing**, $\hat\pi_k = (n_k+\alpha)/(n+V\alpha)$, which is how n-gram models avoid assigning probability zero to unseen events (Ch. 9).

## 3.9 Sampling from a categorical (and the Gumbel-max trick)

**Inverse-CDF sampling.** To sample $k\sim\text{Categorical}(\boldsymbol\pi)$: draw $u\sim\text{Uniform}[0,1]$ and return the first $k$ with $\sum_{j\le k}\pi_j > u$. That's what `torch.multinomial` does conceptually, and what the sampling widget in Ch. 18 does.

!!! theorem "Theorem 3.10 (Gumbel-max trick)"
    Let $g_1,\dots,g_V$ be i.i.d. standard Gumbel ($g = -\log(-\log u)$ with $u\sim\text{Uniform}(0,1)$). Then $\argmax_k\,(z_k + g_k)\sim\softmax(\mathbf{z})$.

??? proof "Proof"
    Standard Gumbel has CDF $F(g) = \exp(-e^{-g})$ and density $f(g) = e^{-g}\exp(-e^{-g})$. Let $G_k = z_k+g_k$ and $S = \sum_ie^{z_i}$. Index $k$ wins iff every other $G_i < G_k$:

    $$
    P(k\text{ wins}) = \int_{-\infty}^{\infty}f(x-z_k)\prod_{i\ne k}F(x-z_i)\,dx
    = \int e^{-(x-z_k)}\prod_{i}\exp\!\big(-e^{-(x-z_i)}\big)dx
    = e^{z_k}\int e^{-x}\exp(-e^{-x}S)\,dx .
    $$

    Substitute $u = e^{-x}$, $du = -e^{-x}dx$: $\;P = e^{z_k}\int_0^\infty e^{-uS}du = e^{z_k}/S = \softmax(\mathbf{z})_k$. $\blacksquare$

The trick turns sampling into an argmax, which is handy for parallel samplers, sampling without replacement (repeat with fresh noise and take the top-$k$), and the Gumbel-softmax relaxation.

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

- The **chain rule of probability** $p(x_{1:T}) = \prod_tp(x_t\mid x_{<t})$ is exact for any distribution, and it is the blueprint of autoregressive LLMs.
- Expectation is linear. Variances of independent sums add. Averages of $n$ things have standard deviation $\propto1/\sqrt n$.
- Every LLM step outputs a **categorical** distribution via softmax.
- **MLE = minimize average negative log-likelihood**. For LLMs this is the next-token cross-entropy loss. For count tables, MLE = normalized counts.
- MAP with a Gaussian prior = L2 weight decay. A Dirichlet prior = add-α smoothing.
- SGD gradients and eval scores are Monte Carlo estimates with $1/\sqrt n$ error.

## Exercises

1. Prove $\Var X = \E[X^2]-(\E X)^2$.
2. Derive the MLE of $\mu$ and $\sigma^2$ for i.i.d. Gaussian data.
3. Show that the NLL of a Gaussian model with fixed $\sigma$ is the mean-squared error up to constants. Why does that justify MSE for regression?
4. A model is evaluated on 400 questions and gets 82%. Give a 95% confidence interval. Is it significantly better than a model that got 79%?
5. Prove $\E[\E[X\mid Y]] = \E X$ (law of total expectation).
6. **Code:** estimate $\pi$ by Monte Carlo (fraction of random points in the unit square that fall inside the quarter circle). Plot the error versus $n$ on log-log axes and confirm the slope is $-1/2$.

??? success "Solutions"
    1. $\E[(X-\mu)^2] = \E[X^2] - 2\mu\E X + \mu^2 = \E[X^2]-\mu^2$.
    2. $\hat\mu = \frac1n\sum x_i$ and $\hat\sigma^2 = \frac1n\sum(x_i-\hat\mu)^2$ (set the partial derivatives of the log-likelihood to 0).
    3. $-\log\mathcal{N}(y;f(x),\sigma^2) = \frac{(y-f(x))^2}{2\sigma^2} + \text{const}$.
    4. SE $=\sqrt{0.82\cdot0.18/400}\approx0.019$, so the CI is $\approx[0.78, 0.86]$. The difference of 3 points has SE $\approx\sqrt{0.019^2+0.020^2}\approx0.028$, giving $z\approx1.1$: not significant. (If both models answered the *same* questions, a paired test is more powerful, see Ch. 29.)
    5. $\sum_yp(y)\sum_xx\,p(x\mid y) = \sum_x x\sum_yp(x,y) = \sum_xx\,p(x)$.
    6. The error is $\propto n^{-1/2}$, so the log-log slope is $-0.5$.

## References

- J. Blitzstein, J. Hwang, [*Introduction to Probability*](https://projects.iq.harvard.edu/stat110), 2nd ed. (free online).
- K. P. Murphy, [*Probabilistic Machine Learning: An Introduction*](https://probml.github.io/pml-book/book1.html), Ch. 2–4 (2022).
- C. M. Bishop, [*Pattern Recognition and Machine Learning*](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/), Ch. 1–2 (2006).
- S. McCandlish et al., [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) (2018).
- C. J. Maddison, D. Tarlow, T. Minka, [A* Sampling](https://arxiv.org/abs/1411.0030) (2014), on the Gumbel-max trick.
