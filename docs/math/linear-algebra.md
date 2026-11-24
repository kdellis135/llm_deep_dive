# 1. Linear Algebra

!!! abstract "In this chapter"
    **Prerequisites:** high-school algebra (solving equations, squaring, square roots) and basic trigonometry ($\sin$, $\cos$).

    **You will learn:**

    - vectors, dot products, lengths and angles
    - matrices as functions that transform space
    - matrix multiplication, transpose, rank, inverses, determinants
    - rotations, eigenvalues, the SVD and low-rank approximation
    - projections and least squares
    - tensors, `einsum`, and the strange geometry of high dimensions

    **Why it matters:** an LLM is, to a first approximation, *a long sequence of matrix multiplications*.

    - Embeddings are vectors.
    - Attention scores are dot products.
    - LoRA is a low-rank matrix.
    - RoPE is a rotation matrix.
    - The cost of everything is counted in matmul FLOPs.

!!! watch "Watch first (highly recommended)"
    3Blue1Brown, *Essence of Linear Algebra*. Its 16 short videos build exactly the geometric intuition this chapter formalizes.

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/videoseries?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab" allowfullscreen loading="lazy"></iframe></div>

    For rigor: Gilbert Strang, [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) and [MIT 18.065 Matrix Methods in Data Analysis and Machine Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/).

    Textbook: Deisenroth, Faisal & Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io/), Ch. 2–4.

!!! tip "How to read the math chapters"
    - Every idea comes in the same order: **one-line idea → definition → worked example with numbers → theorem → step-by-step proof → where it shows up in LLMs.**
    - Proofs are written as numbered steps. Each step says *what* we do and *why* it is allowed.
    - Short proofs are open. Long ones are folded: click the bar to open them.
    - **Refresher** boxes recall background math (law of cosines, quadratics, complex numbers …) right where it is needed.
    - **Check yourself** boxes are 30-second questions. Try them before opening the answer.
    - Work the examples with pen and paper. Reading math is not the same as doing it.

---

## Roadmap

| § | Question it answers | Used later in |
|---|---|---|
| 1.1 | What is a vector? | everything |
| 1.2 | How do we measure similarity and length? | attention (Ch. 12), embeddings (Ch. 26) |
| 1.3 | What does a matrix *do*? | every layer |
| 1.4 | What does multiplying matrices mean, and what does it cost? | FLOP counting (Ch. 15), LoRA (Ch. 23) |
| 1.5 | How much information can a matrix carry? (rank) | LoRA, attention heads |
| 1.6 | When can a map be undone? (inverse, determinant) | normalization, theory |
| 1.7 | Which maps preserve geometry? (rotations) | RoPE (Ch. 11) |
| 1.8 | Which directions does a matrix only stretch? (eigenvectors) | optimization (Ch. 5) |
| 1.9 | What does *every* matrix look like inside? (SVD) | LoRA, compression, Muon |
| 1.10 | What is the closest point in a subspace? (least squares) | linear models (Ch. 6) |
| 1.11 | How does this look in PyTorch? | all code |
| 1.12 | Why is high-dimensional space so strange? | $\sqrt{d_h}$ scaling, superposition |

---

## Warm-up: reading the notation

Three pieces of notation appear on almost every line. Make sure they are comfortable first.

**1. Subscripts pick out entries.**

- $x_3$ means "entry number 3 of $\mathbf{x}$".
- $W_{ij}$ means "the entry of $\mathbf{W}$ in row $i$, column $j$".

**2. The summation sign $\sum$ is a compact "for loop that adds".**

$$
\sum_{i=1}^{4} i^2 \;=\; 1^2 + 2^2 + 3^2 + 4^2 \;=\; 30.
$$

Read it as: "let $i$ run from 1 to 4, compute $i^2$ each time, add everything up."

| Written | Means | Python |
|---|---|---|
| $\sum_{i=1}^{n} x_i$ | $x_1 + x_2 + \dots + x_n$ | `sum(x)` |
| $\sum_{i=1}^{n} x_i y_i$ | $x_1y_1 + \dots + x_ny_n$ | `sum(a*b for a, b in zip(x, y))` |
| $\sum_{i}\sum_{j} A_{ij}$ | add every entry of a table | `A.sum()` |

Two rules you will use constantly:

- **Constants factor out:** $\sum_i c\,x_i = c\sum_i x_i$.
- **Sums split:** $\sum_i (x_i + y_i) = \sum_i x_i + \sum_i y_i$.

**3. "$\in \R^n$" means "is a list of $n$ real numbers".**

- $\mathbf{x}\in\R^3$: a vector with 3 entries.
- $\mathbf{W}\in\R^{2\times 3}$: a matrix with 2 rows and 3 columns.

---

## 1.1 Vectors

**The idea:** a vector is a list of numbers that we can add and scale.

!!! definition "Definition 1.1 (Vector)"
    A **vector** $\mathbf{x} \in \R^n$ is an ordered list of $n$ real numbers, written as a column:

    $$
    \mathbf{x} = \begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}, \qquad \mathbf{x}^\top = (x_1, \dots, x_n).
    $$

    The $\top$ ("transpose") just lays the column on its side.

    Two operations, both done **entry by entry**:

    - **Addition:** $(\mathbf{x}+\mathbf{y})_i = x_i + y_i$
    - **Scaling** by a number $c$: $(c\mathbf{x})_i = c\,x_i$

!!! example "Worked example"
    Let $\mathbf{x} = (1, 2)^\top$ and $\mathbf{y} = (3, -1)^\top$.

    - $\mathbf{x} + \mathbf{y} = (1+3,\; 2+(-1))^\top = (4, 1)^\top$
    - $3\mathbf{x} = (3, 6)^\top$
    - $\mathbf{x} - \mathbf{y} = (1-3,\; 2-(-1))^\top = (-2, 3)^\top$

There are two pictures of a vector. You need both.

| Picture | A vector is … | Addition means … | Example |
|---|---|---|---|
| **Geometric** | an arrow from the origin to the point $(x_1,\dots,x_n)$ | put arrows tip to tail | $(1,2)$ is "1 right, 2 up" |
| **Data** | a list of features | combine features | a GPT-2 token embedding is a vector in $\R^{768}$ |

In the data picture, no single coordinate has to mean anything. What matters is **directions** and **distances** between vectors.

### Linear combinations and span

!!! definition "Linear combination, span"
    - A **linear combination** of $\mathbf{v}_1,\dots,\mathbf{v}_k$ is any sum $c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k$, with numbers $c_i$.
    - Their **span** is the set of *all* such combinations: every point you can reach by scaling and adding them.

!!! example "Worked example: what spans what"
    | Vectors | Span | Why |
    |---|---|---|
    | $(1,0)$, $(0,1)$ | the whole plane $\R^2$ | $(a,b) = a(1,0) + b(0,1)$ for any $a,b$ |
    | $(1,2)$ | a line through the origin | you can only scale it: $c(1,2)$ |
    | $(1,2)$, $(2,4)$ | the *same* line | $(2,4) = 2\cdot(1,2)$ adds no new direction |
    | $(1,2)$, $(1,0)$ | the whole plane | two different directions |

!!! llm "LLM connection: the residual stream"
    A Transformer keeps one vector $\mathbf{h}_t\in\R^d$ at each position $t$, called the **residual stream** (Ch. 13).

    1. Every layer reads $\mathbf{h}_t$.
    2. Every layer *adds* a vector to it.
    3. At the end, $\mathbf{h}_t$ is compared with every token embedding via dot products (next section) to get the next-token scores.

    Language modeling is literally vector arithmetic.

---

## 1.2 The dot product, angles and similarity

**The idea:** the dot product turns two vectors into one number that says how much they point the same way.

!!! definition "Definition 1.2 (Dot product, norm)"
    Multiply matching entries, then add:

    $$
    \mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{n} x_i y_i .
    $$

    The **norm** (length) of a vector is the square root of its dot product with itself:

    $$
    \lVert \mathbf{x} \rVert = \sqrt{\mathbf{x}^\top \mathbf{x}} = \sqrt{x_1^2 + \dots + x_n^2}.
    $$

!!! example "Worked example"
    Let $\mathbf{x} = (1, 2, 3)^\top$ and $\mathbf{y} = (4, -5, 6)^\top$.

    | Step | Computation |
    |---|---|
    | multiply matching entries | $1\cdot4 = 4,\quad 2\cdot(-5) = -10,\quad 3\cdot6 = 18$ |
    | add | $\mathbf{x}^\top\mathbf{y} = 4 - 10 + 18 = 12$ |
    | length of $\mathbf{x}$ | $\lVert\mathbf{x}\rVert = \sqrt{1+4+9} = \sqrt{14}\approx 3.74$ |

    In 2-D, $\lVert(3,4)\rVert = \sqrt{9+16} = 5$. That is Pythagoras: the arrow is the hypotenuse of a 3-4-5 triangle.

Three properties follow straight from the definition. We will use them inside every proof below.

| Property | Formula | Why it's true |
|---|---|---|
| Symmetric | $\mathbf{x}^\top\mathbf{y} = \mathbf{y}^\top\mathbf{x}$ | $x_iy_i = y_ix_i$ |
| Linear | $(a\mathbf{x}+b\mathbf{z})^\top\mathbf{y} = a\,\mathbf{x}^\top\mathbf{y} + b\,\mathbf{z}^\top\mathbf{y}$ | the sum splits and constants factor out |
| Positive | $\mathbf{x}^\top\mathbf{x} = \lVert\mathbf{x}\rVert^2 \ge 0$, and $=0$ only for $\mathbf{x}=\mathbf{0}$ | a sum of squares |

A consequence we need right away is the **expansion of a squared length**. Using linearity and symmetry:

$$
\lVert \mathbf{x}-\mathbf{y}\rVert^2 = (\mathbf{x}-\mathbf{y})^\top(\mathbf{x}-\mathbf{y}) = \mathbf{x}^\top\mathbf{x} - \mathbf{x}^\top\mathbf{y} - \mathbf{y}^\top\mathbf{x} + \mathbf{y}^\top\mathbf{y} = \lVert\mathbf{x}\rVert^2 - 2\,\mathbf{x}^\top\mathbf{y} + \lVert\mathbf{y}\rVert^2 .
$$

It is the vector version of $(a-b)^2 = a^2 - 2ab + b^2$.

### What the dot product means geometrically

!!! recall "Refresher: the law of cosines"
    In a triangle with sides $a$, $b$, $c$, where $\theta$ is the angle between sides $a$ and $b$:

    $$
    c^2 = a^2 + b^2 - 2ab\cos\theta .
    $$

    - When $\theta = 90^\circ$, $\cos\theta = 0$ and this is Pythagoras: $c^2 = a^2+b^2$.
    - The $-2ab\cos\theta$ term corrects Pythagoras for non-right angles.

!!! theorem "Theorem 1.1 (Geometric meaning of the dot product)"
    If $\theta$ is the angle between nonzero vectors $\mathbf{x}$ and $\mathbf{y}$, then

    $$
    \mathbf{x}^\top\mathbf{y} = \lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert\cos\theta .
    $$

???+ proof "Proof"
    **Step 1 · Draw the triangle.** Put $\mathbf{x}$ and $\mathbf{y}$ tail to tail at the origin. The third side goes from the tip of $\mathbf{y}$ to the tip of $\mathbf{x}$: it is the vector $\mathbf{x}-\mathbf{y}$. The angle between the sides $\mathbf{x}$ and $\mathbf{y}$ is $\theta$.

    **Step 2 · Law of cosines** with $a = \lVert\mathbf{x}\rVert$, $b = \lVert\mathbf{y}\rVert$, $c = \lVert\mathbf{x}-\mathbf{y}\rVert$:

    $$
    \lVert \mathbf{x}-\mathbf{y}\rVert^2 = \lVert\mathbf{x}\rVert^2 + \lVert\mathbf{y}\rVert^2 - 2\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert\cos\theta .
    $$

    **Step 3 · Algebra** gives the same left-hand side another way (the expansion above):

    $$
    \lVert \mathbf{x}-\mathbf{y}\rVert^2 = \lVert\mathbf{x}\rVert^2 + \lVert\mathbf{y}\rVert^2 - 2\,\mathbf{x}^\top\mathbf{y}.
    $$

    **Step 4 · Compare.** The two right-hand sides are equal. Cancel $\lVert\mathbf{x}\rVert^2 + \lVert\mathbf{y}\rVert^2$ from both, then divide by $-2$:

    $$
    \mathbf{x}^\top\mathbf{y} = \lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert\cos\theta. \qquad\blacksquare
    $$

So the dot product measures **alignment**. Since lengths are positive, its sign is the sign of $\cos\theta$:

| Angle $\theta$ | $\cos\theta$ | $\mathbf{x}^\top\mathbf{y}$ | Meaning |
|---|---|---|---|
| $0^\circ$ | $1$ | largest possible, $\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert$ | same direction |
| less than $90^\circ$ | positive | positive | roughly the same way |
| $90^\circ$ | $0$ | $0$ | **orthogonal** (perpendicular) |
| more than $90^\circ$ | negative | negative | roughly opposite |
| $180^\circ$ | $-1$ | most negative | opposite |

### Cosine similarity

Dividing out the lengths leaves only the angle. This is the **cosine similarity**:

$$
\cos(\mathbf{x},\mathbf{y}) = \frac{\mathbf{x}^\top\mathbf{y}}{\lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert} \in [-1, 1].
$$

It is the standard similarity measure for text embeddings (Ch. 26), because it ignores how *long* the vectors are and compares only their *direction*.

!!! example "Worked example"
    $\mathbf{x} = (1, 0)$, $\mathbf{y} = (1, 1)$.

    - $\mathbf{x}^\top\mathbf{y} = 1$
    - $\lVert\mathbf{x}\rVert = 1$, $\lVert\mathbf{y}\rVert = \sqrt2$
    - $\cos = 1/\sqrt2 \approx 0.707$, so $\theta = 45^\circ$. That matches the picture: $(1,1)$ points diagonally.

    Doubling $\mathbf{y}$ to $(2,2)$ doubles the dot product but leaves the cosine at $0.707$.

### Why the cosine always lands in $[-1, 1]$ { data-toc-label="Why the cosine always lands in [−1, 1]" }

There is a gap in the story above. In $\R^{768}$ we cannot draw triangles, so "the angle" has no picture. Instead we **define** the angle by $\cos\theta = \mathbf{x}^\top\mathbf{y}/(\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert)$.

For that to make sense, the right side must always be between $-1$ and $1$ — in every dimension. That is exactly the next theorem.

!!! recall "Refresher: the minimum of a quadratic"
    If $a > 0$, the parabola $q(t) = at^2 + bt + c$ has its lowest point at

    $$
    t^\star = -\frac{b}{2a}, \qquad q(t^\star) = c - \frac{b^2}{4a}.
    $$

    (Set the slope $2at + b$ to zero, or complete the square.) Example: $t^2 - 6t + 10$ is smallest at $t = 3$, where it equals $1$.

!!! theorem "Theorem 1.2 (Cauchy–Schwarz inequality)"
    For all vectors $\mathbf{x},\mathbf{y}\in\R^n$:

    $$
    \lvert \mathbf{x}^\top\mathbf{y}\rvert \le \lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert,
    $$

    with equality if and only if one vector is a scalar multiple of the other.

???+ proof "Proof"
    **Step 1 · Easy case.** If $\mathbf{y}=\mathbf{0}$, both sides are $0$ and we're done. From now on $\mathbf{y}\neq\mathbf{0}$.

    **Step 2 · A length is never negative.** For every real number $t$, the vector $\mathbf{x}-t\mathbf{y}$ has a squared length $\ge 0$. Expand it (same expansion as before, with $t\mathbf{y}$ in place of $\mathbf{y}$):

    $$
    0 \le \lVert\mathbf{x} - t\mathbf{y}\rVert^2 = \underbrace{\lVert\mathbf{y}\rVert^2}_{a}\,t^2 \;\underbrace{-\,2\,\mathbf{x}^\top\mathbf{y}}_{b}\;t + \underbrace{\lVert\mathbf{x}\rVert^2}_{c} .
    $$

    **Step 3 · This is a quadratic in $t$** with $a = \lVert\mathbf{y}\rVert^2 > 0$. By the refresher, its minimum value is $c - b^2/(4a)$. The minimum is still $\ge 0$ (every value is), so

    $$
    0 \le \lVert\mathbf{x}\rVert^2 - \frac{4(\mathbf{x}^\top\mathbf{y})^2}{4\lVert\mathbf{y}\rVert^2} = \lVert\mathbf{x}\rVert^2 - \frac{(\mathbf{x}^\top\mathbf{y})^2}{\lVert\mathbf{y}\rVert^2}.
    $$

    **Step 4 · Rearrange.** Multiply both sides by $\lVert\mathbf{y}\rVert^2 > 0$:

    $$
    (\mathbf{x}^\top\mathbf{y})^2 \le \lVert\mathbf{x}\rVert^2\lVert\mathbf{y}\rVert^2 .
    $$

    Take square roots of both (non-negative) sides to get the theorem.

    **Step 5 · When is it an equality?** Exactly when the minimum value is $0$, i.e. $\lVert\mathbf{x}-t^\star\mathbf{y}\rVert = 0$ for the best $t^\star$. A vector has length 0 only if it *is* $\mathbf{0}$, so $\mathbf{x} = t^\star\mathbf{y}$. $\blacksquare$

!!! theorem "Corollary 1.3 (Triangle inequality)"
    $\lVert\mathbf{x}+\mathbf{y}\rVert \le \lVert\mathbf{x}\rVert + \lVert\mathbf{y}\rVert$. A detour is never shorter than the direct route.

???+ proof "Proof"
    **Step 1 · Expand** the left side squared:

    $$
    \lVert\mathbf{x}+\mathbf{y}\rVert^2 = \lVert\mathbf{x}\rVert^2 + 2\,\mathbf{x}^\top\mathbf{y} + \lVert\mathbf{y}\rVert^2 .
    $$

    **Step 2 · Use Cauchy–Schwarz** on the middle term, $\mathbf{x}^\top\mathbf{y} \le \lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert$:

    $$
    \le \lVert\mathbf{x}\rVert^2 + 2\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert + \lVert\mathbf{y}\rVert^2 = (\lVert\mathbf{x}\rVert+\lVert\mathbf{y}\rVert)^2 .
    $$

    **Step 3 · Take square roots** of both sides. $\blacksquare$

### Other ways to measure length

The $\ell_2$ norm above is not the only option. The general family is $\lVert\mathbf{x}\rVert_p = (\sum_i \lvert x_i\rvert^p)^{1/p}$. Three members matter:

| Norm | Formula | For $\mathbf{x} = (3,-4)$ | Where it's used |
|---|---|---|---|
| $\ell_1$ | $\sum_i\lvert x_i\rvert$ | $3+4 = 7$ | sparsity penalties (Ch. 24) |
| $\ell_2$ | $\sqrt{\sum_i x_i^2}$ | $\sqrt{9+16} = 5$ | gradient clipping (Ch. 8), default "length" |
| $\ell_\infty$ | $\max_i\lvert x_i\rvert$ | $4$ | absmax quantization (Ch. 19) |

!!! llm "LLM connection: attention is a dot product"
    In attention (Ch. 12), a query vector $\mathbf{q}$ "looks for" key vectors $\mathbf{k}_j$ pointing in a similar direction.

    - The score is $\mathbf{q}^\top\mathbf{k}_j/\sqrt{d_h}$: a dot product.
    - Large when aligned, near zero when orthogonal, negative when opposed.
    - The $\sqrt{d_h}$ is explained in §1.12.

??? question "Check yourself"
    1. What is $(2, 1)^\top (1, -2)$? What does that tell you about the angle?
    2. Can two vectors in $\R^{1000}$ have cosine similarity $1.2$?

    **Answers.**

    1. $2 - 2 = 0$. They are orthogonal: $90^\circ$.
    2. No. Cauchy–Schwarz guarantees the cosine is in $[-1, 1]$ in every dimension.

---

## 1.3 Matrices are linear maps

**The idea:** a matrix is a recipe for transforming vectors, and it is completely determined by where it sends the basic directions.

### What a matrix is

A matrix $\mathbf{W}\in\R^{m\times n}$ is a grid of numbers with **$m$ rows and $n$ columns**. Entry $W_{ij}$ sits in row $i$, column $j$.

$$
\mathbf{W} = \begin{pmatrix} 1 & 0 & 2 \\ -1 & 3 & 1 \end{pmatrix}\in\R^{2\times 3},
\qquad W_{13} = 2,\quad W_{21} = -1 .
$$

### Multiplying a matrix by a vector

$\mathbf{W}\mathbf{x}$ needs $\mathbf{x}$ to have as many entries as $\mathbf{W}$ has columns. The result has as many entries as $\mathbf{W}$ has rows. There are two ways to compute it. They always agree.

!!! example "Worked example: two ways to compute $\mathbf{W}\mathbf{x}$"
    Take the $\mathbf{W}$ above and $\mathbf{x} = (2, 1, 1)^\top$.

    **Way 1 — rows.** Entry $i$ of the result is (row $i$ of $\mathbf{W}$) · $\mathbf{x}$:

    | Row | Dot product with $\mathbf{x}$ |
    |---|---|
    | $(1, 0, 2)$ | $1\cdot2 + 0\cdot1 + 2\cdot1 = 4$ |
    | $(-1, 3, 1)$ | $-1\cdot2 + 3\cdot1 + 1\cdot1 = 2$ |

    **Way 2 — columns.** The result is a mix of the columns, weighted by the entries of $\mathbf{x}$:

    $$
    2\begin{pmatrix}1\\-1\end{pmatrix} + 1\begin{pmatrix}0\\3\end{pmatrix} + 1\begin{pmatrix}2\\1\end{pmatrix} = \begin{pmatrix}2+0+2\\-2+3+1\end{pmatrix} = \begin{pmatrix}4\\2\end{pmatrix}.
    $$

    Both give $\mathbf{W}\mathbf{x} = (4, 2)^\top$.

In general:

$$
\underbrace{(\mathbf{W}\mathbf{x})_i = \sum_{j} W_{ij}x_j}_{\text{row } i \text{ dotted with } \mathbf{x}}
\qquad\Longleftrightarrow\qquad
\underbrace{\mathbf{W}\mathbf{x} = \sum_{j=1}^n x_j\,\mathbf{w}_{:,j}}_{\text{weighted sum of the columns } \mathbf{w}_{:,j}} .
$$

They agree because both compute the same numbers $W_{ij}x_j$ and add them, just grouped differently.

### Linear maps

!!! definition "Definition 1.3 (Linear map)"
    A function $f:\R^n\to\R^m$ is **linear** if, for all vectors $\mathbf{x},\mathbf{y}$ and numbers $a,b$,

    $$
    f(a\mathbf{x}+b\mathbf{y}) = a f(\mathbf{x}) + b f(\mathbf{y}).
    $$

    In words: **scaling and adding can be done before or after $f$, with the same result.**

!!! example "Worked example: linear or not?"
    | Function | Linear? | Reason |
    |---|---|---|
    | $f(x) = 3x$ | yes | $3(ax+by) = a(3x) + b(3y)$ |
    | $f(x) = x + 1$ | **no** | $f(0) = 1$, but a linear map must send $\mathbf{0}$ to $\mathbf{0}$ (take $a=b=0$) |
    | $f(x) = x^2$ | **no** | $f(2) = 4 \neq 2f(1) = 2$ |
    | $f(\mathbf{x}) = \mathbf{W}\mathbf{x}$ | yes | $\mathbf{W}(a\mathbf{x}+b\mathbf{y}) = a\mathbf{W}\mathbf{x} + b\mathbf{W}\mathbf{y}$ |

The **standard basis vectors** $\mathbf{e}_j$ have a 1 in slot $j$ and 0 everywhere else. In $\R^2$: $\mathbf{e}_1 = (1,0)$, $\mathbf{e}_2 = (0,1)$. Every vector is built from them: $\mathbf{x} = x_1\mathbf{e}_1 + \dots + x_n\mathbf{e}_n$.

!!! theorem "Theorem 1.4 (Every linear map is a matrix)"
    If $f:\R^n\to\R^m$ is linear, then $f(\mathbf{x}) = \mathbf{W}\mathbf{x}$, where $\mathbf{W}\in\R^{m\times n}$ is the matrix whose **$j$-th column is $f(\mathbf{e}_j)$**.

???+ proof "Proof"
    **Step 1 · Write $\mathbf{x}$ in the basis:** $\mathbf{x} = \sum_j x_j\mathbf{e}_j$.

    **Step 2 · Apply $f$ and use linearity** (pull out the numbers $x_j$, split the sum):

    $$
    f(\mathbf{x}) = f\Big(\sum_j x_j\mathbf{e}_j\Big) = \sum_j x_j\, f(\mathbf{e}_j).
    $$

    **Step 3 · Recognize the result.** That is a weighted sum of the vectors $f(\mathbf{e}_j)$ with weights $x_j$. By "Way 2" above, a weighted sum of columns is exactly $\mathbf{W}\mathbf{x}$ when the columns are $f(\mathbf{e}_j)$. $\blacksquare$

!!! example "Worked example: building a rotation matrix"
    Let $f$ rotate the plane $90^\circ$ counter-clockwise.

    - $\mathbf{e}_1 = (1,0)$ (pointing right) goes to $(0,1)$ (pointing up).
    - $\mathbf{e}_2 = (0,1)$ (pointing up) goes to $(-1,0)$ (pointing left).

    Put those as columns:

    $$
    \mathbf{W} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}.
    $$

    Check: $\mathbf{W}(2,1)^\top = (0\cdot2 - 1\cdot1,\; 1\cdot2 + 0\cdot1) = (-1, 2)$. Rotating "2 right, 1 up" by $90^\circ$ gives "1 left, 2 up". ✓

Play with it: the orange and green arrows are the images of $\mathbf{e}_1,\mathbf{e}_2$, i.e. the columns of $\mathbf{W}$. The whole grid follows along.

<div class="widget" data-widget="linear-transform"></div>

!!! intuition "Intuition"
    A neural-network layer $\mathbf{y} = \mathbf{W}\mathbf{x}+\mathbf{b}$:

    - **rotates, stretches, shears and projects** space ($\mathbf{W}\mathbf{x}$),
    - then shifts it ($+\mathbf{b}$).

    On its own this only produces straight-line "warps": grid lines stay straight and parallel. The nonlinearity that follows (Ch. 6) is what lets stacked layers bend space into arbitrary shapes.

---

## 1.4 Matrix multiplication is composition

**The idea:** multiplying two matrices gives the single matrix that does "first one, then the other".

!!! definition "Definition 1.4 (Matrix product)"
    For $\mathbf{A}\in\R^{m\times k}$ and $\mathbf{B}\in\R^{k\times n}$, the product $\mathbf{AB}\in\R^{m\times n}$ has entries

    $$
    (\mathbf{AB})_{ij} = \sum_{r=1}^{k} A_{ir}B_{rj} = (\text{row } i \text{ of } \mathbf{A})\cdot(\text{column } j \text{ of } \mathbf{B}).
    $$

**The shape rule.** The inner sizes must match, and they disappear:

$$
\underbrace{(m\times \boxed{k})}_{\mathbf{A}}\;\underbrace{(\boxed{k}\times n)}_{\mathbf{B}} \;\to\; (m\times n).
$$

| $\mathbf{A}$ | $\mathbf{B}$ | $\mathbf{AB}$ |
|---|---|---|
| $2\times3$ | $3\times4$ | $2\times4$ |
| $768\times768$ | $768\times1$ | $768\times1$ (matrix times vector) |
| $1\times5$ | $5\times1$ | $1\times1$ (a dot product) |
| $5\times1$ | $1\times5$ | $5\times5$ (an **outer product**) |
| $2\times3$ | $2\times3$ | not defined: $3\ne2$ |

!!! example "Worked example"
    $$
    \mathbf{A} = \begin{pmatrix}1&2\\3&4\end{pmatrix},\qquad \mathbf{B} = \begin{pmatrix}0&1\\1&0\end{pmatrix}.
    $$

    | Entry | Row of $\mathbf{A}$ · column of $\mathbf{B}$ | Value |
    |---|---|---|
    | $(1,1)$ | $(1,2)\cdot(0,1)$ | $2$ |
    | $(1,2)$ | $(1,2)\cdot(1,0)$ | $1$ |
    | $(2,1)$ | $(3,4)\cdot(0,1)$ | $4$ |
    | $(2,2)$ | $(3,4)\cdot(1,0)$ | $3$ |

    So $\mathbf{AB} = \begin{psmallmatrix}2&1\\4&3\end{psmallmatrix}$: $\mathbf{B}$ swaps coordinates, so $\mathbf{AB}$ is $\mathbf{A}$ with its columns swapped.

    The other order gives $\mathbf{BA} = \begin{psmallmatrix}3&4\\1&2\end{psmallmatrix}$: the *rows* swapped. Different!

!!! theorem "Theorem 1.5 (Product = composition)"
    $(\mathbf{AB})\mathbf{x} = \mathbf{A}(\mathbf{B}\mathbf{x})$ for every $\mathbf{x}$: the product matrix is the map "apply $\mathbf{B}$, then apply $\mathbf{A}$". Consequently:

    - matrix multiplication is **associative**: $(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$;
    - but in general it is **not commutative**: $\mathbf{AB}\ne\mathbf{BA}$.

???+ proof "Proof"
    **Step 1 · Write entry $i$ of $\mathbf{A}(\mathbf{B}\mathbf{x})$** using the row rule twice:

    $$
    \big(\mathbf{A}(\mathbf{B}\mathbf{x})\big)_i = \sum_r A_{ir}\,(\mathbf{B}\mathbf{x})_r = \sum_r A_{ir}\sum_j B_{rj}x_j .
    $$

    **Step 2 · Swap the order of summation.** Finite sums can be added in any order:

    $$
    = \sum_j\Big(\sum_r A_{ir}B_{rj}\Big)x_j .
    $$

    **Step 3 · Recognize the bracket.** It is $(\mathbf{AB})_{ij}$ by Definition 1.4. So the whole thing is $\big((\mathbf{AB})\mathbf{x}\big)_i$.

    **Step 4 · Associativity.** Applying "$\mathbf{C}$, then $\mathbf{B}$, then $\mathbf{A}$" is the same procedure however you group it, so $(\mathbf{AB})\mathbf{C}$ and $\mathbf{A}(\mathbf{BC})$ act identically on every vector. Matrices that act identically are equal (they have the same columns, by Theorem 1.4).

    **Step 5 · Non-commutativity.** One counterexample is enough. The worked example above has $\mathbf{AB}\ne\mathbf{BA}$. $\blacksquare$

### What a matmul costs

!!! theorem "Fact 1.6 (Cost of a matmul)"
    Computing $\mathbf{AB}$ with $\mathbf{A}\in\R^{m\times k}$ and $\mathbf{B}\in\R^{k\times n}$ takes **about $2mnk$ FLOPs** (floating-point operations).

Where the count comes from, step by step:

1. The result has $m\times n$ entries.
2. Each entry is a dot product of length $k$.
3. A dot product of length $k$ needs $k$ multiplies and $k-1\approx k$ adds: about $2k$ FLOPs.
4. Total: $mn\times 2k = 2mnk$.

!!! example "Worked example: one linear layer"
    A layer with weight $\mathbf{W}\in\R^{4096\times4096}$ applied to one token vector:

    - $m = 4096$, $k = 4096$, $n = 1$.
    - FLOPs $= 2\cdot4096\cdot4096\cdot1 \approx 3.4\times10^{7}$.
    - The layer has $4096^2\approx1.7\times10^7$ parameters.

    So: **2 FLOPs per parameter per token** in the forward pass. In Ch. 15 this grows into the famous $C \approx 6ND$.

!!! warning "Associativity is a performance tool"
    $(\mathbf{AB})\mathbf{x}$ and $\mathbf{A}(\mathbf{B}\mathbf{x})$ give the same answer. Their costs can be wildly different.

    Take $\mathbf{A}\in\R^{d\times r}$, $\mathbf{B}\in\R^{r\times d}$ with $d = 4096$, $r = 8$:

    | Order | Steps | FLOPs |
    |---|---|---|
    | $(\mathbf{AB})\mathbf{x}$ | build $d\times d$ matrix ($2d^2r$), then multiply ($2d^2$) | $\approx 3.0\times10^8$ |
    | $\mathbf{A}(\mathbf{B}\mathbf{x})$ | $\mathbf{B}\mathbf{x}$ ($2rd$), then $\mathbf{A}(\cdot)$ ($2dr$) | $\approx 1.3\times10^5$ |

    A factor of about 2,000. LoRA (Ch. 23) relies on exactly this.

### The transpose

The **transpose** flips a matrix across its diagonal: rows become columns. $(\mathbf{A}^\top)_{ij} = A_{ji}$.

$$
\begin{pmatrix}1&2&3\\4&5&6\end{pmatrix}^\top = \begin{pmatrix}1&4\\2&5\\3&6\end{pmatrix}.
$$

!!! theorem "Theorem 1.7 (Transpose rules)"
    1. $(\mathbf{AB})^\top = \mathbf{B}^\top\mathbf{A}^\top$ — the order reverses, like taking off socks and shoes.
    2. $\mathbf{x}^\top\mathbf{A}\mathbf{y} = (\mathbf{A}^\top\mathbf{x})^\top\mathbf{y}$ — a matrix can be moved to the other side of a dot product by transposing it.

???+ proof "Proof"
    **Rule 1, Step 1 · Look at entry $(i,j)$ of the left side.** By the definition of transpose, then of the product:

    $$
    \big((\mathbf{AB})^\top\big)_{ij} = (\mathbf{AB})_{ji} = \sum_r A_{jr}B_{ri} .
    $$

    **Rule 1, Step 2 · Rewrite each factor as a transpose entry** ($A_{jr} = (\mathbf{A}^\top)_{rj}$, $B_{ri} = (\mathbf{B}^\top)_{ir}$) and reorder the two numbers:

    $$
    = \sum_r (\mathbf{B}^\top)_{ir}(\mathbf{A}^\top)_{rj} = (\mathbf{B}^\top\mathbf{A}^\top)_{ij} .
    $$

    **Rule 2 · Use Rule 1 on $\mathbf{A}^\top\mathbf{x}$.** $(\mathbf{A}^\top\mathbf{x})^\top = \mathbf{x}^\top(\mathbf{A}^\top)^\top = \mathbf{x}^\top\mathbf{A}$. Multiply both sides by $\mathbf{y}$ on the right. $\blacksquare$

!!! llm "LLM connection: the attention bilinear form"
    The attention score between positions $i$ and $j$ uses queries $\mathbf{q}_i = \mathbf{W}_Q^\top\mathbf{x}_i$ and keys $\mathbf{k}_j = \mathbf{W}_K^\top\mathbf{x}_j$. Using Theorem 1.7:

    $$
    \mathbf{q}_i^\top\mathbf{k}_j = (\mathbf{W}_Q^\top\mathbf{x}_i)^\top(\mathbf{W}_K^\top\mathbf{x}_j) = \mathbf{x}_i^\top\,\underbrace{\mathbf{W}_Q\mathbf{W}_K^\top}_{d\times d}\,\mathbf{x}_j .
    $$

    - The $d\times d$ matrix in the middle has rank at most $d_h$ (next section).
    - Interpretability research (Ch. 24) calls it the **QK circuit**.

---

## 1.5 Span, independence, basis, rank

**The idea:** rank counts how many *genuinely different* directions a matrix can output.

### Independence

Some sets of vectors contain redundancy: one of them can be built from the others.

!!! example "Worked example"
    - $(1,2)$ and $(2,4)$ are **dependent**: $(2,4) = 2(1,2)$. Equivalently, $2\cdot(1,2) - 1\cdot(2,4) = \mathbf{0}$ with coefficients that are not all zero.
    - $(1,0)$ and $(0,1)$ are **independent**: $c_1(1,0) + c_2(0,1) = (c_1, c_2)$ is $\mathbf{0}$ only if $c_1 = c_2 = 0$.

!!! definition "Definition 1.5 (Independence, basis, rank, null space)"
    | Term | Definition | Plain words |
    |---|---|---|
    | **linearly independent** | $\sum_i c_i\mathbf{v}_i = \mathbf{0}$ forces every $c_i = 0$ | no vector is redundant |
    | **basis** of a subspace | an independent set that spans it | a minimal set of building blocks |
    | **dimension** | the number of vectors in any basis | number of independent directions |
    | **column space** of $\mathbf{W}$ | the span of its columns | every possible output $\mathbf{W}\mathbf{x}$ |
    | **rank** of $\mathbf{W}$ | the dimension of its column space | how many output directions |
    | **null space** of $\mathbf{W}$ | $\{\mathbf{x} : \mathbf{W}\mathbf{x} = \mathbf{0}\}$ | inputs that get erased |

!!! example "Worked example: rank and null space"
    $$
    \mathbf{W} = \begin{pmatrix}1&2\\2&4\end{pmatrix}.
    $$

    - **Columns:** $(1,2)$ and $(2,4) = 2(1,2)$. Only one independent direction, so $\rank\mathbf{W} = 1$. Every output lies on the line through $(1,2)$.
    - **Null space:** solve $\mathbf{W}\mathbf{x} = \mathbf{0}$, i.e. $x_1 + 2x_2 = 0$ (both rows say the same thing). Solutions: $\mathbf{x} = t(2,-1)$. A line, so $\dim\operatorname{null} = 1$.
    - **Check:** $\rank + \dim\operatorname{null} = 1 + 1 = 2 =$ number of columns. That's the next theorem.

!!! theorem "Theorem 1.8 (Rank–nullity)"
    For $\mathbf{W}\in\R^{m\times n}$:

    $$
    \rank(\mathbf{W}) + \dim\operatorname{null}(\mathbf{W}) = n .
    $$

    Every input direction is either **kept** (contributes to the rank) or **erased** (lies in the null space).

??? proof "Proof"
    **Step 1 · Basis of the null space.** Let $\mathbf{u}_1,\dots,\mathbf{u}_k$ be a basis of the null space, so $k = \dim\operatorname{null}$.

    **Step 2 · Extend it to a basis of all of $\R^n$** by adding vectors $\mathbf{w}_1,\dots,\mathbf{w}_r$. A basis of $\R^n$ has $n$ vectors, so $k + r = n$. We will show $r = \rank\mathbf{W}$.

    **Step 3 · The images $\mathbf{W}\mathbf{w}_i$ span the column space.** Any input is $\mathbf{x} = \sum a_i\mathbf{u}_i + \sum b_i\mathbf{w}_i$. Since $\mathbf{W}\mathbf{u}_i = \mathbf{0}$,

    $$
    \mathbf{W}\mathbf{x} = \sum_i b_i\,\mathbf{W}\mathbf{w}_i .
    $$

    So every output is a combination of the $r$ vectors $\mathbf{W}\mathbf{w}_i$.

    **Step 4 · They are independent.** Suppose $\sum c_i\mathbf{W}\mathbf{w}_i = \mathbf{0}$. Then $\mathbf{W}(\sum c_i\mathbf{w}_i) = \mathbf{0}$, so $\sum c_i\mathbf{w}_i$ is in the null space. That means it equals some $\sum d_j\mathbf{u}_j$, i.e.

    $$
    \sum_i c_i\mathbf{w}_i - \sum_j d_j\mathbf{u}_j = \mathbf{0}.
    $$

    All the $\mathbf{u}$'s and $\mathbf{w}$'s together form a basis, which is independent, so every coefficient is 0. In particular every $c_i = 0$.

    **Step 5 · Conclude.** The $r$ vectors $\mathbf{W}\mathbf{w}_i$ are an independent spanning set of the column space, i.e. a basis. So $\rank\mathbf{W} = r = n - k$. $\blacksquare$

### Low rank means cheap

Two consequences you will use:

1. **$\rank(\mathbf{AB}) \le \min(\rank\mathbf{A},\rank\mathbf{B})$.**
    - Why $\le \rank\mathbf{A}$: every column of $\mathbf{AB}$ is $\mathbf{A}$ times something, so it lies in the column space of $\mathbf{A}$.
    - Why $\le \rank\mathbf{B}$: by rank–nullity, $\rank\mathbf{AB} = n - \dim\operatorname{null}(\mathbf{AB})$, and anything $\mathbf{B}$ erases, $\mathbf{AB}$ also erases.
2. **A product through a narrow middle has low rank.** If $\mathbf{B}\in\R^{d\times r}$ and $\mathbf{A}\in\R^{r\times k}$, then $\mathbf{BA}$ is $d\times k$ but has rank at most $r$.

!!! example "Worked example: rank-1 matrices"
    An **outer product** $\mathbf{u}\mathbf{v}^\top$ has rank 1:

    $$
    \begin{pmatrix}1\\2\\3\end{pmatrix}\begin{pmatrix}4&5\end{pmatrix} = \begin{pmatrix}4&5\\8&10\\12&15\end{pmatrix}.
    $$

    Every column is a multiple of $(1,2,3)^\top$. Six numbers stored, but only five ($3+2$) needed to describe it.

**Storage.** A rank-$r$ matrix $\mathbf{BA}$ of size $d\times k$ needs $r(d+k)$ numbers instead of $dk$.

| $d\times k$ | Full | Rank 8 | Saving |
|---|---|---|---|
| $4096\times4096$ | $16.8$M | $65.5$k | $256\times$ |

That is the whole idea of LoRA and of low-rank compression.

<div class="widget" data-widget="lora"></div>

---

## 1.6 Inverses and determinants

**The idea:** a map can be undone exactly when it doesn't squash space flat. The determinant measures the squashing.

### Inverses

The **identity matrix** $\mathbf{I}$ has 1s on the diagonal and 0s elsewhere. It does nothing: $\mathbf{I}\mathbf{x} = \mathbf{x}$.

A square $\mathbf{W}\in\R^{n\times n}$ is **invertible** if some $\mathbf{W}^{-1}$ undoes it:

$$
\mathbf{W}^{-1}\mathbf{W} = \mathbf{W}\mathbf{W}^{-1} = \mathbf{I}.
$$

!!! example "Worked example: the $2\times2$ inverse"
    For a $2\times2$ matrix there is a formula. Swap the diagonal, negate the off-diagonal, divide by $ad-bc$:

    $$
    \begin{pmatrix}a&b\\c&d\end{pmatrix}^{-1} = \frac{1}{ad-bc}\begin{pmatrix}d&-b\\-c&a\end{pmatrix}.
    $$

    Take $\mathbf{W} = \begin{psmallmatrix}2&1\\1&1\end{psmallmatrix}$. Here $ad - bc = 2 - 1 = 1$, so $\mathbf{W}^{-1} = \begin{psmallmatrix}1&-1\\-1&2\end{psmallmatrix}$.

    **Check:** $\begin{psmallmatrix}2&1\\1&1\end{psmallmatrix}\begin{psmallmatrix}1&-1\\-1&2\end{psmallmatrix} = \begin{psmallmatrix}2-1 & -2+2\\ 1-1 & -1+2\end{psmallmatrix} = \begin{psmallmatrix}1&0\\0&1\end{psmallmatrix}$. ✓

    If $ad - bc = 0$, the formula divides by zero: no inverse exists.

### Determinants

The **determinant** $\det\mathbf{W}$ is the factor by which $\mathbf{W}$ scales area (in 2-D) or volume (in higher dimensions). Its sign says whether orientation flips.

For $2\times2$: $\det\begin{psmallmatrix}a&b\\c&d\end{psmallmatrix} = ad-bc$. That is the signed area of the parallelogram spanned by the two columns. Watch the shaded square in the widget in §1.3.

!!! example "Worked example: reading determinants"
    | Matrix | $\det$ | What happens to the unit square |
    |---|---|---|
    | $\begin{psmallmatrix}2&0\\0&3\end{psmallmatrix}$ | $6$ | stretched to a $2\times3$ rectangle: area $\times6$ |
    | $\begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix}$ (rotation) | $1$ | turned, same area |
    | $\begin{psmallmatrix}1&0\\0&-1\end{psmallmatrix}$ (mirror) | $-1$ | flipped: same area, orientation reversed |
    | $\begin{psmallmatrix}1&2\\2&4\end{psmallmatrix}$ | $0$ | squashed onto a line: area $0$ |

Two facts:

- **Composition multiplies:** $\det(\mathbf{AB}) = \det\mathbf{A}\,\det\mathbf{B}$. Scaling area by 2 and then by 3 scales it by 6.
- **Zero means information is lost.** If $\det\mathbf{W} = 0$, space is squashed into a lower dimension. Different inputs land on the same output, so the map can't be undone.

### The invertible matrix theorem

For a square $n\times n$ matrix, all of these say the same thing:

| Condition | Why it's equivalent |
|---|---|
| $\mathbf{W}$ is invertible | (the definition) |
| $\rank\mathbf{W} = n$ | outputs fill all of $\R^n$, so every output can be traced back |
| $\operatorname{null}\mathbf{W} = \{\mathbf{0}\}$ | rank–nullity: rank $n$ ⇔ null space dimension $0$ |
| $\det\mathbf{W}\neq 0$ | nothing is squashed flat |
| $0$ is not an eigenvalue (§1.8) | $\mathbf{W}\mathbf{v} = 0\cdot\mathbf{v}$ would put $\mathbf{v}\ne\mathbf{0}$ in the null space |

!!! watch "Watch"
    3Blue1Brown: [Linear transformations and matrices](https://www.youtube.com/watch?v=kYB8IZa5AuE) · [Matrix multiplication as composition](https://www.youtube.com/watch?v=XkY2DOUCWMU) · [The determinant](https://www.youtube.com/watch?v=Ip3X9LOh2dk) · [Dot products and duality](https://www.youtube.com/watch?v=LyGKycYT2v0)

---

## 1.7 Orthogonal matrices: rotations

**The idea:** some matrices move vectors around without changing any lengths or angles. Rotations are the main example.

**Orthonormal** vectors are vectors that are each of length 1 and mutually perpendicular. In $\R^2$, $(1,0)$ and $(0,1)$ are orthonormal. So are $\frac{1}{\sqrt2}(1,1)$ and $\frac{1}{\sqrt2}(1,-1)$.

!!! definition "Definition 1.6 (Orthogonal matrix)"
    A square matrix $\mathbf{Q}$ is **orthogonal** if $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$. Then the inverse is free: $\mathbf{Q}^{-1} = \mathbf{Q}^\top$.

Why does $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$ mean "orthonormal columns"? Entry $(i,j)$ of $\mathbf{Q}^\top\mathbf{Q}$ is (row $i$ of $\mathbf{Q}^\top$) · (column $j$ of $\mathbf{Q}$) $=$ (column $i$) · (column $j$). So:

| Entry of $\mathbf{Q}^\top\mathbf{Q}$ | Equals | Requirement $=\mathbf{I}$ means |
|---|---|---|
| diagonal, $i = j$ | $\mathbf{q}_i^\top\mathbf{q}_i = \lVert\mathbf{q}_i\rVert^2$ | every column has length 1 |
| off-diagonal, $i\ne j$ | $\mathbf{q}_i^\top\mathbf{q}_j$ | different columns are perpendicular |

!!! theorem "Theorem 1.9 (Orthogonal maps preserve geometry)"
    Orthogonal maps preserve dot products:

    $$
    (\mathbf{Q}\mathbf{x})^\top(\mathbf{Q}\mathbf{y}) = \mathbf{x}^\top\mathbf{y}.
    $$

    Since lengths and angles are defined from dot products, they are preserved too.

???+ proof "Proof"
    **Step 1 · Transpose of a product** (Theorem 1.7): $(\mathbf{Q}\mathbf{x})^\top = \mathbf{x}^\top\mathbf{Q}^\top$. So

    $$
    (\mathbf{Q}\mathbf{x})^\top(\mathbf{Q}\mathbf{y}) = \mathbf{x}^\top\mathbf{Q}^\top\mathbf{Q}\,\mathbf{y}.
    $$

    **Step 2 · Use the definition** $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$:

    $$
    = \mathbf{x}^\top\mathbf{I}\,\mathbf{y} = \mathbf{x}^\top\mathbf{y}. \qquad\blacksquare
    $$

### The 2-D rotation matrix

Use Theorem 1.4: find where the basis vectors go.

- Rotating $\mathbf{e}_1 = (1,0)$ by angle $\phi$ lands on the unit circle at $(\cos\phi, \sin\phi)$.
- Rotating $\mathbf{e}_2 = (0,1)$ by $\phi$ lands at $(-\sin\phi, \cos\phi)$ (it starts $90^\circ$ ahead).

Those are the columns:

$$
\mathbf{R}(\phi) = \begin{pmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{pmatrix}.
$$

Three properties, each with a one-line check:

| Property | Check |
|---|---|
| It's orthogonal | columns have length $\cos^2\phi+\sin^2\phi = 1$ and dot product $-\cos\phi\sin\phi + \sin\phi\cos\phi = 0$ |
| $\mathbf{R}(\phi)^\top = \mathbf{R}(-\phi)$ | $\cos(-\phi) = \cos\phi$ and $\sin(-\phi) = -\sin\phi$, so transposing just flips the sign of $\sin$ |
| $\mathbf{R}(\alpha)\mathbf{R}(\beta) = \mathbf{R}(\alpha+\beta)$ | see below |

!!! example "Worked check: rotations compose by adding angles"
    Multiply out the top-left entry of $\mathbf{R}(\alpha)\mathbf{R}(\beta)$:

    $$
    \cos\alpha\cos\beta - \sin\alpha\sin\beta = \cos(\alpha+\beta).
    $$

    And the bottom-left entry:

    $$
    \sin\alpha\cos\beta + \cos\alpha\sin\beta = \sin(\alpha+\beta).
    $$

    These are exactly the angle-addition formulas from trigonometry. The other two entries work the same way. So "rotate by $\beta$, then by $\alpha$" = "rotate by $\alpha+\beta$".

!!! llm "LLM connection: RoPE"
    Rotary position embeddings (Ch. 11) rotate the query at position $m$ by $\mathbf{R}(m\theta)$, and the key at position $n$ by $\mathbf{R}(n\theta)$. Their score:

    | Step | Expression | Rule used |
    |---|---|---|
    | 1 | $(\mathbf{R}(m\theta)\mathbf{q})^\top(\mathbf{R}(n\theta)\mathbf{k})$ | the score |
    | 2 | $\mathbf{q}^\top\,\mathbf{R}(m\theta)^\top\,\mathbf{R}(n\theta)\,\mathbf{k}$ | transpose of a product |
    | 3 | $\mathbf{q}^\top\,\mathbf{R}(-m\theta)\,\mathbf{R}(n\theta)\,\mathbf{k}$ | $\mathbf{R}^\top = \mathbf{R}(-\cdot)$ |
    | 4 | $\mathbf{q}^\top\,\mathbf{R}((n-m)\theta)\,\mathbf{k}$ | angles add |

    The result depends only on the **relative** position $n-m$. A few lines of algebra from this section are behind one of the most important ideas in modern LLMs.

---

## 1.8 Eigenvalues, eigenvectors, and the spectral theorem

**The idea:** most vectors get turned by a matrix. Eigenvectors are the special directions that only get stretched.

!!! definition "Definition 1.7 (Eigenvector, eigenvalue)"
    A **nonzero** vector $\mathbf{v}$ is an **eigenvector** of a square matrix $\mathbf{A}$, with **eigenvalue** $\lambda$, if

    $$
    \mathbf{A}\mathbf{v} = \lambda\mathbf{v}.
    $$

    Along $\mathbf{v}$, the matrix acts like the plain number $\lambda$.

!!! example "Worked example: spotting eigenvectors"
    $$
    \mathbf{A} = \begin{pmatrix}2&1\\1&2\end{pmatrix}.
    $$

    | Try $\mathbf{v}$ | $\mathbf{A}\mathbf{v}$ | Eigenvector? |
    |---|---|---|
    | $(1, 0)$ | $(2, 1)$ | no: direction changed |
    | $(1, 1)$ | $(3, 3) = 3\cdot(1,1)$ | **yes**, $\lambda = 3$ |
    | $(1, -1)$ | $(1, -1) = 1\cdot(1,-1)$ | **yes**, $\lambda = 1$ |

    So $\mathbf{A}$ stretches the diagonal direction by 3 and leaves the anti-diagonal alone.

### Finding eigenvalues: $\det(\mathbf{A}-\lambda\mathbf{I}) = 0$ { data-toc-label="Finding eigenvalues: det(A − λI) = 0" }

Guessing doesn't scale. Here is the systematic method, step by step.

1. $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ is the same as $\mathbf{A}\mathbf{v} - \lambda\mathbf{I}\mathbf{v} = \mathbf{0}$, i.e. $(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}$.
2. So $\mathbf{v}\neq\mathbf{0}$ is in the null space of $\mathbf{A}-\lambda\mathbf{I}$.
3. A nonzero null space means $\mathbf{A}-\lambda\mathbf{I}$ is **not** invertible (§1.6).
4. Not invertible means $\det(\mathbf{A}-\lambda\mathbf{I}) = 0$.

This is a polynomial equation in $\lambda$, the **characteristic polynomial**. Its roots are the eigenvalues.

!!! example "Worked example: the same matrix, systematically"
    $$
    \det\begin{pmatrix}2-\lambda & 1\\ 1 & 2-\lambda\end{pmatrix} = (2-\lambda)^2 - 1 = 0
    \;\Longrightarrow\; 2-\lambda = \pm1
    \;\Longrightarrow\; \lambda \in\{1, 3\}.
    $$

    For each $\lambda$, solve $(\mathbf{A}-\lambda\mathbf{I})\mathbf{v} = \mathbf{0}$:

    - $\lambda = 3$: $\begin{psmallmatrix}-1&1\\1&-1\end{psmallmatrix}\mathbf{v} = \mathbf{0}$ gives $v_1 = v_2$, so $\mathbf{v} = (1,1)$.
    - $\lambda = 1$: $\begin{psmallmatrix}1&1\\1&1\end{psmallmatrix}\mathbf{v} = \mathbf{0}$ gives $v_1 = -v_2$, so $\mathbf{v} = (1,-1)$.

!!! example "Worked example: a rotation has no real eigenvectors"
    For the $90^\circ$ rotation $\begin{psmallmatrix}0&-1\\1&0\end{psmallmatrix}$:

    $$
    \det\begin{pmatrix}-\lambda&-1\\1&-\lambda\end{pmatrix} = \lambda^2+1 = 0 \;\Longrightarrow\; \lambda = \pm i .
    $$

    No real solutions. Makes sense: a rotation moves *every* direction. In the widget in §1.3, the dashed lines are eigenvector directions; for a rotation they disappear.

### Symmetric matrices

A matrix is **symmetric** if $\mathbf{A} = \mathbf{A}^\top$ (mirror image across the diagonal). They are the most important case in ML:

- covariance matrices (Ch. 3),
- Hessians, the matrices of second derivatives (Ch. 2, Ch. 5),
- Gram matrices $\mathbf{X}^\top\mathbf{X}$.

Symmetric matrices are as nice as they could possibly be:

!!! theorem "Theorem 1.10 (Spectral theorem)"
    A real symmetric $\mathbf{A}\in\R^{n\times n}$ has:

    1. only **real** eigenvalues, and
    2. an **orthonormal basis** of eigenvectors.

    Equivalently, $\mathbf{A} = \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$, where $\mathbf{Q}$ is orthogonal (its columns are the eigenvectors) and $\boldsymbol\Lambda$ is diagonal (the eigenvalues).

How to read $\mathbf{A}\mathbf{x} = \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top\mathbf{x}$, right to left:

| Step | Operation | Meaning |
|---|---|---|
| 1 | $\mathbf{Q}^\top\mathbf{x}$ | rotate into eigen-coordinates (how much of $\mathbf{x}$ lies along each eigenvector) |
| 2 | $\boldsymbol\Lambda(\cdot)$ | stretch each coordinate by its eigenvalue |
| 3 | $\mathbf{Q}(\cdot)$ | rotate back |

!!! example "Worked check"
    For $\mathbf{A} = \begin{psmallmatrix}2&1\\1&2\end{psmallmatrix}$, normalize the eigenvectors to length 1:

    $$
    \mathbf{Q} = \frac{1}{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\qquad \boldsymbol\Lambda = \begin{pmatrix}3&0\\0&1\end{pmatrix}.
    $$

    Multiply out $\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top = \frac12\begin{psmallmatrix}1&1\\1&-1\end{psmallmatrix}\begin{psmallmatrix}3&3\\1&-1\end{psmallmatrix} = \frac12\begin{psmallmatrix}4&2\\2&4\end{psmallmatrix} = \mathbf{A}$. ✓

!!! recall "Refresher: complex conjugates"
    A complex number is $z = a + bi$ with $i^2 = -1$.

    - Its **conjugate** is $\bar z = a - bi$.
    - $z\bar z = a^2 + b^2 = \lvert z\rvert^2$ is always real and $\ge 0$.
    - $z$ is real exactly when $z = \bar z$.
    - Conjugating a product conjugates each factor: $\overline{zw} = \bar z\,\bar w$.

??? proof "Proof of the spectral theorem (three parts)"
    **Part 1 · Eigenvalues are real.**

    *Step 1.* Let $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$, where $\mathbf{v}$ and $\lambda$ might be complex. Consider the number

    $$
    s = \bar{\mathbf{v}}^\top\mathbf{A}\mathbf{v}.
    $$

    *Step 2 · $s$ equals its own conjugate.* Conjugate each factor ($\mathbf{A}$ is real, so $\bar{\mathbf{A}} = \mathbf{A}$), then transpose the $1\times1$ result (which doesn't change a number), then use $\mathbf{A}^\top = \mathbf{A}$:

    $$
    \bar s = \mathbf{v}^\top\mathbf{A}\bar{\mathbf{v}} = (\mathbf{v}^\top\mathbf{A}\bar{\mathbf{v}})^\top = \bar{\mathbf{v}}^\top\mathbf{A}^\top\mathbf{v} = \bar{\mathbf{v}}^\top\mathbf{A}\mathbf{v} = s .
    $$

    So $s$ is real.

    *Step 3 · Compute $s$ another way.* Using $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$:

    $$
    s = \lambda\,\bar{\mathbf{v}}^\top\mathbf{v} = \lambda\sum_i\lvert v_i\rvert^2 .
    $$

    *Step 4.* $\sum_i\lvert v_i\rvert^2$ is real and positive ($\mathbf{v}\ne\mathbf{0}$). So $\lambda = s / \sum_i\lvert v_i\rvert^2$ is a real number divided by a positive real number: real.

    **Part 2 · Eigenvectors with different eigenvalues are orthogonal.**

    *Step 1.* Let $\mathbf{A}\mathbf{v}_1=\lambda_1\mathbf{v}_1$ and $\mathbf{A}\mathbf{v}_2=\lambda_2\mathbf{v}_2$ with $\lambda_1\ne\lambda_2$.

    *Step 2.* Compute $\mathbf{v}_1^\top\mathbf{A}\mathbf{v}_2$ two ways:

    - grouping as $(\mathbf{A}^\top\mathbf{v}_1)^\top\mathbf{v}_2 = (\mathbf{A}\mathbf{v}_1)^\top\mathbf{v}_2 = \lambda_1\,\mathbf{v}_1^\top\mathbf{v}_2$ (Theorem 1.7 and symmetry);
    - grouping as $\mathbf{v}_1^\top(\mathbf{A}\mathbf{v}_2) = \lambda_2\,\mathbf{v}_1^\top\mathbf{v}_2$.

    *Step 3.* Subtract: $(\lambda_1-\lambda_2)\,\mathbf{v}_1^\top\mathbf{v}_2 = 0$. Since $\lambda_1-\lambda_2\ne0$, we get $\mathbf{v}_1^\top\mathbf{v}_2 = 0$.

    **Part 3 · A full orthonormal basis exists** (by induction on the size $n$).

    *Step 1.* For $n=1$ there is nothing to prove.

    *Step 2.* For general $n$, take one eigenvector and scale it to length 1: $\mathbf{q}_1$, with eigenvalue $\lambda_1$ (one exists, since the characteristic polynomial has a root, which is real by Part 1).

    *Step 3.* Let $\mathbf{q}_1^\perp$ be the set of vectors perpendicular to $\mathbf{q}_1$: an $(n-1)$-dimensional subspace. $\mathbf{A}$ maps it into itself: if $\mathbf{q}_1^\top\mathbf{x} = 0$, then

    $$
    \mathbf{q}_1^\top(\mathbf{A}\mathbf{x}) = (\mathbf{A}\mathbf{q}_1)^\top\mathbf{x} = \lambda_1\,\mathbf{q}_1^\top\mathbf{x} = 0 .
    $$

    *Step 4.* Restricted to that subspace, $\mathbf{A}$ is still symmetric. By the induction hypothesis it has an orthonormal eigenbasis $\mathbf{q}_2,\dots,\mathbf{q}_n$ there. Together with $\mathbf{q}_1$ that is an orthonormal eigenbasis of $\R^n$. $\blacksquare$

### Positive (semi)definite matrices

!!! definition "Definition 1.8 (Positive (semi)definite)"
    A symmetric $\mathbf{A}$ is

    - **positive semidefinite** (PSD, written $\mathbf{A}\succeq 0$) if $\mathbf{x}^\top\mathbf{A}\mathbf{x}\ge 0$ for every $\mathbf{x}$;
    - **positive definite** ($\mathbf{A}\succ 0$) if $\mathbf{x}^\top\mathbf{A}\mathbf{x} > 0$ for every $\mathbf{x}\neq\mathbf{0}$.

The number $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ is a **quadratic form**. For $\mathbf{A} = \begin{psmallmatrix}2&1\\1&2\end{psmallmatrix}$ it is $2x_1^2 + 2x_1x_2 + 2x_2^2$. Is that always $\ge0$? The spectral theorem answers it instantly:

!!! theorem "PSD ⇔ all eigenvalues ≥ 0"
    $$
    \mathbf{x}^\top\mathbf{A}\mathbf{x} = \sum_i\lambda_i\,(\mathbf{q}_i^\top\mathbf{x})^2 ,
    $$

    so $\mathbf{A}$ is PSD exactly when every $\lambda_i\ge 0$ (and positive definite when every $\lambda_i > 0$).

???+ proof "Proof"
    **Step 1 · Substitute** $\mathbf{A} = \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$ and let $\mathbf{c} = \mathbf{Q}^\top\mathbf{x}$, so $c_i = \mathbf{q}_i^\top\mathbf{x}$:

    $$
    \mathbf{x}^\top\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top\mathbf{x} = (\mathbf{Q}^\top\mathbf{x})^\top\boldsymbol\Lambda(\mathbf{Q}^\top\mathbf{x}) = \mathbf{c}^\top\boldsymbol\Lambda\mathbf{c} = \sum_i\lambda_ic_i^2 .
    $$

    **Step 2 · If every $\lambda_i\ge0$**, every term is $\ge0$, so the sum is $\ge0$.

    **Step 3 · If some $\lambda_k < 0$**, pick $\mathbf{x} = \mathbf{q}_k$. Then $\mathbf{c} = \mathbf{e}_k$ and the sum is just $\lambda_k < 0$. Not PSD. $\blacksquare$

    For the example, $\lambda\in\{1,3\}$, so $2x_1^2 + 2x_1x_2 + 2x_2^2 > 0$ for every $\mathbf{x}\ne\mathbf{0}$.

**Every Gram matrix is PSD**, in one line: $\mathbf{x}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{x} = (\mathbf{X}\mathbf{x})^\top(\mathbf{X}\mathbf{x}) = \lVert\mathbf{X}\mathbf{x}\rVert^2 \ge 0$.

!!! llm "Preview: the condition number"
    In Ch. 5, the eigenvalues of the loss Hessian decide how fast gradient descent converges.

    - The ratio $\kappa = \lambda_\max/\lambda_\min$ is the **condition number**.
    - For $\begin{psmallmatrix}2&1\\1&2\end{psmallmatrix}$, $\kappa = 3/1 = 3$: mild.
    - Deep-network Hessians have enormous $\kappa$. That is the enemy.

!!! watch "Watch"
    3Blue1Brown: [Eigenvectors and eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g) · [Change of basis](https://www.youtube.com/watch?v=P2LTAUO1TdA)

---

## 1.9 The singular value decomposition (SVD)

**The idea:** *every* matrix — any shape — is a rotation, then a stretch along the axes, then another rotation.

Eigen-decomposition needs a square matrix, and is only guaranteed to be nice for symmetric ones. The SVD works for **every** matrix. It is arguably the most useful theorem in applied linear algebra.

!!! theorem "Theorem 1.11 (SVD)"
    Any $\mathbf{A}\in\R^{m\times n}$ of rank $r$ can be written

    $$
    \mathbf{A} = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top = \sum_{i=1}^{r}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top .
    $$

The pieces:

| Piece | Shape | What it is |
|---|---|---|
| $\mathbf{V}$ | $n\times n$, orthogonal | input directions $\mathbf{v}_i$ ("right singular vectors") |
| $\boldsymbol\Sigma$ | $m\times n$, diagonal | **singular values** $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$, then zeros |
| $\mathbf{U}$ | $m\times m$, orthogonal | output directions $\mathbf{u}_i$ ("left singular vectors") |

The key relation: **$\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$.** Input direction $\mathbf{v}_i$ is sent to output direction $\mathbf{u}_i$, stretched by $\sigma_i$.

The sum form says $\mathbf{A}$ is a sum of $r$ rank-1 pieces $\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$, ordered from most to least important.

!!! example "Worked example: an SVD by hand"
    $$
    \mathbf{A} = \begin{pmatrix}3&0\\4&5\end{pmatrix}.
    $$

    **Step 1 · Form $\mathbf{A}^\top\mathbf{A}$:**

    $$
    \mathbf{A}^\top\mathbf{A} = \begin{pmatrix}3&4\\0&5\end{pmatrix}\begin{pmatrix}3&0\\4&5\end{pmatrix} = \begin{pmatrix}25&20\\20&25\end{pmatrix}.
    $$

    **Step 2 · Its eigenvalues and eigenvectors.** Same pattern as the $\begin{psmallmatrix}2&1\\1&2\end{psmallmatrix}$ example: eigenvalues $25\pm20 = 45, 5$, with eigenvectors $\mathbf{v}_1 = \frac{1}{\sqrt2}(1,1)$ and $\mathbf{v}_2 = \frac{1}{\sqrt2}(1,-1)$.

    **Step 3 · Singular values** are the square roots: $\sigma_1 = \sqrt{45} = 3\sqrt5\approx6.71$, $\sigma_2 = \sqrt5\approx2.24$.

    **Step 4 · Output directions** $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$:

    $$
    \mathbf{u}_1 = \frac{1}{3\sqrt5}\cdot\frac{1}{\sqrt2}\begin{pmatrix}3\\9\end{pmatrix} = \frac{1}{\sqrt{10}}\begin{pmatrix}1\\3\end{pmatrix},
    \qquad
    \mathbf{u}_2 = \frac{1}{\sqrt5}\cdot\frac{1}{\sqrt2}\begin{pmatrix}3\\-1\end{pmatrix} = \frac{1}{\sqrt{10}}\begin{pmatrix}3\\-1\end{pmatrix}.
    $$

    **Check:** $\mathbf{u}_1^\top\mathbf{u}_2 = \frac{1}{10}(3 - 3) = 0$. ✓ Orthonormal, as promised.

??? proof "Proof of the SVD (it is the construction above, in general)"
    **Step 1 · $\mathbf{A}^\top\mathbf{A}$ is symmetric and PSD.** Symmetric: $(\mathbf{A}^\top\mathbf{A})^\top = \mathbf{A}^\top\mathbf{A}$. PSD: it's a Gram matrix (§1.8).

    **Step 2 · Apply the spectral theorem.** $\mathbf{A}^\top\mathbf{A}$ has orthonormal eigenvectors $\mathbf{v}_1,\dots,\mathbf{v}_n$, with eigenvalues $\lambda_1\ge\dots\ge\lambda_n\ge0$.

    **Step 3 · Exactly $r$ eigenvalues are positive.** $\mathbf{A}^\top\mathbf{A}$ and $\mathbf{A}$ have the same null space: if $\mathbf{A}^\top\mathbf{A}\mathbf{x}=\mathbf{0}$, then $\lVert\mathbf{A}\mathbf{x}\rVert^2 = \mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}=0$, so $\mathbf{A}\mathbf{x} = \mathbf{0}$ (and the converse is obvious). By rank–nullity they have the same rank $r$, so exactly $r$ eigenvalues are nonzero.

    **Step 4 · Define** $\sigma_i = \sqrt{\lambda_i}$ and, for $i\le r$, $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$.

    **Step 5 · The $\mathbf{u}_i$ are orthonormal:**

    $$
    \mathbf{u}_i^\top\mathbf{u}_j = \frac{\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_j}{\sigma_i\sigma_j} = \frac{\lambda_j\,\mathbf{v}_i^\top\mathbf{v}_j}{\sigma_i\sigma_j} = \begin{cases}1 & i=j\\ 0 & i\ne j\end{cases}.
    $$

    (For $i=j$: $\lambda_i/\sigma_i^2 = 1$.)

    **Step 6 · Assemble.** Extend the $\mathbf{u}_i$ to an orthonormal basis of $\R^m$. By construction $\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$ for $i\le r$, and $\mathbf{A}\mathbf{v}_i = \mathbf{0}$ for $i>r$ (those $\mathbf{v}_i$ are in the null space). In matrix form, $\mathbf{A}\mathbf{V} = \mathbf{U}\boldsymbol\Sigma$. Multiply on the right by $\mathbf{V}^\top$ (using $\mathbf{V}\mathbf{V}^\top = \mathbf{I}$). $\blacksquare$

**Geometry.** Reading $\mathbf{A}\mathbf{x} = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top\mathbf{x}$ right to left: rotate ($\mathbf{V}^\top$), stretch each axis by $\sigma_i$ ($\boldsymbol\Sigma$), rotate again ($\mathbf{U}$).

The biggest stretch is the **spectral norm**:

$$
\lVert\mathbf{A}\rVert_2 = \max_{\lVert\mathbf{x}\rVert=1}\lVert\mathbf{A}\mathbf{x}\rVert = \sigma_1 .
$$

### The best low-rank approximation

To compare matrices we need a size for the error. Two standard choices:

| Norm | Definition | Meaning | In terms of $\sigma$'s |
|---|---|---|---|
| spectral $\lVert\mathbf{M}\rVert_2$ | $\max_{\lVert\mathbf{x}\rVert=1}\lVert\mathbf{M}\mathbf{x}\rVert$ | worst-case stretch | $\sigma_1(\mathbf{M})$ |
| Frobenius $\lVert\mathbf{M}\rVert_F$ | $\sqrt{\sum_{i,j}M_{ij}^2}$ | treat the matrix as one long vector | $\sqrt{\sum_i\sigma_i(\mathbf{M})^2}$ (Exercise 4) |

!!! theorem "Theorem 1.12 (Eckart–Young–Mirsky: best low-rank approximation)"
    Keep only the first $k$ terms of the SVD:

    $$
    \mathbf{A}_k = \sum_{i=1}^{k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top .
    $$

    This is the **best** rank-$k$ approximation of $\mathbf{A}$, in both norms:

    $$
    \min_{\rank\mathbf{B}\le k}\lVert\mathbf{A}-\mathbf{B}\rVert_2 = \sigma_{k+1},\qquad
    \min_{\rank\mathbf{B}\le k}\lVert\mathbf{A}-\mathbf{B}\rVert_F = \sqrt{\textstyle\sum_{i>k}\sigma_i^2}.
    $$

!!! example "Worked example: rank-1 approximation of the matrix above"
    $$
    \mathbf{A}_1 = \sigma_1\mathbf{u}_1\mathbf{v}_1^\top = 3\sqrt5\cdot\frac{1}{\sqrt{10}}\begin{pmatrix}1\\3\end{pmatrix}\cdot\frac{1}{\sqrt2}\begin{pmatrix}1&1\end{pmatrix} = \frac32\begin{pmatrix}1&1\\3&3\end{pmatrix} = \begin{pmatrix}1.5&1.5\\4.5&4.5\end{pmatrix}.
    $$

    The error is $\mathbf{A}-\mathbf{A}_1 = \begin{psmallmatrix}1.5&-1.5\\-0.5&0.5\end{psmallmatrix}$. Its squared Frobenius norm is $2.25+2.25+0.25+0.25 = 5 = \sigma_2^2$. ✓ Exactly as the theorem predicts.

??? proof "Proof (spectral norm version)"
    **Step 1 · The truncation achieves error $\sigma_{k+1}$.** The leftover is $\mathbf{A}-\mathbf{A}_k = \sum_{i>k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$. That is itself an SVD, whose largest singular value is $\sigma_{k+1}$.

    **Step 2 · No rank-$k$ matrix does better.** Let $\mathbf{B}$ be any matrix with rank $\le k$.

    *Step 2a.* By rank–nullity, its null space has dimension $\ge n-k$.

    *Step 2b.* The span of $\mathbf{v}_1,\dots,\mathbf{v}_{k+1}$ has dimension $k+1$.

    *Step 2c.* Two subspaces of $\R^n$ whose dimensions add up to more than $n$ must share a nonzero vector. (Otherwise, putting their bases together would give more than $n$ independent vectors in $\R^n$, which is impossible.) Here $(n-k) + (k+1) = n+1 > n$.

    *Step 2d.* So pick a unit vector $\mathbf{w}$ in both. Being in the span, $\mathbf{w}=\sum_{i\le k+1}c_i\mathbf{v}_i$ with $\sum c_i^2 = 1$. Being in the null space, $\mathbf{B}\mathbf{w} = \mathbf{0}$.

    *Step 2e.* Then

    $$
    \lVert(\mathbf{A}-\mathbf{B})\mathbf{w}\rVert^2 = \lVert\mathbf{A}\mathbf{w}\rVert^2 = \Big\lVert\sum_{i\le k+1}c_i\sigma_i\mathbf{u}_i\Big\rVert^2 = \sum_{i\le k+1}\sigma_i^2c_i^2 \ge \sigma_{k+1}^2\sum_i c_i^2 = \sigma_{k+1}^2 .
    $$

    (The middle step is Pythagoras: the $\mathbf{u}_i$ are orthonormal. The inequality uses $\sigma_i\ge\sigma_{k+1}$ for $i\le k+1$.)

    *Step 2f.* So $\mathbf{A}-\mathbf{B}$ stretches the unit vector $\mathbf{w}$ by at least $\sigma_{k+1}$, i.e. $\lVert\mathbf{A}-\mathbf{B}\rVert_2\ge\sigma_{k+1}$. $\blacksquare$

!!! llm "LLM connection"
    | Where | How the SVD / low rank appears |
    |---|---|
    | **LoRA** (Ch. 23) | assumes the fine-tuning *update* $\Delta\mathbf{W}$ has low rank; parameterizes it as $\mathbf{B}\mathbf{A}$ with $r\in\{8,16,64\}$ |
    | **Attention heads** | $\mathbf{W}_Q\mathbf{W}_K^\top$ has rank $\le d_h$ |
    | **Multi-head latent attention** (DeepSeek-V2, Ch. 20) | compresses keys/values through a low-rank bottleneck to shrink the KV cache |
    | **Muon optimizer** (Ch. 5) | orthogonalizes updates: sets every singular value of the update to 1 |
    | **Diagnostics** | the singular-value spectrum of weight matrices shows how much of each matrix is "used" |

```python
import torch
A = torch.randn(512, 256) @ torch.randn(256, 512) / 16  # rank ≤ 256
U, S, Vh = torch.linalg.svd(A, full_matrices=False)
for k in (8, 64, 256):
    Ak = U[:, :k] @ torch.diag(S[:k]) @ Vh[:k]
    rel = torch.linalg.norm(A - Ak) / torch.linalg.norm(A)
    print(k, f"relative Frobenius error {rel:.3f}",
          f"theory {torch.sqrt((S[k:]**2).sum() / (S**2).sum()):.3f}")
```

---

## 1.10 Projections and least squares

**The idea:** the closest point to $\mathbf{y}$ on a line or plane is found by dropping a perpendicular.

### Projecting onto a line

We want the point $\mathbf{p} = c\,\mathbf{a}$ on the line through $\mathbf{a}$ that is closest to $\mathbf{y}$.

1. **The leftover must be perpendicular to the line.** (If it weren't, sliding along the line would get closer.) So $\mathbf{a}^\top(\mathbf{y} - c\,\mathbf{a}) = 0$.
2. **Solve for $c$:** $\mathbf{a}^\top\mathbf{y} - c\,\mathbf{a}^\top\mathbf{a} = 0$, so $c = \dfrac{\mathbf{a}^\top\mathbf{y}}{\mathbf{a}^\top\mathbf{a}}$.
3. **Result:** $\mathbf{p} = \dfrac{\mathbf{a}^\top\mathbf{y}}{\mathbf{a}^\top\mathbf{a}}\,\mathbf{a}$. For a unit vector $\mathbf{u}$ this simplifies to $\mathbf{p} = (\mathbf{u}^\top\mathbf{y})\,\mathbf{u}$.

!!! example "Worked example"
    Project $\mathbf{y} = (2, 3)$ onto the line through $\mathbf{a} = (1, 1)$.

    - $c = \dfrac{2+3}{1+1} = 2.5$, so $\mathbf{p} = (2.5, 2.5)$.
    - Leftover: $\mathbf{y} - \mathbf{p} = (-0.5, 0.5)$.
    - **Check** it's perpendicular: $(1,1)\cdot(-0.5, 0.5) = 0$. ✓

### Least squares

Now project onto a whole subspace: the column space of a matrix $\mathbf{X}$. Points in it are $\mathbf{X}\mathbf{w}$ for some weights $\mathbf{w}$. Finding the closest one is the **least-squares** problem:

$$
\hat{\mathbf{w}} = \argmin_{\mathbf{w}}\lVert\mathbf{X}\mathbf{w}-\mathbf{y}\rVert^2 .
$$

"$\argmin$" means "the $\mathbf{w}$ that makes this smallest". This is fitting a linear model to data: rows of $\mathbf{X}$ are examples, $\mathbf{y}$ holds the targets.

!!! theorem "Theorem 1.13 (Normal equations)"
    $\hat{\mathbf{w}}$ satisfies

    $$
    \mathbf{X}^\top\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^\top\mathbf{y},
    $$

    so $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ when $\mathbf{X}$ has independent columns.

???+ proof "Proof (geometric, no calculus)"
    **Step 1 · Guess the condition.** As with the line, the best fit should leave a residual $\mathbf{r} = \mathbf{y}-\mathbf{X}\hat{\mathbf{w}}$ perpendicular to *every* column of $\mathbf{X}$. All those dot products at once: $\mathbf{X}^\top\mathbf{r}=\mathbf{0}$. Suppose $\hat{\mathbf{w}}$ satisfies this.

    **Step 2 · Compare with any other $\mathbf{w}$.** Let $\mathbf{d} = \mathbf{w}-\hat{\mathbf{w}}$. Then

    $$
    \mathbf{X}\mathbf{w}-\mathbf{y} = \mathbf{X}\mathbf{d} - \mathbf{r}.
    $$

    **Step 3 · The two pieces are perpendicular:** $(\mathbf{X}\mathbf{d})^\top\mathbf{r} = \mathbf{d}^\top(\mathbf{X}^\top\mathbf{r}) = 0$.

    **Step 4 · Pythagoras:**

    $$
    \lVert\mathbf{X}\mathbf{w}-\mathbf{y}\rVert^2 = \lVert\mathbf{X}\mathbf{d}\rVert^2 + \lVert\mathbf{r}\rVert^2 \ge \lVert\mathbf{r}\rVert^2 .
    $$

    So no $\mathbf{w}$ beats $\hat{\mathbf{w}}$.

    **Step 5 · Rewrite the condition.** $\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\hat{\mathbf{w}}) = \mathbf{0}$ is $\mathbf{X}^\top\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^\top\mathbf{y}$: the normal equations. $\blacksquare$

!!! example "Worked example: fitting a line $y = wx$"
    Data points $(1, 1)$, $(2, 2)$, $(3, 2)$. One weight, no intercept.

    - $\mathbf{X} = (1, 2, 3)^\top$, $\mathbf{y} = (1, 2, 2)^\top$.
    - $\mathbf{X}^\top\mathbf{X} = 1+4+9 = 14$.
    - $\mathbf{X}^\top\mathbf{y} = 1+4+6 = 11$.
    - $\hat w = 11/14\approx 0.786$.

    **Check** the residual is perpendicular to $\mathbf{X}$: $\mathbf{r} = (1, 2, 2) - 0.786\cdot(1,2,3) = (0.214, 0.429, -0.357)$, and $1(0.214) + 2(0.429) + 3(-0.357) \approx 0$. ✓

Least squares is "a linear model trained in closed form". Neural networks are what you get when the model is nonlinear and no closed form exists, so we need gradient descent (Ch. 5). In Ch. 2 we re-derive the normal equations by setting a gradient to zero.

---

## 1.11 Tensors, batching, broadcasting and einsum

In code, a **tensor** is an array with any number of axes ("dimensions").

| Axes | Name | Example shape | Example |
|---|---|---|---|
| 0 | scalar | `()` | a loss value |
| 1 | vector | `(C,)` | one token's embedding |
| 2 | matrix | `(T, C)` | one sequence of token vectors |
| 3 | 3-D tensor | `(B, T, C)` | a batch of sequences |
| 4 | 4-D tensor | `(B, H, T, T)` | attention scores for every head |

LLM code is mostly about tensors of shape `(B, T, C)`: **B**atch, **T**ime (sequence position), **C**hannels ($=d$).

Three rules make the code below readable:

1. **`@` multiplies the last two axes.** All earlier axes are "batch" axes: the same matmul is done for each.
2. **`einsum` spells out the sum.** Each letter names an axis. A letter that appears in the inputs but not the output is summed over.
3. **Broadcasting** lines shapes up from the right. A missing axis, or an axis of size 1, is copied to match.

```python
import torch
B, T, C, H = 2, 5, 12, 3          # batch, sequence, width, heads
x = torch.randn(B, T, C)
W = torch.randn(C, 4 * C)

y = x @ W                          # (B,T,C) @ (C,4C) -> (B,T,4C): same W for every (b,t)
print(y.shape)

# Batched matmul: the last two dims multiply, leading dims are batch dims
q = torch.randn(B, H, T, C // H)
k = torch.randn(B, H, T, C // H)
scores = q @ k.transpose(-2, -1)   # (B,H,T,d_h) @ (B,H,d_h,T) -> (B,H,T,T)

# einsum states the contraction explicitly: sum over the repeated index d
scores2 = torch.einsum("bhtd,bhsd->bhts", q, k)
print(torch.allclose(scores, scores2))

# Broadcasting: a size-1 (or missing) dim is stretched to match
bias = torch.randn(C)              # (C,) -> treated as (1,1,C)
print((x + bias).shape)            # (B,T,C)
```

!!! warning "Common pitfall"
    - `x.view(B, T, H, d_h)` reinterprets memory. It does **not** move data.
    - After `transpose`, the tensor is non-contiguous in memory.
    - You then need `.contiguous()` (or `reshape`) before `view` again.

    The attention code in Ch. 12 does exactly this dance.

---

## 1.12 High-dimensional geometry: why everything is nearly orthogonal

**The idea:** in thousands of dimensions, two random directions are almost always nearly perpendicular.

Our 2-D and 3-D intuition misleads us in $\R^{4096}$. One fact is fundamental for LLMs. Its proof needs two rules about averages ("expectations") from Ch. 3:

!!! recall "Preview from Ch. 3: two expectation rules"
    - $\E[X]$ is the average value of a random quantity $X$. $\Var[X] = \E[(X-\E X)^2]$ measures its spread.
    - If $X$ and $Y$ are **independent**, $\E[XY] = \E[X]\,\E[Y]$.
    - If $X_1,\dots,X_d$ are independent, $\Var[\sum_i X_i] = \sum_i\Var[X_i]$.
    - A standard normal $\mathcal{N}(0,1)$ number has mean 0 and variance 1, so $\E[x^2] = 1$.

!!! theorem "Proposition 1.14 (Random vectors are nearly orthogonal)"
    Let $\mathbf{x},\mathbf{y}\in\R^d$ have independent $\mathcal{N}(0,1)$ entries. Then:

    | Quantity | Value |
    |---|---|
    | $\E[\mathbf{x}^\top\mathbf{y}]$ | $0$ |
    | $\Var[\mathbf{x}^\top\mathbf{y}]$ | $d$ |
    | $\E\lVert\mathbf{x}\rVert^2$ | $d$ |

    Hence $\cos(\mathbf{x},\mathbf{y})$ is typically of size $1/\sqrt{d}$.

???+ proof "Proof"
    **Step 1 · Mean of the dot product.** $\mathbf{x}^\top\mathbf{y} = \sum_i x_iy_i$. Each term has $\E[x_iy_i] = \E[x_i]\E[y_i] = 0\cdot0 = 0$ (independence). So the mean is 0.

    **Step 2 · Variance of the dot product.** The terms $x_iy_i$ are independent of each other, so their variances add. Each term has mean 0, so its variance is

    $$
    \E[(x_iy_i)^2] = \E[x_i^2]\,\E[y_i^2] = 1\cdot1 = 1 .
    $$

    Adding $d$ of them: $\Var[\mathbf{x}^\top\mathbf{y}] = d$. So the dot product is typically about $\pm\sqrt d$ (one standard deviation).

    **Step 3 · Length.** $\E\lVert\mathbf{x}\rVert^2 = \sum_i\E[x_i^2] = d$. By the law of large numbers (Ch. 3), the average $\frac1d\sum x_i^2$ concentrates near 1, so $\lVert\mathbf{x}\rVert\approx\sqrt d$. Same for $\mathbf{y}$.

    **Step 4 · Put together:**

    $$
    \cos(\mathbf{x},\mathbf{y}) = \frac{\mathbf{x}^\top\mathbf{y}}{\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert}\approx\frac{\pm\sqrt d}{\sqrt d\cdot\sqrt d} = \pm\frac{1}{\sqrt d}. \qquad\blacksquare
    $$

| Dimension $d$ | Typical $\lvert\cos\rvert\approx 1/\sqrt d$ | Typical angle |
|---|---|---|
| 3 | 0.58 | anywhere |
| 64 | 0.125 | $\approx 83^\circ$ |
| 768 (GPT-2) | 0.036 | $\approx 88^\circ$ |
| 4096 | 0.016 | $\approx 89^\circ$ |

You can even fit **exponentially many** nearly-orthogonal directions in high dimension (the Johnson–Lindenstrauss lemma). Two big consequences:

1. **The $\sqrt{d_h}$ in attention.** The dot product $\mathbf{q}^\top\mathbf{k}$ has variance growing like $d_h$ (Step 2). Dividing by $\sqrt{d_h}$ keeps scores around size 1 (Ch. 12; widget there).
2. **Superposition.** Networks can represent far more "features" than they have dimensions by using nearly-orthogonal directions, at the cost of a little interference. This is central to interpretability (Ch. 24; Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html)).

```python
import torch
for d in (3, 64, 768, 4096):
    x, y = torch.randn(10000, d), torch.randn(10000, d)
    cos = torch.nn.functional.cosine_similarity(x, y, dim=1)
    print(f"d={d:5d}  mean|cos|={cos.abs().mean():.4f}   1/sqrt(d)={d**-0.5:.4f}")
```

---

## Key takeaways

| Concept | One-line summary | Where it matters |
|---|---|---|
| Dot product | $\mathbf{x}^\top\mathbf{y} = \lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert\cos\theta$ measures alignment | attention scores, embedding similarity |
| Cauchy–Schwarz | $\lvert\mathbf{x}^\top\mathbf{y}\rvert\le\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert$, so cosines lie in $[-1,1]$ | makes "angle" meaningful in any dimension |
| Matrices | column $j$ is where $\mathbf{e}_j$ goes; multiplication is composition | every layer |
| Matmul cost | $(m\times k)(k\times n)$ costs $2mnk$ FLOPs | the $6ND$ compute rule |
| Rank | number of independent output directions; rank-$r$ storage is $r(d+k)$ | LoRA, attention heads |
| Determinant | volume scale factor; $0$ means not invertible | theory, invertibility |
| Orthogonal / rotation | preserves lengths and angles; $\mathbf{R}(\alpha)\mathbf{R}(\beta) = \mathbf{R}(\alpha+\beta)$ | RoPE |
| Spectral theorem | symmetric $\mathbf{A} = \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$ | Hessians, conditioning |
| SVD | every $\mathbf{A} = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top$; truncation is the best low-rank approximation | LoRA, compression, Muon |
| Least squares | residual $\perp$ columns: $\mathbf{X}^\top\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^\top\mathbf{y}$ | linear models |
| High dimensions | random cosines are $\sim1/\sqrt d$ | $\sqrt{d_h}$ scaling, superposition |

## Exercises

1. Show that $\lVert\mathbf{x}+\mathbf{y}\rVert^2 + \lVert\mathbf{x}-\mathbf{y}\rVert^2 = 2\lVert\mathbf{x}\rVert^2 + 2\lVert\mathbf{y}\rVert^2$ (the parallelogram law).
2. Prove that $\tr(\mathbf{AB}) = \tr(\mathbf{BA})$ whenever both products are defined. (The **trace** $\tr$ is the sum of the diagonal entries.) Use it to show $\tr(\mathbf{A}) = \sum_i\lambda_i$ for symmetric $\mathbf{A}$.
3. A linear layer has $d_\text{in}=4096$, $d_\text{out}=11008$. How many FLOPs does a forward pass over a batch of $B=8$ sequences of length $T=2048$ take?
4. Show that $\lVert\mathbf{A}\rVert_F^2 = \sum_i\sigma_i^2$.
5. Prove that the eigenvalues of an orthogonal matrix all have absolute value 1.
6. **Code:** verify Theorem 1.12 numerically: for a random $100\times80$ matrix, compare the Frobenius error of the rank-10 SVD truncation against 1000 random rank-10 matrices $\mathbf{B}\mathbf{C}$ fitted by least squares.

??? success "Solutions"
    1. Expand both squares. The cross terms $+2\mathbf{x}^\top\mathbf{y}$ and $-2\mathbf{x}^\top\mathbf{y}$ cancel, leaving $2\lVert\mathbf{x}\rVert^2 + 2\lVert\mathbf{y}\rVert^2$.
    2. $\tr(\mathbf{AB}) = \sum_i(\mathbf{AB})_{ii} = \sum_i\sum_k A_{ik}B_{ki}$. Swap the order of the sums and the two factors: $\sum_k\sum_i B_{ki}A_{ik} = \sum_k(\mathbf{BA})_{kk} = \tr(\mathbf{BA})$. Then $\tr(\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top) = \tr(\boldsymbol\Lambda\mathbf{Q}^\top\mathbf{Q}) = \tr\boldsymbol\Lambda = \sum_i\lambda_i$.
    3. $BT = 16384$ tokens. FLOPs $= 2\cdot BT\cdot d_\text{in}d_\text{out} = 2\cdot 16384\cdot 4096\cdot 11008 \approx 1.48\times10^{12}$.
    4. $\lVert\mathbf{A}\rVert_F^2 = \sum_{ij}A_{ij}^2 = \tr(\mathbf{A}^\top\mathbf{A})$. Substitute the SVD: $\tr(\mathbf{V}\boldsymbol\Sigma^\top\mathbf{U}^\top\mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top) = \tr(\mathbf{V}\boldsymbol\Sigma^\top\boldsymbol\Sigma\mathbf{V}^\top) = \tr(\boldsymbol\Sigma^\top\boldsymbol\Sigma\mathbf{V}^\top\mathbf{V}) = \tr(\boldsymbol\Sigma^\top\boldsymbol\Sigma) = \sum\sigma_i^2$.
    5. If $\mathbf{Q}\mathbf{v} = \lambda\mathbf{v}$, then $\lVert\mathbf{v}\rVert = \lVert\mathbf{Q}\mathbf{v}\rVert = \lvert\lambda\rvert\lVert\mathbf{v}\rVert$. (Theorem 1.9 extends to complex vectors using the conjugate transpose.) Divide by $\lVert\mathbf{v}\rVert\neq0$: $\lvert\lambda\rvert = 1$.
    6. Every random attempt has error $\ge\sqrt{\sum_{i>10}\sigma_i^2}$, and the SVD truncation hits that bound exactly.

## References

- G. Strang, *Introduction to Linear Algebra*, 5th ed.; [MIT 18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/).
- M. P. Deisenroth, A. A. Faisal, C. S. Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io/), Ch. 2–4 (2020).
- S. Axler, [*Linear Algebra Done Right*](https://linear.axler.net/), 4th ed. (free PDF), for the proof-oriented view.
- K. B. Petersen, M. S. Pedersen, [*The Matrix Cookbook*](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
- C. Eckart, G. Young, "The approximation of one matrix by another of lower rank," *Psychometrika* (1936).
- N. Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) (Anthropic, 2022).
