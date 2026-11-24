# 1. Linear Algebra

!!! abstract "In this chapter"
    **Prerequisites:** high-school algebra. **You will learn:** vectors, dot products, norms, matrices as linear maps, matrix multiplication as composition, rank, inverses, determinants, orthogonal matrices, eigen-decomposition, the SVD and low-rank approximation, projections and least squares, tensors and `einsum`, and the strange geometry of high-dimensional spaces.
    **Why it matters:** an LLM is, to a first approximation, *a long sequence of matrix multiplications*. Embeddings are vectors, attention scores are dot products, LoRA is a low-rank matrix, RoPE is a rotation matrix, and the cost of everything is measured in matmul FLOPs.

!!! watch "Watch first (highly recommended)"
    3Blue1Brown, *Essence of Linear Algebra*. Its 16 short videos build exactly the geometric intuition this chapter formalizes.

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/videoseries?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab" allowfullscreen loading="lazy"></iframe></div>

    For rigor: Gilbert Strang, [MIT 18.06 Linear Algebra](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) and [MIT 18.065 Matrix Methods in Data Analysis and Machine Learning](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/). Textbook: Deisenroth, Faisal & Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io/), Ch. 2–4.

---

## 1.1 Vectors

!!! definition "Definition 1.1 (Vector)"
    A **vector** $\mathbf{x} \in \R^n$ is an ordered list of $n$ real numbers, written as a column:

    $$
    \mathbf{x} = \begin{pmatrix} x_1 \\ \vdots \\ x_n \end{pmatrix}, \qquad \mathbf{x}^\top = (x_1, \dots, x_n).
    $$

    Vectors are added componentwise, $(\mathbf{x}+\mathbf{y})_i = x_i + y_i$, and scaled componentwise, $(c\mathbf{x})_i = c\,x_i$.

There are two pictures, and you need both:

- **Geometric:** an arrow from the origin to the point $(x_1,\dots,x_n)$. Addition is "tip to tail", and scaling stretches the arrow.
- **Data:** a list of features. A token's embedding in GPT-2 is a vector in $\R^{768}$. No single coordinate has to mean anything; what matters is the **directions** and the **distances**.

A **linear combination** of vectors $\mathbf{v}_1,\dots,\mathbf{v}_k$ is $c_1\mathbf{v}_1 + \dots + c_k\mathbf{v}_k$. Their **span** is the set of all such combinations.

!!! llm "LLM connection"
    A Transformer's **residual stream** (Ch. 13) is a vector $\mathbf{h}_t\in\R^d$ at each position. Every layer reads it and *adds* a vector to it. The final vector is compared with every token embedding via dot products to produce the next-token logits. Language modeling is literally vector arithmetic.

## 1.2 The dot product, angles and similarity

!!! definition "Definition 1.2 (Dot product, norm)"
    $$
    \mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{n} x_i y_i, \qquad \lVert \mathbf{x} \rVert = \sqrt{\mathbf{x}^\top \mathbf{x}} = \sqrt{\textstyle\sum_i x_i^2}.
    $$

The dot product is symmetric ($\mathbf{x}^\top\mathbf{y} = \mathbf{y}^\top\mathbf{x}$) and linear in each argument: $(a\mathbf{x}+b\mathbf{z})^\top\mathbf{y} = a\,\mathbf{x}^\top\mathbf{y} + b\,\mathbf{z}^\top\mathbf{y}$.

!!! theorem "Theorem 1.1 (Geometric meaning of the dot product)"
    If $\theta$ is the angle between nonzero $\mathbf{x}$ and $\mathbf{y}$, then

    $$
    \mathbf{x}^\top\mathbf{y} = \lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert\cos\theta .
    $$

??? proof "Proof"
    Expand the squared length of $\mathbf{x}-\mathbf{y}$ using linearity:

    $$
    \lVert \mathbf{x}-\mathbf{y}\rVert^2 = (\mathbf{x}-\mathbf{y})^\top(\mathbf{x}-\mathbf{y}) = \lVert\mathbf{x}\rVert^2 - 2\,\mathbf{x}^\top\mathbf{y} + \lVert\mathbf{y}\rVert^2 .
    $$

    The vectors $\mathbf{x}$, $\mathbf{y}$, $\mathbf{x}-\mathbf{y}$ form a triangle, and the angle between the sides $\mathbf{x}$ and $\mathbf{y}$ is $\theta$. The law of cosines gives

    $$
    \lVert \mathbf{x}-\mathbf{y}\rVert^2 = \lVert\mathbf{x}\rVert^2 + \lVert\mathbf{y}\rVert^2 - 2\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert\cos\theta .
    $$

    Equating the two expressions and cancelling gives the result. $\blacksquare$

So the dot product measures **alignment**: it is positive when the vectors point the same way, zero when they are **orthogonal** ($\theta = 90^\circ$), and negative when they point opposite ways. Dividing out the lengths gives the **cosine similarity**

$$
\cos(\mathbf{x},\mathbf{y}) = \frac{\mathbf{x}^\top\mathbf{y}}{\lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert} \in [-1, 1],
$$

which is the standard similarity measure for text embeddings (Ch. 26). That $\cos$ lands in $[-1,1]$ for *any* dimension $n$ is not obvious from the formula. It is the content of:

!!! theorem "Theorem 1.2 (Cauchy–Schwarz inequality)"
    $\lvert \mathbf{x}^\top\mathbf{y}\rvert \le \lVert\mathbf{x}\rVert\,\lVert\mathbf{y}\rVert$, with equality iff one vector is a scalar multiple of the other.

??? proof "Proof"
    If $\mathbf{y}=\mathbf{0}$, both sides are $0$. Otherwise, for every real $t$,

    $$
    0 \le \lVert\mathbf{x} - t\mathbf{y}\rVert^2 = \lVert\mathbf{x}\rVert^2 - 2t\,\mathbf{x}^\top\mathbf{y} + t^2\lVert\mathbf{y}\rVert^2 .
    $$

    This quadratic in $t$ is minimized at $t^\star = \mathbf{x}^\top\mathbf{y}/\lVert\mathbf{y}\rVert^2$. Substituting,

    $$
    0 \le \lVert\mathbf{x}\rVert^2 - \frac{(\mathbf{x}^\top\mathbf{y})^2}{\lVert\mathbf{y}\rVert^2}
    \;\Longrightarrow\; (\mathbf{x}^\top\mathbf{y})^2 \le \lVert\mathbf{x}\rVert^2\lVert\mathbf{y}\rVert^2 .
    $$

    Equality holds iff $\lVert\mathbf{x}-t^\star\mathbf{y}\rVert = 0$, i.e. $\mathbf{x} = t^\star\mathbf{y}$. $\blacksquare$

!!! theorem "Corollary 1.3 (Triangle inequality)"
    $\lVert\mathbf{x}+\mathbf{y}\rVert \le \lVert\mathbf{x}\rVert + \lVert\mathbf{y}\rVert$.

??? proof "Proof"
    $\lVert\mathbf{x}+\mathbf{y}\rVert^2 = \lVert\mathbf{x}\rVert^2 + 2\mathbf{x}^\top\mathbf{y} + \lVert\mathbf{y}\rVert^2 \le \lVert\mathbf{x}\rVert^2 + 2\lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert + \lVert\mathbf{y}\rVert^2 = (\lVert\mathbf{x}\rVert+\lVert\mathbf{y}\rVert)^2$, using Cauchy–Schwarz in the middle step. $\blacksquare$

**Other norms.** $\lVert\mathbf{x}\rVert_1 = \sum_i\lvert x_i\rvert$, $\lVert\mathbf{x}\rVert_\infty = \max_i\lvert x_i\rvert$, and in general $\lVert\mathbf{x}\rVert_p = (\sum_i \lvert x_i\rvert^p)^{1/p}$. Gradient clipping (Ch. 8) uses the $\ell_2$ norm of *all* gradients concatenated, and the absmax quantization in Ch. 19 uses $\ell_\infty$.

!!! llm "LLM connection: attention is a dot product"
    In attention (Ch. 12), a query $\mathbf{q}$ "looks for" keys $\mathbf{k}_j$ that point in a similar direction. The score is $\mathbf{q}^\top\mathbf{k}_j/\sqrt{d_h}$, a dot product: large when aligned, small when orthogonal.

## 1.3 Matrices are linear maps

!!! definition "Definition 1.3 (Linear map)"
    A function $f:\R^n\to\R^m$ is **linear** if $f(a\mathbf{x}+b\mathbf{y}) = a f(\mathbf{x}) + b f(\mathbf{y})$ for all vectors $\mathbf{x},\mathbf{y}$ and scalars $a,b$.

!!! theorem "Theorem 1.4 (Every linear map is a matrix)"
    Let $\mathbf{e}_j$ be the $j$-th standard basis vector (1 in slot $j$, 0 elsewhere). If $f$ is linear, then $f(\mathbf{x}) = \mathbf{W}\mathbf{x}$, where $\mathbf{W}\in\R^{m\times n}$ is the matrix whose **$j$-th column is $f(\mathbf{e}_j)$**.

??? proof "Proof"
    Any $\mathbf{x} = \sum_j x_j\mathbf{e}_j$. By linearity, $f(\mathbf{x}) = \sum_j x_j f(\mathbf{e}_j)$. That is a linear combination of the columns $f(\mathbf{e}_j)$ with coefficients $x_j$, which is precisely the definition of the matrix–vector product $\mathbf{W}\mathbf{x}$ (below). $\blacksquare$

The matrix–vector product has two equivalent readings, and both are used constantly:

$$
\underbrace{\mathbf{W}\mathbf{x} = \sum_{j=1}^n x_j\,\mathbf{w}_{:,j}}_{\text{combination of columns}}
\qquad\Longleftrightarrow\qquad
\underbrace{(\mathbf{W}\mathbf{x})_i = \mathbf{w}_{i,:}\,\mathbf{x} = \sum_{j} W_{ij}x_j}_{\text{dot product with each row}} .
$$

Play with it: the orange and green arrows are the images of $\mathbf{e}_1,\mathbf{e}_2$, i.e. the columns of $\mathbf{W}$, and the whole grid follows along.

<div class="widget" data-widget="linear-transform"></div>

!!! intuition "Intuition"
    A neural-network layer $\mathbf{y} = \mathbf{W}\mathbf{x}+\mathbf{b}$ **rotates, stretches, shears and projects** space, then shifts it. On its own this can only produce linear "warps". The nonlinearity that follows (Ch. 6) is what lets stacked layers bend space into arbitrary shapes.

## 1.4 Matrix multiplication is composition

!!! definition "Definition 1.4 (Matrix product)"
    For $\mathbf{A}\in\R^{m\times k}$ and $\mathbf{B}\in\R^{k\times n}$, $\mathbf{AB}\in\R^{m\times n}$ has entries $(\mathbf{AB})_{ij} = \sum_{r=1}^{k} A_{ir}B_{rj}$.

!!! theorem "Theorem 1.5"
    $(\mathbf{AB})\mathbf{x} = \mathbf{A}(\mathbf{B}\mathbf{x})$: the product matrix is the linear map "apply $\mathbf{B}$, then apply $\mathbf{A}$". Consequently matrix multiplication is **associative**, $(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$, but in general **not commutative**, $\mathbf{AB}\ne\mathbf{BA}$.

??? proof "Proof"
    $\big(\mathbf{A}(\mathbf{B}\mathbf{x})\big)_i = \sum_r A_{ir}(\mathbf{B}\mathbf{x})_r = \sum_r A_{ir}\sum_j B_{rj}x_j = \sum_j\big(\sum_r A_{ir}B_{rj}\big)x_j = \big((\mathbf{AB})\mathbf{x}\big)_i$. Associativity follows because composition of functions is associative. For non-commutativity, take $\mathbf{A}=\begin{psmallmatrix}0&1\\0&0\end{psmallmatrix}$ and $\mathbf{B}=\begin{psmallmatrix}0&0\\1&0\end{psmallmatrix}$: $\mathbf{AB}=\begin{psmallmatrix}1&0\\0&0\end{psmallmatrix}\neq\mathbf{BA}=\begin{psmallmatrix}0&0\\0&1\end{psmallmatrix}$. $\blacksquare$

!!! theorem "Fact 1.6 (Cost of a matmul)"
    Computing $\mathbf{AB}$ with $\mathbf{A}\in\R^{m\times k}$, $\mathbf{B}\in\R^{k\times n}$ takes $mn$ dot products of length $k$: $mnk$ multiplies and about $mnk$ adds, so **$\approx 2mnk$ FLOPs**.

This one fact underlies all of LLM cost accounting. A linear layer with weight $d_\text{in}\times d_\text{out}$ applied to one token costs $2\,d_\text{in}d_\text{out}$ FLOPs, which is **2 FLOPs per parameter per token** in the forward pass. In Ch. 15 this becomes the famous $C \approx 6ND$.

!!! warning "Associativity is a performance tool"
    $(\mathbf{AB})\mathbf{x}$ and $\mathbf{A}(\mathbf{B}\mathbf{x})$ give the same answer but can differ hugely in cost. With $\mathbf{A}\in\R^{d\times r}$, $\mathbf{B}\in\R^{r\times d}$ and $r \ll d$, forming $\mathbf{AB}$ costs $2d^2r$, while $\mathbf{A}(\mathbf{B}\mathbf{x})$ costs $4dr$. LoRA (Ch. 23) relies on exactly this.

**Transpose.** $(\mathbf{A}^\top)_{ij} = A_{ji}$. Two rules you'll use everywhere:

!!! theorem "Theorem 1.7"
    $(\mathbf{AB})^\top = \mathbf{B}^\top\mathbf{A}^\top$ and $\mathbf{x}^\top\mathbf{A}\mathbf{y} = (\mathbf{A}^\top\mathbf{x})^\top\mathbf{y}$.

??? proof "Proof"
    $\big((\mathbf{AB})^\top\big)_{ij} = (\mathbf{AB})_{ji} = \sum_r A_{jr}B_{ri} = \sum_r (\mathbf{B}^\top)_{ir}(\mathbf{A}^\top)_{rj} = (\mathbf{B}^\top\mathbf{A}^\top)_{ij}$. The second rule is the first applied to $\mathbf{x}^\top\mathbf{A}$: since $\mathbf{x}^\top\mathbf{A} = (\mathbf{A}^\top\mathbf{x})^\top$, we get $\mathbf{x}^\top\mathbf{A}\mathbf{y} = (\mathbf{A}^\top\mathbf{x})^\top\mathbf{y}$. $\blacksquare$

!!! llm "LLM connection: the attention bilinear form"
    The attention score between positions $i$ and $j$ is $\mathbf{q}_i^\top\mathbf{k}_j = (\mathbf{W}_Q^\top\mathbf{x}_i)^\top(\mathbf{W}_K^\top\mathbf{x}_j) = \mathbf{x}_i^\top\,\mathbf{W}_Q\mathbf{W}_K^\top\,\mathbf{x}_j$. The product $\mathbf{W}_Q\mathbf{W}_K^\top\in\R^{d\times d}$ has rank at most $d_h$. Interpretability research (Ch. 24) calls it the **QK circuit**.

## 1.5 Span, independence, basis, rank

!!! definition "Definition 1.5"
    Vectors $\mathbf{v}_1,\dots,\mathbf{v}_k$ are **linearly independent** if $\sum_i c_i\mathbf{v}_i = \mathbf{0}$ implies all $c_i=0$. A **basis** of a subspace is an independent set that spans it, and the number of vectors in any basis is its **dimension**. The **column space** of $\mathbf{W}$ is the span of its columns, and the **rank** of $\mathbf{W}$ is the dimension of its column space. The **null space** is $\{\mathbf{x} : \mathbf{W}\mathbf{x} = \mathbf{0}\}$.

!!! theorem "Theorem 1.8 (Rank–nullity)"
    For $\mathbf{W}\in\R^{m\times n}$: $\;\rank(\mathbf{W}) + \dim\operatorname{null}(\mathbf{W}) = n$.

??? proof "Proof sketch"
    Pick a basis $\mathbf{u}_1,\dots,\mathbf{u}_k$ of the null space and extend it to a basis $\mathbf{u}_1,\dots,\mathbf{u}_k,\mathbf{w}_1,\dots,\mathbf{w}_{r}$ of $\R^n$ (so $k + r = n$). The images $\mathbf{W}\mathbf{w}_1,\dots,\mathbf{W}\mathbf{w}_r$ span the column space, because the $\mathbf{u}$'s map to $\mathbf{0}$. They are also independent: if $\sum c_i\mathbf{W}\mathbf{w}_i = \mathbf{0}$, then $\sum c_i\mathbf{w}_i$ lies in the null space, so it is a combination of the $\mathbf{u}$'s, and independence of the full basis forces all $c_i = 0$. Hence $\rank = r = n - k$. $\blacksquare$

Useful consequences: $\rank(\mathbf{AB}) \le \min(\rank\mathbf{A},\rank\mathbf{B})$, and a product $\mathbf{B}\mathbf{A}$ with $\mathbf{B}\in\R^{d\times r}$, $\mathbf{A}\in\R^{r\times k}$ has rank at most $r$. A **low-rank** matrix stores $r(d+k)$ numbers instead of $dk$. That is the whole idea of LoRA and of low-rank compression.

<div class="widget" data-widget="lora"></div>

## 1.6 Inverses and determinants

A square $\mathbf{W}\in\R^{n\times n}$ is **invertible** if there is a $\mathbf{W}^{-1}$ with $\mathbf{W}^{-1}\mathbf{W} = \mathbf{W}\mathbf{W}^{-1} = \mathbf{I}$. The following are equivalent (the *invertible matrix theorem*): $\mathbf{W}$ is invertible $\Leftrightarrow$ $\rank\mathbf{W}=n$ $\Leftrightarrow$ $\operatorname{null}\mathbf{W}=\{\mathbf{0}\}$ $\Leftrightarrow$ $\det\mathbf{W}\neq 0$ $\Leftrightarrow$ $0$ is not an eigenvalue.

The **determinant** is the factor by which $\mathbf{W}$ scales (signed) volume. For $2\times2$, $\det\begin{psmallmatrix}a&b\\c&d\end{psmallmatrix} = ad-bc$, the signed area of the parallelogram spanned by the columns (watch the shaded square in the widget above). Since volumes scale multiplicatively under composition, $\det(\mathbf{AB}) = \det\mathbf{A}\,\det\mathbf{B}$. A zero determinant means space is squashed into a lower dimension, so information is lost and the map can't be undone.

!!! watch "Watch"
    3Blue1Brown: [Linear transformations and matrices](https://www.youtube.com/watch?v=kYB8IZa5AuE) · [Matrix multiplication as composition](https://www.youtube.com/watch?v=XkY2DOUCWMU) · [The determinant](https://www.youtube.com/watch?v=Ip3X9LOh2dk) · [Dot products and duality](https://www.youtube.com/watch?v=LyGKycYT2v0)

## 1.7 Orthogonal matrices: rotations

!!! definition "Definition 1.6"
    $\mathbf{Q}\in\R^{n\times n}$ is **orthogonal** if $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$, i.e. its columns are orthonormal (unit length and mutually orthogonal). Then $\mathbf{Q}^{-1} = \mathbf{Q}^\top$.

!!! theorem "Theorem 1.9"
    Orthogonal maps preserve dot products, and therefore lengths and angles: $(\mathbf{Q}\mathbf{x})^\top(\mathbf{Q}\mathbf{y}) = \mathbf{x}^\top\mathbf{y}$.

??? proof "Proof"
    $(\mathbf{Q}\mathbf{x})^\top(\mathbf{Q}\mathbf{y}) = \mathbf{x}^\top\mathbf{Q}^\top\mathbf{Q}\mathbf{y} = \mathbf{x}^\top\mathbf{I}\mathbf{y} = \mathbf{x}^\top\mathbf{y}$, using Theorem 1.7. $\blacksquare$

The 2-D rotation by angle $\phi$ is

$$
\mathbf{R}(\phi) = \begin{pmatrix}\cos\phi & -\sin\phi\\ \sin\phi & \cos\phi\end{pmatrix}, \qquad \mathbf{R}(\alpha)\mathbf{R}(\beta) = \mathbf{R}(\alpha+\beta),\qquad \mathbf{R}(\phi)^\top = \mathbf{R}(-\phi).
$$

(The composition rule is the angle-addition formulas for $\sin$ and $\cos$.)

!!! llm "LLM connection: RoPE"
    Rotary position embeddings (Ch. 11) rotate the query at position $m$ by $\mathbf{R}(m\theta)$ and the key at position $n$ by $\mathbf{R}(n\theta)$. Then $(\mathbf{R}(m\theta)\mathbf{q})^\top(\mathbf{R}(n\theta)\mathbf{k}) = \mathbf{q}^\top\mathbf{R}(m\theta)^\top\mathbf{R}(n\theta)\mathbf{k} = \mathbf{q}^\top\mathbf{R}((n-m)\theta)\mathbf{k}$, which depends only on the **relative** position $n-m$. Two lines of algebra from this section are behind one of the most important architectural ideas in modern LLMs.

## 1.8 Eigenvalues, eigenvectors, and the spectral theorem

!!! definition "Definition 1.7"
    A nonzero $\mathbf{v}$ is an **eigenvector** of square $\mathbf{A}$ with **eigenvalue** $\lambda$ if $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$: along $\mathbf{v}$, the map is just scaling. Eigenvalues are the roots of $\det(\mathbf{A}-\lambda\mathbf{I}) = 0$.

In the widget above, the dashed lines are the eigenvector directions. A rotation has no real eigenvectors, since every direction moves; its eigenvalues are complex.

**Symmetric matrices** ($\mathbf{A} = \mathbf{A}^\top$) are the most important case in ML: covariance matrices, Hessians, and Gram matrices $\mathbf{X}^\top\mathbf{X}$ are all symmetric.

!!! theorem "Theorem 1.10 (Spectral theorem)"
    A real symmetric $\mathbf{A}\in\R^{n\times n}$ has real eigenvalues and an orthonormal basis of eigenvectors. Equivalently, $\mathbf{A} = \mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top$ with $\mathbf{Q}$ orthogonal and $\boldsymbol\Lambda$ diagonal.

??? proof "Proof (the two key parts)"
    **Eigenvalues are real.** Let $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ with possibly complex $\mathbf{v},\lambda$, and let $\bar{\mathbf{v}}$ denote the complex conjugate. The scalar $s = \bar{\mathbf{v}}^\top\mathbf{A}\mathbf{v}$ equals its own conjugate: $\bar s = \mathbf{v}^\top\mathbf{A}\bar{\mathbf{v}} = (\mathbf{v}^\top\mathbf{A}\bar{\mathbf{v}})^\top = \bar{\mathbf{v}}^\top\mathbf{A}^\top\mathbf{v} = \bar{\mathbf{v}}^\top\mathbf{A}\mathbf{v} = s$, using that $\mathbf{A}$ is real and symmetric. So $s$ is real. Also $s = \lambda\,\bar{\mathbf{v}}^\top\mathbf{v} = \lambda\sum_i\lvert v_i\rvert^2$, and $\sum_i\lvert v_i\rvert^2 > 0$ is real, so $\lambda$ is real.

    **Eigenvectors of distinct eigenvalues are orthogonal.** If $\mathbf{A}\mathbf{v}_1=\lambda_1\mathbf{v}_1$ and $\mathbf{A}\mathbf{v}_2=\lambda_2\mathbf{v}_2$ with $\lambda_1\ne\lambda_2$, then $\lambda_1\mathbf{v}_1^\top\mathbf{v}_2 = (\mathbf{A}\mathbf{v}_1)^\top\mathbf{v}_2 = \mathbf{v}_1^\top\mathbf{A}\mathbf{v}_2 = \lambda_2\mathbf{v}_1^\top\mathbf{v}_2$, so $(\lambda_1-\lambda_2)\mathbf{v}_1^\top\mathbf{v}_2 = 0$ and therefore $\mathbf{v}_1^\top\mathbf{v}_2 = 0$.

    **Full basis.** By induction on $n$: take one unit eigenvector $\mathbf{q}_1$. The subspace $\mathbf{q}_1^\perp$ is mapped into itself by $\mathbf{A}$ (if $\mathbf{q}_1^\top\mathbf{x}=0$ then $\mathbf{q}_1^\top\mathbf{A}\mathbf{x} = (\mathbf{A}\mathbf{q}_1)^\top\mathbf{x} = \lambda_1\mathbf{q}_1^\top\mathbf{x} = 0$), and $\mathbf{A}$ restricted to it is still symmetric. Apply the induction hypothesis there. $\blacksquare$

!!! definition "Definition 1.8 (Positive (semi)definite)"
    Symmetric $\mathbf{A}$ is **positive semidefinite** (PSD, $\mathbf{A}\succeq 0$) if $\mathbf{x}^\top\mathbf{A}\mathbf{x}\ge 0$ for all $\mathbf{x}$, and **positive definite** ($\succ 0$) if the inequality is strict for $\mathbf{x}\neq\mathbf{0}$. By the spectral theorem, $\mathbf{x}^\top\mathbf{A}\mathbf{x} = \sum_i\lambda_i(\mathbf{q}_i^\top\mathbf{x})^2$, so PSD $\Leftrightarrow$ all $\lambda_i\ge 0$.

Every Gram matrix is PSD: $\mathbf{x}^\top(\mathbf{X}^\top\mathbf{X})\mathbf{x} = \lVert\mathbf{X}\mathbf{x}\rVert^2 \ge 0$. In Ch. 5 the eigenvalues of the loss Hessian will decide how fast gradient descent converges. The ratio $\lambda_\max/\lambda_\min$ (the **condition number**) is the enemy.

!!! watch "Watch"
    3Blue1Brown: [Eigenvectors and eigenvalues](https://www.youtube.com/watch?v=PFDu9oVAE-g) · [Change of basis](https://www.youtube.com/watch?v=P2LTAUO1TdA)

## 1.9 The singular value decomposition (SVD)

Eigen-decomposition needs a square (and, to be nice, symmetric) matrix. The SVD works for **every** matrix and is arguably the most useful theorem in applied linear algebra.

!!! theorem "Theorem 1.11 (SVD)"
    Any $\mathbf{A}\in\R^{m\times n}$ of rank $r$ can be written

    $$
    \mathbf{A} = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top = \sum_{i=1}^{r}\sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top,
    $$

    with $\mathbf{U}\in\R^{m\times m}$ and $\mathbf{V}\in\R^{n\times n}$ orthogonal, and $\boldsymbol\Sigma\in\R^{m\times n}$ diagonal with **singular values** $\sigma_1\ge\sigma_2\ge\dots\ge\sigma_r>0$ (and zeros after).

??? proof "Proof (constructive)"
    $\mathbf{A}^\top\mathbf{A}\in\R^{n\times n}$ is symmetric PSD, so by the spectral theorem it has orthonormal eigenvectors $\mathbf{v}_1,\dots,\mathbf{v}_n$ with eigenvalues $\lambda_1\ge\dots\ge\lambda_n\ge0$. Exactly $r$ of them are positive, because $\rank(\mathbf{A}^\top\mathbf{A}) = \rank(\mathbf{A})$ (they share a null space: $\mathbf{A}^\top\mathbf{A}\mathbf{x}=\mathbf{0}\Rightarrow\lVert\mathbf{A}\mathbf{x}\rVert^2 = \mathbf{x}^\top\mathbf{A}^\top\mathbf{A}\mathbf{x}=0$). Define $\sigma_i = \sqrt{\lambda_i}$ and, for $i\le r$, $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i/\sigma_i$. These are orthonormal:

    $$
    \mathbf{u}_i^\top\mathbf{u}_j = \frac{\mathbf{v}_i^\top\mathbf{A}^\top\mathbf{A}\mathbf{v}_j}{\sigma_i\sigma_j} = \frac{\lambda_j\,\mathbf{v}_i^\top\mathbf{v}_j}{\sigma_i\sigma_j} = \delta_{ij}.
    $$

    Extend $\{\mathbf{u}_i\}$ to an orthonormal basis of $\R^m$. By construction, $\mathbf{A}\mathbf{v}_i = \sigma_i\mathbf{u}_i$ for $i\le r$ and $\mathbf{A}\mathbf{v}_i = \mathbf{0}$ for $i>r$, i.e. $\mathbf{A}\mathbf{V} = \mathbf{U}\boldsymbol\Sigma$. Multiply on the right by $\mathbf{V}^\top$. $\blacksquare$

**Geometry:** every linear map is a *rotation* ($\mathbf{V}^\top$), then an axis-aligned *stretch* ($\boldsymbol\Sigma$), then another *rotation* ($\mathbf{U}$). The singular values are the stretch factors, and $\sigma_1 = \max_{\lVert\mathbf{x}\rVert=1}\lVert\mathbf{A}\mathbf{x}\rVert$ is the **spectral norm** $\lVert\mathbf{A}\rVert_2$.

!!! theorem "Theorem 1.12 (Eckart–Young–Mirsky: best low-rank approximation)"
    Truncating the SVD, $\mathbf{A}_k = \sum_{i=1}^{k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$, gives the best rank-$k$ approximation:

    $$
    \min_{\rank\mathbf{B}\le k}\lVert\mathbf{A}-\mathbf{B}\rVert_2 = \lVert\mathbf{A}-\mathbf{A}_k\rVert_2 = \sigma_{k+1},\qquad
    \min_{\rank\mathbf{B}\le k}\lVert\mathbf{A}-\mathbf{B}\rVert_F = \sqrt{\textstyle\sum_{i>k}\sigma_i^2}.
    $$

??? proof "Proof (spectral norm)"
    First, $\mathbf{A}-\mathbf{A}_k = \sum_{i>k}\sigma_i\mathbf{u}_i\mathbf{v}_i^\top$ has largest singular value $\sigma_{k+1}$, so the error $\sigma_{k+1}$ is achieved. Now let $\mathbf{B}$ have rank $\le k$. By rank–nullity, $\dim\operatorname{null}\mathbf{B}\ge n-k$. The span of $\mathbf{v}_1,\dots,\mathbf{v}_{k+1}$ has dimension $k+1$. Two subspaces of $\R^n$ with dimensions summing to more than $n$ must share a nonzero vector, so pick a unit vector $\mathbf{w}=\sum_{i\le k+1}c_i\mathbf{v}_i$ in both (so $\sum c_i^2 = 1$). Then

    $$
    \lVert(\mathbf{A}-\mathbf{B})\mathbf{w}\rVert^2 = \lVert\mathbf{A}\mathbf{w}\rVert^2 = \sum_{i\le k+1}\sigma_i^2c_i^2 \ge \sigma_{k+1}^2 .
    $$

    So $\lVert\mathbf{A}-\mathbf{B}\rVert_2\ge\sigma_{k+1}$. $\blacksquare$

!!! llm "LLM connection"
    - **LoRA** (Ch. 23) assumes the *update* $\Delta\mathbf{W}$ needed for fine-tuning has low rank, and parameterizes it as $\mathbf{B}\mathbf{A}$ with rank $r\in\{8,16,64\}$.
    - **Attention heads** have $\mathbf{W}_Q\mathbf{W}_K^\top$ of rank $\le d_h$.
    - **Multi-head latent attention** (DeepSeek-V2, Ch. 20) compresses keys/values through a low-rank bottleneck to shrink the KV cache.
    - The singular-value spectrum of weight matrices is a diagnostic. The Muon optimizer (Ch. 5) orthogonalizes updates, setting all singular values of the update to 1.

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

## 1.10 Projections and least squares

The **projection** of $\mathbf{y}$ onto the line spanned by unit vector $\mathbf{u}$ is $(\mathbf{u}^\top\mathbf{y})\,\mathbf{u}$. Onto the column space of a matrix $\mathbf{X}$ (with independent columns) it is $\hat{\mathbf{y}} = \mathbf{X}\hat{\mathbf{w}}$, where $\hat{\mathbf{w}}$ solves least squares:

!!! theorem "Theorem 1.13 (Normal equations)"
    $\hat{\mathbf{w}} = \argmin_{\mathbf{w}}\lVert\mathbf{X}\mathbf{w}-\mathbf{y}\rVert^2$ satisfies $\mathbf{X}^\top\mathbf{X}\hat{\mathbf{w}} = \mathbf{X}^\top\mathbf{y}$, so $\hat{\mathbf{w}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ when $\mathbf{X}$ has full column rank.

??? proof "Proof (geometric, no calculus)"
    Suppose $\hat{\mathbf{w}}$ makes the residual $\mathbf{r} = \mathbf{y}-\mathbf{X}\hat{\mathbf{w}}$ orthogonal to every column, i.e. $\mathbf{X}^\top\mathbf{r}=\mathbf{0}$. For any other $\mathbf{w}$, $\mathbf{X}\mathbf{w}-\mathbf{y} = \mathbf{X}(\mathbf{w}-\hat{\mathbf{w}}) - \mathbf{r}$, and the two terms are orthogonal, since $(\mathbf{X}\mathbf{d})^\top\mathbf{r} = \mathbf{d}^\top\mathbf{X}^\top\mathbf{r} = 0$. By Pythagoras, $\lVert\mathbf{X}\mathbf{w}-\mathbf{y}\rVert^2 = \lVert\mathbf{X}(\mathbf{w}-\hat{\mathbf{w}})\rVert^2 + \lVert\mathbf{r}\rVert^2 \ge \lVert\mathbf{r}\rVert^2$. So $\hat{\mathbf{w}}$ is optimal, and the orthogonality condition $\mathbf{X}^\top(\mathbf{y}-\mathbf{X}\hat{\mathbf{w}}) = \mathbf{0}$ is exactly the normal equations. $\blacksquare$

Least squares is a "linear model trained in closed form". Neural networks are what you get when the model is nonlinear and no closed form exists, so we need gradient descent (Ch. 5). In Ch. 2 we'll re-derive the normal equations by setting a gradient to zero.

## 1.11 Tensors, batching, broadcasting and einsum

In code, a **tensor** is an $n$-dimensional array. LLM code is mostly about tensors of shape `(B, T, C)`: batch, time (sequence position), channels (= $d$).

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
    `x.view(B, T, H, d_h)` reinterprets memory; it does **not** move data. After `transpose`, the tensor is non-contiguous, and you need `.contiguous()` (or `reshape`) before `view` again. The attention code in Ch. 12 does exactly this dance.

## 1.12 High-dimensional geometry: why everything is nearly orthogonal

Our 2-D and 3-D intuitions mislead us in $\R^{4096}$. One fact is fundamental for LLMs:

!!! theorem "Proposition 1.14 (Random vectors are nearly orthogonal)"
    Let $\mathbf{x},\mathbf{y}\in\R^d$ have independent $\mathcal{N}(0,1)$ entries. Then $\E[\mathbf{x}^\top\mathbf{y}] = 0$, $\Var[\mathbf{x}^\top\mathbf{y}] = d$, and $\E\lVert\mathbf{x}\rVert^2 = d$. Hence $\cos(\mathbf{x},\mathbf{y})$ is typically of size $1/\sqrt{d}$.

??? proof "Proof (uses expectation rules from Ch. 3)"
    $\E[\mathbf{x}^\top\mathbf{y}] = \sum_i\E[x_i]\E[y_i] = 0$ by independence. The terms $x_iy_i$ are independent with mean 0, so $\Var[\sum_i x_iy_i] = \sum_i\E[x_i^2y_i^2] = \sum_i\E[x_i^2]\E[y_i^2] = d$. Similarly $\E\lVert\mathbf{x}\rVert^2 = \sum_i\E[x_i^2] = d$, and by the law of large numbers $\lVert\mathbf{x}\rVert\approx\sqrt d$. So $\cos\approx \frac{\pm\sqrt d}{\sqrt d\sqrt d} = \pm\frac{1}{\sqrt d}$. $\blacksquare$

At $d=4096$, a typical random cosine is about $0.016$. You can fit **exponentially many** nearly-orthogonal directions in high dimension (Johnson–Lindenstrauss lemma). Two big consequences:

1. **The $\sqrt{d_h}$ in attention.** The variance of $\mathbf{q}^\top\mathbf{k}$ grows like $d_h$, so we divide by $\sqrt{d_h}$ to keep scores $O(1)$ (Ch. 12; widget there).
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

- Vectors are arrows or feature lists. The dot product measures alignment: $\mathbf{x}^\top\mathbf{y} = \lVert\mathbf{x}\rVert\lVert\mathbf{y}\rVert\cos\theta$.
- Matrices **are** linear maps: column $j$ is the image of $\mathbf{e}_j$. Multiplication is composition. A matmul of $(m\times k)(k\times n)$ costs $2mnk$ FLOPs.
- Rank is the dimension of the output space. Low-rank matrices are cheap ($r(d+k)$ numbers), which is the basis of LoRA.
- Orthogonal matrices preserve geometry, and rotations compose by adding angles, which is the basis of RoPE.
- Symmetric matrices diagonalize with orthonormal eigenvectors. The SVD $\mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top$ works for any matrix, and truncating it is the optimal low-rank approximation.
- In high dimensions, random vectors are nearly orthogonal ($\cos\sim 1/\sqrt d$), which motivates $\sqrt{d_h}$ scaling and makes superposition possible.

## Exercises

1. Show that $\lVert\mathbf{x}+\mathbf{y}\rVert^2 + \lVert\mathbf{x}-\mathbf{y}\rVert^2 = 2\lVert\mathbf{x}\rVert^2 + 2\lVert\mathbf{y}\rVert^2$ (parallelogram law).
2. Prove that $\tr(\mathbf{AB}) = \tr(\mathbf{BA})$ whenever both products are defined, and use it to show $\tr(\mathbf{A}) = \sum_i\lambda_i$ for symmetric $\mathbf{A}$.
3. A linear layer has $d_\text{in}=4096$, $d_\text{out}=11008$. How many FLOPs does a forward pass over a batch of $B=8$ sequences of length $T=2048$ take?
4. Show that $\lVert\mathbf{A}\rVert_F^2 = \sum_i\sigma_i^2$.
5. Prove that the eigenvalues of an orthogonal matrix all have absolute value 1.
6. **Code:** verify Theorem 1.12 numerically: for a random $100\times80$ matrix, compare the Frobenius error of the rank-10 SVD truncation against 1000 random rank-10 matrices $\mathbf{B}\mathbf{C}$ fitted by least squares.

??? success "Solutions"
    1. Expand both squares: the cross terms $\pm 2\mathbf{x}^\top\mathbf{y}$ cancel.
    2. $\tr(\mathbf{AB}) = \sum_i\sum_k A_{ik}B_{ki} = \sum_k\sum_i B_{ki}A_{ik} = \tr(\mathbf{BA})$. Then $\tr(\mathbf{Q}\boldsymbol\Lambda\mathbf{Q}^\top) = \tr(\boldsymbol\Lambda\mathbf{Q}^\top\mathbf{Q}) = \tr\boldsymbol\Lambda$.
    3. $2\cdot BT\cdot d_\text{in}d_\text{out} = 2\cdot 16384\cdot 4096\cdot 11008 \approx 1.48\times10^{12}$ FLOPs.
    4. $\lVert\mathbf{A}\rVert_F^2 = \tr(\mathbf{A}^\top\mathbf{A}) = \tr(\mathbf{V}\boldsymbol\Sigma^\top\boldsymbol\Sigma\mathbf{V}^\top) = \tr(\boldsymbol\Sigma^\top\boldsymbol\Sigma) = \sum\sigma_i^2$.
    5. If $\mathbf{Q}\mathbf{v} = \lambda\mathbf{v}$, then $\lVert\mathbf{v}\rVert = \lVert\mathbf{Q}\mathbf{v}\rVert = \lvert\lambda\rvert\lVert\mathbf{v}\rVert$ (Theorem 1.9 extends to complex vectors with conjugate transpose), so $\lvert\lambda\rvert = 1$.
    6. Every random attempt has error $\ge\sqrt{\sum_{i>10}\sigma_i^2}$.

## References

- G. Strang, *Introduction to Linear Algebra*, 5th ed.; [MIT 18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/).
- M. P. Deisenroth, A. A. Faisal, C. S. Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io/), Ch. 2–4 (2020).
- S. Axler, [*Linear Algebra Done Right*](https://linear.axler.net/), 4th ed. (free PDF), for the proof-oriented view.
- K. B. Petersen, M. S. Pedersen, [*The Matrix Cookbook*](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
- C. Eckart, G. Young, "The approximation of one matrix by another of lower rank," *Psychometrika* (1936).
- N. Elhage et al., [*Toy Models of Superposition*](https://transformer-circuits.pub/2022/toy_model/index.html) (Anthropic, 2022).
