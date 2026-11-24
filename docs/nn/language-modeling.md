# 9. Language Modeling Basics

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 3](../math/probability.md) (chain rule of probability, MLE), [Ch. 4](../math/information-theory.md) (cross-entropy, perplexity), [Ch. 6–8](mlp.md). **You will learn:** what a language model is, the autoregressive factorization, n-gram models and why counting is MLE, the same bigram model as a neural network (and proof that both reach the same answer), Bengio's MLP language model and distributed representations, word embeddings as matrix factorization, recurrent networks and why their gradients vanish through time, LSTMs, the attention mechanism's origin in translation, and why Transformers replaced RNNs.
    **Why it matters:** GPT is "just" this chapter's MLP language model with a much better way of reading the context. Every idea here (next-token targets, teacher forcing, embeddings, sampling, perplexity) carries over unchanged.

!!! watch "Watch"
    Andrej Karpathy, [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo) (bigram models, counting and neural) and [makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I) (Bengio et al. 2003).

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/PaCmpygFfXo" allowfullscreen loading="lazy"></iframe></div>

    Reading: D. Jurafsky & J. H. Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) (3rd ed. draft), Ch. 3 (n-grams), Ch. 6 (embeddings), Ch. 8 (RNNs). C. Olah, [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/). A. Karpathy, [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).

---

## 9.1 What a language model is

!!! definition "Definition 9.1 (Language model)"
    A **language model** assigns a probability $p(x_1,\dots,x_T)$ to every finite sequence of tokens from a vocabulary $\mathcal{V}$. An **autoregressive** language model does it through the chain rule of probability (Thm 3.1):

    $$
    p(x_1,\dots,x_T) = \prod_{t=1}^Tp(x_t\mid x_{<t}),
    $$

    so it only needs to model the **next-token distribution** $p_\theta(\cdot\mid x_{<t})$, a categorical distribution over $V$ outcomes.

This factorization is exact: no information is lost by modeling one token at a time (and by Thm 4.4 the entropy of the sequence is the sum of the per-step conditional entropies). It gives the model three abilities:

1. **Scoring** text: $\log p(\mathbf{x}) = \sum_t\log p_\theta(x_t\mid x_{<t})$.
2. **Generating** text: sample $x_1$, then $x_2\sim p_\theta(\cdot\mid x_1)$, and so on (Ch. 18).
3. **Training** from raw text: every position of every document is a labeled example "context → next token". No human labels needed. This is **self-supervised learning**, and it's why LLMs can use the whole internet as training data.

A special end-of-sequence token (`<|endoftext|>` in GPT-2) ends generation. The probabilities of all finite sequences then sum to 1.

## 9.2 n-gram models: counting is maximum likelihood

The **Markov assumption** truncates the context to the last $n-1$ tokens: $p(x_t\mid x_{<t})\approx p(x_t\mid x_{t-n+1:t-1})$. For $n = 2$ (a **bigram** model) the parameters are a $V\times V$ table $P_{ab} = p(b\mid a)$.

!!! theorem "Proposition 9.1 (Bigram MLE)"
    Given a training sequence, the maximum-likelihood bigram table is the normalized count table:

    $$
    \hat P_{ab} = \frac{N_{ab}}{\sum_{b'}N_{ab'}},\qquad N_{ab} = \#\{t: x_t = a,\ x_{t+1} = b\}.
    $$

??? proof "Proof"
    The log-likelihood is $\sum_t\log P_{x_tx_{t+1}} = \sum_a\sum_bN_{ab}\log P_{ab}$. It splits into independent problems, one per row $a$: maximize $\sum_bN_{ab}\log P_{ab}$ subject to $\sum_bP_{ab} = 1$. That's the MLE of a categorical distribution, whose solution is the empirical frequency (Thm 3.8). $\blacksquare$

```python
--8<-- "code/ngram.py:bigram_counts"
```

**Smoothing.** An unseen pair gets probability 0, so any test text containing it has infinite loss. **Add-$\alpha$ smoothing** adds $\alpha$ pseudo-counts to every cell, which is MAP estimation with a Dirichlet prior (§3.8). Better schemes such as Kneser–Ney back off to shorter contexts. For decades, smoothed n-gram models were the state of the art in speech recognition and machine translation.

**The curse of dimensionality.** An $n$-gram table has $V^n$ entries. With $V = 50{,}000$ and $n = 5$ that's $3\times10^{23}$ parameters, nearly all never observed. Worse, the table treats every context as unrelated: seeing "the cat sat on the mat" teaches it nothing about "a dog sat on a rug". The table has no notion of **similarity** between words. Fixing this is what neural language models do.

## 9.3 The bigram model as a neural network

Represent the current token as a one-hot vector $\mathbf{e}_a\in\R^V$ and compute logits $\mathbf{z} = \mathbf{W}^\top\mathbf{e}_a$, which is just row $a$ of a $V\times V$ weight matrix. Then $p(\cdot\mid a) = \softmax(\mathbf{W}_{a,:})$, trained with cross-entropy by gradient descent.

```python
--8<-- "code/ngram.py:bigram_nn"
```

!!! theorem "Proposition 9.2 (Same model, same answer)"
    The cross-entropy minimizer of the neural bigram satisfies $\softmax(\mathbf{W}_{a,:}) = \hat P_{a,:}$ for every context $a$ seen in training: its logits are the log-counts up to a per-row constant, $W_{ab} = \log N_{ab}+c_a$.

??? proof "Proof"
    The loss splits into one term per row $a$: $-\sum_bN_{ab}\log\softmax(\mathbf{W}_{a,:})_b$. By Thm 3.8 (or by setting the gradient $N_a(\mathbf{p}-\hat{\mathbf{P}}_{a,:}) = 0$, using Thm 2.8) it's minimized exactly when the softmax equals the empirical distribution. The softmax is invariant to adding a constant to all logits in a row. (Pairs with $N_{ab} = 0$ have no finite optimum: their logits go to $-\infty$. Weight decay or a finite number of steps stops this, which acts like smoothing.) $\blacksquare$

The test suite checks this: gradient descent drives the neural model's loss to the count model's loss. So why bother with the neural version? Because it **generalizes**. The table can't be extended to long contexts, but a neural network can take any context as input, and **share statistical strength between similar contexts**.

!!! llm "The one-hot × matrix = lookup insight"
    $\mathbf{e}_a^\top\mathbf{W}$ selects row $a$ of $\mathbf{W}$. Multiplying by a one-hot vector is the same as indexing. That's exactly what an **embedding layer** is: `nn.Embedding(V, d)(idx)` is `W[idx]`, a learned lookup table, and its backward pass is a scatter-add into the rows that were used (§7.5). The first layer of every LLM is this lookup.

## 9.4 Bengio's neural probabilistic language model (2003)

[Bengio et al., A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (JMLR 2003) introduced the architecture that, scaled up, is still the outline of an LLM:

1. **Embed** each of the previous $n-1$ tokens with a shared table $\mathbf{C}\in\R^{V\times d_\text{emb}}$.
2. **Concatenate** them into one vector of size $(n-1)d_\text{emb}$.
3. **Hidden layer**: $\mathbf{h} = \tanh(\mathbf{W}_h\mathbf{e}+\mathbf{b}_h)$.
4. **Output layer**: logits $\mathbf{z} = \mathbf{W}_o\mathbf{h}+\mathbf{b}_o\in\R^V$, softmax, cross-entropy.

```python
--8<-- "code/ngram.py:mlp_lm"
```

**Why it generalizes.** Words used in similar contexts receive similar gradients, so their embeddings move to nearby points in $\R^{d_\text{emb}}$. Once "cat" and "dog" are close, everything learned about one transfers to the other. The number of parameters grows *linearly* with the context length, not exponentially. Bengio called this a **distributed representation**: each word is a point in a continuous space, and each dimension of that space is shared by many words.

!!! lab "Lab"
    Run `python code/ngram.py`. The MLP language model with only 3 characters of context gets a much lower loss than the bigram. On this tiny text it partly memorizes, so in Karpathy's makemore Part 2 on 32,000 names, you'll see the honest version with a train/validation split: the MLP reaches a validation loss of about 2.1–2.2 nats against the bigram's 2.45.

## 9.5 Word embeddings

Embeddings learned this way turned out to be useful on their own. **word2vec** ([Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)) trained them cheaply by predicting nearby words (skip-gram) with **negative sampling**: for each real (word, context) pair, push $\sigma(\mathbf{u}_w^\top\mathbf{v}_c)$ toward 1, and for $k$ random "negative" contexts, push it toward 0. The resulting vectors famously support analogies by vector arithmetic: $\mathbf{v}_\text{king}-\mathbf{v}_\text{man}+\mathbf{v}_\text{woman}\approx\mathbf{v}_\text{queen}$.

!!! theorem "Fact 9.3 (word2vec factorizes a PMI matrix)"
    Levy & Goldberg ([2014](https://papers.nips.cc/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html)) showed that at the optimum, skip-gram with $k$ negative samples satisfies $\mathbf{u}_w^\top\mathbf{v}_c = \operatorname{PMI}(w,c)-\log k$, where $\operatorname{PMI}(w,c) = \log\frac{p(w,c)}{p(w)p(c)}$ is the pointwise mutual information (the per-pair version of §4.8).

So word embeddings are a **low-rank factorization** of a word–context statistics matrix, the same idea as the truncated SVD (Thm 1.12). Linear structure such as analogies appears because PMI ratios encode it. Embeddings inside an LLM are similar but **contextual**: by the middle layers, the vector for "bank" depends on whether the sentence is about rivers or money.

## 9.6 Recurrent neural networks

A fixed window can't capture long-range dependencies. A **recurrent neural network (RNN)** carries a hidden state through the whole sequence:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{hh}\mathbf{h}_{t-1}+\mathbf{W}_{xh}\mathbf{x}_t+\mathbf{b}),\qquad \mathbf{z}_t = \mathbf{W}_{hy}\mathbf{h}_t.
$$

The same weights are applied at every step, and in principle $\mathbf{h}_t$ summarizes the entire prefix. Training unrolls the recurrence into a $T$-layer-deep graph and runs backprop on it (**backpropagation through time**, BPTT).

!!! theorem "Proposition 9.4 (Vanishing gradients through time)"
    $\dfrac{\partial\mathbf{h}_t}{\partial\mathbf{h}_k} = \prod_{j=k+1}^{t}\diag(1-\mathbf{h}_j^2)\,\mathbf{W}_{hh}$. If $\lVert\mathbf{W}_{hh}\rVert_2<1$, this norm is at most $\lVert\mathbf{W}_{hh}\rVert_2^{\,t-k}$, which decays exponentially in the distance $t-k$. If the largest singular value is above 1, it can grow exponentially.

??? proof "Proof"
    Chain rule through $\mathbf{h}_j = \tanh(\mathbf{W}_{hh}\mathbf{h}_{j-1}+\dots)$: the Jacobian of one step is $\diag(\tanh'(\cdot))\mathbf{W}_{hh}$ with $\tanh' = 1-\mathbf{h}_j^2\in(0,1]$. Spectral norms are submultiplicative, and $\lVert\diag(1-\mathbf{h}_j^2)\rVert_2\le1$. $\blacksquare$

    (See [Pascanu, Mikolov & Bengio, 2013](https://arxiv.org/abs/1211.5063) for the full analysis.)

It's the depth problem of §8.1 again, with the same matrix multiplied $T$ times. In practice, vanilla RNNs struggle to use information more than a few tens of steps back.

**LSTM** ([Hochreiter & Schmidhuber, 1997](https://www.bioinf.jku.at/publications/older/2604.pdf)) adds a **cell state** $\mathbf{c}_t$ that is updated *additively*, controlled by sigmoid gates:

$$
\mathbf{c}_t = \mathbf{f}_t\odot\mathbf{c}_{t-1}+\mathbf{i}_t\odot\tilde{\mathbf{c}}_t,\qquad \mathbf{h}_t = \mathbf{o}_t\odot\tanh(\mathbf{c}_t),
$$

where the forget gate $\mathbf{f}_t$, input gate $\mathbf{i}_t$, output gate $\mathbf{o}_t$ and candidate $\tilde{\mathbf{c}}_t$ are each a small layer of $[\mathbf{h}_{t-1};\mathbf{x}_t]$. When $\mathbf{f}_t\approx1$, $\partial\mathbf{c}_t/\partial\mathbf{c}_{t-1}\approx\mathbf{I}$: the same **identity highway** as a residual connection (Prop. 8.4), through time instead of through depth. The GRU ([Cho et al., 2014](https://arxiv.org/abs/1406.1078)) is a simpler variant. LSTMs dominated language modeling and translation from about 2014 to 2017.

## 9.7 Attention is born: sequence-to-sequence translation

In encoder–decoder translation ([Sutskever et al., 2014](https://arxiv.org/abs/1409.3215)), an encoder RNN reads the source sentence into one final vector, and a decoder RNN generates the translation from it. Squeezing a whole sentence into one fixed-size vector is a bottleneck.

[Bahdanau, Cho & Bengio (2015)](https://arxiv.org/abs/1409.0473) let the decoder **look back** at every encoder state: at each output step $t$, compute a score $e_{tj} = \operatorname{score}(\mathbf{s}_{t-1},\mathbf{h}_j)$ for each source position $j$, normalize with a softmax to weights $\alpha_{tj}$, and use the weighted average $\sum_j\alpha_{tj}\mathbf{h}_j$ as extra input. The decoder learns *where to look*, soft alignment learned end to end. This is **attention**, and the Transformer (Ch. 12) is what you get when attention becomes the *only* mechanism for mixing information between positions.

## 9.8 Why Transformers replaced RNNs

| | RNN / LSTM | Transformer |
|---|---|---|
| Training parallelism across positions | none: $\mathbf{h}_t$ needs $\mathbf{h}_{t-1}$, so $T$ sequential steps | full: all $T$ positions in one batched matmul |
| Path length between positions $i$ and $j$ | $O(\lvert i-j\rvert)$ multiplications (vanishing gradients) | $O(1)$: one attention hop |
| Memory of the past | fixed-size state $\mathbf{h}_t$ | all previous tokens' keys and values (KV cache) |
| Cost per sequence | $O(Td^2)$ | $O(Td^2 + T^2d)$ |
| Generation cost per token | $O(d^2)$, constant | $O(d^2+Td)$, grows with context |

GPUs are massively parallel machines. A model that can process all $T$ positions of a training sequence at once uses them far better than one that must step through positions one at a time, and on the same compute budget it can be trained on far more data. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017), made this trade, and it won. (The RNN's constant per-token generation cost is attractive, which is why linear-attention and state-space models like **Mamba** try to combine both; see Ch. 20.)

## 9.9 Training layout: teacher forcing and shifted targets

Every LLM training step uses the same data layout. Take a chunk of $T+1$ consecutive token ids and split it:

```text
tokens : [ The,  cat,  sat,  on,  the,  mat ]
inputs : [ The,  cat,  sat,  on,  the ]        x = chunk[:-1]
targets: [ cat,  sat,  on,   the, mat ]        y = chunk[1:]
```

The model outputs logits of shape `(B, T, V)`, one next-token distribution **per position**, and the loss is the cross-entropy averaged over all $B\times T$ positions. So one sequence of length $T$ provides $T$ training examples at once, with context lengths $1,2,\dots,T$. This is **teacher forcing**: during training the model is always conditioned on the *true* prefix, never on its own samples. (The resulting mismatch with generation, where the model conditions on its own possibly wrong outputs, is called **exposure bias**. Post-training with RL, Ch. 21–22, trains on the model's own samples.) In a Transformer this requires a **causal mask**, so that position $t$ can't peek at $x_{t+1}$ (Ch. 12).

## Key takeaways

- An autoregressive LM factorizes $p(\mathbf{x}) = \prod_tp(x_t\mid x_{<t})$ exactly, and trains on raw text with next-token cross-entropy.
- n-gram MLE = normalized counts. Tables explode as $V^n$ and can't generalize across similar words.
- A one-hot vector times a matrix is a lookup, i.e. an embedding. The neural bigram converges to the count model.
- Bengio's MLP LM (embed → concatenate → hidden → softmax) introduced distributed representations. GPT keeps the embed-and-softmax ends and replaces the middle.
- RNNs have vanishing gradients through time. LSTMs fix this with an additive cell state, the same idea as residual connections.
- Attention came from translation. Transformers won because they parallelize over positions and connect any two tokens in one step.

## Exercises

1. Show that the bigram model's MLE loss on its own training data equals the empirical conditional entropy $H(X_{t+1}\mid X_t)$ of the data.
2. For $V = 27$ (letters plus a boundary token), how many parameters do bigram, trigram and 5-gram tables have? How many does a Bengio MLP with context 4, $d_\text{emb} = 10$ and 200 hidden units have?
3. Prove that $\softmax(\mathbf{z}+c\mathbf{1}) = \softmax(\mathbf{z})$, and explain why the neural bigram's logits are only defined up to a per-row constant.
4. In the RNN of §9.6, take $\mathbf{W}_{hh} = 0.9\,\mathbf{I}$ and $\mathbf{h}\approx0$. Roughly what fraction of the gradient from step $t$ reaches step $t-50$?
5. Why can't you use teacher forcing *at generation time*?
6. **Code:** follow makemore Part 2: train the MLP LM on [`names.txt`](https://github.com/karpathy/makemore/blob/master/names.txt) with a train/dev split, and tune $d_\text{emb}$, the hidden size and the learning rate to beat a dev loss of 2.2.

??? success "Solutions"
    1. At the MLE, $\hat P_{ab} = \hat p(b\mid a)$, so the average NLL is $-\sum_{a,b}\hat p(a,b)\log\hat p(b\mid a) = H(X_{t+1}\mid X_t)$ under the empirical distribution.
    2. $27^2 = 729$; $27^3 = 19{,}683$; $27^5\approx1.4\times10^7$. The MLP: $27\cdot10 + (40\cdot200+200) + (200\cdot27+27) = 270+8{,}200+5{,}427 = 13{,}897$.
    3. $e^{z_i+c}/\sum_je^{z_j+c} = e^{z_i}/\sum_je^{z_j}$. Since only the softmax enters the loss, adding a constant to a row of $\mathbf{W}$ changes nothing, so the minimizer is a family.
    4. $\tanh'\approx1$, so the Jacobian is about $0.9^{50}\approx0.005$, i.e. 0.5%.
    5. At generation time the true next token is unknown. That's what you're generating.
    6. Karpathy reaches about 2.17 on dev with $d_\text{emb} = 10$, 200 hidden units and a learning-rate decay.

## References

- Y. Bengio, R. Ducharme, P. Vincent, C. Jauvin, [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (JMLR 2003).
- T. Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (2013).
- O. Levy & Y. Goldberg, [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/2014/hash/feab05aa91085b7a8012516bc3533958-Abstract.html) (NeurIPS 2014).
- S. Hochreiter & J. Schmidhuber, Long Short-Term Memory, *Neural Computation* 9 (1997).
- R. Pascanu, T. Mikolov, Y. Bengio, [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063) (ICML 2013).
- D. Bahdanau, K. Cho, Y. Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) (ICLR 2015).
- A. Karpathy, [makemore](https://github.com/karpathy/makemore).
- D. Jurafsky & J. H. Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/), 3rd ed. draft.
