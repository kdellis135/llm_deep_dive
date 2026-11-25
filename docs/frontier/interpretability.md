---
seo_title: "LLM Interpretability: Probes, Sparse Autoencoders, Circuits"
description: "Linear probes, superposition, sparse autoencoders with a working implementation, circuits and activation patching, attribution graphs inside Claude, and steering vectors."
---

# 24. Interpretability

!!! abstract "In this chapter"
    **Prerequisites:** [§1.12](../math/linear-algebra.md#112-high-dimensional-geometry-why-everything-is-nearly-orthogonal) (near-orthogonality), [Ch. 12](../transformer/attention.md) (QK/OV circuits, induction heads), [Ch. 13](../transformer/block.md) (residual stream, logit lens). **You will learn:** linear probes and the linear representation hypothesis; superposition, and why high-dimensional geometry allows it; sparse autoencoders (dictionary learning) with a working implementation that recovers features from superposition; circuits, activation patching and attribution patching; attribution graphs and what they revealed inside Claude; steering vectors and the single "refusal direction"; locating and editing factual knowledge; and the tools and open problems.
    **Why it matters:** we train LLMs but don't design their internal algorithms. Interpretability tries to reverse-engineer them. It's increasingly practical, for debugging, steering and auditing models, and it's central to making advanced AI safe.

!!! watch "Read & watch"
    - Anthropic, [Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model) (2024), the accessible summary of *Scaling Monosemanticity*.
    - Anthropic, [Tracing the thoughts of a large language model](https://www.anthropic.com/research/tracing-thoughts-language-model) (2025), the accessible summary of *On the Biology of a Large Language Model*.
    - 3Blue1Brown, [How might LLMs store facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8), which ends with superposition.
    - N. Nanda, [A Comprehensive Mechanistic Interpretability Explainer & Glossary](https://www.neelnanda.io/mechanistic-interpretability/glossary), and the [ARENA](https://arena-chapter1-transformer-interp.streamlit.app/) course for hands-on exercises.

---

## 24.1 Three levels of understanding

- **Behavioral**: study inputs and outputs (benchmarks, red-teaming). It tells you *what* a model does, not *how*.
- **Representational**: what information is encoded in the activations, and where? (Probing.)
- **Mechanistic**: what algorithm do the weights implement? (Features and circuits.) This is the goal of **mechanistic interpretability**: explanations as precise as reading the source code.

The residual-stream view (§13.5) makes the mechanistic level tractable. Every component reads from and writes to one shared vector space by linear maps, so contributions can be traced and added up.

## 24.2 Probes and the linear representation hypothesis

A **linear probe** is a logistic regression trained to predict some property (part of speech, truth of a statement, a board-game state) from a layer's activations. High accuracy means the property is linearly decodable there.

The **linear representation hypothesis**: models represent many concepts as **directions** in activation space, so the strength of a concept is the projection onto its direction. Evidence includes word-embedding analogies (§9.5); a "truth direction" in which true and false statements separate ([Marks & Tegmark, 2023](https://arxiv.org/abs/2310.06824)); and **Othello-GPT**, a model trained only on sequences of legal moves, whose activations linearly encode the board state, once you express it as "my piece vs their piece" rather than "black vs white" ([Li et al., 2023](https://arxiv.org/abs/2210.13382); [Nanda et al., 2023](https://arxiv.org/abs/2309.00941)). The last example shows a sequence model building an internal *world model* that nobody gave it.

The caveat: a probe shows information is *present*, not that the model *uses* it. Causal interventions (§24.5) are needed for that.

## 24.3 Superposition

If each neuron represented one concept, we could read models neuron by neuron. In practice most neurons are **polysemantic**: one neuron fires for academic citations, Korean text and HTTP requests. The leading explanation is **superposition** ([Elhage et al., Toy Models of Superposition, 2022](https://transformer-circuits.pub/2022/toy_model/index.html)): the model represents **more features than it has dimensions**, as nearly orthogonal directions, accepting a little interference between them.

Why is that possible? In $d$ dimensions you can have at most $d$ exactly orthogonal directions, but exponentially many *nearly* orthogonal ones.

!!! theorem "Fact 24.1 (Johnson–Lindenstrauss and near-orthogonality)"
    For any $n$ points in $\R^D$ and $\epsilon\in(0,1)$, a random linear map to $k = O(\epsilon^{-2}\log n)$ dimensions preserves all pairwise distances within a factor $1\pm\epsilon$ with high probability. Equivalently, $\R^k$ contains $e^{\Omega(\epsilon^2k)}$ unit vectors whose pairwise cosine similarities are all below $\epsilon$. (Random unit vectors in $\R^k$ have cosine similarity of order $1/\sqrt k$, Prop. 1.14.)

Superposition works when features are **sparse**: if each input activates only a few of the many features, the interference between simultaneously active features is rarely large. Elhage et al.'s toy models show a sharp transition. With dense features the model stores only the $d$ most important features, one per dimension. As sparsity increases, it switches to packing more features than dimensions into geometric arrangements (antipodal pairs, triangles, pentagons). Language is full of sparse features (any given concept is absent from most contexts), so superposition is expected.

## 24.4 Sparse autoencoders: extracting features

If activations are sparse combinations of many feature directions, we can try to recover those directions by **dictionary learning**. A **sparse autoencoder (SAE)** maps an activation $\mathbf{x}\in\R^d$ to a much wider, sparse code $\mathbf{f}\in\R^m$ ($m\gg d$) and back:

$$
\mathbf{f} = \operatorname{ReLU}\big(\mathbf{W}_\text{enc}(\mathbf{x}-\mathbf{b}_\text{dec})+\mathbf{b}_\text{enc}\big),\qquad \hat{\mathbf{x}} = \mathbf{W}_\text{dec}\mathbf{f}+\mathbf{b}_\text{dec} = \mathbf{b}_\text{dec} + \sum_if_i\,\mathbf{d}_i,
$$

$$
\mathcal{L} = \lVert\mathbf{x}-\hat{\mathbf{x}}\rVert^2 + \lambda\sum_if_i\lVert\mathbf{d}_i\rVert.
$$

Each decoder column $\mathbf{d}_i$ is a candidate **feature direction**, and $f_i$ its activation. The L1 penalty (weighted by the decoder norms, so the model can't cheat by shrinking $\mathbf{f}$ and growing $\mathbf{d}$) makes most $f_i$ zero for any given input.

```python
--8<-- "code/sae.py:sae"
```

To check that this works, we make data where we **know** the answer: 64 random feature directions in only 16 dimensions, each input activating about 2 of them.

```python
--8<-- "code/sae.py:toy"
```

The test suite trains the SAE above with a 128-entry dictionary and checks that it recovers at least 90% of the 64 true directions (cosine similarity > 0.9), with about 6 active codes per input. Run `python code/sae.py` and try smaller $\lambda$: reconstruction improves, but codes stop being sparse and the recovered directions turn into mixtures of features. That's the core trade-off in real SAEs.

**At scale.** Anthropic's *Towards Monosemanticity* ([Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)) trained SAEs on a one-layer model and found interpretable features (DNA sequences, Arabic script, base64). *Scaling Monosemanticity* ([Templeton et al., 2024](https://transformer-circuits.pub/2024/scaling-monosemanticity/)) scaled to **Claude 3 Sonnet** with up to 34 million features, finding features for specific entities (the Golden Gate Bridge), abstract concepts (code bugs, sycophancy, secrecy, inner conflict), and multilingual and multimodal features (the same feature fires for the concept in text and images, in many languages). **Steering** with a feature works: clamping the Golden Gate Bridge feature high produced "Golden Gate Claude", which brought up the bridge in every answer and even claimed to be the bridge. Variants such as TopK SAEs ([Gao et al., 2024](https://arxiv.org/abs/2406.04093)) and JumpReLU SAEs (used for Google DeepMind's open [Gemma Scope](https://arxiv.org/abs/2408.05147)) improve the reconstruction–sparsity trade-off.

Limits: SAEs don't reconstruct activations perfectly (the unexplained "dark matter" can matter for behavior), features split into finer features as the dictionary grows, and a feature's label from its top activating examples can be misleading.

## 24.5 Circuits and causal interventions

A **circuit** is a subgraph of components (heads, MLP neurons, features) and the connections between them that implements a behavior. You've seen one: the **induction circuit** of a previous-token head and an induction head (§12.7). A larger example is the **indirect object identification** (IOI) circuit in GPT-2 small ([Wang et al., 2022](https://arxiv.org/abs/2211.00593)): for "When Mary and John went to the store, John gave a drink to ___", about 26 heads in several functional groups (duplicate-token heads, S-inhibition heads, name-mover heads) together produce "Mary".

Circuits are found by **causal interventions**:

- **Activation patching** (causal tracing). Run the model on a *clean* input (with the behavior) and a *corrupted* one (e.g. names swapped). Then rerun the corrupted input but replace one component's activation with its value from the clean run. If a metric $M$ (e.g. the logit difference between the correct and incorrect name) recovers, that component carries the relevant information:

    $$
    \text{effect}(h) = M\big(\text{corrupted}\mid h\leftarrow h_\text{clean}\big) - M(\text{corrupted}).
    $$

- **Attribution patching** ([Nanda, 2023](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching); [Syed et al., 2023](https://arxiv.org/abs/2310.10348)): patching every component one at a time is expensive. A first-order Taylor expansion (Thm 2.9) approximates all effects at once from one forward and one backward pass:

    $$
    \text{effect}(h)\approx(h_\text{clean}-h_\text{corr})^\top\nabla_hM\big|_\text{corrupted}.
    $$

- **Ablation**: set a component to zero or to its mean value and see what breaks.

## 24.6 Attribution graphs: tracing Claude's thoughts

Anthropic's 2025 work ([Ameisen et al., Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html); [Lindsey et al., On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)) replaces the MLPs of a model with a **cross-layer transcoder**, an interpretable sparse replacement trained to reproduce the MLPs' outputs from their inputs, and then traces, for a specific prompt, which features cause which, all the way from input tokens to the output. The resulting **attribution graphs** for Claude 3.5 Haiku showed, among other things:

- **Multi-step reasoning in one forward pass**: for "the capital of the state containing Dallas", features for *Dallas* activate *Texas*, which with *capital* activates *Austin*. The intermediate step is visible internally.
- **Planning ahead**: when writing a rhyming couplet, the model picks the rhyming word for the end of the next line *before* writing the line, then writes toward it.
- **Shared multilingual concepts**: the same features for concepts like "small" and "opposite" across English, French and Chinese, with language-specific features at the input and output ends. Larger models share more.
- **Parallel arithmetic paths**: for addition, one path estimates the rough magnitude and another computes the exact last digit, then they combine, while the model *describes* the schoolbook carry algorithm when asked how it did it.
- **Unfaithful reasoning made visible**: when given a wrong hint on a hard math problem, the model sometimes works backward from the hint to construct plausible-looking intermediate steps (cf. §22.8).
- **Hallucination mechanics**: a default "can't answer" circuit is suppressed by features for *known entities*. Misfires of that suppression on partially familiar names produce confident hallucinations.

## 24.7 Steering with directions

If concepts are directions, you can **add** them to the residual stream at inference to change behavior, with no training:

- **Activation addition** ([Turner et al., 2023](https://arxiv.org/abs/2308.10248)): take the difference of activations between two contrasting prompts (e.g. "Love" − "Hate") and add a multiple of it during generation.
- **Representation engineering** ([Zou et al., 2023](https://arxiv.org/abs/2310.01405)): find directions for honesty, emotion or power-seeking from many contrasting examples, then read or control them.
- **The refusal direction** ([Arditi et al., 2024](https://arxiv.org/abs/2406.11717)): in 13 open chat models, refusal of harmful requests is mediated by **one direction** $\mathbf{r}$: the difference of mean activations on harmful vs harmless prompts. Projecting it out of the residual stream ($\mathbf{x}\leftarrow\mathbf{x}-(\mathbf{r}^\top\mathbf{x})\mathbf{r}$, $\lVert\mathbf{r}\rVert = 1$) at every layer removes refusals almost completely; adding it makes harmless requests get refused. A striking result, and also a warning about how shallow some safety training is for open-weight models.
- **Persona vectors** ([Anthropic, 2025](https://www.anthropic.com/research/persona-vectors)): directions for traits such as sycophancy or "evil", used to monitor shifts in a model's character during training and to flag training data that would cause them.

## 24.8 Where is knowledge stored?

Meng et al., [Locating and Editing Factual Associations in GPT (ROME)](https://arxiv.org/abs/2202.05262) (2022), used causal tracing on prompts like "The Eiffel Tower is located in the city of": restoring clean activations at the **last token of the subject** ("Tower") in **middle-layer MLPs** recovered the correct answer. This is consistent with the key-value memory view of FFNs (§6.7). They then edited a fact ("The Eiffel Tower is in Rome") with a **rank-one update** to one MLP weight matrix, computed in closed form. Edits generalize to paraphrases but have side effects and don't always propagate to related facts ("What country is the Eiffel Tower in?"), which suggests knowledge is stored in a more distributed and entangled way than one matrix.

## 24.9 Tools

- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens): load open models with hooks on every activation; patching, caching, ablation.
- [nnsight](https://nnsight.net/): interventions on any PyTorch model, including remote execution on large models.
- [SAELens](https://github.com/jbloomAus/SAELens) and [Gemma Scope](https://huggingface.co/google/gemma-scope): pretrained SAEs for open models; [Neuronpedia](https://www.neuronpedia.org/): browse features and their activations in the browser.
- Anthropic's open-sourced [circuit-tracer](https://github.com/safety-research/circuit-tracer) library generates attribution graphs for open models.

## 24.10 Why it matters, and what's unsolved

Interpretability is used for **debugging** (why did the model do that?), **steering** (control without retraining), **monitoring** (detect deception, sycophancy, or harmful plans in activations), and **auditing**: in an Anthropic experiment, teams were given a model deliberately trained with a hidden objective, and the teams using interpretability tools found it ([Marks et al., 2025](https://arxiv.org/abs/2503.10965)).

Open problems: SAE features aren't guaranteed to be the model's "true" units; explanations cover only part of the computation; attention circuits are harder to trace than MLPs; and methods must scale to the largest models and to long agentic transcripts. Progress is fast.

## Key takeaways

- Probes show what's linearly decodable; causal interventions show what's used.
- Concepts tend to be directions (the linear representation hypothesis). Models store more sparse features than dimensions (superposition), as nearly orthogonal directions.
- Sparse autoencoders recover features by reconstructing activations through a wide, L1-sparse code. They scaled to millions of interpretable features in Claude 3 Sonnet, and feature clamping steers behavior.
- Circuits are found with activation patching; attribution patching approximates it with gradients. Attribution graphs reveal multi-step reasoning, planning and shared multilingual features.
- Adding or removing directions steers models; refusal in many chat models is a single direction.
- Factual recall concentrates in mid-layer MLPs at the subject's last token, and can be edited with rank-one updates, imperfectly.

## Exercises

1. Show that if two feature directions $\mathbf{d}_1,\mathbf{d}_2$ have cosine similarity $c$, reading feature 1 by projection when only feature 2 is active (with value 1) gives interference $c$. Why does sparsity make this acceptable?
2. Why must the SAE's L1 penalty be weighted by the decoder column norms (or the columns be normalized)?
3. Derive the attribution-patching formula from a first-order Taylor expansion of $M$ around the corrupted activations.
4. Show that projecting out a unit direction $\mathbf{r}$ from $\mathbf{x}$ is a linear map with matrix $\mathbf{I}-\mathbf{r}\mathbf{r}^\top$, and that it can be folded into the weights of every matrix that writes to the residual stream.
5. Using the logit lens (§13.5) on GPT-2 with a factual prompt, at which layer does the correct answer first become the top prediction?
6. **Code:** in `sae.py`, vary `p_active` from 0.01 to 0.3. How does feature recovery change, and why?

??? success "Solutions"
    1. $\hat f_1 = \mathbf{d}_1^\top(1\cdot\mathbf{d}_2) = c$. If features are rarely active together, interference appears rarely, and a ReLU threshold can filter small values.
    2. Otherwise the model can shrink $\mathbf{f}$ by any factor and grow $\mathbf{W}_\text{dec}$ to compensate, making the L1 penalty arbitrarily small without becoming sparser.
    3. $M(h_\text{corr}+\Delta)\approx M(h_\text{corr})+\nabla_hM^\top\Delta$ with $\Delta = h_\text{clean}-h_\text{corr}$.
    4. $\mathbf{x}-\mathbf{r}(\mathbf{r}^\top\mathbf{x}) = (\mathbf{I}-\mathbf{r}\mathbf{r}^\top)\mathbf{x}$. For a matrix $\mathbf{W}_\text{out}$ that writes $\mathbf{W}_\text{out}\mathbf{z}$ into the stream, replace it by $(\mathbf{I}-\mathbf{r}\mathbf{r}^\top)\mathbf{W}_\text{out}$, so the direction is never written. This is how "abliterated" open models are made.
    5. Typically in the middle-to-late layers. Earlier layers predict generic continuations.
    6. Recovery is easy when features are very sparse and degrades as `p_active` grows: with many features active at once, interference grows, the code can't be both sparse and accurate, and the features stop being recoverable (the dense regime of the toy models).

## References

- N. Elhage et al., [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) (Anthropic, 2022).
- T. Bricken et al., [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) (Anthropic, 2023); A. Templeton et al., [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (Anthropic, 2024).
- E. Ameisen et al., [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) and J. Lindsey et al., [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) (Anthropic, 2025).
- K. Wang et al., [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593) (ICLR 2023).
- K. Meng et al., [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) (NeurIPS 2022).
- A. Arditi et al., [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) (NeurIPS 2024).
- K. Li et al., [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382) (ICLR 2023).
- C. Olah et al., [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/) (Distill, 2020).
