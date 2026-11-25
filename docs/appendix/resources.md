---
seo_title: "Best Resources to Learn LLMs: Videos, Books, Code"
description: "The best videos, books, courses, papers and code for learning large language models, including Karpathy and 3Blue1Brown, with what each resource is best for."
---

# Resources

!!! abstract "How to use this page"
    The authoritative sources this site builds on, grouped by type, with what each is best for. Every chapter also ends with its own references.

## Video series

| Resource | Best for |
|---|---|
| Andrej Karpathy, [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | building micrograd, makemore, GPT and the tokenizer from scratch (Ch. 7–14) |
| Andrej Karpathy, [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) | real training runs and performance engineering (Ch. 14–16) |
| Andrej Karpathy, [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) and [How I use LLMs](https://www.youtube.com/watch?v=EWvNQjAaOHw) | the whole pipeline, and practical use |
| 3Blue1Brown, [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab), [Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr), [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | geometric intuition for Part I, backprop, attention, how LLMs store facts |
| Stanford [CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/) | a full university course on Parts III–V, with assignments |
| Stanford [CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) | embeddings, RNNs, attention, Transformers |
| Stanford [CS25: Transformers United](https://web.stanford.edu/class/cs25/) | talks by researchers on current topics |
| MIT [18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) and [18.065](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) (Gilbert Strang) | rigorous linear algebra and matrix methods |
| D. MacKay, [Information Theory, Pattern Recognition and Neural Networks](https://www.inference.org.uk/itprnn_lectures/) | information theory (Ch. 4) |

## Books (free online where linked)

- M. Deisenroth, A. Faisal, C. Ong, [*Mathematics for Machine Learning*](https://mml-book.github.io/) — Part I.
- I. Goodfellow, Y. Bengio, A. Courville, [*Deep Learning*](https://www.deeplearningbook.org/) — Parts I–II.
- S. Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) — the best modern deep-learning textbook, with excellent figures.
- K. Murphy, [*Probabilistic Machine Learning*](https://probml.github.io/pml-book/) (2 vols.) — probability and ML in depth.
- T. Cover & J. Thomas, *Elements of Information Theory* — Ch. 4.
- S. Boyd & L. Vandenberghe, [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/) — Ch. 5.
- D. Jurafsky & J. Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) — NLP foundations (Ch. 9–10).
- N. Lambert, [*Reinforcement Learning from Human Feedback*](https://rlhfbook.com/) — Ch. 21–22.
- R. Sutton & A. Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book.html) — RL foundations.
- C. Manning, P. Raghavan, H. Schütze, [*Introduction to Information Retrieval*](https://nlp.stanford.edu/IR-book/) — Ch. 26.

## Code to read

- Karpathy: [micrograd](https://github.com/karpathy/micrograd), [makemore](https://github.com/karpathy/makemore), [minbpe](https://github.com/karpathy/minbpe), [nanoGPT](https://github.com/karpathy/nanoGPT), [build-nanogpt](https://github.com/karpathy/build-nanogpt), [llm.c](https://github.com/karpathy/llm.c), [nanochat](https://github.com/karpathy/nanochat).
- A. Rush et al., [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
- Meta, [Llama reference implementation](https://github.com/meta-llama/llama-models); Hugging Face [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl).
- [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [FlashAttention](https://github.com/Dao-AILab/flash-attention).
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [SAELens](https://github.com/jbloomAus/SAELens).
- This site's tested code: [`code/`](https://github.com/kdellis135/llm_deep_dive/tree/main/code).

## Interactive visualizations

- B. Bycroft, [LLM Visualization](https://bbycroft.net/llm) — every tensor of a GPT in 3-D.
- Polo Club, [Transformer Explainer](https://poloclub.github.io/transformer-explainer/).
- [Tiktokenizer](https://tiktokenizer.vercel.app/) — see how text is tokenized.
- [TensorFlow Playground](https://playground.tensorflow.org/) — tiny MLPs in the browser.
- G. Goh, [Why Momentum Really Works](https://distill.pub/2017/momentum/); C. Olah, [Visual Information Theory](https://colah.github.io/posts/2015-09-Visual-Information/).
- [Neuronpedia](https://www.neuronpedia.org/) — browse interpretable features.

## Foundational papers (in the order of this site)

1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017).
2. Radford et al., [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019); Brown et al., [GPT-3](https://arxiv.org/abs/2005.14165) (2020).
3. Sennrich et al., [BPE for NMT](https://arxiv.org/abs/1508.07909) (2016); Su et al., [RoFormer](https://arxiv.org/abs/2104.09864) (2021).
4. Kaplan et al., [Scaling Laws](https://arxiv.org/abs/2001.08361) (2020); Hoffmann et al., [Chinchilla](https://arxiv.org/abs/2203.15556) (2022).
5. Rajbhandari et al., [ZeRO](https://arxiv.org/abs/1910.02054) (2020); Shoeybi et al., [Megatron-LM](https://arxiv.org/abs/1909.08053) (2019).
6. Dao et al., [FlashAttention](https://arxiv.org/abs/2205.14135) (2022); Kwon et al., [PagedAttention](https://arxiv.org/abs/2309.06180) (2023); Leviathan et al., [Speculative Decoding](https://arxiv.org/abs/2211.17192) (2023).
7. Touvron et al., [LLaMA](https://arxiv.org/abs/2302.13971) (2023); Meta, [Llama 3](https://arxiv.org/abs/2407.21783) (2024); DeepSeek-AI, [DeepSeek-V3](https://arxiv.org/abs/2412.19437) (2024).
8. Ouyang et al., [InstructGPT](https://arxiv.org/abs/2203.02155) (2022); Bai et al., [Constitutional AI](https://arxiv.org/abs/2212.08073) (2022); Rafailov et al., [DPO](https://arxiv.org/abs/2305.18290) (2023).
9. Wei et al., [Chain-of-Thought](https://arxiv.org/abs/2201.11903) (2022); Shao et al., [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300) (2024); DeepSeek-AI, [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (2025).
10. Hu et al., [LoRA](https://arxiv.org/abs/2106.09685) (2021); Dettmers et al., [QLoRA](https://arxiv.org/abs/2305.14314) (2023).
11. Elhage et al., [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (2021); Templeton et al., [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/) (2024).
12. Lewis et al., [RAG](https://arxiv.org/abs/2005.11401) (2020); Yao et al., [ReAct](https://arxiv.org/abs/2210.03629) (2022).

## Engineering guides

- Anthropic: [Claude API docs](https://platform.claude.com/docs), [prompt engineering](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview), [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents), [Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents), [Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents), [Contextual retrieval](https://www.anthropic.com/news/contextual-retrieval).
- Hugging Face, [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook); Google DeepMind, [How to Scale Your Model](https://jax-ml.github.io/scaling-book/).
- H. He, [Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html).
- H. Husain, [Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/).

## Following the field

- Research blogs: [Anthropic research](https://www.anthropic.com/research) and [Transformer Circuits](https://transformer-circuits.pub/), [Google DeepMind](https://deepmind.google/research/), [OpenAI research](https://openai.com/research/), [Meta AI](https://ai.meta.com/research/).
- Explanatory writing: [Lilian Weng](https://lilianweng.github.io/), [Sebastian Raschka](https://magazine.sebastianraschka.com/), [Jay Alammar](https://jalammar.github.io/), [Simon Willison](https://simonwillison.net/), [Interconnects (Nathan Lambert)](https://www.interconnects.ai/).
- Papers: [arXiv cs.CL](https://arxiv.org/list/cs.CL/recent) and [cs.LG](https://arxiv.org/list/cs.LG/recent); [Hugging Face daily papers](https://huggingface.co/papers).
- Leaderboards (read with the caveats of §29.5): [LMArena](https://lmarena.ai/), [Epoch AI benchmarks](https://epoch.ai/benchmarks), [SWE-bench](https://www.swebench.com/), [MTEB](https://huggingface.co/spaces/mteb/leaderboard).
