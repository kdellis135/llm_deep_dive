# LLM Deep Dive

**From first principles to frontier LLMs: the math, the proofs, the code, and how to apply it.**

📖 **Read the site: https://kdellis135.github.io/llm_deep_dive/**

A structured, 29-chapter path from linear algebra and the chain rule to Transformers, scaling laws, efficient inference, RLHF/DPO/GRPO, reasoning models, interpretability, and building production applications with RAG, agents and the Claude API. Every key formula is derived step by step, every major component is implemented in tested Python, and each chapter links the best videos (Karpathy, 3Blue1Brown, and others) and primary papers.

| Part | Chapters |
|---|---|
| Start here | Setup & study method, Notation |
| I · Math foundations | Linear algebra, Calculus, Probability, Information theory, Optimization |
| II · Neural networks | MLPs, Backpropagation, Training deep nets, Language modeling basics |
| III · The Transformer | Tokenization, Embeddings & positions, Attention, The block, GPT from scratch |
| IV · Training at scale | Pretraining, Distributed training & precision, Scaling laws |
| V · Inference | Decoding & sampling, Efficient inference |
| VI · Frontier models | Modern architectures, Post-training, Reasoning models, Fine-tuning, Interpretability |
| VII · Applying LLMs | Prompt & context engineering, Embeddings & RAG, Tool use & agents, Building with Claude, Evaluation & production |

## Run the code

```bash
pip install torch numpy regex
python code/test_all.py        # All 60 checks passed.
```

`code/` contains the implementations shown on the site: a scalar autograd engine, manual backprop, n-gram and MLP language models, a byte-level BPE tokenizer, attention (with its backward pass and the online-softmax trick), a complete GPT-2, a Llama-style model with RoPE/GQA/SwiGLU/KV cache, mixture-of-experts, sampling, quantization, LoRA, reward/PPO/DPO/GRPO losses, a sparse autoencoder, and a retrieval core (BM25, dense search, RRF).

## Build the site locally

```bash
pip install -r requirements-docs.txt
mkdocs serve                   # http://127.0.0.1:8000
```

The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and deployed to GitHub Pages by `.github/workflows/pages.yml` on every push to `main`.

## License

Text content: CC BY 4.0. Code: MIT.
