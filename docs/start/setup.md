# Setup & How to Study

!!! abstract "In this page"
    Install a Python environment that runs every code sample on this site, learn the repository layout, and adopt a study method that turns reading into understanding.

## 1. Get the code

```bash
git clone https://github.com/kdellis135/llm_deep_dive.git
cd llm_deep_dive
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install torch numpy regex matplotlib jupyter
python code/test_all.py      # should end with:  All 60 checks passed.
```

A CPU is enough for everything on this site except the optional "train GPT-2 for real" sections, which point you to cloud GPUs. If you have an NVIDIA GPU, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/). On Apple Silicon, PyTorch's `mps` device works for the small models.

## 2. Repository layout

```text
llm_deep_dive/
├── docs/                 # this website (Markdown + MathJax + small JS widgets)
├── code/                 # tested reference implementations shown on the site
│   ├── micrograd.py      # Ch 7  — scalar autograd engine + MLP
│   ├── backprop.py       # Ch 7  — hand-written backward for an MLP and LayerNorm
│   ├── optim.py          # Ch 5  — SGD-momentum, AdamW, LR schedule (match torch.optim)
│   ├── ngram.py          # Ch 9  — bigram (counts and neural) and Bengio MLP language model
│   ├── bpe.py            # Ch 10 — byte-level BPE tokenizer with GPT-4 split regex
│   ├── attention.py      # Ch 12 — attention, its backward pass, multi-head, online-softmax tiling
│   ├── gpt.py            # Ch 14 — complete GPT-2 (124M) + tiny training loop
│   ├── sampling.py       # Ch 18 — temperature, top-k, top-p, min-p
│   ├── quant.py          # Ch 19 — int8 / zero-point / group-wise int4
│   ├── llama.py          # Ch 20 — RMSNorm, RoPE, GQA, SwiGLU, KV cache
│   ├── moe.py            # Ch 20 — mixture-of-experts layer with load-balancing loss
│   ├── alignment.py      # Ch 21–22 — reward model, PPO, DPO, GRPO losses
│   ├── lora.py           # Ch 23 — LoRA layer, injection, merging
│   ├── sae.py            # Ch 24 — sparse autoencoder recovering features from superposition
│   ├── rag.py            # Ch 26 — chunking, BM25, dense search, RRF
│   └── test_all.py       # 60 checks, many against PyTorch reference implementations
└── mkdocs.yml
```

To preview the site locally:

```bash
pip install mkdocs-material
mkdocs serve      # open http://127.0.0.1:8000
```

## 3. How to study this material (it matters)

Understanding comes from **retrieval and generation**, not re-reading. A method that works:

1. **Preview.** Read the chapter's "In this chapter" box and the headings (2 minutes).
2. **Read actively.** When you hit a **Theorem**, stop and try to prove it before opening the **Proof**. Even a failed attempt makes the proof stick.
3. **Re-derive on paper.** After each section, close the page and write the key equation from memory, with shapes.
4. **Code it.** Re-type the code rather than copy-pasting it. Change one thing and predict the effect before running it.
5. **Explain it.** Write a 5-sentence explanation for a colleague. If you can't, re-read the gap.
6. **Space it.** Revisit the [Formula Sheet](../appendix/formulas.md) after 1 day, 1 week and 1 month.

!!! tip "Shape discipline"
    About 80% of the bugs (and the confusion) in deep learning are **shape** problems. Whenever you see a tensor expression, annotate its shape, e.g. `(B, T, C)`. Every code sample on this site does this, and you should too.

## 4. Admonition legend

The site uses the same colored boxes everywhere:

!!! definition "Definition"
    A precise statement of what a term means. Terms are used **only** in their defined sense from then on.

!!! theorem "Theorem / Proposition"
    A claim that is proved.

??? proof "Proof (click to expand)"
    The step-by-step derivation. Try first, then open.

!!! intuition "Intuition"
    The picture to keep in your head.

!!! llm "LLM connection"
    Where the idea shows up in real language models.

!!! watch "Watch"
    A curated video that explains this visually.

!!! lab "Lab"
    Code to run or modify.

!!! warning "Common pitfall"
    Mistakes that even experienced practitioners make.

## 5. A note on the math prerequisites

You need high-school algebra and the willingness to read notation slowly. Part I rebuilds everything else. If a symbol is ever unclear, the [Notation](notation.md) page and the [Glossary](../appendix/glossary.md) define it once for the whole site.
