# 10. Tokenization

!!! abstract "In this chapter"
    **Prerequisites:** [Ch. 9](../nn/language-modeling.md). **You will learn:** why LLMs read subword tokens rather than characters or words, Unicode and UTF-8 from the bit level up, byte-pair encoding (BPE) training and encoding with a complete implementation, GPT-4's pre-tokenization regex piece by piece, special tokens and chat templates, WordPiece and the Unigram/SentencePiece alternatives, the vocabulary sizes real models use and the trade-off behind them, and the long list of LLM oddities that are really tokenization bugs.
    **Why it matters:** the tokenizer decides what the model can "see". It's why LLMs struggle to count the r's in "strawberry", why non-English text costs more, why arithmetic used to be unreliable, and why you're billed per token. It's also the one part of an LLM pipeline that is trained separately and frozen.

!!! watch "Watch"
    Andrej Karpathy, [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) (2 h). This chapter follows it closely, and our `bpe.py` follows his [minbpe](https://github.com/karpathy/minbpe).

    <div class="video"><iframe src="https://www.youtube-nocookie.com/embed/zduSFxRajkE" allowfullscreen loading="lazy"></iframe></div>

    Play: [Tiktokenizer](https://tiktokenizer.vercel.app/) shows how real tokenizers split any text you type.

---

## 10.1 Characters, words or subwords?

The model needs a finite vocabulary $\mathcal{V}$. The choice trades vocabulary size against sequence length.

| Unit | Vocabulary | Sequence length | Problems |
|---|---|---|---|
| Characters / bytes | tiny (256 bytes) | very long (≈4× subwords) | attention cost is $O(T^2)$; the model must learn spelling from scratch |
| Words | huge (millions), still incomplete | short | unknown words (OOV), typos, morphology ("run", "runs", "running" unrelated), no sharing across languages |
| **Subwords** | 32k–256k | medium | splits are statistical, not linguistic |

Subword tokenization keeps frequent words whole (" the", " model") and breaks rare words into reusable pieces (" token" + "ization"). With a byte-level base vocabulary, **no input is ever out-of-vocabulary**: in the worst case a string falls back to individual bytes.

## 10.2 Unicode and UTF-8

Text is a sequence of Unicode **code points**: integers from 0 to 0x10FFFF (about 150,000 are assigned to characters). Python's `ord("猫")` is 29,483 (U+732B). Using code points as the vocabulary would be too large and too sparse, so text is first encoded to **bytes** with **UTF-8**, a variable-length encoding:

| Code point range | Bytes | Bit pattern |
|---|---|---|
| U+0000 – U+007F (ASCII) | 1 | `0xxxxxxx` |
| U+0080 – U+07FF | 2 | `110xxxxx 10xxxxxx` |
| U+0800 – U+FFFF | 3 | `1110xxxx 10xxxxxx 10xxxxxx` |
| U+10000 – U+10FFFF | 4 | `11110xxx 10xxxxxx 10xxxxxx 10xxxxxx` |

Examples: `"a"` → `61`; `"é"` (U+00E9) → `C3 A9`; `"猫"` (U+732B) → `E7 8C AB`; `"🐈"` (U+1F408) → `F0 9F 90 88`.

UTF-8 is backward compatible with ASCII, self-synchronizing (continuation bytes always start with `10`, so you can find character boundaries anywhere in a stream), and dominant on the web. The cost: English is 1 byte per character, but Chinese, Japanese and Hindi take 3. A byte-level tokenizer starts from these 256 possible bytes and learns to merge them.

## 10.3 Byte-pair encoding

BPE began as a compression algorithm ([Gage, 1994](https://www.derczynski.com/papers/archive/BPE_Gage.pdf)) and was adapted for neural machine translation by [Sennrich, Haddow & Birch (2016)](https://arxiv.org/abs/1508.07909). GPT-2 made it byte-level ([Radford et al., 2019](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)).

!!! definition "Algorithm 10.1 (BPE training)"
    1. Start with the training text as a sequence of byte ids (0–255). The vocabulary is the 256 single bytes.
    2. Count every adjacent pair of ids.
    3. Take the most frequent pair $(a, b)$, create a new id $k$ (256, 257, …), record the merge $(a, b)\to k$, and replace every occurrence of $a\,b$ with $k$.
    4. Repeat steps 2–3 until the vocabulary reaches the target size.

    The output is an **ordered list of merges**. The vocabulary size is $256+\#\text{merges}$.

A worked example on `aaabdaaabac` (bytes `a`=97, `b`=98, …):

```text
merge 1: (97, 97)   -> 256 b'aa'    (4x)     aaabdaaabac -> [256]a b d [256]a b a c
merge 2: (256, 97)  -> 257 b'aaa'   (2x)     -> [257] b d [257] b a c
merge 3: (257, 98)  -> 258 b'aaab'  (2x)     -> [258] d [258] a c        11 bytes -> 5 tokens
```

<div class="widget" data-widget="bpe"></div>

The two helpers do all the work:

```python
--8<-- "code/bpe.py:helpers"
```

## 10.4 Encoding and decoding

**Decoding** is easy: concatenate each id's bytes and decode the bytes as UTF-8. A sequence of ids might end in the middle of a multi-byte character, so use `errors="replace"`, which inserts � instead of crashing.

**Encoding** new text must reproduce the training-time merges. Apply them **in the order they were learned**: repeatedly find the adjacent pair in the current sequence with the *lowest* merge index and merge it, until no learned merge applies. The full tokenizer:

```python
--8<-- "code/bpe.py:tokenizer"
```

!!! warning "Tokenization is not unique, and that matters"
    The string `" tokenization"` *could* be represented by many id sequences (all bytes, `" token"+"ization"`, …). The encoder always produces one canonical sequence, and the model only ever sees canonical ones during training. If you build token sequences by hand (for example, splitting a prompt in the middle of a word), you can feed the model a non-canonical sequence it has never seen, with odd results. This is why APIs warn against prompts that end with a trailing space: `" "` would normally be merged with the following word, so a prompt ending in a lone space token is out of distribution ([token healing](https://github.com/guidance-ai/guidance) works around this).

## 10.5 Pre-tokenization: the regex split

Run plain BPE on raw text and you get merges like `"dog."`, `"dog!"`, `"dog?"` as separate tokens, wasting vocabulary on combinations of words and punctuation. GPT-2 first **splits text into chunks with a regex** and lets merges happen only *within* chunks. Our tokenizer uses GPT-4's pattern (`cl100k_base`):

```text
'(?i:[sdmt]|ll|ve|re)          contractions: 's 't 'm 'd 'll 've 're (case-insensitive)
|[^\r\n\p{L}\p{N}]?+\p{L}+     a word, optionally preceded by one non-letter (usually a space)
|\p{N}{1,3}                    numbers, in groups of at most 3 digits
| ?[^\s\p{L}\p{N}]++[\r\n]*    punctuation runs (with an optional leading space)
|\s*[\r\n]                     newlines (with preceding whitespace)
|\s+(?!\S)                     whitespace runs, except the last space before a word
|\s+                           any remaining whitespace
```

Applied to a real string:

```text
"Hello world! It's 2024, and I've got 1234567 tokens.\n\n    def f(x):"
→ ['Hello', ' world', '!', ' It', "'s", ' ', '202', '4', ',', ' and', ' I', "'ve", ' got', ' ',
   '123', '456', '7', ' tokens', '.\n\n', '   ', ' def', ' f', '(x', '):']
```

Notice: **the space attaches to the start of the next word** (`' world'`), so `"world"` and `" world"` are different tokens; numbers split into chunks of at most three digits (`'123','456','7'`); indentation becomes runs of spaces, which GPT-4 handles far better than GPT-2 (important for code).

## 10.6 Special tokens and chat templates

Beyond merges, the vocabulary contains **special tokens** that never come from ordinary text: `<|endoftext|>` (id 50256 in GPT-2) separates documents during pretraining, and chat models add role markers. A chat conversation is serialized into one token sequence by a **chat template**, for example in the ChatML style:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4<|im_end|>
```

Llama 3 uses `<|start_header_id|>user<|end_header_id|>` … `<|eot_id|>`. The model learns the roles from these tokens in post-training (Ch. 21). Using the wrong template for an open-weights model is a common, silent cause of bad outputs. Use `tokenizer.apply_chat_template(...)` from Hugging Face instead of building the string yourself. Because special tokens are reserved, a tokenizer should never produce them from user-supplied text, or users could forge role boundaries (a prompt-injection vector, Ch. 27).

## 10.7 Other subword algorithms

- **WordPiece** (BERT; [Schuster & Nakajima, 2012](https://research.google/pubs/japanese-and-korean-voice-search/)): like BPE, but chooses the merge that most increases the training data's likelihood under a unigram model, i.e. it scores pairs by $\frac{\text{count}(ab)}{\text{count}(a)\,\text{count}(b)}$ rather than by raw count.
- **Unigram LM** ([Kudo, 2018](https://arxiv.org/abs/1804.10959)): start from a large candidate vocabulary and *prune*. Each token has a probability; a segmentation's probability is the product of its tokens'. Fit the probabilities with EM, repeatedly drop the tokens whose removal hurts the likelihood least, and at encoding time choose the most probable segmentation with the Viterbi algorithm (dynamic programming). It can also *sample* segmentations, a regularizer for training.
- **SentencePiece** ([Kudo & Richardson, 2018](https://arxiv.org/abs/1808.06226)) is a library that runs BPE or Unigram directly on raw Unicode text, with spaces replaced by `▁` so that decoding is lossless, and **byte fallback** for unseen characters. Llama 1–2, Mistral, T5 and Gemma use it. OpenAI's [tiktoken](https://github.com/openai/tiktoken) is a fast byte-level BPE *encoder* (no training), used by the GPT models and adopted by Llama 3.

## 10.8 Vocabulary size in real models

| Model | Tokenizer | Vocabulary size |
|---|---|---|
| GPT-2 / GPT-3 | byte-level BPE | 50,257 |
| Llama 1 / 2, Mistral 7B | SentencePiece BPE | 32,000 |
| GPT-4 (`cl100k_base`) | byte-level BPE (tiktoken) | ≈100,000 |
| Llama 3 | tiktoken-based BPE | 128,256 |
| GPT-4o (`o200k_base`) | byte-level BPE | ≈200,000 |
| Gemma 2 / 3 | SentencePiece | 256,000 |

The trend is toward bigger vocabularies. The trade-off:

- **Bigger $V$ ⇒ shorter sequences.** More text fits in the context window, attention is cheaper, and generation needs fewer steps. Llama 3's 128k vocabulary encodes the same English text in about 15% fewer tokens than Llama 2's 32k ([Meta, 2024](https://arxiv.org/abs/2407.21783)).
- **Bigger $V$ ⇒ bigger embedding and output matrices** ($V\times d$ each) and a costlier softmax. For a small model this is a large share of the parameters: in GPT-2 small, the $50{,}257\times768$ embedding is 38.6M of 124M parameters. Rare tokens also get few training updates.
- Tao et al., [Scaling Laws with Vocabulary](https://arxiv.org/abs/2407.13623) (2024), argue the optimal vocabulary grows with model size, and that many models use vocabularies that are too small.

**Rule of thumb for English:** one token ≈ 4 characters ≈ 0.75 words. Code and non-English text use more tokens per word.

## 10.9 Tokenization is the root of many LLM oddities

Karpathy's list, each of which you can now explain:

- **Spelling and counting letters** ("how many r's in strawberry?"): the model sees `"str" "aw" "berry"` (or similar), not letters. Knowing a token's spelling is a fact it must memorize.
- **Arithmetic**: digits are grouped into chunks (and, in GPT-2, inconsistently, e.g. `"1234"` might be one token and `"1235"` two). Splitting into ≤3-digit groups (GPT-4) or single digits (Llama 1–2) helps.
- **Non-English "token tax"**: a sentence in Hindi or Burmese can take several times more tokens than its English translation ([Petrov et al., 2023](https://arxiv.org/abs/2305.15425)), so it is slower, costs more, and uses more of the context.
- **Trailing whitespace** and partial words at the end of a prompt (§10.4).
- **Glitch tokens**: `" SolidGoldMagikarp"` was a Reddit username frequent in GPT-2's *tokenizer* training data but almost absent from the *model's* training data. Its embedding was barely trained, and prompts containing it produced bizarre behavior ([Rumbelow & Watkins, 2023](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)). Lesson: train the tokenizer on data like the model's training data.
- **Code**: whitespace handling decides whether Python indentation costs 1 token or 8.

!!! llm "Why you are billed per token"
    Compute per request is roughly proportional to tokens processed (prefill) plus tokens generated (decode), so APIs price by input and output tokens, with output usually several times more expensive (decode is memory-bound and sequential, Ch. 19). Different model families have different tokenizers, so the same text can be a different number of tokens. Count tokens with the provider's tokenizer or token-counting endpoint, not by dividing characters by four (Ch. 28).

## Key takeaways

- LLMs read **byte-level subword tokens**: frequent strings are single tokens, rare strings decompose, and nothing is out-of-vocabulary.
- UTF-8 maps code points to 1–4 bytes. The base vocabulary is the 256 bytes.
- BPE training repeatedly merges the most frequent adjacent pair. Encoding applies the learned merges in order. Decoding concatenates bytes.
- A pre-tokenization regex stops merges from crossing word, number and punctuation boundaries. Spaces attach to the following word.
- Special tokens and chat templates turn conversations into one sequence. Use the model's own template.
- Bigger vocabularies mean shorter sequences but bigger embedding/output matrices. Many LLM quirks (spelling, arithmetic, language cost) are tokenization effects.

## Exercises

1. Encode `"naïve 🤖"` to UTF-8 by hand. How many bytes is it?
2. Run BPE by hand on `"low lower lowest"` (treat spaces as ordinary characters, no regex) for 3 merges.
3. Why does BPE encoding apply merges in training order instead of greedily merging the most frequent pair in the *new* text?
4. Using `tiktoken` (`pip install tiktoken`), compare the token counts of the same paragraph in English and in another language you know, with `cl100k_base` and `o200k_base`.
5. A model has $d = 4096$. How many parameters do its embedding and (untied) output matrices have with $V = 32{,}000$ and with $V = 128{,}256$?
6. **Code:** train `BPETokenizer` on a few MB of text to a 1,024-token vocabulary. Print the 20 longest tokens. What do they tell you about the training text?

??? success "Solutions"
    1. `n a` → `6E 61`; `ï` (U+00EF) → `C3 AF`; `v e` → `76 65`; space → `20`; `🤖` (U+1F916) → `F0 9F A4 96`. Total: 2+2+2+1+4 = 11 bytes.
    2. Pair counts: `lo` 3, `ow` 3, `we` 2, `er` 1, … The most frequent is `lo` (first seen): merge 1 `l o → lo`. Then `lo w` 3 times: merge 2 `lo w → low`. Now `␣ low` and `low e` both occur twice. Our code breaks ties by first occurrence, so merge 3 is `␣ low → ␣low` (a different implementation might pick `lowe`, which is why tie-breaking must be fixed for reproducibility).
    3. The model was trained on sequences produced by the training-time merge order. Encoding must give the same segmentation for the same string regardless of the surrounding text, or the model will see unfamiliar token sequences.
    4. Typically English is the cheapest, and the gap is smaller with `o200k_base`, whose larger vocabulary covers more non-English text.
    5. Each matrix is $V\times d$: $32{,}000\times4096\approx131$M, or $128{,}256\times4096\approx525$M; double for embedding plus output. For an 8B model, that's about 13% of all parameters.
    6. Long tokens are frequent strings in the training text: common words, names, code keywords, repeated boilerplate.

## References

- R. Sennrich, B. Haddow, A. Birch, [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (ACL 2016).
- A. Radford et al., [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019), §2.2.
- T. Kudo, [Subword Regularization](https://arxiv.org/abs/1804.10959) (ACL 2018); T. Kudo & J. Richardson, [SentencePiece](https://arxiv.org/abs/1808.06226) (EMNLP 2018).
- A. Karpathy, [minbpe](https://github.com/karpathy/minbpe).
- OpenAI, [tiktoken](https://github.com/openai/tiktoken).
- A. Petrov et al., [Language Model Tokenizers Introduce Unfairness Between Languages](https://arxiv.org/abs/2305.15425) (NeurIPS 2023).
- C. Tao et al., [Scaling Laws with Vocabulary](https://arxiv.org/abs/2407.13623) (NeurIPS 2024).
