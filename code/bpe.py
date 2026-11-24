"""Byte-level BPE tokenizer (the algorithm behind GPT-2/3/4 and Llama-3 tokenizers).

Follows the structure of Karpathy's minbpe: train() learns merges on UTF-8 bytes,
encode() applies them greedily in the order they were learned, decode() inverts.
"""
import regex as re   # `pip install regex` — supports \p{L} (Unicode letters)

# GPT-4 (cl100k_base) pre-tokenization pattern: split text into chunks so that
# merges never cross word / number / punctuation boundaries.
GPT4_SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)


# --8<-- [start:helpers]
def get_stats(ids, counts=None):
    """Count adjacent pairs: [1, 2, 3, 1, 2] -> {(1,2): 2, (2,3): 1, (3,1): 1}."""
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    """Replace every occurrence of `pair` in `ids` by the new token `idx`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out
# --8<-- [end:helpers]


# --8<-- [start:tokenizer]
class BPETokenizer:
    def __init__(self, pattern=GPT4_SPLIT_PATTERN):
        self.pattern = re.compile(pattern) if pattern else None
        self.merges = {}                                   # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}   # int -> bytes

    def _chunks(self, text):
        return self.pattern.findall(text) if self.pattern else [text]

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        chunks = [list(c.encode("utf-8")) for c in self._chunks(text)]
        for idx in range(256, vocab_size):
            stats = {}
            for ids in chunks:                     # pairs never span two chunks
                get_stats(ids, stats)
            if not stats:
                break
            pair = max(stats, key=stats.get)       # most frequent pair
            chunks = [merge(ids, pair, idx) for ids in chunks]
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose:
                print(f"merge {idx - 255}: {pair} -> {idx} {self.vocab[idx]!r} ({stats[pair]}x)")

    def _encode_chunk(self, ids):
        while len(ids) >= 2:
            stats = get_stats(ids)
            # apply the EARLIEST-learned merge available (lowest new id)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        out = []
        for c in self._chunks(text):
            out.extend(self._encode_chunk(list(c.encode("utf-8"))))
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
# --8<-- [end:tokenizer]


if __name__ == "__main__":
    text = "the cat sat on the mat. the cat ate the rat! 猫は可愛い 🐈 1234567"
    tok = BPETokenizer()
    tok.train(text * 20, 300, verbose=False)
    ids = tok.encode(text)
    print(len(text.encode()), "bytes ->", len(ids), "tokens")
    print([tok.vocab[i] for i in ids][:12])
    assert tok.decode(ids) == text
