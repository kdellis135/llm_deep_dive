"""A dependency-free retrieval core: chunking, BM25, dense cosine search, and
reciprocal-rank fusion. Swap `embed()` for a real embedding model in production."""
import math
import re
import zlib
from collections import Counter

import numpy as np


# --8<-- [start:chunk]
def chunk_text(text, max_words=120, overlap=30):
    """Fixed-size word windows with overlap, so a fact split across a boundary
    still appears whole in at least one chunk."""
    words = text.split()
    step = max_words - overlap
    return [" ".join(words[i:i + max_words]) for i in range(0, max(len(words) - overlap, 1), step)]
# --8<-- [end:chunk]


def tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())


# --8<-- [start:bm25]
class BM25:
    """Okapi BM25 (Robertson & Zaragoza, 2009), the standard lexical retriever."""
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        df = Counter(t for d in self.docs for t in set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def scores(self, query):
        q = tokenize(query)
        out = np.zeros(self.N)
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            for t in q:
                if t in tf:
                    f = tf[t]
                    norm = f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
                    out[i] += self.idf[t] * norm
        return out
# --8<-- [end:bm25]


# --8<-- [start:dense]
def embed(texts, dim=256):
    """Toy embedding via the hashing trick on word unigrams+bigrams, L2-normalised.
    Replace with a real embedding model (e.g. a sentence-transformer or an embeddings API)."""
    vecs = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        toks = tokenize(t)
        for g in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
            vecs[i, zlib.crc32(g.encode()) % dim] += 1.0   # stable hash across runs
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-12)


def dense_search(query_vec, doc_vecs, k=5):
    sims = doc_vecs @ query_vec                 # cosine similarity (vectors are unit-norm)
    top = np.argsort(-sims)[:k]
    return top, sims[top]
# --8<-- [end:dense]


# --8<-- [start:rrf]
def reciprocal_rank_fusion(rankings, k=60):
    """RRF (Cormack et al., 2009): score(d) = sum over rankers of 1 / (k + rank_r(d))."""
    scores = Counter()
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] += 1.0 / (k + rank)
    return [d for d, _ in scores.most_common()]
# --8<-- [end:rrf]
