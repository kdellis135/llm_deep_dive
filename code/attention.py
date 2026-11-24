"""Attention from scratch: single head, causal masking, multi-head, and the
online-softmax / tiling trick behind FlashAttention (in NumPy).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --8<-- [start:sdpa]
def attention(q, k, v, causal=True):
    """Scaled dot-product attention.
    q, k, v: (..., T, d_h).  Returns (..., T, d_h) and the attention weights (..., T, T).
    """
    d_h = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_h)        # (..., T, T): q_i · k_j / sqrt(d_h)
    if causal:
        T = q.size(-2)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
        scores = scores.masked_fill(~mask, float("-inf"))     # position i may not see j > i
    weights = F.softmax(scores, dim=-1)                       # each row sums to 1
    return weights @ v, weights                               # weighted average of values
# --8<-- [end:sdpa]


# --8<-- [start:mha]
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_h = n_head, d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)   # W_Q, W_K, W_V for all heads, fused
        self.proj = nn.Linear(d_model, d_model)      # W_O

    def forward(self, x):                            # x: (B, T, C) with C = d_model
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)        # each (B, T, C)
        # split C into n_head heads of size d_h, move heads next to batch
        q = q.view(B, T, self.n_head, self.d_h).transpose(1, 2)   # (B, H, T, d_h)
        k = k.view(B, T, self.n_head, self.d_h).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_h).transpose(1, 2)
        y, _ = attention(q, k, v, causal=True)       # (B, H, T, d_h): heads run in parallel
        y = y.transpose(1, 2).contiguous().view(B, T, C)          # concatenate heads
        return self.proj(y)                          # mix information across heads
# --8<-- [end:mha]


# --8<-- [start:online]
def attention_tiled_numpy(q, k, v, block=4):
    """Exact (non-causal) attention computed one key/value block at a time with an
    ONLINE softmax: we never materialise the full T x T score matrix.
    q: (Tq, d), k, v: (Tk, d).  This is the core idea of FlashAttention.
    """
    Tq, d = q.shape
    m = np.full(Tq, -np.inf)          # running max of scores per query row
    l = np.zeros(Tq)                  # running sum of exp(score - m)
    acc = np.zeros((Tq, v.shape[1]))  # running (unnormalised) weighted sum of values
    for s in range(0, k.shape[0], block):
        kb, vb = k[s:s + block], v[s:s + block]
        S = q @ kb.T / math.sqrt(d)                    # (Tq, block) scores for this tile
        m_new = np.maximum(m, S.max(axis=1))
        scale = np.exp(m - m_new)                      # rescale old statistics to the new max
        P = np.exp(S - m_new[:, None])
        l = l * scale + P.sum(axis=1)
        acc = acc * scale[:, None] + P @ vb
        m = m_new
    return acc / l[:, None]
# --8<-- [end:online]


# --8<-- [start:backward]
def attention_backward(dy, q, k, v, causal=True):
    """Manual backward pass of attention (single head). dy, q, k, v: (T, d_h).
    Forward:  S = q k^T / sqrt(d_h),  A = softmax(S) row-wise,  y = A v."""
    d_h = q.size(-1)
    S = q @ k.T / math.sqrt(d_h)
    if causal:
        T = q.size(0)
        S = S.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
    A = torch.softmax(S, dim=-1)
    dv = A.T @ dy                                        # y = A v      ->  dv = A^T dy
    dA = dy @ v.T                                        #              ->  dA = dy v^T
    dS = A * (dA - (dA * A).sum(-1, keepdim=True))       # softmax VJP per row (Sec. 7.5)
    dq = dS @ k / math.sqrt(d_h)                         # S = q k^T/sqrt(d_h)
    dk = dS.T @ q / math.sqrt(d_h)
    return dq, dk, dv
# --8<-- [end:backward]


if __name__ == "__main__":
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 8, 16) for _ in range(3))
    y, w = attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    print("matches PyTorch SDPA:", torch.allclose(y, ref, atol=1e-6))
    qn, kn, vn = (np.random.randn(10, 8) for _ in range(3))
    S = qn @ kn.T / math.sqrt(8)
    P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    print("tiled == naive:", np.allclose(attention_tiled_numpy(qn, kn, vn, block=3), P @ vn))
