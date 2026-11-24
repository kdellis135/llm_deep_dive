"""A modern (Llama-3-style) decoder: RMSNorm, RoPE, grouped-query attention,
SwiGLU feed-forward, no biases, untied embeddings, and a KV cache for decoding.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LlamaConfig:
    vocab_size: int = 128256
    dim: int = 4096           # d_model
    n_layers: int = 32
    n_heads: int = 32         # query heads H
    n_kv_heads: int = 8       # key/value heads H_kv  (GQA: H_kv < H; MQA: H_kv = 1)
    ffn_dim: int = 14336      # SwiGLU hidden size
    rope_theta: float = 500000.0
    norm_eps: float = 1e-5
    max_seq_len: int = 8192


# --8<-- [start:rmsnorm]
class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g   — LayerNorm without mean-centering or bias."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms_inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms_inv).type_as(x) * self.weight
# --8<-- [end:rmsnorm]


# --8<-- [start:rope]
def rope_cache(head_dim, max_pos, theta=10000.0, device=None):
    """cos/sin tables of shape (max_pos, head_dim/2) for angles m * theta_i,
    theta_i = theta^(-2i/head_dim), i = 0..head_dim/2-1."""
    inv_freq = theta ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    pos = torch.arange(max_pos, device=device).float()
    ang = torch.outer(pos, inv_freq)                       # (max_pos, head_dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """Rotate consecutive pairs (x[2i], x[2i+1]) by angle m*theta_i.
    x: (B, H, T, d_h); cos, sin: (T, d_h/2) for the positions of x."""
    x1, x2 = x[..., 0::2], x[..., 1::2]                    # even / odd components
    out1 = x1 * cos - x2 * sin                             # 2-D rotation matrix
    out2 = x1 * sin + x2 * cos
    return torch.stack([out1, out2], dim=-1).flatten(-2)   # interleave back
# --8<-- [end:rope]


# --8<-- [start:gqa]
class Attention(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.h, self.h_kv = cfg.n_heads, cfg.n_kv_heads
        self.d_h = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, self.h * self.d_h, bias=False)
        self.wk = nn.Linear(cfg.dim, self.h_kv * self.d_h, bias=False)   # fewer K heads
        self.wv = nn.Linear(cfg.dim, self.h_kv * self.d_h, bias=False)   # fewer V heads
        self.wo = nn.Linear(self.h * self.d_h, cfg.dim, bias=False)
        self.cache_k = self.cache_v = None

    def forward(self, x, cos, sin, start_pos=0, use_cache=False):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.d_h).transpose(1, 2)       # (B, H,    T, d_h)
        k = self.wk(x).view(B, T, self.h_kv, self.d_h).transpose(1, 2)    # (B, H_kv, T, d_h)
        v = self.wv(x).view(B, T, self.h_kv, self.d_h).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)          # positions enter here only
        if use_cache:                     # append this step's K, V to everything seen so far
            if start_pos == 0:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v
        # GQA: each group of H/H_kv query heads shares one K/V head
        rep = self.h // self.h_kv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        S = k.size(2)                                                     # total keys
        # causal mask that also works when queries are the LAST T of S positions
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))
# --8<-- [end:gqa]


# --8<-- [start:swiglu]
class SwiGLU(nn.Module):
    """FFN(x) = W2 (SiLU(W1 x) ⊙ W3 x)   (Shazeer, 2020)."""
    def __init__(self, dim, hidden):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, dim, bias=False)   # down

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
# --8<-- [end:swiglu]


# --8<-- [start:model]
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg.dim, cfg.ffn_dim)

    def forward(self, x, cos, sin, start_pos=0, use_cache=False):
        x = x + self.attn(self.attn_norm(x), cos, sin, start_pos, use_cache)
        return x + self.ffn(self.ffn_norm(x))


class Llama(nn.Module):
    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)   # untied
        cos, sin = rope_cache(cfg.dim // cfg.n_heads, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, idx, start_pos=0, use_cache=False):
        T = idx.size(1)
        cos, sin = self.cos[start_pos:start_pos + T], self.sin[start_pos:start_pos + T]
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x, cos, sin, start_pos, use_cache)
        return self.output(self.norm(x))                                # (B, T, V) logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Greedy decoding with a KV cache: one PREFILL pass over the prompt, then one
        cheap DECODE pass per new token that only processes the newest token."""
        logits = self(idx, start_pos=0, use_cache=True)                  # prefill
        out = idx
        for _ in range(max_new_tokens):
            nxt = logits[:, -1:].argmax(-1)                              # (B, 1)
            out = torch.cat([out, nxt], dim=1)
            logits = self(nxt, start_pos=out.size(1) - 1, use_cache=True)  # decode step
        return out
# --8<-- [end:model]


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, dim=64, n_layers=2, n_heads=8, n_kv_heads=2, ffn_dim=172, max_seq_len=64)
    m = Llama(cfg).eval()
    prompt = torch.randint(0, 100, (2, 5))
    fast = m.generate(prompt, 10)
    slow = prompt
    for _ in range(10):  # recompute everything each step, no cache
        slow = torch.cat([slow, m(slow)[:, -1:].argmax(-1)], dim=1)
    print("KV-cache decoding matches full recompute:", torch.equal(fast, slow))
