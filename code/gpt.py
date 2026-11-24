"""A complete GPT-2-style decoder-only Transformer, written to be read top to bottom.

Shapes use the nanoGPT convention: B = batch, T = sequence length, C = d_model
(embedding width), V = vocabulary size, H = number of heads, d_h = C // H.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --8<-- [start:config]
@dataclass
class GPTConfig:
    vocab_size: int = 50257   # V: GPT-2 BPE vocabulary
    block_size: int = 1024    # T_max: context window
    n_layer: int = 12         # L
    n_head: int = 12          # H
    n_embd: int = 768         # C = d_model
    dropout: float = 0.0
    bias: bool = True
# --8<-- [end:config]


# --8<-- [start:attn]
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)  # fused W_Q, W_K, W_V
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)      # W_O
        self.c_proj.RESIDUAL_SCALE = True    # flag for the GPT-2 init below
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, H, T, d_h)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # softmax(q k^T / sqrt(d_h) + causal mask) v, fused (FlashAttention kernel when available)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                  # concat heads
        return self.c_proj(y)
# --8<-- [end:attn]


# --8<-- [start:mlp]
class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, project back."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.gelu = nn.GELU(approximate="tanh")       # GPT-2 used the tanh approximation
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.c_proj.RESIDUAL_SCALE = True
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.gelu(self.c_fc(x))))
# --8<-- [end:mlp]


# --8<-- [start:block]
class Block(nn.Module):
    """Pre-norm residual block:  x = x + Attn(LN(x));  x = x + MLP(LN(x))."""
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # tokens communicate
        x = x + self.mlp(self.ln_2(x))    # each token computes on its own
        return x
# --8<-- [end:block]


# --8<-- [start:gpt]
class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),    # token embeddings  (V, C)
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),    # learned positions (T_max, C)
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight    # weight tying (Press & Wolf, 2017)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = 0.02
            if getattr(m, "RESIDUAL_SCALE", False):
                std *= (2 * self.cfg.n_layer) ** -0.5      # 2 residual adds per layer
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        pos = torch.arange(T, device=idx.device)                       # (T,)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)      # (B, T, C)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)                                               # residual stream
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)                                       # (B, T, V)
        loss = None
        if targets is not None:   # mean over B*T positions of -log p(next token)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]                   # crop to context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature                    # only last position
            if top_k is not None:
                kth = torch.topk(logits, top_k).values[:, [-1]]
                logits[logits < kth] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)              # sample
            idx = torch.cat([idx, nxt], dim=1)
        return idx
# --8<-- [end:gpt]


# --8<-- [start:params]
def count_params(model, non_embedding=True):
    n = sum(p.numel() for p in model.parameters())        # tied weights counted once
    if non_embedding:
        n -= model.transformer.wpe.weight.numel()
    return n
# --8<-- [end:params]


# --8<-- [start:train]
TEXT = """First Citizen:
Before we proceed any further, hear me speak.
All:
Speak, speak.
First Citizen:
You are all resolved rather to die than to famish?
All:
Resolved. resolved.
First Citizen:
First, you know Caius Marcius is chief enemy to the people.
"""


def train_char_level(steps=300, device="cpu", seed=0, verbose=True):
    torch.manual_seed(seed)
    chars = sorted(set(TEXT))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)

    cfg = GPTConfig(vocab_size=len(chars), block_size=64, n_layer=2, n_head=4, n_embd=64)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)

    def batch(bs=16):
        ix = torch.randint(len(data) - cfg.block_size - 1, (bs,))
        x = torch.stack([data[i:i + cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])   # shifted by one
        return x.to(device), y.to(device)

    for step in range(steps):
        x, y = batch()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if verbose and step % 100 == 0:
            print(f"step {step:4d} | loss {loss.item():.4f} | ln(V) = {math.log(len(chars)):.4f}")
    start = torch.tensor([[stoi["F"]]], device=device)
    sample = "".join(itos[i] for i in model.generate(start, 120, temperature=0.8)[0].tolist())
    return loss.item(), sample
# --8<-- [end:train]


if __name__ == "__main__":
    m = GPT(GPTConfig())
    print(f"GPT-2 small: {count_params(m) / 1e6:.1f}M non-embedding-position params")
    final, sample = train_char_level()
    print("final loss", round(final, 4))
    print(sample)
