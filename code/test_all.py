"""Run every check behind the code shown on the site:  python code/test_all.py
Requires: torch, numpy, regex.
"""
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252

PASSED = []


def check(name, cond):
    assert cond, f"FAILED: {name}"
    PASSED.append(name)
    print(f"  ok  {name}")


def test_micrograd():
    from micrograd import Value, train_demo
    # compare against PyTorch on a non-trivial expression
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g = g + 10.0 / f
    h = (g.tanh() * 0.0 + g.log() + a.exp() * 0.01)
    h.backward()
    at, bt = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True), torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    ct = at + bt
    dt = at * bt + bt ** 3
    ct = ct + ct + 1
    ct = ct + 1 + ct + (-at)
    dt = dt + dt * 2 + (bt + at).relu()
    dt = dt + 3 * dt + (bt - at).relu()
    et = ct - dt
    ft = et ** 2
    gt = ft / 2.0
    gt = gt + 10.0 / ft
    ht = (gt.tanh() * 0.0 + gt.log() + at.exp() * 0.01)
    ht.backward()
    check("micrograd forward == torch", abs(h.data - ht.item()) < 1e-9)
    check("micrograd grad a == torch", abs(a.grad - at.grad.item()) < 1e-9)
    check("micrograd grad b == torch", abs(b.grad - bt.grad.item()) < 1e-9)
    loss, _ = train_demo(verbose=False)
    check("micrograd MLP trains (loss < 0.01)", loss < 0.01)


def test_backprop():
    from backprop import mlp_forward_backward, layernorm_backward
    torch.manual_seed(0)
    X, y = torch.randn(32, 10), torch.randint(0, 5, (32,))
    P = dict(W1=torch.randn(64, 10) / 10 ** 0.5, b1=torch.zeros(64), W2=torch.randn(5, 64) / 8, b2=torch.zeros(5))
    for v in P.values():
        v.requires_grad_()
    loss, grads = mlp_forward_backward(X, y, **{k: v.detach() for k, v in P.items()})
    F.cross_entropy(torch.tanh(X @ P["W1"].T + P["b1"]) @ P["W2"].T + P["b2"], y).backward()
    check("manual MLP backprop == autograd", all(torch.allclose(grads[k], P[k].grad, atol=1e-6) for k in P))
    x, g, b = torch.randn(8, 16, requires_grad=True), torch.randn(16, requires_grad=True), torch.randn(16, requires_grad=True)
    dy = torch.randn(8, 16)
    F.layer_norm(x, (16,), g, b, 1e-5).backward(dy)
    dx, dg, db = layernorm_backward(dy, x.detach(), g.detach())
    check("manual LayerNorm backward == autograd", torch.allclose(dx, x.grad, atol=1e-5) and torch.allclose(dg, g.grad, atol=1e-5) and torch.allclose(db, b.grad, atol=1e-5))


def test_ngram():
    from gpt import TEXT
    from ngram import build_vocab, bigram_counts, nll, train_neural_bigram, train_mlp_lm
    stoi, _ = build_vocab(TEXT)
    ids = [stoi[c] for c in TEXT]
    V = len(stoi)
    P = bigram_counts(ids, V, alpha=1e-9)
    check("bigram rows are distributions", torch.allclose(P.sum(1), torch.ones(V)))
    _, l_nn = train_neural_bigram(ids, V, steps=1500)
    check("neural bigram converges to count-based MLE", abs(l_nn - nll(P, ids)) < 0.02)
    l_mlp, _ = train_mlp_lm(ids, V, steps=1500)
    check("MLP LM (3-char context) beats bigram", l_mlp < nll(P, ids))


def test_bpe():
    from bpe import BPETokenizer, get_stats, merge
    check("get_stats", get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1})
    check("merge", merge([1, 2, 3, 1, 2], (1, 2), 4) == [4, 3, 4])
    text = "hello world! hello there, world. 你好世界 🌍 12345 don't stop"
    tok = BPETokenizer()
    tok.train(text * 10, 290)
    ids = tok.encode(text)
    check("BPE roundtrip", tok.decode(ids) == text)
    check("BPE compresses", len(ids) < len(text.encode("utf-8")))
    plain = BPETokenizer(pattern=None)
    plain.train("aaabdaaabac", 259)
    check("BPE classic example: 3 merges", len(plain.merges) == 3 and plain.decode(plain.encode("aaabdaaabac")) == "aaabdaaabac")


def test_attention():
    from attention import attention, MultiHeadAttention, attention_tiled_numpy
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 7, 16) for _ in range(3))
    y, w = attention(q, k, v)
    check("attention == torch SDPA", torch.allclose(y, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5))
    check("attention rows sum to 1", torch.allclose(w.sum(-1), torch.ones(2, 3, 7)))
    check("causal: upper triangle is zero", torch.all(w.triu(1) == 0).item())
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(2, 5, 32)
    out = mha(x)
    x2 = x.clone(); x2[:, -1] += 1.0                    # perturb the LAST token
    check("MHA causal: earlier outputs unaffected by later tokens", torch.allclose(out[:, :-1], mha(x2)[:, :-1], atol=1e-6))
    rng = np.random.default_rng(0)
    qn, kn, vn = rng.normal(size=(9, 8)), rng.normal(size=(13, 8)), rng.normal(size=(13, 5))
    S = qn @ kn.T / math.sqrt(8)
    P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    check("online-softmax tiled attention is exact", np.allclose(attention_tiled_numpy(qn, kn, vn, block=4), P @ vn))
    from attention import attention_backward
    qa, ka, va = (torch.randn(6, 8, requires_grad=True) for _ in range(3))
    dy = torch.randn(6, 8)
    attention(qa, ka, va)[0].backward(dy)
    dq, dk, dv = attention_backward(dy, qa.detach(), ka.detach(), va.detach())
    check("manual attention backward == autograd", all(torch.allclose(a, b.grad, atol=1e-5) for a, b in [(dq, qa), (dk, ka), (dv, va)]))


def test_gpt():
    from gpt import GPT, GPTConfig, count_params, train_char_level
    m = GPT(GPTConfig())
    n = sum(p.numel() for p in m.parameters())
    check(f"GPT-2 small total params = {n:,} (~124M)", 124_000_000 < n < 124_600_000)
    small = GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32))
    idx = torch.randint(0, 50, (3, 16))
    logits, loss = small(idx, torch.randint(0, 50, (3, 16)))   # random targets
    check("GPT logits shape (B,T,V)", logits.shape == (3, 16, 50))
    check("GPT init loss ≈ ln(V)", abs(loss.item() - math.log(50)) < 0.3)
    loss, sample = train_char_level(steps=300, verbose=False)
    check(f"GPT char-level training reduces loss (final {loss:.3f} < 1.0)", loss < 1.0)


def test_llama():
    from llama import Llama, LlamaConfig, rope_cache, apply_rope, RMSNorm
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, dim=64, n_layers=2, n_heads=8, n_kv_heads=2, ffn_dim=172, max_seq_len=64)
    m = Llama(cfg).eval()
    prompt = torch.randint(0, 100, (2, 5))
    fast = m.generate(prompt, 12)
    slow = prompt
    for _ in range(12):
        slow = torch.cat([slow, m(slow)[:, -1:].argmax(-1)], dim=1)
    check("KV-cache decoding == full recompute", torch.equal(fast, slow))
    # RoPE relative-position property: <R_m q, R_n k> depends only on m - n
    cos, sin = rope_cache(16, 200)
    q, k = torch.randn(1, 1, 1, 16), torch.randn(1, 1, 1, 16)
    def score(mp, npos):
        return (apply_rope(q, cos[mp:mp + 1], sin[mp:mp + 1]) * apply_rope(k, cos[npos:npos + 1], sin[npos:npos + 1])).sum().item()
    check("RoPE score depends only on m - n", abs(score(10, 3) - score(110, 103)) < 1e-4)
    check("RoPE preserves norm", torch.allclose(apply_rope(q, cos[7:8], sin[7:8]).norm(), q.norm(), atol=1e-5))
    x = torch.randn(4, 64) * 5
    y = RMSNorm(64)(x)
    check("RMSNorm output has unit RMS", torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-3))


def test_moe():
    from moe import MoE
    torch.manual_seed(0)
    m = MoE(dim=32, hidden=64, n_experts=4, top_k=2)
    x = torch.randn(2, 5, 32)
    y, aux = m(x)
    check("MoE output shape", y.shape == x.shape)
    one = MoE(dim=32, hidden=64, n_experts=1, top_k=1)
    check("MoE with 1 expert == that expert's FFN", torch.allclose(one(x)[0], one.experts[0](x), atol=1e-6))
    with torch.no_grad():
        m.router.weight.zero_()                      # uniform router probabilities
    _, aux_u = m(x)
    check("balance loss is 1 for uniform router probs", abs(aux_u.item() - 1.0) < 1e-5)
    y.sum().backward()
    check("router receives gradient", m.router.weight.grad is not None and m.router.weight.grad.abs().sum() > 0)


def test_sampling():
    from sampling import top_k_filter, top_p_filter, min_p_filter, sample_next
    z = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])
    check("top-k keeps k", torch.isfinite(top_k_filter(z, 2)).sum().item() == 2)
    p = F.softmax(z, -1)[0]          # [0.553, 0.203, 0.123, 0.075, 0.027]
    kept = torch.isfinite(top_p_filter(z, 0.7)).sum().item()
    check("top-p=0.7 keeps 2 tokens (0.553 < 0.7 <= 0.756)", kept == 2)
    check("top-p=1.0 keeps all", torch.isfinite(top_p_filter(z, 1.0)).sum().item() == 5)
    check("min-p=0.3 keeps tokens with p >= 0.3*pmax", torch.isfinite(min_p_filter(z, 0.3)).sum().item() == int((p >= 0.3 * p.max()).sum()))
    check("temperature 0 is greedy", sample_next(z, temperature=0).item() == 0)


def test_optim():
    from optim import AdamW, SGDMomentum, lr_at
    torch.manual_seed(0)
    for Mine, Ref, kw in [(AdamW, torch.optim.AdamW, dict(lr=1e-2, weight_decay=0.1)),
                          (SGDMomentum, torch.optim.SGD, dict(lr=1e-2, momentum=0.9))]:
        w1 = torch.randn(5, 3, requires_grad=True)
        w2 = w1.detach().clone().requires_grad_(True)
        o1, o2 = Mine([w1], **kw), Ref([w2], **kw)
        for _ in range(20):
            for w, o in ((w1, o1), (w2, o2)):
                w.grad = None
                ((w ** 2).sum() + w.sin().sum()).backward()
                o.step()
        check(f"{Mine.__name__} matches torch.optim.{Ref.__name__}", torch.allclose(w1, w2, atol=1e-6))
    check("lr schedule warmup/cosine endpoints", abs(lr_at(99, 1.0, 100, 1000) - 1.0) < 1e-9 and abs(lr_at(1000, 1.0, 100, 1000) - 0.1) < 1e-9)


def test_lora():
    from lora import LoRALinear, add_lora
    torch.manual_seed(0)
    base = nn.Linear(32, 16)
    x = torch.randn(4, 32)
    ref = base(x).detach()
    lora = LoRALinear(base, r=4, alpha=8)
    check("LoRA is a no-op at init (B = 0)", torch.allclose(lora(x), ref))
    with torch.no_grad():
        lora.B.normal_()
    before = lora(x).detach()
    lora.merge()
    check("LoRA merge gives identical outputs", torch.allclose(lora(x), before, atol=1e-5))
    trainable = sum(p.numel() for p in LoRALinear(nn.Linear(4096, 4096), r=16).parameters() if p.requires_grad)
    check("LoRA trainable params r(d_in+d_out)", trainable == 16 * (4096 + 4096))
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq, self.wv, self.wo = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8)
    t = add_lora(Toy())
    check("add_lora wraps targets only", isinstance(t.wq, LoRALinear) and isinstance(t.wv, LoRALinear) and isinstance(t.wo, nn.Linear))


def test_alignment():
    from alignment import dpo_loss, reward_model_loss, grpo_advantages, ppo_clip_loss, kl_k3, sequence_logprob
    lp = torch.randn(8)
    loss, _ = dpo_loss(lp, lp.clone(), lp.clone(), lp.clone())
    check("DPO loss at policy == reference is ln 2", abs(loss.item() - math.log(2)) < 1e-6)
    check("RM loss at equal rewards is ln 2", abs(reward_model_loss(torch.zeros(4), torch.zeros(4)).item() - math.log(2)) < 1e-6)
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    a = grpo_advantages(r)
    check("GRPO advantages are zero-mean per group", torch.allclose(a.mean(1), torch.zeros(2), atol=1e-6))
    adv = torch.tensor([1.0, -1.0])
    check("PPO at ratio 1 equals -mean(A)", abs(ppo_clip_loss(torch.zeros(2), torch.zeros(2), adv).item() - 0.0) < 1e-6)
    lp1, lp2 = torch.randn(1000), torch.randn(1000)
    check("k3 KL estimator is non-negative", torch.all(kl_k3(lp1, lp2) >= 0).item())
    logits = torch.randn(2, 6, 11)
    toks = torch.randint(0, 11, (2, 6))
    mask = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    manual = sum(F.log_softmax(logits[0, t - 1], -1)[toks[0, t]] for t in range(3, 6))
    check("sequence log-prob sums response tokens only", torch.allclose(sequence_logprob(logits, toks, mask)[0], manual, atol=1e-5))


def test_sae():
    from sae import superposition_data, train_sae, feature_recovery
    x, dirs = superposition_data()
    sae, mse, l0 = train_sae(x)
    power = x.pow(2).sum(-1).mean().item()
    check(f"SAE codes are sparse (mean L0 {l0:.1f} of 128)", l0 < 8)
    check("SAE reconstructs most of the signal", mse < 0.25 * power)
    check("SAE recovers >= 90% of 64 features stored in 16 dims", feature_recovery(sae, dirs) >= 0.9)


def test_quant():
    from quant import quantize_absmax_int8, quantize_zeropoint, quantize_groupwise
    torch.manual_seed(0)
    w = torch.randn(256, 512) * 0.02
    q, s = quantize_absmax_int8(w)
    err8 = (q.float() * s - w).abs().max().item()
    check("int8 absmax error <= s/2", err8 <= s.item() / 2 + 1e-9)
    qz, sz, z = quantize_zeropoint(w)
    check("zero-point int8 roundtrip", (sz * (qz - z) - w).abs().max().item() <= sz.item() / 2 + 1e-6)
    w[0, 0] = 1.0                                   # one big outlier
    e_g = (quantize_groupwise(w, 4, 32) - w).pow(2).mean().item()
    e_t = (quantize_groupwise(w, 4, 512) - w).pow(2).mean().item()
    check("small groups beat per-row scales with an outlier", e_g < e_t)


def test_rag():
    from rag import BM25, embed, dense_search, reciprocal_rank_fusion, chunk_text
    docs = ["The KV cache stores keys and values for past tokens.",
            "LoRA adds a low-rank update to frozen weights.",
            "BM25 is a lexical ranking function based on term frequency.",
            "Rotary embeddings rotate query and key pairs by position."]
    bm = BM25(docs)
    check("BM25 finds lexical match", int(np.argmax(bm.scores("low-rank LoRA update"))) == 1)
    vecs = embed(docs)
    top, _ = dense_search(embed(["what does the KV cache store"])[0], vecs, k=1)
    check("dense search finds KV doc", top[0] == 0)
    check("RRF fuses", reciprocal_rank_fusion([[1, 2, 3], [2, 1, 3]])[:2] in ([1, 2], [2, 1]))
    words = " ".join(f"w{i}" for i in range(250))
    ch = chunk_text(words, 100, 20)
    check("chunks cover every word", set(" ".join(ch).split()) == set(words.split()))


if __name__ == "__main__":
    random.seed(0)
    for t in [test_micrograd, test_backprop, test_ngram, test_bpe, test_attention, test_gpt, test_llama, test_moe, test_sampling,
              test_optim, test_lora, test_alignment, test_quant, test_sae, test_rag]:
        print(t.__name__)
        t()
    print(f"\nAll {len(PASSED)} checks passed.")
