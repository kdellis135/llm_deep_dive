"""A minimal mixture-of-experts (MoE) feed-forward layer with top-k routing and the
Switch-Transformer load-balancing loss."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from llama import SwiGLU


# --8<-- [start:moe]
class MoE(nn.Module):
    def __init__(self, dim, hidden, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(dim, n_experts, bias=False)          # one logit per expert
        self.experts = nn.ModuleList([SwiGLU(dim, hidden) for _ in range(n_experts)])

    def forward(self, x):                                            # x: (B, T, C)
        B, T, C = x.shape
        x = x.reshape(-1, C)                                         # (N, C) tokens, N = B*T
        probs = F.softmax(self.router(x).float(), dim=-1)            # (N, E) router distribution
        weight, idx = probs.topk(self.top_k, dim=-1)                 # (N, k) chosen experts
        weight = (weight / weight.sum(-1, keepdim=True)).type_as(x)  # renormalize over the k chosen
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):                    # loop over experts, not tokens
            token_ids, slot = (idx == e).nonzero(as_tuple=True)      # tokens routed to expert e
            if token_ids.numel():
                out.index_add_(0, token_ids, weight[token_ids, slot, None] * expert(x[token_ids]))
        # Switch-style balance loss: E * sum_e f_e * P_e  (= 1 under perfectly uniform routing)
        f = F.one_hot(idx, self.n_experts).float().sum(1).mean(0) / self.top_k   # fraction routed
        P = probs.mean(0)                                                          # mean router prob
        aux_loss = self.n_experts * (f * P).sum()
        return out.reshape(B, T, C), aux_loss
# --8<-- [end:moe]
