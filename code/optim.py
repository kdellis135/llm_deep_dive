"""Optimizers from scratch, matching torch.optim semantics exactly."""
import math
import torch


# --8<-- [start:sgd]
class SGDMomentum:
    """v <- mu*v + g ;  theta <- theta - lr*v   (PyTorch's convention)."""
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params, self.lr, self.mu = list(params), lr, momentum
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self):
        for p, v in zip(self.params, self.v):
            v.mul_(self.mu).add_(p.grad)
            p.sub_(self.lr * v)
# --8<-- [end:sgd]


# --8<-- [start:adamw]
class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr, (self.b1, self.b2), self.eps, self.wd = lr, betas, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in self.params]   # 1st moment (mean of g)
        self.v = [torch.zeros_like(p) for p in self.params]   # 2nd moment (mean of g^2)
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            p.mul_(1 - self.lr * self.wd)                      # decoupled decay: shrink weights
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)         # m = b1 m + (1-b1) g
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)  # v = b2 v + (1-b2) g^2
            m_hat = m / (1 - self.b1 ** self.t)                # bias correction
            v_hat = v / (1 - self.b2 ** self.t)
            p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))
# --8<-- [end:adamw]


# --8<-- [start:schedule]
def lr_at(step, max_lr, warmup, total, min_ratio=0.1):
    """Linear warmup, then cosine decay to min_ratio * max_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return max_lr * min_ratio
    progress = (step - warmup) / (total - warmup)
    return max_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
# --8<-- [end:schedule]
