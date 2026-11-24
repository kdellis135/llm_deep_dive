"""LoRA (Hu et al., 2021): freeze W, learn a low-rank update  W' = W + (alpha/r) * B A."""
import math
import torch
import torch.nn as nn


# --8<-- [start:lora]
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                       # frozen pretrained weights
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.empty(r, d_in))       # down-projection
        self.B = nn.Parameter(torch.zeros(d_out, r))      # up-projection, ZERO init
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # same as nn.Linear's default
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
        self.merged = False

    def forward(self, x):
        y = self.base(x)
        if not self.merged:   # (x A^T) B^T has cost r(d_in + d_out) per token, not d_in*d_out
            y = y + self.scaling * (self.drop(x) @ self.A.T @ self.B.T)
        return y

    @torch.no_grad()
    def merge(self):
        """Fold the update into W for zero-overhead inference."""
        if not self.merged:
            self.base.weight += self.scaling * (self.B @ self.A)
            self.merged = True


def add_lora(model: nn.Module, target_names=("q_proj", "v_proj", "wq", "wv"), r=8, alpha=16):
    """Replace every nn.Linear whose attribute name is in target_names with a LoRALinear."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_names:
            setattr(model, name, LoRALinear(module, r=r, alpha=alpha))
        else:
            add_lora(module, target_names, r, alpha)
    return model
# --8<-- [end:lora]
