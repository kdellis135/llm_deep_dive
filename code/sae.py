"""A sparse autoencoder (dictionary learning) for interpretability, plus a toy dataset of
sparse features stored in superposition (more features than dimensions)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --8<-- [start:toy]
def superposition_data(n=4096, n_features=64, d=16, p_active=0.03, seed=0):
    """Each sample activates a few of n_features ground-truth features; features are random
    unit directions in R^d with d << n_features, so they must share dimensions."""
    g = torch.Generator().manual_seed(seed)
    dirs = F.normalize(torch.randn(n_features, d, generator=g), dim=1)        # (F, d)
    active = (torch.rand(n, n_features, generator=g) < p_active).float()
    coeffs = active * torch.rand(n, n_features, generator=g)                   # sparse, >= 0
    return coeffs @ dirs, dirs                                                 # x: (n, d)
# --8<-- [end:toy]


# --8<-- [start:sae]
class SparseAutoencoder(nn.Module):
    """f = ReLU(W_enc (x - b_dec) + b_enc);  x_hat = W_dec f + b_dec.
    Loss = ||x - x_hat||^2 + lam * sum_i |f_i| * ||W_dec[:, i]||   (Anthropic, 2023-24)."""
    def __init__(self, d, n_dict):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(n_dict, d) * 0.1)
        self.b_enc = nn.Parameter(torch.zeros(n_dict))
        self.W_dec = nn.Parameter(torch.randn(d, n_dict) * 0.1)
        self.b_dec = nn.Parameter(torch.zeros(d))

    def forward(self, x):
        f = F.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)    # (n, n_dict) sparse codes
        return f @ self.W_dec.T + self.b_dec, f

    def loss(self, x, lam=0.3):
        x_hat, f = self(x)
        mse = (x - x_hat).pow(2).sum(-1).mean()
        sparsity = (f * self.W_dec.norm(dim=0)).sum(-1).mean()       # L1, scaled by decoder norms
        return mse + lam * sparsity, mse, (f > 0).float().sum(-1).mean()   # also report L0


def train_sae(x, n_dict=128, steps=3000, lam=0.3, lr=3e-3, seed=0):
    torch.manual_seed(seed)
    sae = SparseAutoencoder(x.shape[1], n_dict)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    for _ in range(steps):
        loss, mse, l0 = sae.loss(x, lam)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return sae, mse.item(), l0.item()
# --8<-- [end:sae]


def feature_recovery(sae, true_dirs, thresh=0.9):
    """Fraction of ground-truth features matched (cosine > thresh) by some decoder column."""
    dec = F.normalize(sae.W_dec.detach(), dim=0)                     # (d, n_dict)
    cos = F.normalize(true_dirs, dim=1) @ dec                        # (F, n_dict)
    return (cos.max(dim=1).values > thresh).float().mean().item()


if __name__ == "__main__":
    x, dirs = superposition_data()
    sae, mse, l0 = train_sae(x)
    print(f"mse {mse:.4f}  (signal power {x.pow(2).sum(-1).mean():.4f})  mean L0 {l0:.2f}")
    print("features recovered:", feature_recovery(sae, dirs))
