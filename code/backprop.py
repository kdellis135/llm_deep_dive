"""Backpropagation by hand for a 2-layer MLP and for LayerNorm, checked against autograd."""
import torch
import torch.nn.functional as F


# --8<-- [start:mlp]
def mlp_forward_backward(X, y, W1, b1, W2, b2):
    """Forward + manual backward for  logits = tanh(X W1^T + b1) W2^T + b2,  mean cross-entropy.
    X: (B, n_in), y: (B,) class ids, W1: (n_h, n_in), W2: (K, n_h). Returns loss and grads."""
    B = X.shape[0]
    # ---- forward, keeping every intermediate we will need ----
    Z1 = X @ W1.T + b1                   # (B, n_h)  pre-activations
    H1 = torch.tanh(Z1)                  # (B, n_h)  hidden activations
    Z2 = H1 @ W2.T + b2                  # (B, K)    logits
    loss = F.cross_entropy(Z2, y)
    # ---- backward: one vector-Jacobian product per forward line, in reverse ----
    dZ2 = F.softmax(Z2, dim=1)           # dL/dZ2 = (p - y) / B   (Thm 2.8, mean over batch)
    dZ2[torch.arange(B), y] -= 1
    dZ2 /= B
    dW2 = dZ2.T @ H1                     # (K, n_h)   same shape as W2
    db2 = dZ2.sum(0)                     # (K,)       bias was broadcast over B -> sum over B
    dH1 = dZ2 @ W2                       # (B, n_h)   gradient flows back through W2^T
    dZ1 = dH1 * (1 - H1 ** 2)            # (B, n_h)   tanh'(z) = 1 - tanh(z)^2
    dW1 = dZ1.T @ X                      # (n_h, n_in)
    db1 = dZ1.sum(0)                     # (n_h,)
    return loss, dict(W1=dW1, b1=db1, W2=dW2, b2=db2)
# --8<-- [end:mlp]


# --8<-- [start:layernorm]
def layernorm_backward(dy, x, gamma, eps=1e-5):
    """Gradients of y = gamma * (x - mean) / sqrt(var + eps) + beta over the last dim."""
    mu = x.mean(-1, keepdim=True)
    rstd = (x.var(-1, unbiased=False, keepdim=True) + eps).rsqrt()
    xhat = (x - mu) * rstd
    g = dy * gamma                                         # gradient w.r.t. xhat
    dx = rstd * (g - g.mean(-1, keepdim=True) - xhat * (g * xhat).mean(-1, keepdim=True))
    dgamma = (dy * xhat).reshape(-1, x.shape[-1]).sum(0)
    dbeta = dy.reshape(-1, x.shape[-1]).sum(0)
    return dx, dgamma, dbeta
# --8<-- [end:layernorm]
