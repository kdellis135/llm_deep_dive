"""Character-level language models before Transformers (in the spirit of Karpathy's makemore):
a counting bigram model, the same model as a one-layer neural net, and Bengio et al.'s (2003) MLP LM."""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --8<-- [start:data]
def build_vocab(text):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return stoi, {i: c for c, i in stoi.items()}
# --8<-- [end:data]


# --8<-- [start:bigram_counts]
def bigram_counts(ids, V, alpha=1.0):
    """MLE of p(next | current) is the normalized count table (Thm 3.8).
    alpha > 0 adds pseudo-counts (add-alpha smoothing = MAP with a Dirichlet prior)."""
    N = torch.full((V, V), alpha)
    for a, b in zip(ids, ids[1:]):
        N[a, b] += 1
    return N / N.sum(dim=1, keepdim=True)             # row a is the distribution p(. | a)


def nll(P, ids):
    """Average negative log-likelihood (nats per character) of a sequence under P."""
    x, y = torch.tensor(ids[:-1]), torch.tensor(ids[1:])
    return -P[x, y].log().mean().item()


def sample_bigram(P, start, n, generator=None):
    out = [start]
    for _ in range(n):
        out.append(torch.multinomial(P[out[-1]], 1, generator=generator).item())
    return out
# --8<-- [end:bigram_counts]


# --8<-- [start:bigram_nn]
def train_neural_bigram(ids, V, steps=500, lr=10.0):
    """The same model as a neural net: logits = one_hot(x) @ W, i.e. row W[x].
    Gradient descent on cross-entropy converges to log of the (unsmoothed) count table."""
    x, y = torch.tensor(ids[:-1]), torch.tensor(ids[1:])
    W = torch.zeros(V, V, requires_grad=True)
    for _ in range(steps):
        loss = F.cross_entropy(W[x], y)                # W[x] == F.one_hot(x, V).float() @ W
        W.grad = None
        loss.backward()
        with torch.no_grad():
            W -= lr * W.grad
    return W.detach(), loss.item()
# --8<-- [end:bigram_nn]


# --8<-- [start:mlp_lm]
class MLPLM(nn.Module):
    """Bengio et al. (2003): embed the last `context` tokens, concatenate, tanh hidden layer, softmax."""
    def __init__(self, V, context=3, d_emb=10, d_hidden=200):
        super().__init__()
        self.context = context
        self.emb = nn.Embedding(V, d_emb)                    # C: (V, d_emb) lookup table
        self.hidden = nn.Linear(context * d_emb, d_hidden)
        self.out = nn.Linear(d_hidden, V)

    def forward(self, ctx):                                   # ctx: (B, context) token ids
        e = self.emb(ctx).flatten(1)                          # (B, context*d_emb)
        return self.out(torch.tanh(self.hidden(e)))          # (B, V) logits


def make_windows(ids, context, pad):
    """Every position is a training example: (previous `context` ids, next id)."""
    padded = [pad] * context + ids
    X = torch.tensor([padded[i:i + context] for i in range(len(ids))])
    return X, torch.tensor(ids)


def train_mlp_lm(ids, V, context=3, steps=2000, seed=0):
    torch.manual_seed(seed)
    X, Y = make_windows(ids, context, pad=0)
    model = MLPLM(V, context)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        ix = torch.randint(0, len(X), (64,))
        loss = F.cross_entropy(model(X[ix]), Y[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return F.cross_entropy(model(X), Y).item(), model
# --8<-- [end:mlp_lm]


if __name__ == "__main__":
    from gpt import TEXT
    stoi, itos = build_vocab(TEXT)
    ids = [stoi[c] for c in TEXT]
    V = len(stoi)
    P = bigram_counts(ids, V, alpha=0.0 + 1e-9)
    print("count bigram NLL", round(nll(P, ids), 4))
    W, l = train_neural_bigram(ids, V)
    print("neural bigram NLL", round(l, 4))
    l3, _ = train_mlp_lm(ids, V)
    print("MLP LM NLL", round(l3, 4))
    g = torch.Generator().manual_seed(0)
    print("".join(itos[i] for i in sample_bigram(bigram_counts(ids, V), stoi["F"], 80, g)))
