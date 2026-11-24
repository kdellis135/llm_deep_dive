"""Decoding strategies operating on a (B, V) tensor of next-token logits."""
import torch
import torch.nn.functional as F


# --8<-- [start:sampling]
def apply_temperature(logits, T):
    return logits / max(T, 1e-8)                 # T<1 sharpens, T>1 flattens


def top_k_filter(logits, k):
    """Keep the k largest logits, set the rest to -inf."""
    kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits, p):
    """Nucleus sampling (Holtzman et al., 2019): keep the smallest set of tokens
    whose cumulative probability >= p."""
    sorted_logits, idx = torch.sort(logits, descending=True, dim=-1)
    cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    remove = cum > p
    remove[..., 1:] = remove[..., :-1].clone()   # shift right: keep the token that crosses p
    remove[..., 0] = False                       # always keep the most likely token
    mask = remove.scatter(-1, idx, remove)       # back to vocabulary order
    return logits.masked_fill(mask, float("-inf"))


def min_p_filter(logits, min_p):
    """min-p sampling: drop tokens whose prob < min_p * (max prob)."""
    probs = F.softmax(logits, dim=-1)
    thresh = min_p * probs.max(dim=-1, keepdim=True).values
    return logits.masked_fill(probs < thresh, float("-inf"))


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, min_p=None, generator=None):
    if temperature == 0:                         # greedy decoding
        return logits.argmax(dim=-1, keepdim=True)
    logits = apply_temperature(logits, temperature)
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    if min_p is not None:
        logits = min_p_filter(logits, min_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)
# --8<-- [end:sampling]


if __name__ == "__main__":
    z = torch.tensor([[2.0, 1.0, 0.5, 0.0, -1.0]])
    print(F.softmax(z, -1))
    print("top-k=2:", F.softmax(top_k_filter(z, 2), -1))
    print("top-p=0.8:", F.softmax(top_p_filter(z, 0.8), -1))
