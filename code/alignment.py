"""Losses used in post-training: reward modeling, PPO, DPO, GRPO."""
import torch
import torch.nn.functional as F


# --8<-- [start:logprobs]
def sequence_logprob(logits, tokens, mask):
    """Sum of log p(token_t | tokens_<t) over response positions.
    logits: (B, T, V) predicting tokens[:, 1:]; tokens: (B, T); mask: (B, T) 1 on response tokens.
    """
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)                    # (B, T-1, V)
    tok_logp = logp.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)    # (B, T-1)
    return (tok_logp * mask[:, 1:]).sum(-1)                                 # (B,)
# --8<-- [end:logprobs]


# --8<-- [start:rm]
def reward_model_loss(r_chosen, r_rejected):
    """Bradley-Terry: P(chosen > rejected) = sigmoid(r_c - r_r); minimise its NLL."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()
# --8<-- [end:rm]


# --8<-- [start:dpo]
def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta=0.1):
    """Direct Preference Optimization (Rafailov et al., 2023).
    All inputs are sequence log-probs (B,). The implicit reward is beta * log(pi/ref)."""
    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()
    accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, accuracy
# --8<-- [end:dpo]


# --8<-- [start:ppo]
def ppo_clip_loss(logp_new, logp_old, advantages, clip_eps=0.2):
    """PPO clipped surrogate (Schulman et al., 2017), per token. Returns a loss to MINIMISE."""
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    return -torch.min(unclipped, clipped).mean()
# --8<-- [end:ppo]


# --8<-- [start:grpo]
def grpo_advantages(rewards, eps=1e-6):
    """GRPO (Shao et al., 2024): sample G responses per prompt, use the group's own
    statistics as the baseline instead of a learned value function.
    rewards: (num_prompts, G) -> advantages of the same shape."""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def kl_k3(logp, logp_ref):
    """Unbiased, always-positive per-token KL estimator used by GRPO:
    KL ≈ exp(ref - logp) - (ref - logp) - 1."""
    d = logp_ref - logp
    return torch.exp(d) - d - 1
# --8<-- [end:grpo]
