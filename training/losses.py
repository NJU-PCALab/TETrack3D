import torch
from torch.nn import functional as F


def mask_loss(logits, targets):
    return F.binary_cross_entropy_with_logits(logits, targets)


def objectness_loss(logits, targets, valid_mask):
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=logits.new_tensor([2.0]),
        reduction="none",
    )
    return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1e-6)


def box_regression_loss(predictions, targets, positive_mask):
    loss = F.smooth_l1_loss(predictions, targets, reduction="none").mean(dim=2)
    return (loss * positive_mask).sum() / positive_mask.sum().clamp_min(1e-6)


def center_regression_loss(predictions, targets, box_size, foreground_mask):
    normalized = F.mse_loss(predictions, targets, reduction="none")
    normalized = normalized / box_size.unsqueeze(1).clamp_min(1e-6)
    normalized = normalized.mean(dim=2)
    return (normalized * foreground_mask).sum() / foreground_mask.sum().clamp_min(
        1e-6
    )


def state_evolution_loss(predicted_state, observed_future_state):
    return F.smooth_l1_loss(predicted_state, observed_future_state)


def temporal_distribution_alignment_loss(
    source_tokens,
    target_tokens,
    source_weights,
    target_weights,
    epsilon=0.1,
    sinkhorn_iterations=10,
):
    source_mass = source_weights.float().clamp_min(0)
    target_mass = target_weights.float().clamp_min(0)
    valid = (source_mass.sum(dim=1) > 1e-6) & (
        target_mass.sum(dim=1) > 1e-6
    )
    if not valid.any():
        return source_tokens.new_zeros(())

    source_tokens = F.normalize(source_tokens[valid].float(), dim=-1)
    target_tokens = F.normalize(target_tokens[valid].float(), dim=-1)
    source_mass = source_mass[valid]
    target_mass = target_mass[valid]
    source_mass = source_mass / source_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
    target_mass = target_mass / target_mass.sum(dim=1, keepdim=True).clamp_min(1e-6)

    cost = 1.0 - torch.bmm(source_tokens, target_tokens.transpose(1, 2))
    kernel = torch.exp(-cost / epsilon).clamp_min(1e-8)
    source_scaling = torch.ones_like(source_mass)
    target_scaling = torch.ones_like(target_mass)
    for _ in range(sinkhorn_iterations):
        source_scaling = source_mass / torch.bmm(
            kernel, target_scaling.unsqueeze(-1)
        ).squeeze(-1).clamp_min(1e-8)
        target_scaling = target_mass / torch.bmm(
            kernel.transpose(1, 2), source_scaling.unsqueeze(-1)
        ).squeeze(-1).clamp_min(1e-8)

    transport = (
        source_scaling.unsqueeze(-1)
        * kernel
        * target_scaling.unsqueeze(1)
    )
    return (transport * cost).sum(dim=(1, 2)).mean()
