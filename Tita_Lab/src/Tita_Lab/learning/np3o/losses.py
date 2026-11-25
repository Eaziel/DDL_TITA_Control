"""Compute the losses used to train TITA's critic and policy."""

from __future__ import annotations

import torch

from .critic import TitaValueNetworks


def compute_value_losses(
        *,
        value_networks: TitaValueNetworks,
        observations: torch.Tensor,
        reward_returns: torch.Tensor,
        cost_returns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reward and cost value prediction losses.

    Args:
        value_networks: Current critic.
        observations: Saved observations, shape (B, 586).
        reward_returns: Fixed reward value targets, shape (B, 1).
        cost_returns: Fixed cost value targets, shape (B, C).

    Returns:
        Reward and cost mean squared errors, in that order.
    """

    # 用当前critic重新预测，再把两种预测分别与固定目标比较。
    predictions: tuple[torch.Tensor, torch.Tensor] = value_networks(observations)
    reward_values, cost_values = predictions
    reward_loss: torch.Tensor = (reward_values - reward_returns).square().mean()
    cost_loss: torch.Tensor = (cost_values - cost_returns).square().mean()
    return reward_loss, cost_loss


def compute_policy_losses(
        *,
        ratio: torch.Tensor,
        normalized_advantages: torch.Tensor,
        normalized_cost_advantages: torch.Tensor,
        cost_offset: torch.Tensor,
        k: torch.Tensor,
        clip_param: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the clipped reward loss, cost penalty, and per-cost violations.

    Args:
        ratio: New-to-old action density ratios, shape (B, 1).
        normalized_advantages: Normalized reward advantages, shape (B, 1).
        normalized_cost_advantages: Normalized cost advantages, shape (B, C).
        cost_offset: Constraint offsets in normalized units, shape (C,).
        k: Weights for positive violations, shape (C,).
        clip_param: Half width of the ratio clipping interval around 1.

    Returns:
        Two scalar losses and the C constraint violation values before ReLU.
    """

    clipped_ratio: torch.Tensor = ratio.clamp(
        min=1.0 - clip_param,
        max=1.0 + clip_param,
    )
    reward_loss: torch.Tensor = -torch.minimum(
        ratio * normalized_advantages,
        clipped_ratio * normalized_advantages,
    ).mean()

    # 每一种cost分别对小批量里的样本求平均。
    cost_surrogate: torch.Tensor = torch.maximum(
        ratio * normalized_cost_advantages,
        clipped_ratio * normalized_cost_advantages,
    ).mean(dim=0)

    # 每种约束只惩罚正的违反量，再按k加权求和。
    violation: torch.Tensor = cost_surrogate + cost_offset
    cost_loss: torch.Tensor = (k * torch.relu(violation)).sum()

    return reward_loss, cost_loss, violation

        