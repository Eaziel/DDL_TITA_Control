"""Gaussian action policy for TITA."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from .actor import TitaHistoryActor


class TitaGaussianPolicy(nn.Module):
    """Sample joint actions from a history-based Gaussian policy."""

    def __init__(self, init_noise_std: float = 1.0) -> None:
        """Initialize the action network and trainable standard deviations."""
        super().__init__()

        if init_noise_std <= 0.0:
            raise ValueError("init_noise_std must be positive.")

        self.actor = TitaHistoryActor()
        # 每个动作维度一个可学习标准差，shape=(8,)，所有环境共享。
        # 沿用旧TITA直接学习std的参数化方式。
        self.std = nn.Parameter(init_noise_std * torch.ones(8))
        # 保存最近一次根据观测建立的分布；创建对象时还没有对应观测。
        self._distribution: Normal | None = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action means from packed observations."""
        if obs.ndim != 2 or obs.shape[1] != 586:
            raise ValueError(f"Expected observations with shape (N, 586), got {tuple(obs.shape)}.")

        # 586列中，前33列是当前观测，最后330列是过去十帧。
        proprio = obs[:, :33]
        history = obs[:, -330:].reshape(obs.shape[0], 10, 33)
        return self.actor(proprio, history)  # (N, 8)，动作均值。

    def update_distribution(self, obs: torch.Tensor) -> None:
        """Build the Gaussian distribution for the given observations."""
        mean = self(obs)  # (N, 8)，调用本类forward。
        # (8,) -> (N, 8)：各环境使用同一组std，但各自具有不同的均值。
        scale = self.std.expand_as(mean)
        self._distribution = Normal(loc=mean, scale=scale, validate_args=True)

    def _get_distribution(self) -> Normal:
        """Return the most recently built action distribution."""
        if self._distribution is None:
            raise RuntimeError("Call update_distribution() or act() first.")
        return self._distribution

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample actions for environment interaction."""
        self.update_distribution(obs)
        # a = mean + std * epsilon，epsilon的各分量独立服从N(0, 1)。
        return self._get_distribution().sample()  # (N, 8)。

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action means without sampling."""
        return self(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return joint action log densities under the current distribution."""
        # 给定观测时，8个动作分量独立；联合密度的对数是8项之和。
        return self._get_distribution().log_prob(actions).sum(dim=-1)  # (N,)。

    @property
    def action_mean(self) -> torch.Tensor:
        """Return the current action means."""
        return self._get_distribution().mean  # (N, 8)。

    @property
    def action_std(self) -> torch.Tensor:
        """Return the current action standard deviations."""
        return self._get_distribution().stddev  # (N, 8)。

    @property
    def entropy(self) -> torch.Tensor:
        """Return the summed action entropy for each environment."""
        return self._get_distribution().entropy().sum(dim=-1)  # (N,)。
