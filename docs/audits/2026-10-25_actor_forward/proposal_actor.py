"""History-based action network for TITA."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: Sequence[int],
    *,
    layer_norm: bool = False,
) -> nn.Sequential:
    """Build an ELU multilayer perceptron with a linear output."""
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for index, hidden_dim in enumerate(hidden_dims):
        # 每层先做线性变换 Wx+b，再经过ELU非线性变换。
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        # 沿用旧工厂的准确位置：第一隐层不加LayerNorm。
        if layer_norm and index > 0:
            layers.append(nn.LayerNorm(hidden_dim))
        previous_dim = hidden_dim

    # 输出层不加激活函数，动作均值不在这里裁剪。
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class TitaHistoryActor(nn.Module):
    """Predict action means from current and historical observations."""

    def __init__(
        self,
        num_proprio: int = 33,
        history_length: int = 10,
        num_actions: int = 8,
    ) -> None:
        """Initialize the history encoder and action network."""
        super().__init__()
        self.num_proprio = num_proprio
        self.history_length = history_length

        # 旧网络的26个中间输出：10个辅助预测量，加16个学习得到的特征。
        # 本步只迁前向；辅助预测的监督损失后续接入。
        self.prediction_dim = 10
        self.latent_dim = 16
        encoded_dim = self.prediction_dim + self.latent_dim

        # 默认维度：330 -> 512 -> 256 -> 128 -> 26。
        self.mlp_encoder = _build_mlp(
            num_proprio * history_length,
            encoded_dim,
            (512, 256, 128),
            layer_norm=True,
        )
        # 默认维度：26+33=59 -> 512 -> 256 -> 128 -> 8。
        self.actor = _build_mlp(
            encoded_dim + num_proprio,
            num_actions,
            (512, 256, 128),
        )

    def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        """Return the action means for a batch of observations."""
        # obs为(N,33)，obs_hist为(N,10,33)，后者不含当前帧。
        # 丢掉最老一帧，补入当前帧；得到最近九帧历史加当前帧。
        history_with_current = torch.cat(
            (obs_hist[:, 1:, :], obs.unsqueeze(1)), dim=1
        )
        batch_size = obs.shape[0]
        history_flat = history_with_current.reshape(batch_size, -1)  # (N,330)。
        features = self.mlp_encoder(history_flat)  # (N,26)。

        # 顺序沿用旧网络：26维编码在前，33维当前观测在后。
        actor_input = torch.cat((features, obs), dim=-1)  # (N,59)。
        return self.actor(actor_input)  # (N,8)，动作分布的均值。
