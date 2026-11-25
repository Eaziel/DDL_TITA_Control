"""Reward and cost value networks for TITA."""

from __future__ import annotations

import torch
from torch import nn

from .actor import _build_mlp


class TitaValueNetworks(nn.Module):
    """Estimate reward and cost returns from packed observations."""

    def __init__(self):
        """Initialize observation encoders and value heads."""

        super().__init__()

        # 扫描：187 -> 128 -> 64 -> 32。
        # 旧mlp_factory末层已有ELU，last_act=True又追加一次；保持原计算。
        self.scan_encoder = nn.Sequential(
            nn.Linear(187, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.ELU()
        )

        # 每帧33个数映射为30个特征；十帧共用同一套参数。
        self.history_frame_encoder = nn.Sequential(
            nn.Linear(33, 30),
            nn.ELU(),
        )

        # 沿时间轴处理历史：(N,30,10) -> (N,20,4) -> (N,10,3)。
        self.history_temporal_encoder = nn.Sequential(
            nn.Conv1d(30, 20, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv1d(20, 10, kernel_size=2, stride=1),
            nn.ELU(),
            nn.Flatten(start_dim=1),  # (N,10,3) -> (N,30)。
        )
        self.history_output = nn.Sequential(
            nn.Linear(30, 32),
            nn.ELU(),
        )

        # 当前33 + 仿真状态/参数36 + 扫描编码32 + 历史编码32 = 133。
        # 两个输出网络各自有参数，共享上面的观测编码器。
        self.reward_head = _build_mlp(133, 1, (512, 256, 128))
        self.cost_head = nn.Sequential(
            _build_mlp(133, 6, (512, 256, 128)),
            nn.Softplus(),  # 六类cost回报非负，保持旧网络的输出约束。
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reward and cost value estimates for each environment."""

        if obs.ndim != 2 or obs.shape[1] != 586:
            raise ValueError(
                f"Expected observations with shape (N, 586), got {tuple(obs.shape)}."
            )

        batch_size = obs.shape[0]
        proprio = obs[:, :33]  # (N,33)，当前观测。
        priviliged = obs[:, 220:256]  # (N,36)，六维状态加30维动力学参数。
        scan_features = self.scan_encoder(obs[:, 33:220])  # (N,32)。

        # 价值网络使用原样过去十帧，不追加当前帧，沿用旧critic。
        history = obs[:, 256:].reshape(batch_size, 10, 33)
        frame_features = self.history_frame_encoder(
            history.reshape(batch_size * 10, 33)
        )  # (N*10,30)。

        # Conv1d要求顺序为：批量、通道、序列长度。
        temporal_input = frame_features.reshape(batch_size, 10, 30).transpose(1, 2)
        temporal_features = self.history_temporal_encoder(temporal_input)  # (N,30)。
        history_features = self.history_output(temporal_features)  # (N,32)。

        # 拼接顺序与旧evaluate/evaluate_cost一致。
        features = torch.cat(
            (proprio, priviliged, scan_features, history_features), dim=-1,
        )  # (N,133)。

        reward_values = self.reward_head(features)  # (N,1)，未来累计奖励估计。
        cost_values = self.cost_head(features)  # (N,6)，六类未来累计cost估计。
        return reward_values, cost_values
