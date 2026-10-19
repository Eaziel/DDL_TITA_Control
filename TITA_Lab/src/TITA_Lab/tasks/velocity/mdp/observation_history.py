# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import torch


class ProprioHistory:
    def __init__(
        self,
        num_envs: int,
        obs_dim: int = 33,
        history_length: int = 10,
        device: str | torch.device = "cpu",
    ) -> None:

        if num_envs <= 0 or obs_dim <= 0 or history_length <= 0:
            raise ValueError("缓存三个维度必须为正整数.")

        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.history_length = history_length

        self._history = torch.zeros(
            (num_envs, history_length, obs_dim),
            dtype=torch.float32,
            device=device,
        )

    def read_then_push(
        self,
        current_obs: torch.Tensor,
        episode_length: torch.Tensor,
    ) -> torch.Tensor:

        if current_obs.shape != (self.num_envs, self.obs_dim):
            raise ValueError("current_obs 的形状不匹配.")
        if episode_length.shape != (self.num_envs,):
            raise ValueError("episode_length 的形状不匹配.")

        # 保存本次要返回的旧历史
        past = self._history.clone()

        # 普通步：丢掉最旧帧，追加当前帧
        shifted = torch.cat(
            (self._history[:, 1:, :], current_obs.unsqueeze(1)),  # unsq 加时间轴
            dim=1,
        )

        # 旧 Gym 的 开局 规则：用当前帧填满历史
        repeated_current = current_obs.unsqueeze(1).expand(
            -1,
            self.history_length,
            -1,  # (N, 1, 33)  --expand(-1, 10, -1)-->  (N, 10, 33)
        )
        self._history = torch.where(  # torch.where(条件, A, B)
            (episode_length <= 1)[:, None, None],  # 刚开局（第0、1步）。`[:, None, None]`：每环境一个 bool，广播用
            repeated_current,  # 刚开局，填满
            shifted,
        )

        # 先读取再push新的观测，这里返回的是push新观测前的
        return past

    # 历史缓存“只读取，不更新”。
    # Lab 可能为了确定观测维度、记录数据等原因，多次调用观测函数。
    # 读取一次观测，不应当等于时间前进了一步。因此，读取历史和写入新帧必须分开。
    def read(self) -> torch.Tensor:
        """Return a copy of the stored history without updating it."""

        return self._history.clone()

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:
            self._history.zero_()
        else:
            self._history[env_ids, :, :] = 0.0
