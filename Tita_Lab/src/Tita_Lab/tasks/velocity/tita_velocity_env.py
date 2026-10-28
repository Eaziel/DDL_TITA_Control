# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn

from .cost_manager import TitaCostManager
from .mdp.observation_history import ProprioHistory

if TYPE_CHECKING:
    from .config.Tita.env_cfg import VelocityEnvCfg


class TitaVelocityEnv(ManagerBasedRLEnv):
    """TITA environment with observation history and constraint costs."""

    def __init__(
        self,
        cfg: VelocityEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize task histories and cost collection."""

        # 父类建立动作/观测管理器时会读取这两份数据，故先创建。
        self.proprio_history = ProprioHistory(
            num_envs=cfg.scene.num_envs,
            obs_dim=33,
            history_length=10,
            device=cfg.sim.device,
        )
        # Shared history-slot indices, shape (N,); sampled by a startup event.
        self.action_lag_indices = torch.zeros(cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device)

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # 此时机器人和物理数据已建立，可以解析cost中的关节/轮体。
        self.cost_manager = TitaCostManager(self.cfg.costs, self)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Advance the environment and return task rewards and costs."""

        # 此时 obs_buf 保存的仍是上一次返回的观测。
        self.proprio_history.read_then_push(
            self.obs_buf["policy"],
            self.episode_length_buf,
        )

        obs, reward, terminated, time_outs, extras = super().step(action)

        # 找到本步已经计算好的失败奖励。
        termination_index = self.reward_manager.active_terms.index("termination")
        termination_reward = self.reward_manager._step_reward[:, termination_index] * self.step_dt

        # 普通项合计夹到非负，再加回失败奖励。
        ordinary_reward = reward - termination_reward
        reward.copy_(ordinary_reward.clamp_min(0.0) + termination_reward)

        # 无reset时在这里计算；有reset时返回reset前保存的本步值。
        # clone让本次返回的矩阵不被下一步覆盖，形状(N, 6)。
        extras["costs"] = self.cost_manager.compute().clone()

        # 检查本步返回的cost：每台机器人六项，数值有限且非负。
        costs = extras["costs"]
        assert costs.shape == (self.num_envs, 6)
        assert torch.isfinite(costs).all(), "Non-finite cost detected."
        assert (costs >= 0.0).all(), "Negative cost detected."

        return obs, reward, terminated, time_outs, extras

    def _reset_idx(
        self,
        env_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Reset selected robots and their task histories."""

        # 重置物理状态前，保留刚结束这一步的六维观测样本。
        self.observation_manager.compute_group("privileged_state")

        # 保存失败瞬间的cost，供本步step返回。
        self.cost_manager.compute()

        super()._reset_idx(env_ids)
        self.proprio_history.reset(env_ids)

        # 父类创建本轮log后，再加入cost统计并更新加速度差分基准。
        self.extras["log"].update(self.cost_manager.reset(env_ids))
