# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn

from .mdp.observation_history import ProprioHistory


class TitaVelocityEnv(ManagerBasedRLEnv):
    """TITA environment with proprioceptive observation history."""

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:

        # 缓存要在 super().__init__() 之前创建
        self.proprio_history = ProprioHistory(
            num_envs=cfg.scene.num_envs,
            obs_dim=33,
            history_length=10,
            device=cfg.sim.device,
        )

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # 此时 obs_buf 保存的仍是上一次返回的观测。
        self.proprio_history.read_then_push(
            self.obs_buf["policy"],
            self.episode_length_buf,
        )

        obs, reward, terminated, time_outs, extras = super().step(action)

        # 找到本步已经计算好的失败奖励。
        termination_index = self.reward_manager.active_terms.index("termination")
        termination_reward = (
            self.reward_manager._step_reward[:, termination_index] * self.step_dt
        )

        # 普通项合计夹到非负，再加回失败奖励。
        ordinary_reward = reward - termination_reward
        reward.copy_(ordinary_reward.clamp_min(0.0) + termination_reward)

        return obs, reward, terminated, time_outs, extras

    def _reset_idx(
        self,
        env_ids: Sequence[int] | torch.Tensor,
    ) -> None:

        super()._reset_idx(env_ids)
        self.proprio_history.reset(env_ids)
