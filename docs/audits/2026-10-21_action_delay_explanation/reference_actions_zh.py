# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.envs.mdp.actions import JointAction, JointActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from TITA_Lab.tasks.velocity.tita_velocity_env import TitaVelocityEnv


class TitaJointAction(JointAction):
    """Joint commands with filtering and delay."""

    cfg: TitaJointActionCfg

    def __init__(self, cfg: TitaJointActionCfg, env: TitaVelocityEnv) -> None:
        """Initialize joint command buffers."""
        super().__init__(cfg, env)
        if cfg.command_type not in ("position", "effort"):
            raise ValueError(f"Unsupported TITA command type: {cfg.command_type}")
        if cfg.lag_timesteps < 1:
            raise ValueError("lag_timesteps must be at least one.")

        # 两份滤波用的动作记录都是(N, J)；腿有6个关节，轮有2个。
        self.clipped_actions = torch.zeros_like(self.raw_actions)
        self.previous_clipped_actions = torch.zeros_like(self.raw_actions)
        if cfg.command_type == "position":
            self._offset = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()

        # 腿和轮都引用环境中的同一个(N,)张量，启动事件会填写其中的选择编号。
        self._lag_indices = env.action_lag_indices
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        # 保存的目标：腿是关节角(rad)，轮是前馈力矩(N·m)。
        # 初始时，腿的三个位置都填默认角，轮的三个位置都填零前馈。
        self._initial_command = (
            self._offset.clone() if cfg.command_type == "position" else torch.zeros_like(self.raw_actions)
        )
        self._lag_buffer = self._initial_command.unsqueeze(1).repeat(1, cfg.lag_timesteps, 1)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Update filtered joint commands from policy actions."""
        self.previous_clipped_actions.copy_(self.clipped_actions)
        self.clipped_actions.copy_(actions.clamp(-self.cfg.raw_clip, self.cfg.raw_clip))

        # 每策略步滤波一次：u_t=0.2*c_(t-1)+0.8*c_t，c为裁剪后的动作。
        previous_weight = self.cfg.previous_weight
        filtered = previous_weight * self.previous_clipped_actions + (1.0 - previous_weight) * self.clipped_actions
        super().process_actions(filtered)
        self._raw_actions.copy_(actions)

    def apply_actions(self) -> None:
        """Apply delayed joint commands."""
        # 每物理步更新一次；形状(N, L, J)，第二维按从旧到新排列。
        self._lag_buffer[:, :-1, :] = self._lag_buffer[:, 1:, :].clone()
        self._lag_buffer[:, -1, :] = self.processed_actions
        delayed = self._lag_buffer[self._env_ids, self._lag_indices, :]

        if self.cfg.command_type == "position":
            self._asset.set_joint_position_target_index(target=delayed, joint_ids=self._joint_ids)
        else:
            self._asset.set_joint_effort_target_index(target=delayed, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Restore initial command histories for the selected robots."""
        if isinstance(env_ids, Sequence):
            env_ids = list(env_ids)
        super().reset(env_ids)
        selection = slice(None) if env_ids is None else env_ids
        self._lag_buffer[selection, :, :] = self._initial_command[selection, :].unsqueeze(1)
        # 保留滤波所需的动作记录，以及每台机器人之前抽到的选择编号。


@configclass
class TitaJointActionCfg(JointActionCfg):
    """Configuration for filtered and delayed TITA joint commands."""

    class_type: type[TitaJointAction] = TitaJointAction
    command_type: Literal["position", "effort"] = "position"
    raw_clip: float = 100.0
    previous_weight: float = 0.2
    lag_timesteps: int = 1  # 保留最近几次物理步的目标；1表示只保留当前目标，没有延迟。
