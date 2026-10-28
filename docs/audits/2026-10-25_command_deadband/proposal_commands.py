# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class TitaVelocityCommand(UniformVelocityCommand):
    """Velocity commands with the legacy TITA planar speed deadband."""

    cfg: TitaVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Sample velocity commands with a planar speed deadband."""
        super()._resample_command(env_ids)

        # 命令缓冲为(N, 3)，每行[vx, vy, wz]；这里取本次重采样的M行。
        vel_xy = self.vel_command_b[env_ids, :2]  # (M, 2)，单位m/s。
        speed_xy = torch.linalg.vector_norm(vel_xy, dim=1)  # (M,)。

        # 沿用旧Gym：速度模长大于阈值（默认0.2 m/s）才保留平移命令。
        moving = speed_xy > self.cfg.lin_vel_deadband  # (M,)，bool。

        # (M, 1)广播到两列；将处理后的结果写回原始命令缓冲。
        self.vel_command_b[env_ids, :2] = vel_xy * moving.unsqueeze(1)


@configclass
class TitaVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for TITA velocity commands."""

    class_type: type[TitaVelocityCommand] = TitaVelocityCommand
    lin_vel_deadband: float = 0.2
    """Planar command speeds at or below this threshold are zeroed [m/s]."""
