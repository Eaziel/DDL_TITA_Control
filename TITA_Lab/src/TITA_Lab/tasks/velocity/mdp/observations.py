# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from ..tita_velocity_env import TitaVelocityEnv

def joint_pos_rel_zero_wheels(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        wheel_cfg: SceneEntityCfg,
)->torch.Tensor:
    """Return joint position offsets [rad], with wheel entries zeroed."""
    # 从 Scene 取得机器人
    robot: Articulation = env.scene[asset_cfg.name]
    # 计算全部关节的 当前位置－默认位置。减法生成新张量
    joint_pos_rel = robot.data.joint_pos.torch - robot.data.default_joint_pos.torch
    # 在新张量里，把两个轮子的位置偏差置零
    joint_pos_rel[:, wheel_cfg.joint_ids] = 0.0
    # 按指定的八关节顺序返回
    return joint_pos_rel[:, asset_cfg.joint_ids]

# MDP 观测函数，把历史交给 ObservationManager
def proprio_history(env: TitaVelocityEnv) -> torch.Tensor:
    """Return past proprioceptive frames in whole-frame order."""

    # 展开为长度10*33=330的向量
    return env.proprio_history.read().reshape(env.num_envs, -1)
