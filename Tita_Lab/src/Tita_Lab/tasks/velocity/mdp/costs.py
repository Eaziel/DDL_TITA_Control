# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg
    from isaaclab.sensors import ContactSensor


def joint_position_limit_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Flag environments with a joint outside its soft position limits."""

    asset: Articulation = env.scene[asset_cfg.name]

    # 当前关节角(N,J)，单位rad；J由asset_cfg选中的关节决定。
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    # (N,J,2)，最后一维依次为软下限、软上限。
    limits = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, :]

    out_of_limits = (joint_pos < limits[..., 0]) | (joint_pos > limits[..., 1])
    # 逐行检查：任一关节越界即为True，再转成(N,)的0/1。
    return torch.any(out_of_limits, dim=1).float()


def joint_effort_limit_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """Flag environments whose applied joint effort exceeds the threshold."""

    asset: Articulation = env.scene[asset_cfg.name]

    # 限幅后的驱动力矩(N,J)，与旧Gym的self.torques对应。
    torques = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    # 求解器力矩上限(N,J)，当前腿60、轮15 N·m。
    limits = asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids]

    # soft_ratio是阈值比例；旧值为1.0。
    return torch.any(torch.abs(torques) > limits * soft_ratio, dim=1).float()


def joint_velocity_limit_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """Flag environments whose joint speed exceeds the threshold."""

    asset: Articulation = env.scene[asset_cfg.name]

    # 当前关节角速度(N,J)，单位rad/s。
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    # 求解器关节速度上限(N,J)，当前腿25、轮20 rad/s。
    limits = asset.data.joint_vel_limits.torch[:, asset_cfg.joint_ids]

    return torch.any(torch.abs(joint_vel) > limits * soft_ratio, dim=1).float()


def wheel_contact_force_cost(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500.0,
) -> torch.Tensor:
    """Flag environments with excessive wheel normal contact force."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 选中轮体的当前世界系法向力，(N,W,3)，单位N；本任务W=2。
    forces_w = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :]

    # 每轮力的大小(N,W)，任一轮超过阈值则该环境为1。
    forces_magnitudes = torch.linalg.vector_norm(forces_w, dim=-1)
    return torch.any(forces_magnitudes > threshold, dim=1).float()


def wheel_stumble_cost(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    horizontal_ratio: float = 5.0,
) -> torch.Tensor:
    """Flag environments with excessive horizontal-to-vertical wheel contact force."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :]

    # 每轮世界水平力sqrt(Fx²+Fy²)，以及竖直力abs(Fz)，均为(N,W)。
    horizontal_force = torch.linalg.vector_norm(forces_w[..., :2], dim=-1)
    vertical_force = torch.abs(forces_w[..., 2])

    # 旧规则：水平力超过竖直力的5倍，记一次侧向碰撞判断。
    stumble = horizontal_force > horizontal_ratio * vertical_force
    return torch.any(stumble, dim=1).float()


class JointAccelerationCost(ManagerTermBase):
    """Measure joint acceleration above the legacy TITA threshold."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]

        # 上一次策略步结束时的关节速度，(N, J)，单位rad/s。
        self._previous_joint_vel = self._asset.data.joint_vel.torch.clone()

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        joint_vel = self._asset.data.joint_vel.torch
        joint_ids = asset_cfg.joint_ids

        # 每0.01 s差分一次，得到(N, J_selected)的rad/s²。
        acceleration = (joint_vel[:, joint_ids] - self._previous_joint_vel[:, joint_ids]) / env.step_dt
        acceleration_limit = self._asset.data.joint_vel_limits.torch[:, joint_ids] / (2.0 * env.step_dt)

        excess = (acceleration.abs() - acceleration_limit).clamp_min(0.0)
        self._previous_joint_vel.copy_(joint_vel)

        # 每台机器人一个值，(N,)；此处尚未乘dt。
        return excess.mean(dim=1)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Use reset joint velocities as the next acceleration reference."""

        selection = slice(None) if env_ids is None else env_ids

        # 在机器人reset之后调用，只更新被重置机器人的速度参考。
        self._previous_joint_vel[selection, :] = self._asset.data.joint_vel.torch[selection, :]
