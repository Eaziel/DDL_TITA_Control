# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def undesired_contacts_current(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Count selected bodies exceeding the current normal contact-force threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # (N, B_all, 3) -> (N, B_selected, 3)
    forces_w = contact_sensor.data.net_normal_forces_w.torch
    selected_forces_w = forces_w[:, sensor_cfg.body_ids, :]

    # (N, B_selected, 3) -> (N, B_selected) -> (N,)
    force_magnitudes = torch.linalg.vector_norm(
        selected_forces_w,
        dim=-1,
    )
    return torch.sum(
        (force_magnitudes > threshold).float(),
        dim=1,
    )


def joint_power_signed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Sum signed actuator mechanical power over the selected joints."""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
        
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torques * joint_vel, dim=1)


class JointAccelerationPenalty(ManagerTermBase):
    """Penalize joint acceleration estimated between policy steps."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene["robot"]
        self._previous_joint_vel = torch.zeros_like(self._robot.data.joint_vel.torch)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        joint_vel = self._robot.data.joint_vel.torch
        joint_acc = (joint_vel - self._previous_joint_vel) / env.step_dt
        penalty = torch.sum(torch.square(joint_acc), dim=1)
        self._previous_joint_vel.copy_(joint_vel)
        return penalty

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_joint_vel[env_ids, :] = self._robot.data.joint_vel.torch[env_ids, :]

