# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def illegal_contact_current(
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return whether selected bodies exceed a contact-force threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # shape: (N, B_all, 3)
    #   N     = num_envs        环境数
    #   B_all = 传感器监测的所有 body 数
    #   3     = 力的 xyz 分量
    forces_w = contact_sensor.data.net_normal_forces_w.torch

    # shape: (N, B, 3)
    #   只保留配置里指定的那些 body，B = len(sensor_cfg.body_ids)
    selected_forces_w = forces_w[:, sensor_cfg.body_ids, :]

    # shape: (N, B)
    #   对最后一维（xyz）求模长 → 每个 body 一个“力的大小”
    force_magnitudes = torch.linalg.vector_norm(
        selected_forces_w, dim=-1
    )

    # shape: (N,)  bool
    #   每个环境一个判断：这个环境里“有没有任何一个 body”超阈值
    return torch.any(force_magnitudes > threshold, dim=1)

