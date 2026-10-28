# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


# Explicit：Lab 模型每个物理步读取状态，算力矩、限幅，再交给引擎
# Implicit：把目标和增益交给物理引擎，由引擎中的 drive 求解
class TitaPDActuator(IdealPDActuator):
    """Explicit PD with TITA's independent gain and motor-strength factors."""

    cfg: TitaPDActuatorCfg

    def __init__(self, cfg: TitaPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # 无量纲输出比例，用于随机化电机输出力矩，实现：有些给力过大，有些给力不足。
        self.motor_strength = torch.ones_like(self.stiffness)
        # 用于随机化，如：实际kp = kp * kp_factor = 40.0 * 1.1 = 44，目前先设置为1！
        self.kp_factor = torch.ones_like(self.stiffness)
        # 同上
        self.kd_factor = torch.ones_like(self.damping)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        feedforward = control_action.joint_efforts
        if self.cfg.scale_effort_with_kp:
            feedforward = self.kp_factor * feedforward

        self.computed_effort = self.motor_strength * (
            self.kp_factor * self.stiffness * error_pos + self.kd_factor * self.damping * error_vel + feedforward
        )

        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action


@configclass
class TitaPDActuatorCfg(IdealPDActuatorCfg):
    """Configure TITA's explicit PD and wheel feedforward calculation."""

    class_type: type[TitaPDActuator] = TitaPDActuator
    scale_effort_with_kp: bool = False
