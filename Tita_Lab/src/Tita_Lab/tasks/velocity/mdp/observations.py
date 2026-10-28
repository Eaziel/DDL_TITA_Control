# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from ..actuators import TitaPDActuator
from .actions import TitaJointAction
from .events import RandomizeTitaMaterial

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import ContactSensor

    from ..tita_velocity_env import TitaVelocityEnv


def joint_pos_rel_zero_wheels(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_cfg: SceneEntityCfg,
) -> torch.Tensor:
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


class TitaDynamicsParameters(ManagerTermBase):
    """Randomized TITA dynamics parameters relative to the nominal robot."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv) -> None:
        """Store nominal inertial properties before startup randomization."""

        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]

        if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 1:
            raise ValueError("Select exactly one base body for dynamics observations.")
        self._base_id = asset_cfg.body_ids[0]

        # Observation terms are constructed before startup events run.
        self._nominal_mass = self._asset.data.body_mass.torch[:, self._base_id].clone()
        self._nominal_com = self._asset.data.body_com_pos_b.torch[:, self._base_id].clone()

        material = env.event_manager.get_term_cfg("randomize_material").func
        if not isinstance(material, RandomizeTitaMaterial):
            raise TypeError("Dynamics observations require RandomizeTitaMaterial.")

        # The event samples these per-robot values in its constructor.
        self._friction = material.friction_coeffs.to(self.device).unsqueeze(1)
        self._restitution = material.restitution_coeffs.to(self.device).unsqueeze(1)

        for actuator in self._asset.actuators.values():
            if not isinstance(actuator, TitaPDActuator):
                raise TypeError("Dynamics observations require TitaPDActuator groups.")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return base perturbations, material coefficients, and motor factors."""

        delta_mess = self._asset.data.body_mass.torch[:, self._base_id] - self._nominal_mass

        delta_com = self._asset.data.body_com_pos_b.torch[:, self._base_id, :] - self._nominal_com

        factors = []
        for name in ("motor_strength", "kp_factor", "kd_factor"):
            # Scatter actuator-local columns into the articulation's joint order.

            values = torch.ones_like(self._asset.data.joint_pos.torch)
            for actuator in self._asset.actuators.values():
                values[:, actuator.joint_indices] = getattr(actuator, name)

            # Gather into the task's explicit eight-joint observation order.
            factors.append(values[:, asset_cfg.joint_ids])

        # (N, 30) = mass 1 + CoM 3 + friction 1 + restitution 1 + 3*8 factors.
        return torch.cat(
            (
                delta_mess.unsqueeze(1),
                delta_com,
                self._friction,
                self._restitution,
                *factors,
            ),
            dim=-1,
        )


class TitaPrivilegedState(ManagerTermBase):
    """Base velocity, filtered wheel contacts, and action-history slot features."""

    def __init__(self, cfg: ObservationTermCfg, env: TitaVelocityEnv) -> None:
        """Initialize per-policy-step state samples and contact memory."""

        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]

        if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
            raise ValueError("Select exactly two wheel bodies in the required order.")

        self._robot: Articulation = env.scene[asset_cfg.name]
        self._sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

        legs = env.action_manager.get_term("leg_position")
        if not isinstance(legs, TitaJointAction):
            raise TypeError("Privileged state requires TitaJointAction terms.")
        self._lag_denominator = max(legs.cfg.lag_timesteps - 1, 1)

        self._linear_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_contacts = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._filtered_contacts = torch.zeros_like(self._last_contacts)

        # Dimension probing and reset before the first policy step keep a cold sample.
        self._sample_step = env.common_step_counter

    def __call__(
        self,
        env: TitaVelocityEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return six privileged features with one contact update per policy step."""

        if self._sample_step != env.common_step_counter:
            forces_z = self._sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, 2]
            contact = forces_z > 1.0

            # c_filtered(t) = c(t) OR c(t-1); samples are 10 ms apart.

            self._filtered_contacts.copy_(contact | self._last_contacts)
            self._last_contacts.copy_(contact)
            self._linear_velocity.copy_(self._robot.data.root_lin_vel_b.torch)
            self._sample_step = env.common_step_counter

        # This encodes the history SLOT, not the physical delay duration.
        lag_slot = env.action_lag_indices.to(dtype=torch.float32).unsqueeze(1) / self._lag_denominator

        return torch.cat(
            (
                2.0 * self._linear_velocity,
                self._filtered_contacts.float() - 0.5,
                lag_slot,
            ),
            dim=-1,
        )

    # Inherited reset() preserves contact memory and the current sample, as in Gym.
