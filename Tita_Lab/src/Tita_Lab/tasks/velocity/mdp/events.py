# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import sample_uniform

from ..actuators import TitaPDActuator
from .actions import TitaJointAction

if TYPE_CHECKING:
    from omni.physics.tensors.api import ArticulationView

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from ..tita_velocity_env import TitaVelocityEnv


def push_by_setting_xy_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Replace world-frame horizontal root COM velocity with random samples."""

    # Preserve the physics-result sample before this event changes root velocity.
    env.observation_manager.compute_group("privileged_state")

    asset: Articulation = env.scene[asset_cfg.name]
    selection = slice(None) if env_ids is None else env_ids
    velocity_w = asset.data.root_vel_w.torch[selection].clone()
    velocity_w[:, :2] = sample_uniform(
        velocity_range[0],
        velocity_range[1],
        (velocity_w.shape[0], 2),
        device=asset.device,
    )
    asset.write_root_velocity_to_sim_index(root_velocity=velocity_w, env_ids=env_ids)


def randomize_tita_motor_factors(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    motor_strength_range: tuple[float, float],
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
) -> None:
    """Sample independent motor factors for every environment and joint."""
    if asset_cfg.joint_ids != slice(None):
        raise ValueError("Select the entire robot without a joint subset.")

    asset: Articulation = env.scene[asset_cfg.name]
    selection = slice(None) if env_ids is None else env_ids
    for actuator in asset.actuators.values():
        if not isinstance(actuator, TitaPDActuator):
            raise TypeError("Motor-factor randomization requires TitaPDActuator groups.")
        for factors, bounds in (
            (actuator.motor_strength, motor_strength_range),
            (actuator.kp_factor, kp_range),
            (actuator.kd_factor, kd_range),
        ):
            factors[selection, :] = sample_uniform(
                bounds[0], bounds[1], factors[selection, :].shape, device=asset.device
            )


def randomize_tita_action_lag(
    env: TitaVelocityEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Sample a shared action-history slot for each robot."""

    legs = env.action_manager.get_term("leg_position")
    wheels = env.action_manager.get_term("wheel_effort")

    if not isinstance(legs, TitaJointAction) or not isinstance(wheels, TitaJointAction):
        raise TypeError("Action lag randomization requires TitaJointAction terms.")
    if legs.cfg.lag_timesteps != wheels.cfg.lag_timesteps:
        raise ValueError("Leg and wheel command histories must have the same length.")

    selection = slice(None) if env_ids is None else env_ids

    # Write in place: both action terms retain references to this (N,) tensor.
    env.action_lag_indices[selection] = torch.randint(
        0, legs.cfg.lag_timesteps, env.action_lag_indices[selection].shape, device=env.device
    )


class RandomizeTitaMaterial(ManagerTermBase):
    """Assign shared friction and restitution to all collision shapes of each robot."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv) -> None:
        # 该实现依赖 Kit，等仿真启动、事件实例开始构造时再导入。
        from isaaclab_physx.physics import PhysxManager

        super().__init__(cfg, env)
        if not isinstance(env.sim.backend_manager, PhysxManager):
            raise ValueError("RandomizeTitaMaterial supports Isaac Sim PhysX only.")

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        if asset_cfg.body_ids != slice(None):
            raise ValueError("Select the entire robot without a body subset.")
        self.asset: Articulation = env.scene[asset_cfg.name]

        num_buckets = cfg.params["num_buckets"]
        friction_range = cfg.params["friction_range"]
        bucket_ids = torch.randint(0, num_buckets, (env.num_envs,), device="cpu")
        friction_buckets = sample_uniform(friction_range[0], friction_range[1], (num_buckets,), device="cpu")
        self.friction_coeffs = friction_buckets[bucket_ids]

        restitution_range = cfg.params["restitution_range"]
        restitution_ids = torch.randint(0, num_buckets, (env.num_envs,), device="cpu")
        restitution_buckets = sample_uniform(restitution_range[0], restitution_range[1], (num_buckets,), device="cpu")
        self.restitution_coeffs = restitution_buckets[restitution_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
    ) -> None:

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device="cpu", dtype=torch.int32)
        else:
            env_ids = env_ids.to(device="cpu", dtype=torch.int32)

        view: ArticulationView = self.asset.root_view

        materials = wp.to_torch(view.get_material_properties()).clone()
        friction = self.friction_coeffs[env_ids].unsqueeze(-1)
        restitution = self.restitution_coeffs[env_ids].unsqueeze(-1)
        materials[env_ids, :, 0] = friction
        materials[env_ids, :, 1] = friction
        materials[env_ids, :, 2] = restitution
        view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32),
            wp.from_torch(env_ids, dtype=wp.int32),
        )
