"""Candidate stateful observation; not written into the task's source tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from TITA_Lab.tasks.velocity.mdp.actions import TitaJointAction

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
    from isaaclab.sensors import ContactSensor

    from TITA_Lab.tasks.velocity.tita_velocity_env import TitaVelocityEnv


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
            # c_filtered(t) = c(t) OR c(t-1); each sample is 10 ms apart.
            self._filtered_contacts.copy_(contact | self._last_contacts)
            self._last_contacts.copy_(contact)
            self._linear_velocity.copy_(self._robot.data.root_lin_vel_b.torch)
            self._sample_step = env.common_step_counter

        # This encodes the history SLOT, not the physical delay duration.
        lag_slot = env.action_lag_indices.to(dtype=torch.float32).unsqueeze(1) / self._lag_denominator
        return torch.cat(
            (2.0 * self._linear_velocity, self._filtered_contacts.float() - 0.5, lag_slot),
            dim=-1,
        )

    # Inherited reset() preserves contact memory and the current sample, as in Gym.
