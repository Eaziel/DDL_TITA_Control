"""Candidate observation term; not installed into the user's source tree."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from TITA_Lab.tasks.velocity.actuators import TitaPDActuator
from TITA_Lab.tasks.velocity.mdp.events import RandomizeTitaMaterial

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg, SceneEntityCfg


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
        self._nominal_com = self._asset.data.body_com_pos_b.torch[:, self._base_id, :].clone()

        material = env.event_manager.get_term_cfg("randomize_material").func
        if not isinstance(material, RandomizeTitaMaterial):
            raise TypeError("Dynamics observations require RandomizeTitaMaterial.")
        # The event samples these per-robot values in its constructor.
        self._friction = material.friction_coeffs.to(self.device).unsqueeze(1)
        self._restitution = material.restitution_coeffs.to(self.device).unsqueeze(1)

        for actuator in self._asset.actuators.values():
            if not isinstance(actuator, TitaPDActuator):
                raise TypeError("Dynamics observations require TitaPDActuator groups.")

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Return base perturbations, material coefficients, and motor factors."""
        delta_mass = self._asset.data.body_mass.torch[:, self._base_id] - self._nominal_mass
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
            (delta_mass.unsqueeze(1), delta_com, self._friction, self._restitution, *factors),
            dim=-1,
        )
