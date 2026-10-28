# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.envs.mdp.actions import JointAction, JointActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from TITA_Lab.tasks.velocity.tita_velocity_env import TitaVelocityEnv


class TitaJointAction(JointAction):
    """Filtered TITA joint commands with per-robot physical-step latency."""

    cfg: TitaJointActionCfg

    def __init__(self, cfg: TitaJointActionCfg, env: TitaVelocityEnv) -> None:
        """Initialize joint command histories and shared delay selection."""
        super().__init__(cfg, env)
        if cfg.command_type not in ("position", "effort"):
            raise ValueError(f"Unsupported TITA command type: {cfg.command_type}")
        if cfg.lag_timesteps < 1:
            raise ValueError("lag_timesteps must be at least one.")

        # Policy-step FIR samples, both (N, J); legs J=6 and wheels J=2.
        self.clipped_actions = torch.zeros_like(self.raw_actions)
        self.previous_clipped_actions = torch.zeros_like(self.raw_actions)
        if cfg.command_type == "position":
            self._offset = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()

        # Both action groups reference the same (N,) tensor; startup fills it in place.
        self._lag_indices = env.action_lag_indices
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        # Store final targets: leg positions [rad] or wheel feedforward efforts [N*m].
        # Default angles/zero effort equal the old Gym zero-filled action-offset history.
        self._initial_command = (
            self._offset.clone() if cfg.command_type == "position" else torch.zeros_like(self.raw_actions)
        )
        self._lag_buffer = self._initial_command.unsqueeze(1).repeat(1, cfg.lag_timesteps, 1)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Update filtered joint commands from policy actions."""
        self.previous_clipped_actions.copy_(self.clipped_actions)
        self.clipped_actions.copy_(actions.clamp(-self.cfg.raw_clip, self.cfg.raw_clip))

        # c_t=clip(a_t), u_t=0.2*c_(t-1)+0.8*c_t; one update per policy step.
        previous_weight = self.cfg.previous_weight
        filtered = previous_weight * self.previous_clipped_actions + (1.0 - previous_weight) * self.clipped_actions
        super().process_actions(filtered)
        self._raw_actions.copy_(actions)

    def apply_actions(self) -> None:
        """Apply delayed joint position or feedforward effort commands."""
        # (N, L, J), oldest to newest; one push per physics step.
        self._lag_buffer[:, :-1, :] = self._lag_buffer[:, 1:, :].clone()
        self._lag_buffer[:, -1, :] = self.processed_actions
        delayed = self._lag_buffer[self._env_ids, self._lag_indices, :]

        if self.cfg.command_type == "position":
            self._asset.set_joint_position_target_index(target=delayed, joint_ids=self._joint_ids)
        else:
            self._asset.set_joint_effort_target_index(target=delayed, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Restore initial command histories for the selected robots."""
        if isinstance(env_ids, Sequence):
            env_ids = list(env_ids)
        super().reset(env_ids)
        selection = slice(None) if env_ids is None else env_ids
        self._lag_buffer[selection, :, :] = self._initial_command[selection, :].unsqueeze(1)
        # Preserve FIR reference samples and the robot's sampled delay index.


@configclass
class TitaJointActionCfg(JointActionCfg):
    """Configuration for filtered and delayed TITA joint commands."""

    class_type: type[TitaJointAction] = TitaJointAction
    command_type: Literal["position", "effort"] = "position"
    raw_clip: float = 100.0
    previous_weight: float = 0.2
    lag_timesteps: int = 1  # Number of stored targets; 1 gives zero latency.
