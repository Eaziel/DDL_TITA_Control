"""Candidate startup randomization for the TITA physical-step delay."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from TITA_Lab.tasks.velocity.mdp.actions import TitaJointAction

if TYPE_CHECKING:
    from TITA_Lab.tasks.velocity.tita_velocity_env import TitaVelocityEnv


def randomize_tita_action_lag(env: TitaVelocityEnv, env_ids: torch.Tensor | None) -> None:
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
        0,
        legs.cfg.lag_timesteps,
        env.action_lag_indices[selection].shape,
        device=env.device,
    )
