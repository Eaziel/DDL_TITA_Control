"""Candidate push event with privileged-state capture before velocity changes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    import torch

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


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
