"""Environment interface for TITA NP3O training."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import gymnasium as gym
import torch

from Tita_Lab.tasks.velocity.observation_layout import pack_tita_observations

if TYPE_CHECKING:
    from Tita_Lab.tasks.velocity.tita_velocity_env import TitaVelocityEnv


class TitaNP3OEnvWrapper:
    """Expose the TITA Lab task through the NP3O rollout interface."""

    def __init__(self, env: gym.Env) -> None:
        """Store the environment and its training dimensions."""

        self.env = env
        # env可能带Gymnasium包装层；_env直接指向真正的TitaVelocityEnv。
        self._env = cast("TitaVelocityEnv", env.unwrapped)
        self.num_envs = self._env.num_envs
        self.device = self._env.device
        self.num_actions = self._env.action_manager.total_action_dim
        self.max_episode_length = self._env.max_episode_length

        # 五组按既定顺序拼成586列；价值网络所需数据也已包含在其中。
        self.num_obs = 586
        self.num_privileged_obs = None
        self.cost_names = self._env.cost_manager.active_terms
        self.num_cost = len(self.cost_names)

        # 供算法读取的六项参数，分别为(1,6)和(1,1,6)；不乘进本步cost。
        self.cost_k_values = self._env.cost_manager.k_values
        self.cost_d_values_tensor = self._env.cost_manager.d_values

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Return the policy-step counts of the current episodes."""
        return self._env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Set the policy-step counts of the current episodes."""
        self._env.episode_length_buf = value

    def reset(self) -> torch.Tensor:
        """Reset all environments and return the packed observations."""
        observations, _ = self.env.reset()
        return pack_tita_observations(observations=observations)

    def get_observations(self) -> torch.Tensor:
        """Return the most recently computed observations."""
        # 读取reset/step已经算好的结果，不再次采噪声或更新历史。
        return pack_tita_observations(self._env.obs_buf)

    def get_privileged_observations(self) -> None:
        """Use the packed observations as the critic input."""
        return None

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, None, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance the task and return one NP3O rollout transition."""
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        obs = pack_tita_observations(observations=observations)

        # (N,6)，已经乘过一次step_dt，并保留了自动reset前的本步数值。
        costs = extras["costs"]
        # (N,)，失败或超时都表示本回合结束；旧存储使用整数0/1。
        dones = (terminated | truncated).to(dtype=torch.long)

        # 仅往副本添加旧训练循环需要的键，不改Lab自己的extras字典。
        infos = dict(extras)
        if not self._env.cfg.is_finite_horizon:
            infos["time_outs"] = truncated
        if bool(dones.any()) and "log" in extras:
            infos["episode"] = dict(extras["log"])
        infos["depth"] = None  # 旧TITA默认不启用深度相机分支。

        # 第二项None表示不另传一张critic专用观测，数据已在obs的586列中。
        return obs, None, rewards, costs, dones, infos

    def close(self) -> None:
        """Close the wrapped environment."""
        self.env.close()
