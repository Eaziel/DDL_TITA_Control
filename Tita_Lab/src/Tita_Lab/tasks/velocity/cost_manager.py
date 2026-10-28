# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerBase, ManagerTermBase, ManagerTermBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class CostTermCfg(ManagerTermBaseCfg):
    """Configure a TITA cost term and its algorithm parameters."""

    # k、d留给约束算法，不参与这里的cost数值计算。
    k: float = 0.0
    d: float = 0.0


class TitaCostManager(ManagerBase):
    """Collect per-step costs and episode statistics for TITA."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv) -> None:
        self._term_names: list[str] = []
        self._term_cfgs: list[CostTermCfg] = []
        super().__init__(cfg, env)

        # 每行一台机器人，每列一项cost；当前形状(N, 6)。
        self.cost_buf = torch.zeros((env.num_envs, len(self._term_names)), device=env.device)
        # 每台机器人在当前回合内各项cost的累计值，(N,6)；reset只清选中行。
        self._episode_sums = torch.zeros_like(self.cost_buf)

        # 六项cost的算法参数；k为(1,6)，d为(1,1,6)。
        self.k_values = torch.tensor([term.k for term in self._term_cfgs], device=env.device).view(1, -1)
        self.d_values = torch.tensor([term.d for term in self._term_cfgs], device=env.device).view(1, 1, -1)

        # 初次reset不算一次训练步；同一步重复调用也不能重复累加。
        self._last_step = env.common_step_counter

    @property
    def active_terms(self) -> list[str]:
        """Names of the cost columns in their configured order."""
        return self._term_names.copy()

    def compute(self) -> torch.Tensor:
        """Return costs for the most recently completed policy step."""
        if self._last_step == self._env.common_step_counter:
            return self.cost_buf

        for index, term in enumerate(self._term_cfgs):
            value = term.func(self._env, **term.params)
            if value.shape != (self.num_envs,):
                raise ValueError(f"Cost '{self._term_names[index]}' must return shape ({self.num_envs},).")

            # 原始cost只在这里乘一次策略周期；k不乘在这里。
            self.cost_buf[:, index] = value * self._env.step_dt

        self._episode_sums.add_(self.cost_buf)
        self._last_step = self._env.common_step_counter
        return self.cost_buf

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Report episode costs and reset the selected histories."""

        selection = slice(None) if env_ids is None else env_ids
        statistics = {}

        for index, name in enumerate(self._term_names):
            # 沿用旧Gym：累计值除以最大回合秒数，非实际存活秒数。
            statistics[f"Episode_Cost/{name}"] = (
                self._episode_sums[selection, index].mean() / self._env.max_episode_length_s
            )

        self._episode_sums[selection, :] = 0.0

        for term in self._term_cfgs:
            if isinstance(term.func, ManagerTermBase):
                term.func.reset(env_ids)

        # 保留刚结束那一步的cost_buf，step仍要把它交给训练端。
        return statistics

    def _prepare_terms(self) -> None:
        """Resolve configured cost functions and scene selections."""

        for name, term in vars(self.cfg).items():
            if term is None:
                continue
            if not isinstance(term, CostTermCfg):
                raise TypeError(f"Cost '{name}' must use CostTermCfg.")

            # 原生接口解析关节/轮体名称，并将类对象变成实例。
            self._resolve_common_term_cfg(name, term)
            self._term_names.append(name)
            self._term_cfgs.append(term)
