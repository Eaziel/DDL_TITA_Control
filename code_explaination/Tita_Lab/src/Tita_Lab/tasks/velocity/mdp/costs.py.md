# costs.py：计算每台机器人的约束数值

**阅读目标：** 读懂六项原始cost函数的取数、阈值和跨步状态；它们输出每环境一个数，dt与列拼接交给收集器。

源码目录：Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/。
源码文件：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/costs.py`。

整理：2026-10-26。五个无状态函数与加速度类均已写入，收集器/环境接线已完成；四处笔误已修，用户已有4环境200步真实cost接口日志。此处解释原始计算，终止步数值与训练效果仍未单独验证。

## 1. 当前五个函数

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor


def joint_position_limit_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Flag environments with a joint outside its soft position limits."""

    asset: Articulation = env.scene[asset_cfg.name]

    # 当前关节角(N,J)，单位rad；J由asset_cfg选中的关节决定。
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    # (N,J,2)，最后一维依次为软下限、软上限。
    limits = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, :]

    out_of_limits = (joint_pos < limits[..., 0]) | (joint_pos > limits[..., 1])
    # 逐行检查：任一关节越界即为True，再转成(N,)的0/1。
    return torch.any(out_of_limits, dim=1).float()


def joint_effort_limit_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """Flag environments whose applied joint effort exceeds the threshold."""

    asset: Articulation = env.scene[asset_cfg.name]

    # 限幅后的驱动力矩(N,J)，与旧Gym的self.torques对应。
    torques = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    # 求解器力矩上限(N,J)，当前腿60、轮15 N·m。
    limits = asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids]

    # soft_ratio是阈值比例；旧值为1.0。
    return torch.any(torch.abs(torques) > limits * soft_ratio, dim=1).float()


def joint_velocity_limit_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """Flag environments whose joint speed exceeds the threshold."""

    asset: Articulation = env.scene[asset_cfg.name]

    # 当前关节角速度(N,J)，单位rad/s。
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    # 求解器关节速度上限(N,J)，当前腿25、轮20 rad/s。
    limits = asset.data.joint_vel_limits.torch[:, asset_cfg.joint_ids]

    return torch.any(torch.abs(joint_vel) > limits * soft_ratio, dim=1).float()


def wheel_contact_force_cost(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500.0,
) -> torch.Tensor:
    """Flag environments with excessive wheel normal contact force."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # 选中轮体的当前世界系法向力，(N,W,3)，单位N；本任务W=2。
    forces_w = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :]

    # 每轮力的大小(N,W)，任一轮超过阈值则该环境为1。
    forces_magnitudes = torch.linalg.vector_norm(forces_w, dim=-1)
    return torch.any(forces_magnitudes > threshold, dim=1).float()


def wheel_stumble_cost(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    horizontal_ratio: float = 5.0,
) -> torch.Tensor:
    """Flag environments with excessive horizontal-to-vertical wheel contact force."""

    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :]

    # 每轮世界水平力sqrt(Fx²+Fy²)，以及竖直力abs(Fz)，均为(N,W)。
    horizontal_force = torch.linalg.vector_norm(forces_w[..., :2], dim=-1)
    vertical_force = torch.abs(forces_w[..., 2])

    # 旧规则：水平力超过竖直力的5倍，记一次侧向碰撞判断。
    stumble = horizontal_force > horizontal_ratio * vertical_force
    return torch.any(stumble, dim=1).float()
```

位置、速度、力矩的(N,J)逐关节比较，再any(dim=1)得到每台机器人一个0/1。位置用soft_joint_pos_limits，力矩读已限幅的applied_effort。

接触两函数读取(N,W,3)的当前世界系法向接触力。vector_norm(..., dim=-1)把三个分量合成每轮力大小；另一函数比较水平分量大小与abs(Fz)的五倍，再对轮体any。它们没有历史，也不读取经过OR的轮接触标记。

这五项均返回(N,)浮点张量，暂不乘时间或算法k。当前64环境旧公式对照通过；法向力与旧合接触力的物理输入差异仍保留。

## 2. 加速度类（已见写入）

顶部导入按[17](../../../../../../../docs/migration/17_cost收集与加速度.md)第2节增加，然后追加：

```python
class JointAccelerationCost(ManagerTermBase):
    """Measure joint acceleration above the legacy TITA threshold."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # 上一次策略步结束时的关节速度，(N, J)，单位rad/s。
        self._previous_joint_vel = self._asset.data.joint_vel.torch.clone()

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        joint_vel = self._asset.data.joint_vel.torch
        joint_ids = asset_cfg.joint_ids
        # 每0.01 s差分一次，得到(N, J_selected)的rad/s²。
        acceleration = (joint_vel[:, joint_ids] - self._previous_joint_vel[:, joint_ids]) / env.step_dt
        acceleration_limit = self._asset.data.joint_vel_limits.torch[:, joint_ids] / (2.0 * env.step_dt)
        excess = (acceleration.abs() - acceleration_limit).clamp_min(0.0)
        self._previous_joint_vel.copy_(joint_vel)
        # 每台机器人一个值，(N,)；此处尚未乘dt。
        return excess.mean(dim=1)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Use reset joint velocities as the next acceleration reference."""
        selection = slice(None) if env_ids is None else env_ids
        # 必须在机器人reset之后调用；只更新被重置机器人的行。
        self._previous_joint_vel[selection, :] = self._asset.data.joint_vel.torch[selection, :]
```

_previous_joint_vel持有上一策略步的速度；__call__做速度差分、减阈值、夹零、求均值，再保存本次速度。每个实例有自己的数据，与奖励实例相互独立。

reset只更新选中机器人的速度参考。环境先让机器人重置，TitaCostManager.reset再调用这个reset，因此下一步不会把回合边界当成突然加速。

## 3. 谁调用与怎样输出

CostsCfg选择函数；TitaCostManager解析对象并逐步调用，将每个(N,)写入(N,6)的一列，只在这里乘一次0.01 s。环境负责在自动reset前保存这一步结果，最终经extras["costs"]交给未来训练端。该接线在用户本轮磁盘文件中尚未完整保存，待保存后核对，见17。

## 补充问答

**Q：** cost在我学的RL理论里没有提到。cost是mdp中的吗？讲一下什么是cost？

**A：** 普通MDP常用reward表示目标，也可以改用cost表示要最小化的量，最大化reward与最小化其相反数是等价表述。本项目在reward之外另设六个cost，对应带约束MDP（CMDP）：提高奖励，同时要求各项期望累计cost不超过各自上限。cost可以是0/1，也可以是连续值。

```text
目标：最大化 E[Σ γ^t r_t]
约束：每项 E[Σ γ^t c_i,t] ≤ d_i
```

这里c_i,t为第i类当步cost，d_i是该类期望折扣累计cost的上限。约束通常针对期望累计量，不能把训练目标当作每一步都不越界的硬保证。理论定义见[CPO论文第4节](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf)。

TITA的reward已有跟踪/姿态等合成分数，另外六列cost交给NP3O的cost价值估计及约束损失。mdp/costs.py负责由环境数据算出这些数，训练代码负责怎样使用它们；函数返回1本身不会阻止物理运动。
