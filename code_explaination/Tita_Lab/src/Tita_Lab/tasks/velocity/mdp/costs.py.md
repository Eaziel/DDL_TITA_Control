# costs.py：六类约束代价怎样从关节与接触数据算出

内容纯AI生成，由人类要求生成，未经人类修改。源码及学习用途核对：2026-11-14。

被解释文件：[mdp/costs.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/costs.py)。

本文件为每台机器人计算六类原始cost。读完应能区分“是否越限”的0/1判断与“超出多少”的连续量，知道各个张量沿哪个维度归约，并明确这些函数不会直接替机器人限矩、reset或更新策略。

这些物理量怎样经过storage、GAE再影响策略，见[一批经历怎样完成策略更新](../../../../../../../docs/一批经历怎样完成策略更新.md)。本页先把每个原始项f_i的来源、单位和判据解释清楚。

本文核对当前源码与本地 IsaacLab-main / Sim 6.2 调用关系；未代运行仿真或训练。

## 1. 当前完整源码

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg
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

        # 在机器人reset之后调用，只更新被重置机器人的速度参考。
        self._previous_joint_vel[selection, :] = self._asset.data.joint_vel.torch[selection, :]
```

## 2. cost在强化学习里表示什么

常见MDP用状态、动作、转移、奖励和折扣描述任务。若还要单独记录不希望发生的现象，可以定义额外的代价 $c_i(s,a)$，并对未来累计代价提出要求；这类扩展常称约束MDP。

例如用 $J_R$ 表示期望累计奖励，用 $J_{C_i}$ 表示第i类期望累计代价，抽象目标可写作：

$$
\max_\pi J_R(\pi),\qquad
J_{C_i}(\pi)=\mathbb E_\pi\left[\sum_t\gamma^t c_{i,t}\right]\le b_i.
$$

$\pi$是策略，$\mathbb E$表示按策略与环境随机性求平均，$\gamma$是折扣，$b_i$是抽象的允许上限。当前check把配置中的d_i按这个累计cost上限解释，k_i另作策略惩罚权重；两者都不同于本文件里的关节／接触物理阈值。第3节以下的函数先判断本步发生什么，尚不计算累计量或策略损失。

本文件只计算原始项 $f_i$。[cost_manager.py](../cost_manager.py.md)再乘一次策略dt得到六列本步cost，wrapper交给[storage](../../../learning/np3o/rollout_storage.py.md)。storage结合cost critic的旧预测，逐列计算cost回报目标与优势；[check第17节](../../../../../scripts/check_np3o_interface.py.md)已用优势、k/d和累计cost估计构造策略惩罚并执行单批更新。这里算出非零cost不会立即阻止动作；它通过这条学习路径影响之后的动作分布。单批更新也不能证明学到的策略已满足真实约束。

## 3. 六列分别是什么，按哪个顺序返回

| cost列 | 配置名 | 本文件实现 | 原始输出 |
| --- | --- | --- | --- |
| 0 | acc_smoothness | JointAccelerationCost | 超阈值加速度的关节均值 |
| 1 | dof_vel_limits | joint_velocity_limit_cost | 任一关节速度超阈值则1，否则0 |
| 2 | feet_contact_forces | wheel_contact_force_cost | 任一轮法向接触合力的大小超阈值则1，否则0 |
| 3 | pos_limit | joint_position_limit_cost | 任一关节位置越软限位则1，否则0 |
| 4 | stumble | wheel_stumble_cost | 任一轮水平/竖直力比较满足条件则1，否则0 |
| 5 | torque_limit | joint_effort_limit_cost | 任一关节已施加力矩超阈值则1，否则0 |

函数在源码中的排列不决定cost列序，配置对象的条目顺序与管理器的收集顺序才决定。全部函数输出 `(N,)`，N为并行环境数；manager横向保存成 `(N,6)`。

以下用指示函数 $\mathbf1\{条件\}$ 表示条件为真取1，否则取0；“任一”即沿同一机器人的关节/轮子维度做OR，不把不同环境合到一起。

## 4. 位置软限位：当前角度与两端阈值比较

`joint_pos` 为 `(N,J)`，`limits` 为 `(N,J,2)`。最后两列依次是软下限与软上限：

$$
f_{pos,n}=\mathbf1\{\exists j:q_{n,j}<l^{soft}_{n,j}
\ \text{或}\ q_{n,j}>u^{soft}_{n,j}\}.
$$

`limits[...,0]`中的省略号代表前面所有轴，即取所有环境、所有选中关节的下限，得到 `(N,J)`。`|` 对两份布尔张量逐元素OR，`torch.any(...,dim=1)`再沿关节轴判断任一越界，最后 `.float()`变成0/1浮点。

软限位由Lab根据机器人关节原始限位与 `ArticulationCfg.soft_joint_pos_limit_factor` 提供。设原始中点 $m=(l+u)/2$、半宽 $h=(u-l)/2$、系数为 $\alpha$，软区间为 $[m-\alpha h,m+\alpha h]$。系数1沿用原区间，系数小于1可提前警告。它是数据接口里的阈值，不表示本函数会施加新的物理约束。

代码用严格 `<`/`>`，在端点上不算越界。例如区间 `[−1,1]`、角度1.05返回越限，角度1.0不会。

## 5. 力矩与速度：先取绝对值，再判断是否超过阈值

力矩项：

$$
f_{\tau,n}=\mathbf1\{\exists j:|\tau^{applied}_{n,j}|>
\rho_\tau\tau^{limit}_{n,j}\}.
$$

`torch.abs`去掉正负方向；−70N·m与+70N·m都可能过载。当前 `soft_ratio=1.0`，腿阈值60N·m、轮阈值15N·m。读取的是 `applied_effort`，即执行器限幅后的驱动力矩；没有读取限幅前的 `computed_effort`。

因此当前执行器若严格把输出裁到同一上限，严格大于上限的判断通常不会触发。比如计算想要80、输出被裁成60，比较 `60>60` 为False。本项沿用此处已有Gym口径，不能把0自动解释成“策略从未请求过过大力矩”。这也是理解变量选择比只看cost名称重要的原因。

速度项同样计算：

$$
f_{\dot q,n}=\mathbf1\{\exists j:|\dot q_{n,j}|>
\rho_v\dot q^{limit}_{n,j}\}.
$$

当前腿速度上限25rad/s、轮20rad/s，比例1。速度项读实际状态；力矩项读执行器输出。两个函数都只给数值判断，不在函数内把关节速度或力矩写回物理引擎。

## 6. 轮接触力：两个轮中任一个超过500N

传感器 `net_normal_forces_w.torch` 提供聚合法向力，选轮体后形状 `(N,2,3)`。norm沿xyz得到 `(N,2)`，any沿轮子维得到 `(N,)`：

$$
f_{force,n}=\mathbf1\{\exists j\in\{L,R\}:
\sqrt{(F^w_{x,n,j})^2+(F^w_{y,n,j})^2+(F^w_{z,n,j})^2}>500\}.
$$

它检测的是当前法向合力大小。即使传感器支持总接触力字段，此函数也没有自动把摩擦力加进去。当前力来自哪些碰撞对方不在本函数细分，完整物理口径见 [terminations.py](terminations.py.md)。

`threshold=500.0` 是函数默认参数，当前cfg未覆盖它。两轮 `[400,600]`N时输出1；`[400,500]`N时输出0，因为使用严格大于。

## 7. stumble比较水平分量与竖直分量

代码没有做除法，直接比较：

$$
F_{h,n,j}=\sqrt{(F^w_{x,n,j})^2+(F^w_{y,n,j})^2},\qquad
F_{v,n,j}=|F^w_{z,n,j}|,
$$

$$
f_{stumble,n}=\mathbf1\{\exists j:F_{h,n,j}>5F_{v,n,j}\}.
$$

`forces_w[...,:2]`取Fx/Fy，`forces_w[...,2]`取Fz。前者norm消去长度2的分量轴，后者abs直接得到竖直分量大小。

例如某轮Fx/Fy为30/40N、Fz为5N，水平长度50，比较50>25为True。若全部力为零，0>0为False；若Fz为零、水平力非零，可以触发，代码无需处理除零。

“水平”和“竖直”按世界坐标轴区分；法向方向则按接触面区分。坡面法向力也可以有水平分量，所以本项不是直接测轮子摩擦，也不是直接证明发生绊倒。它是沿用旧任务的力分量判据，具体地形上的触发要结合接触数据解释。

## 8. 加速度cost测的是超出阈值多少

`JointAccelerationCost`保存全部关节的上次速度，调用时选出任务8列：

$$
a_{t,n,j}=\frac{\dot q_{t,n,j}-\dot q_{t-1,n,j}}{\Delta t},\qquad
A^{limit}_{n,j}=\frac{\dot q^{limit}_{n,j}}{2\Delta t}.
$$

当前dt0.01s，腿阈值1250rad/s²、轮1000rad/s²。原始cost为：

$$
f_{acc,n}=\frac1J\sum_{j=1}^J
\max\bigl(|a_{t,n,j}|-A^{limit}_{n,j},0\bigr).
$$

`.abs()`逐元素取绝对值；`.clamp_min(0.0)`将负超出量清零；`.mean(dim=1)`沿关节取平均，分母是选中关节总数J，不是只除以超限关节数。

例如8关节中只有一个轮关节加速度1200rad/s²，其余未超限，超出量200，原始cost为200/8=25；manager乘dt后本步cost为0.25。此处原始量带rad/s²单位，其他五项为0/1，后续算法需要尊重这套既定尺度。

它与奖励中的 `JointAccelerationPenalty` 不同：奖励对全部加速度平方求和，即使很小也有扣分；本项只计算超过阈值的部分，再作均值。两者保存独立历史，不应让一个实例被反复调用后意外推进两次参考。

## 9. 有记忆的项由谁更新与reset

构造时 `.clone()`复制当前关节速度；`__call__`先差分，再 `copy_`保存这次速度。外部管理器保存类实例，所以每步不会丢掉这份记忆。

自定义cost管理器使用 `common_step_counter`防止同一策略步重复计算：reset前先取失败瞬间的cost，step结尾再取时复用缓存。否则同一步再次调用加速度类，第二次差分会因参考已经更新而变成零。

机器人reset完成后，cost管理器显式调用该类 `reset(env_ids)`，把被重置行的参考速度改为新回合初始速度。其他行不变。它不会清模型权重，也不是Python根据方法名自动调用。

原始项不乘dt、不乘k/d、不相互合并；读完本文件后，再沿 [cost管理器说明](../cost_manager.py.md) 看这六类结果怎样进入 `(N,6)`、回合统计和wrapper。

## 源码与API依据

- [本项目cost管理器](../cost_manager.py.md)
- [本项目执行器限幅](../actuators.py.md)
- [Lab PhysX关节软限位初始化](/home/eaziel/IsaacLab-main/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py)
- [PyTorch any](https://docs.pytorch.org/docs/2.11/generated/torch.any.html)
- [PyTorch mean](https://docs.pytorch.org/docs/2.11/generated/torch.mean.html)
