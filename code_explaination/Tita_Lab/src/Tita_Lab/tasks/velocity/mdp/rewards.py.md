# rewards.py：物理量怎样计算成每步奖励

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-02。

被解释文件：[mdp/rewards.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/rewards.py)。

本文件生成自定义奖励项的原始数值，管理器再乘权重与策略周期，环境最后按旧TITA规则合成总分。读完应能解释接触计数、机械功率、动作变化和加速度分别在惩罚什么，以及带记忆的奖励类为何需要reset。

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

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

from .actions import TitaJointAction

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def undesired_contacts_current(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Count selected bodies exceeding the current normal contact-force threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # (N, B_all, 3) -> (N, B_selected, 3)
    forces_w = contact_sensor.data.net_normal_forces_w.torch
    selected_forces_w = forces_w[:, sensor_cfg.body_ids, :]

    # (N, B_selected, 3) -> (N, B_selected) -> (N,)
    force_magnitudes = torch.linalg.vector_norm(
        selected_forces_w,
        dim=-1,
    )
    return torch.sum(
        (force_magnitudes > threshold).float(),
        dim=1,
    )


def joint_power_signed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Sum signed actuator mechanical power over the selected joints."""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")

    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torques * joint_vel, dim=1)


def action_rate_clipped_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """按旧Gym的裁剪动作，计算每个环境相邻策略步的变化平方和。"""
    penalty = torch.zeros(env.num_envs, device=env.device)
    for name in ("leg_position", "wheel_effort"):
        term = env.action_manager.get_term(name)
        if not isinstance(term, TitaJointAction):
            raise TypeError("Clipped action-rate requires TitaJointAction terms.")

        # 分别得到(N,6)/(N,2)的差值，再对关节维求和，结果均为(N,)。
        delta = term.clipped_actions - term.previous_clipped_actions
        penalty += torch.sum(torch.square(delta), dim=1)

    return penalty


class JointAccelerationPenalty(ManagerTermBase):
    """Penalize joint acceleration estimated between policy steps."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene["robot"]
        self._previous_joint_vel = torch.zeros_like(self._robot.data.joint_vel.torch)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        joint_vel = self._robot.data.joint_vel.torch
        joint_acc = (joint_vel - self._previous_joint_vel) / env.step_dt
        penalty = torch.sum(torch.square(joint_acc), dim=1)
        self._previous_joint_vel.copy_(joint_vel)
        return penalty

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_joint_vel[env_ids, :] = self._robot.data.joint_vel.torch[env_ids, :]
```

## 2. 先把一项原始值、加权贡献与总奖励分开

本文件每个奖励函数返回 `(N,)`，一台机器人一个数。记第i项原始值为 $f_i(s_t,a_t)$，配置权重为 $w_i$，策略周期为 $\Delta t=0.01s$，Lab默认贡献为：

$$
r_{i,t}=\Delta t\,w_i f_i(s_t,a_t),\qquad
r_t^{Lab}=\sum_i r_{i,t}.
$$

例如某惩罚函数返回4、权重−0.05，本步贡献为−0.002。函数名字叫reward不要求函数自己返回正收益：平方误差通常非负，负权重把它变成扣分。

乘dt使按时间累计的量在改变控制频率时仍能保留相近积分口径。对连续评分率 $g(t)$，$\sum_t g_t\Delta t$ 是时间积分的离散近似。具体权重仍是任务设计参数；碰撞和失败这样的离散事件也沿用了框架的dt缩放，不要把其数值自动解释成自然物理定律。

本地 `RewardManager._step_reward[:,i]` 保存 $w_if_i$，总奖励与回合累计才乘dt。自定义环境需要从这一内部列取出termination项，再乘0.01，详见 [环境说明](../tita_velocity_env.py.md)。

## 3. 默认复用项如何提供主要任务目标

速度跟踪等普通项直接复用Lab。当前配置包含平移/转向跟踪、横向角速度、机身倾斜、离地高度，以及本文件的动作变化、碰撞、功率和加速度等项。

平移跟踪评分为：

$$
f_{track}=\exp\left(-\frac{\|\boldsymbol v^{cmd}_{xy}-\boldsymbol v^b_{xy}\|_2^2}{\sigma^2}\right),
\qquad \sigma=0.5.
$$

$\exp(x)=e^x$；误差为零时评分为1，误差变大时趋近0。这里std是控制误差评分宽度的参数，与 `policy.std` 的动作采样标准差分别属于两项计算。转向评分只比较命令第三列和实际z角速度。

本文件主要补足不能直接照搬默认口径的项。以下按数据走向理解每一项。

## 4. 关节位置偏差平方：`joint_pos_target_l2`

先读选中关节角，`wrap_to_pi` 把相差整圈的角折回主区间，再减同一个标量target，平方并沿关节列相加：

$$
f_{pos,n}=\sum_{j\in\mathcal J}
\big(\operatorname{wrap}(q_{n,j})-q^{target}\big)^2.
$$

`l2` 在该函数中表示平方误差和，没有再开平方。`torch.square(x)`逐元素平方，`torch.sum(...,dim=1)`把 `(N,J)` 沿关节维变成 `(N,)`。例如偏差 `[0.1,-0.2]`，返回0.05。

`SceneEntityCfg` 的 `joint_ids` 是管理器根据关节名解析的编号，函数不按名字逐个搜索。当前 `RewardsCfg` 没有启用本函数对应项；函数存在和被配置使用是两件需要分别核实的事实。

## 5. 小腿接触计数：`undesired_contacts_current`

传感器读取 `(N,B_all,3)` 的世界系聚合法向接触力，选择左右小腿 `left_leg_3/right_leg_3` 后为 `(N,2,3)`。`vector_norm(...,dim=-1)` 得到每个小腿的合力大小 `(N,2)`。

```python
return torch.sum((force_magnitudes > threshold).float(), dim=1)
```

比较产生布尔值，`.float()`把False/True变成0/1，再对两条小腿求和。结果为0、1或2，表示本步超阈值的小腿数，不是接触点数，也不是力的总大小。当前threshold=0.1N、weight=−1，一条小腿超阈值，本步扣0.01。

“聚合法向力”和w坐标的完整推导见 [接触与终止说明](terminations.py.md)。本项没有筛选碰撞对方；地面或其他物体造成的小腿法向接触均可能计入。它读取当前字段，不沿传感器三帧历史取最大值。

## 6. 机械功率：`joint_power_signed`

函数读取执行器限幅后的驱动力矩与实际关节角速度，并计算：

$$
P_n=\sum_j\tau^{applied}_{n,j}\dot q_{n,j}.
$$

旋转机械做功满足 $\mathrm dW=\tau\,\mathrm dq$，除以时间得到 $P=\tau\dot q$。这里q单位rad，角速度rad/s，力矩N·m，功率单位W。rad在量纲上无量纲，但表达角度时仍保留rad说明数值意义。

`torques * joint_vel` 是逐元素乘法，同一行同一关节相乘；随后 `sum(dim=1)` 把关节功率相加。它没有进行矩阵乘法，也没有先取绝对值。

例如两关节力矩 `[2,-1]`、角速度 `[3,4]`，各自功率 `[6,-4]`W，总和2W。正功率表示驱动力矩沿运动方向输出机械能，负功率表示驱动力矩抵抗运动。不同关节正负会抵消；当前函数保留这一带符号口径。若功率和为负，配合负权重甚至会产生正贡献，这一结果直接来自当前公式。

本项没有电压、电流、绕组损耗、驱动器损耗、电机效率或再生回收模型。不能据此把结果当作整机输入电功率，更不能直接用于RoboMaster式整机电功率上限验证。机械功率积分可表达机械功，但转换成实际耗电仍需要额外模型与实测参数。

`asset_cfg=None`时才在函数内部创建 `SceneEntityCfg("robot")`，默认选择整机。显式传入选择时则使用选中关节。本地 `asset.actuators.applied_effort` 是执行器输出的限幅后驱动力矩，不是把接触约束、被动结构等全部广义力合并后的总力矩。

## 7. 相邻动作变化：`action_rate_clipped_l2`

当前读取两个自定义动作项已经保存的裁剪缓存，计算：

$$
f_{rate,n}=\sum_{j=1}^{8}(c_{t,n,j}-c_{t-1,n,j})^2.
$$

$c_t$ 是原始动作裁剪到±100后的结果，尚未FIR、缩放、延迟或PD。两个动作项分别提供6列与2列；循环把两组平方和加到 `(N,)` 的 `penalty`。

本函数没有除以dt，因此名称中的rate在当前代码中表示相邻动作变化惩罚，不是严格数值微分 $\mathrm da/\mathrm dt$。例如一个动作从0变到0.5、其余不变，原始惩罚为0.25，乘权重−0.01和dt后为−0.000025。

环境状态通常随时间连续变化，惩罚相邻动作大幅变化可以鼓励控制指令变化较小；碰撞、重置、命令切换与噪声仍可能使输入突变，所以连续物理过程不构成“动作必然平滑”的证明。

PPO的近端约束比较的是更新前后策略在同一批数据上的概率分布。这里比较同一机器人相邻时间的两个动作值。一个限制学习更新幅度，一个为环境中的动作时间变化评分，二者评价的对象和时间索引不同。FIR又是直接改变执行的目标；奖励惩罚只通过未来训练影响策略偏好。

`isinstance(term,TitaJointAction)`保证拿到的对象具有这些缓存，失败抛 `TypeError`。属性名相近不保证任意默认动作项都能替换进来。

## 8. 加速度惩罚为什么需要一个类

### 8.1 差分需要记住上一次速度

`JointAccelerationPenalty` 保存 `_previous_joint_vel`，形状与全机关节速度相同。每策略步末计算：

$$
\widehat{\ddot q}_{t,n,j}
=\frac{\dot q_{t,n,j}-\dot q_{t-1,n,j}}{0.01},\qquad
f_{acc,n}=\sum_j\widehat{\ddot q}_{t,n,j}^{\,2}.
$$

“估计”表示用两个离散时刻的差替代连续导数。当前口径是0.01秒策略步差分，不是0.0025秒物理步差分。若某关节速度1.0→1.2rad/s，加速度估计20rad/s²，平方贡献400。

函数先计算，再 `self._previous_joint_vel.copy_(joint_vel)` 保存这次速度。若先复制，速度差就会变成零。`copy_`不会让前面已经计算出的平方和跟着消失。

### 8.2 类、实例和`__call__`怎样接线

配置 `func=mdp.JointAccelerationPenalty` 提供类。Lab先构造实例，再把实例存回 `term_cfg.func`。之后管理器的 `term_cfg.func(env)` 调用实例 `__call__`，返回原始惩罚；并不是每个step重新执行构造函数。

这个类不是神经网络层，缓存也不是可学习权重。`__call__`是Python调用协议；Torch的nn.Module另外把实例调用接到forward。看到括号调用，应先确定对象属于哪一类。

### 8.3 谁调用reset，为什么在机器人重置之后

父环境先执行reset事件，写入新关节位置与速度，之后 `RewardManager.reset(env_ids)` 遍历其登记的类奖励实例，调用这些实例的 `reset`。本类因此将被重置行的速度参考改成新状态：

```python
self._previous_joint_vel[env_ids, :] = self._robot.data.joint_vel.torch[env_ids, :]
```

这样下一步比较的是“新回合上一步速度”与新速度，避免用上一回合末速度差分。`env_ids=None`时转换为 `slice(None)` 选择全部行。历史缓存、cost类有各自的reset路径；Python不会因为方法同名就遍历所有对象。

## 9. 这些项最后怎样进入训练奖励

配置权重决定正负与大小，RewardManager先得到各项之和。`TitaVelocityEnv.step` 再实现：

$$
r_t=\max\left(0,\sum_{i\ne termination}\Delta t\,w_if_{i,t}\right)
+\Delta t\,w_{termination}f_{termination,t}.
$$

先把普通项合计夹到非负，再加失败项；没有把每个负项分别清零。例如普通合计−0.3、失败项−2，最终是−2。当前失败权重−200、dt0.01，所以失败一次的贡献是−2，而非−200。

回报是训练端把许多步奖励按折扣累计得到的量，reward critic学习估计这类未来回报。本文只描述环境本步原始项和最终奖励；当前短检查尚未进行这一步学习。

## 源码与API依据

- [RewardManager：调用、weight与dt、类reset](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/reward_manager.py)
- [Lab默认速度/姿态奖励](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/rewards.py)
- [ManagerBase：类变成实例](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/manager_base.py)
- [PyTorch sum](https://docs.pytorch.org/docs/2.11/generated/torch.sum.html)
