# commands.py：目标速度怎样生成并进入观测与奖励

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-02。

被解释文件：[mdp/commands.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/commands.py)。

本文件规定机器人“这段时间应当运动多快”。读完应能区分目标速度、机器人实际速度和关节动作，并找到10秒重新采样、航向转角速度以及0.2m/s死区各自的调用位置。

本文按当前文件与本地 IsaacLab-main / Sim 6.2 接线解释。源码只读；数学与调用关系核对不等于重新运行仿真或验证训练效果。

## 1. 当前完整源码

下面是本次读取的完整源码，保留原注释，供后文逐段对照。

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from isaaclab.utils import configclass


class TitaVelocityCommand(UniformVelocityCommand):
    """Velocity commands with the legacy TITA planar speed deadband."""

    cfg: TitaVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Sample velocity commands with a planar speed deadband."""
        super()._resample_command(env_ids)

        # 命令缓冲为(N, 3)，每行[vx, vy, wz]。
        # env_ids选出本次需要重新采样的M个环境。
        vel_xy = self.vel_command_b[env_ids, :2]  # (M, 2)，单位m/s。
        speed_xy = torch.linalg.vector_norm(vel_xy, dim=1)  # (M,)。

        # 只有速度模长大于阈值（默认0.2 m/s）才保留平移命令。
        moving: torch.Tensor = speed_xy > self.cfg.lin_vel_deadband  # (M,)，bool。

        # (M, 1)广播到两列，并把处理后的结果写回原始命令缓冲。
        self.vel_command_b[env_ids, :2] = vel_xy * moving.unsqueeze(1)


@configclass
class TitaVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for TITA velocity commands."""

    class_type: type[TitaVelocityCommand] = TitaVelocityCommand

    lin_vel_deadband: float = 0.2
    """Planar command speeds at or below this threshold are zeroed [m/s]."""
```

## 2. 命令给出任务目标，策略决定如何完成

当前命令张量是 `vel_command_b`，形状 `(N,3)`，每行依次为：

$$
c_n=[v^{cmd}_{x,n},\ v^{cmd}_{y,n},\ \omega^{cmd}_{z,n}].
$$

N是并行机器人数量。前两列单位m/s，第三列单位rad/s。下标 `b` 表示线速度以机身坐标轴表达，`x/y` 分别是机身前向/侧向。命令是希望机器人达到的速度，实际速度另从物理状态读取。

策略看到命令和自身状态后输出8维关节动作；执行器根据动作目标计算关节力矩。命令张量本身不会直接把机身速度改成目标值。推扰事件会改写物理速度，那是 [events.py](events.py.md) 的另一项工作。

## 3. 配置怎样让这个类实际运行

`TitaVelocityCommand` 继承Lab的 `UniformVelocityCommand`，只扩展平移速度死区。父类已负责随机采样、计时、航向控制和可视化，所以本文件可以很短。

当前配置对象提供这些值：

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `resampling_time_range` | `(10,10)` s | 每次命令保持10秒后再抽取 |
| `heading_command` | True | 允许根据目标朝向产生转向速度 |
| `heading_control_stiffness` | 0.5 | 航向误差到角速度的比例系数 |
| `rel_heading_envs` | 1.0 | 全部环境采用航向目标模式 |
| `rel_standing_envs` | 0.0 | 不额外抽取要求三维命令全零的站立环境 |
| `lin_vel_deadband` | 0.2 m/s | 小平移命令的清零阈值 |

`TitaVelocityCommandCfg` 中 `class_type = TitaVelocityCommand` 保存类对象。MDP导出的是这个Cfg名字，配置对象仍然持有实际实现类；`CommandManager` 用 `class_type` 创建实例，因此不需要在配置文件额外直接导入实现类。

配置项名 `base_velocity` 成为管理器中的查询键。`env.command_manager.get_command("base_velocity")` 返回该项当前的 `(N,3)` 命令。若还有另一项命令，它有自己的名字与数据约定；`track_ang_vel_z_exp` 按此键取到速度命令，再读取第2列，管理器不会猜哪一列代表转向。

## 4. 重新采样到底重新抽什么

父类给每台机器人保存距下次采样还剩多久的 `time_left`。每个策略步减去0.01s，时间到时只对相应 `env_ids` 重新采样；回合reset也会采样。`env_ids` 例如 `[1,3]` 表示本次只更新第1、3台机器人的目标，不改其他行。

父类 `_resample_command` 从范围抽取：

$$
v_x^{cmd}\sim U(-1,1),\qquad
v_y^{cmd}\sim U(-1,1),\qquad
\omega_z^{sample}\sim U(-1,1).
$$

$U(l,u)$ 是区间上的均匀分布：长度相同的子区间具有相同概率。开启航向模式后，还抽目标朝向 $\psi^{cmd}\sim U(-3.14,3.14)$，保存在独立的 `heading_target` 中。它不是输出张量的第4列。

10秒是任务设计选择：在一段仿真时间中保持同一平移目标与目标朝向，给机器人持续跟踪的任务。它不是PPO规定的时间，也不是每10秒才输出一次动作。期间策略仍每0.01秒输出8个动作。若机器人提前失败，reset会提前生成新回合的命令。

## 5. 子类接着把太小的平移目标清零

```python
super()._resample_command(env_ids)
vel_xy = self.vel_command_b[env_ids, :2]
speed_xy = torch.linalg.vector_norm(vel_xy, dim=1)
moving = speed_xy > self.cfg.lin_vel_deadband
self.vel_command_b[env_ids, :2] = vel_xy * moving.unsqueeze(1)
```

`super()`先调用父实现填好本次选中行，然后子类继续处理同一实例的命令缓冲。`vel_xy` 为 `(M,2)`，M是此次选中的环境数。`vector_norm(...,dim=1)` 沿两列求欧氏长度：

$$
s_n=\sqrt{(v_{x,n}^{cmd})^2+(v_{y,n}^{cmd})^2}.
$$

`moving` 为 `(M,)` 的布尔张量；布尔值只有True/False。数学上相当于指示函数 $m_n=\mathbf1\{s_n>0.2\}$，满足条件取1，否则取0。随后：

$$
(v_{x,n}^{cmd},v_{y,n}^{cmd})\leftarrow
m_n(v_{x,n}^{cmd},v_{y,n}^{cmd}).
$$

例如三台被选中的机器人抽到 `[0.1,0.1]`、`[0.3,0]`、`[0.12,0.16]`，速度长度分别约0.1414、0.3、0.2。按严格大于号，数学结果是第一和第三行清零，第二行保留。浮点计算在边界附近受数值表示影响，因此源码中的比较对象始终是实际计算的张量值。

`unsqueeze(1)` 把 `(M,)` 变成 `(M,1)`；乘 `(M,2)` 时沿列广播，每行同一个判断同时乘两列。广播就是自动沿长度为1的维度使用同一数值，不混合不同环境的数据。

`vel_xy` 通过环境编号进行高级索引，读取通常得到独立结果。代码最后明确写回 `self.vel_command_b[env_ids,:2]`，所以命令缓冲才真正改变。只修改临时 `vel_xy` 并不能保证原缓冲同步。死区只动前两列，机器人仍可以在平移目标为零时接到转向目标。

## 6. 转向速度在父类的另一段方法中持续更新

父类 `_update_command` 每个命令更新周期执行：

$$
e_\psi=\operatorname{wrap}_{[-\pi,\pi]}(\psi^{cmd}-\psi),\qquad
\omega_z^{cmd}=\operatorname{clip}(0.5e_\psi,-1,1).
$$

$\psi$ 是当前世界航向角。`wrap_to_pi` 把相差整圈的角度折回主区间，从而按短角度误差计算，例如目标179°、当前−179°，误差应约−2°。0.5的单位可理解为1/s：rad乘该系数得到rad/s。

这段计算常叫航向比例外环。它根据“朝向还差多少”给出“应当转多快”，机器人具体怎样产生这个转动由策略与关节控制完成。若航向误差为0.6rad，目标角速度就是0.3rad/s；当前航向变化后，即使10秒还没到，误差与角速度命令也会变化。

`rel_heading_envs=1.0` 使这段结果覆盖所有环境的第2列。父类最初随机抽出的角速度对这些环境不会作为最终转向命令持续使用。`rel_standing_envs` 控制另一份站立标记，标记为站立时三列命令都清零；它与平移死区的两列清零规则分别存在。

## 7. 最后哪些计算会读取这三个数

观测项 `generated_commands(command_name="base_velocity")` 把三列放进policy观测，并分别乘 `(2,2,0.25)`。actor因此知道任务目标。

平移跟踪奖励读取前两列，与机身实际前向/侧向速度比较；转向跟踪奖励读取第2列，与实际 $\omega_z$ 比较。例如平移误差平方 $E=\|v_{xy}^{cmd}-v_{xy}\|_2^2$，当前配置返回 $\exp(-E/0.5^2)$，误差越小越接近1。这里指数函数把误差转换成连续评分。

地形课程也会读取命令的前两列，估计期望运动距离。可视化箭头读取速度命令，用于画图。它们都是具体读取者；“消费者”一词在旧说明中只是指这些读取对象。

训练循环取得观测、给出动作、推进物理后，父环境计算本步奖励，然后调用命令管理器更新命令，再生成下一份观测。因此目标切换后，新目标随下一份观测交给策略；本步奖励评价的是本步动作对应的目标。

## 源码与API依据

- [UniformVelocityCommand：采样与航向更新](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py)
- [CommandTerm/CommandManager：计时、reset与创建实例](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/command_manager.py)
- [PyTorch vector_norm](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.vector_norm.html)
