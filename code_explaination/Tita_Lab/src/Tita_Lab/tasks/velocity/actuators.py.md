# actuators.py：位置目标与轮前馈怎样变成施加力矩

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-02。

被解释文件：[actuators.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/actuators.py)。

本文件定义TITA的显式PD执行器。读完应能从目标角、当前角、目标角速度和力矩前馈推到实际公式，理解Kp/Kd倍率与电机强度分别乘在哪里，并找到Lab真正调用compute的位置。

本次只读核对当前文件和本地 IsaacLab-main / Sim 6.2，不代运行事件、物理仿真或训练。

## 1. 当前完整源码

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


# Explicit：Lab 模型每个物理步读取状态，算力矩、限幅，再交给引擎
# Implicit：把目标和增益交给物理引擎，由引擎中的 drive 求解
class TitaPDActuator(IdealPDActuator):
    """Explicit PD with TITA's independent gain and motor-strength factors."""

    cfg: TitaPDActuatorCfg

    def __init__(self, cfg: TitaPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # 无量纲输出比例，用于随机化电机输出力矩，实现：有些给力过大，有些给力不足。
        self.motor_strength = torch.ones_like(self.stiffness)
        # 用于随机化，如：实际kp = kp * kp_factor = 40.0 * 1.1 = 44，目前先设置为1！
        self.kp_factor = torch.ones_like(self.stiffness)
        # 同上
        self.kd_factor = torch.ones_like(self.damping)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        feedforward = control_action.joint_efforts
        if self.cfg.scale_effort_with_kp:
            feedforward = self.kp_factor * feedforward

        self.computed_effort = self.motor_strength * (
            self.kp_factor * self.stiffness * error_pos + self.kd_factor * self.damping * error_vel + feedforward
        )

        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action


@configclass
class TitaPDActuatorCfg(IdealPDActuatorCfg):
    """Configure TITA's explicit PD and wheel feedforward calculation."""

    class_type: type[TitaPDActuator] = TitaPDActuator
    scale_effort_with_kp: bool = False
```

## 2. 从动作文件结束的位置继续

[actions.py](mdp/actions.py.md)已经把策略8个原始数变成6个腿位置目标与2个轮力矩前馈，并做了延迟选择。setter把这些值写进目标缓存，还没有计算完整关节驱动力矩。

当前场景配置创建两个 `TitaPDActuator` 实例：

| 组 | 关节数J | 名义Kp | 名义Kd | 力矩上限 |
| --- | --- | --- | --- | --- |
| legs | 6 | 40 | 1 | 60N·m |
| wheels | 2 | 0 | 0.5 | 15N·m |

每个实例处理 `(N,J)` 数组，N是并行机器人数。Kp单位N·m/rad，Kd单位N·m·s/rad。配置标量会按相应批量与关节展开为执行器计算可用的参数，不表示只有一个机器人能使用。

`TitaPDActuatorCfg.class_type`指向实现类；机器人Articulation建立执行器集合时按配置构造实例。USD提供关节、刚体、碰撞与惯性等物理结构，Lab执行器在这些关节上定义力矩计算。执行器不另造一套articulation root，也不自动修复USD关节结构。

## 3. 显式计算与隐式drive分别在哪里算反馈

显式执行器在Lab侧读取当前关节状态，计算一份力矩并提交物理引擎。隐式执行器把目标与增益交给后端drive，由物理求解过程处理反馈约束。当前使用自定义显式类，才能按旧TITA公式明确控制每个倍率与轮前馈的乘法位置。

两种路径均与物理求解器相连；“显式”不代表不用物理引擎，也不代表一定更精确。它说明控制反馈的实现位置。当前 `sim.use_newton_actuators=False` 保留本地Lab执行路径，使这个Python `compute`真正参与控制；不能只看写了类就断定所有后端设置都会调用它。

## 4. 构造函数为何先创建三个全1矩阵

```python
super().__init__(cfg, *args, **kwargs)
self.motor_strength = torch.ones_like(self.stiffness)
self.kp_factor = torch.ones_like(self.stiffness)
self.kd_factor = torch.ones_like(self.damping)
```

`super().__init__`先初始化 `IdealPDActuator`。`*args`接收额外位置参数，`**kwargs`接收额外关键字参数；调用父类时星号再把它们展开。这样本子类能在不重复列出全部父类构造参数的情况下继续传递机器人关节等信息。

`torch.ones_like`创建与给定张量形状、设备、类型相同的全1数组。腿组的三个因子为 `(N,6)`，轮组为 `(N,2)`。每个环境、每个关节都有自己的倍率。

设名义刚度 $K_p$、阻尼 $K_d$，因子 $f_p,f_d$，强度s，则等效反馈中的增益部分为 $f_pK_p$、$f_dK_d$；之后s还乘整份力矩。全1使初始化时等于名义配置，startup事件再把它们独立采样到0.8到1.2。

这些因子是物理随机化参数，不是 `nn.Parameter`，本文件没有神经网络和梯度更新。`stiffness`与 `damping`仍保存名义PD参数；将随机化单独作为因子，可以观测这些变化并保留名义基准。

## 5. `compute`的三个输入分别是什么

```python
compute(control_action, joint_pos, joint_vel)
```

`control_action`是 `ArticulationActions` 数据对象，包含该执行器对应的目标关节位置、目标关节速度、力矩前馈与关节编号。名称里的Actions在这里指关节控制数据；它已经不是策略刚输出的无量纲8列数组。

`joint_pos`和 `joint_vel`是本物理步的实际关节位置和速度，分别单位rad、rad/s。所有目标与实测量在当前执行器内按同一 `(N,J)` 排列，因此相减按同一机器人、同一关节对应。

`compute`是这个类约定的普通方法名，需要Lab执行器集合显式调用；Python没有规定“所有叫compute的方法都自动运行”。本项目另有观测manager.compute、cost_manager.compute，它们分别属于不同对象、执行不同算法。

## 6. 反馈误差与前馈先构成力矩

```python
error_pos = control_action.joint_positions - joint_pos
error_vel = control_action.joint_velocities - joint_vel
feedforward = control_action.joint_efforts
```

数学上：

$$
e_q=q^{des}-q,\qquad e_{\dot q}=\dot q^{des}-\dot q.
$$

比例反馈 $K_pe_q$ 根据位置偏差施力；速度反馈 $K_de_{\dot q}$ 根据速度偏差施力。前馈 $\tau^{ff}$ 是直接给定的力矩分量，不先由这两个误差算出来。

当前腿通常使用目标角速度0和前馈0，因此：

$$
\tau_{leg}^{calc}=s\big[f_p\,40(q^{des}-q)-f_d\,1\dot q\big].
$$

当前轮Kp=0、目标角速度0，且 `scale_effort_with_kp=True`，因此先 `feedforward = kp_factor * feedforward`，再得到：

$$
\tau_{wheel}^{calc}=s\big[-f_d\,0.5\dot q+f_p\tau^{ff}\big],
\qquad \tau^{ff}=5u_{delayed}.
$$

轮的Kp因子在这里仍有作用，因为旧公式用该因子缩放轮前馈；它不是通过零刚度的位置误差起作用。腿的该开关默认False，所以腿前馈若非零也不会在这一分支额外乘Kp因子。

统一形式为：

$$
\tau^{calc}_{n,j}=s_{n,j}\left[
 f_{p,n,j}K_{p,n,j}e_{q,n,j}
+f_{d,n,j}K_{d,n,j}e_{\dot q,n,j}
+\widetilde\tau^{ff}_{n,j}\right].
$$

其中 $\widetilde\tau^{ff}=f_p\tau^{ff}$ 当开关开启，否则直接为 $\tau^{ff}$。所有乘法都是逐元素乘，不是矩阵乘法。

## 7. 用具体数字走完腿与轮

腿某关节：目标−1.4rad、实测−1.5rad、速度0.2rad/s，s=1.1、$f_p=0.9$、$f_d=1.2$，前馈0：

$$
e_q=0.1,\quad e_{\dot q}=-0.2,\qquad
\tau^{calc}=1.1(0.9\times40\times0.1-1.2\times0.2)=3.696\mathrm{N\cdot m}.
$$

轮某关节：延迟后滤波动作0.4，因此前馈2N·m；轮速3rad/s，因子同上：

$$
\tau^{calc}=1.1(0.9\times2-1.2\times0.5\times3)=0.
$$

前馈与阻尼在这个例子抵消，所以动作非零也可以输出零净驱动力矩。若轮速不同，结果随当前物理状态改变；同一策略目标在4个物理步保持时，反馈力矩仍可每步不同。

## 8. 限幅以后，输出对象保留什么

```python
self.applied_effort = self._clip_effort(self.computed_effort)
control_action.joint_efforts = self.applied_effort
control_action.joint_positions = None
control_action.joint_velocities = None
return control_action
```

`computed_effort`保存限幅前计算值，`applied_effort`保存限幅后驱动力矩。本地 `IdealPDActuator._clip_effort`按关节力矩上限裁剪：

$$
\tau^{applied}=\operatorname{clip}(\tau^{calc},-\tau^{limit},\tau^{limit}).
$$

腿想输出80会裁成60，轮想输出−20会裁成−15。限幅是直接改变提交给后端的数值；奖励或cost的越限评分只是测量结果，两者作用不同。

赋 `joint_positions=None`、`joint_velocities=None` 表示这次输出只提供已计算好的力矩，不继续在这个输出对象里提供位置/速度目标。`None`不等于给位置和速度目标都设为0。

`return control_action`返回传入并被修改的同一个数据对象，调用者继续将其散回整机输出。`applied_effort`也用于功率奖励与力矩cost；因此这些项读到的是控制模型的限幅后输出，不是网络原始数或全系统所有约束力。

## 9. Lab在物理步中怎样调用它

当前父环境每个策略step处理一次动作，随后按decimation执行物理循环。每轮调用动作项apply写目标、Scene写控制数据、Sim推进、Scene更新状态。

本地 `ActuatorCollection.compute`先从整机目标按执行器关节编号取出位置/速度/前馈，构造 `ArticulationActions`，然后执行：

```python
control_action = actuator.compute(
    control_action,
    joint_pos=joint_pos.torch[:, joint_indices],
    joint_vel=joint_vel.torch[:, joint_indices],
)
```

`actuator`此时就是已经构造的腿组或轮组 `TitaPDActuator` 实例；Python根据对象实际类型选择子类compute。返回后集合把结果按关节编号写回整机命令，再交后端。自定义类没有自己写while循环，是环境与执行器集合负责反复调用它。

这条接线与计算公式能由当前源码确认；电机的真实带宽、效率、驱动器延迟和饱和特性是否被充分近似，需要后续物理验证与实测。当前理想PD加随机倍率并没有包含全部电机电气模型。

## 源码与API依据

- [Lab IdealPDActuator与实际_clip_effort](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/actuators/actuator_pd.py)
- [Lab ActuatorCollection.compute](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/actuators/actuator_collection.py)
- [本项目电机随机化](mdp/events.py.md)
- [本项目控制目标](mdp/actions.py.md)
- [PyTorch ones_like](https://docs.pytorch.org/docs/2.11/generated/torch.ones_like.html)
