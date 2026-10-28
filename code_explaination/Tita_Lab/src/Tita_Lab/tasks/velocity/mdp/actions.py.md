# actions.py：动作怎样变成延迟后的关节目标

**阅读目标：** 沿一份动作的变化读懂裁剪、FIR、缩放、目标历史与提交；区分策略步处理和物理步应用。

源码：[actions.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/actions.py)。更新：2026-10-27。**重复copy_已由用户修正，当前计算体与下方版本一致；新venv实际FIR/action-rate复核通过。** 两处旧延迟笔误也已修，第6节保留重复更新问题的原因和修正。

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.envs.mdp.actions import JointAction, JointActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from Tita_Lab.tasks.velocity.tita_velocity_env import TitaVelocityEnv


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
            # 腿的位置目标 = 默认关节角 + scale * 滤波动作 [rad]。
            self._offset = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()

        # Both groups reference the same (N,) tensor; startup fills it in place.
        self._lag_indices = env.action_lag_indices
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        # Final targets: leg positions [rad] or wheel feedforward efforts [N*m].
        # Default angles/zero effort match Gym's zero-filled action-offset history.
        self._initial_command = (
            self._offset.clone() if cfg.command_type == "position" else torch.zeros_like(self.raw_actions)
        )
        self._lag_buffer = self._initial_command.unsqueeze(1).repeat(1, cfg.lag_timesteps, 1)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Update filtered joint commands from policy actions."""

        self.previous_clipped_actions.copy_(self.clipped_actions)
        self.clipped_actions.copy_(actions.clamp(-self.cfg.raw_clip, self.cfg.raw_clip))

        # c_t=clip(a_t, ±100)，u_t=0.2*c_(t-1)+0.8*c_t。
        # 保存的是上一裁剪动作c_(t-1)，因此是两点FIR（Finite Impulse Response，有限脉冲响应），
        # 不是递归EMA（Exponential Moving Average，指数移动平均）：u_t = α * u_{t-1} + (1-α) * c_t
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
            # 这里只提交轮子的前馈力矩；阻尼反馈仍由TitaPDActuator计算。
            self._asset.set_joint_effort_target_index(target=delayed, joint_ids=self._joint_ids)

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
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
    raw_clip: float = 100.0  # 策略原始动作的无量纲裁剪上限。
    previous_weight: float = 0.2  # FIR上一裁剪动作的权重，当前动作权重为0.8。
    lag_timesteps: int = 1  # Number of stored targets; 1 gives zero latency.
```

## 1. ActionManager把八列动作分给两个实例

leg_position收到前六列，wheel_effort收到后两列。每组的raw_actions、processed_actions、裁剪参考和目标历史都属于自己的TitaJointAction实例，shape中的J分别为6和2。

cfg是这一个动作项的配置，里面的scale/command_type/lag_timesteps分别决定物理量映射、目标类型和保存数量。环境整份cfg属于另一个对象。

## 2. process_actions每策略步调用一次

先把此前裁剪动作复制到previous_clipped_actions，再将当前输入裁到±100。两点FIR为0.2×上一裁剪动作+0.8×当前裁剪动作。

父类process_actions把滤波结果变成processed_actions。腿使用默认角+scale×滤波动作，轮使用scale×滤波动作作为前馈。随后_raw_actions.copy_(actions)恢复原始输入，供观测读取；处理好的目标另有自己的存储。

| 变量 | 内容 | 主要读取位置 |
| --- | --- | --- |
| raw_actions | 原始输入，(N,J) | 观测 |
| clipped_actions | 本次裁剪输入 | FIR、动作变化惩罚 |
| previous_clipped_actions | 上次裁剪输入 | FIR、动作变化惩罚 |
| processed_actions | 已缩放/加默认角的目标 | 目标历史更新 |
| _lag_buffer | (N,L,J)，由旧到新排列的目标 | apply_actions按编号选择 |

## 3. apply_actions每物理步调用一次

当前策略周期包含四个物理步，所以同一processed_actions会加入历史四次。先移走最旧记录，再写入当前目标，最后每个环境按自己的_lag_indices选择。

例如三台机器人的编号为[0,2,1]，分别取buffer[0,0,:]、buffer[1,2,:]、buffer[2,1,:]。L=3时延迟分别为两个、零个、一个物理步，即5/0/2.5 ms。

腿提交位置目标，轮提交力矩前馈，随后TitaPDActuator读取当前关节状态计算反馈并限矩。这里已完成目标提交，不再调用未实现的父类apply_actions。

## 4. reset清什么、保留什么

父类清原始动作，任务类恢复选中环境的目标历史：腿填默认角，轮填零前馈。裁剪/FIR参考与选择编号保持不变，沿用旧Gym的跨回合规则。

Sequence先转换成list，再用显式剩余维度索引(N,L,J)，避免tuple被解释为多轴索引。其他环境的历史不变。

## 5. 导出、配置与验证

.pyi公开TitaJointActionCfg及随机编号事件；ActionsCfg两项都设置lag_timesteps=3；环境在父构造前创建共享编号，startup原位抽样。完整例子见[12动作延迟](../../../../../../../docs/migration/12_动作延迟迁移.md)。

早期候选/历史版本有CPU流程和Gym力矩对照；2026-10-27发现的重复更新已修，新venv当前代码CPU再次核查通过，见[模块结果](../../../../../../../docs/audits/2026-10-27_saved_stack_sim62/README.md)。用户新版短随机运行报告自行结束，日志未打印FIR/action-rate值，验证范围仍分开记录。

## 6. 已修复：裁剪历史被重复更新（2026-10-27）

缺陷版本的函数开头先执行两条copy_，在中间英文字符串后又执行同样两条。假设此前clipped_actions=0、当前输入10：第一次得到previous=0、current=10；第二次又得到previous=10、current=10。滤波变成0.2×10+0.8×10=10，动作变化平方变成(10−10)²=0。

正确行为是每策略步仅推进一次这两份缓存，输出0.2×0+0.8×10=8，动作变化平方为(10−0)²=100。这里的100是原始惩罚，未乘权重或dt。本页开头的完整方法只包含一组copy_。

新venv实际父类及奖励函数的[CPU证据](../../../../../../../docs/audits/2026-10-27_sim62_action_fir/README.md)：输入10→0，当前输出10→0、候选输出8→2；四环境六腿/两轮含裁剪边界下，候选目标与公式最大差0，原始输入保留，action-rate恢复动作差分。没有写源码或推进物理。

## 补充问答

**Q：**

> super().func()到底是啥？他在代码的位置会影响啥？我之前学ros2只知道他会继承父类，但现在的代码，需要我深入了解这玩意到底是干啥的（我不太会问，你应该能洞察到我的真正疑惑吧？）。还有return super,都是啥啊？真的需要认真了解一下这些。

**A：** 继承关系由`class TitaJointAction(JointAction)`声明。super()在这个关系中帮助查找实现；执行到super().func(...)时，实际调用找到的方法。

在普通实例方法里，super()返回一个查找代理，按当前对象的MRO，从定义该方法的类之后继续寻找。当前单继承链是TitaJointAction→JointAction→ActionTerm→ManagerTermBase。调用super().process_actions(...)会找到JointAction的实现；super().apply_actions()先经过没有该实现的JointAction，最后找到ActionTerm的抽象方法体并抛异常。父类方法不会因为被继承就自动额外执行。

父方法使用的self仍是同一个TitaJointAction对象，所以对self._raw_actions等属性的写入立即影响该实例。没有另建一份独立父对象。

一个仅用于理解语法的例子：

```python
class Parent:
    def compute(self, x):
        self.last_x = x
        return 2 * x


class Child(Parent):
    def compute(self, x):
        x += 1
        result = super().compute(x)
        return result + 3


obj = Child()
print(obj.compute(2))  # 9
print(obj.last_x)      # 3
```

obj.compute(2)先选中Child.compute，x变3，父方法用同一obj记下last_x=3并返回6；子方法从调用处继续，把结果加3后返回9。父调用放在哪里，就在那个位置执行，返回后继续它后面的代码。

项目中的调用位置由数据依赖决定：

```python
super().__init__(cfg, env)
self.clipped_actions = torch.zeros_like(self.raw_actions)
```

父初始化先创建raw_actions等对象，后一句才能按它的形状分配缓存。子类自己定义__init__后，Python不会另外替它执行父__init__，所以这里显式调用。

```python
super().process_actions(filtered)
self._raw_actions.copy_(actions)
```

父实现先把filtered写进同一实例的_raw_actions，并计算processed=raw*scale+offset；后一句再把raw改回原始策略输入。processed已经是运算结果，raw的后续覆盖不会回溯改变它。若倒过来，父方法会再次把raw覆盖为filtered，破坏保留原输入的约定。当前动作观测本身读取外层ActionManager.action；不要把它与term.raw_actions混为同一个缓存。

TitaVelocityEnv的历史缓存则要先于super().__init__创建：父初始化内部调用self.load_managers，建立ObservationManager，后者调用mdp.proprio_history(self)探测维度。这里的self仍是同一个自定义环境，观测函数需要它的proprio_history已经存在。因此super初始化既可以在前，也可以在后，要看哪些数据先被谁读取。

| 写法 | 执行与返回 |
| --- | --- |
| super().func() | 调用找到的方法；结果未被这句保存，父方法对self的修改仍保留；后面的代码继续执行 |
| return super().func() | 先调用并取得结果，然后退出当前方法，把结果交给当前调用者 |
| value=super().func(); 后续计算; return value | 保留父结果，允许子方法继续加工后再返回 |
| return super() | 返回查找代理，没有调用func |
| return super | 返回内置super类本身，没有创建代理或调用func |

如果父方法隐式返回None，return super().func()也返回None；如果父方法抛异常，则没有正常返回值，异常向外传播。__init__通常写super().__init__(...)再继续初始化，按Python约定返回None。

return只影响当前方法的返回，不会取消此前对对象或Tensor的修改。例如你当前apply_actions用setter写入目标即可，方法自然返回None；下一阶段scene.write_data_to_sim会继续读取这些已经写好的目标。

依据：[Python 3.12 super](https://docs.python.org/3.12/library/functions.html#super)、[return语句](https://docs.python.org/3.12/reference/simple_stmts.html#the-return-statement)、[本地JointAction](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py)、[ObservationManager维度探测](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/observation_manager.py)。

**Q：** 历史槽数是啥？？

**A：** lag_timesteps表示保存最近几次物理步目标。当前3对应三个位置，从旧到新排列；选择哪个位置，决定使用两步前、一步前还是本次目标。具体数值见第9节和12迁移说明。

**Q：** 但是我不知道哪里的代码实现了，存t-2和t-1,以及t步的？

**A：** apply_actions中的两次赋值负责保存历史：

```python
# 丢掉最旧记录，后两份各向前移动一个位置。
self._lag_buffer[:, :-1, :] = self._lag_buffer[:, 1:, :].clone()
# 最后一个位置写入当前处理好的目标。
self._lag_buffer[:, -1, :] = self.processed_actions
```

假设写入本步之前是[u(t-3),u(t-2),u(t-1)]，第一行变为[u(t-2),u(t-1),u(t-1)]，第二行再变为[u(t-2),u(t-1),u(t)]。这些t表示物理步，间隔2.5 ms；策略每10 ms输出一次，目标会连续加入四次。

不是每步重新抽编号：startup事件创建后抽一次0/1/2，各机器人可不同，后续物理步都按自己的固定编号取目标，reset也保留它。每步变化的是数组内的目标，物理继续运行，只是执行较早目标。

## 已修复问题

早期版本曾多调用未实现的父类apply_actions，另有历史切片[:1]和process_actions方法名两处笔误。当前已由用户修复；问题来由见10、12章，不再把旧实现混进本页主线。
