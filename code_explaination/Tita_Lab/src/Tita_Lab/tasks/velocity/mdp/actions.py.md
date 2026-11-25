# actions.py：8维动作怎样成为带延迟的关节目标

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-03；本次补写第6节。

被解释文件：[mdp/actions.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/actions.py)。

本文件接收策略输出的8个数，把前6个解释为腿关节的位置调整量，把后2个解释为轮子的力矩前馈。读完后应能沿一个数找到裁剪、两点滤波、单位换算、历史保存和延迟选择的实际存储位置；真正的PD力矩计算接着发生在 [actuators.py](../actuators.py.md)。

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

        # 保存上个策略步的裁剪动作；shape为(N, J)，腿J=6、轮J=2。
        self.previous_clipped_actions.copy_(self.clipped_actions)
        self.clipped_actions.copy_(actions.clamp(-self.cfg.raw_clip, self.cfg.raw_clip))

        # c_t=clip(a_t, ±100)，u_t=0.2*c_(t-1)+0.8*c_t。
        # 保存的是上一裁剪动作c_(t-1)，因此是两点FIR（Finite Impulse Response，有限脉冲响应），
        # 不是递归EMA（Exponential Moving Average，指数移动平均）：u_t = α * u_{t-1} + (1-α) * c_t
        # previous_weight是上一动作的权重，当前配置为0.2。
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

## 2. 先确定谁在调用这个类

策略交给环境的 `actions` 是二维张量，形状为 `(N,8)`。张量是带形状的数值数组；这里第0维有N行，一行对应一个并行环境中的机器人，第1维有8列，一列对应一个关节动作。批量计算表示同一段计算同时作用于N行，行之间没有因为放在同一张量中而互相控制。

当前动作列依次为左腿1/2/3、右腿1/2/3、左轮、右轮。`ActionManager` 根据配置项的声明顺序与维度拆分8列，创建两个 `TitaJointAction` 实例：

| 配置项名 | 收到的张量 | `command_type` | 目标单位 |
| --- | --- | --- | --- |
| `leg_position` | `(N,6)` | `"position"` | rad，弧度 |
| `wheel_effort` | `(N,2)` | `"effort"` | N·m，力矩 |

`TitaJointActionCfg` 保存参数，`class_type = TitaJointAction` 保存要构造的类。管理器据此执行类似 `cfg.class_type(cfg, env)` 的调用，得到具有自己缓存的实例。类不会因为写在文件里就自动运行。`.pyi` 导出配置名后，`env_cfg.py` 中的动作配置才把这个实现接入环境。

`@configclass` 是Lab提供的装饰器：装饰器接收下面定义的类并扩展其配置能力，例如初始化与复制。`cfg: TitaJointActionCfg` 是类型注解，用于表达属性应当是什么类型，本句没有创建新的配置实例。`Literal["position", "effort"]` 表达允许的字符串类型；实际运行的防错仍由构造函数中的 `if` 与 `raise ValueError` 完成。

## 3. 一个实例先建立哪些存储

### 3.1 `super()`调用同一个对象的父类实现

`class TitaJointAction(JointAction)` 声明继承。子类定义自己的 `__init__` 后，需要显式执行 `super().__init__(cfg, env)`，父类才会建立机器人引用、关节编号、缩放、偏移、原始动作和处理后动作等属性。

这里的 `super()` 是沿继承顺序查找方法的代理；父方法使用的 `self` 仍是当前这一份 `TitaJointAction` 实例。没有再创建一台独立的“父对象”。先执行父初始化，后面才可以读取 `self.raw_actions`。

`raw_actions` 是父类的属性接口，读取后得到 `_raw_actions` 张量。`torch.zeros_like(self.raw_actions)` 创建形状、数据类型和设备相同、内容为零的独立张量。腿的形状是 `(N,6)`，轮的形状是 `(N,2)`。

| 属性 | 在这一动作项中保存的内容 |
| --- | --- |
| `clipped_actions` | 当前策略步裁剪后的动作 |
| `previous_clipped_actions` | 上一策略步裁剪后的动作 |
| `_raw_actions` / `raw_actions` | 策略送进该动作项的原始输入 |
| `_processed_actions` / `processed_actions` | 已换算为位置或力矩的当前目标 |
| `_initial_command` | 腿默认角，或轮零前馈 |
| `_lag_buffer` | 按时间从旧到新保存的若干份目标 |

`self._offset = ...clone()` 把腿的默认角复制成偏移。`.torch` 是Lab物理数据 `ProxyArray` 的Torch访问入口；`.clone()` 才在这里生成独立副本。关节状态变化不会把这份默认目标一起改掉。

### 3.2 腿与轮共享的是延迟编号

```python
self._lag_indices = env.action_lag_indices
self._env_ids = torch.arange(self.num_envs, device=self.device)
```

第一句把已有张量的引用保存在属性中，不复制数值。环境的 `action_lag_indices`、腿的 `_lag_indices`、轮的 `_lag_indices` 指向同一份 `(N,)` 存储。第二句建立整数序列 `[0,1,...,N-1]`，后面用来逐行选择各机器人的历史。

初始化历史的代码：

```python
self._initial_command.unsqueeze(1).repeat(1, cfg.lag_timesteps, 1)
```

`unsqueeze(1)` 在第1维插入长度1的轴，把 `(N,J)` 变成 `(N,1,J)`。J是这个动作项控制的关节数。`repeat(1,L,1)` 沿中间轴复制L份，得到 `(N,L,J)`；L即 `lag_timesteps`。例如一个腿目标 `[0.1,-0.2]` 保存3份后是 `[[0.1,-0.2],[0.1,-0.2],[0.1,-0.2]]`。这只是初始化时的重复值，此后各位置会依次保存不同时刻的目标。

## 4. 每个策略步，先裁剪再滤波

当前策略周期为0.01s，物理步为0.0025s。`process_actions()` 每个策略步执行一次，随后同一处理结果用于4个物理步。

对一个动作分量，定义原始输入为 $a_t$，裁剪结果为 $c_t$，滤波结果为 $u_t$：

$$
c_t=\operatorname{clip}(a_t,-100,100),\qquad
u_t=0.2c_{t-1}+0.8c_t.
$$

`clip` 表示超过上限取上限、低于下限取下限，区间内保留原值。代码里的 `clamp` 完成同样运算。滤波输出 $u_t$ 是两份裁剪动作的加权和。

两条 `copy_` 的顺序直接实现“保存上次，再写这次”：

```python
self.previous_clipped_actions.copy_(self.clipped_actions)
self.clipped_actions.copy_(actions.clamp(-self.cfg.raw_clip, self.cfg.raw_clip))
```

`copy_` 把数值写入已有张量，末尾下划线是Torch常见的原位修改命名。假设当前 `clipped_actions` 保存上次的0，新的输入为10：第一句让 `previous` 得到0，第二句让 `clipped` 得到10，滤波输出为8。下次输入0时，先得到 `previous=10`、`clipped=0`，滤波输出为2。

两点FIR是有限脉冲响应滤波：这里只取两次裁剪动作的加权和。以采样角频率 $\Omega$ 表示变化快慢，其频率响应为：

$$
H(e^{i\Omega})=0.8+0.2e^{-i\Omega}.
$$

`i` 是虚数单位，$\Omega$ 单位为rad/采样。缓慢变化对应 $\Omega\approx0$，增益接近1；交替正负的最快变化对应 $\Omega=\pi$，增益为0.6，因此能减弱部分高频变化。它不会保证动作绝对平滑，也不限制每步变化必须小于某个固定数。

## 5. 父类把滤波值换算成实际目标

```python
super().process_actions(filtered)
self._raw_actions.copy_(actions)
```

本地 `JointAction.process_actions` 先保存输入，再计算仿射变换。仿射变换就是“乘比例，再加偏移”：

$$
p_{t,j}=s_j u_{t,j}+o_j.
$$

对于腿，$o_j=q_{0,j}$ 是默认角，髋关节比例为0.25，其他腿关节为0.5，单位为rad/动作单位。若默认角为−1.5rad，当前滤波值为0.2、比例为0.5，则位置目标为−1.4rad。

对于轮，偏移使用当前配置的0，比例为5N·m/动作单位，因此 $\tau^{ff}_{t,j}=5u_{t,j}$。这是前馈力矩，后续仍有轮速阻尼反馈。

父方法计算后，第二句把 `_raw_actions` 改回最初的策略输入。父方法已生成的 `processed_actions` 保存另一份运算结果；覆盖原始动作缓存不会倒过来重算目标。若把两句顺序颠倒，父方法就会在最后把原始缓存覆盖为滤波值。

没有变量接收 `super().process_actions(...)`，因为父方法的结果主要体现在对 `self` 属性的写入。函数可以通过修改对象产生效果，同时返回 `None`。`return super().某方法(...)` 则会把父方法返回值交给调用者，并立即结束当前方法。

## 6. 控制延迟、三个历史位置与编号，怎样联系起来

### 6.1 先看为什么要让目标晚一点生效

真实控制链中，策略算出动作后，还要经过程序调度、通信和执行器处理，新的控制目标可能晚一点才被采用。本任务用“暂时继续采用旧目标”的方法模拟其中一种固定控制延迟。

例如某腿关节原来一直收到0.1rad的位置目标，现在策略给出的新目标经过滤波、缩放与默认角处理后变成0.3rad。无延迟时，本物理步就把0.3交给PD；有延迟时，先继续交0.1，过一段时间再交0.3。PD与物理引擎随后根据目标和实际状态计算力矩及运动，目标改成0.3并不让关节瞬间转到0.3。

为了以后还能找到旧目标，程序保存最近几次的目标。这里保存的是控制目标，观测十帧历史由另一个对象ProprioHistory保存。

### 6.2 “槽”就是数组里保存一份目标的位置

当前配置 `lag_timesteps=3`，表示保存三份物理步目标，代码用L表示这个数量。先只看一台机器人、一个关节，数组可写成：

```text
位置编号：          0                  1                 2
保存内容：   两个物理步前的目标   上一个物理步的目标   本物理步的目标
```

旧说明把每个位置称为slot或“槽”。它只是数组中的一个位置，没有额外的物理器件或新类型。编号0、1、2用于告诉索引操作“从哪个位置取值”；L=3说明数组有几个位置。

实际 `_lag_buffer` 为 `(N,L,J)`：N台机器人，每台保存L份目标，每份有J个关节值。腿组J=6、轮组J=2。对一台机器人而言，一个位置保存的是该组全部关节的一份目标向量；前面的单关节例子只取其中一个数，便于观察时间变化。

### 6.3 新目标每物理步写入一次，旧目标向前移

对应源码：

```python
self._lag_buffer[:, :-1, :] = self._lag_buffer[:, 1:, :].clone()
self._lag_buffer[:, -1, :] = self.processed_actions
```

三维索引依次选择环境、时间位置、关节。`:`取该维全部；`:-1`取除末尾以外的位置；`1:`从位置1取到末尾；`-1`取最后位置。第一句把后面两份记录移到前面，第二句将当前处理目标写到最后。`.clone()`先复制右侧，避免同一数组中重叠读写。

仍用原来0.1rad、新目标0.3rad的例子。策略周期0.01s内有四个物理步，所以这四次写入采用相同的 `processed_actions=0.3`：

| 目标提交次数 | 更新后的三份目标 | 若固定取编号0 | 若固定取编号1 | 若固定取编号2 |
| --- | --- | --- | --- | --- |
| 新目标到来前 | `[0.1,0.1,0.1]` | 0.1 | 0.1 | 0.1 |
| 第1次 | `[0.1,0.1,0.3]` | 0.1 | 0.1 | 0.3 |
| 第2次 | `[0.1,0.3,0.3]` | 0.1 | 0.3 | 0.3 |
| 第3次 | `[0.3,0.3,0.3]` | 0.3 | 0.3 | 0.3 |
| 第4次 | `[0.3,0.3,0.3]` | 0.3 | 0.3 | 0.3 |

每次提交相隔一个物理步，即2.5ms。从首次可提交新目标的第1次算起，编号2立刻得到新值；编号1晚一次，延迟2.5ms；编号0晚两次，延迟5ms。下一次策略更新时，同样继续移动历史，不会重新把三份都清为零。

### 6.4 编号决定实际从历史中取哪份目标

对应源码：

```python
delayed = self._lag_buffer[self._env_ids, self._lag_indices, :]
```

`_env_ids`列出机器人编号，`_lag_indices`给出每台机器人选择的历史位置。例如三台机器人的选择编号为 `[0,2,1]`，分别取 `buffer[0,0,:]`、`buffer[1,2,:]`、`buffer[2,1,:]`，返回 `(N,J)`。

这里的高级索引按同一位置的两个编号配对：第0台取它的位置0，第1台取它的位置2，第2台取它的位置1。每台机器人分别选择，不把不同机器人的历史混在一起。

选择编号用数学符号ℓ表示。当前每台机器人在startup时随机抽一次ℓ，腿与轮共享这个整数，之后持续按该编号读取。编号保持不变，历史内容每物理步移动，所以新的目标会在相应延迟后到达选中位置。

一般公式为：

$$
\Delta t_\ell=(L-1-\ell)\,dt_{physics}.
$$

L−1是最新位置的编号；L−1−ℓ是选中记录距最新记录隔了多少个物理步。当前L=3、物理步2.5ms，便得到0/1/2号对应5/2.5/0ms。

腿的setter将选中结果写入位置目标，轮的setter写入力矩前馈目标，随后执行器读取这些目标计算控制输出。真正实现控制延迟的是这套“保存旧目标、选择历史位置、提交目标”的代码。

### 6.5 为什么随后还把编号写成一维观测

执行器已经在上面采用选中的目标。观测代码还会另外读取这个延迟设置，向训练侧提供信息，帮助价值网络区分不同控制响应条件：

$$
z_{lag}=\frac{\ell}{\max(L-1,1)}.
$$

当前L=3时，ℓ=0/1/2分别得到z_lag=0/0.5/1。这个数占一列，因为每台机器人只需要一个共享编号，就能确定整台机器人的腿与轮采用哪种延迟。三种可能取值仍是一个标量；并未把三个位置的目标向量全部塞进这一列。

| 控制实际使用的整数ℓ | 采用的记录 | 控制延迟 | 观测中的z_lag |
| --- | --- | --- | --- |
| 0 | 两个物理步前 | 5ms | 0 |
| 1 | 上一个物理步 | 2.5ms | 0.5 |
| 2 | 本物理步最新目标 | 0ms | 1 |

把编号除以最大编号，是当前任务选择的数值表示方法。它将这一特征放到0到1的范围，仍保留三种设置的区别；物理延迟由动作代码中的整数索引产生，观测z_lag提供关于这项设置的信息。

例如z_lag=0.5表示“这一台机器人选编号1”，对应2.5ms。数值0.5本身没有毫秒单位。critic也没有专门把0.5转换成毫秒的程序，它把这一列和其他观测一起输入网络，通过后续训练学习它与未来reward/cost的关系。

这项被加入 `privileged_state` 组：三维机身线速度、左右轮接触两维、延迟设置一维，总数3+2+1=6。它是仿真已知的控制设置；观测组可以同时包含物理状态读数与这类参数。设置不同可能使后续响应不同，因此有理由把它提供给价值预测。这种设计的实际收益仍需要训练验证。

### 6.6 从哪个文件追这几项数据

| 工作 | 实际文件与对象 |
| --- | --- |
| 创建每台机器人一个的整数编号 | [tita_velocity_env.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/tita_velocity_env.py:42)创建 `action_lag_indices` |
| startup随机抽编号 | [events.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/events.py:79)的 `randomize_tita_action_lag` |
| 保存目标、按编号执行 | 本文件 `TitaJointAction.apply_actions`，即上面第6.3—6.4节 |
| 把编号换成一列观测 | [observations.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/observations.py:164)的 `TitaPrivilegedState.__call__` |
| 读取这列进行价值预测 | [critic.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/learning/np3o/critic.py:71)取得 `obs[:,220:256]`，其中含完整六维组 |

六维组的逐列拼接和为何供critic使用，继续读[observations.py第6节](observations.py.md#6-六维状态速度3列接触2列延迟编号1列)。

## 7. reset恢复什么，哪些量继续保留

`env_ids` 可为指定环境编号、序列、切片或 `None`。`Sequence` 是序列类型接口；把普通序列变为 `list`，可以避免元组在张量索引中被解释为多个维度。`slice(None)` 等价于选取所有行的 `:`。

父类reset清选中机器人的原始动作。自定义部分将该机器人的全部目标历史填成默认腿角/零轮前馈，右侧 `(M,1,J)` 自动沿L维广播成 `(M,L,J)`。M是本次重置的机器人数。其他机器人历史不变。

当前实现保留FIR的两份裁剪参考，也保留startup抽取的延迟编号。这是当前迁移规则。动作观测 `mdp.last_action` 读取外层 `ActionManager.action`；动作变化惩罚读取这里的两个裁剪缓存。读取者不同，不能把两份数据自动当成同一变量。

读到这里，一个动作已经被解释成带延迟的物理目标。下一段力矩计算沿 [执行器说明](../actuators.py.md) 继续。

## 源码与API依据

- [Lab JointAction：缩放、偏移与reset](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py)
- [Lab ActionManager：拆分动作与调用动作项](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/action_manager.py)
- [环境step：处理一次动作、物理步内应用](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)
- [PyTorch Tensor索引与视图](https://docs.pytorch.org/docs/2.11/tensor_view.html)
