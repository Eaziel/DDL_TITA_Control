# observation_history.py：十帧观测怎样保存、推进和重置

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-02。

被解释文件：[mdp/observation_history.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/observation_history.py)。

本文件实现一个有状态的普通Python对象：它保存每台机器人过去10帧、每帧33个数。读完后应能解释一次读数为何不应推进时间，以及环境step开头那次写入怎样保证历史里不包含即将返回的当前帧。

本文按当前文件与本地 IsaacLab-main / Sim 6.2 接线解释。源码只读；数学与调用关系核对不等于重新运行仿真或验证训练效果。

## 1. 当前完整源码

下面是本次读取的完整源码，保留原注释，供后文逐段对照。

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence

import torch


class ProprioHistory:
    def __init__(
        self,
        num_envs: int,
        obs_dim: int = 33,
        history_length: int = 10,
        device: str | torch.device = "cpu",
    ) -> None:

        if num_envs <= 0 or obs_dim <= 0 or history_length <= 0:
            raise ValueError("缓存三个维度必须为正整数.")

        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.history_length = history_length

        self._history = torch.zeros(
            (num_envs, history_length, obs_dim),
            dtype=torch.float32,
            device=device,
        )

    def read_then_push(
        self,
        current_obs: torch.Tensor,
        episode_length: torch.Tensor,
    ) -> torch.Tensor:

        if current_obs.shape != (self.num_envs, self.obs_dim):
            raise ValueError("current_obs 的形状不匹配.")
        if episode_length.shape != (self.num_envs,):
            raise ValueError("episode_length 的形状不匹配.")

        # 保存本次要返回的旧历史
        past = self._history.clone()

        # 普通步：丢掉最旧帧，追加当前帧
        shifted = torch.cat(
            (self._history[:, 1:, :], current_obs.unsqueeze(1)),  # unsq 加时间轴
            dim=1,
        )

        # 旧 Gym 的 开局 规则：用当前帧填满历史
        repeated_current = current_obs.unsqueeze(1).expand(
            -1,
            self.history_length,
            -1,  # (N, 1, 33)  --expand(-1, 10, -1)-->  (N, 10, 33)
        )
        self._history = torch.where(  # torch.where(条件, A, B)
            (episode_length <= 1)[:, None, None],  # 刚开局（第0、1步）。`[:, None, None]`：每环境一个 bool，广播用
            repeated_current,  # 刚开局，填满
            shifted,
        )

        # 先读取再push新的观测，这里返回的是push新观测前的
        return past

    # 历史缓存“只读取，不更新”。
    # Lab 可能为了确定观测维度、记录数据等原因，多次调用观测函数。
    # 读取一次观测，不应当等于时间前进了一步。因此，读取历史和写入新帧必须分开。
    def read(self) -> torch.Tensor:
        """Return a copy of the stored history without updating it."""

        return self._history.clone()

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:

        if env_ids is None:
            self._history.zero_()
        else:
            self._history[env_ids, :, :] = 0.0
```

## 2. 为什么当前33个数之外还要过去十帧

当前policy观测包含角速度、重力方向、速度命令、关节位置偏差、关节速度和上次动作，共33维。单帧只给一个时刻的信息；若保存连续观测，网络还可以从变化中估计运动趋势，以及摩擦、延迟等未直接提供给actor的影响。

历史输入只是提供更多数据，并不自动规定网络学到什么。当前actor用MLP编码历史，critic用时间卷积编码历史；缓存类只保存数值，不建立神经网络、不计算梯度，也不更新策略参数。

用 $x_t\in\mathbb R^{33}$ 表示第t次返回的当前观测。正常推进阶段，希望返回的历史为：

$$
H_t=[x_{t-10},x_{t-9},\ldots,x_{t-1}]\in\mathbb R^{10\times33}.
$$

它与当前 $x_t$ 分开。开局不足10帧时采用代码中的填充规则，下面具体追踪。

## 3. `__init__`分配三维张量

`num_envs` 为环境数N，`history_length` 为10，`obs_dim` 为33。三个值必须为正，防止创建无意义的空缓存。`torch.zeros((N,10,33),dtype=torch.float32,device=device)` 创建浮点零数组：

| 索引轴 | 含义 |
| --- | --- |
| 第0轴，N | 哪一台机器人 |
| 第1轴，10 | 哪一个历史时刻，旧在前、新在后 |
| 第2轴，33 | 这一帧的哪个特征 |

例如 `_history[2,9,4]` 是第2台机器人最新历史帧中的第4个特征。Python索引从0开始。`device` 决定数据放CPU内存还是GPU显存；当前环境用 `cfg.sim.device` 创建缓存，使历史与policy观测在同一设备计算。

构造参数默认 `device="cpu"` 是类可独立使用时的默认值；环境实际传入GPU设备时就使用传入值。默认参数不强迫所有调用都在CPU运行。

## 4. `read()`返回副本，不改变任何历史

```python
return self._history.clone()
```

`clone()`复制张量数值，返回同形状的新存储。读取者修改返回结果时不会修改 `_history`。`proprio_history(env)` 观测函数随后把 `(N,10,33)` 展开成 `(N,330)`。

Lab可能在初始化时为确定维度调用观测函数，也可能在同一策略步中为记录或输出重复调用。若每次读取都追加一帧，网络历史长度虽然仍是10，但各格不再对应10个策略时刻。因此本类把 `read` 与推进分开，环境明确决定什么时候推进。

## 5. `read_then_push`先保留旧结果，再构造新历史

输入 `current_obs` 必须为 `(N,33)`，`episode_length` 必须为 `(N,)`。后者是每台机器人当前回合已走过的策略步数，不是秒数，也不是物理步数。形状不符会抛 `ValueError`，避免错误数据被广播成看似可用的结果。

### 5.1 先复制调用前的历史

```python
past = self._history.clone()
```

`past` 保存旧内容。随后即使 `self._history` 被重新赋值，`past` 仍保持旧内容；最后 `return past` 把这份旧内容交给调用者。

当前 `TitaVelocityEnv.step` 没接收返回值，它需要的是该方法把新历史写入对象的效果。方法既可以修改对象，也可以返回一个值；调用者可以只使用其中一种结果。

### 5.2 普通情况下丢最旧帧，再追加本次帧

```python
shifted = torch.cat(
    (self._history[:, 1:, :], current_obs.unsqueeze(1)),
    dim=1,
)
```

`[:,1:,:]` 保留所有环境、除最旧格以外的9帧、全部特征，形状 `(N,9,33)`。`unsqueeze(1)` 把 `(N,33)` 变成 `(N,1,33)`，增加一帧长度的时间维度。

`torch.cat` 沿指定轴拼接已有维度；第1轴长度从9加1成为10，其他轴的大小必须一致。这一步生成新张量，并没有混合不同特征的数值。例如简化成每帧1个数、只存3帧，旧 `[1,2,3]` 加入4后得到 `[2,3,4]`。实际代码保存10帧，每一格是一整行33个数。[torch.cat官方说明](https://docs.pytorch.org/docs/2.11/generated/torch.cat.html)

### 5.3 开局时用本次帧填满

```python
repeated_current = current_obs.unsqueeze(1).expand(-1, self.history_length, -1)
```

`expand`把长度1的时间轴按需展开成10，`-1`表示该轴保持原长度。它通常以共享存储的视图表达重复值，不像 `repeat` 那样复制每一格。这里随后交给 `torch.where` 生成结果，代码没有对展开视图逐格修改。

```python
self._history = torch.where(
    (episode_length <= 1)[:, None, None],
    repeated_current,
    shifted,
)
```

条件最初为 `(N,)`；`[:,None,None]` 插入两个长度1的轴，成为 `(N,1,1)`，广播到每台机器人的整块 `(10,33)`。`None` 在这个索引语境中用于新增轴，不是缺失观测。

`torch.where(condition,A,B)` 逐元素选择：条件True取A，False取B。它不是只执行某一条Python分支；A和B在调用前已构造完成。本代码使回合步数0或1的行取“当前帧重复十次”，其他行取“9帧旧记录加1帧新记录”。不同机器人的回合进度可不同，所以需要逐行条件。[torch.where官方说明](https://docs.pytorch.org/docs/2.11/generated/torch.where.html)

## 6. 沿实际环境step看开局和正常推进

环境先 `reset()`，此时缓存清零，得到当前观测 $x_0$。之后每次 `step` 开头传入的仍是上一次返回的policy观测，物理仿真尚未向前推进。

| 调用 | 开头传入的帧 | 开头回合步数 | 写入后的历史 | 物理结束后新当前帧 |
| --- | --- | --- | --- | --- |
| 第1次step | $x_0$ | 0 | 十份 $x_0$ | $x_1$ |
| 第2次step | $x_1$ | 1 | 十份 $x_1$ | $x_2$ |
| 第3次step | $x_2$ | 2 | 九份 $x_1$ 加 $x_2$ | $x_3$ |
| 后续稳定阶段 | $x_t$ | t | 之前9帧加 $x_t$ | $x_{t+1}$ |

因此第3次step返回时，新当前帧是 $x_3$，历史最后一帧是 $x_2$。刚reset后返回的history是零；下一次step才按照开局规则填充。开局的重复帧是明确规定的填充，不表示机器人已经走过这些重复时刻。

每个策略步间隔0.01s，十帧对应过去十个采样位置，最旧帧距当前约0.10s；十个历史采样点彼此首尾间隔0.09s。actor之后用九帧历史加当前帧构造自己的输入，那发生在actor中，缓存本身仍保留这里的过去十帧。

## 7. reset只清指定机器人的历史

`env_ids is None` 时 `zero_()` 原位清全部缓存；有编号时 `self._history[env_ids,:,:] = 0.0` 只清选中行。并行环境中一台跌倒，其他机器人的历史继续保留。

谁执行这次reset？自定义环境在父 `_reset_idx(env_ids)` 完成后显式调用 `self.proprio_history.reset(env_ids)`。普通对象取名为reset不会自动被Lab发现；只有实际调用或管理器接线才会执行它。

`self._history = torch.where(...)` 是让属性指向新的结果张量，`zero_()` 是修改属性目前指向的张量。两种操作都能改变以后 `read()` 看到的内容，区别在于是否保留同一份存储。这个区别也解释了为什么共享延迟编号需要原位更新，而本类内部历史可以重新赋值。

## 源码与API依据

- [本项目环境的推进与reset](../tita_velocity_env.py.md)
- [观测读取函数](observations.py.md)
- [PyTorch Tensor.expand](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.expand.html)
- [PyTorch Tensor.clone](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.clone.html)
