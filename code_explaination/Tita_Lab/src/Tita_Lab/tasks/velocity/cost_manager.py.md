# cost_manager.py：六个cost函数怎样成为一份稳定的训练输出

内容纯AI生成，由人类要求生成，未经人类修改。源码及学习用途核对：2026-11-14。

被解释文件：[cost_manager.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/cost_manager.py)。

本文件把六个原始cost统一调用、各乘一次策略周期、保存本步结果和回合累计。读完应能从配置找到运行实例，理解@property、列表推导式、view和逐列写入，并追清reset前后为什么必须保留本步矩阵。

本页说明环境端怎样提供cost和k/d；它们进入GAE与策略目标后的含义，先看[一批经历怎样完成策略更新](../../../../../../docs/一批经历怎样完成策略更新.md)。本管理器的回合统计与学习端的累计cost估计并非同一个量，第9节会分别说明。

本文解释当前文件及本地 IsaacLab-main / Sim 6.2 调用顺序。只读检查与用户已有200步日志分别陈述，不把它们当成完整训练或接触物理验证。

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
```

## 2. 先确定这个管理器负责哪一段

[costs.py](mdp/costs.py.md)中的函数每次返回 `(N,)` 原始值。本管理器按固定顺序排列成 `(N,6)`，交给环境 `extras["costs"]`，wrapper再把它作为单独的cost输出。

它不决定动作、不求策略梯度，也不训练cost critic。cost critic的六个预测值与本管理器的六个本步测量值有不同角色：一个预测未来累计，一个提供当前观测到的代价数据。

`CostTermCfg`是每一项的配置类型，继承 `ManagerTermBaseCfg` 的func与params，再增加k、d两个算法参数。`TitaCostManager`继承 `ManagerBase`，复用Lab的名称解析、参数检查和类实例化，而六列收集逻辑是项目自己实现的。

## 3. 构造时，空列表怎样在super期间填好

```python
self._term_names = []
self._term_cfgs = []
super().__init__(cfg, env)
```

两个属性必须先创建，因为父 `ManagerBase.__init__` 内会调用 `self._prepare_terms()`。`self`仍是当前 `TitaCostManager`，所以调用的是本类覆盖的方法，而不是先等待子类构造函数结束。

父类保存环境引用、复制配置，然后 `_prepare_terms`执行：

```python
for name, term in vars(self.cfg).items():
    ...
    self._resolve_common_term_cfg(name, term)
    self._term_names.append(name)
    self._term_cfgs.append(term)
```

`vars(self.cfg)`取得配置实例属性字典，`.items()`依次给 `(名字,配置对象)`，循环将它们解包为name与term。值为None表示禁用，`continue`跳到下一项；类型不为 `CostTermCfg` 则报错。

`_resolve_common_term_cfg`沿用父类实现，检查func签名与params，并在物理对象就绪时解析 `SceneEntityCfg` 的关节/刚体编号；类项如 `JointAccelerationCost`变为可调用实例。它不会因为被收进列表就自动在后台持续计算，持续调用发生在后面的compute。

`append`把一项放到列表末尾。`_term_names[index]`与 `_term_cfgs[index]`保持对应；顺序来自当前配置声明：加速度、关节速度、轮接触力、位置限位、stumble、力矩限位。

父构造返回后，这两个列表已填好，随后才能按项数建立缓冲。这个执行顺序是继承与方法覆盖的具体作用，不是空列表被Python自动补全。

## 4. 两份(N,6)各存什么

```python
self.cost_buf = torch.zeros((env.num_envs, len(self._term_names)), device=env.device)
self._episode_sums = torch.zeros_like(self.cost_buf)
```

`cost_buf[n,i]`是第n台机器人最近完成策略步的第i类cost。`_episode_sums[n,i]`是同一机器人当前回合到目前为止的累计值。两者都是 `(N,6)`，内容含义不同。

例如同一列连续三步为0.01、0.02、0，则第三步 `cost_buf=0`，`_episode_sums=0.03`。reset时清累计值，但保留刚结束那一步的 `cost_buf`，因为该步数据还要返回训练端。

张量在 `env.device` 上建立，以便同原始cost直接运算；`zeros_like`保留形状、类型与设备。这里没有神经网络参数，缓冲不会因优化器训练自动调整。

## 5. k/d的一行代码包含三个普通步骤

```python
self.k_values = torch.tensor([term.k for term in self._term_cfgs], device=env.device).view(1, -1)
self.d_values = torch.tensor([term.d for term in self._term_cfgs], device=env.device).view(1, 1, -1)
```

`[term.k for term in ...]`是列表推导式：遍历六份配置、取每项k、按同一顺序放入列表。当前得到 `[0.1,0.3,0.1,0.3,0.1,0.3]`；d列表为六个0。

`torch.tensor(...)`把Python数列建立为Torch张量，最初形状 `(6,)`。`view`以新形状查看同一批元素：`view(1,-1)`得到 `(1,6)`，`view(1,1,-1)`得到 `(1,1,6)`，其中−1让Torch推算剩余轴长度。

增加长度1的轴便于与含环境/时间轴的六列数据广播，例如 `(N,6)`乘 `(1,6)`会对每行使用同一组k。本管理器没有把k/d乘进 `cost_buf`，它只保存配置，并经wrapper的cost_k、cost_d交给训练侧。当前check_policy_update再把两者整理成 `(6,)`，逐类构造约束目标。

在当前check中，k是每类正代理违反量的惩罚权重，d按折扣累计cost的允许上限解释；d不是本步cost的物理判定阈值。二者是固定配置张量，不是nn.Parameter，也没有通过优化器或乘子更新来学习。策略损失先对每类违反量取正部，再乘对应k并求和；六列原始cost仍分别保留供GAE使用。具体公式与当前估计的局限见[check第17节](../../../../scripts/check_np3o_interface.py.md)。

## 6. `@property`为何让active_terms不用括号

```python
@property
def active_terms(self) -> list[str]:
    return self._term_names.copy()
```

Python的property把方法变成属性读取入口。写 `manager.active_terms` 时执行这个方法，返回名字列表；不需要再写 `()`。普通compute仍写 `manager.compute()`，因为它没有这个装饰器。

`list[str]`是返回类型注解，表达结果为字符串列表。`.copy()`复制列表容器，读取者在返回列表中append或删除项，不会把管理器自己的列序改掉。字符串对象不可变，这里无需为名字再深复制。

## 7. compute逐列调用原始项，并且只乘一次dt

### 7.1 本步是否已经计算过

```python
if self._last_step == self._env.common_step_counter:
    return self.cost_buf
```

`common_step_counter`每次策略step加1；它与每台机器人的回合长度分别计数。`_last_step`表示缓冲最后对应哪个公共策略步。相等时直接返回缓存，既不重复差分，也不重复加回合累计。

初始化把_last_step设成当前公共步号，初次reset因此返回全零缓存，不把“建立环境”算成一次训练交互。

### 7.2 逐列循环不等于逐机器人循环

```python
for index, term in enumerate(self._term_cfgs):
    value = term.func(self._env, **term.params)
    self.cost_buf[:, index] = value * self._env.step_dt
```

`enumerate`每轮同时给出列号index和对应配置term。`**term.params`将字典变成命名实参。普通函数直接计算，类实例则执行其 `__call__`。每次调用同时计算N台机器人的一项，返回 `(N,)`；形状不符立即报错。

数学上：

$$
C_{n,i,t}=\Delta t\,f_i(s_{n,t},a_{n,t}),\qquad \Delta t=0.01.
$$

如果原始指示值为1，本步这一列为0.01。原始项不先乘dt，wrapper不再乘，当前manager是唯一缩放处。

### 7.3 整份本步结果再加入回合累计

```python
self._episode_sums.add_(self.cost_buf)
self._last_step = self._env.common_step_counter
```

`add_`原位逐元素加，表达 $S_{n,i}\leftarrow S_{n,i}+C_{n,i,t}$。完成全部列计算后再标记步号，后续同一步读取复用完整矩阵。

返回的是 `cost_buf`本身，不是自动冻结的副本。环境用 `.clone()`保存返回给训练端的数据，因此下一步覆盖manager缓存时不会改掉上一步已经交出的cost。

## 8. 为什么reset前计算、reset后只清累计与差分参考

一次物理推进使机器人接触过大，同时触发失败：

1. 父环境尚未改姿态/速度，自定义 `_reset_idx`先调用cost_manager.compute，保存失败瞬间六项并加入旧回合累计。
2. 父reset把该机器人放回初始状态，重置其他管理器。
3. 本管理器reset读取旧回合累计，形成日志，再清选中行累计。
4. 对有状态的cost实例调用reset，用新关节速度更新差分参考。
5. step结尾再调compute时公共步号相同，直接返回第1步保存的失败cost。

若等到第5步才首次计算，读到的可能已是重置后的安全状态；若reset把cost_buf也清零，失败cost同样会丢失。缓存与累计分别保留，正是为满足两个不同时间的使用要求。

## 9. reset统计不是按实际存活时间平均

```python
self._episode_sums[selection, index].mean() / self._env.max_episode_length_s
```

先对本次重置的M台机器人取该列累计的均值，再除配置的最大回合时长20s。公式为：

$$
L_i=\frac{1}{20}\frac1M\sum_{n\in\mathcal E_{reset}}S_{n,i}.
$$

某回合只活2秒也仍除20，沿用旧口径。若两台累计0.2、0.6，则均值0.4，日志值0.02。不能把这个日志数字直接称为“每台实际存活时间内的违规比例”。

返回字典的键为 `Episode_Cost/项名`。环境先让父类建立本轮 `extras["log"]`，再用 `.update()`加入这些统计。`reset`不是只清数据的动作，它还返回刚结束回合的报告；副作用与返回值同样各有作用。

这份_episode_sums是本回合已发生cost的**不折扣累加**，用途是报告。当前策略更新的initial_cost则取storage.cost_returns[0]的环境均值，来自折扣、GAE和旧critic自举，用于近似累计cost。check没有拿这里的Episode_Cost日志代替initial_cost。两者也都不同于cost_loss_sum：后者累计的是若干次优化器更新前的损失数值，仅供打印。先分清数据来自哪段计算，再判断“累计”的含义。

## 源码与API依据

- [Lab ManagerBase：父构造调用_prepare_terms](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/manager_base.py)
- [本项目环境中的调用位置](tita_velocity_env.py.md)
- [本项目六项原始cost](mdp/costs.py.md)
- [PyTorch Tensor.view](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.view.html)
- [Python property](https://docs.python.org/3.12/library/functions.html#property)
