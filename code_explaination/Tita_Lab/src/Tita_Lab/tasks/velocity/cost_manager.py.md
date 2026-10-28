# cost_manager.py：六列cost的保存与重置

**阅读目标：** 理解一个自定义管理器如何从六项配置建立计算对象，每步输出(N,6)，并累计、清理各回合统计。

目标源码目录：Tita_Lab/src/Tita_Lab/tasks/velocity/。
目标文件：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/cost_manager.py`。

更新：2026-10-22。四处笔误均已由用户修正，当前4环境/200步真实cost输出检查通过，日志与修正经过见[18](../../../../../../docs/migration/18_cost接线检查与首次运行.md)。下面参考实现的计算与当前源码一致，格式可能不同；助手未改源码。

```python
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

    # k、d留给约束算法；它们不参与本文件的cost数值计算。
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
        # 每台机器人在当前回合内各项cost的累计值，(N, 6)。
        # 每步累加已乘step_dt的cost，reset只清选中机器人的行。
        self._episode_sums = torch.zeros_like(self.cost_buf)
        # 与旧算法的广播形状一致：(1, 6)和(1, 1, 6)。
        self.k_values = torch.tensor([term.k for term in self._term_cfgs], device=env.device).view(1, -1)
        self.d_values = torch.tensor([term.d for term in self._term_cfgs], device=env.device).view(1, 1, -1)
        # 初次reset不算一次训练步；同一策略步重复调用也不能重复累加。
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
            # 沿用旧Gym：回合累计值除以最大回合秒数，非实际存活秒数。
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
            # 原生接口负责解析关节/轮体名称，并将类对象变成实例。
            self._resolve_common_term_cfg(name, term)
            self._term_names.append(name)
            self._term_cfgs.append(term)
```


## 1. 配置保存内容，管理器负责执行

CostTermCfg是“一项的设置”：继承func（计算函数或类）与params（给函数的参数），增加k/d。CostsCfg中acc_smoothness、pos_limit等属性，各保存一个这样的配置对象。@configclass让这些配置可按Lab的配置方式构造、复制、验证；它不负责每步执行cost。

TitaCostManager是环境持有的一个对象：self.cost_manager = TitaCostManager(self.cfg.costs, self)。构造函数中的cfg收到六项配置，env收到外部环境对象；后一个self与管理器方法中的self不是同一个对象。管理器方法里的self表示管理器，self._env指向环境，所以能读到同一批机器人的最新状态。

## 2. 空列表是在super执行期间填满的

```python
self._term_names: list[str] = []
self._term_cfgs: list[CostTermCfg] = []
super().__init__(cfg, env)
```

类型注解说明列表元素类型；=[]才实际建立空列表。父类保存配置副本与环境引用，再调用self._prepare_terms()。这里self仍是当前TitaCostManager对象，因此调用下面重写的方法，填入六个名称和配置，随后才从super返回。本地依据：ManagerBase.__init__，不是Python自动按函数名搜索。

```python
for name, term in vars(self.cfg).items():
    self._resolve_common_term_cfg(name, term)
    self._term_names.append(name)
    self._term_cfgs.append(term)
```

vars(cfg)取得实例属性字典，items()依次给出属性名与属性值。name例如"pos_limit"，term是该项CostTermCfg对象，含func/params/k/d；term只是循环变量名。实际实现另有None和类型检查。通用解析会把SceneEntityCfg里的关节/轮体名称解析成索引，遇到类项则构造实例并放回term.func。

两列表一一对应：_term_names[3]是"pos_limit"，_term_cfgs[3]是其配置。列表长6，和环境数量N无关。下划线前缀表示内部使用的约定，不是Python强制禁止访问。

## 3. k/d一行代码分成三步

```python
values = []
for term in self._term_cfgs:
    values.append(term.k)
k_tensor = torch.tensor(values, device=env.device)
self.k_values = k_tensor.view(1, -1)
```

values是[0.1,0.3,0.1,0.3,0.1,0.3]，tensor形状(6,)，view后形状(1,6)。view不改变六个数，-1让PyTorch按总元素数推断该维长度；1×6=6。同理view(1,1,-1)是(1,1,6)。这些是六类cost的算法参数，没有在当前compute中乘进cost。

旧NP3O接收k，旧RolloutStorageWithCost中cost_returns是(T,N,6)，减d时(1,1,6)可沿时间和环境维使用同一组阈值。这说明形状来自后续使用位置；此处先保持迁移约定，不展开算法公式。

## 4. @property让方法按属性方式读取

```python
@property
def active_terms(self) -> list[str]:
    return self._term_names.copy()
```

读取manager.active_terms会执行方法体并取得列表。不能写manager.active_terms()，否则相当于先取得列表、再把列表当函数调用。没有setter时，外部不能直接给active_terms赋值。copy()返回新列表，外部append不会更改管理器内部名单。每次访问都会执行方法体，不是提前计算后自动缓存。父类num_envs/device也提供类似属性。

## 5. 本步矩阵与回合累计矩阵

```python
self.cost_buf = torch.zeros((env.num_envs, len(self._term_names)), device=env.device)
# 每台机器人在当前回合内各项cost的累计值，(N, 6)。
self._episode_sums = torch.zeros_like(self.cost_buf)
```

zeros_like建立与cost_buf形状、dtype、device相同的新零张量，两个矩阵分别保存数据。cost_buf每步覆盖；_episode_sums每步累加，只在对应机器人reset时清其所在行。episode是一次回合，sums是累计值。

只看位置cost列，N=2时：第1步本步[0.01,0]、累计[0.01,0]；第2步本步[0.01,0.01]、累计[0.02,0.01]。此时只重置环境0，累计变成[0,0.01]，本步矩阵保留原值。这是已乘step_dt后的累计，不是纯次数，也没有折扣。

## 6. compute逐列调用，函数内部同时计算N台机器人

```python
for index, term in enumerate(self._term_cfgs):
    value = term.func(self._env, **term.params)
    self.cost_buf[:, index] = value * self._env.step_dt
self._episode_sums.add_(self.cost_buf)
```

enumerate同时提供0到5的列编号与配置对象。**term.params把字典展开成关键字参数；若params={"asset_cfg": robot_cfg}，该调用等价于term.func(self._env, asset_cfg=robot_cfg)。value为(N,)，[:,index]选所有机器人对应的同一列。(self.num_envs,)中的逗号表示一个元素的tuple，用于检查一维shape。

循环只遍历六项cost；每个函数内部用张量同时计算N台机器人。add_在原累计张量上加本步矩阵，return self.cost_buf返回已有张量；环境再clone一个副本用于本次输出。

_last_step保存已经计算的common_step_counter。进入compute发现编号相同就直接返回，不再调用函数、更新加速度历史或累加统计；策略步编号增加后才计算新的一步。

## 7. reset明确分两种状态处理

```python
selection = slice(None) if env_ids is None else env_ids
```

条件表达式：env_ids为None就用全选，否则用给定编号。[:,index]选一列，selection让统计只取结束的机器人。mean对这批机器人的同一项累计值求平均，再除最大回合20秒，写入普通字典statistics。

随后清这些机器人的累计行。isinstance(term.func, ManagerTermBase)判断该项是否是有reset接口的类实例；只有这类项才调用reset(env_ids)。加速度项因此在机器人已经重置后更新速度参考。cost_buf不清，因为刚结束的一步仍要返回给训练端。

管理器reset没有自动重置机器人；机器人由环境父类重置。本对象只处理自己保存的状态。

## 8. 环境决定调用时机

构造环境时建立一个cost_manager。普通步在super().step返回后compute；遇到终止则在_reset_idx的super之前compute，机器人重置后再调用cost_manager.reset；最终step返回前再compute会复用本步值。最后extras["costs"]持有(N,6)副本。

调用都由TitaVelocityEnv中的明确语句接入。继承ManagerBase不会自动让这个对象参与环境每步和reset循环。

## 计算与累计分别保存什么

设第 i 个原始函数在第 n 个环境返回 $f_{n,i}(t)$，每策略步的矩阵和回合累计为：

$$
C_{n,i}(t)=\Delta t\,f_{n,i}(t),\qquad
S_{n,i}\leftarrow S_{n,i}+C_{n,i}(t).
$$

C对应cost_buf，S对应_episode_sums，两者都是(N,6)。_last_step防止同一common_step_counter重复执行右边的累加。reset只清选中行的S，并保留刚结束一步的C供训练读取。

对本次重置集合 $\mathcal R$，日志第i项使用：

$$
\mathrm{log}_i=\frac{1}{\lvert\mathcal R\rvert\,T_{max}}
\sum_{n\in\mathcal R}S_{n,i}.
$$

分母是最大回合秒数，不是每个回合实际存活时长；这是沿用旧统计规则。k/d不出现在这些环境收集式里。

## 补充问答

**Q：** 还有个关键变量你没给注释：_episode_sums

**A：** 它存每台机器人本回合每项cost的累计值，形状(N,6)。本步cost_buf覆盖写，_episode_sums累加；仅在对应机器人结束并记录统计后清选中行。关键注释已经补入本页和17的参考实现，用户源码不由助手改动。

**Q：** 自己写manager和cfg我真的有点心理恐惧了，讲一下这超长代码吧？这里可能RL原理占小头，编程层面的思维逻辑和流程占大头。

**A：** 看清四个职责即可：CostTermCfg存一项设置，CostsCfg把六项放一起，TitaCostManager调用六项并保存结果，TitaVelocityEnv安排调用时机。当前新增的主要是这层配置、执行与状态保存，不需要先理解NP3O优化公式。
