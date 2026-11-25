# tita_velocity_env.py：一次step怎样安排动作、历史、奖励、cost与reset

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-02。

被解释文件：[tita_velocity_env.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/tita_velocity_env.py)。

本文件在Lab通用环境上增加TITA需要的调用顺序。读完应能跟随一次step区分策略时间与物理时间，知道父类完成哪些工作、自定义代码补哪几处，以及各类reset由谁明确调用。第2节先建立完整顺序，再沿源码逐段解释。

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

from isaaclab.envs import ManagerBasedRLEnv, VecEnvStepReturn

from .cost_manager import TitaCostManager
from .mdp.observation_history import ProprioHistory

if TYPE_CHECKING:
    from .config.Tita.env_cfg import VelocityEnvCfg


class TitaVelocityEnv(ManagerBasedRLEnv):
    """TITA environment with observation history and constraint costs."""

    def __init__(
        self,
        cfg: VelocityEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize task histories and cost collection."""

        # 父类建立动作/观测管理器时会读取这两份数据，故先创建。
        self.proprio_history = ProprioHistory(
            num_envs=cfg.scene.num_envs,
            obs_dim=33,
            history_length=10,
            device=cfg.sim.device,
        )
        # Shared history-slot indices, shape (N,); sampled by a startup event.
        self.action_lag_indices = torch.zeros(cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device)

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        # 此时机器人和物理数据已建立，可以解析cost中的关节/轮体。
        self.cost_manager = TitaCostManager(self.cfg.costs, self)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Advance the environment and return task rewards and costs."""

        # 此时 obs_buf 保存的仍是上一次返回的观测。
        self.proprio_history.read_then_push(
            self.obs_buf["policy"],
            self.episode_length_buf,
        )

        obs, reward, terminated, time_outs, extras = super().step(action)

        # 找到本步已经计算好的失败奖励。
        termination_index = self.reward_manager.active_terms.index("termination")
        termination_reward = self.reward_manager._step_reward[:, termination_index] * self.step_dt

        # 普通项合计夹到非负，再加回失败奖励。
        ordinary_reward = reward - termination_reward
        reward.copy_(ordinary_reward.clamp_min(0.0) + termination_reward)

        # 无reset时在这里计算；有reset时返回reset前保存的本步值。
        # clone让本次返回的矩阵不被下一步覆盖，形状(N, 6)。
        extras["costs"] = self.cost_manager.compute().clone()

        # 检查本步返回的cost：每台机器人六项，数值有限且非负。
        costs = extras["costs"]
        assert costs.shape == (self.num_envs, 6)
        assert torch.isfinite(costs).all(), "Non-finite cost detected."
        assert (costs >= 0.0).all(), "Negative cost detected."

        return obs, reward, terminated, time_outs, extras

    def _reset_idx(
        self,
        env_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Reset selected robots and their task histories."""

        # 重置物理状态前，保留刚结束这一步的六维观测样本。
        self.observation_manager.compute_group("privileged_state")

        # 保存失败瞬间的cost，供本步step返回。
        self.cost_manager.compute()

        super()._reset_idx(env_ids)
        self.proprio_history.reset(env_ids)

        # 父类创建本轮log后，再加入cost统计并更新加速度差分基准。
        self.extras["log"].update(self.cost_manager.reset(env_ids))
```

## 2. 一次step在训练闭环里的位置

当前包装关系是：A为真正的 `TitaVelocityEnv`，B为 `gym.make()`返回的Gymnasium接口层，C为项目的 `TitaNP3OEnvWrapper`。训练或检查循环调用C.step，C调用B.step，最终进入本文件A.step。包装不会复制出三套物理机器人。

一次策略step接收 `(N,8)` 动作，推进4个0.0025s物理步，因此策略周期0.01s。N台机器人批量推进；N不把每台机器人的时间变成N倍。回合上限20s对应最多2000个策略步。

当前主要调用顺序如下，表中的每一行都对应自定义环境或父类step里的实际调用：

| 顺序 | 执行者与操作 | 数据意义 |
| --- | --- | --- |
| 1 | 本类更新ProprioHistory | 把上次返回的policy帧放进过去历史 |
| 2 | 父类ActionManager.process_action | 拆分8列、裁剪/FIR/缩放，一次 |
| 3 | 父类物理循环4轮 | 动作项apply写延迟目标，执行器计算力矩、Sim推进、Scene更新 |
| 4 | 父类更新计数器 | 每环境回合步数+1，公共策略步号+1 |
| 5 | TerminationManager、RewardManager | 计算本步结束条件与原始加权奖励 |
| 6 | 若有结束行，调用本类_reset_idx | 先保存6维状态和cost，再执行父reset |
| 7 | CommandManager、interval事件 | 更新下一份命令，必要时推扰 |
| 8 | ObservationManager | 生成将返回的新观测，已重置行取新回合状态 |
| 9 | 本类恢复旧TITA总奖励规则 | 普通项合计夹零，加失败项 |
| 10 | 本类读取/保存cost | 本步只计算一次，clone后放extras |
| 11 | wrapper整理数据 | 五组观测拼586，cost独立返回 |

表格省略记录器/GUI手动reset等分支，常规失败/超时路径可据此逐步阅读。具体代码以本地父类为准，不能用旧Lab的循环顺序代替。

## 3. 初始化为何一部分在super前，一部分在后

先创建 `ProprioHistory(N,33,10,device)` 与 `(N,)` 的 `action_lag_indices`，然后调用父构造函数。

父构造会建立Scene、ActionManager、ObservationManager等对象。动作项构造时引用共享延迟编号；观测管理器准备历史项时会调用 `mdp.proprio_history(env)`探测输出形状。因此这两份对象必须先存在。父构造中的self仍指当前子类实例，任何对这些属性的访问都会来到这一实例上。

`super().__init__(cfg=cfg,render_mode=render_mode,**kwargs)`沿继承顺序调用父方法；`**kwargs`把收到的额外关键字参数继续传递，不是把cfg转换成字典。

父构造结束后机器人与物理数据已经建立，才创建 `TitaCostManager(self.cfg.costs,self)`，解析cost需要的关节、轮体和有状态类。它是项目额外管理器，所以父框架不会凭名字自动添加其step/reset调用，本类在下面明确接线。

## 4. step开头：当前动作使用的新观测，怎样变成下一次的历史

此时 `self.obs_buf["policy"]`仍是上次reset/step返回的帧 $x_t$。本类把它放进历史，然后才 `super().step(action)`推进物理。父类最后返回新当前帧 $x_{t+1}$，历史末尾是 $x_t$。

`read_then_push`返回旧历史，但这里没用变量接收；它同时修改了 `self.proprio_history`内部存储，这个修改就是环境需要的效果。Python函数返回值与对象修改可以同时存在，忽略返回值不等于函数没运行。

历史有开局填充和选中行reset规则，详见 [历史缓存](mdp/observation_history.py.md)。父框架的观测getter可以多次读取缓存，而step只推进一次，从而把“读数据”和“时间前进”分开。

## 5. 父step计算完后，为什么还要改总奖励

```python
termination_index = self.reward_manager.active_terms.index("termination")
termination_reward = self.reward_manager._step_reward[:, termination_index] * self.step_dt
ordinary_reward = reward - termination_reward
reward.copy_(ordinary_reward.clamp_min(0.0) + termination_reward)
```

`active_terms`是管理器的奖励项名列表，`.index("termination")`查失败项在哪一列。该名称必须与配置一致。`_step_reward`是本地RewardManager内部字段，当前保存每项“原始值×权重”，尚未乘dt，所以读取后要乘0.01恢复本步失败贡献。

令普通项加权后本步合计为 $R_{ordinary}$、失败贡献为 $R_{fail}$，父类总分为两者之和；本类最终要：

$$
R=\max(R_{ordinary},0)+R_{fail}.
$$

例如普通项−0.3、失败项−2，最终−2；普通项+0.5且失败−2，最终−1.5。普通项包括正跟踪奖励与负惩罚，并非“所有叫penalty的函数以外的项”。先把它们合计，再夹零。

`reward.copy_`写回父类那份奖励缓冲，维持调用者与 `self.reward_buf`指向同一份更新后数值。它不会修改各项回合统计；父RewardManager早已按各项贡献累计日志。因此日志各项之和未必等于夹零后训练实际收到的总分。

这里依赖 `_step_reward`内部字段，而非一个跨版本稳定的公开取值保证；以后本地Lab改变该字段含义，需重新核对。当前实现与本地源码对应，不据此新增兼容代码。

## 6. cost的返回先解决“失败那一步不能丢”

```python
extras["costs"] = self.cost_manager.compute().clone()
```

`extras`是附加信息字典，原生五项返回值仍是观测、奖励、失败、超时、extras；本类把六列cost放在字典里，不把原生返回协议直接改成六项。wrapper才把它整理成NP3O的独立cost返回值。

无reset时，此处首次compute，用当前公共步号计算六列并累计；有reset时，`_reset_idx`已在重置前计算，本次compute只取同一步缓存。`.clone()`确保下一步覆盖manager缓冲时，上一次返回的cost仍保留数值。

后面三个assert分别检查shape `(N,6)`、所有值有限、所有值非负。`torch.isfinite`逐元素判断非NaN/非无穷，`.all()`把全部判断AND成一个布尔标量；它们是运行防错，不证明六项公式或物理接触一定正确。细节见 [check脚本](../../../../scripts/check_np3o_interface.py.md)。

## 7. _reset_idx中父方法前后各保存什么

### 7.1 父reset前保存刚完成物理推进的样本

先 `compute_group("privileged_state")`保存本步线速度和接触；该组按公共步号每步只采样一次。再 `cost_manager.compute()`保存本步失败/超限代价。此时物理状态尚未被reset事件改写。

两份缓存服务不同返回数据：6维样本在下一份打包观测中，cost矩阵作为这一转移的代价返回。它们都不能等机器人回到初始姿态后才首次采样。

### 7.2 父reset完成通用流程

`super()._reset_idx(env_ids)`先更新所选环境的地形课程、重置Scene内部状态、执行reset事件，写入新位姿与关节状态；再按顺序重置观测、动作、奖励、课程、命令、事件和结束管理器等，并把回合步数清零。

地形课程的“升级/降级”改变所选地形难度行与下一回合原点，不是改变策略网络级别，也不是保证每个新地形更容易。此处只是说明它在reset里的调用位置，地形参数沿 [terrain_cfg.py](terrain_cfg.py.md) 阅读。

父类不会搜索整个进程里所有名为reset的方法。它对已经持有的管理器逐个调用；管理器再根据自己保存的项调用类实例reset。

### 7.3 父reset之后处理项目自己的状态

`self.proprio_history.reset(env_ids)`只清所选行历史。`self.cost_manager.reset(env_ids)`先报告旧回合累计、清累计，并把加速度cost的速度参考更新成刚重置后的速度。返回的统计用 `self.extras["log"].update(...)`合入父类刚建立的日志。

不能把cost.reset提前到父reset之前，否则差分参考可能仍是旧回合末速度，且父类后续建立新日志时可能覆盖提前写的内容。方法的位置由数据依赖决定，不是统一要求super永远放在第一行。

## 8. 把所有reset按对象归属列清楚

| 要清/更新的数据 | 谁持有 | 明确调用者 |
| --- | --- | --- |
| 机器人初始物理状态 | Scene/Articulation | 父reset中的reset事件 |
| 动作原始缓存、延迟目标历史 | TitaJointAction | ActionManager.reset |
| 奖励加速度参考 | JointAccelerationPenalty | RewardManager.reset |
| 十帧观测 | ProprioHistory | 本类_reset_idx |
| cost回合累计与cost加速度参考 | TitaCostManager/JointAccelerationCost | 本类调用cost_manager.reset，再由manager分发 |
| 6维状态的接触记忆 | TitaPrivilegedState | 继承的reset保留，按现有Gym时序处理 |
| policy/critic权重 | 神经网络对象 | 本文件不重置、不训练 |

`reset`可以表示不同对象各自的初始化操作。同名方法的关系来自明确调用链，不是全局同名广播。

## 9. 返回时，新观测与本步奖励可能跨回合

父环境在同一次step内自动reset结束行，然后计算最终观测。因此该行返回的观测可已属于新回合，但terminated/time_outs仍标记刚结束的旧回合，奖励与cost也对应旧回合最后一步。

6维额外状态另有上文的预采样保留规则。训练存储之后必须按dones、timeouts及最终观测约定处理回报，不能见到新姿态就把上一回合的失败忽略。当前wrapper/check已能传递与检查这些接口，完整NP3O存储与更新仍未实现。

用户11月2日提供的4环境200步输出验证了接口形状与有限值检查未中断，见[证据](../../../../../../docs/audits/2026-11-02_value_networks/README.md)。这不是本轮助手再次运行，也不能覆盖轮接触物理、15秒推扰、20秒回合或训练收敛。

## 源码与API依据

- [Lab ManagerBasedRLEnv：step与_reset_idx](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)
- [Lab ManagerBasedEnv：初始化与管理器建立](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/manager_based_env.py)
- [Lab RewardManager：_step_reward的当前含义](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/reward_manager.py)
- [本项目cost管理器](cost_manager.py.md)
- [本项目观测采样](mdp/observations.py.md)


## 历史原文（折叠）

以下仅保留当时要求归档的原文。旧路径、旧行号及“当前”措辞均属于当时状态，部分链接已失效；学习现有实现以本页上方2026-11-02正文为准。

<details>
<summary>此前创建/重置说明及已修复的缩进问题（历史原文）</summary>

# tita_velocity_env.py：历史缓存的创建与重置

日期：2026-10-17。状态：**用户已写入环境子类和注册入口；当前_reset_idx误嵌套在__init__内部，尚未形成重置方法重写。** 已做当前文件与父类调用链的源码/AST核对，未实例化环境；助手未修改TITA_Lab。历史每步更新和输出尚未接通。

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/`。
源码文件：[tita_velocity_env.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/tita_velocity_env.py:16)。

整体目的先看[迁移入口](../../../../../../docs/migration/README.md)及[05第6.1节](../../../../../../docs/migration/05_observations迁移.md#61-env_cfg-声明环境和管理器执行)：这是旧观测历史的接线方案，沿用既有manager-based流程。

## 当前源码中的缩进问题

当前文件节选，保留错误的缩进以便定位：

```python
class TitaVelocityEnv(ManagerBasedRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):  # 签名在此缩写
        # 此处省略父类初始化与缓存构造
        def _reset_idx(
            self,
            env_ids: Sequence[int] | torch.Tensor,
        ) -> None:
            super()._reset_idx(env_ids)
            self.proprio_history.reset(env_ids)
```

第34行def比__init__多缩进一级，因此定义的是初始化函数内的局部函数，不是TitaVelocityEnv的方法。AST确认类的直接方法只有__init__；这段语法可以解析通过，但环境重置仍会沿用父类方法，自定义缓存不会由此被清空。把def _reset_idx及其函数体整体左移一级，使它与def __init__同级。下方完整参考代码展示修正后的结构；没有代替用户修改源码。

## 目标代码：重置方法位于类方法层级

```python
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

from .mdp.observation_history import ProprioHistory


class TitaVelocityEnv(ManagerBasedRLEnv):
    """Manager-based TITA environment with a proprioceptive history buffer."""

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.proprio_history = ProprioHistory(
            num_envs=self.num_envs,
            obs_dim=33,
            history_length=10,
            device=self.device,
        )

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self.proprio_history.reset(env_ids)
```

## 这一步接通什么

用户已写入ProprioHistory和这里的构造代码，注册入口也已选择子类。设计上这一步让环境对象拥有一个缓存，并负责在指定环境重置时清空它；当前重置部分仍需修正上述缩进。一个向量化环境对象管理N个并行环境，共用一个形状为(N,10,33)的缓存实例。尚未实例化环境，不将构造语句存在记作已运行。

`class TitaVelocityEnv(ManagerBasedRLEnv)` 表示继承：场景、动作、奖励、观测等 manager 及步进逻辑沿用父类。`super().__init__(...)` 先完成官方环境初始化，然后才能用 `self.num_envs` 和 `self.device` 创建匹配环境数量、设备的缓存。`self.proprio_history` 是环境对象持有的 ProprioHistory 实例。

`_reset_idx` 是对父类同名方法的重写：先 `super()._reset_idx(env_ids)` 完成官方的场景、manager 和回合计数重置，再清这些环境的自定义历史。不要只重写公开 `reset()`：自动结束回合也会在 `step()` 内调用 `_reset_idx()`，不会经过公开 `reset()`。

本地源码核对：父类 [构造函数](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:63)、[场景和 manager 初始化](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:132)、[公开 reset 调用处](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:434)、[step 内自动重置调用处](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:265)、[父类 _reset_idx](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:395)。当前正常构造路径没有调用环境的 `_reset_idx`，因此在父类构造完成后创建缓存符合当前生命周期；`sim.reset()` 是仿真初始化，不能等同于环境的 `_reset_idx()`。

配套[注册入口](config/TITA/__init__.py.md)现已由用户修改，直接指向此模块的类；无需给mdp的导出文件添加这个环境类。

完成边界：修正缩进后，源码层面建立“创建缓存、随重置清空”的路径，实际效果仍待运行验证。当前任务没有read_then_push调用，观测出口仍只有已有policy当前帧。后续应复用管理器已经加噪和缩放的同一份观测，在正确时机更新一次并提供历史输出，一并核对观测空间、额外compute与终止观测。当前Events/Rewards/Terminations仍有Cartpole内容；本轮只讲清整体目的并整理文档，不继续新增接线代码。

## 2026-10-17续接：配置、环境与缓存的关系

```python
self.proprio_history = ProprioHistory(
    num_envs=self.num_envs,
    obs_dim=33,
    history_length=10,
    device=self.device,
)
```

上面节选自前文__init__方法中父类初始化之后的语句。env_cfg.py提供场景、动作、观测等配置；运行时的TitaVelocityEnv对象持有scene和各manager。ProprioHistory类定义提供存取方法，调用ProprioHistory(...)才创建真正持有历史张量的实例，并通过self.proprio_history保留在环境对象上。一个向量化环境对象服务N个并行环境，因此创建一个(N,10,33)缓存实例即可。num_envs/device从父类已建立的环境读取，使缓存数量和设备与环境一致。

```python
def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
    super()._reset_idx(env_ids)
    self.proprio_history.reset(env_ids)
```

官方公开reset和step中的自动重置均调用self._reset_idx。当前self是TitaVelocityEnv实例，所以会进入这里的重写方法，即使reset/step方法本身继承自父类。super()._reset_idx(env_ids)继续执行官方重置职责；返回后再清同一批环境的自定义缓存。这保证重置某几个环境时，不影响其余环境的历史。缓存.reset与环境._reset_idx是两个对象上的两个不同方法。

此前entry_point指向官方类；当前用户已改为TITA_Lab.tasks.velocity.tita_velocity_env:TitaVelocityEnv，Gym将据此选择子类。env_cfg_entry_point继续提供原VelocityEnvCfg配置。这个子类沿用ManagerBasedRLEnv的管理器与物理步进流程，此步只补缓存生命周期。

最新只读复核：缓存类、环境子类及新入口都已保存；重置方法缩进待修，更新和输出尚未接。没有实例化环境、重新执行CPU缓存检查或改动TITA_Lab。已有CPU证据只覆盖缓存张量逻辑，不证明上述生命周期接线已运行。

</details>
