# tita_velocity_env.py：历史缓存与最终奖励合成

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/`。
源码文件：[tita_velocity_env.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/tita_velocity_env.py)。

更新：2026-10-19。历史接线与最终奖励合成step均已保存，当前AST与此前方案对应；真实RL环境尚未运行。普通项、active_terms、step_dt及裁剪动机见末尾原问题Q/A。

## 1. 当前代码

```python
from collections.abc import Sequence

import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg, VecEnvStepReturn

from .mdp.observation_history import ProprioHistory


class TitaVelocityEnv(ManagerBasedRLEnv):
    """TITA environment with proprioceptive observation history."""

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:

        # 缓存要在 super().__init__() 之前创建
        self.proprio_history = ProprioHistory(
            num_envs=cfg.scene.num_envs,
            obs_dim=33,
            history_length=10,
            device=cfg.sim.device,
        )

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # 此时 obs_buf 保存的仍是上一次返回的观测。
        self.proprio_history.read_then_push(
            self.obs_buf["policy"],
            self.episode_length_buf,
        )

        obs, reward, terminated, time_outs, extras = super().step(action)

        # 找到本步已经计算好的失败奖励。
        termination_index = self.reward_manager.active_terms.index("termination")
        termination_reward = (
            self.reward_manager._step_reward[:, termination_index] * self.step_dt
        )

        # 普通项合计夹到非负，再加回失败奖励。
        ordinary_reward = reward - termination_reward
        reward.copy_(ordinary_reward.clamp_min(0.0) + termination_reward)

        return obs, reward, terminated, time_outs, extras

    def _reset_idx(
        self,
        env_ids: Sequence[int] | torch.Tensor,
    ) -> None:

        super()._reset_idx(env_ids)
        self.proprio_history.reset(env_ids)
```

## 2. 构造时先准备缓存

环境对象通过self.proprio_history持有一个ProprioHistory实例。一个环境对象管理N个并行环境，因此这一份缓存形状为(N,10,33)。

缓存使用传入cfg.scene.num_envs和cfg.sim.device创建，随后super初始化官方环境与manager。ObservationManager在构造期间会调用history观测函数推导输出形状，因而此时缓存已经需要存在。

## 3. 每次step先保存上一份policy

进入step时，self.obs_buf["policy"]保存上一次返回的观测o_t，episode_length_buf对应同一时刻。先将它写入历史，再由父类处理本次动作、推进物理和生成o_(t+1)。

```text
上次返回：policy=o20，history=o10…o19
本次step开头保存o20：缓存变o11…o20
父类生成o21：本次返回policy=o21，history=o11…o20
```

输入复用管理器已经加噪和缩放的policy。这里使用read_then_push的更新效果；MDP函数另外通过read只读取数，所以额外观测计算保持同一历史窗口。当前每策略步的推进位置只有这里。

## 4. 重置指定环境的历史

_reset_idx先调用父类，执行官方场景、Event及各manager的重置，再清相同env_ids的缓存。公开reset和step中的自动结束都经过这条路径。随后观测得到零历史，下一次step按旧开局规则回填。

## 5. 接线与验证范围

当前任务注册入口指向TitaVelocityEnv；env_cfg继续提供配置，history观测函数读取缓存并返回(N,330)。普通PPO配置当前只消费policy组，模型使用history仍待迁移。

此前CPU原型`/tmp/tita_history_integration_20261017/`以3环境、36步及异步重置检查返回窗口，额外读取不推进；使用真实Lab拼接方法与父环境替身，未实例化真实RL环境或加载动力学。

完整Gym/Lab时序、配套文件及源码入口见[05 Observations](../../../../../../docs/migration/05_observations迁移.md#5-历史lab-中怎样接入)。

## 6. 当前实现：处理最终奖励

当前保存的step包含以下最终分数处理；__init__与_reset_idx继续负责已有历史接线：

```python
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.proprio_history.read_then_push(
            self.obs_buf["policy"],
            self.episode_length_buf,
        )

        obs, reward, terminated, time_outs, extras = super().step(action)

        # The native cache stores func(...) * weight, without step_dt.
        termination_index = self.reward_manager.active_terms.index("termination")
        termination_reward = (
            self.reward_manager._step_reward[:, termination_index] * self.step_dt
        )
        ordinary_reward = reward - termination_reward
        reward.copy_(ordinary_reward.clamp_min(0.0) + termination_reward)

        return obs, reward, terminated, time_outs, extras
```

前半段沿用历史推进，父类计算奖励并完成重置/观测；后半段只重组最终分数。active_terms.index("termination")定位失败项列，_step_reward保存func乘weight，所以乘step_dt取回该项本步贡献。总分减去失败项得到普通合计，普通合计夹非负后加回失败项。

reward与环境/管理器的总分缓冲区共享存储，copy_写回以保持一致。取缓存不会再次执行JointAccelerationPenalty，因此不会把前次速度参考推进两次。当前管理器reset不清本步缓存，自动重置后仍能取刚才的失败贡献；_step_reward是内部字段，后续升级核对其单位与重置行为。

配置仍10项，其他文件不必为这次合成增加条目。Gym原过程、逐项日志和当前默认Recorder边界见[08第10节](../../../../../../docs/migration/08_rewards迁移.md#10-将各项合成为最终分数)。此前方案的4轮CPU父类/场景替身与旧Gym公式对照通过；当前step已保存且AST对应，本轮未运行仿真。

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> 我没搞懂。为啥会调用这个？

**A：** 你问的是父类初始化时为何会执行mdp.proprio_history。ObservationsCfg.HistoryCfg把函数交给ObsTerm.func，ObservationManager准备该项时执行func(当前环境, **params)，根据返回张量.shape确定维度。这个回调需要读取self.proprio_history，所以缓存创建在super之前。带真实调用语句及self归属的详细解释见[05末尾问答](../../../../../../docs/migration/05_observations迁移.md#补充问答)。


**Q：**

> 还有reset,我记得obs hist也写了reset。有这么多reset，我没搞懂啊，我还记得reset是lab的方法，我们重写了是吗？是环境重置的时候，会自动调用所有被登记的reset？这个代码框架有点大，说实话我都忘了obs hist文件里头的内容拿来干啥了。

**A：** 当前自定义环境写的是：

```python
def _reset_idx(self, env_ids):
    super()._reset_idx(env_ids)
    self.proprio_history.reset(env_ids)
```

第一行调用父环境的_reset_idx，进入Lab现有的场景、Event和manager重置流程；第二行显式调用自定义历史对象的reset。这两个调用处理不同对象的状态。

| 方法 | 是否重写基类 | 清理什么 | 谁调用它 |
| --- | --- | --- | --- |
| TitaVelocityEnv._reset_idx | 重写ManagerBasedRLEnv._reset_idx | 协调父类流程，再清历史 | 官方公开reset或step的结束处理 |
| JointAccelerationPenalty.reset | 重写ManagerTermBase.reset | 选中环境的上一关节速度参考 | RewardManager的类项循环 |
| ProprioHistory.reset | 普通自定义方法；此类不继承Lab基类 | 选中环境的十帧观测缓存 | 上述self.proprio_history.reset(env_ids) |

父环境中真正调用其他对象的方法是：

```python
self.observation_manager.reset(env_ids)
self.action_manager.reset(env_ids)
self.reward_manager.reset(env_ids)
```

这里只摘与当前问题直接相关的调用，实际还包含Scene、Event、命令等。RewardManager随后遍历自己保存的类项配置，调用奖励实例.reset；ObservationManager清理自己管理的类项、延迟和原生历史。当前ProprioHistory既未继承ManagerTermBase，也未作为类项登记进manager；它靠子类代码的第二行明确接线。没有一个全局机制会因为方法名叫reset就自动调用。

Lab公开reset()和每步自动结束都会进入self._reset_idx，Python方法分派选中子类的重写；子类用super明确保留父类流程。本项目没有重写环境的公开reset方法。源码：[公开reset入口](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:434)、[step入口](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:265)、[父类依次重置manager](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:414)。

ProprioHistory用于保存过去十帧、每帧33维的自身观测。环境对象持有一个(N,10,33)缓存；step更新，mdp.proprio_history读取并展平为(N,330)，ObservationManager把它放到obs["history"]中。history提供旧TITA后续历史编码网络需要的时间窗口；当前生成的普通PPO仍只读policy，模型消费history尚未迁移。

两份跨步状态都要按同一env_ids处理，但内容不同：奖励缓存同步为重置后的关节速度，以免跨回合计算加速度；历史缓存清零，以免新回合读到旧回合轨迹。单个环境结束只清它对应的行，其他并行环境继续保留各自数据。

**Q：**

> 现在是新的step的问题。我没搞懂什么叫普通项？为啥要 * step_dt？active_terms指的就是在rewardcfg里写的项？
> 还有为啥要这个计算公式啊，你说的旧gym的那个，他为啥这样搞？

**A：** 普通项在当前合成中指除termination以外的9项，含负权重惩罚。active_terms是管理器收集的奖励项名字列表，当前index("termination")为9，对应(N,10)缓存的第9列。父类reward已经累计raw*weight*step_dt；缓存保存的是raw*weight，所以从缓存取失败贡献时才乘step_dt恢复本步分数。接着用reward-termination_reward拆出普通合计，只夹这部分再加回失败贡献。

旧配置注释称夹普通合计是为避免过早结束问题。其训练动机可理解为减少“继续运行总在扣分、结束回合省掉后续扣分”的倾向，失败项后加则保留失败扣分。它是奖励设计选择，PPO允许负奖励；夹到0也会让原本不同的负合计获得相同即时分数。详细的1 s时间缩放示例、缓存单位及三种合成数值见[08末尾本题Q/A](../../../../../../docs/migration/08_rewards迁移.md#补充问答)。当前源码已保存，只读检查不代表真实环境运行。

## 历史解释原文

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
