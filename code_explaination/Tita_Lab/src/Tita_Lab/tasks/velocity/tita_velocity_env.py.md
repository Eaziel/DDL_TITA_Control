# tita_velocity_env.py：安排历史、采样和奖励合成的时机

**阅读目标：** 理解自定义环境只补哪些调用时机：历史更新、奖励最终合成、reset前保存cost与六维状态，以及reset后清相应缓存。

源码：[tita_velocity_env.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/tita_velocity_env.py)。更新：2026-10-23。以下为本次建议的清理后类（不是已写入快照）；移除临时打印/断言，保留全部历史、奖励、cost与reset接线。当前源码已删RESET/CONTACT/COST打印，首步VIEW及cost断言仍在；导入省略，历史原文保留在末尾折叠区。

```python
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
        # 腿轮共享的动作延迟编号，shape=(N,)，由startup事件采样。
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

        return obs, reward, terminated, time_outs, extras

    def _reset_idx(
        self,
        env_ids: Sequence[int] | torch.Tensor,
    ) -> None:
        """Reset selected robots and their task histories."""

        # 重置物理状态前，保留刚结束这一步的六维观测样本。
        self.observation_manager.compute_group("privileged_state")

        # 父类修改姿态、关节速度和传感器数据前，保存刚结束的一步。
        self.cost_manager.compute()

        super()._reset_idx(env_ids)
        self.proprio_history.reset(env_ids)

        # 父类创建本轮log后，再加入cost统计并更新加速度差分基准。
        self.extras["log"].update(self.cost_manager.reset(env_ids))
```

## 1. 初始化先准备两份数据

proprio_history持有(N,10,33)的policy历史，action_lag_indices持有(N,)的腿轮共享选择编号。它们在super().__init__之前建立，因为父类建立观测/动作管理器时就会读取它们。

随后父类建立Scene、动作、观测、事件等通用对象。本类保留这套框架，只增加任务需要的调用时机。

## 2. step开头先处理旧policy

此时obs_buf["policy"]还是上次返回的观测。read_then_push保存它，父类再推进四次物理步、计算奖励、处理结束环境并生成新观测。返回的history因此包含过去十帧，当前policy单独返回。

## 3. 父类step返回后形成最终奖励

父类已得到十项的加权时间积分。_step_reward中的termination列保存未乘step_dt的本步贡献，乘0.01恢复失败分数；从总分减去它，对其余合计夹零，再加回失败分数。reward.copy_更新原奖励缓冲。

例如普通合计-0.3、失败分数-2，结果为0-2=-2。这个位置依赖本地_step_reward内部字段；以后启用父类record_post_step记录时还需核对记录发生的先后。

## 4. reset先保留本步六维样本，再重置其他对象

compute_group先要求六维观测对象保存速度/接触，父类再重置Scene、事件和各管理器，最后显式清ProprioHistory对应行。六维对象按common_step_counter识别同一步，后续观测复用已保存样本。

奖励类reset由RewardManager分发，动作类reset由ActionManager分发，历史缓存reset由这里明确调用。方法同名不会使任意对象自动加入某个管理器。

当前代码的原生step/reset CPU流程检查通过；物理状态/setter为替身，真实训练未运行。证据见[当前检查](../../../../../../docs/audits/2026-10-21_current_observations/README.md)。

## 5. 独立cost的调用需要保留

初始化在父类完成后构造TitaCostManager；step返回前把compute结果clone到extras["costs"]。发生reset时，父类改变物理状态前先compute，父类reset后调用cost_manager.reset收集回合统计并重置需要差分的内部状态。这些调用属于迁移功能，不随诊断打印删除。

## 6. 本次临时诊断已经完成用途

2026-10-23先由[接触日志](../../../../../../docs/audits/2026-10-23_reset_contacts/result.json)定位自碰撞路径，再发现USD比旧URDF多出左右leg_1/leg_2四处碰撞。用户现已在USD关闭四处并报告重新开启自碰撞后现象正常；静态结果见[修正后状态](../../../../../../docs/audits/2026-10-23_collision_model/after_user_usd_edit.json)。新运行未附完整日志，不能宣称训练成功。

当前VIEW/RESET/CONTACT/COST打印与Scene.base_contact_debug均已删除，正式钩子保留；每步cost断言仍在。上方完整类是清理参考，省略了这些断言，不冒充当前逐字源码。可选box方案当前跳过，取舍见[19](../../../../../../docs/migration/19_碰撞模型对齐与诊断清理.md)。

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

Lab公开reset()和每步自动结束都会进入self._reset_idx，Python方法分派选中子类的重写；子类用super明确保留父类流程。本项目没有重写环境的公开reset方法。源码：[公开reset入口](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/manager_based_env.py)、[step入口](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)、[父类依次重置manager](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)。

ProprioHistory用于保存过去十帧、每帧33维的自身观测。环境对象持有一个(N,10,33)缓存；step更新，mdp.proprio_history读取并展平为(N,330)，ObservationManager把它放到obs["history"]中。history提供旧TITA后续历史编码网络需要的时间窗口；当前生成的普通PPO仍只读policy，模型消费history尚未迁移。

两份跨步状态都要按同一env_ids处理，但内容不同：奖励缓存同步为重置后的关节速度，以免跨回合计算加速度；历史缓存清零，以免新回合读到旧回合轨迹。单个环境结束只清它对应的行，其他并行环境继续保留各自数据。

**Q：**

> 现在是新的step的问题。我没搞懂什么叫普通项？为啥要 * step_dt？active_terms指的就是在rewardcfg里写的项？
> 还有为啥要这个计算公式啊，你说的旧gym的那个，他为啥这样搞？

**A：** 普通项在当前合成中指除termination以外的9项，含负权重惩罚。active_terms是管理器收集的奖励项名字列表，当前index("termination")为9，对应(N,10)缓存的第9列。父类reward已经累计raw*weight*step_dt；缓存保存的是raw*weight，所以从缓存取失败贡献时才乘step_dt恢复本步分数。接着用reward-termination_reward拆出普通合计，只夹这部分再加回失败贡献。

旧配置注释称夹普通合计是为避免过早结束问题。其训练动机可理解为减少“继续运行总在扣分、结束回合省掉后续扣分”的倾向，失败项后加则保留失败扣分。它是奖励设计选择，PPO允许负奖励；夹到0也会让原本不同的负合计获得相同即时分数。详细的1 s时间缩放示例、缓存单位及三种合成数值见[08末尾本题Q/A](../../../../../../docs/migration/08_rewards迁移.md#补充问答)。当前源码已保存，只读检查不代表真实环境运行。

**Q：** 我没懂两处self.observation_manager.compute_group("privileged_state")在干啥。

**A：** 这里的一处在_reset_idx的父调用之前。compute_group会调用六维观测对象，把本步速度/接触存入对象的成员，再由父方法重置机器人。返回张量没有接收，但成员变量已更新，最后统一观测读取仍使用同一步样本。第二处在events.py的推扰写速度之前；两处不是每步都各执行一次，它们只在对应事件发生时提前取数。完整变量例子见mdp/observations.py.md的同题问答。

**Q：** compute函数屡次出现，这玩意到底是干啥的？

对应运行语句例如：

```python
self.curriculum_manager.compute(env_ids=env_ids)
extras["costs"] = self.cost_manager.compute().clone()
```

**A：** compute意思是“计算”，是开发者起的普通方法名，不是Python提供的特殊方法。先看点号左边的对象：RewardManager.compute汇总奖励并累计回合统计；ObservationManager.compute组装观测；CurriculumManager.compute执行课程项并保存课程状态；TitaCostManager.compute算出或复用本步(N,6)cost。

本地CurriculumManager.compute的关键代码是：

```python
for name, term_cfg in zip(self._term_names, self._term_cfgs):
    state = term_cfg.func(self._env, env_ids, **term_cfg.params)
    self._curriculum_state[name] = state
```

当前只有terrain_levels项，所以会调用登记的terrain_levels_vel。该函数读取机器人位置和速度命令，调用TerrainImporter修改等级与env_origins，并返回平均等级；管理器将这个返回值保存供日志读取。CurriculumManager.compute自身没有return语句，Python返回None；调用它的目的包含更新内部状态，不必有接收变量。

什么时候执行由调用点决定，函数名不会自动决定时机：原生环境在reset前明确调用课程compute；本项目显式在step末尾与reset前调用cost compute。cost用_last_step避免同一个策略步重复计算和累加，因此第二次调用可直接返回已保存的cost_buf。[原生调用点](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)、[课程compute实现](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/curriculum_manager.py)。

## 历史原文（折叠）

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
