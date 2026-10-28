# 17 六项cost怎样随step输出

**目标与结果：** 补齐加速度项，并让六项cost每策略步只计算一次、只乘一次dt、按固定列序输出。

整理：2026-10-26。六函数、配置、管理器及step/reset接线已写，四处笔误已修。用户4环境/200步真实cost输出正常，见18；旧候选检查保留为历史，不能代替当前终止瞬间或训练验证。 当前证据与整体状态以[项目记忆](../../PROJECT_MEMORY.md)为准。

## 1. Gym计算什么

旧`_cost_acc_smoothness()`按策略周期计算关节加速度，只取超过阈值的部分，再对八个关节求平均：

$$
a_{t,j}=\frac{\dot q_{t,j}-\dot q_{t-1,j}}{\Delta t},\qquad
\bar a_j=\frac{\dot q_{j,lim}}{2\Delta t},
$$

$$
f_{acc}(t)=\frac1J\sum_{j=1}^{J}\max(\lvert a_{t,j}\rvert-\bar a_j,0),\qquad
c_{acc}(t)=\Delta t\,f_{acc}(t).
$$

$J=8$，$\Delta t=0.01\ \mathrm{s}$；$\dot q_{j,lim}$ 是各关节速度上限。原始项先沿关节求平均，管理器再乘策略时间步。

它与已有加速度奖励不同：奖励用加速度平方和，这里用超出阈值部分的均值。例如dt=0.01 s、某腿关节速度上限25 rad/s，该关节由0变20 rad/s，加速度2000 rad/s²，阈值1250，超额750。其他七关节无超额时，原始cost为750/8=93.75，收集后该列为0.9375。

Gym的compute_cost()只对每列乘一次dt，不把k乘进环境cost。class_to_dict()遍历dir()，实际旧列序是acc_smoothness、dof_vel_limits、feet_contact_forces、pos_limit、stumble、torque_limit。这里显式按这个顺序声明配置，k依次为0.1/0.3/0.1/0.3/0.1/0.3，d全零。

Gym先算cost，再reset。reset把关节速度清零，post_physics_step尾部再将此时的dof_vel保存为last_dof_vel。因此新回合第一步的差分基准是重置后的速度。

来源：[原始加速度cost](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1374)、[收集](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:726)、[step顺序](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:413)、[关节reset](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:827)。

## 2. 在mdp/costs.py追加加速度项

现有`from __future__ import annotations`保留。顶部增加`from collections.abc import Sequence`和运行时导入`from isaaclab.managers import ManagerTermBase`。TYPE_CHECKING中的managers导入加入`ManagerTermBaseCfg`。然后追加：

```python
class JointAccelerationCost(ManagerTermBase):
    """Measure joint acceleration above the legacy TITA threshold."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # 上一次策略步结束时的关节速度，(N, J)，单位rad/s。
        self._previous_joint_vel = self._asset.data.joint_vel.torch.clone()

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        joint_vel = self._asset.data.joint_vel.torch
        joint_ids = asset_cfg.joint_ids
        # 每0.01 s差分一次，得到(N, J_selected)的rad/s²。
        acceleration = (joint_vel[:, joint_ids] - self._previous_joint_vel[:, joint_ids]) / env.step_dt
        acceleration_limit = self._asset.data.joint_vel_limits.torch[:, joint_ids] / (2.0 * env.step_dt)
        excess = (acceleration.abs() - acceleration_limit).clamp_min(0.0)
        self._previous_joint_vel.copy_(joint_vel)
        # 每台机器人一个值，(N,)；此处尚未乘dt。
        return excess.mean(dim=1)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Use reset joint velocities as the next acceleration reference."""
        selection = slice(None) if env_ids is None else env_ids
        # 必须在机器人reset之后调用；只更新被重置机器人的行。
        self._previous_joint_vel[selection, :] = self._asset.data.joint_vel.torch[selection, :]
```

这个对象独立保存速度，不读取奖励类的_previous_joint_vel。奖励已经先执行并更新自己的历史，用它计算cost会读到错误的差分基准。

## 3. 建立任务自己的cost收集器

新增`Tita_Lab/src/Tita_Lab/tasks/velocity/cost_manager.py`。当前本地Lab没有原生CostManager/CostTermCfg；以下两个名字是任务扩展。继承原生ManagerBase是为了复用SceneEntityCfg解析、参数检查和类实例化，不修改Lab。

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

`compute()`由环境显式调用；Lab不会自动识别这个新管理器。它把每个函数的(N,)写入一列，得到(N,6)。`_last_step`只负责每策略步算一次：某步有机器人结束，reset之前先算；step返回前再调用时，直接返回之前保存的矩阵。

`reset()`由环境显式分发。它清除选中环境的回合累计值，调用有状态项的reset，把加速度参考改为重置后的关节速度。当前cost_buf不清除，因为其中是刚结束那一步的结果。

参考原生[ManagerBase](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/manager_base.py)、[通用参数解析](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/manager_base.py)及[RewardManager](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/reward_manager.py)。原生RewardManager求和得到一个奖励值；这里保留六列。[官方Manager文档](https://isaac-sim.github.io/IsaacLab/develop/source/api/lab/isaaclab.managers.html)说明了基类与自定义项关系；该网页为develop，具体API和本轮代码以本地源码及CPU检查为准。

## 4. .pyi导出后写cfg

在mdp/__init__.pyi的__all__与.costs导入块都追加JointAccelerationCost，保留已有五个函数。

env_cfg.py增加`from ...cost_manager import CostTermCfg as CostTerm`。在VelocityEnvCfg之前定义：

```python
@configclass
class CostsCfg:
    """Six cost channels used by the legacy TITA constraint algorithm."""

    # 声明顺序就是cost输出列序，显式对应旧Gym实际顺序。
    acc_smoothness = CostTerm(
        func=mdp.JointAccelerationCost,
        k=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TITA_JOINT_NAMES, preserve_order=True)},
    )
    dof_vel_limits = CostTerm(
        func=mdp.joint_velocity_limit_cost,
        k=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TITA_JOINT_NAMES, preserve_order=True)},
    )
    feet_contact_forces = CostTerm(
        func=mdp.wheel_contact_force_cost,
        k=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_leg_4", "right_leg_4"])},
    )
    pos_limit = CostTerm(
        func=mdp.joint_position_limit_cost,
        k=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TITA_JOINT_NAMES, preserve_order=True)},
    )
    stumble = CostTerm(
        func=mdp.wheel_stumble_cost,
        k=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_leg_4", "right_leg_4"])},
    )
    torque_limit = CostTerm(
        func=mdp.joint_effort_limit_cost,
        k=0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=TITA_JOINT_NAMES, preserve_order=True)},
    )
```

所有d沿用CostTermCfg的0默认值；三个可选阈值也沿用已有函数默认值。然后在VelocityEnvCfg的rewards、terminations旁新增`costs: CostsCfg = CostsCfg()`。两个轮体的判断最终都any，轮体先后不影响本次结果。

## 5. 环境显式接入

在tita_velocity_env.py中，增加future注解、TYPE_CHECKING及TitaCostManager导入；环境cfg类型改为VelocityEnvCfg。以下是合并现有功能后的参考文件主体：

```python
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

    cfg: VelocityEnvCfg

    def __init__(self, cfg: VelocityEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        """Initialize task histories and cost collection."""
        # 父类建立动作/观测管理器时会读取这两份数据，故先创建。
        self.proprio_history = ProprioHistory(
            num_envs=cfg.scene.num_envs, obs_dim=33, history_length=10, device=cfg.sim.device
        )
        self.action_lag_indices = torch.zeros(cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device)
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        # 此时机器人和物理数据已建立，可以解析cost中的关节/轮体。
        self.cost_manager = TitaCostManager(self.cfg.costs, self)

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Advance the environment and return task rewards and costs."""
        self.proprio_history.read_then_push(self.obs_buf["policy"], self.episode_length_buf)
        obs, reward, terminated, time_outs, extras = super().step(action)

        termination_index = self.reward_manager.active_terms.index("termination")
        termination_reward = self.reward_manager._step_reward[:, termination_index] * self.step_dt
        ordinary_reward = reward - termination_reward
        reward.copy_(ordinary_reward.clamp_min(0.0) + termination_reward)

        # 无reset时在这里计算；有reset时返回reset前已经保存的本步值。
        # clone让本次返回的矩阵不被下一步覆盖，形状(N, 6)。
        extras["costs"] = self.cost_manager.compute().clone()
        return obs, reward, terminated, time_outs, extras

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Reset selected robots and their task histories."""
        self.observation_manager.compute_group("privileged_state")
        # 在父类修改姿态、关节速度和传感器数据前，保存刚结束的一步。
        self.cost_manager.compute()
        super()._reset_idx(env_ids)
        self.proprio_history.reset(env_ids)
        # 父类创建本轮log后，再加入cost统计并更新加速度差分基准。
        self.extras["log"].update(self.cost_manager.reset(env_ids))
```

父类构造没有调用任务的_reset_idx，cost_manager可在super().__init__后建立；公共env.reset发生在构造完成之后。初次reset时common_step_counter未增加，compute只返回初始零矩阵，reset再保存重置后的速度。

当前六项仅依赖关节位置、关节速度、驱动力矩和轮接触力。interval推扰只写机身速度，command更新也不改变这四类输入，因此没有结束的环境可以在super().step后采集。未来若事件会改关节或接触数据，须重新核对采集位置。现有旧Gym与Lab推扰时序差异仍留在00缺口清单，本次不宣称全流程逐步等价。

训练端未来读取`extras["costs"]`，算法参数读取`env.unwrapped.cost_manager.k_values/d_values`。本次不改变Lab的五元step返回结构，不把cost塞进观测或奖励。NP3O消费仍待接入。

## 6. 完成标准

代码写好后，先核对CostManager列名与六列数值，特别是结束那一步的cost没有被reset清掉、加速度参考在reset后重新设置。CPU检查使用原生ManagerBase及原生RL step/reset流程，状态和物理推进为替身；真实环境仍需用户运行。

## 补充问答

**Q：** 自己写manager和cfg我真的有点心理恐惧了，讲一下这超长代码吧？这里可能RL原理占小头，编程层面的思维逻辑和流程占大头。

**A：** 本轮先停新功能，把对象和调用顺序讲清。CostTermCfg保存一项的func/params/k/d，CostsCfg汇总六项；TitaCostManager保存这六项并负责调用、统计，TitaVelocityEnv决定何时调用。父类ManagerBase构造中有self._prepare_terms()，此时self仍是TitaCostManager实例，所以执行子类方法，填好六项列表。继承提供解析工具，step/reset仍靠环境中的显式调用接入。详细顺序与变量例子见[cost_manager.py说明](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/cost_manager.py.md)。

**Q：** 比如@property（用户原句节选）

**A：** @property使读取manager.active_terms时执行active_terms方法并返回结果，无需写括号。当前没有setter，对该公开属性不能直接赋值；它每次返回_term_names.copy()，让外部修改拿到的列表时不影响管理器内部名单。

**Q：** self.k_values = torch.tensor( [term.k for term in self._term_cfgs], device=env.device ).view(1, -1) self.d_values = torch.tensor( [term.d for term in self._term_cfgs], device=env.device ).view(1, 1, -1)这里的语法。（保留用户代码，合并换行）

**A：** [term.k for term in self._term_cfgs]是列表推导式，等价于建立空列表，再for遍历六个配置并append(term.k)。得到六个Python浮点数后，torch.tensor将其变成设备上的一维张量(6,)，view(1,-1)推断剩余维度为6，得到(1,6)；d的view(1,1,-1)得到(1,1,6)。六对应cost种类，不是环境数。k/d形状沿用旧算法：批次侧使用k，时间×环境×cost的存储侧使用d；这里没有复制成每环境一份。参考[Python列表推导式](https://docs.python.org/3.12/tutorial/datastructures.html#list-comprehensions)、[Tensor.view](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html)。

**Q：** 还有个关键变量你没给注释：_episode_sums

**A：** 它是(N,6)的本回合累计值；cost_buf是相同形状的本步结果。每步_episode_sums.add_(cost_buf)，重置时仅清选中环境的累计行，保留cost_buf供训练端取得最后一步。假设仅看位置cost，第1步cost_buf[:,3]=[0.01,0]，第2步=[0.01,0.01]，累计变为[0.02,0.01]；只reset环境0后累计为[0,0.01]，本步结果仍保留。它不计算折扣、优势或梯度。
