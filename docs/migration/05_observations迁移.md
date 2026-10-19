# 05 Observations：当前33维、噪声和十帧历史

更新：2026-10-18。当前33维及噪声、缓存read、历史函数/导出、history组、提前构造、step推进和reset均已写入并做静态核对。第5节整理这套完整接线，第6节解释顺序。未运行真实RL环境。

本章迁移的是环境提供给学习程序的数据。先处理“现在看到什么”，再处理“之前看到什么”。`N` 表示并行环境数。

## 1. 当前帧：Gym 原来怎么做

旧 `compute_observations()` 先生成一个33维自身观测。下面省略注释行：

```python
self.dof_pos[:, [3, 7]] = 0
obs_buf = torch.cat((
    self.base_ang_vel * self.obs_scales.ang_vel,
    self.projected_gravity,
    self.commands[:, :3] * self.commands_scale,
    self.reindex((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos),
    self.reindex(self.dof_vel * self.obs_scales.dof_vel),
    self.action_history_buf[:, -1],
), dim=-1)
```

六部分依次为：

| 内容 | 宽度 | 作用 |
| --- | --- | --- |
| 机身角速度 | 3 | 身体正在绕各轴怎样转动 |
| 机身系下的单位重力方向 | 3 | 身体怎样倾斜 |
| vx、vy、yaw_rate目标 | 3 | 这一段希望怎么运动 |
| 相对默认角的关节位置，轮角屏蔽 | 8 | 腿偏离名义姿态多少 |
| 实际关节角速度，包含轮速 | 8 | 各关节怎样运动 |
| 上一次输入的策略动作 | 8 | 刚才给了什么控制信号 |

因此`3+3+3+8+8+8=33`，局部obs_buf形状为(N,33)。六个子张量按上述顺序沿特征轴拼接。

旧代码随后给这33列加噪，再拼成完整输出：

```python
if self.cfg.noise.add_noise:
    obs_buf += (2 * torch.rand_like(obs_buf) - 1) * noise_vec.to(self.device)

self.obs_buf = torch.cat(
    [obs_buf, heights, priv_latent, self.obs_history_buf.view(self.num_envs, -1)],
    dim=-1,
)
```

局部 `obs_buf` 是当前33维；`self.obs_buf` 是包含高度、特权量、历史的完整观测。旧 runner 取得完整输出，再由网络按需要切片。当前迁移先对应其中的自身观测与历史，不能把整份旧观测的宽度当成33。

## 2. 当前帧：Lab 中怎样写

### 2.1 env_cfg.py 声明六个观测项

以下为当前 [env_cfg.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py) 已有代码的整理写法，省略行尾解释。文件已有 `ObsTerm`、`ObsGroup`、`SceneEntityCfg`、`configclass` 和 `mdp` 的导入；噪声类为：

```python
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
```

关节顺序列表放在配置类外：

```python
TITA_JOINT_NAMES = [
    "joint_left_leg_1",
    "joint_left_leg_2",
    "joint_left_leg_3",
    "joint_right_leg_1",
    "joint_right_leg_2",
    "joint_right_leg_3",
    "joint_left_leg_4",
    "joint_right_leg_4",
]
```

当前帧部分如下；同一个ObservationsCfg还登记了history组，见第5.3节：

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.8, n_max=0.8, operation="add"),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05, operation="add"),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_zero_wheels,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=TITA_JOINT_NAMES, preserve_order=True,
                ),
                "wheel_cfg": SceneEntityCfg(
                    "robot", joint_names=["joint_left_leg_4", "joint_right_leg_4"],
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01, operation="add"),
            scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=TITA_JOINT_NAMES, preserve_order=True,
                ),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5, operation="add"),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

外层环境配置已经通过 `observations: ObservationsCfg = ObservationsCfg()` 登记它。

### 2.2 MDP 函数只补轮角屏蔽

角速度、重力、命令、关节速度和上次动作都有 Lab 函数。相对位置还需屏蔽两轮，因此当前 [mdp/observations.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/observations.py) 增加了：

```python
def joint_pos_rel_zero_wheels(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = robot.data.joint_pos.torch - robot.data.default_joint_pos.torch
    joint_pos_rel[:, wheel_cfg.joint_ids] = 0.0
    return joint_pos_rel[:, asset_cfg.joint_ids]
```

先对全部关节做减法，得到新张量；按资产原生索引清两轮；最后按指定八关节顺序取列。`wheel_cfg` 负责“清哪些列”，`asset_cfg` 负责“返回哪些列及顺序”。两者都指向同一个机器人。

当前 `mdp/__init__.pyi` 已导出这个函数。该工程的 `lazy_export()` 在运行时也会读取 `.pyi`，因此它不只是给编辑器看的类型文件。

### 2.3 由 ObservationManager 执行计算和拼接

`func=mdp.base_ang_vel` 保存函数对象，不是在定义配置时立刻取角速度。环境初始化时创建 ObservationManager；每次计算观测时，它依次处理每个条目：

```text
调用 func 取数据 → clone → modifiers → noise → clip → scale → delay/history
  → 按组拼接
```

当前没有配置 modifiers、clip、内置 delay/history，所以主要执行取数、加噪、缩放和拼接。`clone` 避免后续处理直接改写资产状态或命令。

`concatenate_terms=True` 配合默认 `concatenate_dim=-1`，沿最后一维拼接；环境行不会混在一起。条目按声明顺序排列，`policy` 是返回字典的组名：

```text
policy 列：0:3角速度 | 3:6重力 | 6:9命令 | 9:17位置 | 17:25速度 | 25:33动作
```

这个组的形状由源码推导为 `obs["policy"]:(N,33)`；history组另在第5.3节登记。尚未做真实环境读数检查。

## 3. 当前帧：为什么这样迁移

### 3.1 保留原特征含义，明确参考值和顺序

`base_ang_vel` 读取机身系实际角速度，单位 rad/s。`projected_gravity` 读取世界单位重力方向旋转到机身系后的结果；正立时约为 `[0,0,-1]`，倾斜改变x/y分量，仅改变航向不会改变它。

`generated_commands` 取得 `base_velocity` 的三维目标，第三列已经是航向外环算出的目标转速，不是目标航向角。观测 scale 只改变给网络的数值，不改变命令管理器保存的物理目标。

`rel` 是 relative，即相对默认参考值：

```text
joint_pos_rel = q - q_default
joint_vel_rel = dq - dq_default
joint_vel     = dq
```

默认膝角 -1.5 rad，当前 -1.4 rad，位置偏差就是 +0.1 rad。本任务使用joint_vel读取实际角速度dq，直接对应旧dof_vel；当前默认速度参考为0。

轮子对应轮轴转动关节。按旧特征定义，轮角两列在加噪前置零，轮速保留。Lab在q-q_default产生的新张量上做这一步，物理状态保持原值。

新观测顺序与新 Actions 一致：左腿1/2/3、右腿1/2/3、左轮、右轮。按名称解析，不照搬旧 reindex 下标。`last_action` 取得环境刚接收的完整八维动作，处于动作项缩放之前；在 `o_t → a_t → 物理 → o_(t+1)` 中，下一观测记录的是刚执行的 `a_t`，不需要再取 `prev_action`。

### 3.2 噪声参数必须补偿处理顺序

旧 Gym 先缩放，再加噪；Lab 当前管理器先加噪，再缩放。对一个条目，记原始值为 `x`、scale为 `s`、旧最终噪声半宽为 `A`：

```text
Gym：y = s*x + A*r
Lab：y = s*(x + b*r)
```

`r` 为 `[-1,1)` 均匀随机数。为了让相同r产生相同结果，需要 `s*b=A`，所以 **Lab 半宽 b=A/s**。这里各项scale为正，且没有中间裁剪，换算成立。

旧 `noise_vec` 的角速度半宽直接用0.2，没有乘角速度scale 0.25；关节速度则用 `1.5*0.05`。因此要从实际计算取幅度，不能把配置里所有 noise_scales 原样复制：

| 条目 | 旧最终半宽 A | scale s | Lab 的 noise 半宽 b |
| --- | --- | --- | --- |
| 角速度 | 0.2 | 0.25 | 0.8 |
| 重力方向 | 0.05 | 1 | 0.05 |
| 速度命令 | 0 | `(2,2,0.25)` | 不设置noise |
| 相对关节位置 | 0.01 | 1 | 0.01 |
| 关节速度 | 0.075 | 0.05 | 1.5 |
| 上次动作 | 0 | 1 | 不设置noise |

例如 `x=1`、`s=0.25`、`r=0.5`：Gym 为 `0.25+0.2*0.5=0.35`；Lab 为 `(1+0.8*0.5)*0.25=0.35`。如果把Lab半宽也填0.2，结果就变为0.275。

分别采样时两者输出分布对应，不保证同seed逐元素随机序列相同。旧 `noise_level=1` 是全局幅度乘数；Lab 不会自动读取它，后续若要调节，应显式让四项边界同乘该系数。

`Unoise` 是 `UniformNoiseCfg` 的别名，`n_min/n_max` 为均匀噪声边界，`operation="add"` 指加法。它不是高斯标准差。轮角在函数中先归零，之后仍加±0.01位置噪声，因此轮角两列是纯噪声；若用最终scale=0代替，会把这份噪声一起消掉。

## 4. 历史：Gym 原来怎么保存

### 4.1 当前帧与历史是两份数据

旧任务除了当前 `o_t:(N,33)`，还保存 `H_t:(N,10,33)`。每0.01 s记录一帧，提供近期动作与状态响应的变化线索；这可以帮助网络估计部分未直接观测的状态，但不保证重建所有隐藏状态。

正常情况下，环境返回：

```text
当前：o_t
历史：[o_(t-10), …, o_(t-1)]
```

历史由旧到新，每帧33项连续排列；展开后为330列，不包含本次当前帧。

### 4.2 先输出旧历史，再保存当前帧

旧代码先拼接已有历史，随后更新缓存：

```python
self.obs_buf = torch.cat(
    [obs_buf, heights, priv_latent, self.obs_history_buf.view(self.num_envs, -1)],
    dim=-1,
)
self.obs_history_buf = torch.where(
    (self.episode_length_buf <= 1)[:, None, None],
    torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
    torch.cat([self.obs_history_buf[:, 1:], obs_buf.unsqueeze(1)], dim=1),
)
```

这里 `obs_buf` 已经加噪和缩放。正常更新丢掉最旧帧、追加当前帧；不会重新给存过的历史加噪。

重置时先把对应环境的缓存清零。首次计算先输出这份零历史，之后在 `episode_length<=1` 分支用当前帧回填十份。因此“首次返回零历史”和“随后回填当前帧”是前后两件事，不能合并为一条。

旧默认 Actor 从完整观测分别取当前33列与末尾330列，再将历史中的后9帧和当前帧组成自己的编码窗口。其他价值/辅助计算也会读取原历史，所以环境侧仍保留旧过去十帧的定义，网络侧组窗留在模型迁移时处理。

## 5. 历史：Lab 中怎样接入

**本节说明当前已写入的完整接线。** 缓存、只读取数、组登记、提前构造、step推进与reset均已出现于源码；下面按数据从哪里来、由谁输出、何时更新讲清楚。源码存在不等于已运行。

目标接口为：

```python
obs["policy"]   # (N,33)，保留当前六项
obs["history"]  # (N,330)，过去十帧，按整帧展平
```

### 5.1 observation_history.py：保存和读取历史

文件：[mdp/observation_history.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/observation_history.py)。下面按构造、读取、更新、重置顺序列出当前类的计算，省略原行尾解释；`read()`已写入：

```python
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

    def read(self) -> torch.Tensor:
        return self._history.clone()

    def read_then_push(
        self,
        current_obs: torch.Tensor,
        episode_length: torch.Tensor,
    ) -> torch.Tensor:
        if current_obs.shape != (self.num_envs, self.obs_dim):
            raise ValueError("current_obs 的形状不匹配.")
        if episode_length.shape != (self.num_envs,):
            raise ValueError("episode_length 的形状不匹配.")

        past = self._history.clone()
        shifted = torch.cat(
            (self._history[:, 1:, :], current_obs.unsqueeze(1)), dim=1,
        )
        repeated_current = current_obs.unsqueeze(1).expand(
            -1, self.history_length, -1,
        )
        self._history = torch.where(
            (episode_length <= 1)[:, None, None],
            repeated_current,
            shifted,
        )
        return past

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> None:
        if env_ids is None:
            self._history.zero_()
        else:
            self._history[env_ids, :, :] = 0.0
```

这个普通 Python 类持有跨步状态，PyTorch 函数完成数组运算：

- `_history[:,1:,:]` 丢最旧帧，得到 `(N,9,33)`；`unsqueeze(1)` 将当前帧变为 `(N,1,33)`，拼接后恢复 `(N,10,33)`。
- `episode_length<=1` 为每个环境选择开局回填或正常滑动，两个新增轴用于广播。
- `past` 是更新前的独立快照；内部 `_history` 已更新后，返回的 `past` 仍是旧历史。
- `read()` 只返回副本，不推进。`reset(env_ids)` 只清指定环境的全部时间帧与特征。

### 5.2 observations.py：把缓存作为观测项返回

文件：[mdp/observations.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/observations.py)。保留已有函数和 `from __future__ import annotations`，当前类型导入块为：

```python
if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from ..tita_velocity_env import TitaVelocityEnv
```

已写入的模块级函数为：

```python
def proprio_history(env: TitaVelocityEnv) -> torch.Tensor:
    """Return past proprioceptive frames in whole-frame order."""
    return env.proprio_history.read().reshape(env.num_envs, -1)
```

它取出 `(N,10,33)` 并按整帧展开为 `(N,330)`。`TYPE_CHECKING` 中的环境导入只供类型检查，避免运行时循环导入；已有 future annotations 支持这份注解。

文件：[mdp/__init__.pyi](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/__init__.pyi)。当前导出部分为：

```python
__all__ = [
    "joint_pos_target_l2",
    "joint_pos_rel_zero_wheels",
    "proprio_history",
    "illegal_contact_current",
]

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import joint_pos_rel_zero_wheels, proprio_history
from .rewards import joint_pos_target_l2
from .terminations import illegal_contact_current
```

这样 cfg 才能通过 `mdp.proprio_history` 访问新函数。

### 5.3 env_cfg.py：登记 history 组

当前 `ObservationsCfg` 内已有以下HistoryCfg，它与PolicyCfg同级；history字段与policy字段同级。第2节当前帧内容保留：

```python
    @configclass
    class HistoryCfg(ObsGroup):
        """Past proprioceptive observations."""

        proprio = ObsTerm(func=mdp.proprio_history)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    history: HistoryCfg = HistoryCfg()
```

缓存里是以前已经处理好的帧，因此不重新加噪。也不设置内置 `history_length`：函数已经返回完整十帧，不能再对它堆十帧。

这一步让 ObservationManager 知道还有一个输出组，并自动推导其形状与观测空间，不需要在 step 返回字典中临时塞一个未登记的键。

### 5.4 tita_velocity_env.py：控制何时写入与重置

文件：[tita_velocity_env.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/tita_velocity_env.py)。当前缓存已在super之前创建，step推进和reset均已写入。下面为当前类的整理写法，省略原行尾解释：

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
            self.obs_buf["policy"], self.episode_length_buf,
        )
        return super().step(action)

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        super()._reset_idx(env_ids)
        self.proprio_history.reset(env_ids)
```

当前注册 `entry_point` 已指向 `TITA_Lab.tasks.velocity.tita_velocity_env:TitaVelocityEnv`，无需再次修改；配置入口仍提供原来的 `VelocityEnvCfg`。

## 6. 历史：为什么需要这些文件和顺序

### 6.1 env_cfg 声明，环境和管理器执行

旧 Gym 在一个大环境文件里完成存储、推进、取数和拼接；Lab 中按职责分开：

| 文件或对象 | 负责什么 |
| --- | --- |
| `env_cfg.py` 的 ObservationsCfg | 声明有哪些组、每项调用谁、使用什么参数 |
| `mdp/observations.py` | 计算一次观测项，这里是读取已有缓存 |
| `ProprioHistory` 对象 | 保存跨步数据，提供读、写、清零方法 |
| `TitaVelocityEnv` 对象 | 持有缓存，决定何时推进与重置 |
| 官方 ObservationManager | 按 cfg 调用函数、处理条目、拼接并返回观测组 |

因此，写 MDP 和这个环境子类仍在完成观测功能。cfg 中填写 `func` 不会自动提供那个函数尚未实现的跨步存储与生命周期。此前只写 cfg 时，实际运行者也一直存在，就是官方环境类与管理器。

### 6.2 为什么不直接设置 history_length=10

Lab 有内置历史，但直接给现有六项开启默认历史，会改变两件事：

1. 内置更新先 append 当前条目再输出，所以历史包含当前帧；旧环境输出的是更新前历史。
2. 默认每个term独立维护并展平，再拼term，布局是“角速度十帧→重力十帧→……”，旧布局是“完整33维一帧→下一帧”。

当前自定义缓存按整帧保存、先读取旧窗口，再推进一次，保留旧任务对历史布局和时序的要求。

### 6.3 为什么在 step 开头写上一帧

原 Gym 在生成当前观测后保存它。上述方案把这次写入推迟到下一次 step 开头：

```text
上一次已返回：policy=o20，history=o10…o19
下一次step开头：把已有 o20 写入 → 缓存变为 o11…o20
父类推进物理：生成 o21
本次返回：policy=o21，history=o11…o20
```

外部读到的窗口不变，但内部写入时刻有意调整。输入直接复用 `self.obs_buf["policy"]`，保留当时实际返回的那份噪声和缩放结果，不重新调用六个函数另算一份。

ObsTerm 只读，因此初始化推导形状、额外取数或读取终止观测不会多推进一次。`read_then_push()` 在这里仅使用更新副作用，返回的旧快照不再接收；对外读取统一经 `read()`。不能再在step末尾追加一次。

### 6.4 为什么缓存要在 super 初始化之前创建

父类构造会创建ObservationManager。管理器在准备观测组时调用配置登记的函数，以返回张量的.shape确定维度与空间；history组的函数此时已经要读取缓存。因此初始化顺序为：

```text
ProprioHistory创建缓存
  → super初始化环境与ObservationManager
  → 管理器调用history观测函数并确定(N,330)输出形状
```

创建缓存时使用传入cfg的环境数量与设备。重置时先走父类_reset_idx，再清对应环境缓存；随后的观测读到零历史，下一次step按开局规则回填。关于“谁在调用哪个函数”的详细追问见本章末尾Q/A。

## 7. 当前结果、未完成项与验证

| 部分 | 当前文件状态 |
| --- | --- |
| 六项当前观测、顺序、轮角屏蔽、四项noise | 已写入并静态核对；33维为源码推导 |
| 缓存类、只读函数/导出、history组、提前构造、重置与入口 | 已写入，重置方法为类方法 |
| 本章第5.4节step推进 | 已写入；方法体与本章代码静态对应，真实逐步更新待运行验证 |
| 高度与特权观测分组 | 尚未迁移 |
| 旧观测最终±100裁剪 | 尚未对齐；Lab条目clip位于scale之前，不能直接照抄范围 |
| 旧模型如何消费历史 | 尚未迁移；当前普通PPO仍只选policy组 |

当前 `rsl_rl_ppo_cfg.py` 的 `obs_groups` 为 `{"actor":["policy"],"critic":["policy"]}`。环境提供history，不代表这个网络已经使用history，更不代表旧编码器、Cost和NP3O已迁完。

此前局部检查包括：轮角函数的独立CPU张量检查、噪声换算与旧公式的受控随机数对照、缓存更新与旧Gym语句对照，以及三环境36步含异步重置的接线原型。接线原型位于 `/tmp/tita_history_integration_20261017/`，使用真实Lab观测拼接方法和CPU父环境替身，检查了返回窗口与只读行为。**这些都不是真实环境、物理仿真或训练验证。** 详细检查经过保留在 [项目历史](../PROJECT_HISTORY.md)，本章只保留结论。

## 8. 对照源码

| 要查什么 | Gym | Lab |
| --- | --- | --- |
| 当前帧与完整输出 | [配置](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:34)、[compute_observations](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:335) | [官方观测配置](/home/eaziel/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:159)、[MDP观测函数](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py:64) |
| noise与scale | [normalization / noise](/home/eaziel/Downloads/tita_rl-master/configs/legged_robot_config.py:189) | [compute_group处理](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:417)、[uniform_noise](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/utils/noise/noise_model.py:46)、[组开关准备](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:580) |
| 历史更新与清零 | [先拼接再更新](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:380)、[reset_idx](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:734) | [内置history](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:438)、[CircularBuffer](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py:100) |
| 配置到执行 | [旧step](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:295) | [创建managers](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:363)、[初始化推导维度](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:591)、[环境step](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:205) |
| 旧网络取当前帧和历史 | [act_teacher](/home/eaziel/Downloads/tita_rl-master/modules/actor_critic.py:764)、[Actor组窗](/home/eaziel/Downloads/tita_rl-master/modules/actor_critic.py:169) | [当前PPO配置](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/agents/rsl_rl_ppo_cfg.py) |

## 补充问答

问题保留用户原句；回答按对应代码整理。

<a id="33-enable_corruption-控制什么"></a>

**Q：**

> 回答一下enable_corruption是不是让noise这一项被考虑了？类似源码的if flag 加noise的计算？
> if so,那为啥api不叫enable noise？

**A：** 是。当前ObservationManager在准备观测项时处理：

```python
if not group_cfg.enable_corruption:
    term_cfg.noise = None
```

True保留已配置的noise，False移除它；命令和动作项本来没配noise，所以开启后也不会增加噪声。scale、clip、modifiers和history按各自配置处理。这个开关在初始化准备阶段生效。

corruption表示观测受到扰动。当前源码没有说明作者为什么选这个API名字，因此只能确认行为，不能据名字推导额外功能。

**Q：**

> 当前默认关节速度为零，两者目前数值相同。
> 这句话啥意思？？rel是啥含义？

**A：** rel是relative，表示相对默认参考值：joint_vel_rel=dq-dq_default；joint_vel=dq。当前dq_default=0，因此实际速度为2 rad/s时，两者都返回2。这里为零的是保存的参考值，实际运动仍由dq给出。位置同理，joint_pos_rel=q-q_default。

**Q：**

> 好了。轮子位置指的是轮子旋转角度？确实不需要这个，轮子无限转圈。是这样理解的吧！

**A：** 对，这里指轮轴转动关节的角度。当前任务沿用旧TITA的特征定义：屏蔽两轮转角、保留两轮转速。清零发生在观测的新张量中，随后仍会加位置噪声，所以最终轮角两列可能是纯噪声。是否需要轮子相位取决于任务；本次迁移保持旧任务选择。

**Q：**

> 我没搞懂。为啥会调用这个？

**A：** 你问的是父类初始化ObservationManager时为何执行proprio_history。调用入口就是自己登记的这一项：

```python
proprio = ObsTerm(func=mdp.proprio_history)
```

func保存函数对象。管理器准备各项时实际执行：

```python
obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)
```

这里self是ObservationManager，self._env才是TitaVelocityEnv。处理history/proprio项时，保存的func就是mdp.proprio_history，它接收当前环境、读取缓存并返回(N,330)，管理器据此确定维度。因此该调用发生在父类构造期间，缓存需要先创建。类型注解torch.Tensor只声明返回类型，列数由真实返回值的.shape提供。

源码：[准备观测项](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:591)。

[返回迁移入口](README.md) · [查看来源与覆盖表](00_迁移映射索引.md)
