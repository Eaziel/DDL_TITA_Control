# observations.py：机器人数据怎样成为33维观测、30维参数与6维状态

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-03；本次补写第6节。

被解释文件：[mdp/observations.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/observations.py)。

本文件提供四个观测入口：关节位置偏差、历史读取、30维动力学参数和6维额外状态。读完后应能把返回值中的每一组数字对应到物理量、存储对象与更新时机，并理解为什么有的观测用函数、有的需要保存状态的类。

本文核对当前源码与本地 IsaacLab-main / Sim 6.2 调用关系；未代运行仿真或训练。

## 1. 当前完整源码

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from ..actuators import TitaPDActuator
from .actions import TitaJointAction
from .events import RandomizeTitaMaterial

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import ContactSensor

    from ..tita_velocity_env import TitaVelocityEnv


def joint_pos_rel_zero_wheels(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    wheel_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return joint position offsets [rad], with wheel entries zeroed."""
    # 从 Scene 取得机器人
    robot: Articulation = env.scene[asset_cfg.name]
    # 计算全部关节的 当前位置－默认位置。减法生成新张量
    joint_pos_rel = robot.data.joint_pos.torch - robot.data.default_joint_pos.torch
    # 在新张量里，把两个轮子的位置偏差置零
    joint_pos_rel[:, wheel_cfg.joint_ids] = 0.0
    # 按指定的八关节顺序返回
    return joint_pos_rel[:, asset_cfg.joint_ids]


# MDP 观测函数，把历史交给 ObservationManager
def proprio_history(env: TitaVelocityEnv) -> torch.Tensor:
    """Return past proprioceptive frames in whole-frame order."""

    # 展开为长度10*33=330的向量
    return env.proprio_history.read().reshape(env.num_envs, -1)


class TitaDynamicsParameters(ManagerTermBase):
    """Randomized TITA dynamics parameters relative to the nominal robot."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv) -> None:
        """Store nominal inertial properties before startup randomization."""

        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]

        if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 1:
            raise ValueError("Select exactly one base body for dynamics observations.")
        self._base_id = asset_cfg.body_ids[0]

        # Observation terms are constructed before startup events run.
        self._nominal_mass = self._asset.data.body_mass.torch[:, self._base_id].clone()
        self._nominal_com = self._asset.data.body_com_pos_b.torch[:, self._base_id].clone()

        material = env.event_manager.get_term_cfg("randomize_material").func
        if not isinstance(material, RandomizeTitaMaterial):
            raise TypeError("Dynamics observations require RandomizeTitaMaterial.")

        # The event samples these per-robot values in its constructor.
        self._friction = material.friction_coeffs.to(self.device).unsqueeze(1)
        self._restitution = material.restitution_coeffs.to(self.device).unsqueeze(1)

        for actuator in self._asset.actuators.values():
            if not isinstance(actuator, TitaPDActuator):
                raise TypeError("Dynamics observations require TitaPDActuator groups.")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return base perturbations, material coefficients, and motor factors."""

        delta_mess = self._asset.data.body_mass.torch[:, self._base_id] - self._nominal_mass

        delta_com = self._asset.data.body_com_pos_b.torch[:, self._base_id, :] - self._nominal_com

        factors = []
        for name in ("motor_strength", "kp_factor", "kd_factor"):
            # Scatter actuator-local columns into the articulation's joint order.

            values = torch.ones_like(self._asset.data.joint_pos.torch)
            for actuator in self._asset.actuators.values():
                values[:, actuator.joint_indices] = getattr(actuator, name)

            # Gather into the task's explicit eight-joint observation order.
            factors.append(values[:, asset_cfg.joint_ids])

        # (N, 30) = mass 1 + CoM 3 + friction 1 + restitution 1 + 3*8 factors.
        return torch.cat(
            (
                delta_mess.unsqueeze(1),
                delta_com,
                self._friction,
                self._restitution,
                *factors,
            ),
            dim=-1,
        )


class TitaPrivilegedState(ManagerTermBase):
    """Base velocity, filtered wheel contacts, and action-history slot features."""

    def __init__(self, cfg: ObservationTermCfg, env: TitaVelocityEnv) -> None:
        """Initialize per-policy-step state samples and contact memory."""

        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]

        if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
            raise ValueError("Select exactly two wheel bodies in the required order.")

        self._robot: Articulation = env.scene[asset_cfg.name]
        self._sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

        legs = env.action_manager.get_term("leg_position")
        if not isinstance(legs, TitaJointAction):
            raise TypeError("Privileged state requires TitaJointAction terms.")
        self._lag_denominator = max(legs.cfg.lag_timesteps - 1, 1)

        self._linear_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_contacts = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._filtered_contacts = torch.zeros_like(self._last_contacts)

        # Dimension probing and reset before the first policy step keep a cold sample.
        self._sample_step = env.common_step_counter

    def __call__(
        self,
        env: TitaVelocityEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return six privileged features with one contact update per policy step."""

        if self._sample_step != env.common_step_counter:
            forces_z = self._sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, 2]
            contact = forces_z > 1.0

            # c_filtered(t) = c(t) OR c(t-1); samples are 10 ms apart.

            self._filtered_contacts.copy_(contact | self._last_contacts)
            self._last_contacts.copy_(contact)
            self._linear_velocity.copy_(self._robot.data.root_lin_vel_b.torch)
            self._sample_step = env.common_step_counter

        # This encodes the history SLOT, not the physical delay duration.
        lag_slot = env.action_lag_indices.to(dtype=torch.float32).unsqueeze(1) / self._lag_denominator

        return torch.cat(
            (
                2.0 * self._linear_velocity,
                self._filtered_contacts.float() - 0.5,
                lag_slot,
            ),
            dim=-1,
        )

    # Inherited reset() preserves contact memory and the current sample, as in Gym.
```

## 2. 一项观测怎样被管理器调用

当前 `env_cfg.py` 把函数或类放入 `ObsTerm(func=..., params=...)`。这句配置保存“调用谁、带哪些参数”；`ObservationManager` 才负责调用它。管理器把 `params` 字典展开成关键字实参，例如：

```python
value = term_cfg.func(env, **term_cfg.params)
```

若 `params={"asset_cfg": selection}`，展开后的含义就是 `func(env,asset_cfg=selection)`。`SceneEntityCfg` 是场景对象及关节/刚体选择的配置；管理器先根据名称解析编号，函数运行时直接使用 `joint_ids`、`body_ids` 取数据。

无须保存上一时刻信息的计算适合普通函数。有名义质量、接触记忆等状态时，本文件用继承 `ManagerTermBase` 的类：初始化保存引用和缓存，之后由 `__call__` 返回观测。类对象起初放在 `func` 中，Lab构造管理器时会执行：

```python
term_cfg.func = term_cfg.func(cfg=term_cfg, env=env)
```

赋值前右侧是类，调用是在创建实例；赋值后 `func` 保存实例，后面的 `func(env,...)` 会执行该实例的 `__call__`。`__call__` 是Python允许实例使用括号调用的特殊方法，与神经网络的 `forward` 约定分别由各自框架接线。

管理器还会在准备观测时调用观测函数，读取输出形状。配置条目声明了计算，框架就有理由在第一次step之前执行它。历史缓存必须提前创建；接触状态也必须防止“探测维度的一次读取”被误算成一个新时刻。

## 3. 关节相对位置：先计算偏差，再屏蔽轮角

```python
joint_pos_rel = robot.data.joint_pos.torch - robot.data.default_joint_pos.torch
joint_pos_rel[:, wheel_cfg.joint_ids] = 0.0
return joint_pos_rel[:, asset_cfg.joint_ids]
```

设当前角度为 $q_j$、默认角度为 $q_{0,j}$，`rel` 是relative，即相对默认值的差：

$$
q_j^{rel}=q_j-q_{0,j}.
$$

例如默认膝角−1.5rad，当前−1.4rad，偏差为+0.1rad。默认角没有使测量失去物理意义，它定义了零偏差的参考姿态。

`robot.data.joint_pos` 是Lab对关节状态的数据接口，`.torch` 取Torch形式，通常为 `(N,J_all)`。N是并行环境数量，J_all是机器人全部关节数。减法得到新的结果张量，所以把轮子列设零不会把物理轮角改成零。

轮子可以连续旋转，同一个接地运动状态可能对应相差多圈的角度；本任务沿用旧实现，去掉这种绝对轮角输入。轮速仍然保留。最后通过 `asset_cfg.joint_ids` 把列按任务要求排成左腿3个、右腿3个、左轮、右轮，共8列。

随后观测管理器还可能加噪声。因此“轮角列在本函数输出时为零”与“policy最终轮角观测严格为零”不同：当前配置在返回值上加±0.01的均匀噪声，轮角列也会得到噪声值。

完整policy33列由多个项共同组成：

| 列范围，左闭右开 | 内容 | 当前缩放 |
| --- | --- | --- |
| `0:3` | 机身角速度 | 0.25 |
| `3:6` | 机身坐标系中的重力方向 | 1 |
| `6:9` | `[vx_cmd,vy_cmd,wz_cmd]` | `[2,2,0.25]` |
| `9:17` | 本函数返回的8维位置偏差 | 1 |
| `17:25` | 8维实际关节速度 | 0.05 |
| `25:33` | 环境收到的8维原始动作 | 1 |

左闭右开表示包括起始列、不包括结束列，例如9:17恰好8列。本文件不单独生成全部33列；其他项复用Lab已有观测函数。

### 3.1 返回值之后，噪声和缩放怎样继续处理

当前Lab顺序为函数结果、噪声、裁剪、缩放，然后按组拼接。本任务四项噪声在最终输入中的半宽为：角速度0.2、重力方向0.05、位置0.01、速度0.075。

以角速度为例，配置原始噪声 $\epsilon\sim U(-0.8,0.8)$，缩放0.25后：

$$
y=0.25(\omega+\epsilon)
=0.25\omega+\epsilon',\qquad \epsilon'\sim U(-0.2,0.2).
$$

这解释了配置为何使用0.8而非0.2。`enable_corruption=True` 使本组噪声配置生效；它本身不产生一个新噪声分布。这里corruption描述对观测的扰动处理，具体分布由每项的noise配置提供。

扫描187维复用Lab `height_scan`：按当前配置，单点为 $5\operatorname{clip}(z_{ref}-0.4-z_{hit},-1,1)$。射线给出地面交点高度，函数把它转成相对高度输入。它不读取真实摄像机图片，也不由这里的缓存推算；射线起点抬高20m用于相交检测，不应再额外从结果中减20m。

## 4. 历史入口只读已有十帧

```python
return env.proprio_history.read().reshape(env.num_envs, -1)
```

`read()`返回 `(N,10,33)` 副本。`reshape(N,-1)` 保留N行，把每一行内部的时间轴和特征轴展开；`-1`让Torch根据元素总数推算为330列。顺序是最旧帧完整33列、下一帧完整33列，直到最新历史帧。

`reshape`不求和，不训练参数，也不删数。第n个机器人的第t帧第j个特征位于展开后的第 `33*t+j` 列。函数每次读取不会推进历史，推进在 [环境step](../tita_velocity_env.py.md) 开头显式完成。

## 5. 动力学参数类先保存名义值，再读取随机后的差

“名义值”是随机化前机器人原本的参数。构造 `TitaDynamicsParameters` 时，动作/事件等对象已建立，startup事件尚未执行物理随机化，因此本类保存：

```python
self._nominal_mass = self._asset.data.body_mass.torch[:, self._base_id].clone()
self._nominal_com = self._asset.data.body_com_pos_b.torch[:, self._base_id].clone()
```

质量形状 `(N,)`，质心位置形状 `(N,3)`。必须复制名义数据，后续质量或质心改变时，参考值仍保持原样。每次调用再计算：

$$
\Delta m_n=m_n-m_{0,n},\qquad
\Delta\boldsymbol c_n=\boldsymbol c_n-\boldsymbol c_{0,n}.
$$

质量单位kg，质心偏移单位m；`body_com_pos_b` 是相应刚体局部参考系中的质心位置，不是世界中机器人移动的位置。`self._base_id` 只选base_link，构造函数要求恰好一个机身刚体。源码把质量差变量拼为 `delta_mess`，该变量实际保存的数学量仍是 $\Delta m$。

例如名义机身质量8kg、随机后9.2kg，则质量观测为1.2；质心从 `[0,0,0.03]` 变为 `[0.01,-0.02,0.04]`，偏移观测为 `[0.01,-0.02,0.01]`。

### 5.1 材质数值从事件实例取得

```python
material = env.event_manager.get_term_cfg("randomize_material").func
self._friction = material.friction_coeffs.to(self.device).unsqueeze(1)
```

事件配置中的 `func` 此时已经是 `RandomizeTitaMaterial` 实例。本类读取事件为每台机器人抽到的系数。`isinstance` 检查实际对象类型，防止拿到一个没有这些属性的其他事件实现。

`.to(self.device)` 使这份张量与本观测的其他张量在同一设备；如果设备/类型已符合要求，`.to` 可以返回原张量。`unsqueeze(1)` 把每机器人一个值的 `(N,)` 变成一列 `(N,1)`。恢复系数同理。

这里读取的是当前事件实现提供的随机化参数，startup会把它们写进机器人碰撞形状。若以后另写事件再次改变物理材质，不能假定本类会自动同步那个新来源；当前迁移只使用这套startup参数。

### 5.2 三组8列因子，为什么先散到全部关节再重排

执行器分腿和轮两组，每组因子按该组自己的关节顺序保存。观测却需要统一的8关节顺序，因此使用两次索引：

```python
values = torch.ones_like(self._asset.data.joint_pos.torch)
for actuator in self._asset.actuators.values():
    values[:, actuator.joint_indices] = getattr(actuator, name)
factors.append(values[:, asset_cfg.joint_ids])
```

`actuators.values()`遍历字典中保存的执行器对象。`getattr(actuator,name)`根据字符串读属性，例如name为 `"kp_factor"` 就相当于 `actuator.kp_factor`。

先创建全部关节列的1矩阵，把每个执行器的局部数据按其 `joint_indices` 填回整台机器人的关节编号，再按任务 `asset_cfg.joint_ids` 取得8列。这样不会把“执行器内部第0列”误当成“任务动作第0列”。

外层循环依次处理 `motor_strength`、`kp_factor`、`kd_factor`，因此 `factors` 是包含三个 `(N,8)` 张量的Python列表。

### 5.3 最后的cat精确得到30维

$$
d_n=[\Delta m_n,\Delta\boldsymbol c_n,\mu_n,e_n,
\boldsymbol s_n,\boldsymbol f_{p,n},\boldsymbol f_{d,n}].
$$

$\mu$ 是摩擦系数，e是恢复系数，s是电机强度倍率，$f_p/f_d$ 分别是Kp/Kd倍率。数量为 $1+3+1+1+8+8+8=30$。

`delta_mess.unsqueeze(1)` 把 `(N,)` 变成 `(N,1)`，其余各项已是二维。`*factors` 是可迭代对象展开语法，把列表中的三个张量依次放入传给cat的元组；没有执行矩阵乘法。`torch.cat(...,dim=-1)` 沿最后一维横向拼列，同一机器人的各项仍在同一行。

## 6. 六维状态：速度3列、接触2列、延迟编号1列

### 6.1 为什么在这里放一个延迟设置

当前动作代码为了模拟控制目标晚一点生效，会保存最近三个物理步的目标，再按每台机器人固定的编号取一份执行。数组每个位置曾称为“槽”；编号0/1/2分别取两步前、一步前、当前的目标，对应5/2.5/0ms延迟。控制延迟怎样产生，先读[actions.py第6节](actions.py.md#6-控制延迟三个历史位置与编号怎样联系起来)，本节只解释它怎样成为观测。

仿真知道这台机器人抽到了哪种延迟。把这个设置提供给critic，可以让价值预测考虑控制响应条件：两台机器人姿态、速度相近，动作的生效时刻仍可能因延迟不同而不同，后续reward/cost也可能不同。这里说明设计用途，不表示已证明预测提升。

`privileged_state`是任务自己命名的观测组，包含机身运动、接触判断和延迟设置；其中的每一个数字不要求都来自独立物理传感器。当前高斯策略直接算动作时只取policy33与history330；critic会读取这6列及30维参数。辅助学习还可以按已有目标布局使用部分信息，具体取数由相应代码决定。

### 6.2 六维到底是哪六个数

| 本组第几列 | 内容 | 数量 |
| --- | --- | --- |
| 0、1、2 | 机身线速度vx、vy、vz，各乘2 | 3 |
| 3、4 | 左、右轮滤波后的接触标记，各减0.5 | 2 |
| 5 | 这台机器人的共享延迟编号，按下文除以最大编号 | 1 |

所以6=3+2+1。左右腿和轮共用一个编号，延迟只占一列；三个可选编号并不使这一项变成三列。每行返回：

$$
p_n=[2v_x^b,2v_y^b,2v_z^b,
\widetilde C_L-0.5,\widetilde C_R-0.5,
i_n/\max(L-1,1)].
$$

`root_lin_vel_b` 在当前Lab中是根刚体质心线速度的机身坐标表达，乘2沿用观测缩放。轮接触判断来自当前世界坐标系法向力的z分量：

$$
C_{t,j}=\mathbf1\{F^w_{z,t,j}>1\mathrm N\},\qquad
\widetilde C_{t,j}=C_{t,j}\lor C_{t-1,j}.
$$

$\lor$ 是逻辑OR，任一次为True就为True。`contact | self._last_contacts` 对布尔张量逐元素计算OR。随后把原始当前判断存入 `_last_contacts`，不是把滤波结果存回去。若原始接触序列是 `[True,False,False]`，滤波结果为 `[True,True,False]`，可以填住一个采样间隔的短暂失联；不会一直保持接触。

`.float()`把False/True变成0.0/1.0，再减0.5得到−0.5/+0.5。接触两个采样相隔0.01s。这个短时逻辑滤波不改变物理接触，也不把所有物理子步做时间平均。

### 6.3 整数编号怎样变成(N,1)的最后一列

对应本文件的两处代码：

```python
self._lag_denominator = max(legs.cfg.lag_timesteps - 1, 1)
```

```python
lag_slot = env.action_lag_indices.to(dtype=torch.float32).unsqueeze(1) / self._lag_denominator
```

`action_lag_indices`本来是每台机器人一个整数，形状(N,)。假设三台机器人的编号为 `[1,2,0]`，依次对应2.5、0、5ms控制延迟。

`.to(dtype=torch.float32)`使整数成为浮点数，数值仍为 `[1.0,2.0,0.0]`。`.unsqueeze(1)`增加列轴，得到(N,1)的 `[[1.0],[2.0],[0.0]]`。当前L=3，分母max(3−1,1)=2，最后得到 `[[0.5],[1.0],[0.0]]`。每台机器人因此新增一个数，能与其他二维观测沿列拼接。

例如某台机器人三维线速度为 `[0.2,-0.1,0]`，左右轮滤波接触为 `[True,False]`，编号为1，则本组这一行是：

```text
[0.4, -0.2, 0.0, 0.5, -0.5, 0.5]
 └─速度×2──┘    └─接触─┘   └延迟编号÷2
```

最后的0.5表示编号1。除以2是把选择编号表示到0到1区间；它没有新增一次延迟，也没有把0.5作为位置目标交给PD。L=1时分母取1避免除零，这时唯一编号为0，对应零延迟。

### 6.4 这一列怎样进入critic

`torch.cat`把上述3列、2列、1列合成(N,6)，作为 `privileged_state`。打包函数将本组放在586列中的220:226，延迟值就在第225列；critic再用220:256一起取得这6列与后面的30维参数。

critic把这项设置与其他特征共同用于价值预测。它通过训练学习不同输入与未来累计reward/cost的关系，输入中出现0.5不会自动触发任何“延迟0.5”的物理操作。真实的目标选择始终由actions.py使用原始整数编号完成。

## 7. 同一策略步读多次，为什么只更新一次接触

`common_step_counter` 每次父环境step完成物理推进后加1；4个物理步合计只加1。`_sample_step`保存本对象已经采样的策略步号：

```python
if self._sample_step != env.common_step_counter:
    # 更新接触记忆和线速度
    self._sample_step = env.common_step_counter
```

这让同一步中的第二次、第三次调用复用已有样本。否则重复调用会把“前一策略步接触”覆盖成“本步第一次读取的接触”，OR就失去原含义。

初始化把 `_sample_step` 设为当前计数，同时速度与接触缓存为零，所以维度探测时返回冷启动样本，不抢先推进。类没有覆盖reset，继承的reset保留接触记忆与本步样本，这也是当前迁移规则。

父环境会在reset后生成新观测，interval推扰也在最终观测前执行。为了沿用旧时序，自定义环境在reset改物理状态之前、推扰事件在改写速度之前，都调用一次 `compute_group("privileged_state")`。这次调用先保存刚完成物理推进的速度/接触；同一策略步最终再读时不会被reset或推扰后的值覆盖。两处调用分别保护不同的修改入口，依据是明确的调用顺序，不是因为compute这个名字有特殊能力。

这样得到的整份观测存在有意保留的时序差别：刚reset后的policy可以是新姿态，而这6列仍是reset前保存的本步样本。本文描述当前Gym对齐实现；它没有声称所有列永远来自同一个物理瞬间。

## 源码与API依据

- [ObservationManager：维度探测与noise/clip/scale](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/observation_manager.py)
- [ManagerBase：名称解析与类实例化](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/manager_base.py)
- [本项目固定观测布局](../observation_layout.py.md)
- [接触力的物理与索引解释](terminations.py.md)
- [PyTorch cat](https://docs.pytorch.org/docs/2.11/generated/torch.cat.html)
