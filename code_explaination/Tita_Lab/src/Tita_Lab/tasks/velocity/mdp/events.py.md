# events.py：随机化与推扰怎样真正改变机器人

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-02。

被解释文件：[mdp/events.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/events.py)。

本文件改变物理参数、动作延迟编号或机身速度。读完后应能从配置的触发时机找到执行函数，再看清采样结果写进哪份数据；尤其区分“生成观测中的参数”和“修改物理引擎实际使用的参数”。

本次只读核对当前文件和本地 IsaacLab-main / Sim 6.2，不代运行事件、物理仿真或训练。

## 1. 当前完整源码

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import sample_uniform

from ..actuators import TitaPDActuator
from .actions import TitaJointAction

if TYPE_CHECKING:
    from omni.physics.tensors.api import ArticulationView

    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    from ..tita_velocity_env import TitaVelocityEnv


def push_by_setting_xy_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Replace world-frame horizontal root COM velocity with random samples."""

    # Preserve the physics-result sample before this event changes root velocity.
    env.observation_manager.compute_group("privileged_state")

    asset: Articulation = env.scene[asset_cfg.name]
    selection = slice(None) if env_ids is None else env_ids
    velocity_w = asset.data.root_vel_w.torch[selection].clone()
    velocity_w[:, :2] = sample_uniform(
        velocity_range[0],
        velocity_range[1],
        (velocity_w.shape[0], 2),
        device=asset.device,
    )
    asset.write_root_velocity_to_sim_index(root_velocity=velocity_w, env_ids=env_ids)


def randomize_tita_motor_factors(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    motor_strength_range: tuple[float, float],
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
) -> None:
    """Sample independent motor factors for every environment and joint."""
    if asset_cfg.joint_ids != slice(None):
        raise ValueError("Select the entire robot without a joint subset.")

    asset: Articulation = env.scene[asset_cfg.name]
    selection = slice(None) if env_ids is None else env_ids
    for actuator in asset.actuators.values():
        if not isinstance(actuator, TitaPDActuator):
            raise TypeError("Motor-factor randomization requires TitaPDActuator groups.")
        for factors, bounds in (
            (actuator.motor_strength, motor_strength_range),
            (actuator.kp_factor, kp_range),
            (actuator.kd_factor, kd_range),
        ):
            factors[selection, :] = sample_uniform(
                bounds[0], bounds[1], factors[selection, :].shape, device=asset.device
            )


def randomize_tita_action_lag(
    env: TitaVelocityEnv,
    env_ids: torch.Tensor | None,
) -> None:
    """Sample a shared action-history slot for each robot."""

    legs = env.action_manager.get_term("leg_position")
    wheels = env.action_manager.get_term("wheel_effort")

    if not isinstance(legs, TitaJointAction) or not isinstance(wheels, TitaJointAction):
        raise TypeError("Action lag randomization requires TitaJointAction terms.")
    if legs.cfg.lag_timesteps != wheels.cfg.lag_timesteps:
        raise ValueError("Leg and wheel command histories must have the same length.")

    selection = slice(None) if env_ids is None else env_ids

    # Write in place: both action terms retain references to this (N,) tensor.
    env.action_lag_indices[selection] = torch.randint(
        0, legs.cfg.lag_timesteps, env.action_lag_indices[selection].shape, device=env.device
    )


class RandomizeTitaMaterial(ManagerTermBase):
    """Assign shared friction and restitution to all collision shapes of each robot."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv) -> None:
        # 该实现依赖 Kit，等仿真启动、事件实例开始构造时再导入。
        from isaaclab_physx.physics import PhysxManager

        super().__init__(cfg, env)
        if not isinstance(env.sim.backend_manager, PhysxManager):
            raise ValueError("RandomizeTitaMaterial supports Isaac Sim PhysX only.")

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        if asset_cfg.body_ids != slice(None):
            raise ValueError("Select the entire robot without a body subset.")
        self.asset: Articulation = env.scene[asset_cfg.name]

        num_buckets = cfg.params["num_buckets"]
        friction_range = cfg.params["friction_range"]
        bucket_ids = torch.randint(0, num_buckets, (env.num_envs,), device="cpu")
        friction_buckets = sample_uniform(friction_range[0], friction_range[1], (num_buckets,), device="cpu")
        self.friction_coeffs = friction_buckets[bucket_ids]

        restitution_range = cfg.params["restitution_range"]
        restitution_ids = torch.randint(0, num_buckets, (env.num_envs,), device="cpu")
        restitution_buckets = sample_uniform(restitution_range[0], restitution_range[1], (num_buckets,), device="cpu")
        self.restitution_coeffs = restitution_buckets[restitution_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
    ) -> None:

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device="cpu", dtype=torch.int32)
        else:
            env_ids = env_ids.to(device="cpu", dtype=torch.int32)

        view: ArticulationView = self.asset.root_view

        materials = wp.to_torch(view.get_material_properties()).clone()
        friction = self.friction_coeffs[env_ids].unsqueeze(-1)
        restitution = self.restitution_coeffs[env_ids].unsqueeze(-1)
        materials[env_ids, :, 0] = friction
        materials[env_ids, :, 1] = friction
        materials[env_ids, :, 2] = restitution
        view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32),
            wp.from_torch(env_ids, dtype=wp.int32),
        )
```

## 2. Event表示按指定时机调用一项操作

环境创建配置时，`EventTerm(func=...,mode=...,params=...)` 保存实现、触发模式和参数。`EventManager` 创建所需实例并在相应时机调用。函数的名称没有特殊调度能力；调度依据是配置与环境实际调用。

当前事件安排为：

| 时机 | 当前操作 | 实现位置 |
| --- | --- | --- |
| startup，建立场景后一次 | 机器人材质、机身质量、机身质心、电机倍率、动作延迟编号 | 本文件与Lab已有函数 |
| reset，每个新回合 | 关节初态、机身位姿与速度 | 直接复用Lab函数 |
| interval，间隔事件 | 每15秒改写世界水平速度 | 本文件push函数 |

startup采样的参数会跨回合保留，reset不自动再次采样它们。当前 `is_global_time=True` 的推扰按公共仿真时间计时，不随每台机器人reset重新计算一个完整15秒倒计时。

“域随机化”在这里指让不同机器人经历不同物理参数或扰动，希望训练接触多种条件。它不要求每步都换参数，也不能据配置开启就保证真机鲁棒性；效果要靠训练与验证。

## 3. `env_ids`表示本次改哪些机器人

函数一般收到环境env、环境编号env_ids与配置params。`env_ids=None` 时使用 `slice(None)`，相当于所有行；否则选中M台机器人。N是全部并行机器人数，M可以小于N。

`asset_cfg.name` 是Scene中的对象键，当前为 `"robot"`，`env.scene[asset_cfg.name]`返回机器人Articulation对象。它是物理资产对象，不是单个关节，不是训练wrapper。

`TYPE_CHECKING`块中的导入只为类型检查工具提供名字，运行时不会执行。`from __future__ import annotations` 延后注解的解析，允许类尚未真正导入时先写类型名。这个机制解释了为什么源码可写 `asset: Articulation`，同时不为了注解在模块导入阶段启动Kit。

## 4. 推扰：改写世界水平速度，其他速度保留

### 4.1 先保存本步额外状态

```python
env.observation_manager.compute_group("privileged_state")
```

这句在物理推进后、修改根速度前调用6维观测。该观测实例用 `common_step_counter` 保证每步只保存一次速度与接触，因此最终观测再次读取时保留推扰前的样本。否则推扰新写入的速度会改变原本要保留的旧Gym时序。

这句不会保存所有观测，也不会让物理再走一步；它只通过调用这一个组的观测项产生相应缓存效果。

### 4.2 复制六维速度，再替换两列

```python
velocity_w = asset.data.root_vel_w.torch[selection].clone()
velocity_w[:, :2] = sample_uniform(l, u, (velocity_w.shape[0], 2), device=asset.device)
```

当前 `root_vel_w` 是根刚体质心的空间速度，选中行后为 `(M,6)`；前三列是世界坐标线速度 `[vx,vy,vz]`，后三列是角速度 `[wx,wy,wz]`。`.clone()`创建临时可修改副本，随后只将vx/vy两列改为新采样值。

`sample_uniform(l,u,shape,device)`是Lab随机采样函数，当前范围−1到1，生成 `(M,2)`。因此操作是：

$$
v_{x,n}^{new}\sim U(-1,1),\qquad
v_{y,n}^{new}\sim U(-1,1).
$$

这是替换，不是 $v^{new}=v^{old}+\delta v$。原vx若0.4、新采样−0.7，结果是−0.7，不是−0.3。vz和三列角速度保留从原状态复制的值。

最后 `write_root_velocity_to_sim_index(root_velocity=velocity_w,env_ids=env_ids)` 把新根速度写回物理状态。代码没指定作用力或持续时间；它用速度跳变实现扰动，不能从这里直接读出多少N、多少秒的真实推力。

## 5. 电机随机化：三类倍率分别写进每组执行器

每个 `TitaPDActuator` 保存 `motor_strength`、`kp_factor`、`kd_factor`，形状是 `(N,J_group)`，腿组J=6、轮组J=2。startup将每个因子独立从当前 `[0.8,1.2]` 范围采样。

核心循环先遍历执行器，再遍历三组“张量与范围”的配对：

```python
for factors, bounds in (
    (actuator.motor_strength, motor_strength_range),
    (actuator.kp_factor, kp_range),
    (actuator.kd_factor, kd_range),
):
    factors[selection, :] = sample_uniform(
        bounds[0], bounds[1], factors[selection, :].shape, device=asset.device
    )
```

外面的括号是含三个二元组的元组。第一次循环，`factors`引用 `actuator.motor_strength`、`bounds`引用强度范围；第二次换成Kp因子与其范围；第三次换成Kd因子。`for factors,bounds in ...` 是解包，把每个二元组的两个元素分配给两个变量。

`bounds[0]/[1]`取下/上限；`.shape`使采样矩阵恰好对应选中的M行、该组的J列。左侧索引赋值原位修改执行器保存的因子；若只写 `factors = 新张量`，通常只是局部变量换指向，执行器属性仍指旧对象。

这些因子在控制公式中产生：

$$
\tau^{calc}=s\,[f_pK_p(q^{des}-q)+f_dK_d(\dot q^{des}-\dot q)+\tau^{ff}],
$$

轮前馈另按该组配置乘 $f_p$。s、$f_p$、$f_d$是无量纲倍率。名义Kp40、因子1.1给44，之后强度s还会整体乘输出。初始化全1让基准计算保持名义参数，startup再覆盖它们。完整力矩链见 [执行器](../actuators.py.md)。

函数要求选择整台机器人，当前不支持只选一个关节子集；`isinstance` 检查确保对象确实有这三个因子属性。

## 6. 延迟随机化：抽取共享编号，保留共享存储

函数读取腿与轮动作项，检查都属于 `TitaJointAction`，且历史长度L相同。否则无法保证同一个编号在两组都合法。

```python
env.action_lag_indices[selection] = torch.randint(
    0, L, env.action_lag_indices[selection].shape, device=env.device
)
```

`torch.randint(low,high,size)`生成整数，下限包含、上限不包含。L=3时每行抽0、1或2，当前startup抽一次。这个编号选的是历史位置，实际延迟为 $(L-1-i)dt_{physics}$。

关键是左侧有 `[selection]`：把新编号写入原张量。腿、轮在构造时都引用 `env.action_lag_indices`；原位写入后，三个引用看到同一组更新。若改成整个属性重新赋一个新张量，已有动作项仍可能持有旧张量，随机化就无法传递过去。

动作历史每物理步保存一份目标，再按编号选择，详见 [actions.py](actions.py.md)。这里没有存历史，也没有输出力矩；它只决定每台机器人取哪一个时间位置。

## 7. 材质类：先抽每台机器人的参数，再写入碰撞形状

### 7.1 为什么用类，并在构造时采样

`RandomizeTitaMaterial`保存 `friction_coeffs` 与 `restitution_coeffs`。事件管理器创建这个实例后，观测类即可读取已抽好的参数；startup调用实例时再把它们写入物理。构造参数与实际应用分成两段，是当前观测与事件初始化顺序的需要。

构造中延迟导入 `PhysxManager`，此时仿真上下文已经启动。`isinstance(env.sim.backend_manager,PhysxManager)`检查实际后端，本实现只支持Isaac Sim PhysX。编辑器没给backend_manager补全不改变属性的实际所属对象；类型补全与运行存在性要分开核对。

### 7.2 bucket采样究竟抽了什么

当前64个bucket，摩擦范围0.2到2.75。先抽64个候选摩擦系数，再让每台机器人抽候选编号：

$$
\mu^{bucket}_b\sim U(0.2,2.75),\quad
I_n\sim\operatorname{Uniform}\{0,\ldots,63\},\quad
\mu_n=\mu^{bucket}_{I_n}.
$$

编号数组 `(N,)`、候选数组 `(64,)`，用后者按前者索引，得到 `(N,)` 的机器人摩擦。固定这次候选集合后，机器人使用其中某个值，不是每台都重新连续采样一个独立值；不同机器人可以选同一个bucket。

恢复系数用独立的64个候选和独立编号，在0到1中采样。恢复系数描述碰撞法向速度恢复程度；0接近不反弹，1对应理想无损法向恢复的端点含义。实际解算仍受材质组合规则与接触过程影响，不单由此一个数决定全部碰撞行为。

### 7.3 PhysX材质数组的三列

```python
view = self.asset.root_view
materials = wp.to_torch(view.get_material_properties()).clone()
```

这里 `root_view` 是物理后端的articulation视图，提供读写整批机器人物理属性的接口。材质数组形状 `(N,S,3)`，S是每台机器人的碰撞形状容量，最后三列依次为静摩擦、动摩擦、恢复系数。

`body`是刚体，`shape`是刚体上的碰撞形状，一个刚体可有多个shape。事件按机器人把它所有shape赋相同随机值：

```python
materials[env_ids, :, 0] = friction
materials[env_ids, :, 1] = friction
materials[env_ids, :, 2] = restitution
```

`friction/restitution`从 `(M,)`经 `unsqueeze(-1)`成为 `(M,1)`，沿S列广播。静/动摩擦在本实现使用同一摩擦系数，恢复系数另抽。

机器人每台可以不同；同台机器人的腿、轮等shape共享该次值。地面是另一个资产，事件没把地面也写成这些随机值。机器人spawn时复制地面基准材质，是提供初始材质；startup对机器人覆盖随机值之后，二者不必相同。

### 7.4 Torch、Warp与CPU各负责哪部分

Warp是NVIDIA的数组与并行计算工具，能服务多个仿真后端；MuJoCo Warp使用它，不代表所有Warp数组都在使用MuJoCo。当前仍是PhysX，只是此视图以Warp形式提供数组。

`wp.to_torch`通常建立共享底层数据的Torch表示，`wp.from_torch`反向建立Warp表示，两者不是把数据转交另一个物理引擎。后面的 `.clone()`才给材质创建独立可编辑副本。[Warp官方互操作说明](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html)

构造中的候选和编号放CPU；调用中env_ids转CPU int32，配合此后端材质属性接口。设备由具体接口的数据要求决定，不能因为动力学跑GPU就把所有属性接口一概改成cuda。

`.to(device="cpu",dtype=torch.int32)`同时指定存放位置与整数类型；实际需要变化时产生转换结果。`wp.from_torch(materials,dtype=wp.float32)`与 `wp.from_torch(env_ids,dtype=wp.int32)`在调用setter前提供匹配的Warp数组表示。最后 `view.set_material_properties(data,indices)`才把编辑后的属性应用回物理后端。

若只修改Torch副本而不调用setter，不能据观测或副本的值宣称物理材质已经更改。`get/set_material_properties`的本地实现属于Kit扩展，API源码列在文末；缺编辑器提示不说明没有这个方法。

## 8. 本文件以外的reset与质量随机化仍有明确来源

当前质量事件复用Lab `randomize_rigid_body_mass`，给base_link质量加−1到3kg，并要求重算惯性；质心事件复用 `randomize_rigid_body_com`，各轴偏移−0.1到0.1m。它们不需要为了“完整”再复制一遍到本文件。

关节reset按默认关节角乘0.5到1.5、初速度为零；机身reset在默认根状态与地形原点上叠加配置的位姿和速度范围，z偏移0到0.2m。默认根高度来自机器人资产配置；它与reset随机偏移是两项相加的来源。出生穿透问题需要沿这条链核查真实几何，本轮文档更新没有修改高度或随机范围。

事件解决的是在指定时刻改变环境条件。如何从事件实例读出30维参数由 [observations.py](observations.py.md) 说明；模型如何利用观测，则继续由policy/critic定义，事件本身不完成学习。

## 源码与API依据

- [Lab EventManager：startup/reset/interval](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/event_manager.py)
- [Lab已有物理随机化与reset函数](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/events.py)
- [PhysX ArticulationView材质API](/home/eaziel/my_venvs/lab3_sim62/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.4.4+110.4.0.lx64.r.cp312.u7f4/omni/physics/tensors/api.py)
- [Warp from_torch](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.from_torch.html)
- [PyTorch Tensor.to](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.to.html)
