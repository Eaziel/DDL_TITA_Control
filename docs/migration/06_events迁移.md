# 06 Events：回合初值、机身参数与定时推扰

更新：2026-10-19。**当前两个reset项和两个startup项均已保存并静态核对；第6节定时推扰为待写方案。** 第1—4节解释每回合的关节/机身初值，第5节解释创建时的质量与质心随机化，第6节给出每15 s改变实际水平速度的推扰。摩擦、恢复系数和控制随机化仍待迁移；未运行真实环境。

## 1. Gym 原来怎么做

旧 `reset_idx(env_ids)` 为指定环境开始新回合，主要调用：

```python
self._reset_dofs(env_ids)
self._reset_root_states(env_ids)
self._resample_commands(env_ids)
```

随后清动作、接触、历史、延迟等缓冲。迁移时不把整个reset_idx照搬成一个Event；物理状态、命令与各自缓存分别由对应对象管理。

### 1.1 八个关节

旧 `_reset_dofs()` 的核心是：

```python
self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(
    0.5, 1.5, (len(env_ids), self.num_dof), device=self.device
)
self.dof_vel[env_ids] = 0.0
```

每环境、每关节抽一个比例，乘保存的默认角，关节速度归零。默认角0仍为0；0.8 rad变为0.4～1.2 rad；-1.5 rad变为-2.25～-0.75 rad。这是乘法随机化，不是在默认角上加±0.5 rad。

### 1.2 机身根状态

当前粗糙地形对应旧 `custom_origins` 分支，规则为：

- 位置等于默认位置加该环境的地形原点，再给xy各加U(-1,1) m，z额外加U(0,0.2) m。
- 姿态保持默认旋转，不随机转向。
- 世界系三个线速度各取U(-0.5,0.5) m/s，三个角速度各取U(-0.5,0.5) rad/s。

默认机身z为0.3 m，所以相对地形原点的出生z为0.3～0.5 m。平地旧分支不增加这份xy扰动；下文按当前生成地形配置选范围。

## 2. Lab 中怎样写

当前[env_cfg.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py)已登记以下两个TITA重置项，使用EventTerm、SceneEntityCfg、mdp和configclass：

```python
@configclass
class EventCfg:
    """Reset TITA joints and base state at the start of each episode."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.2),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
```

`VelocityEnvCfg` 已有 `events: EventCfg = EventCfg()`，保持该登记即可。两个条目均有本地Lab实现，本步无需新增MDP函数或改环境子类。

## 3. 为什么这样迁移

### 3.1 Event 指定在何时改变状态

这里 `mode="reset"` 表示每次这些环境开始新回合时执行，不是每个物理步执行。环境结束后会自动走重置；公开reset也会走重置。

官方 `ManagerBasedRLEnv._reset_idx()` 调用EventManager执行reset项。当前 `TitaVelocityEnv._reset_idx()` 先调用这个父类，再清自定义历史，因此新增两个Event会自然接入已有流程：

```text
重置指定env_ids
  → 父类重置场景内部缓冲
  → EventManager执行关节、机身reset项
  → 官方manager清理观测/动作/奖励等，命令manager重采样
  → 子类清这些环境的ProprioHistory
```

不需要再手动调用这两个重置项。创建时的物理参数随机化使用startup模式，见第5节；定时推扰使用interval模式，后续再迁移。

### 3.2 init_state 是参考，Event 是实际开局状态

`tita.py` 的init_state保存默认姿态和速度。Action位置偏置、相对关节观测和本次重置都可以使用这份参考。

reset Event每回合重新抽样并写入实际物理状态，不会把默认参考本身改成随机值。例如默认膝角仍为-1.5，某次实际重置到-1.8，观测相对位置便为-0.3 rad。

### 3.3 两个原函数对应两个Lab条目

`reset_joints_by_scale` 读取default_joint_pos/vel，乘给定区间的随机值，再写回仿真。SceneEntityCfg("robot")指当前机器人，未限选关节时处理其全部八关节。

`reset_root_state_uniform` 取默认根位置，加env_origins和pose偏移；零旋转偏移保持默认姿态。velocity_range是加在默认根速度上的随机偏移，当前默认根线/角速度均为0，与旧直接采样相对应。

同名键在两个字典中单位不同：pose_range的xyz为m、roll/pitch/yaw为rad；velocity_range的xyz为m/s、roll/pitch/yaw为rad/s。它们都修改实际状态，区别于Commands提供的期望速度。

本地reset_root_state_uniform是可调用的ManagerTerm类，EventManager负责创建和调用；配置时填类引用即可，无需自行实例化。

## 4. 差异与验证边界

Lab关节条目还会将位置/速度夹到软限位，旧 `_reset_dofs()` 没有这一步。当前资产soft_joint_pos_limit_factor继承1.0；本轮只读USD得到leg_2限位约[-1.920,3.491] rad、leg_3约[-2.670,-0.698] rad，上述非零默认角的采样区间均在限位内。若以后改默认角或限位，需要重核这项差异。

当前源码已保存两项重置和两项质量/质心随机化；不能把它称为旧domain_rand全部完成。摩擦、恢复系数、电机强度、Kp/Kd、延迟仍待迁移，推扰待写方案见第6节。终止条件、20 s时长、10项奖励及最终奖励合成已接入，见07/08章；真实reset和自动结束流程仍待运行。

重置部分已有本地API、参数、默认状态与USD限位静态核对；尚未导入真实配置或执行重置/仿真。

## 5. 创建时给每台机器人不同的基座质量和质心

### 5.1 Gym 原来怎么做

有效配置来自`tita_constraint_config.py`的domain_rand：

```python
randomize_base_mass = True
added_mass_range = [-1., 3.]
randomize_base_com = True
added_com_range = [-0.1, 0.1]
```

`_create_envs()`每创建一台机器人，就调用一次`_process_rigid_body_props()`。其中两段计算是：

```python
rng_mass = self.cfg.domain_rand.added_mass_range
rand_mass = np.random.uniform(rng_mass[0], rng_mass[1], size=(1, ))
props[0].mass += rand_mass

rng_com = self.cfg.domain_rand.added_com_range
rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
props[0].com += gymapi.Vec3(*rand_com)
```

按旧实现，props[0]是要修改的基座。质量加一个U(-1,3) kg的数；质心在基座坐标系中沿x/y/z各加一个U(-0.1,0.1) m的偏移。这些值在创建时抽取，之后该环境的各回合沿用。随机的是物理模型参数；第1—4节随机的是每回合的运动状态。

创建函数随后提交：

```python
self.gym.set_actor_rigid_body_properties(
    env_handle, actor_handle, body_props, recomputeInertia=True
)
```

所以迁移质量时还要核对惯量处理，不能只找到一个名字相似的质量函数就算完成。

### 5.2 Lab 中怎样写

**以下两项已由用户写入，参数与此前方案的AST核对通过。** 它们位于现有EventCfg的`reset_robot_joints`之前，原有两个reset项保留。类说明目前仍写reset，可随本次扩展改为`Configure physical randomization, resets, and disturbances.`。

```python
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
            },
        },
    )
```

两项均用本地Lab原生实现。任务mdp的`.pyi`已经转发官方MDP，不需要新建任务`mdp/events.py`或新增导出；`VelocityEnvCfg.events: EventCfg = EventCfg()`也已存在。

### 5.3 参数分别在改变什么

`SceneEntityCfg("robot", body_names="base_link")`先选择Scene中的robot，再按名字选其基座刚体。当前USD中实际路径为`/TITARobot/Geometry/base_link`；仿真里的身体编号由框架解析，不写死为0，也不修改所有腿段。

质量项使用加法与均匀分布：

\[
m_e=m_0+\Delta m_e,\qquad\Delta m_e\sim U(-1,3)\ \mathrm{kg}.
\]

2026-10-19静态读取当前USD，基座质量约13.2 kg，单位设置为kg、m，因此该方案得到约12.2～16.2 kg。这里是基座质量范围，整机还包含其他刚体。负的增量表示比标称值轻，最终质量仍为正；不是把基座质量直接设置在[-1,3] kg。

质心项计算：

\[
\mathbf c_e=\mathbf c_0+
[\Delta x_e,\Delta y_e,\Delta z_e].
\]

当前USD的基座局部质心约为`(0.020452, 0.000003546, 0.018710)` m，配置给的是在这份参考上的偏移。三轴各在±0.1 m内取值，位移向量长度可以大于0.1 m。物理模型中的重心位置因此变化，机身外观和出生位置不被平移。

这组随机化让策略在质量与重心不同的机器人上经历运动，学习适应物理参数变化。本步沿用旧任务的范围，尚未依据当前仿真表现调整范围。

### 5.4 startup 如何接入执行流程

环境创建各manager后，`ManagerBasedRLEnv.load_managers()`中已有：

```python
if "startup" in self.event_manager.available_modes:
    self.event_manager.apply(mode="startup")
```

EventManager按mode保存配置，在startup执行时调用两项。`func`指向两个可调用的ManagerTerm类；管理器负责实例化与调用，用户只填类引用。

流程对应为：

```text
创建整批环境和机器人、建立物理视图
  → 初始化manager
  → 执行startup：每台机器人抽质量增量和质心偏移
  → 开始训练
  → 回合结束时只执行reset项，重置关节/机身状态
```

例如环境0和环境1可以拥有不同质量；同一环境重置后继续用自己的那份质量。重新创建环境对象时会重新采样。PhysX写入这些模型属性会使用CPU侧数据，也适合放在初始化时执行。

### 5.5 惯量怎样对应，还有什么差异

质量影响平动，转动惯量影响转动响应。Lab质量项中的`recompute_inertia=True`，当前本地实现具体计算为：

\[
\mathbf I_e=\mathbf I_0\frac{m_e}{m_0}.
\]

即按质量比例缩放原惯量，假设物体形状与相对质量分布保持一致；不是根据新增载荷的安装位置重新构造一个载荷模型。随后质心项只修改COM位姿中的位置部分，保持COM朝向，不再重算惯量。

旧Gym提交时也要求重算惯量，但仅凭调用端`recomputeInertia=True`，还不能证明其内部重算算法与上述比例缩放完全相同。本步对应了采样范围、修改对象和一次性时机，惯量构造的严格等价仍未验证。

旧代码还把`[质量增量, 质心偏移x, y, z]`存入`mass_params_tensor:(N,4)`供特权观测使用。当前只迁移它们对物理模型的作用；特权观测分组继续作为00表中的待迁移项，未把这些参数加入当前policy观测。

### 5.6 已有检查与真实运行边界

已定向核对本地PhysX实现、官方velocity示例、Gym采样/调用点以及当前USD的基座质量和质心。抽取原生两个Event类、质量随机化辅助函数及均匀采样函数，使用256环境、9身体的CPU场景/资产写入替身检查：

- 配置参数匹配原生签名，临时完整配置副本AST和项目Ruff检查通过。
- 只修改选中的基座，其他身体保持原值；基座索引故意设为3，未依赖body 0。
- 质量/质心偏移落在配置范围内，惯量按质量比缩放，COM朝向不变。
- 部分环境更新不影响其余环境；再次调用仍从默认值取样，不累加先前偏移。

上述方案检查证据：`/tmp/tita_events_mass_com_20261019_7u_hokh0/result.json`及同目录`inspection.py`、`env_cfg_proposed.py`。本轮当前保存文件的复核见`/tmp/tita_push_events_20261019_9ju291ua/result.json`，确认四个Event及两个startup的func/mode/实体/范围/惯量开关均正确。`/tmp`可能被清理；这里保留带日期的检查结论。助手没有修改TITA_Lab中的源码。

这不是物理后端运行检查。后续真实小环境验证应读取启动后的质量/质心，再重置确认它们保持不变，同时核对关节/根状态确实按reset项重置。尚未导入真实任务、运行仿真或训练。

## 6. 运动过程中每15秒扰动一次水平速度

### 6.1 Gym 原来怎么做

有效配置为：

```python
push_robots = True
push_interval_s = 15
max_push_vel_xy = 1
```

`_parse_cfg()`把15 s换成策略步数；当前策略步长0.01 s，对应1500步。`_post_physics_step_callback()`用全局common_step_counter计时，到了间隔，对整批机器人调用`_push_robots()`：

```python
self.root_states[:, 7:9] = torch_rand_float(
    -max_vel, max_vel, (self.num_envs, 2), device=self.device
)
self.gym.set_actor_root_state_tensor(
    self.sim, gymtorch.unwrap_tensor(self.root_states)
)
```

这里把实际世界系水平速度vx、vy各自设成U(-1,1) m/s。它不是给机器人一个新目标，也不是指定持续推力；它直接改变物理状态，模拟一次速度扰动。z方向线速度和三个角速度保留。

整批环境同时触发，但每个环境的vx、vy独立采样。计时不因某台机器人开始新回合而重来。例如全局第10 s重置的机器人，在第15 s仍会和其他机器人一起受到扰动。

### 6.2 为什么需要一个任务函数

本地Lab原生`push_by_setting_velocity()`当前实现含有：

```python
vel_w = asset.data.root_vel_w.torch[env_ids]
vel_w += math_utils.sample_uniform(
    ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device
)
```

它给原速度加随机增量。假如原vx为2 m/s、抽到0.3，旧Gym得到0.3 m/s，这段原生实现得到2.3 m/s。本轮补一个很小的函数，明确使用赋值，保留旧规则。

职责分工是：EventCfg保存何时执行、选择哪个机器人和采样范围；任务MDP函数负责读取速度、覆盖xy、写回仿真；EventManager负责计时和调用。无需再给函数写计时器。

### 6.3 Lab 中怎样写：函数、导出、配置

以下均为待用户写入的方案。

**第一处：** 新建`TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/events.py`：

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def push_by_setting_xy_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Replace world-frame horizontal root COM velocity with random samples."""
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
```

**第二处：** 在`mdp/__init__.pyi`现有`__all__`中追加一个名字，并在自定义相对导入区域加入一行：

```python
# 加到现有__all__列表内。
"push_by_setting_xy_velocity",

# 加到from .observations等导入旁，保留原有内容。
from .events import push_by_setting_xy_velocity
```

**第三处：** 在现有`EventCfg`里加入下面的条目，保留已有四项：

```python
    push_robot = EventTerm(
        func=mdp.push_by_setting_xy_velocity,
        mode="interval",
        interval_range_s=(15.0, 15.0),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": (-1.0, 1.0),
        },
    )
```

两个15表示固定15 s间隔。`is_global_time=True`让整批环境共用一个计时器，回合reset不重置它；这只统一触发时刻，不统一抽到的速度。

### 6.4 函数里每个张量是什么

`asset.data.root_vel_w`是当前根刚体质心的世界系速度，`.torch`得到`(N,6)`张量，每行为`[vx,vy,vz,wx,wy,wz]`。前3列单位m/s，后3列单位rad/s。

全局interval调用会传`env_ids=None`，所以selection取slice(None)，选择全部行。若以后对部分环境调用，selection就是其索引张量；取出的velocity_w为`(M,6)`。clone创建独立副本，暂存修改，随后通过写入API正式提交物理状态。

`sample_uniform(...,(M,2),...)`返回M行水平速度，赋给`[:, :2]`；剩余4列保留。最后`write_root_velocity_to_sim_index()`接收`(M,6)`部分数据以及原env_ids；None代表全部环境。它修改实际运动状态，不改Commands中的目标速度。

### 6.5 检查结果与尚未对齐的时序

临时三个候选文件的AST与项目Ruff检查通过。CPU检查使用真实sample_uniform、抽取的原生EventManager.apply/reset和场景/资产写入替身，确认覆盖赋值、保留vz/角速度、None整批调用、部分环境更新和全局计时器跨reset保留。当前0.01 s步长下，前两次触发在1500/3000步。证据：`/tmp/tita_push_events_20261019_9ju291ua/result.json`及同目录脚本/三个候选文件。

旧Gym在计算终止/奖励/重置前调用推扰；本地Lab的interval Event在本步奖励和自动reset之后、最终观测之前执行。若推扰与结束同一步发生，刚reset环境是否还会受到该次扰动，两边存在差异。当前方案先对应速度规则、15 s间隔与全局同步；严格的调用顺序仍留在覆盖表中，不宣称整段step完全等价。

旧`disturbance=False`是另一条施加随机力的分支，本步不启用。尚未创建真实任务/物理视图或运行仿真、训练；CPU替身检查不能验证PhysX的实际动态响应。

## 7. 对照源码

| 内容 | Gym | Lab |
| --- | --- | --- |
| 关节重置 | [_reset_dofs](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:827) | [reset_joints_by_scale](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:2265) |
| 机身重置 | [_reset_root_states](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:798) | [reset_root_state_uniform](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:2094) |
| 重置流程 | [reset_idx](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:734) | [_reset_idx](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:395) |
| 配置写法 | [init_state](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:43) | [官方velocity重置项](/home/eaziel/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:241)、[EventTermCfg](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/manager_term_cfg.py:291) |
| 质量/质心范围 | [domain_rand](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:123) | 已保存的add_base_mass、base_com |
| 创建时修改/提交body属性 | [_process_rigid_body_props](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:596)、[recomputeInertia调用](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:244) | [质量类/惯量缩放](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:651)、[COM类](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:905) |
| startup配置与触发 | [_create_envs](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:164) | [官方velocity两项](/home/eaziel/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:209)、[startup调用](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:133) |
| COM局部坐标与PhysX写入 | props[0].com | [body_com_pose_b](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:867)、[set_coms_index](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:2498) |
| 定时推扰的速度赋值 | [_push_robots](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1089)、[全局计数触发](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:552) | 本章待写的push_by_setting_xy_velocity、push_robot |
| 现成推扰的差异 | root_states[:,7:9]直接赋值 | [原生push的+=](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py:2067) |
| interval计时/reset与执行位置 | 全局common_step_counter | [全局interval调用](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py:220)、[reset保留全局计时器](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/event_manager.py:136)、[环境执行位置](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:287) |
| 实际根速度读写 | root_states[:,7:13] | [root_vel_w](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:1376)、[PhysX根质心速度写入](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py:697) |

官方在线参考：[startup/reset教程](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/create_manager_base_env.html)、[质量随机化API](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.events.randomize_rigid_body_mass)、[COM随机化API](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.events.randomize_rigid_body_com)。具体参数及计算以本地已核对源码为准。

[返回迁移入口](README.md) · [查看来源与覆盖表](00_迁移映射索引.md)
