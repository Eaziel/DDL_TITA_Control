# 01 Physics：物理步长、策略周期和 PhysX 参数

**目标与结果：** 确定一次策略动作对应多少物理时间，并把旧求解参数放到Lab正确的配置对象。

整理：2026-10-26。物理后端与时间配置已写；已有200策略步用户环境日志。此章解释控制/物理周期与求解参数，不把短运行当作求解数值或训练效果验证。 当前证据与整体状态以[项目记忆](../../PROJECT_MEMORY.md)为准。

## 1. Gym 原来怎么做

### 1.1 配置给出物理步长和控制间隔

旧 `configs/legged_robot_config.py` 中的配置是：

```python
class sim:
    dt = 0.0025
    substeps = 1
    gravity = [0.0, 0.0, -9.81]
    up_axis = 1  # z 轴朝上

    class physx:
        num_threads = 10
        solver_type = 1
        num_position_iterations = 4
        num_velocity_iterations = 0
        contact_offset = 0.01
        rest_offset = 0.0
        bounce_threshold_velocity = 0.5
        max_depenetration_velocity = 1.0
        max_gpu_contact_pairs = 2**23
        default_buffer_size_multiplier = 5
        contact_collection = 2
```

TITA 在 `configs/tita_constraint_config.py` 中另外设置：

```python
class control(LeggedRobotCfg.control):
    # 省略其他控制参数。
    decimation = 4
```

`dt` 和 `decimation` 来自两个配置块，却共同决定策略与物理仿真的时间关系。

### 1.2 step 实际执行四次物理步

旧 `envs/legged_robot.py` 的关键循环是：

```python
for _ in range(self.cfg.control.decimation):
    self.torques = self._compute_torques(self.actions).view(self.torques.shape)
    self.gym.set_dof_actuation_force_tensor(
        self.sim, gymtorch.unwrap_tensor(self.torques)
    )
    self.gym.simulate(self.sim)
    self.gym.fetch_results(self.sim, True)
    self.gym.refresh_dof_state_tensor(self.sim)
self.post_physics_step()
```

策略输出一次动作，`env.step()` 就执行四次 `gym.simulate()`。每次前进 0.0025 s，四次合计 0.01 s，然后计算终止、奖励和下一份观测。

```text
策略给出 a_t
  → 算力矩，物理前进 2.5 ms
  → 算力矩，物理前进 2.5 ms
  → 算力矩，物理前进 2.5 ms
  → 算力矩，物理前进 2.5 ms
  → 返回新观测，策略再给出 a_(t+1)
```

**四步保持同一个策略动作，但力矩可以变化。** 控制器每个物理步都读取更新后的关节角和角速度，再算一次反馈力矩。

### 1.3 substeps 在 simulate 内部细分时间

`substeps` 属于物理引擎内部。设物理步长为 `dt`、内部子步数为 `S`、decimation 为 `D`：

| 时间层次 | 计算方法 | 当前 TITA |
| --- | --- | --- |
| 引擎内部积分子步 | `dt / S` | 2.5 ms |
| 一次 `gym.simulate()` | `dt` | 2.5 ms |
| 一次策略动作对应的环境步 | `D * dt` | 10 ms |

当前 `S=1`，内部没有进一步细分。假如只把 `substeps` 改为 2，一次 simulate 会走两个 1.25 ms 子步，总共仍前进 2.5 ms；策略周期仍为 10 ms。

内部子步不会自动让 Python 中的 `_compute_torques()` 多执行一次。位置/速度求解迭代次数也不等于子步数：迭代是在一个时间步内求解约束，不是再次推进时间。

时间关系可统一写成：

$$
\Delta t_{sub}=\frac{\Delta t_{sim}}{S},\qquad
\Delta t_{policy}=D\Delta t_{sim},\qquad
f_{sim}=\frac1{\Delta t_{sim}},\quad f_{policy}=\frac1{\Delta t_{policy}}.
$$

当前 $S=1,D=4$，所以物理调用频率400Hz，策略频率100Hz；求解迭代次数不出现在这些时间换算中。

## 2. Lab 中怎样写

### 2.1 环境配置负责时间和后端

在当前 [env_cfg.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/config/Tita/env_cfg.py) 中，物理预设为：

```python
@configclass
class VelocityPhysicsCfg(PresetCfg):
    isaacsim_physx: PhysxCfg = PhysxCfg(
        solver_type=1,
        bounce_threshold_velocity=0.5,
        gpu_max_rigid_contact_count=2**23,
    )
    default: PhysxCfg = isaacsim_physx
```

同一文件的 `VelocityEnvCfg.__post_init__()` 中已有：

```python
self.decimation = 4
self.sim.dt = 0.0025
self.sim.gravity = (0.0, 0.0, -9.81)
self.sim.render_interval = self.decimation
self.sim.physics = VelocityPhysicsCfg()
```

这两段是现有配置的节选，其他环境字段仍保留。这里使用 PhysX；生成器最初提供的 Newton 预设没有继续作为默认后端。

Lab 中：

```text
physics_dt = cfg.sim.dt                  = 0.0025 s
step_dt    = cfg.sim.dt * cfg.decimation = 0.01 s
```

当前 PhysX 路径中，`ManagerBasedRLEnv.step()` 先处理一次策略动作，再循环四次应用动作、写入仿真、推进物理、更新场景。这部分循环复用 Lab，不需要把旧 Gym 循环复制过来。

`render_interval=4` 另外指定每四个物理步渲染一次。渲染间隔与策略间隔可以不同；这里让它们取同一个数值。

### 2.2 机器人属性放在资产配置

旧 Gym 将很多参数集中在 `sim.physx` 中。Lab 中有一部分属于机器人及碰撞体，当前已放进 [assets/tita.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/assets/tita.py) 的 `UsdFileCfg`：

```python
articulation_props={
    "/.*": [
        PhysxArticulationCfg(
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        )
    ]
},
rigid_props={
    "/.*": [PhysxRigidBodyCfg(max_depenetration_velocity=1.0)]
},
collision_props={
    "/.*": [PhysxCollisionCfg(contact_offset=0.01, rest_offset=0.0)]
},
```

这三个字典是 `UsdFileCfg(...)` 的参数片段。`"/.*"` 用于选择生成根节点下的后代，再由相应属性写入器筛选 articulation、刚体或碰撞体。TITA 的这些对象位于 USD 子层级，因此要明确选择后代路径。

## 3. 为什么这样迁移

首先保留时间关系：旧 TITA 的 `substeps=1`，所以 Lab 直接使用相同的 `dt=0.0025` 和 `decimation=4`，就能保留 2.5 ms 物理步与 10 ms 策略周期。

这里沿用旧TITA的control.decimation=4。配合dt=0.0025 s，保留策略周期0.01 s；引擎内部substeps=1，控制器在每个物理步重新计算力矩。

其次按参数作用对象拆分，而不是照抄旧类的层级：

| Gym 参数 | 当前 Lab 落点 | 原因 |
| --- | --- | --- |
| `dt`、`gravity` | `env_cfg.sim` | 整个仿真的时间与重力 |
| `control.decimation` | `env_cfg.decimation` | 环境每次动作要走几个物理步 |
| `solver_type=1` | `PhysxCfg` | PhysX 求解器选择，1 为 TGS |
| `bounce_threshold_velocity` | `PhysxCfg` | 接触反弹的速度阈值 |
| 位置/速度求解迭代次数 | `PhysxArticulationCfg` | 机器人的 articulation 属性 |
| `max_depenetration_velocity` | `PhysxRigidBodyCfg` | 刚体穿透修正速度上限 |
| `contact_offset`、`rest_offset` | `PhysxCollisionCfg` | 碰撞体的接触生成与静止偏移 |
| GPU 接触容量 | 当前设 `gpu_max_rigid_contact_count=2**23` | 容量配置；旧 contact pairs 与新 contact count 不能只凭同一数值认定完全等价 |

旧 `num_threads`、`default_buffer_size_multiplier`、`contact_collection` 尚未逐项建立等价配置，仍需结合 Lab 当前接口与传感器采样方式核对。没有同名字段，不意味着可以不检查它原来影响什么。

如果将来迁移 `substeps>1` 的工程，可以先按 `dt_Lab=dt_Gym/S`、`D_Lab=D_Gym*S` 匹配时间尺度；但控制器、延迟和传感器的调用次数也会受影响，不能仅靠这两个公式宣布执行过程等价。本项目当前不需要这一步。

## 资产参数怎样对应

旧asset选项还包括刚体阻尼和速度上限。当前已显式补零阻尼、关节腿25/轮20 rad/s，以及刚体线/角速度上限。USD中的Newton速度属性不能自动成为PhysX约束，必须设置对应后端参数；遗漏的发现经过见[09 F2/F3](09_阶段审查.md)。

旧max_gpu_contact_pairs和新gpu_max_rigid_contact_count不是同名同单位承诺；当前复用2**23作为容量配置，不据数值相同宣称缓冲行为等价。线程、默认buffer倍数和接触采样开关没有逐字段硬套。

## 4. 当前结果与验证边界

- 当前文件已经配置 PhysX、重力、时间参数及上述资产物理属性。
- 静态推导对应物理 400 Hz、策略 100 Hz。这是仿真时间频率，不是程序运行速度或屏幕 FPS。
- 此前 USD 静态检查匹配到 1 个 articulation root、9 个刚体和 9 个碰撞体；尚未验证真实运行中的求解与接触表现。
- 当前回合时长已按[07 Terminations](07_terminations迁移.md)设置为20 s；本章的2.5 ms/10 ms描述步进周期，回合时长描述一次任务最长持续多久。

## 5. 对照源码

| 要查什么 | Gym | Lab |
| --- | --- | --- |
| 时间和物理参数 | [sim 配置](/home/eaziel/Downloads/tita_rl-master/configs/legged_robot_config.py:218)、[TITA control](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:63) | [SimulationCfg](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/sim/simulation_cfg.py)、[PhysxCfg](/home/eaziel/IsaacLab-main/source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py) |
| 时间循环 | [step](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:295)、[_parse_cfg](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:993) | [physics_dt / step_dt](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/manager_based_env.py)、[step](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/manager_based_rl_env.py) |
| 机器人属性 | [创建环境与资产](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:164) | [PhysX schemas 配置](/home/eaziel/IsaacLab-main/source/isaaclab_physx/isaaclab_physx/sim/schemas/schemas_cfg.py) |

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> 为啥deci=4？？咋算的？？gym的substep是啥？？

**A：** 4直接来自旧TITA的control配置。已知目标策略周期为0.01 s时，可以用0.01/0.0025=4反算；只知道物理步长，无法单独决定decimation。源码未说明作者为何选择100 Hz。

Gym的substeps在一次simulate内部再细分积分时间。若dt=0.0025、substeps=2，一次simulate仍前进2.5 ms，内部走两个1.25 ms子步。Python的_compute_torques只在外层循环调用，内部子步不会额外调用它。第1.3节列出了这三层时间关系。

[返回迁移入口](README.md) · [查看来源与覆盖表](00_迁移映射索引.md)
