# 02 Scene：地形、高度扫描和接触力

更新：2026-10-18。下文的 Scene 配置已写入项目；几何路径和部分张量逻辑做过静态/CPU 检查，传感器尚未在 RL 环境中运行验证。

Scene 负责创建机器人、地面和传感器。**传感器提供数据；观测、奖励和终止函数决定怎样使用数据。** 本章先讲这些数据在旧任务中是什么，再说明 Lab 的对应实现。

## 1. Gym 原来怎么做

### 1.1 创建机器人与地形

旧 `LeggedRobot.create_sim()` 建立仿真和地面，`_create_envs()` 循环创建环境与机器人。机器人文件、默认关节角、地形和随机化参数分散在配置中。

TITA 默认使用 `trimesh`。虽然配置保留了多类地形的比例字段，当前 `utils/terrain.py` 中相应五类分支已经注释；实际执行的是正/负金字塔台阶：

```python
if choice < 0.5:
    step_height *= -1
terrain_utils.pyramid_stairs_terrain(
    terrain, step_width=0.61, step_height=0.5 * step_height,
    platform_size=3.0,
)
```

前面 `step_height=0.05+0.18*difficulty`，所以实际传入高度为 `±(0.025+0.09*difficulty)` m。不能仅按失效注释迁移地形分布。

### 1.2 采样机器人附近的地面高度

旧 `_init_height_points()` 在机身附近建立 17×11=187 个点。`_get_heights()` 将点按机器人航向旋转，再加机身位置，得到要查询的世界系 xy 坐标；随后查询地形高度表，取三个相邻格的最小值作为该点地面高度。

得到的 `measured_heights` 是 `(N,187)`：每个环境187个地面世界坐标 z 值。各元素描述采样点上的地面高度，机器人离地高度由机身z再减去这些地面z得到。

这些值有两种已确认的用途：

- 计算机身相对局部地面的高度，用于高度奖励。例如机身 z 从0.35升到0.45，而地面从0升到0.1，离地高度仍为0.35 m。
- 构造地形观测，供旧默认模型的价值/成本分支使用。旧默认 Actor 读取当前自身观测与历史，并不直接读取这187个高度量。

### 1.3 读取接触求解结果

旧环境取得 Gym 的刚体接触力张量，用轮子世界 z 向接触力 `Fz>1 N` 判断着地，用机身接触判断终止，用腿部接触构造惩罚。

地面高度描述周围几何，接触力描述已经发生的相互作用。前方有台阶时，高度查询可以提前看到它；机器人还没碰到台阶时，不会因此产生接触力。

旧任务另创建轮子六维 force sensor，但当前 LeggedRobot 观测/奖励主要使用 `contact_forces`。不能把六维关节/力传感器与刚体接触力数组视为同一个接口。

## 2. Lab 中怎样写

当前 [env_cfg.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py) 的 Scene 如下，省略原注释：

```python
@configclass
class VelocitySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/height_scanner",
        update_period=0.01,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=[1.6, 1.0],
            ordering="yx",
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[^/]*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
```

这段代码使用当前文件已有的导入。`VelocityEnvCfg` 再通过 `scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=4.0)` 把它交给环境。

`InteractiveScene` 根据配置创建并登记对象，因此无需再写 Gym 的环境/actor 创建循环。`{ENV_REGEX_NS}` 展开到各个环境的路径；`/World/ground` 则是共享地形。Python 字段 `robot`、`height_scanner`、`contact_forces` 是运行时取对象的键。

## 3. 为什么地形这样迁移

`TerrainImporterCfg` 负责将地形放入场景并提供出生原点，`terrain_generator` 决定具体生成什么形状。当前 `ROUGH_TERRAINS_CFG` 是 Lab 通用混合地形，**尚未对应旧 TITA 实际的正/负台阶分布**。

`max_init_terrain_level=5` 是出生难度限制相关配置，不会自动实现训练中的升降级规则。地形生成、初始难度选择、Curriculum 更新是三个相关环节，仍需分别迁移。当前通用 ROUGH 的生成器 curriculum 默认关闭。

物理材质与视觉材质也分开：前者决定摩擦、恢复系数等接触参数，后者决定外观。当前静/动摩擦均为1，恢复系数为0；这些固定值不代表已迁入旧材质随机化。

`restitution=0` 保留旧固定地面值，这也是 Lab 通用材质的默认值。两个接触材质怎样合成，另外由 combine mode 决定；选中 multiply 时相乘，模式冲突时还受 PhysX 优先级规则影响。

本章使用的PhysX材质类为`isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg`。

## 4. 为什么高度扫描改用 RayCaster

### 4.1 用射线求交获得同类地形数据

Gym 查高度表，Lab 当前使用 RayCaster 与地形网格求交。二者都回答“这些 xy 位置上的地面有多高”，因此可以对应到同一项任务功能。

Lab 的计算过程是：

```text
在机身附近排列187条向下射线
  → 按机身位置与航向变换射线起点
  → 与 /World/ground 网格求交
  → 返回每条射线命中点的世界坐标
  → MDP 取命中点 z，生成任务所需特征
```

无需自己编写三角形求交算法。它仍是虚拟的几何查询，不是向地面施力，也不意味着真机上存在同样的传感器。

### 4.2 参数对应什么过程

| 参数 | 当前含义 |
| --- | --- |
| `prim_path` | 在真实机身刚体 `Geometry/base_link` 下创建扫描坐标节点 |
| 路径末尾 `/height_scanner` | 新建的普通 Xform 名称，无质量、无碰撞；默认 SensorFrameCfg 负责创建 |
| `offset.pos=(0,0,20)` | 射线起点抬高20 m，便于从地形上方向下求交；不抬高机器人 |
| `ray_alignment="yaw"` | 网格跟随位置与航向，忽略滚转/俯仰造成的扫描平面倾斜 |
| `size=[1.6,1.0]`、`resolution=0.1` | x 从-0.8到0.8、y 从-0.5到0.5，含端点，共17×11条射线 |
| `ordering="yx"` | 展平后 y 变化更快，对应旧 ij 网格的点序 |
| `mesh_prim_paths` | 指定被查询的地形网格 |
| `update_period=0.01` | 10 ms 仿真时间更新周期，与当前策略周期一致 |
| `debug_vis=False` | 不显示调试交点，计算仍保留 |
| `global_world_only=True` | Newton 使用的字段，当前 PhysX 忽略 |

扫描节点可以是新建的子节点，但它的机身祖先必须存在。当前 PhysX 实现通过刚体祖先跟踪位姿，因此要按 TITA 的实际 USD 层级填写路径。

### 4.3 射线命中点与最终高度观测不是同一个量

下面是已初始化环境中的读取方式，不是要直接粘进配置类的语句：

```python
scanner = env.scene["height_scanner"]
hit_positions_w = scanner.data.ray_hits_w.torch  # 预期 (N,187,3)
terrain_z = hit_positions_w[..., 2]             # (N,187)，单位 m
```

`ray_hits_w` 由运行时传感器的数据类提供，`_w` 表示世界坐标，`.torch` 取得供 PyTorch 运算的视图。平地 z=0 时，即使射线起点在20 m以上，命中点 z 仍为0；它不是射线长度。

旧高度观测还要计算：

```python
heights = torch.clip(
    self.root_states[:, 2].unsqueeze(1) - 0.4 - self.measured_heights,
    -1.0, 1.0,
) * self.obs_scales.height_measurements
```

即 `机身z - 地面z - 0.4`，再裁剪并乘5。这属于后续 Observations/MDP 的职责。Lab 的 `mdp.height_scan()` 默认偏置为0.5，也不自动带齐上述处理，不能直接默认调用就认为计算相同。

旧高度表取邻格最小值，当前射线取三角网格交点，两者在台阶边缘可能不同。未命中射线会返回 inf，真实运行时需要检查命中情况。

## 5. 为什么接触力使用 ContactSensor

### 5.1 先有物理接触求解，再有传感器报告

PhysX 根据几何、运动状态、摩擦等参数求解接触。`activate_contact_sensors=True` 给机器人刚体开启接触报告；`ContactSensorCfg` 再指定读取范围和缓冲方式。传感器本身不额外制造接触力。

当前 PhysX 接触传感器会递归筛选所选子树中启用报告的刚体，并建立逐刚体 view，因此可以处理 TITA 的嵌套刚体。实际绑定数量和顺序仍待运行确认。

配置与读取的关系是：

```python
# Scene 中声明：
contact_forces = ContactSensorCfg(...)

# 已初始化环境中读取：
contact = env.scene["contact_forces"]
forces_w = contact.data.net_normal_forces_w.torch
```

`contact_forces` 是你起的场景键名；`net_normal_forces_w` 是 Lab 数据类已有的属性，所以你不需要在 cfg 中自己声明这个名字。

### 5.2 “聚合法向接触力”是什么

同一个刚体可能同时有多个接触点。把这些接触点的法向力转到世界坐标后做向量相加，就是该刚体的 `net_normal_forces_w`：

```text
net：同一刚体上各接触点的向量和
normal：每个接触面的法向分量
forces：力，单位 N
_w：用世界坐标表示
```

例如同一个轮子从地面两个接触点分别受到 `(0,0,60)`、`(0,0,40)` N，又从墙面受到 `(-30,0,0)` N，聚合结果就是 `(-30,0,100)` N。墙的法向可以沿水平轴，所以“法向力”仍有 xyz 三个分量。

聚合发生在同一身体、同一采样时刻；不把整台机器人合成一个力，也不对历史求和。向量可能互相抵消，聚合后也无法还原每个接触点的位置和受力。

它只报告法向分量，不包含轮地切向摩擦。这不表示 PhysX 没有求解摩擦。旧兼容属性 `net_forces_w` 在当前后端也返回法向量并给出兼容警告，不能按名字理解成完整接触反力。

### 5.3 形状、历史与判据

按当前 USD 的9个刚体静态推导：

| 数据属性，通过 `.torch` 读取 | 预期形状 | 含义 |
| --- | --- | --- |
| `net_normal_forces_w` | `(N,9,3)` | 各身体当前的法向接触合力 |
| `net_normal_forces_w_history` | `(N,3,9,3)` | 保存3次力采样 |
| `current_air_time` / `current_contact_time` | `(N,9)` | 连续腾空/接触时长，单位s |

运行时按传感器的身体名字选择；本地SceneEntityCfg优先通过`find_sensors()`解析名字与索引。当前未配置接触对象过滤，因此不是专门的“轮子—地面”配对数据，也不是六维力/力矩 wrench。

接触传感器默认 `update_period=0`，按物理步调度；正历史长度保留物理步采样。这里3帧力历史，不是旧任务100帧接触状态缓冲。底层读取接触报告使用2.5 ms物理步长。

`track_air_time=True` 开启力阈值判定与计时。当前 PhysX 默认按合力范数超过1 N判断接触；旧 TITA 按轮子 `Fz>1 N`，再与上一策略帧状态取 OR，并按策略周期计时。侧墙接触时范数可以非零而Fz为0，因此这两种判据及时间窗口需要另行对齐。

## 6. 当前迁移到哪里

当前已配置机器人、固定材质、地形导入、187点扫描、全身接触报告与灯光。此前做过 USD 路径/刚体数量静态检查，以及实际网格生成函数的CPU点数与顺序检查；没有验证传感器成功初始化或真实读数。

尚未完成：旧台阶地形与课程、策略高度任务特征及旧接触计时。当前高度/小腿碰撞奖励已在08章接入并静态/局部CPU核对，尚未运行；碰撞读取入口已统一为明确的法向力属性。机身当前接触力终止已在[07章](07_terminations迁移.md)接入，尚未运行。Lab 已有 `height_scan`、`undesired_contacts`、`illegal_contact` 等函数，但其默认偏置、力范数和历史最大值窗口必须与旧公式分别核对。

数据接入顺序是：Scene创建传感器，MDP函数读取或处理数据，ObservationsCfg、RewardsCfg、TerminationsCfg选择相应函数。本章不讨论用户暂缓的真机接触力来源。

## 7. 对照源码

| 要查什么 | Gym | Lab |
| --- | --- | --- |
| 创建场景与地形 | [create_sim](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:843)、[实际台阶](/home/eaziel/Downloads/tita_rl-master/utils/terrain.py:78) | [官方 Scene 例子](/home/eaziel/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:81)、[InteractiveSceneCfg](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py:22) |
| 高度点与查询 | [_init_height_points / _get_heights](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1035) | [RayCasterCfg](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_cfg.py:25)、[ray_hits_w](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_data.py:58)、[新建扫描节点](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/spawners/sensors/sensors.py:204) |
| 高度的用途 | [高度观测](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:379)、[高度奖励](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1165)、[模型取数](/home/eaziel/Downloads/tita_rl-master/modules/actor_critic.py:764) | [height_scan](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py:281) |
| 接触绑定与输出 | [接触状态计算](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:436) | [ContactSensorCfg](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py:19)、[PhysX 绑定](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py:351)、[输出数据类](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor_data.py:74) |
| 接触奖励/终止 | [奖励函数起点](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1153)、[check_termination](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:699) | [undesired_contacts](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:293)、[illegal_contact](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py:174) |

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> 为啥有这个？？不应该吧

**A：** 你问的是路径末尾的`/height_scanner`。它是传感器创建的普通坐标节点，源USD只需存在真实机身父节点`Geometry/base_link`。当前SensorFrameCfg负责创建这个节点；PhysX传感器沿祖先找到机身刚体并跟踪它的位姿。第4.2节的参数表给出了完整路径含义。

**Q：**

> 你说的这个又是啥？我代码里没有啊？但你也没说修改。

**A：** 你问的是`net_normal_forces_w`。配置中的`contact_forces`负责创建接触传感器；创建后的数据对象由Lab提供这个属性：

```python
sensor = env.scene["contact_forces"]
forces_w = sensor.data.net_normal_forces_w.torch
```

这段是环境运行时的读取方式。该属性已经定义在Lab的数据类里；任务配置只需选传感器与参数。

**Q：**

> 我还想问，聚合法向接触力是啥？

**A：** 同一个身体、同一时刻，各接触点的法向力转成世界坐标后做向量相加。身体轴仍保留，每个身体各有一个xyz向量。第5.2节有地面60 N、40 N加墙面30 N的例子；这个接口报告法向分量，切向摩擦由物理引擎另行求解。

**Q：**

> 我只是想让pylance认识。以及想知道为啥这里没给地面设恢复系数

**A：** 当前代码已经直接导入并构造`PhysxRigidBodyMaterialCfg`，也显式写了`restitution=0.0`：

```python
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
```

这个值对应旧固定地面恢复系数，也是当前基类默认值。早期只导入旧名、构造处仍用sim_utils前缀，会留下未使用导入；编辑器对另一条表达式的类型推断也不会因此改变。

**Q：**

> 你说的“我查了你本地源码：旧的 `RigidBodyMaterialCfg` ..”意思是，lab开发者开发了新的接口，但仍然保留了旧的？

**A：** 对。当前接口拆为求解器公共的RigidBodyMaterialBaseCfg与PhysX专用的PhysxRigidBodyMaterialCfg，旧RigidBodyMaterialCfg作为弃用的兼容接口保留。当前项目按本地版本直接使用PhysX专用类；旧导入警告属于历史问题。

**Q：**

> 小腿接触力是啥玩意？？靠，看来我压根没理解当时接触力sensor的创建和原理！

**A：** 这里的小腿接触力，专指其他碰撞体直接接触小腿碰撞几何时，作用到这段小腿刚体上的接触反力。PhysX按碰撞几何、姿态和相对运动求解接触约束；接触反力参与刚体动力学，ContactSensor再读取它的报告并按身体整理。它是一个仿真软件数据接口。

先看受力归到哪个身体：

| 物理情况 | 接触报告归属 |
| --- | --- |
| 轮子压在地面上 | 地面对轮子的支持力记在轮子身体上 |
| 小腿悬空，通过关节承载 | 载荷通过关节约束传递；小腿本体未直接接触别的碰撞体时，其接触读数可以为0 |
| 小腿撞到台阶侧面 | 台阶对小腿的接触反力记在该小腿身体上 |
| 小腿碰到另一条腿，且该碰撞对启用 | 小腿可能有自碰撞接触报告 |

因此轮子接地带来的关节支撑载荷，不会自动复制成小腿的接触力。ContactSensor读的是接触这部分受力；关节传力、重力、驱动力等在动力学中各有来源。

你当前的创建代码是：

```python
robot.spawn.activate_contact_sensors = True
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/[^/]*",
    history_length=3,
    track_air_time=True,
)
```

第一行在生成机器人时启用接触报告。InteractiveScene再依据ContactSensorCfg创建运行时传感器对象，登记为scene.sensors["contact_forces"]。本地PhysX初始化从Robot资产子树递归收集名称匹配、启用报告的刚体，因此适用于当前嵌套身体层级。history_length=3保留三次力采样；track_air_time额外累计腾空/接触时长。本次碰撞计数只读当前力，没有用这些历史或计时量。

读取链条为：

```python
contact_sensor = env.scene.sensors["contact_forces"]
forces_w = contact_sensor.data.net_normal_forces_w.torch
```

forces_w形状为(N,B,3)：环境数、传感器身体数、世界系xyz力分量，单位N。按当前USD9个刚体推导，全部绑定时预期(N,9,3)；实际顺序与数量须在环境初始化后核实。每个身体各有自己的向量，身体之间不会被合成一个数。

这里normal表示接触面的法向。例如地面法向大致向上，台阶竖直侧面的法向大致水平。一个身体的多个接触点法向力做世界系向量和，得到net_normal_forces_w；它不是只取Fz，也不是把力的大小相加，方向相反时可以抵消。切向摩擦由物理引擎另外求解，这个属性只读法向分量。

奖励中的body_names=["left_leg_3","right_leg_3"]从这份按身体分开的数据中选择两条腿，得到(N,2,3)，再对xyz求模、比较0.1 N并计数。例：左腿(0,0,20) N、右腿(0,0,0) N，得到[True,False]→[1.0,0.0]→1。轮子没有被选择，正常轮地接触不会计入这项。

当前未过滤接触对象，所以这个力可能来自地形、其他物体或允许发生的自碰撞；选择小腿不等于限定“小腿—地面”这一对。当前保存奖励函数已经使用上面的明确法向力属性。本地net_forces_w兼容getter会发警告再回退到该属性；当前直接读取避免了这层兼容行为。源码依据：[PhysX传感器绑定](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py:351)、[读取报告](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py:547)、[兼容getter](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/base_contact_sensor_data.py:196)。当前说明与CPU检查不代表传感器已经运行。

**Q：**

> 意思就是小腿link碰到东西了？

**A：** 对，就是小腿link的碰撞几何直接接触到了其他碰撞体。

[返回迁移入口](README.md) · [查看来源与覆盖表](00_迁移映射索引.md)
