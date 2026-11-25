# tita.py：机器人资产、默认状态和执行器

**阅读目标：** 读懂机器人资产配置如何指定USD、默认关节状态、刚体/碰撞属性和名义执行器；任务中的随机PD由Scene另行选择。

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/assets/`。
源码文件：[tita.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/assets/tita.py)。

更新：2026-11-08；本轮只核对源码与文档，未重开USD或仿真。资产IdealPD与Scene的TitaPD分别维护。用户在源USD关闭四处额外碰撞并报告恢复自碰撞后正常；当前源码已显式enabled_self_collisions=True。覆盖已修正，当前保留现有形状；精确box/cylinder仅列为可选对照。助手未修改用户源码/资产或运行模拟器。

## 1. 当前代码

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import (
    ArticulationCfg,  # BaseArticulation：与其他后端（如Newton Mjwarp）共用的基类
)
from isaaclab_physx.sim.schemas import (
    PhysxArticulationCfg,
    PhysxCollisionCfg,
    PhysxRigidBodyCfg,
)

from . import TITA_LAB_ASSETS_DIR  # . 表示当前的 Tita_Lab.assets 包。这个路径常量已在它的 __init__.py 中定义

TITA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(TITA_LAB_ASSETS_DIR / "robots" / "tita" / "usd" / "tita.usd"),
        articulation_props={
            "/.*": [
                PhysxArticulationCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                )
            ]
        },
        rigid_props={
            "/.*": [
                PhysxRigidBodyCfg(
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,  # [m/s]
                    # 旧Gym使用rad/s，此处PhysX配置使用deg/s。
                    max_angular_velocity=math.degrees(1000.0),
                    max_depenetration_velocity=1.0,
                )
            ]
        },
        collision_props={"/.*": [PhysxCollisionCfg(contact_offset=0.01, rest_offset=0.0)]},
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_.*_leg_1": 0.0,
            "joint_.*_leg_2": 0.8,
            "joint_.*_leg_3": -1.5,
            "joint_.*_leg_4": 0.0,
        },
        pos=(0.0, 0.0, 0.45),
    ),
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=["joint_.*_leg_[123]"],
            stiffness=40.0,
            damping=1.0,
            actuator_effort_limit=60.0,
            joint_effort_limit=60.0,
            joint_velocity_limit=25.0,
        ),
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=["joint_.*_leg_4"],
            stiffness=0.0,
            damping=0.5,
            actuator_effort_limit=15.0,
            joint_effort_limit=15.0,
            joint_velocity_limit=20.0,
        ),
    },
)
```

## 2. 配置对象按什么顺序组成

| 部分 | 创建的配置与作用 |
| --- | --- |
| usd_path | 从assets包的目录常量拼接当前机器人USD路径 |
| spawn | UsdFileCfg描述怎样加载模型、写入物理属性 |
| articulation_props / rigid_props / collision_props | 分别选择articulation、刚体、碰撞体并写入对应参数 |
| init_state | 保存默认位置和默认关节角，作为创建、重置、动作偏置与相对观测的参考 |
| actuators | 将六个腿关节和两轮分别交给两组IdealPD执行器 |

模块导入时构造TITA_CFG配置对象。Scene通过TITA_CFG.replace(prim_path=...)取得指定生成路径的配置，之后由InteractiveScene创建运行时机器人。

## 3. 执行器怎样连接Actions

IdealPD根据目标与实际关节状态计算力矩：

$
\tau=K_p(q_{\mathrm{des}}-q)
+K_d(\dot q_{\mathrm{des}}-\dot q)+\tau_{\mathrm{ff}}.
$

q是关节角，单位rad；q_dot是角速度，单位rad/s；tau_ff是动作执行模块传来的前馈力矩，单位N·m。Kp将角度误差换算为力矩，Kd将速度误差换算为力矩。默认腿增益40/1，轮子增益0/.5：轮子不追踪旋转角度，但仍有速度阻尼与前馈力矩。

init_state.pos目前是(0,0,.45)m，关节默认角仍为0/.8/−1.5/0。它是资产的根位置参考；RL重置还会组合环境地形原点及reset事件的偏移，不能仅凭这个z值推断所有采样姿态都不会穿透。

actuator_effort_limit限制模型输出的总力矩，joint_effort_limit配置求解器侧上限。速度任务会在Scene覆盖为TitaPDActuator并接上startup因子；滤波、物理步延迟和startup随机因子均已接入。参数映射与完整公式见[03 Actions](../../../../../docs/migration/03_actions迁移.md)。

## 4. 相关源码与验证

[01 Physics](../../../../../docs/migration/01_physics.md)解释物理属性归属，[02 Scene](../../../../../docs/migration/02_scene迁移.md)解释资产怎样进入场景，[USD检查快照](../../../../../docs/USD_CHECK_2026-10-15.md)保留当时的层级与关节检查证据。用户报告导入脚本可运行；助手未在本次文档整理中运行仿真。

## 5. 四处额外碰撞已关闭，精确几何保留为可选对照

旧URDF中左右leg_1/leg_2没有有效collision。用户已在源USD关闭这四个Mesh的Collision Enabled，并报告重新开启自碰撞后运行正常。本次静态检查确认四处False、其余五处True，见[修正后状态](../../../../../docs/audits/2026-10-23_collision_model/after_user_usd_edit.json)。无需再添加此前建议的Python关闭规则。

本轮已核实PhysxArticulationCfg片段明确写为：

```python
PhysxArticulationCfg(
    enabled_self_collisions=True,
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=0,
)
```

用户随后问“我不加行吗？”，可以。本阶段保留现有机身/小腿/轮子碰撞形状，继续地形迁移。旧box/cylinder尺寸与可选Sim操作仍见[19](../../../../../docs/migration/19_碰撞模型对齐与诊断清理.md)，不作为开训前置条件；若以后需要更严格的旧任务复现或出现与接触形状相关的问题，再单独对照。当前模型的形状差异继续保留记录。

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> 这是啥意思？你看我tita.py好着没？

**A：** 你问的是属性字典中的"/.*"以及此前说明中的writer。这个后缀与机器人生成路径拼接，选择其后代；属性写入函数再按已有ArticulationRootAPI、RigidBodyAPI、CollisionAPI选择对应对象。当前资产的这些物理对象位于USD子层级，因此此显式路径写法可覆盖它们。

属性写入发生在加载后的场景；writer就是Lab内部执行这些赋值的函数。旧静态检查匹配到1个articulation root、9个刚体和9个碰撞体，当前三组字典的值和目标路径已核对。完整运行效果仍待仿真验证。

本地实现也允许适用的裸配置通过默认后缀递归选择；当前显式字典写法有效。源码：[from_files目标选择](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py)。

本次[09审查](../../../../../docs/migration/09_阶段审查.md)确认：当前PhysX关节速度上限和刚体阻尼/速度参数需要补齐；已禁用root_joint，未把Newton自碰撞属性误当PhysX配置。

**Q：** 我还是得按照tita gym的身体box近似来？咋搞啊，我总感觉box近似会影响训练效果，但好像没啥影响，都是防止碰撞。

**A：** 迁移阶段建议先对齐旧任务的碰撞定义，包括哪些link有碰撞、形状尺寸和局部位置，而不是统一把所有零件改成box。碰撞形状负责生成接触力，接触点和法线会影响摩擦、滚动与机身转矩，也决定何时触发机身接触终止；它不只是防穿模。适合任务的简单形状可以形成稳定、可验证的模型，更多视觉细节不保证训练更好。旧任务本身也有意省略了四个链接的碰撞。先建立匹配旧模型的基线，再单独评估更真实几何；box碰撞可以与单独指定的质量、质心、惯量共存。

**Q：** 我直接去sim里，关usd的就行呗？就是删除leg1和2的，collision api？这四个都删掉？

**A：** 可以直接编辑实际加载的源USD：`Tita_Lab/src/Tita_Lab/assets/data/robots/tita/usd/tita.usd`。四个目标是左右leg_1、左右leg_2各自的碰撞Mesh，不是截图中的四种API。选中对应Mesh，在`PhysicsCollisionAPI`中取消`Collision Enabled`（USD属性`physics:collisionEnabled`），保存源USD即可；这保留视觉与刚体，只关闭碰撞，之后也能重新开启。截图中的`PhysxCollisionAPI`提供PhysX接触参数，`PhysxConvexHullCollisionAPI`与`PhysicsMeshCollisionAPI`配置碰撞近似，`MaterialBindingAPI`绑定材质，不需要一起删除。控制启用的`PhysicsCollisionAPI`可能在截图上方。字段依据为本地Isaac Lab `sim/schemas/schemas_cfg.py`中的`UsdPhysicsCollisionCfg.collision_enabled`。源USD修改可替代上面的Python关闭规则；保存后恢复`enabled_self_collisions=True`并重启原诊断运行。仅在运行场景的`/World/envs/...`临时修改，不代表已保存源资产。当前仍待用户执行与复跑。

## 历史解释原文

<details>
<summary>展开此前资产解释与快照（含已被替换的配置，保留原文）</summary>

# tita.py 代码解释与检查

解释的文件目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/assets/`。

解释的文件：[tita.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/assets/tita.py)。

更新日期：2026-10-15。以下是本次只读检查时的源码片段；未替用户修改源码，未启动仿真或验证模块实际导入。源码修改后应更新对应片段及状态。

同日后续状态：用户已报告 `scripts/import_assets.py` 可以正常运行、关节正常；当前脚本已实际引用本文件的 `TITA_CFG`。下文第 4 节“没有实际导入模块或启动仿真”描述的是此前助手检查的边界；不能据此否定用户后续运行结果。RL 环境仍未接入该配置，助手本次未复跑。

## 1. 模块结构与配置对象

### 源码

```python
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import (
    ArticulationCfg,  # BaseArticulation：与其他后端（如Newton Mjwarp）共用的基类
)

TITA_CFG = ArticulationCfg(
```

### 解释

`tita.py` 是现有 `TITA_Lab.assets` 包中的一个模块，`TITA_CFG` 是模块级变量，保存一个 `ArticulationCfg` 实例。这里存机器人定义，没有创建场景或启动仿真，职责划分正确。后续环境配置可从 `TITA_Lab.assets.tita` 引用此对象。

`ArticulationCfg` 旁边的 `BaseArticulation` 注释是从原脚本带来的旧注释，与当前导入对象不对应，建议删除或改成配置类的说明。这是说明准确性问题，不影响 Python 语法。

## 2. USD 路径

### 源码

```python
from . import TITA_LAB_ASSETS_DIR  # . 表示当前的 TITA_Lab.assets 包。这个路径常量已在它的 __init__.py 中定义

# 当前 UsdFileCfg 内的路径字段节选：
usd_path=str(TITA_LAB_ASSETS_DIR / "robots" / "tita" / "usd" / "tita.usd"),
```

### 解释

用户已修正为上面的包内路径写法，当前表达式正确。之前的 `TITA_non_RL/assets/tita.usd` 是依赖工作目录的旧路径；中间曾写为 `/data/robots/tita/usd/tita.usd`，其开头 `/` 指 Linux 文件系统根目录，不是包内 assets/data。两种旧写法均已失效。

当前实际复制到 RL 包内的文件是：

```text
/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/assets/data/robots/tita/usd/tita.usd
```

这里有一层用户创建的 `usd/`，应按这个真实目录拼接，不要求改目录名。

现有 `assets/__init__.py` 已定义 `TITA_LAB_ASSETS_DIR`，指向 assets/data。当前实现从所在包导入该常量，使用 Path 的 `/` 运算符拼出 USD 文件位置，最后用 `str(...)` 传给配置：

```python
from . import TITA_LAB_ASSETS_DIR

# 在 UsdFileCfg 中使用：
usd_path=str(TITA_LAB_ASSETS_DIR / "robots" / "tita" / "usd" / "tita.usd")
```

导入语句开头的 `.` 表示当前模块所在的 `TITA_Lab.assets` 包；该常量来自包的 `__init__.py`。`str(...)` 将 Path 对象转成路径字符串。后面拼接的 "robots" 等片段不要带开头的 `/`，否则会重新从文件系统根目录开始解析。

## 3. 默认关节角与执行器

### 源码

```python
joint_pos={
    "joint_.*_leg_1": 0.0,
    "joint_.*_leg_2": 0.8,
    "joint_.*_leg_3": -1.5,
    "joint_.*_leg_4": 0.0,
},
```

```python
"joints_1": ImplicitActuatorCfg(
    joint_names_expr=["joint_left_leg_1", "joint_right_leg_1"],
    joint_effort_limit=None,
    joint_velocity_limit=None,
    stiffness=None,
    damping=None,
),
```

### 解释

默认角字典中的键是正则匹配表达式，分别覆盖左右同类关节。本次按实际 USD 核对，8 个转动关节都恰好匹配一个默认角，默认角均满足限位。四个执行器分组同样覆盖 8 个关节。

`stiffness=None`、`damping=None` 表示继承 USD 的对应值。当前复制的 USD 中 8 个角驱动的两项增益均为 0，所以此配置不会仅靠位置目标产生 PD 持位作用。应在后续确定动作到力矩的实现时再明确执行器类型与参数，不在本次模块拆分阶段盲目填写增益。

## 4. 当前验证边界

- Python AST 语法检查通过；没有实际导入模块或启动仿真。
- RL 包内 USD 与当前 nonRL USD 文件内容一致；组合错误 0，9 个刚体、8 个转动关节。
- 当前资产 default prim 已为 `/TITARobot`；Robot Schema 两个列表各有 9 个有效目标，停用 root_joint 仍登记在 joints 列表中。
- 已核实 `/TITARobot/Physics/root_joint` 的 jointEnabled=False；当前包内 USD 的 SHA256 为 `4ccf8847bc46e68de993a317eb6ff5f296a83524674a47f76313b54d1412be7c`。
- 同日后续：用户已在 `env_cfg.py` 的 Scene 中引用 TITA_CFG，并接入 Lab 通用 ROUGH 地形；资产初始高度现为 0.3 m。其他 MDP 仍为 Cartpole，新增 Scene 与物理属性尚未运行验证。

## 5. Gym 物理参数的资产侧落点（2026-10-15，用户已写入）

当前源码：

```python
spawn=sim_utils.UsdFileCfg(
    usd_path=str(TITA_LAB_ASSETS_DIR / "robots" / "tita" / "usd" / "tita.usd"),
    articulation_props={
        "/.*": [
            PhysxArticulationCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            )
        ]
    },
    rigid_props={
        "/.*": [
            PhysxRigidBodyCfg(max_depenetration_velocity=1.0)
        ]
    },
    collision_props={
        "/.*": [
            PhysxCollisionCfg(contact_offset=0.01, rest_offset=0.0)
        ]
    },
),
```

三个 `*_props` 都是字典：键指定在哪些路径上寻找对象，列表中的配置对象指定写哪些属性。这里的 `"/.*"` 是正则表达式后缀，拼在机器人生成路径后面，匹配其所有层级的后代路径。

例如，`scripts/import_assets.py` 使用 `{ENV_REGEX_NS}/TITA_Robot`。在第 0 个环境中，生成节点为 `/World/envs/env_0/TITA_Robot`，拼接后的匹配表达式是 `/World/envs/env_0/TITA_Robot/.*`。原 USD default prim `/TITARobot` 对应这个生成节点，其下的 `Geometry/base_link` 等相对层级保留。

路径匹配之后，Lab 内部写属性的函数还会检查对象已有的 USD 物理 API。所谓 writer，就是这些将配置值写入已加载场景属性的内部函数，不需要用户另写工具脚本，也不表示回写磁盘上的源 USD。

按当前 USD 静态检查，三组配置的实际目标如下：

| 字段 | 在匹配路径中进一步选择 | 当前目标 | 写入的参数 |
| --- | --- | --- | --- |
| `articulation_props` | 已有 `ArticulationRootAPI` 的节点 | 1 个：原 USD 的 `/TITARobot/Geometry/base_link` | 位置迭代 4、速度迭代 0 |
| `rigid_props` | 已有 `RigidBodyAPI` 的节点 | 9 个：机身和左右各 4 个腿/轮刚体 | 最大穿透修正速度 1.0 m/s |
| `collision_props` | 已有 `CollisionAPI` 的节点 | 9 个：各刚体下的碰撞形状节点 | contact_offset=0.01 m、rest_offset=0.0 m |

这不是给所有子节点都添加刚体或碰撞体；当前显式映射不会开启缺失物理 API 的自动创建。普通层级节点和纯视觉对象不会因为匹配 `"/.*"` 就被转成刚体。

**更正此前说明：** 不能笼统说“裸 fragment 列表不会自动递归”。当前字段注释与实际文件生成器实现不一致：`from_files.py` 为 articulation 的裸配置传入默认后缀 `"(/.*)?"`；刚体/碰撞体的 `_body_family_targeting` 也会在已有物理 API 的资产上采用这个后缀。对当前 TITA 资产，裸配置也能搜索根及其后代。用户现有 `{"/.*": [...]}` 是显式选择后代的有效写法，可以保留。

本地执行依据：[文件生成器的目标选择](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py:360)、[articulation 配置分发](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py:456)、[articulation 属性写入](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas.py:337)、[刚体属性写入](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas.py:714)、[碰撞属性写入](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/schemas/schemas.py:971)。

检查结论：这三组配置的字段、数值和目标选择正确，Python AST 检查通过，USD 组合错误为 0；所有对应目标均可被 `"/.*"` 覆盖。助手未修改 Python/USD，未实际执行生成器或启动仿真；用户此前的导入运行报告不能自动作为本次新增属性的运行验证。

## Actions 配套的显式执行器（2026-10-16，历史建议；当前已落盘，见下节）

替换前四组执行器中的一组历史节选：

```python
"joints_1": ImplicitActuatorCfg(
    joint_names_expr=["joint_left_leg_1", "joint_right_leg_1"],
    joint_effort_limit=None,
    joint_velocity_limit=None,
    stiffness=None,
    damping=None,
),
```

其余 joints_2/3/4 同样是 ImplicitActuatorCfg 和 None 增益。None 表示继承 USD，本轮 pxr 静态重读确认 8 个关节 Drive 的 stiffness/damping 都是 0，所以接入位置动作不等于接入了旧 Gym 的 PD。

本轮建议配合新的 ActionsCfg，把四组替换为两个不重叠的 IdealPDActuatorCfg 组：legs 选择 leg_[123]，Kp=40、Kd=1、模型和求解器力矩上限均 60 N·m；wheels 选择 leg_4，Kp=0、Kd=0.5、两上限均 15 N·m。限值对照原 URDF 和当前 USD 静态核实。

IdealPDActuator 的通用公式把位置反馈、速度反馈与力矩前馈相加，最后裁剪；当前零速度目标下，轮子收到 Actions 的 5a 前馈后得到 5a-0.5*dq。这与仅把轮子换成普通位置驱动有不同物理含义。

完整构造参数片段、API 字段及验证边界集中维护在 [03 Actions](../../../../../docs/migration/03_actions迁移.md)。本轮未修改此 Python 文件或 USD，也未运行控制；滤波、延迟和随机化仍待相应动作/执行器与事件逻辑补齐。

## 当前两组执行器静态检查（2026-10-16）

当前用户实际源码节选（TITA_CFG 的构造参数）：

```python
actuators={
    "legs": IdealPDActuatorCfg(
        joint_names_expr=["joint_.*_leg_[123]"],
        stiffness=40.0,
        damping=1.0,
        actuator_effort_limit=60.0,
        joint_effort_limit=60.0,
    ),
    "wheels": IdealPDActuatorCfg(
        joint_names_expr=["joint_.*_leg_4"],
        stiffness=0.0,
        damping=0.5,
        actuator_effort_limit=15.0,
        joint_effort_limit=15.0,
    )
},
```

两组均为 IdealPDActuatorCfg：legs 匹配 6 个腿关节，Kp=40、Kd=1；wheels 匹配 2 个轮子，Kp=0、Kd=0.5。组间无重叠，完整覆盖 USD 8 个驱动关节；模型/求解器两层上限均为腿 60、轮 15 N·m，与当前 USD 静态读值一致。

执行器使用目标、当前 q/dq 以及力矩前馈计算总力矩。零速度目标下，腿为位置 PD，轮子为 5a-0.5*dq；不能把 Actions 的后两列前馈当作已含阻尼的最终力矩。相关链路及后续随机化见 [03 Actions](../../../../../docs/migration/03_actions迁移.md)。

此前未使用的 ImplicitActuatorCfg 导入，最新文件已由用户删除；助手只核对并维护说明，未改 Python/USD 或启动仿真。

</details>
