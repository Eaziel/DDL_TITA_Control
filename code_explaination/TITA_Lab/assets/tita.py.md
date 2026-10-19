# tita.py：机器人资产、默认状态和执行器

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/assets/`。
源码文件：[tita.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/assets/tita.py)。

更新：2026-10-18。当前配置使用两组IdealPD执行器，名义增益为腿40/1、轮0/0.5，限矩60/15 N·m。已有静态核对，RL环境控制尚未运行。旧配置原文折叠在末尾。

## 1. 当前代码

```python
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

from . import TITA_LAB_ASSETS_DIR  # . 表示当前的 TITA_Lab.assets 包。这个路径常量已在它的 __init__.py 中定义

TITA_CFG = ArticulationCfg(
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
        rigid_props={"/.*": [PhysxRigidBodyCfg(max_depenetration_velocity=1.0)]},
        collision_props={"/.*": [PhysxCollisionCfg(contact_offset=0.01, rest_offset=0.0)]},
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_.*_leg_1": 0.0,
            "joint_.*_leg_2": 0.8,
            "joint_.*_leg_3": -1.5,
            "joint_.*_leg_4": 0.0,
        },
        pos=(0.0, 0.0, 0.3),
    ),
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

IdealPD总力矩为Kp(q_des-q)+Kd(dq_des-dq)+tau_ff。当前腿收到位置目标，得到40(q_des-q)-dq；轮子收到Actions的5a力矩前馈，得到5a-0.5*dq。

actuator_effort_limit限制模型输出的总力矩，joint_effort_limit配置求解器侧上限。旧滤波、延迟与物理随机化仍待迁入。参数映射与完整公式见[03 Actions](../../../docs/migration/03_actions迁移.md)。

## 4. 相关源码与验证

[01 Physics](../../../docs/migration/01_physics.md)解释物理属性归属，[02 Scene](../../../docs/migration/02_scene迁移.md)解释资产怎样进入场景，[USD检查快照](../../../docs/USD_CHECK_2026-10-15.md)保留当时的层级与关节检查证据。用户报告导入脚本可运行；助手未在本次文档整理中运行仿真。

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> 这是啥意思？你看我tita.py好着没？

**A：** 你问的是属性字典中的"/.*"以及此前说明中的writer。这个后缀与机器人生成路径拼接，选择其后代；属性写入函数再按已有ArticulationRootAPI、RigidBodyAPI、CollisionAPI选择对应对象。当前资产的这些物理对象位于USD子层级，因此此显式路径写法可覆盖它们。

属性写入发生在加载后的场景；writer就是Lab内部执行这些赋值的函数。旧静态检查匹配到1个articulation root、9个刚体和9个碰撞体，当前三组字典的值和目标路径已核对。完整运行效果仍待仿真验证。

本地实现也允许适用的裸配置通过默认后缀递归选择；当前显式字典写法有效。源码：[from_files目标选择](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py:360)。


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

完整构造参数片段、API 字段及验证边界集中维护在 [03 Actions](../../../docs/migration/03_actions迁移.md)。本轮未修改此 Python 文件或 USD，也未运行控制；滤波、延迟和随机化仍待相应动作/执行器与事件逻辑补齐。

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

执行器使用目标、当前 q/dq 以及力矩前馈计算总力矩。零速度目标下，腿为位置 PD，轮子为 5a-0.5*dq；不能把 Actions 的后两列前馈当作已含阻尼的最终力矩。相关链路及后续随机化见 [03 Actions](../../../docs/migration/03_actions迁移.md)。

此前未使用的 ImplicitActuatorCfg 导入，最新文件已由用户删除；助手只核对并维护说明，未改 Python/USD 或启动仿真。

</details>
