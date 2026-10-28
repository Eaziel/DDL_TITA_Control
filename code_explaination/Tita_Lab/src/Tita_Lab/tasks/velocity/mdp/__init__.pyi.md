# mdp/__init__.pyi：配置可引用的名字

**阅读目标：** 理解函数/配置类如何经.pyi与lazy_export供mdp包使用，并区分公开名字与真正创建实例的class_type。

源码：/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/__init__.pyi。
更新：2026-10-25。内容纯AI生成，由人类要求生成，未经人类修改。以下为当前保存内容，cost、地形课程与命令Cfg均已导出。

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "joint_pos_target_l2",
    "joint_pos_rel_zero_wheels",
    "proprio_history",
    "illegal_contact_current",
    "undesired_contacts_current",
    "joint_power_signed",
    "JointAccelerationPenalty",
    "push_by_setting_xy_velocity",
    "RandomizeTitaMaterial",
    "randomize_tita_motor_factors",
    "TitaJointActionCfg",
    "action_rate_clipped_l2",
    "randomize_tita_action_lag",
    "TitaDynamicsParameters",
    "TitaPrivilegedState",
    "joint_position_limit_cost",
    "joint_effort_limit_cost",
    "joint_velocity_limit_cost",
    "wheel_contact_force_cost",
    "wheel_stumble_cost",
    "JointAccelerationCost",
    "terrain_levels_vel",
    "TitaVelocityCommandCfg",
]

# Forward stable MDP terms lazily, then override with environment-specific terms below.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from isaaclab_tasks.core.velocity.mdp.curriculums import terrain_levels_vel

from .actions import TitaJointActionCfg
from .commands import TitaVelocityCommandCfg
from .costs import (
    JointAccelerationCost,
    joint_effort_limit_cost,
    joint_position_limit_cost,
    joint_velocity_limit_cost,
    wheel_contact_force_cost,
    wheel_stumble_cost,
)
from .events import (
    RandomizeTitaMaterial,
    push_by_setting_xy_velocity,
    randomize_tita_action_lag,
    randomize_tita_motor_factors,
)
from .observations import TitaDynamicsParameters, TitaPrivilegedState, joint_pos_rel_zero_wheels, proprio_history
from .rewards import (
    JointAccelerationPenalty,
    action_rate_clipped_l2,
    joint_pos_target_l2,
    joint_power_signed,
    undesired_contacts_current,
)
from .terminations import illegal_contact_current
```

同目录__init__.py执行lazy_export()，它在运行时解析相邻.pyi的导入声明。此.pyi同时服务类型工具与Lab的运行时惰性加载，Python本身不会自动执行.pyi导入。

from .commands import TitaVelocityCommandCfg让env_cfg可通过mdp.TitaVelocityCommandCfg找到配置类。Cfg里的class_type已经持有TitaVelocityCommand实现类；CommandManager据此构造对象，无需再把实现类单独导出到mdp包。

terrain_levels_vel直接引用官方课程函数。CostTermCfg、TitaCostManager和ProprioHistory由环境直接导入并持有，不需要为了运行而全部列入此导出文件。具体创建、调用链与用户问答见commands.py.md及迁移21章。
