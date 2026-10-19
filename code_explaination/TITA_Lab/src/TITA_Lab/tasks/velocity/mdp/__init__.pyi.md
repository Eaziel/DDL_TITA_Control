# mdp/__init__.pyi：登记自定义MDP函数导出

源码目录：`TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/`。
源码文件：[__init__.pyi](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/__init__.pyi)。

更新：2026-10-19。当前轮角、历史、终止、碰撞、功率函数与加速度类均已导出；以下为保存源码，省略版权头。名称/导入路径静态检查通过，真实惰性导入尚未运行。

本轮新增推扰函数导出的方案在文末，仍待用户写入。

```python
__all__ = [
    "joint_pos_target_l2",
    "joint_pos_rel_zero_wheels",
    "proprio_history",
    "illegal_contact_current",
    "undesired_contacts_current",
    "joint_power_signed",
    "JointAccelerationPenalty",
]

# Forward stable MDP terms lazily, then override with environment-specific terms below.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import joint_pos_rel_zero_wheels, proprio_history
from .rewards import JointAccelerationPenalty, joint_pos_target_l2, joint_power_signed, undesired_contacts_current
from .terminations import illegal_contact_current
```

`from isaaclab.envs.mdp import *` 保留官方通用条目；后面的相对导入加入本任务函数；`__all__` 声明公开名称。调用端因此可以写 `mdp.joint_pos_rel_zero_wheels`、`mdp.proprio_history` 和 `mdp.illegal_contact_current`。

终止项先在任务模块实现illegal_contact_current，再通过.pyi导出，TerminationsCfg才能写mdp.illegal_contact_current。现有通用转发和观测导出保留。完整功能见[07章](../../../../../../../docs/migration/07_terminations迁移.md)。

本工程的 `mdp/__init__.py` 调用lazy_export。本地实现会读取相邻.pyi中的导出关系，所以这个文件既帮助类型检查，也参与运行时惰性导出；无需在.py里再复制一套导入。

ProprioHistory工具类由环境直接从其模块导入，不需要为本次观测调用额外导出。函数、history组和环境step的配合见 [05第5节](../../../../../../../docs/migration/05_observations迁移.md#5-历史lab-中怎样接入)。

依据：[lazy_export实现](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/utils/module.py:84)。当前只核对名称与路径，没有导入项目或运行RL环境。


## 当前已导出：当前帧碰撞计数函数

undesired_contacts_current已加入上述__all__与from .rewards import，保存代码AST核对通过。lazy_export据此从任务rewards模块找到函数，供RewardsCfg.collision调用；base_height_l2/action_rate_l2继续由官方MDP转发提供。

函数与当前getter建议见[rewards.py说明](rewards.py.md)，完整功能与用户追问见[08章](../../../../../../../docs/migration/08_rewards迁移.md)。真实惰性导入与环境运行仍未验证。

## 当前已导出：功率函数与加速度类

joint_power_signed导出函数，JointAccelerationPenalty导出类，两者已在上述__all__与from .rewards import中。RewardsCfg可以写mdp.joint_power_signed和mdp.JointAccelerationPenalty；lazy_export只负责找到公开对象，后续实例构造由ManagerBase负责。

类的创建、每步__call__及reset调用链见[rewards.py补充问答](rewards.py.md#补充问答)。当前10项配置与导出已核对，功率None接口已保存；真实惰性导入及环境运行未验证。

## 待写：定时推扰函数

在现有__all__列表中追加名字：

```python
"push_by_setting_xy_velocity",
```

在自定义相对导入区域、from .observations前加入：

```python
from .events import push_by_setting_xy_velocity
```

这两处让lazy_export能按mdp.push_by_setting_xy_velocity找到新events模块中的函数；原有官方转发和自定义导出继续保留。函数本身见[events.py说明](events.py.md)，完整配置见[06第6节](../../../../../../../docs/migration/06_events迁移.md#6-运动过程中每15秒扰动一次水平速度)。候选.pyi的AST/Ruff检查已通过，实际文件尚未修改。
