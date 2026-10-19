# mdp/observations.py：轮角屏蔽与历史取数

源码目录：`TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/`。
源码文件：[observations.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/observations.py)。

更新：2026-10-18。两个函数及导出均已写入；history组和环境step推进也已写入。以下为当前源码，省略版权头；尚未运行真实环境。

## 当前代码

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from ..tita_velocity_env import TitaVelocityEnv

def joint_pos_rel_zero_wheels(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        wheel_cfg: SceneEntityCfg,
)->torch.Tensor:
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
```

## joint_pos_rel_zero_wheels：生成八维相对位置

`env.scene[asset_cfg.name]` 取得场景中的robot对象。两个SceneEntityCfg都指向robot，但用途不同：asset_cfg指定输出八个关节及其顺序，wheel_cfg指定要屏蔽的两个轮子。管理器初始化时将名字解析成joint_ids。

计算先对全部关节做 `q-q_default`，减法生成新张量；在这份新张量中按原生轮索引置零；最后按asset_cfg的顺序取列，返回 `(N,8)`。因此不会改写真实关节角、默认角或轮速。

轮角指轮轴旋转关节角，单位rad。本任务按旧设计屏蔽相位/累计转角，保留轮速。后续ObservationManager才加位置噪声，因此最终两轮位置列仍可能有纯噪声。

## proprio_history：读取已有十帧

`env.proprio_history` 是环境持有的ProprioHistory对象，`read()` 返回缓存副本 `(N,10,33)`；reshape按整帧展开，返回 `(N,330)`。这个函数不重新取机器人状态，也不更新缓存。

TYPE_CHECKING内的TitaVelocityEnv导入供类型检查；future annotations让运行时不立即求值这份类型注解，避免循环导入。这里的torch.Tensor注解声明返回张量对象，输出列数由函数实际返回值决定。

## 怎样接入与验证

[__init__.pyi](__init__.pyi.md) 导出两个函数；[env_cfg.py](../config/TITA/env_cfg.py.md) 配置轮角函数参数并登记history组；环境step负责每策略步推进一次缓存。当前这三部分均已写入，真实读取与推进结果仍待运行验证。

此前轮角函数做过独立CPU检查：非原生顺序、非零轮角、输出形状与原状态不变均通过。当前函数/导出与配置仅做静态核对，未实例化Isaac Lab。

Gym公式、Lab接线及迁移理由见 [05 Observations](../../../../../../../docs/migration/05_observations迁移.md)。API对照：[标准joint_pos_rel](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py:202)、[SceneEntityCfg解析](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/manager_base.py:403)。
