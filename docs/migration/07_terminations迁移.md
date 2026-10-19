# 07 Terminations：一个回合什么时候结束

更新：2026-10-18。**本章终止函数、配置、导出和20 s时长已由用户写入，静态核对与保存函数的独立CPU检查通过。** 助手未修改TITA_Lab；真实接触传感器、环境和仿真尚未运行。

## 1. Gym 原来怎样做

旧 `asset.terminate_after_contacts_on=["base"]` 选择机身，`check_termination()` 判断当前接触力：

```python
self.reset_buf = torch.any(
    torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
    dim=1,
)
self.time_out_buf = self.episode_length_buf > self.max_episode_length
self.reset_buf |= self.time_out_buf
```

每环境得到一个布尔值：机身接触力模大于1 N，或者回合超时，就重置该环境。轮子正常接触地面不会仅因这一条触发终止。

回合时长来自父类env配置的20 s，策略周期为0.01 s，最大步数为ceil(20/0.01)=2000。旧代码使用严格大于比较，Lab原生time_out使用大于等于；本次保留20 s参数、采用Lab标准边界，比较符的一策略步差异单独记录。

## 2. Lab 中怎样写

### 2.1 env_cfg.py 声明两条结束条件

当前env_cfg.py已将TerminationsCfg替换为以下两项，移除了Cartpole的cart_out_of_bounds：

```python
@configclass
class TerminationsCfg:
    """Conditions that end a TITA episode."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact_current,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
            "threshold": 1.0,
        },
    )
```

time_out是Lab已有函数；illegal_contact_current是下面新增的任务函数。VelocityEnvCfg已有 `terminations: TerminationsCfg = TerminationsCfg()`，保留即可。

同文件VelocityEnvCfg.__post_init__中的当前时长为：

```python
self.episode_length_s = 20.0
```

### 2.2 新增mdp/terminations.py计算当前接触终止

源码路径：`TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/terminations.py`。用户已创建，以下代码与当前保存实现的计算一致（排版与docstring略有不同）：

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def illegal_contact_current(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return whether selected bodies currently exceed a contact-force threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = contact_sensor.data.net_normal_forces_w.torch
    selected_forces_w = forces_w[:, sensor_cfg.body_ids, :]
    force_magnitudes = torch.linalg.vector_norm(selected_forces_w, dim=-1)
    return torch.any(force_magnitudes > threshold, dim=1)
```

SceneEntityCfg按传感器中的身体名字base_link选择机身。管理器先将身体名字解析为body_ids，函数按索引取对应的力。

按当前Scene静态推导，全部接触力形状(N,9,3)；选一个base后为(N,1,3)，求范数得到(N,1)，阈值比较并沿身体轴any，最后为(N,)布尔张量。实际身体绑定和传感器输出仍需运行验证。

### 2.3 在mdp/__init__.pyi导出

当前.pyi保留版权头，并登记以下导出：

```python
__all__ = [
    "joint_pos_target_l2",
    "joint_pos_rel_zero_wheels",
    "proprio_history",
    "illegal_contact_current",
]

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import joint_pos_rel_zero_wheels, proprio_history
from .rewards import joint_pos_target_l2
from .terminations import illegal_contact_current
```

本项目lazy_export读取.pyi，加入导出后cfg才能访问mdp.illegal_contact_current。

## 3. 为什么这样迁移

### 3.1 为什么不用现成illegal_contact

本地Lab的illegal_contact使用net_normal_forces_w_history，并取历史力模最大值；当前Scene保存3次物理步采样。旧Gym判断的是本次读取的当前力。直接换成现成项会扩大采样窗口，所以本次加一个任务内小函数，使用net_normal_forces_w当前帧。

仍沿用当前PhysX公开的聚合法向接触力接口；与旧接触报告分量、碰撞几何的差异继续保留在02/00章，不以这个函数宣称整个接触模型等价。传感器history_length=3保持原配置，可供后续其他计算使用。

### 3.2 超时分类与回合时长

func返回是否结束；TerminationTermCfg.time_out告诉管理器该项是时间截断，通常返回为truncated。机身接触项默认time_out=False，属于任务终止terminated。它们都会结束回合，但对终止惩罚和学习程序的末尾价值处理有区别。20 s时长来自episode_length_s。

### 3.3 终止与重置怎样连接

```text
每个策略步推进物理
  → TerminationManager计算两项并取逻辑或
  → 对结束的env_ids调用_reset_idx
  → 父类执行reset Events，放回关节与机身状态
  → 当前环境子类清相应历史缓存
```

Terminations决定何时结束，reset Events决定新回合从什么状态开始。不需要手写一段发现True后调用Event函数的循环。

## 4. 当前状态与检查

- EventCfg的两个TITA重置项和VelocityEnvCfg.events登记已与06章提案做AST比对，一致；尚未运行真实重置。
- 本章函数、导出、两条Terminations和20 s时长均已保存；配置、函数入口和.pyi导出静态检查通过。
- 2026-10-18直接抽取用户保存的函数做独立CPU检查：只给轮子大力不触发、机身向量范数超过1 N触发、恰好1 N不触发；结果为[False,True,False]，输出(3,)bool。报告在`/tmp/tita_termination_review_20261018/saved_code_result.json`，没有导入项目或验证真实传感器。此前提案检查另见2026-10-17项目历史。
- 两项速度跟踪奖励已在08章接入；其他奖励、物理domain_rand、成本与模型仍待迁移，尚无完整训练或复现结果。

## 5. 对照源码

| 内容 | 本地入口 |
| --- | --- |
| Gym结束判断 | [check_termination](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:699) |
| Gym机身选择与20 s | [TITA asset](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:90)、[父类env](/home/eaziel/Downloads/tita_rl-master/configs/legged_robot_config.py:4) |
| Lab标准配置参考 | [官方velocity TerminationsCfg](/home/eaziel/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:312) |
| Lab函数与窗口 | [time_out](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py:32)、[illegal_contact](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/terminations.py:174) |
| 布尔结果分类与合并 | [TerminationManager.compute](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/termination_manager.py:153) |
| 自动重置调用 | [ManagerBasedRLEnv.step](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:246) |

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> any是什么函数？

**A：** 这里用torch.any沿选中身体轴检查是否至少有一个True。当前函数把(N,B,3)力向量沿xyz求范数，得到(N,B)，再比较阈值并沿dim=1合并，返回(N,)布尔张量。dim、body_ids及完整数值例子集中在[终止函数的补充问答](../../code_explaination/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/terminations.py.md#补充问答)。

[返回迁移入口](README.md) · [查看来源与覆盖表](00_迁移映射索引.md)
