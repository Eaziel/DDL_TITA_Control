# terminations.py：接触力怎样变成回合结束标记

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-02。

被解释文件：[mdp/terminations.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/terminations.py)。

本文件把选中刚体的当前法向接触力变成每台机器人一个True/False。读完后应能解释传感器测什么、w表示什么、三个索引各选什么，以及vector_norm和any分别消去了哪个维度。

本文核对当前源码与本地 IsaacLab-main / Sim 6.2 调用关系；未代运行仿真或训练。

## 1. 当前完整源码

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
    """Return whether selected bodies exceed a contact-force threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # shape: (N, B_all, 3)
    #   N     = num_envs        环境数
    #   B_all = 传感器监测的所有 body 数
    #   3     = 力的 xyz 分量
    forces_w = contact_sensor.data.net_normal_forces_w.torch

    # shape: (N, B, 3)
    #   只保留配置里指定的那些 body，B = len(sensor_cfg.body_ids)
    selected_forces_w = forces_w[:, sensor_cfg.body_ids, :]

    # shape: (N, B)
    #   对最后一维（xyz）求模长 → 每个 body 一个“力的大小”
    force_magnitudes = torch.linalg.vector_norm(selected_forces_w, dim=-1)

    # shape: (N,)  bool
    #   每个环境一个判断：这个环境里“有没有任何一个 body”超阈值
    return torch.any(force_magnitudes > threshold, dim=1)
```

## 2. 先确定“接触力”在当前代码里指什么

物理引擎用碰撞几何判断表面接触，并在求解运动约束时计算接触作用。法向力沿局部接触面的法线方向，摩擦力位于局部接触面切平面中。法线随接触面方向变化，因此法向力并不总沿世界z轴。

本函数读取 `contact_sensor.data.net_normal_forces_w.torch`：

- `contact_sensor` 是场景中按 `sensor_cfg.name` 查到的传感器对象。
- `data` 提供传感器结果，`.torch` 取Torch张量形式。
- `normal` 明确选择法向部分。
- `net` 表示对同一个受测刚体的接触作用进行聚合。
- `w` 是world，表示三个分量沿世界坐标轴表达；它不表示weight或功率单位W。

对某个刚体，若有多个接触点，其聚合法向力可表达为：

$$
\boldsymbol F_b^w=\sum_{k\in\mathcal C_b} f_{n,k}\boldsymbol n_k^w.
$$

$\mathcal C_b$ 是此刚体的接触点集合，$f_{n,k}$ 是相应法向力大小，$\boldsymbol n_k^w$ 是世界坐标表达的法向方向。聚合先作向量求和；之后才计算这个合力的模。因此合力模不等于每个接触力大小的简单相加，方向不同会影响结果。

当前Sim6.2的 `net_forces_w` 可以包含法向与摩擦两部分；本文件明确读取 `net_normal_forces_w`，并没有因此自动改成总力。法向/总力的区别会影响阈值的物理意义。[Lab接触传感器文档](https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/contact_sensor.html)

## 3. 传感器测哪些刚体，函数再选择哪些刚体

当前场景配置创建 `contact_forces` 传感器，并由prim表达式绑定机器人中的相应刚体。这个对象持有所有受测刚体的数据。`SceneEntityCfg("contact_forces",body_names=["base_link"])` 则进一步选择机身。

Lab根据传感器自己的body名称顺序解析 `sensor_cfg.body_ids`。如果base_link在该传感器第3个位置，编号就类似 `[3]`；具体编号必须由当前实际名称解析得到，不能凭USD树显示顺序猜，也不能直接用机器人关节编号代替。

读取全数据后，三个维度为：

$$
F\in\mathbb R^{N\times B_{all}\times3}.
$$

N是环境数，B_all是传感器受测刚体数，最后3列是世界Fx/Fy/Fz，单位N。代码：

```python
selected_forces_w = forces_w[:, sensor_cfg.body_ids, :]
```

第一个冒号取全部机器人；中间编号取选中刚体；最后冒号取xyz全部分量。若选机身一个body，结果是 `(N,1,3)`；保留中间长度1的维度，后面仍可统一处理多个body。

本函数没有按碰撞对方筛选。来自地面、机器人其他部件或其他物体的接触都可能计入聚合结果，实际是否产生取决于碰撞开关与碰撞过滤。因此名字 `base_contact` 只说明检测机身，不能凭名字断定“只在机身碰地面时触发”。

## 4. 第一次消维：把xyz变成一个力的大小

```python
force_magnitudes = torch.linalg.vector_norm(selected_forces_w, dim=-1)
```

`torch.linalg` 是Torch的线性代数模块。`vector_norm` 默认计算2范数，也叫欧氏长度：

$$
M_{n,b}=\sqrt{F_{x,n,b}^2+F_{y,n,b}^2+F_{z,n,b}^2}.
$$

`dim` 指定沿哪个轴计算，`-1`表示最后一个轴。函数从 `(N,B,3)` 沿xyz三列计算，输出 `(N,B)`。它不是只取最后一列Fz；最后一个冒号之前选出了三个分量，范数现在把三个分量一起使用。

例如两个环境、一个body的力为 `[[[3,4,0]],[[0,0,0.5]]]`，范数结果为 `[[5],[0.5]]`，形状 `(2,1)`。范数始终非负，力方向不同但长度相同会给相同结果。

编辑器无法提示dim时，应区分静态类型信息与运行API。当前Torch函数确实提供dim参数，缺补全不能据此判断参数不存在；此处不需要为补全改变计算公式。[vector_norm官方API](https://docs.pytorch.org/docs/2.11/generated/torch.linalg.vector_norm.html)

## 5. 第二次消维：把每个body的判断变成每台机器人的判断

```python
return torch.any(force_magnitudes > threshold, dim=1)
```

`>`先逐元素比较，得到 `(N,B)` 的布尔张量。`torch.any(...,dim=1)` 沿body维执行逻辑OR：同一机器人有任一个选中body超过阈值就返回True，结果 `(N,)`。

`torch.any` 是Torch函数，不是Python内置 `any`；它能指定张量维度并返回张量。前面的范数消掉xyz轴，这里的any再消掉body轴，环境轴始终保留。

对上面的力例子，阈值1N时，比较结果为 `[[True],[False]]`，最终为 `[True,False]`。若某环境选中两个body、大小为 `[0.2,2.0]`，any为True；它不返回2.0，也不返回超阈值body的数量。

函数的数学输出为：

$$
d_n=\bigvee_{b\in\mathcal B_{selected}}
\mathbf1\{\|\boldsymbol F^w_{n,b}\|_2>F_{threshold}\}.
$$

$\bigvee$表示把一组布尔值作OR。代码使用严格大于号，恰好等于阈值时条件为False。

## 6. 结束标记怎样让环境reset

配置 `TerminationsCfg.base_contact` 把本函数交给 `DoneTerm`，参数选择base_link、阈值1N。`TerminationManager` 每个策略步在物理推进后调用各项，记录哪些行失败、哪些行超时。环境再根据结束行执行 `_reset_idx`。

`time_out` 是另一项，按20秒回合上限判断，并标为超时。终止并非只有一种来源：当前base_contact属于失败，time_out属于截断。wrapper随后把二者OR成训练循环要用的dones，同时单独提供 `time_outs`。

本函数读取当前传感器结果，没有读取 `_history`、没有沿历史取最大值，也没有做接触OR滤波。传感器配置 `history_length=3` 只说明另外保存了历史，不代表访问当前字段时会自动把三帧聚合。6维状态中的两次接触OR在另一个类中单独实现。

当前函数在策略步末判定，而非每个物理子步都直接结束回合；因此“当前样本”的时间口径是实际读取的传感器结果。是否满足想要的碰撞判定，仍要结合传感器更新时机与物理接触验证，不能只靠布尔张量形状确认。

若要理解“碰一下为什么有的惩罚、有的reset”，可对照：奖励碰撞项对小腿计数，cost对轮子检查阈值，本文件对机身给失败标记。三者可以读取同一传感器，同时选择不同body、执行不同归约。

## 源码与API依据

- [Lab ContactSensorData：力的接口与shape](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/sensors/contact_sensor/base_contact_sensor_data.py)
- [TerminationManager：失败与超时](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/termination_manager.py)
- [PyTorch any](https://docs.pytorch.org/docs/2.11/generated/torch.any.html)
