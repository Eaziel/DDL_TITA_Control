# 15 Cost：先完成三项关节越界判断

**目标与结果：** 为每个环境判断关节位置、速度与力矩是否超出旧任务阈值，分别保留三列。

整理：2026-10-26。三项关节cost、软位置范围与导出已写；另三项、收集器及环境接线也已完成，见16—18。此章解释原始判断，不再把17称作未写候选。 当前证据与整体状态以[项目记忆](../../PROJECT_MEMORY.md)为准。

## 1. Gym怎样计算cost

旧任务分别计算六项cost，保留六列，供NP3O使用。Reward回答任务完成得怎样，cost另外记录约束情况；这些函数的输出不直接并入现有奖励总和。

本次三个函数都返回(N,)的0/1张量：只要某个被选关节越界，该环境为1；全部没越界则为0。越界判断使用严格大于/小于，恰好在边界上仍为0。

| Gym函数 | 使用的数据 | 判定阈值 |
| --- | --- | --- |
| _cost_pos_limit | 当前关节角 | 原位置范围中央的90% |
| _cost_torque_limit | 限幅后的驱动力矩 | 原力矩上限×1.0 |
| _cost_dof_vel_limits | 当前关节角速度 | 原速度上限×1.0 |

旧compute_cost只对每列乘策略dt=0.01。配置costs.scales中的0.3/0.1实际另存为cost_k_values，交给算法形成约束损失；不在下列函数中乘它们，也不在这里乘dt。时间积分留到统一收集六列时做一次。

位置软范围以原区间中点为中心缩小，公式为：

$$
m_j=\frac{q_{lo,j}+q_{hi,j}}2,\quad r_j=q_{hi,j}-q_{lo,j},\qquad
q^{soft}_{lo,j}=m_j-0.9\frac{r_j}2,\quad
q^{soft}_{hi,j}=m_j+0.9\frac{r_j}2.
$$

即保留原范围中央90%，两侧各收进总宽度的5%；这只是判断范围，不会主动改变物理硬限位。

例如髋关节原范围约[-0.7854,0.7854] rad，软范围约[-0.7069,0.7069] rad。q=0.72 rad仍在物理范围内，但位置cost已经为1。软边界用于任务判断，不缩小求解器的物理关节范围。

## 2. Lab怎样实现三个函数

新建源码文件：Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/costs.py。完整内容如下：

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def joint_position_limit_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Flag environments with a joint outside its soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    # 当前关节角(N,J)，单位rad；J由asset_cfg选中的关节决定。
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    # (N,J,2)，最后一维依次为软下限、软上限；由资产配置的0.9生成。
    limits = asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, :]
    out_of_limits = (joint_pos < limits[..., 0]) | (joint_pos > limits[..., 1])
    # (N,J)逐行归并：任一关节越界即为True，再转成(N,)的0/1。
    return torch.any(out_of_limits, dim=1).float()


def joint_effort_limit_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """Flag environments whose applied joint effort exceeds the threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    # 读取限幅后的驱动力矩(N,J)，与旧Gym的self.torques保持同一计算阶段。
    torques = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    # 求解器力矩上限(N,J)，当前腿60、轮15 N·m。
    limits = asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids]
    # soft_ratio是阈值比例；旧值为1.0，不是算法中的cost权重。
    return torch.any(torch.abs(torques) > limits * soft_ratio, dim=1).float()


def joint_velocity_limit_cost(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    soft_ratio: float = 1.0,
) -> torch.Tensor:
    """Flag environments whose joint speed exceeds the threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    # 当前关节角速度(N,J)，单位rad/s。
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    # 求解器关节速度上限(N,J)，当前腿25、轮20 rad/s。
    limits = asset.data.joint_vel_limits.torch[:, asset_cfg.joint_ids]
    return torch.any(torch.abs(joint_vel) > limits * soft_ratio, dim=1).float()
```

asset_cfg指定机器人和参与判断的关节。全部八关节均可参与；当前USD轮轴保留了原URDF约±999999 rad的大范围，正常转动不会轻易触及位置边界。

布尔矩阵中每一行属于一台机器人。例如两行分别为[False,True,False]、[False,False,False]，any(dim=1)得到[True,False]，float后为[1,0]。这保留“是否发生”的判断，不累计越界关节数量或越界幅度。

## 3. .pyi导出

在mdp/__init__.pyi现有__all__列表追加：

```python
"joint_position_limit_cost",
"joint_effort_limit_cost",
"joint_velocity_limit_cost",
```

并在相对导入区增加，保留所有现有导入：

```python
from .costs import (
    joint_position_limit_cost,
    joint_effort_limit_cost,
    joint_velocity_limit_cost,
)
```

## 4. cfg设置位置软边界

在env_cfg.py的VelocitySceneCfg中，robot定义后增加一行：

```python
robot: ArticulationCfg = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
# 任务判断使用原关节位置范围中央的90%；物理关节限位保持原范围。
robot.soft_joint_pos_limit_factor = 0.9
```

字段属于机器人ArticulationCfg。只在RL任务Scene中设置，与本任务的cost判定对应；保留原有spawn、接触、材质和执行器配置。

本地Lab已有joint_pos_limits、joint_vel_limits等reward函数，但它们返回越界幅度之和；本任务需要0/1，并且旧速度阈值来自URDF/求解器速度上限，因此用joint_vel_limits数据而不是执行器产生的soft_joint_vel_limits。

## 5. 一个必须保留的旧行为

旧self.torques在_compute_torques返回前已经限幅；soft_torque_limit=1.0时，再严格比较是否超过同一限值，通常得到0。这里仍使用applied_effort，所以保留这个行为。

若改用computed_effort，测量的就会是“限幅前需求是否超额”，含义不同。是否修正旧任务的这一设计，需要之后单独决定，不在迁移时悄悄改变。

## 6. 本次终点与后续接线

本章介绍其中三个关节判断与位置软范围。另三项和六列收集、reset前采集已经在16—18章接通；当前check也已将六项cost存入rollout并计算各自GAE。单独定义函数不会自动执行，实际调用由环境创建的TitaCostManager组织。不能在super().step返回后直接重读机器人算结束步cost，因为部分环境已经重置。

另一个接线约定：旧class_to_dict使用dir(obj)，所以实际六列按名称排序为acc_smoothness、dof_vel_limits、feet_contact_forces、pos_limit、stumble、torque_limit，和类正文书写顺序不同。最终收集时显式固定这个顺序，并使k、d保持一致。

早期[128环境候选检查](../audits/2026-10-21_joint_costs/result.json)保留当时torque名称；本轮[64环境检查](../audits/2026-10-21_contact_costs/result.json)已直接读取当前三个函数，结果与旧公式一致，并发现导出名称不匹配。均未运行真实仿真。

## 7. 对照源码

- [Gym三项判断](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1358)、[成本收集](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:726)。
- [Gym参数](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:101)、[六项cost配置](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:178)。
- [Lab软位置范围](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py)、[原生位置/速度reward](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/rewards.py)。

## 补充问答

**Q：** cost在我学的RL理论里没有提到。cost是mdp中的吗？讲一下什么是cost？

**A：** 普通MDP常用reward表示目标，也可以改用cost表示要最小化的量，最大化reward与最小化其相反数是等价表述。本项目在reward之外另设六个cost，对应带约束MDP（CMDP）：提高奖励，同时要求各项期望累计cost不超过各自上限。cost可以是0/1，也可以是连续值。

$$
\max_\theta\ J_r(\theta)=\mathbb E_{\pi_\theta}\!\left[\sum_{t\ge0}\gamma^t r_t\right]
\quad\text{使得}\quad
J_{c_i}(\theta)=\mathbb E_{\pi_\theta}\!\left[\sum_{t\ge0}\gamma^t c_{i,t}\right]\le d_i\quad\forall i.
$$

这是说明CMDP含义的标准折扣形式，具体NP3O中的归一化与阈值处理还需按旧代码核对。

这里c_i,t为第i类当步cost，d_i是该类期望折扣累计cost的上限。约束通常针对期望累计量，不能把训练目标当作每一步都不越界的硬保证。理论定义见[CPO论文第4节](https://proceedings.mlr.press/v70/achiam17a/achiam17a.pdf)。

TITA的reward已有跟踪/姿态等合成分数，另外六列cost交给NP3O的cost价值估计及约束损失。mdp/costs.py负责由环境数据算出这些数，训练代码负责怎样使用它们；函数返回1本身不会阻止物理运动。

**Q：** 这个字段属于机器人的 `ArticulationCfg`。Lab会据此提供代码读取的：`asset.data.soft_joint_pos_limits`。为啥？？

**A：** 配置里的robot是ArticulationCfg；运行时env.scene["robot"]是Lab依据配置创建的Articulation实例。该实例用cfg保存配置，用data保存状态和属性数据。

在本地PhysX机器人初始化中，可以直接找到计算调用：

```python
wp.launch(
    articulation_kernels.update_soft_joint_pos_limits,
    dim=(self.num_instances, self.num_joints),
    inputs=[
        self.data.joint_pos_limits,
        self.cfg.soft_joint_pos_limit_factor,
    ],
    outputs=[self.data.soft_joint_pos_limits],
    device=self.device,
)
```

输入是物理关节范围和配置倍数，kernel按中点±倍数×半宽算出结果，写入data保存的软限位数组。data类本来就定义了soft_joint_pos_limits属性；0.9只改变初始化时写入的数值，即使不填、使用默认1.0，这个属性也存在。再用.torch取得(N,J,2)的PyTorch形式。

源码：[初始化调用](/home/eaziel/IsaacLab-main/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py)、[计算公式](/home/eaziel/IsaacLab-main/source/isaaclab_physx/isaaclab_physx/assets/articulation/kernels.py)、[数据属性](/home/eaziel/IsaacLab-main/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py)。
