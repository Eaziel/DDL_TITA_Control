# 16 Cost：轮接触力过大与侧向碰撞

**目标与结果：** 判断两轮是否受力过大或出现较大的侧向碰撞，每项返回每环境一个0/1。

整理：2026-10-26。两项接触cost及导出已写，六项管理器和环境接线均已完成。此章解释轮力阈值与侧向碰撞判断；真实接触力口径差异仍保留。 当前证据与整体状态以[项目记忆](../../PROJECT_MEMORY.md)为准。

## 1. Gym怎样计算

旧任务用当前两轮受力计算两项0/1 cost：

| 项 | 每轮判断 | 每环境输出 |
| --- | --- | --- |
| feet_contact_forces | 力向量范数>500 N | 任一轮满足为1，否则0 |
| stumble | sqrt(Fx²+Fy²)>5×abs(Fz) | 任一轮满足为1，否则0 |

stumble用世界水平力明显大于竖直力作为轮子撞到竖直面的判断。例如[120,0,20] N满足120>100，[50,0,100] N不满足50>500。它是一条任务设计规则，不是对所有碰撞的精确分类。

这两项没有上一帧状态，直接读取本次力。原始输出为(N,)浮点0/1，不在函数内乘策略dt或算法k。

令 $\mathcal W$ 是左右两轮，原始函数逐环境返回：

$$
f_{force}=\mathbf1\{\exists b\in\mathcal W:\lVert\mathbf F_b^w\rVert_2>500\ \mathrm N\},
$$

$$
f_{stumble}=\mathbf1\{\exists b\in\mathcal W:\sqrt{F_{b,x}^2+F_{b,y}^2}>5\lvert F_{b,z}\rvert\}.
$$

$\mathbf1$ 表示条件满足取1，否则取0；代码中的any沿轮体轴实现“至少一轮满足”。

## 2. Lab使用已有ContactSensor

当前contact_forces已覆盖机器人身体，函数通过sensor_cfg.body_ids选出两轮，得到(N,2,3)的世界系法向力。沿xyz求范数得到(N,2)，沿轮体维any得到(N,)。

当前PhysX接口给出聚合法向力，旧Gym使用合接触力。下列代码对照的是同一输入向量上的计算公式，不能据此认定真实受力数值完全等价。墙面法向力本身可以沿世界x/y方向，因此法向力向量也有Fx/Fy分量。

## 3. 在costs.py追加实现

先在现有TYPE_CHECKING块增加from isaaclab.sensors import ContactSensor。然后追加：

```python
def wheel_contact_force_cost(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 500.0,
) -> torch.Tensor:
    """Flag environments with excessive wheel normal contact force."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # 选中轮体的当前世界系法向力，(N,W,3)，单位N；本任务W=2。
    forces_w = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :]
    # 每个轮体的合成力大小(N,W)，任一轮超过阈值则该环境为1。
    force_magnitudes = torch.linalg.vector_norm(forces_w, dim=-1)
    return torch.any(force_magnitudes > threshold, dim=1).float()


def wheel_stumble_cost(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    horizontal_ratio: float = 5.0,
) -> torch.Tensor:
    """Flag environments with excessive horizontal-to-vertical wheel contact force."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = sensor.data.net_normal_forces_w.torch[:, sensor_cfg.body_ids, :]
    # 每轮世界水平力sqrt(Fx²+Fy²)，以及竖直力abs(Fz)，均为(N,W)。
    horizontal_force = torch.linalg.vector_norm(forces_w[..., :2], dim=-1)
    vertical_force = torch.abs(forces_w[..., 2])
    # 旧规则：水平力超过竖直力的5倍，记一次侧向碰撞判断。
    stumble = horizontal_force > horizontal_ratio * vertical_force
    return torch.any(stumble, dim=1).float()
```

horizontal_ratio为无量纲倍数，threshold的单位为N。两函数都采用严格大于，恰好等于阈值时输出0。

## 4. .pyi统一命名并导出（已完成）

用户源码把力矩函数命名为joint_effort_limit_cost，现有__all__与.costs导入均已统一，并已增加wheel_contact_force_cost、wheel_stumble_cost。

完整.costs导入块为：

```python
from .costs import (
    joint_effort_limit_cost,
    joint_position_limit_cost,
    joint_velocity_limit_cost,
    wheel_contact_force_cost,
    wheel_stumble_cost,
)
```

Scene的传感器已经存在。本次先完成五个无历史依赖的函数；最终cost cfg将用SceneEntityCfg("contact_forces", body_names=["left_leg_4","right_leg_4"], preserve_order=True)指定两轮。尚不新增一个未实现的CostTermCfg，也不把cost误接成reward或观测组。

## 5. 后续目标与验证边界

最后一项acc_smoothness需要上一策略步的关节速度。下一步连同六列收集、reset前采集、跨回合缓存规则一起实现；当前函数和导出本身不会触发每步计算。

[当前五函数检查](../audits/2026-10-21_cost_collection/current_five_result.json)直接读取用户已保存的五个函数，分别对照旧Gym表达式，64环境边界检查差0、导出无缺失。最初候选检查仍保存在2026-10-21_contact_costs中。未运行PhysX、传感器或训练。

来源：[Gym两项接触cost](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1382)、[500 N阈值](/home/eaziel/Downloads/tita_rl-master/configs/legged_robot_config.py:187)、[本地接触传感器说明](/home/eaziel/IsaacLab-main/docs/source/concepts/sensors/contact_sensor.rst)。
