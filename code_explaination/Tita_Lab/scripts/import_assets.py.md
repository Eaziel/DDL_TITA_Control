# import_assets.py：独立资产与物理循环

内容纯AI生成，由人类要求生成，未经人类修改。源码核对与整理：2026-11-08。

对应源码：`Tita_Lab/scripts/import_assets.py`。本脚本在平地建立TITA，设置关节状态、执行器目标和材质，再推进物理。它用于独立资产检查；RL任务由TitaVelocityEnv另行建立。

## 1. 当前完整源码

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from typing import TYPE_CHECKING

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import instantiate

parser = argparse.ArgumentParser(description="Spawn and simulate the TITA robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# 注册并解析仿真启动参数，例如 --device 和 --viz。
add_launcher_args(parser)
args_cli = parser.parse_args()


import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from Tita_Lab.assets.tita import TITA_CFG

if TYPE_CHECKING:
    from isaaclab.assets.articulation import BaseArticulation
    from isaaclab.scene import InteractiveScene


class TITARobotSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(  # 配置地面的摩擦系数、恢复系数
            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="average",
                restitution_combine_mode="average",
            )
        ),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # robot
    tita_robot = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/TITA_Robot")


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: "InteractiveScene",
):
    """Simulate the robot with randomized joint states and materials."""
    sim_dt = sim.get_physics_dt()

    robot: BaseArticulation = scene[
        "tita_robot"
    ]  # scene 是 InteractiveScene 类的实例，内部用字典保存不同类别的实体，例如机器人、普通刚体和传感器。
    joint_pos = robot.data.default_joint_pos.torch.clone()
    joint_vel = robot.data.default_joint_vel.torch.clone()

    # 随机化
    joint_pos *= torch.empty_like(joint_pos).uniform_(0.5, 1.5)
    joint_vel.zero_()

    # 保证采样结果没有越过关节限位
    limits = robot.data.joint_pos_limits.torch
    joint_pos.clamp_(limits[..., 0], limits[..., 1])

    # 官方方式：摩擦和恢复系数分别准备 64 个随机值
    mu_buckets = torch.empty(64, device="cpu").uniform_(0.2, 2.75)
    e_buckets = torch.empty(64, device="cpu").uniform_(0.0, 1.0)

    # 每个环境分别选取一个摩擦值、一个恢复系数
    n = robot.num_instances
    mu = mu_buckets[torch.randint(64, (n,), device="cpu")]
    e = e_buckets[torch.randint(64, (n,), device="cpu")]

    # 同一环境内，整个机器人的碰撞体共用这组参数
    view = robot.root_view
    materials = wp.to_torch(view.get_material_properties()).clone()
    materials[..., 0] = mu[:, None]  # 静摩擦
    materials[..., 1] = mu[:, None]  # 动摩擦，与静摩擦相同
    materials[..., 2] = e[:, None]  # 恢复系数

    env_ids = torch.arange(n, dtype=torch.int32, device="cpu")
    view.set_material_properties(
        wp.from_torch(materials, dtype=wp.float32),
        wp.from_torch(env_ids, dtype=wp.int32),
    )

    print("每个环境的摩擦和恢复系数：", torch.stack((mu, e), dim=-1))

    # 实际起始状态
    robot.write_joint_position_to_sim_index(position=joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
    # 驱动要维持的目标
    robot.actuators.target_command.set_position_index(value=joint_pos)
    robot.actuators.target_command.set_velocity_index(value=joint_vel)

    while sim.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


def main():
    """Spawn and simulate the configured TITA asset."""
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=0.005,
        # 保留当前执行器计算路径。
        use_newton_actuators=False,
    )

    # 进入时启动运行环境，离开时自动清理并关闭。
    with launch_simulation(sim_cfg, args_cli):
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(
            [3.5, 0.0, 3.2],
            [0.0, 0.0, 0.5],
        )

        scene_cfg = TITARobotSceneCfg(
            num_envs=args_cli.num_envs,
            env_spacing=4.0,
        )
        scene: InteractiveScene = instantiate(scene_cfg)

        # 启动物理仿真，并刷新机器人状态缓存。
        sim.reset()
        scene.update(sim.get_physics_dt())

        robot: BaseArticulation = scene["tita_robot"]

        # 第0台机器人的世界位置 [x, y, z]，单位 m。
        root_position = robot.data.root_pos_w.torch[0].detach().cpu().tolist()
        print("[CHECK] robot world position:", root_position)

        # 相机对准机器人的实际位置。
        x, y, z = root_position
        sim.set_camera_view(
            eye=(x + 1.5, y + 1.5, z + 1.0),
            target=(x, y, z),
        )

        run_simulator(sim, scene)


if __name__ == "__main__":
    main()
```

## 2. 脚本入口怎样建立场景

```python
add_launcher_args(parser)
args_cli = parser.parse_args()

with launch_simulation(sim_cfg, args_cli):
    sim = sim_utils.SimulationContext(sim_cfg)
    scene: InteractiveScene = instantiate(scene_cfg)
    sim.reset()
    scene.update(sim.get_physics_dt())
    run_simulator(sim, scene)
```

这是完整源码中的调用关系摘录。add_launcher_args让解析器认识device、viz等启动选项，parse_args读取命令行；main创建本次SimulationCfg并进入launch_simulation上下文。with进入时准备运行环境，退出时清理。当前文件没有AppLauncher或simulation_app变量。

scene_cfg描述地面、灯光和机器人；instantiate根据配置创建运行时InteractiveScene。场景字段tita_robot成为`scene["tita_robot"]`的键；prim_path决定USD生成位置，两者用途不同。

sim.reset使物理后端建立运行状态，scene.update刷新资产缓存，之后才能可靠读取机器人位置等数据。打印的世界位置用于把相机对准实际生成位置，窗口朝向不改变机器人动力学。

本脚本dt=.005s，每次循环调用一次sim.step，因此是5ms物理步。RL任务另有dt=.0025s、decimation=4；这两个独立入口的步长不应混为一谈。

## 3. 默认状态怎样变成实际状态和执行器目标

```python
joint_pos = robot.data.default_joint_pos.torch.clone()
joint_vel = robot.data.default_joint_vel.torch.clone()
joint_pos *= torch.empty_like(joint_pos).uniform_(0.5, 1.5)
joint_vel.zero_()
limits = robot.data.joint_pos_limits.torch
joint_pos.clamp_(limits[..., 0], limits[..., 1])
```

默认状态是参考值。clone取得独立张量，乘上逐关节采样因子，再将角度限制在上下限之间；zero_将速度设为0。joint_pos/joint_vel为(N,J)，limits为(N,J,2)，最后两列分别是下限与上限。

```python
robot.write_joint_position_to_sim_index(position=joint_pos)
robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
robot.actuators.target_command.set_position_index(value=joint_pos)
robot.actuators.target_command.set_velocity_index(value=joint_vel)
```

前两句写入实际状态，决定机器人此刻的关节角和速度；后两句写入执行器目标，决定PD随后追踪的参考。两者可以取相同初值，但不是同一份物理量。

## 4. 64个候选值怎样赋给所有碰撞形状

```python
mu_buckets = torch.empty(64, device="cpu").uniform_(0.2, 2.75)
e_buckets = torch.empty(64, device="cpu").uniform_(0.0, 1.0)
mu = mu_buckets[torch.randint(64, (n,), device="cpu")]
e = e_buckets[torch.randint(64, (n,), device="cpu")]
materials[..., 0] = mu[:, None]
materials[..., 1] = mu[:, None]
materials[..., 2] = e[:, None]
```

mu_buckets/e_buckets各有64个随机候选。每个环境分别抽一个整数索引，得到(N,)的摩擦/恢复系数；设该机器人有S个碰撞形状，则materials为(N,S,3)。

mu[:,None]将(N,)整理为(N,1)，沿形状轴广播给本机器人全部S个碰撞形状：

$$
P_{n,s,:}=(\mu_n,\mu_n,e_n),
\qquad s=0,\ldots,S-1.
$$

前三个材质分量依次是静摩擦、动摩擦与恢复系数。这里让同一机器人各形状共用一组参数，没有让所有机器人取同一组参数。

wp.to_torch取得Warp数据的Torch视图；clone后修改独立副本，再通过wp.from_torch交给set_material_properties写回后端。这组材质接口使用CPU数组；关节状态仍可在仿真设备上，两种数据所在设备由各接口约定决定。更细的索引/广播说明见[非RL材质解释](../../TITA_non_RL/scripts/run_sim.py.md)。

## 5. 每次循环怎样推进机器人

```python
while sim.is_running():
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)
```

循环先向后端提交目标与命令，推进一个物理步，再更新场景缓存。sim.is_running读取仿真运行状态；当前代码不是只刷新画面的render循环。

## 6. 手动运行与验证范围

```bash
source "$HOME/my_venvs/lab3_sim62/.venv/bin/activate"
cd /mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab
python scripts/import_assets.py --num_envs 1 --device cuda:0 --viz kit
```

这是独立资产检查，不创建RL的命令、奖励、cost或训练runner。它使用argparse，不接收任务Hydra式physics覆盖。11月8日只核对代码与说明，没有重跑脚本或验证物理表现。

历史的静态render、相机与CUDA图捕获排查保留在[原文](../../../docs/history/2026-10-22_import_assets排查原文.md)，不作为当前待执行步骤。任务侧运行与GAE证据见[项目记忆](../../../PROJECT_MEMORY.md)。

## 补充问答

**Q：** 啥反应没有，f是干啥的？

**A：** Viewport的F用于聚焦到选中对象；先在Stage选择，再让Viewport获得键盘焦点。是否成功不能单独判断模型和动力学是否正确，当前脚本还会读取实际位置来设置相机。

**Q：** 有机器人。但是穿模，无物理效果，机器人就是原始USD的姿态，不动。

**A：** 当时是临时的render诊断，没有推进物理。当前源码运行run_simulator，循环包含sim.step。对应旧过程仍保留在上面的历史原文链接，不再要求重做静态诊断。
