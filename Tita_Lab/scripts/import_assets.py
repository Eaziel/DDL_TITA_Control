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
