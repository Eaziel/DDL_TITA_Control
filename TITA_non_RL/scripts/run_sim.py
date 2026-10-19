# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import (
    ArticulationCfg,
    BaseArticulation,  # BaseArticulation：与其他后端（如Newton Mjwarp）共用的基类
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

TITA_HWBR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="TITA_non_RL/assets/tita.usd"),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_.*_leg_1": 0.0,
            "joint_.*_leg_2": 0.8,
            "joint_.*_leg_3": -1.5,
            "joint_.*_leg_4": 0.0,
        },
        pos=(0.0, 0.0, 0.6),
    ),
    actuators={
        "joints_1": ImplicitActuatorCfg(
            joint_names_expr=["joint_left_leg_1", "joint_right_leg_1"],
            joint_effort_limit=None,
            joint_velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "joints_2": ImplicitActuatorCfg(
            joint_names_expr=["joint_left_leg_2", "joint_right_leg_2"],
            joint_effort_limit=None,
            joint_velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "joints_3": ImplicitActuatorCfg(
            joint_names_expr=["joint_left_leg_3", "joint_right_leg_3"],
            joint_effort_limit=None,
            joint_velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
        "joints_4": ImplicitActuatorCfg(
            joint_names_expr=["joint_left_leg_4", "joint_right_leg_4"],
            joint_effort_limit=None,
            joint_velocity_limit=None,
            stiffness=None,
            damping=None,
        ),
    },
)


class TITARobotSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg( # 配置地面的摩擦系数、恢复系数
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
    tita_robot = TITA_HWBR_CFG.replace(
        prim_path="{ENV_REGEX_NS}/TITA_Robot"
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
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
    materials[..., 2] = e[:, None]   # 恢复系数

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

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=0.005,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # Design scene
    scene_cfg = TITARobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
