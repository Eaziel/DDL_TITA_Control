# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from isaaclab_tasks.utils import PresetCfg

from ... import mdp

##
# Pre-defined configs
##

from TITA_Lab.assets.tita import TITA_CFG  # isort:skip

# 明确八个关节的排列
# 这个顺序与当前 Actions 的“六个腿关节＋两个轮子”一致。我们显式指定它，后续就能知道观测每一列对应哪个关节。
TITA_JOINT_NAMES = [
    "joint_left_leg_1",
    "joint_left_leg_2",
    "joint_left_leg_3",
    "joint_right_leg_1",
    "joint_right_leg_2",
    "joint_right_leg_3",
    "joint_left_leg_4",
    "joint_right_leg_4",
]


@configclass
class VelocityPhysicsCfg(PresetCfg):
    """Physics presets for the tita environment."""

    isaacsim_physx: PhysxCfg = PhysxCfg(
        solver_type=1,  # TGS
        bounce_threshold_velocity=0.5,
        gpu_max_rigid_contact_count=2**23,
    )
    default: PhysxCfg = isaacsim_physx


##
# Scene definition
##


@configclass
class VelocitySceneCfg(InteractiveSceneCfg):
    """Configuration for a tita scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=PhysxRigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robot
    robot: ArticulationCfg = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True

    # sensors: the concrete implementation is selected automatically from the active physics backend;
    # backend-specific fields such as ``global_world_only`` are ignored by the backends that do not use them
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/height_scanner",
        update_period=0.01,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=[1.6, 1.0],
            ordering="yx",
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )
    # 接触力
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Nominal action mapping for TITA legs and wheels."""

    leg_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "joint_left_leg_1",
            "joint_left_leg_2",
            "joint_left_leg_3",
            "joint_right_leg_1",
            "joint_right_leg_2",
            "joint_right_leg_3",
        ],
        preserve_order=True,
        scale={
            "joint_.*_leg_1": 0.25,
            "joint_.*_leg_[23]": 0.5,
        },
        use_default_offset=True,
    )

    wheel_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["joint_left_leg_4", "joint_right_leg_4"],
        preserve_order=True,
        scale=5.0,
    )


@configclass
class CommandsCfg:
    """Velocity and heading command specifications for TITA."""

    # 速度命令生成器的配置；CommandManager据此创建运行时生成器，
    # 并以配置条目名"base_velocity"登记。
    # get_command("base_velocity")返回torch.Tensor，shape=(num_envs, 3)。
    # 每行：[vx_cmd, vy_cmd, wz_cmd]，均以机身坐标系表达：
    #   第0/1列：前向/侧向目标线速度，单位m/s；
    #   第2列：绕z轴目标角速度，单位rad/s。
    # 当前航向模式下，目标航向另存于生成器的heading_target中，
    # 第2列由航向误差计算得到；对外输出仍然只有三列。
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=True,
        heading_control_stiffness=0.5,  # 航向外环的kp！航向外环：kp*（目标朝向-当前朝向）
        rel_heading_envs=1.0,
        rel_standing_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),  # 希望沿机身前向运动多快，单位 m/s
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),  # 希望机身转多快，单位 rad/s
            heading=(-3.14, 3.14),  # 希望最终朝向哪个方向，单位 rad
        ),
        debug_vis=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # Lab API 先加噪声后缩放，而Gym先缩放观测后加噪声
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.8, n_max=0.8, operation="add"),  # 加噪声
            scale=0.25,
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05, operation="add"),  # 加噪声
            scale=1.0,
        )

        velocity_commands = ObsTerm(  # 不加噪声
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )

        # 八个关节的（当前角度-默认角度）。例如膝关节默认 −1.5 rad，当前 −1.4 rad，结果是 +0.1
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_zero_wheels,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=TITA_JOINT_NAMES,
                    preserve_order=True,
                ),
                "wheel_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["joint_left_leg_4", "joint_right_leg_4"],
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01, operation="add"),  # 加噪声
            scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        # 八个关节的实际角速度，统一乘 0.05，对应旧任务的速度缩放
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=TITA_JOINT_NAMES,
                    preserve_order=True,
                ),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5, operation="add"),  # 加噪声
            scale=0.05,
        )

        # 环境接收的完整八维动作，处于动作项缩放之前
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True  # 启用噪声
            self.concatenate_terms = True  # 告诉观测管理器：把这一组中的各个观测项合并！
            # 没有修改 concatenate_dim，所以使用默认值 -1，沿最后一维拼接。

    # 历史观测组
    @configclass
    class HistoryCfg(ObsGroup):
        """Past proprioceptive observations."""

        proprio = ObsTerm(func=mdp.proprio_history)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()  # 把该观测组命名为 policy，因此最终通过 obs["policy"] 取得整组张量。
    history: HistoryCfg = HistoryCfg()  # obs["history"]


@configclass
class EventCfg:
    """Reset TITA joints and base state at each episode."""

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )

    # 身体质心
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "com_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_xy_velocity,
        mode="interval",
        interval_range_s=(15.0, 15.0),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": (-1.0, 1.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.2),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for TITA velocity tracking."""

    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0, # 每项贡献 = 函数返回值 * weight * step_dt
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
        },
    )

    # 惩罚绕机身 x、y 轴过快转动。
    # ang_vel_xy_l2:
    #   读取：asset.data.root_ang_vel_b.torch，shape：(环境数, 3)，每行为 [wx, wy, wz]
    #   计算：torch.sum(torch.square(angular_velocity[:, :2]), dim=1)，即wx**2 + wy**2
    ang_vel_xy = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    # flat_orientation_l2:
    #   读取：asset.data.projected_gravity_b.torch，shape：：(环境数, 3)，世界坐标系中朝下的方向是 [0, 0, -1]
    #       此处为机身坐标系下的重力，反映机身怎样倾斜。
    #   计算：gx**2 + gy**2
    orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # 机身参考原点相对扫描区域平均地面的高度
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.35,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    # 相邻策略步的八维原始动作之差。
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    # 统计两侧小腿当前超过接触力阈值的数量。
    collision = RewTerm(
        func=mdp.undesired_contacts_current,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_leg_3", "right_leg_3"],
            ),
            "threshold": 0.1,
        },
    )

    powers = RewTerm(
        func=mdp.joint_power_signed,
        weight=-2.0e-5,
    )

    dof_acc = RewTerm(
        func=mdp.JointAccelerationPenalty,
        weight=-2.5e-7,
    )

    # 给 因机身触地而Terminate 的情况惩罚（正常超时而termin,不惩罚）
    termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-200.0,
        params={"term_keys": "base_contact"}
    )


@configclass
class TerminationsCfg:
    """Conditions that end a TITA episode."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact_current,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_link"],
            ),
            "threshold": 1.0,
        },
    )


##
# Environment configuration
##


@configclass
class VelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cart-pole environment."""

    # Scene settings
    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.gravity = (0.0, 0.0, -9.81)
        # visualizer camera settings
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))
        # simulation settings
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
        self.sim.physics = VelocityPhysicsCfg()
