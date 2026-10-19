# env_cfg.py：按配置块组织TITA任务

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/`。
源码文件：[env_cfg.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py)。

更新：2026-10-19。下面按现行配置的职责顺序解释：物理、场景、动作、命令、观测、Events、奖励、终止，最后由VelocityEnvCfg组合。源码由用户编写；已有静态/局部CPU检查，真实RL环境尚未运行。第6节两个startup、两个reset已保存；定时推扰为本轮待写方案。旧解释原文折叠在末尾。

## 导入与关节顺序

以下是本文件的配置类型导入节选：

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
```

这些名字指向配置类。调用RewTerm(...)、ObsTerm(...)等构造配置对象，记录函数引用与参数；环境创建各manager后，由manager执行函数。函数和类的实际实现分别在官方MDP或任务的mdp模块中。

```python
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
```

TITA_JOINT_NAMES按左腿1/2/3、右腿1/2/3、左轮、右轮排列，供关节观测按名称选择并保持当前八维顺序。


## 1. VelocityPhysicsCfg：物理后端

```python
@configclass
class VelocityPhysicsCfg(PresetCfg):
    """Physics presets for the tita environment."""

    isaacsim_physx: PhysxCfg = PhysxCfg(
        solver_type=1,  # TGS
        bounce_threshold_velocity=0.5,
        gpu_max_rigid_contact_count=2**23,
    )
    default: PhysxCfg = isaacsim_physx
```

PhysxCfg保存求解器及接触容量参数；PresetCfg组织可选后端，default选择当前PhysX配置。物理步长和策略间隔由最后的VelocityEnvCfg设置。

Gym对应计算与完整迁移理由见[01章](../../../../../../../../docs/migration/01_physics.md)。


## 2. VelocitySceneCfg：创建场景实体

<details>
<summary>查看当前VelocitySceneCfg源码</summary>

```python
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
```

</details>

InteractiveScene根据这些字段创建并登记地形、robot、高度扫描、接触传感器和灯光。robot是资产配置的副本，替换生成路径；height_scanner和contact_forces作为场景键供MDP读取。地形、射线和接触力的物理过程见02章。

Gym对应计算与完整迁移理由见[02章](../../../../../../../../docs/migration/02_scene迁移.md)。


## 3. ActionsCfg：把八维动作变成关节目标

```python
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
```

ActionManager按条目声明顺序分配动作列：前六列交给腿位置动作，后两列交给轮子力矩动作。位置项用默认角加缩放动作生成目标；轮子项生成5a的力矩前馈。tita.py中的IdealPD执行器再结合当前q/dq计算总力矩。

Gym对应计算与完整迁移理由见[03章](../../../../../../../../docs/migration/03_actions迁移.md)。


## 4. CommandsCfg：生成速度任务目标

```python
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
```

CommandManager根据base_velocity配置创建UniformVelocityCommand对象。该对象采样和更新目标速度，对外返回(N,3)的[vx,vy,wz]。heading模式由当前航向误差计算目标wz。采样、航向反馈、奖励和观测的连接见04章。

Gym对应计算与完整迁移理由见[04章](../../../../../../../../docs/migration/04_commands迁移.md)。


## 5. ObservationsCfg：组织当前帧和过去十帧

<details>
<summary>查看当前ObservationsCfg源码</summary>

```python
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
```

</details>

PolicyCfg和HistoryCfg是两个观测组配置类；policy/history字段分别创建组配置对象。ObservationManager调用每个ObsTerm.func，完成加噪、缩放和组拼接。当前policy为(N,33)，history为(N,330)；历史由函数读取已有缓存。六项含义、噪声换算及完整缓存接线见05章。

Gym对应计算与完整迁移理由见[05章](../../../../../../../../docs/migration/05_observations迁移.md)。


## 6. EventCfg：回合初值与启动时的物理参数

<details>
<summary>查看已保存的两个reset项源码（节选）</summary>

```python
@configclass
class EventCfg:
    """Reset TITA joints and base state at each episode."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
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
```

</details>

EventManager在reset模式被触发时，对指定env_ids执行关节和根状态重置。两项分别读取默认状态、按配置抽样并写入仿真；init_state作为保存的参考值继续供动作和观测使用。

### 已保存：两个startup条目

下面两项已保存于EventCfg的reset_robot_joints之前；上述两个reset项继续保留。当前四项及质量/质心参数的AST核对通过。

```python
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
```

EventTerm是EventTermCfg的导入别名。这里构造的是两个配置对象：func保存具体实现的类引用，mode保存触发类别，params保存调用参数；没有在定义EventCfg时就抽质量或修改仿真。类说明可改为`Configure startup physical randomization and episode resets.`。

`SceneEntityCfg("robot", body_names="base_link")`中，robot选择Scene的资产对象；base_link选择其基座刚体。框架用名称解析实际body_ids，所以不写死某个身体列号。质量函数从`asset.data.body_mass.torch:(N,B)`中修改选中的列；当前只选一个基座，因此每个环境抽一个质量增量。

`operation="add"`与`distribution="uniform"`合起来表示：原基座质量加一个[-1,3] kg的均匀随机数。当前USD基座约13.2 kg，最终约12.2～16.2 kg。`recompute_inertia=True`使本地质量类按新旧质量比缩放默认惯量，保持一致的质量比例。

COM函数读取`asset.data.body_com_pose_b.torch:(N,B,7)`。每个身体的前3个数为它在自身link坐标系中的质心位置，后4个数为质心坐标系的朝向。函数为每个环境抽一个`(dx,dy,dz)`，只加到选中基座的位置部分，朝向保留；xyz范围单位均为m。它改变物理模型的质心，不移动外观或出生位置。

管理器保存配置和触发调用的本地关键语句是：

```python
# EventManager._prepare_terms(): 按执行模式保存配置。
self._mode_term_cfgs[term_cfg.mode].append(term_cfg)

# ManagerBasedRLEnv.load_managers(): 环境初始化时触发startup。
self.event_manager.apply(mode="startup")

# EventManager.apply(): 对该模式下的配置逐项调用。
term_cfg.func(self._env, env_ids, **term_cfg.params)
```

调用前，管理器会把可调用类实例化，所以最后一行执行原生项的__call__。startup调用时env_ids默认None，原生项据此选择整批环境。后续reset只触发reset模式，不重新抽这两个物理参数。这与旧Gym在_create_envs中执行一次_process_rigid_body_props相对应。

两项实现已由任务mdp的官方转发导出，无需新建任务events.py或修改.pyi。惯量具体算法与旧Gym是否严格等价仍未验证；旧mass_params_tensor供特权观测使用的路径仍待迁移。当前方案已有原生类/采样函数抽取的CPU替身检查和临时完整配置Ruff检查，没有导入真实任务或运行物理后端。

Gym对应计算与完整迁移理由见[06章](../../../../../../../../docs/migration/06_events迁移.md)。

### 待写：interval模式的水平速度推扰

```python
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
```

EventManager共用一个15 s计时器，间隔到达时调用整个批次；各环境的vx/vy仍独立取样。全局timer在回合reset时保留，函数会收到env_ids=None。params只选择机器人整体，不选某个关节或身体列；具体函数读写根质心的实际速度。

这里的func是待新增的普通函数，计时交给manager；与上面两个可调用类的初始化方式不同。函数将实际水平速度覆盖成U(-1,1)，保留vz和角速度；Commands中的目标不变。函数/导出/配置需共同接上，详见[06第6节](../../../../../../../../docs/migration/06_events迁移.md#6-运动过程中每15秒扰动一次水平速度)及[events.py说明](../../mdp/events.py.md)。当前Lab interval执行于奖励/reset后，与旧Gym回调位置不同，此差异保留在覆盖表中。


## 7. RewardsCfg：组合10项奖励

```python
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
```

2026-10-19当前10项的函数/类、权重与两个自定义名称导出已核对。每项返回(N,)，RewardManager乘weight及step_dt=0.01后相加。两项速度跟踪读取base_velocity及机身速度；机身平稳、高度、动作变化与当前小腿碰撞沿用前述计算；termination读取base_contact并排除超时。

powers调用任务函数，用限幅后关节力矩乘关节速度求带符号机械功率。dof_acc保存JointAccelerationPenalty类对象，原生管理器创建实例、每步调用其__call__，并在reset时同步选中环境的速度缓存。当前已显式写func=；该字段先保存类对象，管理器创建实例后按__call__计算。

类的构造/调用/重置原代码见[rewards.py补充问答](../../mdp/rewards.py.md#补充问答)。功率函数已保存None默认值与条件构造，加速度reset已补slice，当前静态核对通过。历史CPU替身检查与当前文件核对分别记录，未运行真实RL环境。接触力口径、射线/旧高度图采样、动作裁剪/reset等差异仍在[08章](../../../../../../../../docs/migration/08_rewards迁移.md)记录。

当前已接入[08第10节最终奖励合成](../../../../../../../../docs/migration/08_rewards迁移.md#10-将各项合成为最终分数)：在既有TitaVelocityEnv.step里复用原生缓存，执行普通合计夹非负、失败项后加。单项函数、配置与合成step均已保存；历史缓存推进保留，真实运行尚未验证。

## 8. TerminationsCfg：决定何时结束回合

```python
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
```

TerminationManager计算20秒超时与机身当前法向力模大于1 N两项条件，再合并结果。time_out=True分类为时间截断；机身接触分类为任务终止。环境随后对结束的行执行reset，具体初值由EventCfg提供。

Gym对应计算与完整迁移理由见[07章](../../../../../../../../docs/migration/07_terminations迁移.md)。


## 9. VelocityEnvCfg：组合整份任务配置

```python
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
```

VelocityEnvCfg继承官方ManagerBasedRLEnvCfg，把以上配置对象交给对应的scene与manager。sim.dt=0.0025、decimation=4给出0.01 s策略周期，episode_length_s=20给出回合时长。自定义TitaVelocityEnv接收这份cfg，沿用官方环境流程并管理历史缓存。

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> envcfg和现在写的这个有啥区别！？

**A：** 你问的是新增ProprioHistory以后，env_cfg与自定义代码各做什么。顺着数据看：

| 位置 | 负责的工作 |
| --- | --- |
| env_cfg.py | 声明history组，登记读取函数及参数 |
| ProprioHistory实例 | 保存(N,10,33)跨步张量，提供读、更新、重置方法 |
| TitaVelocityEnv实例 | 持有缓存，在step开头更新，在_reset_idx清对应环境 |
| mdp.proprio_history | 读取缓存、展开为(N,330) |
| ObservationManager | 按cfg调用该函数，把结果作为history组输出 |

这些文件共同完成“过去十帧观测”这个功能。配置选择实现，缓存保存状态，环境安排写入时刻，管理器组织输出。完整代码顺序见[05第5—6节](../../../../../../../../docs/migration/05_observations迁移.md#5-历史lab-中怎样接入)。

**Q：**

> uh，我哪有policycfg？

**A：** PolicyCfg定义在ObservationsCfg内部，是一个嵌套类；policy: PolicyCfg = PolicyCfg()创建并登记该组的配置对象。运行时输出字典的键是policy，因此用obs["policy"]读取当前33维观测。配置类名、组字段名和返回的张量分别处在定义、登记、取数三个位置。

**Q：**

> 以及，base_velocity我得加点注释，这个玩意最终是啥内容，shape是啥，看代码我看不来，因为鬼知道那个类最后被咋处理了

**A：** base_velocity保存UniformVelocityCommandCfg配置；运行时生成器持有(N,3)的[vx_cmd,vy_cmd,wz_cmd]张量，CommandManager.get_command("base_velocity")返回该张量。可复制的注释及生成器的真实取数路径见[04 Commands补充问答](../../../../../../../../docs/migration/04_commands迁移.md#23-base_velocity配置对象与实际命令张量)。


## 历史解释原文

以下为当时的原文与阶段快照，现行状态按上文及PROJECT_MEMORY核实。

<details>
<summary>展开此前解释与阶段快照（含已失效建议，保留原文）</summary>

# env_cfg.py 代码解释

解释的文件目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/`。

解释的文件：[env_cfg.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py)。

2026-10-15：依据本地 Isaac Lab `release/3.0.0` / `9ef05c8972634889968b332bfc4b80860789c3ff` 静态核对。用户已将 PhysX 配置值、dt=0.0025、decimation=4 写入 Python，助手未运行仿真；default 的类型注解仍待修正。旧 Gym 来源是 `/home/eaziel/Downloads/tita_rl-master/configs/legged_robot_config.py:218`。

持续维护的迁移入口：[迁移说明](../../../../../../../../docs/migration/README.md)。时间尺度的来源、公式和执行关系见 [Physics 章节](../../../../../../../../docs/migration/01_physics.md)。

## 1. Physics 类是后端预设集合

历史模板源码节选（用户现已简化，下面不再代表当前实现）：

```python
@configclass
class VelocityPhysicsCfg(PresetCfg):
    """Physics presets for the cart-pole environment."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    # 中间的其他预设在此节选中省略。
    default: NewtonCfg = newton_mjwarp
```

`PresetCfg` 的不同字段是可选择的配置，`default` 指定没有显式选择时采用的项。这段不表示几个后端同时执行。旧模板默认是 Newton MJWarp；迁移原 Isaac Gym PhysX 行为时，先选择 Isaac Sim PhysX。

本地依据：[PresetCfg](/home/eaziel/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py:101)、[PhysxCfg](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py:33)。

可采用的精简预设示例：

```python
@configclass
class VelocityPhysicsCfg(PresetCfg):
    """Physics presets for the TITA environment."""

    isaacsim_physx: PhysxCfg = PhysxCfg(
        solver_type=1,
        bounce_threshold_velocity=0.5,
        gpu_max_rigid_contact_count=2**23,
    )
    default: PhysxCfg = isaacsim_physx
```

若保留原有其他分支，只调整 isaacsim_physx 分支和 default 也可以。`gpu_max_rigid_contact_count` 是 Lab 的接触流缓冲区容量，不能把旧 Gym 的 contact pairs 名称与该字段认作跨版本严格一一对应。

同日最新源码中的 default：

```python
default: NewtonCfg = isaacsim_physx
```

该值现已指向 PhysxCfg 实例，类型注解应同步改为上述建议中的 `PhysxCfg`。不能把值的实际类型和残留的类型注解混为一谈；本次没有修改该行或验证运行。

## 2. 仿真公共设置位于环境的 sim

当前源码：

```python
def __post_init__(self) -> None:
    self.decimation = 4
    self.sim.dt = 0.0025
    self.episode_length_s = 5
    self.sim.gravity = (0.0, 0.0, -9.81)
    self.sim.default_visualizer_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))
    self.sim.render_interval = self.decimation
    self.sim.physics = VelocityPhysicsCfg()
```

`self.sim` 是 SimulationCfg；`self.sim.physics` 是物理后端配置。dt、gravity 属于前者，solver_type 等属于后者。

与原 Gym 时间设置对应的片段（用户当前已采用这些数值；其余配置不在此片段中）：

```python
self.decimation = 4
self.sim.dt = 0.0025
self.sim.gravity = (0.0, 0.0, -9.81)
self.sim.render_interval = self.decimation
self.sim.physics = VelocityPhysicsCfg()
```

原 Gym substeps=1 时，物理步长保持 0.0025 s；策略每 4 个物理步决策一次，即 0.01 s。substeps 与策略 decimation 不是同一概念。Lab SimulationContext 显式设置 stage 为 Z-up，不复制 Gym 的 up_axis 枚举值 1。

其中 4 直接来自原 `configs/tita_constraint_config.py:71`，不是从 substeps=1 推导出来的。详细执行循环、S=2 的假设示例与力矩更新边界统一维护在上述 Physics 章节。

`decimation` 表示一次策略动作对应的物理步数。本地 Lab 的 PhysX 通常执行路径中，一次 `env.step(action)` 先处理一次策略动作，再循环 4 次应用动作和推进物理，最后返回下一份观测及奖励。原 TITA 每个物理步还会重新计算反馈力矩，因此策略动作保持 10 ms，并不要求这 4 次力矩完全相同。对应源码：[ManagerBasedRLEnv.step](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:174)。

本地依据：[SimulationCfg](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/simulation_cfg.py:35)、[stage 约定](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py:261)。

## 3. 旧 sim.physx 中的资产参数需要另归资产配置

旧 Gym 源码：

```python
num_position_iterations = 4
num_velocity_iterations = 0
contact_offset = 0.01
rest_offset = 0.0
max_depenetration_velocity = 1.0
```

对应 Lab 3 的 API：

| 旧参数 | 当前 API | 配置位置 |
| --- | --- | --- |
| 位置/速度迭代次数 | PhysxArticulationCfg 的 solver_position_iteration_count / solver_velocity_iteration_count | TITA_CFG.spawn.articulation_props |
| contact_offset / rest_offset | PhysxCollisionCfg 的同名字段 | TITA_CFG.spawn.collision_props |
| max_depenetration_velocity | PhysxRigidBodyCfg 的同名字段 | TITA_CFG.spawn.rigid_props |

以上从 `isaaclab_physx.sim.schemas` 导入。PhysxCfg 中 min/max_iteration_count 是场景对 actor 迭代次数的限幅，不是同名替换的单一迭代次数。

当前旧 num_threads、default_buffer_size_multiplier、contact_collection 在所查 PhysxCfg / SimulationCfg 中没有直接对应的公开字段，不造同名参数填入。接触收集的实际需求在后续传感器配置时另行核对。

资产侧补充见同一归档中的 assets/tita.py 说明；这些参数位置核对不代表跨版本物理行为已验证。

## 4. VelocitySceneCfg：声明场景对象（2026-10-15）

历史源码（用户现已接入 TITA 与生成地形，最新状态见第 5 节）：

```python
@configclass
class VelocitySceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
```

`VelocitySceneCfg` 继承 `InteractiveSceneCfg`，描述地形、机器人、传感器和灯光等实体。实际创建由框架中的 `InteractiveScene(self.cfg.scene)` 完成；配置类本身不手写 Gym 的逐环境创建循环。上述历史版本只有平地、Cartpole、灯光三项；用户随后修改了机器人、地形并加入两种传感器，最新检查见第 5 节。

`robot` 字段名会成为 Scene 中的访问名称，后面的动作/观测等可通过 `asset_name="robot"` 或 `SceneEntityCfg("robot")` 指向它。`prim_path` 是 USD 场景路径；`{ENV_REGEX_NS}` 由 Scene 替换为多环境命名空间，使机器人分别出现在 `/World/envs/env_0/Robot` 等环境中。`/World/ground` 和 `/World/DomeLight` 则是共享的世界路径。

当前环境总装中的源码：

```python
scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=4.0)
```

这两个构造参数来自父类，控制场景实例数量与默认环境间距；不用在子类中重新声明。原 Gym 的 TITA 数量也是 4096；基础平地间距为 3 m，但默认 trimesh 使用地形平台原点，不能直接用间距值判断粗糙地形布局是否等价。

参数与行为来源统一维护在 [00 迁移映射索引](../../../../../../../../docs/migration/00_迁移映射索引.md) 的 Scene 各行。可先读 `_create_envs():164` 的资产加载部分，再读 `create_sim():843` 的地形分支；原默认地形是 trimesh，用户当前选择的 Lab 官方通用粗糙地形尚未对齐旧任务的具体分布。

同日后续核对：用户已导入 `from TITA_Lab.assets.tita import TITA_CFG`，并已采用下面的机器人引用（排版略作展开）：

```python
robot: ArticulationCfg = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

`replace` 复制既有机器人配置并指定当前场景的生成路径。机器人 USD、名义初始状态和物理属性继续由 `assets/tita.py` 定义。机器人引用和 `TerrainImporterCfg` 已由用户写入；传感器建议见下节，其他 MDP 配置仍为 Cartpole，未运行 RL 任务。

本地 API：[InteractiveSceneCfg](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene_cfg.py:22)、[Scene 的构造入口](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_env.py:175)、[官方速度任务 Scene](/home/eaziel/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:81)。

## 5. 当前地形与传感器复查（2026-10-15，后续核对）

当前 `terrain` 内的关键源码节选：

```python
terrain_type="generator",
terrain_generator=ROUGH_TERRAINS_CFG,
max_init_terrain_level=5,
collision_group=-1,
```

`TerrainImporterCfg` 负责把地形放进场景并提供环境原点；`terrain_generator` 负责生成具体地形几何。当前使用 Lab 官方通用 `ROUGH_TERRAINS_CFG`，包含台阶、方块、随机起伏和斜坡，API 接法正确。`physics_material` 设置摩擦/恢复等接触属性，`visual_material` 设置外观。

**原任务地形的新核实：** 旧 `utils/terrain.py:78` 的 `make_terrain()` 中，五类地形分支已全部注释；实际调用的是 `pyramid_stairs_terrain`，宽度参数 0.61 m、平台参数 3 m、台阶高度参数 `±0.5*(0.05+0.18*difficulty)`。课程生成时 `difficulty=row/num_rows`，10 行对应 0～0.9，前半列为负台阶高度。当前通用 ROUGH 预设可用于接通 Scene，但不能称为原 TITA 地形已对齐；下一步应按实际台阶规则配置，不照搬失效的五类比例注释。

用户加入高度扫描与接触传感器后，检查曾发现以下待修正字段。这是历史快照；用户现已修正扫描路径、网格点序并开启接触报告，最新说明见第 6 节。

```python
# height_scanner 内：
prim_path="{ENV_REGEX_NS}/Robot/base",
pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),

# contact_forces：
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True
)
```

当时的 `/Robot/base` 来自参考机器人，不是 TITA 的机身路径。TITA 机身为 `/Robot/Geometry/base_link`；默认传感器 spawner 可能在旧路径创建一个没有物理刚体祖先的 Xform，使扫描不随实际机身运动。用户现已将扫描器挂在实际机身下，并将原默认 xy 点序改为 yx。下面保留修正时采用的配置写法（排版与用户文件略有不同）：

```python
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

# 在 VelocitySceneCfg 中，放在 robot 之后、灯光之前：
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/height_scanner",
    update_period=0.01,
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(
        resolution=0.1,
        size=(1.6, 1.0),
        ordering="yx",
    ),
    mesh_prim_paths=["/World/ground"],
    debug_vis=False,
)
```

高度扫描是随机器人位置和航向移动的地形采样网格。旧 TITA 查询高度表，Lab 的射线传感器返回射线与地形的交点；这不是相机图像。x=-0.8～0.8、y=-0.5～0.5，间距 0.1 m，包含端点，共 17×11=187 点。`ordering="yx"` 保留旧 `torch.meshgrid(x,y)` 默认 ij 后展平的顺序；已抽取当前本地 `grid_pattern()` 在 CPU 执行，结果为 (187,3)，顺序与旧网格一致。

`ray_alignment="yaw"` 使网格随航向旋转，射线保持向下；`offset.pos.z=20` 是射线起点的高度偏移。扫描器 Xform 放在实际机身刚体下。运行后的 `data.ray_hits_w.torch` 预期为 (N,187,3)，最后一维是世界坐标 xyz；地形高度是其中的 z。旧观测的 `base_z-0.4-height`、裁剪和缩放留到 Observations/MDP 阶段。旧高度查询取三个相邻高度格的最小值，射线与实际三角网格求交并不逐点严格等价。

### 为什么末尾新加 /height_scanner（2026-10-15，用户追问）

本地 `RayCasterCfg` 的默认源码：

```python
spawn: SensorFrameCfg | None = SensorFrameCfg()
```

本地 `spawn_sensor_frame()` 创建节点的源码：

```python
prim = create_prim(
    prim_path,
    "Xform",
    translation=translation,
    orientation=orientation,
    stage=stage,
)
```

因此建议路径末尾的 `/height_scanner` 指定运行时新建的传感器坐标节点，不要求源 USD 已有该节点。父级 `Geometry/base_link` 是已有机身刚体，传感器节点是没有刚体/碰撞 API 的普通 Xform；PhysX 实现会向上寻找刚体祖先并读取其运动。这个名字由配置者选择，可以换成其他未占用的合法节点名。Python 字段 `height_scanner` 决定 `scene["height_scanner"]` 的访问名称；USD 路径末尾的同名部分只决定节点名字，两者没有强制同名关系。此前解释漏掉了默认 spawn 的创建步骤，不能让读者误以为在引用 USD 中已有的传感器。

这里确认的是显式新建子节点的配置与实现；未执行场景创建，不以字段注释对其他路径写法的说明代替运行验证。

| 参数 | 当前建议值的含义 |
| --- | --- |
| `prim_path` | 创建/跟踪扫描器坐标节点的 USD 路径；父级应位于实际机身下 |
| `update_period=0.01` | 以仿真时间计的 10 ms 更新间隔；实际刷新服从 Scene 更新和懒读取调度，不是额外的实时线程 |
| `offset.pos=(0,0,20)` | 当前实现将它加到局部射线起点上，再按 ray_alignment 变换；yaw 模式下向上抬高 20 m，不抬高机器人 |
| `ray_alignment="yaw"` | 网格跟随机身位置及航向，保持水平；射线方向保持世界系向下，不随机身滚转/俯仰倾斜 |
| `pattern_cfg` | 描述每条射线的起点与方向；GridPatternCfg 默认方向为 (0,0,-1) |
| `resolution=0.1` | 网格 x/y 方向相邻采样点间距 0.1 m |
| `size=(1.6,1.0)` | 网格长度/宽度；以中心为原点，x 为 -0.8～0.8，y 为 -0.5～0.5，含端点共 187 点 |
| `ordering="yx"` | y 索引在展平后变化更快，对齐旧 ij 网格的展开顺序；改变的是排列，不是采样区域 |
| `mesh_prim_paths` | 当前 PhysX 用于射线求交的静态地形路径，这里选择 /World/ground |
| `debug_vis=False` | 关闭采样交点的调试显示，传感器数据计算仍可运行 |
| `global_world_only=True` | 仅 Newton 有效；当前 PhysX 忽略此项，检测范围由 mesh_prim_paths 决定 |

接触传感器的 `history_length=3` 设置长度为 3 的力采样历史缓冲；`track_air_time=True` 开启接触/腾空持续时间跟踪。其 `update_period` 未显式设置时默认为 0，表示按物理步调度。开启 `activate_contact_sensors` 负责生成机器人时启用接触报告，ContactSensorCfg 负责组织读取到的数据，两者职责不同。

依据：[RayCasterCfg.spawn](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_cfg.py:39)、[SensorFrameCfg](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/spawners/sensors/sensors_cfg.py:369)、[spawn_sensor_frame](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sim/spawners/sensors/sensors.py:204)、[PhysX 刚体祖先与传感器 frame 跟踪](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/ray_caster/ray_caster.py:51)。

当时接触配置已存在，但尚缺报告开关；用户现已补上。保留历史与腾空跟踪选项的写法如下：

```python
# 接在 Scene 的 robot = TITA_CFG.replace(...) 后：
robot.spawn.activate_contact_sensors = True

contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/[^/]*",
    update_period=0.0,
    history_length=3,
    track_air_time=True,
    debug_vis=False,
)
```

第一行在生成机器人时开启刚体接触报告；第二部分在 Scene 中登记读取数据的传感器。此前 USD 静态检查为 9 个刚体、0 个磁盘 PhysxContactReportAPI，spawn 开关当时为默认 False；用户现已将开关写为 True。报告 API 将在生成时加入已加载场景，不要求用户另外回写源 USD。

用户的 `"/Robot/[^/]*"` 可以保留：当前 PhysX ContactSensor 实现将路径拆为父表达式和叶名表达式，再遍历父节点下已有 contact reporter 的后代，使用逐刚体路径列表创建 view，明确支持嵌套刚体。此处的解析逻辑与 UsdFileCfg 属性字典的整路径匹配不同，不能只按正则字面误判为仅能读取直接子节点。按当前 USD，预期覆盖机身和左右各 4 个腿/轮刚体，共 9 个。

`history_length=3` 为物理采样历史，不等于旧环境的 100 帧策略接触缓冲。`track_air_time=True` 的默认判定依据是接触力范数阈值；旧代码按轮子 Fz>1 N 再与上一策略帧取 OR。因此当前选项可以保留用于传感器能力验证，MDP 迁移时仍需核对判定、滤波和计时频率。

本地 Lab 3 / PhysX 应明确读取 `data.net_normal_forces_w.torch`，预期形状 (N,9,3)，单位 N、世界坐标系。这个接口是聚合接触法向力；不是关节内部受力，也不能直接宣称包含全部摩擦分量。当前 `net_forces_w` 在 PhysX 上是带警告的法向力兼容别名。轮子六维 `force_sensor_tensor` 在旧环境类内只见获取、刷新与包装，当前观测和奖励主要使用 `contact_forces`；暂不额外迁移六维传感器。

验证边界：当前文件的 Python AST 通过，用户已写入两种传感器并完成三处参数修正；此前 USD 静态匹配 9 个刚体、真实扫描父节点存在，参考路径 `/TITARobot/base` 不存在。仅网格生成函数已在 CPU 执行。助手未修改任务源码，未启动仿真、绑定传感器或验证上述运行时张量。用户要求真机接触力来源以后再讨论，本轮未检索该问题。

依据：[GridPatternCfg](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns_cfg.py:34)、[ContactSensorCfg](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_cfg.py:19)、[PhysX 嵌套接触目标解析](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py:351)、[旧高度查询](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1051)、[旧地形实际生成](/home/eaziel/Downloads/tita_rl-master/utils/terrain.py:78)。

## 6. 配置项与运行时数据属性（2026-10-15，三处修正已落盘）

当前源码节选：

```python
robot.spawn.activate_contact_sensors = True

# height_scanner 内：
prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link/height_scanner",
update_period=0.01,
# GridPatternCfg 内：
ordering="yx",
```

用户已完成上述配置修正。`net_normal_forces_w` 则不是一个待补的配置字段，而是 Lab 运行时数据对象已经定义的属性。下面是已初始化环境中的读取示例：

```python
contact_sensor = env.scene["contact_forces"]
forces_w = contact_sensor.data.net_normal_forces_w.torch

scanner = env.scene["height_scanner"]
terrain_z = scanner.data.ray_hits_w.torch[..., 2]
```

`ContactSensorCfg` 描述如何创建传感器；`env.scene["contact_forces"]` 才是实际传感器对象；`.data` 返回数据对象；`.net_normal_forces_w` 取其已有 ProxyArray 属性，`.torch` 提供 Torch 视图。后续 MDP 函数读取并加工这些值，不能把读取表达式放在配置类中提前求值，也不用往 ContactSensorCfg 中添加同名参数。

跨文件的系统说明、数据形状、法向/摩擦含义及 Gym 差异统一维护在 [02 Scene 迁移](../../../../../../../../docs/migration/02_scene迁移.md)。当前配置已静态核对，运行时绑定和张量尚未仿真验证。

## 7. 接口名与物理量：聚合法向力（2026-10-16）

已初始化环境中的读取示例：

```python
forces_w = env.scene["contact_forces"].data.net_normal_forces_w.torch
vertical_force = forces_w[..., 2]
resultant_magnitude = torch.linalg.vector_norm(forces_w, dim=-1)
```

`net_normal_forces_w` 按身体分别汇总当前各接触点的法向力向量，以世界坐标表示。`vertical_force` 只取世界 z 分量；合力模会同时考虑 xyz。法向是相对每个接触面定义的，墙面法向可以水平、斜坡法向可以同时有 x/z 分量。

“聚合”不是把所有身体或所有历史帧一起相加。示意：同一轮子的两个地面接触点各向上 60 N、40 N，则汇总为 (0,0,100) N；若再受墙面 (-30,0,0) N 法向力，汇总为 (-30,0,100) N。身体维仍分别保留，预期完整数组 (N,9,3)。详细接口—物理过程对应表及用途已更新到 02 章第 2.1/4.1 节。

## 8. 材质导入、Any 与恢复系数（2026-10-16）

此前源码节选（现已被用户替换为直接导入并使用 PhysxRigidBodyMaterialCfg，恢复系数显式 0.0；以下保留问题由来）：

```python
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

# VelocitySceneCfg.terrain 内：
physics_material=sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
    static_friction=1.0,
    dynamic_friction=1.0,
),
```

`RigidBodyMaterialCfg` 是直接导入的名字，而 `sim_utils.RigidBodyMaterialCfg` 是通过模块对象取属性；后者不会算作使用前者，所以未使用提示成立。颜色变淡与未使用有关，Any 另涉及静态类型解析，二者需分开看。

本地旧材质名走动态兼容转发，materials 包的类型存根未静态导出它；PhysX 包则明确导出 `PhysxRigidBodyMaterialCfg`。从 `isaaclab_physx.sim.spawners.materials` 导入并在构造处实际使用 `PhysxRigidBodyMaterialCfg(...)`、删除旧名字的多余导入，已由用户完成；助手未修改 Python 或验证编辑器提示变化。

`restitution` 从材质基类继承，默认 0.0，与旧 Gym 地面配置一致；显式添加 `restitution=0.0` 只是使配置更直观。`restitution_combine_mode="multiply"` 是两材料的系数合成规则，不是恢复系数本身。完整建议片段、模式优先级及证据集中维护在 [02 Scene 迁移第 2.2 节](../../../../../../../../docs/migration/02_scene迁移.md)。

## 9. ActionsCfg 迁移方案（2026-10-16，历史建议；当前已落盘，见第 10 节）

当时仍为模板源码，以下保留历史节选；用户现已替换，见第 10 节：

```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)
```

这里的类是给 ActionManager 读取的一组条目。配置对象的字段决定资产、关节、动作缩放和目标类型；不是在 class 定义时执行控制循环。TITA 应替换掉 slider_to_cart，通过现成 JointPositionActionCfg 选择六个腿关节，通过 JointEffortActionCfg 选择两轮。

本轮名义建议：腿 use_default_offset=True，leg_1 动作缩放 0.25、leg_2/3 缩放 0.5；轮子前馈缩放 5 N·m，并在 tita.py 配 IdealPDActuatorCfg 的轮子 stiffness=0、damping=0.5。ActionManager 先取前六维作为腿位置动作，再取后两维作为轮子前馈；preserve_order=True 使各项内按关节名列表排序。完整替换片段、腿轮实际公式和当前 API 定位见 [03 Actions](../../../../../../../../docs/migration/03_actions迁移.md)。

旧 Gym 还开启滤波、物理步延迟和随机因子；这些尚未由本段名义配置复现。Actions 写完也不能直接运行完整任务：当前其他管理器仍有 Cartpole 关节名。助手只维护文档，未替换 env_cfg.py 或 tita.py，未运行仿真。

## 10. 当前 ActionsCfg 检查与二维动作理解（2026-10-16）

当前用户实际源码：

```python
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
```

已静态核对两项的类型、关节名与列序、preserve_order、默认偏置、0.25/0.5/5.0 缩放：共 6+2 维，按当前 USD 无重复且完整覆盖 8 个关节。两份 Python 可被 AST 解析；这不是创建环境或运行控制验证。

策略动作张量为 (num_envs,8)：每行是一个环境，每列是一路控制信号，没有历史时间维。配置将前六列送给 leg_position、后两列送给 wheel_effort；前者写位置目标，后者写力矩前馈，IdealPD 执行器再结合当前状态计算实际力矩。因此同为 (N,8) 的 actions 和 torques 具有不同含义。

旧 Gym 的 `actions[:, :8]` 选所有环境的前八列，`actions_scaled[:, [0,4]] *= 0.5` 把旧布局的两列再缩放并写回；不能照搬到当前新 6+2 布局。当前髋列为 0/3、轮子列为 6/7，按名字的 scale 字典已正确处理。

scale 是数值到物理量的任务映射，并非 RL 必需步骤；也不是裁剪。旧 use_filter 为两点 FIR，lag 缓冲在物理步更新，motor/kp/kd 因子改变实际控制，推机器人归 Event 而非 Command。详细源码、数值例和时间含义集中在 [03 Actions 第 7 节](../../../../../../../../docs/migration/03_actions迁移.md)。

此前提示的 default 残留 NewtonCfg 类型注解，最新文件已由用户改为 PhysxCfg。助手未修改 Python；其他 Cartpole 管理器仍待迁移。

## 11. CommandsCfg 与覆盖状态（2026-10-16，基础已落盘）

当前 VelocityEnvCfg 的已有管理器字段节选：

```python
observations: ObservationsCfg = ObservationsCfg()
actions: ActionsCfg = ActionsCfg()
commands: CommandsCfg = CommandsCfg()
events: EventCfg = EventCfg()
```

前次为建议，本轮已核实用户加入 CommandsCfg/commands 字段。以下为当前配置，省略用户的行尾注释：

```python
@configclass
class CommandsCfg:
    """Velocity and heading command specifications for TITA."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_heading_envs=1.0,
        rel_standing_envs=0.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
        debug_vis=True,
    )
```

当前已在 VelocityEnvCfg 内登记：

```python
commands: CommandsCfg = CommandsCfg()
```

CommandsCfg 是配置容器，base_velocity 是 CommandManager 的条目键。UniformVelocityCommandCfg 描述每环境的采样周期、速度范围与航向外环，运行时对象维护 (N,3) 的机身速度目标和单独的 heading_target。后续 Observations/Rewards 通过相同条目名取目标，不靠 Command 直接设置机器人实际运动。

此配置保留旧任务 10 s 重采样、速度范围、航向误差比例系数及 yaw_rate 限幅；原平移模长≤0.2 m/s 的置零规则在标准类中没有对应字段，已明确登记为后续专用 CommandTerm 待补。官方示例 rel_standing_envs=0.02 并不等价，因此这里使用 0。

Actions 表中的 domain_rand 是参数来源索引：旧控制计算依赖其中的强度、增益和延迟；不表示全部域随机化已迁移。当前 TITA 的摩擦、恢复、质量、质心、推扰及这些控制随机化均尚未迁入。覆盖状态与各项落点集中在 [00 索引](../../../../../../../../docs/migration/00_迁移映射索引.md)，Commands 的参数、运行实现与待补规则在 [04 章](../../../../../../../../docs/migration/04_commands迁移.md)。助手未修改 Python 或运行仿真。

当前机制补充（2026-10-16，未改配置）：旧 _resample_commands 为 env_ids 指定的一批环境抽取新 vx/vy/heading 任务目标；_post_physics_step_callback 是由主流程显式调用的普通方法，每个策略步执行，内部仅在周期满足时换任务，但每步根据当前朝向重算 yaw_rate。10 s 目标保持、10 ms 策略/航向反馈、2.5 ms 关节控制与物理属于不同时间尺度。目标前三列同时进入策略观测和速度追踪奖励，第四列朝向目标由环境先换为目标转速。完整单环境示例、回调顺序和已确认的 Lab 调度差异统一维护在 [04 Commands 第 1/2 节](../../../../../../../../docs/migration/04_commands迁移.md)。

本轮字段核实（2026-10-16）：heading_command=True 启用航向生成规则，rel_heading_envs 是每次重采样时选择航向模式的概率，rel_standing_envs 是选择全部速度目标归零的概率。当前 1.0/0.0 对应全部航向模式且无额外静止抽样。配置通过 class_type 绑定 Lab 的 UniformVelocityCommand，实际转速计算在其 _update_command 中，不需要在本文件重复写。调用链和旧回调各项的去向统一见 [04 章 2.5/4.1](../../../../../../../../docs/migration/04_commands迁移.md)。本轮 AST 核对了基础参数、Ranges 与 commands 登记；未实例化环境。

## 12. ObservationsCfg 起点（2026-10-16，新增三项已落盘）

前一阶段五项快照：当时 ObservationsCfg 内部嵌套的 PolicyCfg 有以下条目和组设置。当前六项进展见第 13 节，不再将本快照当作最新状态：

```python
base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)

projected_gravity = ObsTerm(func=mdp.projected_gravity)

velocity_commands = ObsTerm(
    func=mdp.generated_commands,
    params={"command_name": "base_velocity"},
    scale=(2.0, 2.0, 0.25),
)

joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

def __post_init__(self) -> None:
    self.enable_corruption = False
    self.concatenate_terms = True
```

本轮已核实用户写入新增三项；五项的函数、缩放、command_name 及声明顺序通过 AST 静态检查。base_ang_vel 是机身系角速度，projected_gravity 是单位重力方向在机身系的表示，velocity_commands 读取 CommandsCfg 的 base_velocity 条目；新增三项各三维，关节两项按八关节各八维，推算当前 policy 为 (N,25)，未运行验证。

func 字段保存函数对象，ObservationManager.compute_group 运行时调用函数、clone 结果并应用 scale；concatenate_terms=True 使管理器在最后按默认 dim=-1 用 torch.cat 沿列拼接。clone 之后的缩放不会修改原状态/目标；enable_corruption=False 暂不加入观测噪声。policy 是外层字段登记的组名，管理器返回 obs["policy"] 这组张量。

每项物理含义、重力归一化证据、数值例子、列区间和管理器源码统一维护在 [05 Observations 第 2 节](../../../../../../../../docs/migration/05_observations迁移.md)。基础 33 维仍需关节排列/轮角屏蔽与上一动作等对齐，历史、噪声及特权输入也未完成；助手未修改用户 Python 或运行仿真。

## 13. 关节项与上一动作（2026-10-16，已落盘）

用户已加入八关节常量、索引配置、0.05 缩放和上一动作；速度引用现已改成 mdp.joint_vel。以下保留屏蔽轮角之前的基础版本，当前位置条目已切换自定义函数，见第 14 节：

```python
joint_pos_rel = ObsTerm(
    func=mdp.joint_pos_rel,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot", joint_names=TITA_JOINT_NAMES, preserve_order=True
        ),
    },
)
joint_vel = ObsTerm(
    func=mdp.joint_vel,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot", joint_names=TITA_JOINT_NAMES, preserve_order=True
        ),
    },
    scale=0.05,
)
actions = ObsTerm(func=mdp.last_action)
```

TITA_JOINT_NAMES 已在模块级、imports 后定义：左腿1/2/3、右腿1/2/3、左轮、右轮，与当前动作顺序一致；完整列表及原始计算入口见 [05 章第 4 节](../../../../../../../../docs/migration/05_observations迁移.md)。不要将普通列表放进 PolicyCfg 作为观测字段。

asset_cfg 传给函数，由管理器将关节名称一次解析为 joint_ids；preserve_order=True 令结果按名称列表排列。joint_pos_rel 返回相对默认角，joint_vel 返回实际速度，再乘 0.05；当前默认速度为零，改用实际速度是为了直接对应旧 dof_vel 计算。last_action 取环境接收到的完整八维动作，未填 action_name，因此包含两种动作项；在下一次观测中，它记录刚执行的输入，未经过动作项缩放。

当前已有六项，预期共 33 维；速度引用与轮角屏蔽均已接入，不能把相同维数视作完整观测已对齐。保持当前 enable_corruption=False，本轮未运行仿真。

## 14. rel 参考值与轮角屏蔽接入（2026-10-16，已落盘）

两个标准函数的源码要点是：

```python
# joint_vel
return asset.data.joint_vel.torch[:, asset_cfg.joint_ids]

# joint_vel_rel
return (
    asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    - asset.data.default_joint_vel.torch[:, asset_cfg.joint_ids]
)
```

这里 rel 是相对默认值。当前默认速度参考为零，来自 InitialStateCfg.joint_vel，未被 tita.py 覆盖；例如实际速度 2 rad/s，两种函数都返回 2，含义不是机器人实际静止。位置则以 tita.py 的 0/0.8/-1.5/0 为参考。

当前的位置条目为（省略行尾注释与部分排版差异）：

```python
joint_pos_rel = ObsTerm(
    func=mdp.joint_pos_rel_zero_wheels,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot", joint_names=TITA_JOINT_NAMES, preserve_order=True
        ),
        "wheel_cfg": SceneEntityCfg(
            "robot", joint_names=["joint_left_leg_4", "joint_right_leg_4"]
        ),
    },
)
```

此片段已由用户写入，两个 SceneEntityCfg 名称与 preserve_order、函数引用及导出已静态核对；前者指定返回顺序，后者指定需要置零的两轮。保存的函数通过了独立 CPU 张量检查，仍返回八维位置，整组预期 33 维。当前只发现新函数返回注解 torch.tensor 应改 torch.Tensor；函数本体与验证边界见 [mdp/observations.py 的解释](../../mdp/observations.py.md)，导出方式见 [mdp/__init__.pyi 的解释](../../mdp/__init__.pyi.md)。未运行环境。

</details>
