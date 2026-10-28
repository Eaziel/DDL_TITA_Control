# 06 Events：初始状态、物理参数与推扰

**目标与结果：** 明确机器人每回合从什么状态开始，以及哪些参数在启动时改变、哪些扰动定时发生。

整理：2026-10-27。startup参数、reset初值、interval推扰均已接线。新版后端判断已适配，Sim6.2首次任务初始化/zero运行片段已有；质量/COM/材质/初值接口核查可沿用。新旧短运行均无15秒推扰和20秒超时的明确日志，当前证据与整体状态见[项目记忆](../../PROJECT_MEMORY.md)、[新版核查](../audits/2026-10-27_sim62_first_run/README.md)。

## 1. Gym怎样改变环境

旧代码把三种时机分散在不同函数中：创建机器人时设置质量、质心、材质和电机因子；reset_idx重置关节与根状态；回调每15 s扰动速度。Lab用EventTerm统一声明触发时机，具体计算由所选函数或类执行。

| mode | 当前用途 | 发生时机 |
| --- | --- | --- |
| startup | 机器人之间的固定参数差异 | 环境启动、物理对象就绪后 |
| reset | 新回合的姿态、位置和速度 | 被选环境重置时 |
| interval | 运动过程中推扰 | 按仿真时间计时 |

startup随机因子跨回合保留；reset不会把它们重抽一次。

## 2. Lab的完整配置

```python
@configclass
class EventCfg:
    """Reset TITA joints and base state at each episode."""

    randomize_material = EventTerm(
        func=mdp.RandomizeTitaMaterial,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_range": (0.2, 2.75),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
        },
    )

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

    randomize_motors = EventTerm(
        func=mdp.randomize_tita_motor_factors,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "motor_strength_range": (0.8, 1.2),
            "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2),
        },
    )

    randomize_action_lag = EventTerm(
        func=mdp.randomize_tita_action_lag,
        mode="startup",
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
```

VelocityEnvCfg.events选择EventCfg；管理器按字段名记录条目，再按mode调用。SceneEntityCfg("robot")选择整台机器人，body_names="base_link"只选择基座。名字由管理器解析成实际索引。

## 3. 每个新回合从什么状态开始

Gym的_reset_dofs使用默认角乘U(0.5,1.5)，关节速度置0。Lab的reset_joints_by_scale做同类缩放，并额外夹到soft limits；当前采样区间在限位内，因此这次夹紧不会改变分布：

```text
hip：0；第二关节：[0.4,1.2] rad；第三关节：[-2.25,-0.75] rad；轮：0
八关节初速度：0 rad/s
```

粗糙地形中，Gym根位置为默认位置加地形原点，xy再各加±1 m，z再加[0,0.2] m；线/角速度六分量各为±0.5，姿态用固定默认四元数。Lab的reset_root_state_uniform按上述pose_range和velocity_range完成。当前Lab四元数为(x,y,z,w)，默认(0,0,0,1)。

平地分支的旧Gym没有额外xy随机化。当前使用ROUGH，上述配置对应粗糙地形；以后切平地时要单独决定xy范围。

## 4. 创建时改变质量与质心

旧_process_rigid_body_props只改基座：质量增加U(-1,3) kg，局部COM各轴加U(-0.1,0.1) m。当前基座静态质量约13.2 kg，结果约12.2～16.2 kg。

Lab质量项从默认质量抽样，recompute_inertia=True按I_new=I_default*(m_new/m_default)缩放惯量；COM项从默认局部COM加偏移，保留惯性坐标系朝向。这是本地API明确执行的模型。旧Gym的recomputeInertia内部重建没有做二进制复现，不能据参数同名断言惯量完全相同。

## 5. 创建时随机材质

旧_process_rigid_shape_props为摩擦和恢复系数分别建立64个候选，每台机器人各选一个，全身碰撞形状共享。两组选择独立，静动摩擦相同，跨回合保留。Lab使用一个合并类实现：

```python
class RandomizeTitaMaterial(ManagerTermBase):
    """Assign shared friction and restitution to all collision shapes of each robot."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv) -> None:
        # 该实现依赖 Kit，等仿真启动、事件实例开始构造时再导入。
        from isaaclab_physx.physics import PhysxManager

        super().__init__(cfg, env)
        if not isinstance(env.sim.backend_manager, PhysxManager):
            raise ValueError("RandomizeTitaMaterial supports Isaac Sim PhysX only.")

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        if asset_cfg.body_ids != slice(None):
            raise ValueError("Select the entire robot without a body subset.")
        self.asset: Articulation = env.scene[asset_cfg.name]

        num_buckets = cfg.params["num_buckets"]
        friction_range = cfg.params["friction_range"]
        bucket_ids = torch.randint(0, num_buckets, (env.num_envs,), device="cpu")
        friction_buckets = sample_uniform(friction_range[0], friction_range[1], (num_buckets,), device="cpu")
        self.friction_coeffs = friction_buckets[bucket_ids]

        restitution_range = cfg.params["restitution_range"]
        restitution_ids = torch.randint(0, num_buckets, (env.num_envs,), device="cpu")
        restitution_buckets = sample_uniform(restitution_range[0], restitution_range[1], (num_buckets,), device="cpu")
        self.restitution_coeffs = restitution_buckets[restitution_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
    ) -> None:

        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device="cpu", dtype=torch.int32)
        else:
            env_ids = env_ids.to(device="cpu", dtype=torch.int32)

        view: ArticulationView = self.asset.root_view

        materials = wp.to_torch(view.get_material_properties()).clone()
        friction = self.friction_coeffs[env_ids].unsqueeze(-1)
        restitution = self.restitution_coeffs[env_ids].unsqueeze(-1)
        materials[env_ids, :, 0] = friction
        materials[env_ids, :, 1] = friction
        materials[env_ids, :, 2] = restitution
        view.set_material_properties(
            wp.from_torch(materials, dtype=wp.float32),
            wp.from_torch(env_ids, dtype=wp.int32),
        )
```

构造函数保存每个环境的mu_i和e_i；调用时将材质表(N,S,3)的三列写为[mu_i,mu_i,e_i]。S是后端形状数。env_ids选择行，unsqueeze(-1)让每环境一个数广播到其所有shape。get返回Warp数组；to_torch共享存储，clone创建工作副本，set再将结果提交后端。

Scene为机器人明确绑定独立物理材质配置，初始同地面[1,1,0]，双方combine mode均为average；startup只修改robot。接触对为mu_pair=(mu_robot+1)/2、e_pair=e_robot/2。copy复制初始配置，实际随机化范围由asset_cfg决定。旧Gym摩擦有average依据；恢复合成仍是迁移选择，未复现旧二进制。

恢复系数描述法向反弹，摩擦限制切向相对运动；bounce_threshold_velocity还会影响低速碰撞是否使用恢复。报错文字仍含旧RandomizeTitaFriction名字，仅影响提示，可在后续顺手更正。

## 6. 每15秒扰动实际速度

```python
def push_by_setting_xy_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Replace world-frame horizontal root COM velocity with random samples."""

    asset: Articulation = env.scene[asset_cfg.name]
    selection = slice(None) if env_ids is None else env_ids
    velocity_w = asset.data.root_vel_w.torch[selection].clone()
    velocity_w[:, :2] = sample_uniform(
        velocity_range[0],
        velocity_range[1],
        (velocity_w.shape[0], 2),
        device=asset.device,
    )
    asset.write_root_velocity_to_sim_index(root_velocity=velocity_w, env_ids=env_ids)
```

旧_push_robots覆盖实际世界系vx/vy；当前函数保持相同赋值语义，保留vz与三个角速度。原生push_by_setting_velocity使用增量，不能直接当成相同计算。

is_global_time=True使所有环境共用计时器，部分reset不重新开始。原生计时器CPU检查在1500、3000个10 ms步触发。旧Gym在奖励/终止之前推扰，Lab默认interval位于奖励/reset之后；调用位置差异仍保留。

## 7. 电机随机化如何接上

randomize_tita_motor_factors在startup填入每组执行器的motor_strength、kp_factor、kd_factor。完整函数、力矩公式和两组配置统一放在[03 Actions](03_actions迁移.md#3-电机强度与kpkd因子)，这里不重复一份控制模型。

## 8. 验证与源码

原生质量/COM类、原生reset EventManager、全局timer以及任务材质/推扰函数的CPU结果见[09审查](09_阶段审查.md)。写入端使用Tensor/Warp替身，未推进物理。

Gym：[创建参数处理](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:558)、[reset_idx](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:734)、[推扰](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1089)。Lab：[MDP events](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/events.py)、[EventManager](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/event_manager.py)。Torch/Warp接口详细问答见[events.py说明](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/events.py.md#补充问答)。

## 补充问答

**Q：** 我想问了，“w”到底表示啥？？接触力有个w，这里还有w！

**A：** 末尾_w表示world，世界坐标系。root_vel_w的线速度/角速度与net_normal_forces_w的力都沿世界坐标轴分解；机身相关接口的_b表示沿body坐标轴表达。世界轴不跟随机器人旋转，机身轴会跟随机器人旋转。

例如假设世界+x向东、+y向北、机身+x为前方；机器人朝北前进1 m/s，同一速度在世界系为(0,1,0)，机身系为(1,0,0)。此前列出的wx/wy/wz是代写角速度ωx/ωy/ωz，和后缀_w的world不同。原生接口定义与变量命名解释见[events.py问答](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/events.py.md#补充问答)。

**Q：** 而且我没搞懂，为啥要robot.spawn.physics_material = terrain.physics_material.copy()给robot.spawn.physics_material？？这不就机器人和地面材质一样了吗？？

**A：** 对，场景创建时先得到相同的初始接触参数[1,1,0]；copy创建新的配置对象。startup的SceneEntityCfg("robot")只选择机器人，随后将其三列改为[mu_i,mu_i,e_i]；地面保留[1,1,0]，双方average规则保留。随机化写的是机器人后端材质表，没有回头修改地面的Python配置。spawn负责创建时的材质生成和碰撞体绑定；完整对象层次与补全提示见[env_cfg.py问答](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/config/Tita/env_cfg.py.md#补充问答)。
