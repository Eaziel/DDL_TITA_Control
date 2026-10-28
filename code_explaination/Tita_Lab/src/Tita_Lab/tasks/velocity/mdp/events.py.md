# mdp/events.py：推扰、材质与电机因子采样

**阅读目标：** 理解事件怎样在startup/reset/interval触发时改变参数或状态，并追踪随机值实际写进哪个对象。

> 2026-10-20已复核材质、质量/COM、推扰与因子接线；当前CPU证据统一见[09](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/docs/migration/09_阶段审查.md)。

源码目录：`Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/`。
源码文件：[events.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/events.py)。更新：2026-10-21。材质、电机因子、共享延迟和推扰前采样均已保存。当前CPU流程通过；首次真实日志仅覆盖较早无延迟基线，15 s真实推扰与参数效果未评估。

```python
def push_by_setting_xy_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Replace world-frame horizontal root COM velocity with random samples."""

    # Preserve the physics-result sample before this event changes root velocity.
    env.observation_manager.compute_group("privileged_state")

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

## 1. 配置提供调用条件，函数执行物理状态修改

EventCfg.push_robot选择函数、范围、15 s间隔和全局计时；EventManager负责保存配置、计时并调用函数。这个普通函数只在被调用时读取速度、修改xy并写回，不持有跨步缓存，也不需要自己实现reset。

参数env是当前任务环境；env_ids选择本次处理哪些环境；velocity_range是两个水平分量共用的采样区间；asset_cfg.name指定Scene资产。返回类型None表示函数通过写入仿真产生作用，不返回奖励或观测。

`TYPE_CHECKING`导入用于类型提示，future annotations延后注解解析；运行时真正使用的是torch和sample_uniform。asset的Articulation注解帮助编辑器理解机器人对象，`env.scene[asset_cfg.name]`实际取得已经创建的对象。

## 2. 选择环境、复制数据

```python
selection = slice(None) if env_ids is None else env_ids
velocity_w = asset.data.root_vel_w.torch[selection].clone()
```

全局interval调用传env_ids=None。这里将None转换成slice(None)用于取张量的全部行，避免直接tensor[None]增加一维。若传入索引张量，则只取指定环境。

root_vel_w是根刚体质心的世界系速度，torch视图为(N,6)，每行[vx,vy,vz,ωx,ωy,ωz]。选中M个环境后得到(M,6)；全量时M=N。clone复制出独立数据用于修改，最后再通过写入API提交。

## 3. 两列覆盖，其余列保留

```python
velocity_w[:, :2] = sample_uniform(
    velocity_range[0],
    velocity_range[1],
    (velocity_w.shape[0], 2),
    device=asset.device,
)
```

sample_uniform接收下界、上界、输出shape和设备。这里返回(M,2)，各行和各分量独立均匀取样，赋给前两列vx/vy。其余四列保留原来的vz和三个角速度。

使用赋值意味着新vx直接等于抽到的数。旧vx=2、抽到0.3时，新vx=0.3；不是2.3。当前本地官方push_by_setting_velocity使用+=，因此本任务用这个小函数对应Gym的直接赋值。

## 4. 写回物理后端

```python
asset.write_root_velocity_to_sim_index(root_velocity=velocity_w, env_ids=env_ids)
```

PhysX接口接收(M,6)的所选环境数据；env_ids=None表示全量，否则按给定索引更新。本次写入改变实际速度，Commands中的目标速度不变。该方法针对根刚体质心，线速度单位m/s，角速度单位rad/s，均在世界系。

## 5. 如何接入与检查

通过任务`mdp/__init__.pyi`导出函数，再由EventCfg.push_robot引用。完整三处代码、旧Gym规则、global计时及执行顺序差异见[06第6节](../../../../../../../docs/migration/06_events迁移.md#6-每15秒扰动实际速度)。

当前推扰全量/部分选择及原生interval/reset计时器CPU检查见09；写入端为替身，真实推扰未运行。

## 6. 已保存：RandomizeTitaMaterial类

以下为当前保存的类源码。其职责为创建时给每台机器人独立采样一对系数，全身碰撞形状共用，跨回合保留。2026-10-27后端实例检查与延后导入已由用户写入，新环境首次运行片段已提供。

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

__init__先抽摩擦编号与候选，再独立抽恢复系数编号与候选，缓存各为(N,)。__call__接收EventManager传来的params，将所选环境编号准备成CPU int32，取机器人root_view的(N,S,3)材质表；两组系数扩成(M,1)，广播写入静摩擦、动摩擦、恢复系数三列。

view是self.asset.root_view的带类型注解别名。wp.to_torch提供共享存储的Torch视图，clone创建独立修改副本，wp.from_torch包装修改后的数据，setter提交后端。具体类型与导入问题见本页Q/A。

当前asset_cfg.name为robot，只写机器人；Scene复制初始地面材质配置不改变这个选择范围。两边初始参数相同，startup后地面保持[1,1,0]，机器人变成[mu_i,mu_i,e_i]，合成规则仍均为average。

旧环境材质采样与写入范围检查见09；新版后端实例检查、导入时机和首次运行见本页适配节。类型注解导入仍为omni.physics.tensors.api。

## 7. 已保存：电机因子采样与startup参数

2026-10-20当前函数和下述导入已保存；函数AST对应此前候选，EventCfg.randomize_motors及四项params已补齐。普通导入区已有：

```python
from ..actuators import TitaPDActuator
```

当前函数：

```python
def randomize_tita_motor_factors(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    motor_strength_range: tuple[float, float],
    kp_range: tuple[float, float],
    kd_range: tuple[float, float],
) -> None:
    """Sample independent motor factors for every environment and joint."""
    if asset_cfg.joint_ids != slice(None):
        raise ValueError("Select the entire robot without a joint subset.")

    asset: Articulation = env.scene[asset_cfg.name]
    selection = slice(None) if env_ids is None else env_ids
    for actuator in asset.actuators.values():
        if not isinstance(actuator, TitaPDActuator):
            raise TypeError("Motor-factor randomization requires TitaPDActuator groups.")
        for factors, bounds in (
            (actuator.motor_strength, motor_strength_range),
            (actuator.kp_factor, kp_range),
            (actuator.kd_factor, kd_range),
        ):
            factors[selection, :] = sample_uniform(
                bounds[0], bounds[1], factors[selection, :].shape, device=asset.device
            )
```

asset_cfg按Scene名称选择机器人。本函数只实现整台机器人的控制因子随机化，所以joint_ids必须为slice(None)；两个分组都必须已经配置为TitaPDActuator，否则报错，避免默默漏掉一组。

selection在env_ids=None时为slice(None)，代表所有行。第二维的冒号保留整组关节：腿(N,6)、轮(N,2)，部分环境时第一维变成所选数量。外层循环取执行器实例；内层循环把“某个因子张量”和它的范围配成二元组，分三次调用sample_uniform；张量元素分别抽样，三个因子彼此独立。

赋值直接改变执行器持有的Tensor。后续compute读取同一个实例的这三份数据，所以无需写PhysX视图或修改求解器增益。EventCfg使用startup，只抽一次；本函数没有reset缓存。TitaPDActuator继承的原生reset也不改这些固定参数。

完整Scene/导出/Event配置与旧公式见[03第4.4节](../../../../../../../docs/migration/03_actions迁移.md#3-电机强度与kpkd因子)。与真实原生执行器和旧Gym方法的128环境CPU对照通过，未运行完整任务或仿真。当前文件AST已核对，startup参数接线也已保存；此前CPU结果不代表真实环境已运行。

## 8. 已写入：共享动作延迟索引

普通导入增加.actions.TitaJointAction；TYPE_CHECKING增加..tita_velocity_env.TitaVelocityEnv。当前函数如下：

```python
def randomize_tita_action_lag(env: TitaVelocityEnv, env_ids: torch.Tensor | None) -> None:
    """Sample a shared action-history slot for each robot."""
    legs = env.action_manager.get_term("leg_position")
    wheels = env.action_manager.get_term("wheel_effort")
    if not isinstance(legs, TitaJointAction) or not isinstance(wheels, TitaJointAction):
        raise TypeError("Action lag randomization requires TitaJointAction terms.")
    if legs.cfg.lag_timesteps != wheels.cfg.lag_timesteps:
        raise ValueError("Leg and wheel command histories must have the same length.")

    selection = slice(None) if env_ids is None else env_ids
    # Write in place: both action terms retain references to this (N,) tensor.
    env.action_lag_indices[selection] = torch.randint(
        0,
        legs.cfg.lag_timesteps,
        env.action_lag_indices[selection].shape,
        device=env.device,
    )
```

动作项按leg_position/wheel_effort字段名取得并检查类型与缓存长度；随机整数写入环境共享张量，形状(N,)，上界不含在采样值中。不能重新绑定env.action_lag_indices，因为两个动作项已经持有原张量引用。

EventCfg以mode="startup"登记该函数，每台机器人创建后抽一次。reset只恢复目标历史，不再次采样。2026-10-21该函数、startup配置与.pyi已保存；原生EventManager候选CPU检查与完整接线见[12](../../../../../../../docs/migration/12_动作延迟迁移.md)。

## 9. 已写入：推扰前保存六维观测

push_by_setting_xy_velocity开头新增：

```python
# 先保存本步速度与接触，再改写推扰速度。
env.observation_manager.compute_group("privileged_state")
```

旧任务先计算base_lin_vel，再在回调中覆盖根速度。新增组在本次速度改写前采样，之后按策略步号复用。如果同一步已经在reset前采样，此调用也保留原样本。其余速度采样/setter、interval配置不改。已与14观测组一起写入，当前CPU流程通过，见[14](../../../../../../../docs/migration/14_特权状态与观测布局.md)。

## 后端类型检查

```python
def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv) -> None:
    # 该实现依赖Kit，等事件实例开始构造时再导入。
    from isaaclab_physx.physics import PhysxManager

    super().__init__(cfg, env)
    if not isinstance(env.sim.backend_manager, PhysxManager):
        raise ValueError("RandomizeTitaMaterial supports Isaac Sim PhysX only.")
```

用户已用这段替换原构造函数开头，后续asset_cfg读取、系数采样和写入保留。新backend_manager是管理器实例，physics_manager只是弃用别名；isinstance用来判断实例所属的类型。实现类在本函数内导入，因为其实现依赖Kit；模块顶部导入会影响启动前的配置加载。没有新增MDP名称，不需要修改.pyi。用户日志已见startup完成并进入zero循环，尚未提供材质数值逐项回读；完整依据见[25](../../../../../../../docs/migration/25_Lab_main与Sim62_入口和配置.md)。

## 补充问答

**Q：if not isinstance(env.sim.backend_manager, PhysxManager)，backend_manager，没有代码提示！**

**A：** 新SimulationContext在__init__内有该属性和PhysicsManager类型注解。当前工作区应使用IsaacLab-main源码路径，并选择lab3_sim62的venv解释器；重载编辑器后再看补全；运行成功不能保证编辑器分析到同一份源码。具体位置与建议见[25的问答](../../../../../../../docs/migration/25_Lab_main与Sim62_入口和配置.md)。助手没有修改编辑器设置。

**Q：** 我想问了，“w”到底表示啥？？接触力有个w，这里还有w！

**A：** 接口名末尾的`_w`表示world，即世界坐标系。`root_vel_w`给出的线速度和角速度都沿世界坐标轴分解；`net_normal_forces_w`给出的力也沿世界xyz分解。对这些向量接口，`_b`表示body，即沿机身坐标轴表达，机身轴会随机器人转动。

例如假设世界+x向东、+y向北，机身+x为正前方。当机器人朝北以1 m/s前进，同一个线速度写成世界系分量是(0,1,0)，写成机身系分量是(1,0,0)。坐标变了，描述的仍是同一次运动。

变量名velocity_w沿用这个约定，提示它存的是世界系分量；名字本身不会完成坐标变换。此前写出的wx/wy/wz，则是用字母w代替希腊字母ω来表示角速度分量，更清楚的写法是ωx/ωy/ωz。后缀_w的world与角速度符号ω是两个含义。

本地定义：[root_vel_w是root_com_vel_w的别名](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py)、[六维质心速度及世界系定义](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py)、[世界系法向接触力](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/sensors/contact_sensor/base_contact_sensor_data.py)。

**Q：** .to是啥方法？

**A：** 这里env_ids是torch.Tensor，调用的是PyTorch的Tensor.to，用于转换张量的device和/或dtype：

```python
env_ids = env_ids.to(device="cpu", dtype=torch.int32)
```

device="cpu"让索引数据位于CPU内存，dtype=torch.int32让索引元素为32位整数。例如CUDA上的int64索引[0,2]，得到CPU上的int32索引[0,2]；本例索引值与shape保持。当前Lab材质随机化也按CPU索引调用后端。

to返回结果，由等号把变量重新绑定到结果；发生转换时会产生转换后的副本。默认copy=False，若device和dtype本来就匹配，可以直接返回原张量。它不负责选择环境或写入仿真。[PyTorch 2.11官方说明](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.to.html)。

**Q：** .to_torch 和 from_torch细说？

**A：** 代码中的wp是warp模块的别名，这两个是Warp提供的跨库函数：wp.to_torch接收Warp数组，返回Torch张量；wp.from_torch接收Torch张量，返回Warp数组。

```python
material_wp = self.asset.root_view.get_material_properties()
material_view = wp.to_torch(material_wp)
materials = material_view.clone()
```

material_wp与material_view这两个Python对象共享同一块数组存储：通过任一视图改数据，另一边会看到。to_torch这一步不复制底层数值，也不把CPU数据自动搬到GPU。clone才创建独立副本materials，供当前随机化计算修改。

```python
material_wp_for_write = wp.from_torch(materials, dtype=wp.float32)
indices_wp = wp.from_torch(env_ids, dtype=wp.int32)
self.asset.root_view.set_material_properties(material_wp_for_write, indices_wp)
```

material_wp_for_write又与materials共享存储，from_torch同样不复制数值。对本例的标量float32/int32数组，shape和device也保留。dtype指定兼容的Warp元素类型，不能把它当作任意数值类型转换；例如需要从int64变成int32时，先使用Tensor.to，再包装成Warp数组。

整个流程为：PhysX读取材质到Warp数组A→Torch视图A→clone成独立Torch数组B→修改B→包装成Warp视图B→setter提交PhysX。数组共享只描述Warp/Torch对象之间的关系；getter在本地实现中维护一个会被后端刷新读取的缓冲区，修改它不等于已经提交仿真。实际修改物理材质的是最后的setter调用。

依据：[Warp官方互操作说明](https://nvidia.github.io/warp/latest/user_guide/interoperability/pytorch.html)、[本地from_torch](/home/eaziel/my_venvs/lab3_sim62/.venv/lib/python3.12/site-packages/warp/_src/torch.py)、[本地to_torch](/home/eaziel/my_venvs/lab3_sim62/.venv/lib/python3.12/site-packages/warp/_src/torch.py)。

**Q：** warp不是mujoco后端吗？这里啥意思？physx借用了mujoco后端的设计思想？还是说warp不是mujoco后端？那是什么？

**A：** NVIDIA Warp是用于CPU/GPU并行计算的Python框架，提供数组和编写计算kernel的能力，可以把相应Python kernel编译为CPU/CUDA代码。MuJoCo Warp（MJWarp）是建立在Warp上的MuJoCo物理仿真实现，两者属于不同层次。[NVIDIA Warp介绍](https://developer.nvidia.com/warp-python)、[MJWarp项目](https://github.com/google-deepmind/mujoco_warp)及其[Warp依赖](https://github.com/google-deepmind/mujoco_warp/blob/main/pyproject.toml)。

本地PhysX配置创建数据接口的源码是：

```python
self.simulation_view = omni.physics.tensors.create_simulation_view("warp", stage_id=cfg.stage_id)
```

被调函数的本地签名是：

```python
def create_simulation_view(frontend_name, stage_id=-1, backend="physx"):
```

因此"warp"传给frontend_name，选择用Warp数组表达接口数据；backend仍使用默认"physx"，物理求解由PhysX完成。当前Torch/Warp转换用于连接两套数组接口，这段代码没有调用MuJoCo。数组前端回答“数据用什么对象表达”，物理后端回答“由谁计算动力学与接触”。

依据：[本地PhysxBackend](/home/eaziel/IsaacLab-main/source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py)、[本地create_simulation_view签名和前后端分支](/home/eaziel/my_venvs/lab3_sim62/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.4.4+110.4.0.lx64.r.cp312.u7f4/omni/physics/tensors/api.py)。

**Q：** get_material_properties这个方法，和set_material_properties，没有代码提示，显示(function) set_material_properties: Any

**A：** 两个方法确实由本地omni.physics.tensors.api.ArticulationView实现。get返回形状(count,max_shapes,3)的材质数组，set接收完整材质数组和本次更新的articulation索引。它们属于机器人root_view所持有的后端批量视图，而非torch.Tensor或Warp数组自身的方法。

类型链中存在明确的薄弱处：任务注解来自isaaclab.assets.Articulation工厂；其通用BaseArticulation.root_view没有具体返回类型；PhysX专用子类才把root_view标注为physx.ArticulationView。外层工厂与后端专用类型之间的关系，编辑器不一定能完整推导。实际Tensor API的两个方法也没有Python参数/返回类型注解，类型信息主要写在docstring里。若Omni扩展路径未被Pylance解析，提示还会进一步缺失；本轮没有检查编辑器路径，不能认定这也是已证实原因。

可选的提示改进：在已有TYPE_CHECKING分支加入具体视图类型，然后给局部变量view注解，再经view调用getter/setter：

```python
if TYPE_CHECKING:
    from omni.physics.tensors.api import ArticulationView
```

```python
view: ArticulationView = self.asset.root_view
materials = wp.to_torch(view.get_material_properties()).clone()
# 完成现有的摩擦赋值后：
view.set_material_properties(
    wp.from_torch(materials, dtype=wp.float32),
    wp.from_torch(env_ids, dtype=wp.int32),
)
```

前提是Pylance能找到omni.physics.tensors；该类型注解只帮助编辑器，不创建新视图，也不切换物理后端。Any反映静态类型信息不足，不能单靠它判定运行时API不存在。当前代码运行是否成功，仍需后续真实环境验证。

2026-10-19核对包导出后修正：此前给出的顶层导入不准确。本地omni.physics.tensors/__init__.py只从api导出create_simulation_view、reset和SimulationView，没有导出ArticulationView；具体视图类型应直接从omni.physics.tensors.api导入。Lab自身的PhysX返回注解仍写顶层别名，不能据此假定安装包已重导出该类。

本地定义：[通用Articulation工厂](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/assets/articulation/articulation.py)、[通用root_view](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/assets/articulation/base_articulation.py)、[PhysX root_view返回类型](/home/eaziel/IsaacLab-main/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py)、[getter](/home/eaziel/my_venvs/lab3_sim62/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.4.4+110.4.0.lx64.r.cp312.u7f4/omni/physics/tensors/api.py)、[setter](/home/eaziel/my_venvs/lab3_sim62/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.4.4+110.4.0.lx64.r.cp312.u7f4/omni/physics/tensors/api.py)。

**Q：** 找不到omni，帮我解决一下？添加什么就好了？（不要修改，告诉我让我修改）这是关于from omni.physics.tensors import ArticulationView警告。

**A：** 在现有python.analysis.extraPaths数组中加入本机已确认存在的扩展根目录：

```json
"/home/eaziel/my_venvs/lab3_sim62/.venv/lib/python3.12/site-packages/isaacsim/extscache/omni.physics.tensors-110.4.4+110.4.0.lx64.r.cp312.u7f4"
```

这一层下面才是omni/physics/tensors/api.py。extraPaths需要包的导入根目录；只添加site-packages，不会把isaacsim/extscache里的每个扩展自动加入搜索路径。具体类型导入同时按上题改为from omni.physics.tensors.api import ArticulationView。保存后可执行Developer: Reload Window重新加载。源码路径已确认，用户尚未报告VS Code警告消失；助手没有修改设置或源码。参见[Pylance extraPaths说明](https://github.com/microsoft/pylance-release/blob/main/docs/settings/python_analysis_extraPaths.md)。

**Q：** for factors, bounds in ( (actuator.motor_strength, motor_strength_range), (actuator.kp_factor, kp_range), (actuator.kd_factor, kd_range), ):这到底是啥？我的意思是，我没看懂语法和执行过程的数据会是什么！

**A：** 先看最小的Python写法：

```python
for x, y in ((1, 2), (3, 4)):
    print(x, y)
# 第一次x=1、y=2；第二次x=3、y=4。
```

for每次取一个二元组，把两个成员分别绑定给左边两个变量。这叫解包。我们的外层元组有三项，每一项都是“一个Tensor对象、一个范围元组”。

| 次数 | factors指向什么 | bounds是什么 |
| --- | --- | --- |
| 1 | actuator.motor_strength | motor_strength_range，即(0.8,1.2) |
| 2 | actuator.kp_factor | kp_range，即(0.8,1.2) |
| 3 | actuator.kd_factor | kd_range，即(0.8,1.2) |

它不会把这三个Tensor拼接起来，也没有复制它们。第一次循环可展开成：

```python
factors = actuator.motor_strength
bounds = motor_strength_range
factors[selection, :] = sample_uniform(
    bounds[0], bounds[1], factors[selection, :].shape, device=asset.device
)
```

factors是原Tensor的另一个名字，切片赋值会修改原存储，所以actuator.motor_strength随之改变。bounds[0]/[1]分别是0.8/1.2；shape决定返回几行、几列独立样本。全量startup时腿为(N,6)、轮为(N,2)；若只选M个环境则为(M,J)。sample_uniform内部等价于rand*(1.2-0.8)+0.8。

**Q：** 这整个函数的作用我也不清楚。

**A：** randomize_tita_motor_factors的全部作用是“找到选中机器人的执行器，给三份因子张量填随机数”。asset_cfg.name取机器人，asset.actuators.values()依次取legs/wheels实例；内层循环分别填m/p/d。函数不算当下力矩；之后框架调用compute时，compute读取这些持久Tensor。

startup Event是本函数的调用者。当前EventCfg.randomize_motors已提供四项参数。采样函数只填已有执行器对象，不负责创建对象。

**Q：** 我没懂两处self.observation_manager.compute_group("privileged_state")在干啥。

**A：** 这里的一处先调用六维观测对象保存当下速度/接触，再执行速度覆盖。例如0.8 m/s被推扰改为-0.3 m/s，保存的仍是修改前0.8；最终观测按同一步号复用。另一处在环境reset之前。compute_group本身返回六列张量，未接收返回值不影响该对象已经保存的成员数据。
