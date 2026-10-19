# mdp/rewards.py：任务特有的奖励计算

源码目录：`TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/`。
源码文件：[rewards.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/rewards.py)。

更新：2026-10-19。10项配置/导出、功率None接口与加速度类/slice注解均已保存；环境最终合成step也已接入。当前AST与此前方案对应；历史CPU替身与Ruff检查分别记录，真实RL环境未运行。

## 1. 当前已有函数

```python
def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)
```

该函数按asset_cfg选择关节位置、将角度wrap到(-pi,pi)，再计算相对target的平方误差和。本轮保留它；现有TITA奖励主要复用官方MDP函数，不需要把官方实现全部复制到这里。

## 2. 当前已保存的接触计数函数

```python
def undesired_contacts_current(
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Count selected bodies exceeding the current normal contact-force threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # (N, B_all, 3) -> (N, B_selected, 3)
    forces_w = contact_sensor.data.net_normal_forces_w.torch
    selected_forces_w = forces_w[:, sensor_cfg.body_ids, :]

    # (N, B_selected, 3) -> (N, B_selected) -> (N,)
    force_magnitudes = torch.linalg.vector_norm(
        selected_forces_w, dim=-1,
    )
    return torch.sum(
        (force_magnitudes > threshold).float(), dim=1,
    )
```

env是运行时环境对象；sensor_cfg由RewardsCfg的params传入。sensor_cfg.name选择scene.sensors中的contact_forces；body_names在配置解析阶段转成该传感器的body_ids。函数调用时直接用这些索引筛身体，不假定其顺序与关节顺序一致。

| 代码与对象 | 张量形状/类型 | 本步意义 |
| --- | --- | --- |
| contact_sensor.data.net_normal_forces_w.torch | (N,B_all,3)，float | 每个传感器身体的世界系聚合法向力 |
| forces_w[:,sensor_cfg.body_ids,:] | (N,B_selected,3) | 当前配置选择左右leg_3，B_selected=2 |
| vector_norm(...,dim=-1) | (N,2) | 对每个身体的xyz向量求力模，单位N |
| force_magnitudes>threshold | (N,2)，bool | 每个身体是否严格超过0.1 N |
| .float()再sum(dim=1) | (N,)，float | 每个环境超阈值的身体数量，0/1/2 |

最后一维为xyz，所以dim=-1计算单个身体的力模；计数时dim=1才把同一环境的多个身体合起来。函数不修改传感器数据，也不保存跨步状态。

## 3. 怎样进入奖励流程

```python
    collision = RewTerm(
        func=mdp.undesired_contacts_current,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_leg_3", "right_leg_3"]
            ),
            "threshold": 0.1,
        },
    )
```

配置与mdp/__init__.pyi导出均已保存。RewardManager随后以func(env,**params)调用它，再乘weight与step_dt。返回2表示两条腿均超阈值，当前贡献为2*(-1)*0.01=-0.02。函数返回原始非负计数，配置决定惩罚权重。

## 4. 与现有接触计算怎样对应

旧Gym的_reward_collision同样读取当前力、严格比较0.1 N并按身体计数；Lab通用undesired_contacts取历史窗口最大值。Scene已有3次历史，因此补当前帧函数可避免把历史接触当作本次接触继续计数。

终止函数illegal_contact_current最后用any，输出“有没有身体触发”的布尔值；本函数最后用sum，输出触发身体数量。它们分别服务结束条件和奖励统计，选择的身体及阈值也由各自cfg给出。

Lab当前PhysX读数为法向力，旧Gym张量为合接触力；本轮对齐统计计算，不宣称实际力向量相同。聚合读数也未按碰撞对象过滤。函数CPU对照包含仅轮子受力、0/1/2条腿、阈值边界和历史反例；真实身体绑定与动力学尚待验证。

完整Gym计算、Lab代码与输入差异见[08 Rewards](../../../../../../../docs/migration/08_rewards迁移.md)。公开名称登记见[__init__.pyi说明](__init__.pyi.md)。

## 5. 当前明确读取的力接口

```python
    forces_w = contact_sensor.data.net_normal_forces_w.torch
```

当前源码已经使用上述入口。该属性返回每个身体的世界系聚合法向力；本地net_forces_w兼容getter会先警告后转到这里。最新保存函数在CPU替身上得到[0,1,2,0,1,0]且无兼容警告；未运行真实传感器。

## 6. 已保存：带符号机械功率

```python
def joint_power_signed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Sum signed actuator mechanical power over the selected joints."""
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot")
        
    asset: Articulation = env.scene[asset_cfg.name]
    torques = asset.actuators.applied_effort.torch[:, asset_cfg.joint_ids]
    joint_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
    return torch.sum(torques * joint_vel, dim=1)
```

None默认值允许配置不传asset_cfg；只有未传入时才构造robot的默认选择，显式传入的配置保留。当前用户已保存这个版本；此前必填参数错误与Ruff B008处理经过见末尾原问题Q/A。函数输出逐环境的带符号关节机械功率(N,)，限幅后力矩与同一关节的速度相乘再求和，不包含电气损耗模型。

历史None方案通过Ruff及原生管理器CPU空参数/显式选关节检查，当前文件AST核对通过；真实任务未运行。完整来源与单位见[08第8节](../../../../../../../docs/migration/08_rewards迁移.md#8-带符号机械功率沿用旧torque乘velocity)。

## 7. 已保存：持有上一策略步速度的奖励项类

当前文件已有以下相关导入：

```python
from collections.abc import Sequence

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
```

当前保存的类为：

```python
class JointAccelerationPenalty(ManagerTermBase):
    """Penalize joint acceleration estimated between policy steps."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._robot: Articulation = env.scene["robot"]
        self._previous_joint_vel = torch.zeros_like(self._robot.data.joint_vel.torch)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        joint_vel = self._robot.data.joint_vel.torch
        joint_acc = (joint_vel - self._previous_joint_vel) / env.step_dt
        penalty = torch.sum(torch.square(joint_acc), dim=1)
        self._previous_joint_vel.copy_(joint_vel)
        return penalty

    def reset(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_joint_vel[env_ids, :] = self._robot.data.joint_vel.torch[env_ids, :]
```

类继承ManagerTermBase，属于一个奖励条目。RewardsCfg.func提供类，原生RewardManager创建并持有它。__init__保存robot引用并用zeros_like分配独立(N,8)速度缓存；__call__让实例能像函数一样被调用，每策略步返回(N,)的加速度平方和；reset由管理器按结束环境调用。

__call__中joint_vel是资产当前数据视图。相减和平方生成新结果；在penalty计算完后copy_将当前值复制进已有缓存。这样下一次物理数据变化不会改写前次参考；若直接把_previous_joint_vel赋成joint_vel，两者会引用同一份变化的数据。

这里的分母env.step_dt为0.01 s。原生joint_acc按物理步更新，本项主动保存策略边界速度，对应旧last_dof_vel/self.dt。旧last-current与这里current-last在平方后相同。

reset接受None、slice、Tensor或Sequence。全量None转slice(None)，然后[env_ids,:]只覆盖对应环境的全部关节。原生重置Events先写机器人状态，RewardManager再重置该项；因此缓存取重置后的速度，当前任务为0。部分重置不触及其他环境。

该类每策略步只应由RewardManager调用一次；后续奖励裁剪须复用已经得到的贡献。当前保存类的CPU检查已覆盖连续策略步、缓存无别名、部分重置和原生全量slice重置。当前用户已在env_ids注解中补上slice。

## 8. 已保存的配置与导出

```python
    powers = RewTerm(
        func=mdp.joint_power_signed,
        weight=-2.0e-5,
    )

    dof_acc = RewTerm(
        func=mdp.JointAccelerationPenalty,
        weight=-2.5e-7,
    )
```

两项已位于RewardsCfg内termination之前，__init__.pyi也已登记两个名称及导入。实际dof_acc代码已显式写func=，见[导出说明](__init__.pyi.md)。完整Gym公式、物理意义、步长和CPU验证见[08第8—9节](../../../../../../../docs/migration/08_rewards迁移.md#8-带符号机械功率沿用旧torque乘velocity)。单项计算由管理器完成；当前最终奖励合成已接入现有step，历史推进保留。

## 补充问答

**Q：**

> `float()` 没有代码提示。

**A：** 此处是torch.Tensor.float方法，等价于.to(torch.float32)。阈值比较得到(N,2)的bool张量，转换把True/False变为1.0/0.0，形状与设备保留；随后sum(dim=1)得到每环境的浮点计数。布尔张量直接sum也能计数，但默认产生整数结果。本地Torch 2.12的类型存根有float方法，CPU转换核对通过。用户随后报告给force_magnitudes添加: torch.Tensor后补全恢复，并要求忽略此问题；不继续检查或修改Pylance设置。

```python
mask = torch.tensor([[True, False], [True, True], [False, False]])
mask.float()               # [[1., 0.], [1., 1.], [0., 0.]]，float32
mask.float().sum(dim=1)   # [1., 2., 0.]
```

依据：[PyTorch 2.12 Tensor.float](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.float.html)。接触读数为何归到轮子或小腿、软件传感器如何创建，见[02 Scene末尾原问题问答](../../../../../../../docs/migration/02_scene迁移.md#补充问答)。PPO与动作变化惩罚的关系见[08末尾原问题问答](../../../../../../../docs/migration/08_rewards迁移.md#补充问答)。

2026-10-18用户追加的“观测连续与action_rate”“P=Fv/P=UI”“RoboMaster电功率限制”三个原问题及完整回答，同样归在[08末尾Q/A](../../../../../../../docs/migration/08_rewards迁移.md#补充问答)。其中joint_power_signed只计算关节tau*dq；静止出力仍可能通电发热，当前IdealPD没有电气/损耗模型，不能由该指标推断电池输入或保证不超功率上限。加速度类与功率None接口均已保存，电功率模型仍未实现。

**Q：**

> 我想问，为啥“原生 RewardManager 自动调用它的 reset()”？？

**A：** 由框架中的显式调用链负责。首先，奖励管理器识别已经创建好的类项实例：

```python
if isinstance(term_cfg.func, ManagerTermBase):
    self._class_term_cfgs.append(term_cfg)
```

term_cfg.func此时是JointAccelerationPenalty实例；继承ManagerTermBase使它满足判断，被登记进管理器的类项列表。环境重置指定环境时执行：

```python
info = self.reward_manager.reset(env_ids)
```

RewardManager.reset内部再执行：

```python
for term_cfg in self._class_term_cfgs:
    term_cfg.func.reset(env_ids=env_ids)
```

此处func持有的是实例，因此调用到你写的reset。该方法把选中环境的_previous_joint_vel同步成重置后的关节速度，避免把上个回合终点当作新回合的差分参考；其余并行环境保留自己的缓存。父类先执行reset Event写入机器人状态，再重置奖励管理器，时序正好满足这项需求。

源码：[登记类项](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:243)、[环境调用](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:421)、[管理器遍历reset](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:121)。普通函数没有被登记进这个列表，ManagerTermBase的默认reset为空操作；有状态的自定义类重写它来同步自己的状态。

**Q：**

> 以及，__call__干啥用了？？python的类名，在这里，是不是可以看作func就是构造函数__init__()？

**A：** 先看同一个类在两个时刻的调用：

```python
term_class = mdp.JointAccelerationPenalty  # 保存类对象，不创建实例
term = term_class(cfg=term_cfg, env=env)   # 创建实例，执行初始化__init__
values = term(env)                       # 调用已有实例，执行__call__
term.reset(env_ids=env_ids)               # 显式调用该实例的reset方法
```

类对象可以被调用来创建实例。__init__初始化该实例，并不返回奖励值；严格说对象的创建由__new__负责。实例实现__call__以后也能写成term(env)，该调用执行type(term).__call__(term, env)，不会重新创建实例或再次执行__init__。于是同一对象可以保留跨步速度缓存：第一次调用算分并存当前速度，第二次继续利用这份缓存。

Lab实际创建类项的语句是：

```python
if inspect.isclass(term_cfg.func):
    term_cfg.func = term_cfg.func(cfg=term_cfg, env=self._env)
```

右边先以类对象创建实例，左边再把实例写回管理器内部的配置副本。ManagerBase会deepcopy配置，因此不要理解成用户最初的RewardsCfg对象也必然被同样改写。

随后每个策略步计算奖励：

```python
value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
```

函数型条目直接运行函数；类项运行已有实例的__call__。两者都返回(N,)指标，管理器随后统一乘权重和时间。这就是func这个字段能接普通函数，也能接JointAccelerationPenalty类的原因。此前省略func=的写法只是把该类作为第一位置参数传入；当前用户已显式写func=。

源码：[创建类实例](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/manager_base.py:400)、[每步调用](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:147)、[配置字段](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/manager_term_cfg.py:32)。语言语义：[Python __init__](https://docs.python.org/3.12/reference/datamodel.html#object.__init__)、[Python __call__](https://docs.python.org/3.12/reference/datamodel.html#object.__call__)。

**Q：**

> 我没搞懂啊，__call__这个方法也是python给类的内置方法吗？因为有代码提示。
> 到底为啥会调用call啊？？你给的这句看起来就普普通通啊，而且之前func传的不是类吗？

**A：** __call__是Python规定的特殊方法名。Python见到“实例后面加括号”，会查找实例类型上的__call__来执行。一个最小例子：

```python
class Twice:
    def __call__(self, x):
        return 2 * x

obj = Twice()  # 调用类，创建实例
obj(3)         # 调用实例，进入Twice.__call__，结果为6
```

对这个例子，obj(3)相当于Twice.__call__(obj,3)。不用显式写出方法名，括号本身就触发调用协议。普通实例若自身的类和父类都不提供__call__，则不能这样调用；Python的object基类本身也没有该实现。[Python官方调用协议](https://docs.python.org/3.12/reference/datamodel.html#object.__call__)。

当前ManagerTermBase已经定义__call__占位方法，默认抛NotImplementedError；JointAccelerationPenalty重写它，所以编辑器也能从继承关系提供提示。依据：[本地基类](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/manager_base.py:104)。

func只是一个字段名，当前持有的对象决定括号含义：

```python
term_cfg.func = mdp.JointAccelerationPenalty   # 当前持有类对象
term_cfg.func = term_cfg.func(cfg=term_cfg, env=env)  # 创建实例，改为持有实例
value = term_cfg.func(env)                    # 调用这个实例的__call__
```

管理器实际把实例放在内部配置副本，之后不会每步重新构造。在最后一行中，__call__的self是奖励实例，env是传入的环境对象。变量名func不会强制值必须是普通函数。

**Q：**

> 我感觉你可以很清晰的讲一些东西，就是说，什么叫“登记”之类的，肯定有一句代码能对应上？

**A：** 这里登记具体就是把配置对象引用存进管理器的列表，方便之后遍历；不是Python关键字。本地原代码：

```python
self._term_names.append(term_name)
self._term_cfgs.append(term_cfg)
if isinstance(term_cfg.func, ManagerTermBase):
    self._class_term_cfgs.append(term_cfg)
```

_term_cfgs保存所有奖励条目的配置，compute遍历它们计算；_class_term_cfgs额外保存类项配置，reset遍历它们调用实例的reset。对类项，同一配置对象可被两个列表引用，配置的func持有同一个奖励实例。源码：[保存与分类](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:239)、[计算循环](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:141)、[重置循环](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:122)。

观察项也有对应的保存语句self._group_obs_term_cfgs[group_name].append(term_cfg)，见[ObservationManager](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:588)。当前保存进去的是mdp.proprio_history这个观测函数；ProprioHistory缓存对象由环境另行持有，其reset调用链见[环境说明末尾Q/A](../tita_velocity_env.py.md#补充问答)。框架不会扫描整个程序去寻找所有名字为reset的方法。

**Q：**

> 因为ruff插件警告让我写到函数体里，我不想看到黄下划线。

**A：** 用插件自带Ruff和本工程pyproject.toml复现后，之前默认值SceneEntityCfg("robot")触发B008；第6节None+if版本通过现有全部lint规则。默认值表达式在定义函数时执行一次，因此Ruff提示共享默认对象；None分支在每次未传参数的调用中创建默认配置。关键是保留形参的= None，并且只有None时才构造，不覆盖显式传入的选择。[官方B008](https://docs.astral.sh/ruff/rules/function-call-in-default-argument/)。

当前空params和显式asset_cfg选joint_ids=[0]均通过原生管理器参数准备/compute的CPU替身检查，功率分别[2,-6,0] W与[8,-6,0] W。测试只运行临时副本，未修改源码、未导入真实任务或运行仿真。证据：`/tmp/tita_call_reset_ruff_20261019_hr1anlng/result.json`。
