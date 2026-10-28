# mdp/observations.py：轮角屏蔽与历史取数

**阅读目标：** 理解轮角处理、历史读取、30维随机参数与六维采样的张量来源、排列和缓存时机。

源码目录：`Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/`。
源码文件：[observations.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/observations.py)。

更新：2026-10-21。两个函数及两个观测类均已保存；当前scan/dynamics与六维类的CPU检查通过，证据见docs/audits/2026-10-21_current_observations/。真实传感器数值和训练未验证。

## 当前代码

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from ..tita_velocity_env import TitaVelocityEnv

def joint_pos_rel_zero_wheels(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        wheel_cfg: SceneEntityCfg,
)->torch.Tensor:
    """Return joint position offsets [rad], with wheel entries zeroed."""
    # 从 Scene 取得机器人
    robot: Articulation = env.scene[asset_cfg.name]
    # 计算全部关节的 当前位置－默认位置。减法生成新张量
    joint_pos_rel = robot.data.joint_pos.torch - robot.data.default_joint_pos.torch
    # 在新张量里，把两个轮子的位置偏差置零
    joint_pos_rel[:, wheel_cfg.joint_ids] = 0.0
    # 按指定的八关节顺序返回
    return joint_pos_rel[:, asset_cfg.joint_ids]

# MDP 观测函数，把历史交给 ObservationManager
def proprio_history(env: TitaVelocityEnv) -> torch.Tensor:
    """Return past proprioceptive frames in whole-frame order."""

    # 展开为长度10*33=330的向量
    return env.proprio_history.read().reshape(env.num_envs, -1)
```

## joint_pos_rel_zero_wheels：生成八维相对位置

`env.scene[asset_cfg.name]` 取得场景中的robot对象。两个SceneEntityCfg都指向robot，但用途不同：asset_cfg指定输出八个关节及其顺序，wheel_cfg指定要屏蔽的两个轮子。管理器初始化时将名字解析成joint_ids。

计算先对全部关节做 `q-q_default`，减法生成新张量；在这份新张量中按原生轮索引置零；最后按asset_cfg的顺序取列，返回 `(N,8)`。因此不会改写真实关节角、默认角或轮速。

轮角指轮轴旋转关节角，单位rad。本任务按旧设计屏蔽相位/累计转角，保留轮速。后续ObservationManager才加位置噪声，因此最终两轮位置列仍可能有纯噪声。

## proprio_history：读取已有十帧

`env.proprio_history` 是环境持有的ProprioHistory对象，`read()` 返回缓存副本 `(N,10,33)`；reshape按整帧展开，返回 `(N,330)`。这个函数不重新取机器人状态，也不更新缓存。

TYPE_CHECKING内的TitaVelocityEnv导入供类型检查；future annotations让运行时不立即求值这份类型注解，避免循环导入。这里的torch.Tensor注解声明返回张量对象，输出列数由函数实际返回值决定。

## 怎样接入与验证

[__init__.pyi](__init__.pyi.md) 导出两个函数；[env_cfg.py](../config/Tita/env_cfg.py.md) 配置轮角函数参数并登记history组；环境step负责每策略步推进一次缓存。当前这三部分均已写入，用户已完成4环境20零动作步；逐项数值及长期历史时序仍需真实验证。

此前轮角函数做过独立CPU检查：非原生顺序、非零轮角、输出形状与原状态不变均通过。当前函数/导出与配置仅做静态核对，未实例化Isaac Lab。

Gym公式、Lab接线及迁移理由见 [05 Observations](../../../../../../../docs/migration/05_observations迁移.md)。API对照：[标准joint_pos_rel](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/observations.py)、[SceneEntityCfg解析](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/manager_base.py)。

## TitaDynamicsParameters输出30维随机动力学参数

状态：2026-10-21，类已由用户写入并通过CPU检查。保留上述两个函数，新类位于同一模块。导入ManagerTermBase、TitaPDActuator、RandomizeTitaMaterial，并在TYPE_CHECKING导入ObservationTermCfg，完整接线见[13](../../../../../../../docs/migration/13_训练观测迁移.md)。

```python
class TitaDynamicsParameters(ManagerTermBase):
    """Randomized TITA dynamics parameters relative to the nominal robot."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv) -> None:
        """Store nominal inertial properties before startup randomization."""

        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._asset: Articulation = env.scene[asset_cfg.name]

        if isinstance(asset_cfg.body_ids, slice) or len(asset_cfg.body_ids) != 1:
            raise ValueError("Select exactly one base body for dynamics observations.")
        self._base_id = asset_cfg.body_ids[0]

        # Observation terms are constructed before startup events run.
        self._nominal_mass = self._asset.data.body_mass.torch[:, self._base_id].clone()
        self._nominal_com = self._asset.data.body_com_pos_b.torch[:, self._base_id].clone()

        material = env.event_manager.get_term_cfg("randomize_material").func
        if not isinstance(material, RandomizeTitaMaterial):
            raise TypeError("Dynamics observations require RandomizeTitaMaterial.")

        # The event samples these per-robot values in its constructor.
        self._friction = material.friction_coeffs.to(self.device).unsqueeze(1)
        self._restitution = material.restitution_coeffs.to(self.device).unsqueeze(1)

        for actuator in self._asset.actuators.values():
            if not isinstance(actuator, TitaPDActuator):
                raise TypeError("Dynamics observations require TitaPDActuator groups.")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return base perturbations, material coefficients, and motor factors."""

        delta_mess = self._asset.data.body_mass.torch[:, self._base_id] - self._nominal_mass

        delta_com = self._asset.data.body_com_pos_b.torch[:, self._base_id, :] - self._nominal_com

        factors = []
        for name in ("motor_strength", "kp_factor", "kd_factor"):
            # Scatter actuator-local columns into the articulation's joint order.

            values = torch.ones_like(self._asset.data.joint_pos.torch)
            for actuator in self._asset.actuators.values():
                values[:, actuator.joint_indices] = getattr(actuator, name)

            # Gather into the task's explicit eight-joint observation order.
            factors.append(values[:, asset_cfg.joint_ids])

        # (N, 30) = mass 1 + CoM 3 + friction 1 + restitution 1 + 3*8 factors.
        return torch.cat(
            (
                delta_mess.unsqueeze(1),
                delta_com,
                self._friction,
                self._restitution,
                *factors,
            ),
            dim=-1,
        )
```

构造时，cfg.params中的SceneEntityCfg已被管理器解析为一个基座body_id和有明确顺序的joint_ids。当前初始化顺序为ObservationManager构造在startup之前，所以clone得到名义基座质量(N,)与局部COM(N,3)。clone使其不随物理属性更新而变化。

材质配置的func已被EventManager替换为RandomizeTitaMaterial实例；读取其构造时选择的每环境系数，再移动到env.device。类不抽随机数、不写仿真。此依赖针对当前自定义材质事件，改随机化流程时需同步检查。

__call__先作差，得到质量增量与局部COM增量。三次循环分别处理motor_strength、kp_factor、kd_factor；先按执行器局部索引写入底层关节列，再按任务joint_ids取列。返回cat生成独立(N,30)结果，顺序为1+3+1+1+8+8+8。

类继承ManagerTermBase，ObservationManager负责创建、求值及调用reset。默认reset不修改名义值，所以跨回合保留基准，也不需要在环境里手动接线。初始化维度探测会调用一次__call__，此时质量/COM增量为零；startup之后正常读到实际增量。

八环境原生CPU管理器与事件对照通过，包含关节重排、重复读取与部分reset；资产setter为替身，未运行PhysX。另六维观测也已实现并在本轮CPU流程检查通过；五组尚未接入训练消费。

## TitaPrivilegedState输出六维速度、接触和延迟编号

2026-10-21当前类和导入已保存，当前CPU检查通过；真实接触数值未验证。以下片段直接取自当前文件。

```python
class TitaPrivilegedState(ManagerTermBase):
    """Base velocity, filtered wheel contacts, and action-history slot features."""

    def __init__(self, cfg: ObservationTermCfg, env: TitaVelocityEnv) -> None:
        """Initialize per-policy-step state samples and contact memory."""

        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]

        if isinstance(sensor_cfg.body_ids, slice) or len(sensor_cfg.body_ids) != 2:
             raise ValueError("Select exactly two wheel bodies in the required order.")

        self._robot: Articulation = env.scene[asset_cfg.name]
        self._sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

        legs = env.action_manager.get_term("leg_position")
        if not isinstance(legs, TitaJointAction):
            raise TypeError("Privileged state requires TitaJointAction terms.")
        self._lag_denominator = max(legs.cfg.lag_timesteps - 1, 1)

        self._linear_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_contacts = torch.zeros(
            self.num_envs, 2, dtype=torch.bool, device=self.device
        )
        self._filtered_contacts = torch.zeros_like(self._last_contacts)

        # Dimension probing and reset before the first policy step keep a cold sample.
        self._sample_step = env.common_step_counter

    def __call__(
            self,
            env: TitaVelocityEnv,
            asset_cfg: SceneEntityCfg,
            sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Return six privileged features with one contact update per policy step."""

        if self._sample_step != env.common_step_counter:
            forces_z = self._sensor.data.net_normal_forces_w.torch[
                :, sensor_cfg.body_ids, 2
            ]
            contact = forces_z > 1.0

            # c_filtered(t) = c(t) OR c(t-1); samples are 10 ms apart.

            self._filtered_contacts.copy_(contact | self._last_contacts)
            self._last_contacts.copy_(contact)
            self._linear_velocity.copy_(self._robot.data.root_lin_vel_b.torch)
            self._sample_step = env.common_step_counter

        # This encodes the history SLOT, not the physical delay duration.
        lag_slot = (
            env.action_lag_indices.to(dtype=torch.float32).unsqueeze(1)
            / self._lag_denominator
        )

        return torch.cat(
            (
                2.0 * self._linear_velocity,
                self._filtered_contacts.float() - 0.5,
                lag_slot,
            ),
            dim=-1,
        )

    # Inherited reset() preserves contact memory and the current sample, as in Gym.
```

ObservationManager创建实例并进行一次维度探测，sample_step初始化为当前全局步号，因此探测不推进接触滤波。类每个新策略步读取机身COM线速度与两个轮体法向力Z，更新一次c_t OR c_(t-1)。参数SceneEntityCfg已解析轮体索引，配置明确左/右顺序。

返回cat的新(N,6)张量。sample_step使环境重置前、推扰前、最终观测以及终止观测的多次读取共享同一采样。默认reset保留前次原始接触与本步缓存，对应旧自动重置行为；首次step前保持冷启动样本。

完整配置、两个取样位置及CPU时序证据见[14](../../../../../../../docs/migration/14_特权状态与观测布局.md)。法向力与旧合接触力的信号差异、首次reset多一个零动作step的差异仍保留。

## 补充问答（2026-10-21）

**Q：** 我没理解你说的：依赖顺序仍然是：动作延迟 → 上条的地形/动力学参数 → 本条。

**A：** 之前把建议开发顺序说成依赖顺序，范围说得过强。六维类读取动作模块提供的action_lag_indices及lag_timesteps，因此要先有这两个量；scan和dynamics没有依赖这个六维类。最后的586拼接需要五组都已经存在。

**Q：** 我没动接触为啥要或滤波？有什么意义吗？

**A：** 传感器先给出力，再用Fz>1 N得到布尔判断。OR只处理这个判断，不改物理接触或受力。当前与上一次原始判断任一为True，输出就True。连续三次原始值[True,False,False]输出[True,True,False]，可保留一次短暂掉到阈值以下的接触。代价是真离地时也可能延后一次策略采样才显示无接触。这是旧Gym已有的规则。

```python
# (N,2)，当前策略步左右轮的原始接触判断。
contact = forces_z > 1.0
# 使用上一策略步的原始判断；不能改成与上一滤波结果OR，否则True可能一直保留。
self._filtered_contacts.copy_(contact | self._last_contacts)
self._last_contacts.copy_(contact)
```

**Q：** 关键变量你还是得注释啊，是变量的含义，比如_lag_denominator这些！

**A：** 可给当前初始化语句补以下注释，不改变计算：

```python
# 将保存目标的位置编号除以L-1，映射到[0,1]；L=1时除以1，避免零除。
self._lag_denominator = max(legs.cfg.lag_timesteps - 1, 1)
# 已采样的策略步号；同一步再调用时，复用保存的速度与接触。
self._sample_step = env.common_step_counter
```

lag_timesteps=3时编号0/1/2除以2，得到0/0.5/1；对应真实延迟5/2.5/0 ms。这个特征编码的是选择编号。_linear_velocity保存本步机身系速度(N,3)，_last_contacts保存上一策略步原始接触(N,2)，_filtered_contacts保存当前OR输出(N,2)。

**Q：** 为啥有时候是self device，有时候又是什么sim device，有时候还是device=cpu？

**A：** device指定张量所在设备及相应张量运算执行设备。cpu表示主内存与CPU；cuda:0表示第0块GPU。这里ManagerTermBase.device返回env.device，后者返回env.sim.device。asset.device取机器人自己的设备，当前配置中与环境一致。cfg.sim.device来自配置，父环境构造之前还没有self.sim，因而历史与共享编号先用cfg里的值创建。

材质事件这条PhysX属性读写路径使用CPU数据，和官方_Physx材质随机化实现一致。friction_coeffs.to(self.device)把用于观测计算的系数放到环境设备；转换结果由成员变量接住，原CPU系数仍保留。物理计算在GPU不意味着所有属性API都在GPU；应检查具体接口。设备名由哪个对象提供，与张量的(N,D)形状无关。一般张量计算不能把CPU和GPU数组直接混用，需要明确转换。[PyTorch设备文档](https://docs.pytorch.org/docs/2.14/tensor_attributes.html#torch-device)。

**Q：** common_step_counter，这是啥？这玩意每一物理sim步+1吗？

**A：** 它在ManagerBasedRLEnv.step中的四次物理调用之后加1，当前每10 ms加1，是所有并行环境共用的累计策略步号。各环境自己的episode_length_buf另计回合长度并在reset时清零；common_step_counter不随某个回合清零。

**Q：** 我没懂最后的cat。拼接30和6d的内容，但还是稍微需要解释一下。

**A：** 共有三个不同位置的拼接。动力学类先拼出30列；六维类先拼出6列；observation_layout最后取五组拼586列。动力学类里的cat本身不含那六列。

```python
# 质量差原本是(N,)，增加列轴后为(N,1)，才能和其他二维数据横向拼。
# factors保存三个(N,8)张量；*factors将它们逐个放入这个元组。
return torch.cat(
    (delta_mess.unsqueeze(1), delta_com, self._friction, self._restitution, *factors),
    dim=-1,
)
```

当前源码变量拼作delta_mess，功能为质量差。上面各项列数依次为1、3、1、1、8、8、8，相加30。*factors等价于在这里写factors[0], factors[1], factors[2]，不是乘法。二维张量的dim=-1就是列轴，N行保持不变，每台机器人自己的数据排在同一行。六维类另一次cat的列数为3+2+1。[torch.cat文档](https://docs.pytorch.org/docs/2.14/generated/torch.cat.html)。

**Q：** 我没懂两处self.observation_manager.compute_group("privileged_state")在干啥。

**A：** 它要求管理器立刻调用本组state.func，也就是已经建立的TitaPrivilegedState对象。新策略步中，该对象把当下的速度与接触复制到自己的成员变量，记录_sample_step，再返回六列结果。两处提前调用的返回值虽然没有变量接收，成员变量的更新仍然发生。

环境_reset_idx开头先调用，再让父方法重置机器人与传感器；push函数先调用，再覆盖机器人速度。例如物理步结束速度为0.8 m/s，推扰准备改为-0.3 m/s，提前调用会保存0.8。最后统一计算观测时，因为策略步号未变，仍返回保存的0.8乘2，即1.6；没有第二次从机器人取速度。同一步已在reset前保存时，push前调用也复用已有样本。

没有reset或push的普通步，在最终计算观测时首次采样。本组没有噪声或原生历史项，两处调用只算这六列，不更新其他观测组，也不推进仿真。它保留旧Gym先取速度/接触，再改状态的顺序。此行为取决于自定义类中的_sample_step判断，不能推广成所有compute_group调用都自动缓存结果。
