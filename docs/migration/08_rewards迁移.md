# 08 Rewards：从旧计算到Lab配置

更新：2026-10-19。用户已保存10项配置、功率None默认值、加速度类/导出，以及最终奖励合成step；当前文件AST与此前方案对应。历史CPU替身检查与当前文件静态检查分别记录，尚未运行真实RL环境。本轮补普通项、active_terms、时间缩放和旧裁剪动机的原问题Q/A。

本章每个功能依次写Gym原计算、Lab代码、迁移理由和差异。用户在env_cfg.py中组合奖励；有现成计算就选官方MDP，有统计语义差异才补任务函数。

## 1. 所有奖励共用的计算流程

旧_prepare_reward_function把非零权重乘self.dt；这里self.dt是策略周期0.01 s。Lab的RewardManager.compute同样执行：

```python
value = term_cfg.func(env, **term_cfg.params) * term_cfg.weight * dt
```

ManagerBasedRLEnv传入dt=self.step_dt=sim.dt*decimation=0.0025*4=0.01。因此下面weight沿用旧配置，管理器负责乘时间；不要提前再乘一次0.01。

每个奖励函数返回(N,)张量，N是环境数。管理器对同一环境的各项贡献相加，并记录各项回合累计值。原始函数可以返回正的误差指标，再通过负weight变为惩罚。

旧任务还有一个独立的合成规则：

```python
r_step = torch.clamp(ordinary_reward_sum, min=0.0) + failure_reward
```

普通项已经包含各自权重和时间缩放。Lab默认管理器直接求和；10项现已接完，当前用第10节完成上述裁剪顺序。比如普通项合计-0.003，旧任务正常步返回0，同时失败时返回-2；默认Lab分别为-0.003和-2.003。最终总奖励不能整体夹非负，否则失败惩罚也会被夹掉。

## 2. 速度跟踪：实际运动接近命令

### 2.1 Gym原计算

有效权重为tracking_lin_vel=1、tracking_ang_vel=0.5，tracking_sigma=0.25。两个函数计算：

```python
lin_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
ang_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
r_lin = torch.exp(-lin_error / self.cfg.rewards.tracking_sigma)
r_ang = torch.exp(-ang_error / self.cfg.rewards.tracking_sigma)
```

commands前三列为[vx_cmd, vy_cmd, wz_cmd]；两类实际速度也在机身坐标系。航向模式把航向误差换成第三列的目标转速。指数奖励完全跟踪时为1，误差增大时接近0。

### 2.2 Lab代码：已写入

```python
    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
```

### 2.3 为什么对应

command_name选择CommandsCfg.base_velocity。官方函数调用env.command_manager.get_command("base_velocity")获得(N,3)张量，再明确取[:, :2]或[:, 2]。默认asset_cfg=SceneEntityCfg("robot")，分别读取root_lin_vel_b.torch和root_ang_vel_b.torch。

Lab分母是std**2，所以std=0.5对应旧sigma=0.25；若直接填0.25，分母会变成0.0625。目标vx=0.5、实际vx=0且vy误差为0时，r_lin=exp(-0.25/0.25)≈0.368。

两项每步合计为(1*r_lin+0.5*r_ang)*0.01，完全跟踪时为0.015。这里在相同命令与实际速度下对应；旧命令在奖励前更新、Lab在奖励后更新，且平移死区尚未补齐，整段轨迹仍可能不同。

## 3. 失败惩罚：机身接触结束时扣分

### 3.1 Gym原计算

```python
return self.reset_buf * ~self.time_out_buf
```

旧权重为-200。reset_buf含所有结束原因，排除time_out_buf后，非超时失败返回1。机身接触与超时同一步触发时也返回0。

### 3.2 Lab代码：已写入

```python
    termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-200.0,
        params={"term_keys": "base_contact"},
    )
```

### 3.3 为什么对应

term_keys选择现有TerminationsCfg.base_contact。is_terminated_term是可调用的奖励项类；ManagerBase先创建实例，RewardManager每策略步再调用它。该类读取选择的结束标记、排除time_outs，返回(N,)的0/1值。

执行顺序为：TerminationManager判定结束→RewardManager读取标记并算分→环境重置。当前只有一个非超时结束条件，与旧式对应；触发贡献为1*(-200)*0.01=-2。若两项跟踪完全命中且其他普通项为0，当步总奖励为0.015-2=-1.985。

## 4. 机身平稳：倾斜程度与xy角速度

### 4.1 Gym原计算

```python
def _reward_ang_vel_xy(self):
    return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

def _reward_orientation(self):
    return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
```

有效权重分别为-0.05、-1。前者为wx²+wy²，后者为重力单位方向的gx²+gy²，两者都是每环境一个非负指标。

### 4.2 Lab代码：已写入

```python
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
```

### 4.3 为什么对应

默认读取scene.robot。root_ang_vel_b.torch与projected_gravity_b.torch都是(N,3)，取[:, :2]得到(N,2)，平方后沿dim=1求和得到(N,)。这些是资产状态，观测的scale/noise不会修改奖励读到的数据。

第一项衡量绕机身x、y轴转动的快慢，单位(rad/s)²；z轴留给任务转向。机身角速度分量不一般等于欧拉角导数。

第二项把世界重力单位方向(0,0,-1)逆旋转到机身坐标系。PhysX初始化时归一化重力；属性按根姿态计算并按仿真时间戳缓存。水平正立时xy分量为0；侧倾30°时指标为sin²(30°)=0.25，贡献-1*0.25*0.01=-0.0025。该指标完全倒置时也可为0，保留的是旧公式的横向分量度量。

静止但倾斜，角速度项可以为0，姿态项仍有值；恰好经过水平但正在快速翻转，则相反。两个指标一起约束机身平稳，无需新增传感器。

## 5. 机身高度：相对周围地面保持0.35 m

### 5.1 Gym原计算

有效配置为base_height_target=0.35、base_height=-1。旧函数为：

```python
base_height = torch.mean(
    self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
    dim=1,
)
return torch.square(base_height - self.cfg.rewards.base_height_target)
```

root_states[:,2]是机身参考原点的世界高度(N,)；measured_heights是周围采样点的地面世界高度(N,P)。unsqueeze(1)变(N,1)，广播后每点相减，再对P个点取平均，得到机身相对这片地面的平均高度。

### 5.2 Lab代码：已写入RewardsCfg

```python
    # 机身参考原点相对扫描区域平均地面的高度，目标为0.35 m。
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.35,
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
```

### 5.3 为什么对应，以及高度扫描在这里怎样使用

Lab的base_height_l2读取root_pos_w.torch[:,2]与height_scanner.data.ray_hits_w.torch[...,2]。射线命中点形状为(N,P,3)，取最后一列后为(N,P)。当前P=187。官方函数执行：

```python
adjusted_target_height = target_height + torch.mean(hit_heights_w, dim=1)
penalty = torch.square(root_height_w - adjusted_target_height)
```

旧mean(z_base-h_i)-0.35与新z_base-(0.35+mean(h_i))代数等价；给定同一组有限地面高度，平方误差相同。这里先求平均高度误差，再平方；它与每个点各自平方再平均不同。

例如地面世界高度为1.0 m、机身为1.35 m，误差为0；机身为1.45 m，原始误差平方为0.01，负权重和0.01 s缩放后贡献-0.0001。目标0.35 m度量机身参考原点到扫描区域平均地面的高度，与轮子净空、碰撞体底面的距离分别有各自几何含义。

传入sensor_cfg使高度跟随地形世界高度变化。扫描器offset中的20 m是射线起点的抬高量，奖励读取命中点世界z；这里不把射线起点当机身高度。传感器被奖励读取，也不意味着187个高度已经进入策略观测。

当前旧_get_heights从高度图邻近三个网格值中取最小值；Lab射线使用命中几何。公式对应不代表两种输入逐点相同。当前本地base_height_l2直接mean所有命中z，若有未命中产生inf，结果会非有限；运行传感器时必须检查命中有效性，本文未宣称已验证。

## 6. 动作变化：相邻策略输出保持连续

### 6.1 Gym原计算

```python
return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
```

有效权重action_rate=-0.01。actions为经过左右重排和clip_actions=100裁剪、尚未控制缩放的八维动作。函数逐列相减、平方、按八个通道求和。

### 6.2 Lab代码：已写入同一个RewardsCfg

```python
    # 相邻策略步的八维原始动作之差。
    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
```

### 6.3 为什么对应，以及动作历史由谁维护

官方函数读取env.action_manager.action和prev_action，都是(N,8)。ActionManager.process_action每策略步先把原_action复制到_prev_action，再保存新输入；奖励阶段两者分别表示本次与上次策略输出。随后各ActionTerm才做自己的缩放和偏置，因此这个奖励使用原始八维输入，包含六腿与两轮，无需再建立历史缓存。

它计算sum((a_t-a_previous)²)，没有除以step_dt。若八个通道都从0变到0.1，原始值为8*0.1²=0.08，每步贡献为0.08*(-0.01)*0.01=-0.000008。相同置换应用于前后两个动作向量时，平方和不变，因此仅左右重排本身不会改变标量结果。

目前有两条输入/边界差异留待后续处理：

1. Gym在奖励前把输入裁剪到[-100,100]，Lab ActionManager保存未裁剪输入；当前尚未迁入原始动作裁剪。超过范围时，奖励计算会不同。ActionTerm的processed_actions裁剪时机另见03章。
2. Lab在reset中把action和prev_action都清零，下一回合首次比较以0为参考。旧reset_idx虽清零last_actions，但post_physics_step末尾又对全部环境执行last_actions[:]=actions[:]；被重置环境因此可能保留上一回合最后动作作为下一步参考。这里采用Lab当前原生边界，尚未照搬这个旧行为。

## 7. 小腿碰撞：统计当前有几条腿超阈值

### 7.1 Gym原计算

旧配置penalize_contacts_on=["leg_3"]，函数为：

```python
return torch.sum(
    1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
    dim=1,
)
```

有效权重collision=-1。每个选中身体的合接触力先求模，严格大于0.1 N时计1，然后按身体求和。两条腿同时超阈值得到2，与力是否远高于阈值无关。这项只负责分数；回合结束仍由TerminationsCfg决定。

### 7.2 Lab任务函数：已保存到mdp/rewards.py

以下为当前保存函数；用户自行添加了ContactSensor类型注解，对应导入也已存在：

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

当前保存代码已统一使用明确法向力入口：

```python
    forces_w = contact_sensor.data.net_normal_forces_w.torch
```

本地兼容入口net_forces_w会警告后回退到法向力。已观察到用户保存了上面的明确读取；当前函数在CPU替身上得到[0,1,2,0,1,0]且无兼容警告，这不代表真实PhysX接触已运行。

官方undesired_contacts读取net_normal_forces_w_history，并取历史窗口最大力。当前Scene保留3次历史；即使本次已无接触，较早样本仍可使它扣分。旧任务读当前帧，因此这里用任务函数保留统计时机。

### 7.3 导出任务函数：已保存到mdp/__init__.pyi

当前公开名称与相对导入均已登记：

```python
__all__ = [
    "joint_pos_target_l2",
    "joint_pos_rel_zero_wheels",
    "proprio_history",
    "illegal_contact_current",
    "undesired_contacts_current",
]

# Forward stable MDP terms lazily, then override with environment-specific terms below.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import joint_pos_rel_zero_wheels, proprio_history
from .rewards import joint_pos_target_l2, undesired_contacts_current
from .terminations import illegal_contact_current
```

本项目mdp/__init__.py调用lazy_export，运行时会读取.pyi中的导出关系。于是配置中的mdp.undesired_contacts_current才会解析到刚追加的函数。通用base_height_l2、action_rate_l2继续通过isaaclab.envs.mdp的转发获得。

### 7.4 Lab配置：已写入同一个RewardsCfg

```python
    # 只统计两侧leg_3当前超过0.1 N阈值的身体数量。
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
```

两个身体名已从当前USD的刚体prim静态核实。它们是身体名；配置管理器通过contact_forces传感器把名字解析成该传感器中的body_ids，运行时不应手填猜测的序号。

### 7.5 为什么对应，以及当前保留的差异

数据流为(N,B_all,3)法向力→按body_ids选(N,2,3)→对xyz求模得到(N,2)→阈值判断→转float并计数得到(N,)。一条腿超阈值时，每步贡献-1*1*0.01=-0.01；两条为-0.02。轮子正常接地不会因为本项被计数，因为body_names只选择两条leg_3。

当前计数函数与终止函数illegal_contact_current的末尾归约不同：这里把超阈值身体的数量求和；终止函数用any得到每环境一个布尔值。原始指标为0/1/2，负号由weight提供。

Lab当前PhysX接口net_normal_forces_w是聚合法向力；旧Gym net_contact_forces是合接触力。相同输入向量下，计数公式对应，但不能据此声明真实传感器力值完全相同。该聚合读数也未按碰撞对象过滤，接地、自碰撞或其他物体接触都可能贡献；旧函数同样不区分碰撞对象。真实绑定、接触口径、当前USD9个碰撞体与旧URDF5个碰撞体的动力学差异仍待运行对照。

## 8. 带符号机械功率：沿用旧torque乘velocity

### 8.1 Gym原计算

有效权重powers=-2e-5。旧_reward_powers实际执行的是：

```python
return torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1)
```

torques是最后一个物理子步下发的力矩，已经过motor_strength和力矩限幅；dof_vel是策略步结束时更新的关节速度。所有8个关节的tau_j*dq_j求和，得到每环境的带符号机械功率，单位W。源码中绝对值版本被注释，迁移依据是正在执行的这一行。

### 8.2 Lab任务函数：已保存

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

2026-10-19用户已写入None默认值和条件构造，当前空params与该接口匹配；显式传入asset_cfg时保留调用者选择。此前必填参数错误及Ruff B008的处理经过见项目历史与末尾原问题问答，不再作为当前待修问题。

两份张量按相同joint_ids选关节后相乘、沿关节轴求和，(N,8)→(N,)。当前TITA为显式IdealPD，applied_effort是限幅后下发力矩，对应旧取数位置。这里未对4个物理子步做积分/平均，沿用最后子步力矩与策略边界速度；轨迹与真实电耗未验证。

此前None方案的Ruff/CPU替身检查见历史证据`/tmp/tita_call_reset_ruff_20261019_hr1anlng/result.json`，该路径可能不再存在。本轮当前文件核对：/tmp/tita_reward_sum_saved_20261019_1h4kb34t/result.json，仅AST/源码检查，未运行仿真。

### 8.3 符号和权重怎样进入分数

tau与dq同向时，tau*dq为正；反向时为负。各关节正负项可以相抵，例如2*4+(-3)*2=2 W。乘负weight后，正功率产生负贡献，负功率产生正贡献；这里保留原式，不能把它称为总取绝对值的能耗惩罚。

每步贡献为sum(tau*dq)*(-2e-5)*0.01。机械功率也不等于电池电功率；当前目标是复现旧计算，不在本步改奖励定义。

## 9. 策略步关节加速度：用前后速度做差分

### 9.1 Gym原计算

有效权重dof_acc=-2.5e-7。旧函数为：

```python
return torch.sum(
    torch.square((self.last_dof_vel - self.dof_vel) / self.dt),
    dim=1,
)
```

last_dof_vel在上个策略步末尾保存；当前dof_vel来自本次4个物理步之后。self.dt=0.01 s，所以它估计的是两个策略边界之间的平均关节加速度，再对8个关节的加速度平方求和。

当前Lab PhysX的asset.data.joint_acc在每次data.update中按物理步更新，不能直接替换为mdp.joint_acc_l2。举例：最后一个2.5 ms物理步速度从0变1 rad/s，前面三个物理步速度均为0，则策略步估计为100 rad/s²，最后物理步估计为400 rad/s²；平方分别为10000和160000。

### 9.2 为什么本项用类

本项需要跨策略步保存上一份关节速度。Lab允许奖励条目继承ManagerTermBase：RewardsCfg仍用RewTerm选择它，RewardManager会自动创建实例、每步调用__call__，并在相应环境重置时调用reset。这个状态由奖励项自己持有，不需要修改TitaVelocityEnv或观测历史。

当前mdp/rewards.py已有以下导入：

```python
from collections.abc import Sequence

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
```

当前文件已保存以下类：

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

### 9.3 速度缓存与重置怎样对应旧任务

构造时zeros_like分配独立的(N,8)缓存。每次__call__先用当前速度与缓存相减，除env.step_dt、平方求和得到(N,)，随后copy_把当前值写入缓存，供下一个策略步使用。旧式last-current与此处current-last符号相反，平方后相同。不能把缓存直接赋为joint_vel引用，因为该Torch视图会随物理状态更新。

原生重置顺序是Scene reset→重置Events写机器人状态→RewardManager.reset→本项reset。此时将选中环境的缓存同步为重置后的实际关节速度。当前reset关节速度为0；旧post_physics_step末尾也会在reset后保存dof_vel，因此第一步参考一致。未重置环境的缓存保持不变。

RewardManager把全量env_ids=None转为slice(None)，本实现同时兼容slice、Tensor、Sequence和直接None。索引始终保留剩余关节轴[env_ids,:]，避免元组被解释成多轴索引。本项每策略步由管理器调用一次；后续总奖励裁剪应复用已算贡献，不能再次调用该项计算同一个策略步。

例：一个关节在10 ms内从0变到1 rad/s，指标为(1/0.01)²=10000，每步贡献10000*(-2.5e-7)*0.01=-0.000025。该项作用于实际关节运动；已有action_rate作用于策略输入，两者读数和约束对象不同。

### 9.4 导出与RewardsCfg：两项一起接入

mdp/__init__.pyi将两个公开名称加入__all__，并从.rewards导入。以下名称与导入均已保存，分行格式与当前源码等价：

```python
__all__ = [
    "joint_pos_target_l2",
    "joint_pos_rel_zero_wheels",
    "proprio_history",
    "illegal_contact_current",
    "undesired_contacts_current",
    "joint_power_signed",
    "JointAccelerationPenalty",
]

# Forward stable MDP terms lazily, then override with environment-specific terms below.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import joint_pos_rel_zero_wheels, proprio_history
from .rewards import (
    JointAccelerationPenalty,
    joint_pos_target_l2,
    joint_power_signed,
    undesired_contacts_current,
)
from .terminations import illegal_contact_current
```

当前RewardsCfg已在termination之前保存两项；下面显式写func=以便阅读：

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

func填写类对象JointAccelerationPenalty，构造由管理器负责。用户已显式写func=，reset注解也已补slice。10项及最终合成均已写入；下节解释当前合成实现。

## 10. 将各项合成为最终分数

### 10.1 Gym原计算

本文把当前除termination以外的9项合称“普通项”，包括正的跟踪奖励和负权重的各类惩罚。旧compute_reward先加这些项，再把普通合计夹到非负，最后追加termination。其准备函数主动把termination排除在普通项循环外：

```python
self.rew_buf[:] = 0.0
for i in range(len(self.reward_functions)):
    name = self.reward_names[i]
    rew = self.reward_functions[i]() * self.reward_scales[name]
    self.rew_buf += rew
    self.episode_sums[name] += rew
if self.cfg.rewards.only_positive_rewards:
    self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)
if "termination" in self.reward_scales:
    rew = self._reward_termination() * self.reward_scales["termination"]
    self.rew_buf += rew
    self.episode_sums["termination"] += rew
```

因此要实现r=max(普通项合计,0)+失败项。各项已经包含weight和0.01 s缩放。例如普通合计-0.003，正常返回0，失败返回-2；普通合计+0.004，失败则返回-1.996。对最终总和统一clamp_min(0)会抹掉失败惩罚。

### 10.2 Lab接在现有环境step中

10项与8.2节None接口已保存。原生RewardManager对全部条目直接求和；当前TitaVelocityEnv.step再处理最终分数，env_cfg.py继续选择各项，mdp/rewards.py继续计算各项。本地ManagerBasedRLEnvCfg没有对应的only_positive_rewards开关。

当前保存的TITA_Lab/src/TITA_Lab/tasks/velocity/tita_velocity_env.py.step为：

```python
def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # 此时 obs_buf 保存的仍是上一次返回的观测。
        self.proprio_history.read_then_push(
            self.obs_buf["policy"],
            self.episode_length_buf,
        )

        obs, reward, terminated, time_outs, extras = super().step(action)

        # 找到本步已经计算好的失败奖励。
        termination_index = self.reward_manager.active_terms.index("termination")
        termination_reward = (
            self.reward_manager._step_reward[:, termination_index] * self.step_dt
        )

        # 普通项合计夹到非负，再加回失败奖励。
        ordinary_reward = reward - termination_reward
        reward.copy_(ordinary_reward.clamp_min(0.0) + termination_reward)

        return obs, reward, terminated, time_outs, extras
```

### 10.3 为什么这样取数

父类已经推进物理、计算每个奖励、处理自动重置并生成新观测。其返回reward为(N,)张量，也就是env.reward_buf与RewardManager._reward_buf持有的同一缓冲区。

本地RewardManager还缓存(N,K)的_step_reward，K当前为10；第j列保存该项func返回值乘weight，原生实现写的是value/dt，因此这里再乘self.step_dt才恢复这一策略步的贡献。active_terms给出对应列名顺序；通过index("termination")定位，既不假定它是最后一列，也不依赖关节/身体顺序。

从总分减去已算出的失败贡献，就得到普通项合计；只夹普通合计，再加回失败贡献。copy_把结果写回原reward缓冲区，调用者、环境和管理器看到相同的最终分数。现有历史推进仍在super().step之前，既有_reset_idx和所有单项配置保留。

父类自动重置后仍能读取_step_reward：当前原生RewardManager.reset只清回合累计值并重置类项，不清这份本步缓存。这一点已按本地源码和CPU重置路径核对。不要在父类返回后重新读取已经reset的结束管理器来推断刚才失败了没有，也不要重新调用奖励函数；JointAccelerationPenalty第二次调用会读到自己刚更新的速度参考。

_step_reward属于管理器内部字段，本方案明确依赖当前本地实现的缓存含义与reset行为；后续升级Lab时核对这一处。读取它可以保留原生管理器的计算和重置流程，也无需复制整个step或创建第二套管理器。各项episode日志仍按原始带权贡献累计，与旧Gym的逐项日志口径对应；各项日志之和不必等于经过普通合计裁剪后的总回报。当前默认未启用Recorder；若以后在父类record_post_step内部记录最终总分，需把同一处理放到记录发生前，该阶段暂未实现。

### 10.4 文件职责与状态

| 文件 | 当前职责与状态 |
| --- | --- |
| config/TITA/env_cfg.py | 10项函数/类、权重与现有参数已保存，dof_acc已显式写func= |
| mdp/rewards.py | 功率None接口、加速度类及slice注解已保存；管理器负责类项调用/reset |
| mdp/__init__.pyi | 两个新名称已登记 |
| tita_velocity_env.py | 历史接线与本节最终奖励合成已保存，AST与此前方案对应 |

本轮对保存源码的AST核对通过；此前方案的CPU替身检查记录在第11节，尚未运行真实RL环境。助手不修改TITA_Lab。

## 11. 覆盖状态、剩余项与检查边界

| 旧有效奖励 | 当前状态 |
| --- | --- |
| tracking_lin_vel、tracking_ang_vel | 已写入，sigma/std对应，源公式与静态配置已核对 |
| ang_vel_xy、orientation | 已写入，函数/权重AST核对通过 |
| termination | 已写入，选base_contact、排除超时，触发贡献-2 |
| base_height、action_rate、collision | 已写入，参数/自定义函数/导出AST核对通过；保存碰撞函数CPU通过，getter已统一 |
| powers | 配置与None接口已保存；历史方案带符号功率CPU通过，当前AST核对通过 |
| dof_acc | JointAccelerationPenalty/-2.5e-7已保存；原生类项构造/登记、逐步调用、部分/全量reset与独立缓存CPU通过 |
| 普通总和夹非负、失败项后加 | 第10节已保存，当前AST与此前4轮CPU替身方案对应；真实运行尚未验证 |

前一轮从实际Gym/Lab文件抽取函数，用CPU张量和场景/管理器替身检查：

- 高度5组：水平地面、整体升高1 m、高于目标高度0.1/0.05 m、起伏地面但均值为0。给定相同187个有限地面高度时，两侧公式一致；原始值约[0,0,0.01,0,0.0025]。
- 动作4组：不变、八通道各变0.1、单腿通道变化、仅两轮变化；两侧原始值约[0,0.08,0.01,8]。额外核对共同置换不改变平方和。
- 碰撞6组：仅轮子受力、一条腿、两条腿、恰好阈值、两个小分量合成后超阈值、历史有接触但当前无接触。旧函数与当前帧方案均得到[0,1,2,0,1,0]；最后一例Lab通用历史函数得到1，验证了补任务函数的必要性。
- 抽取实际RewardManager.compute，核对三项权重与step_dt=0.01缩放；检查三项待写配置语法，并静态读取当前USD核实两侧leg_3刚体名。

前一轮方案证据：`/tmp/tita_reward_batch_20261018_ioy_cy4x/result.json`及同目录脚本。本轮保存8项配置/参数/导出AST、保存碰撞函数6例与实际兼容getter警告、bool转float32计数核对见`/tmp/tita_contact_questions_20261018_xbwueun9/result.json`。旧跟踪、失败、姿态公式检查分别记录在项目历史；旧/tmp路径可能不再存在。CPU检查不覆盖射线实际命中、接触报告绑定、动作完整生命周期、真实reset、动力学或训练。没有导入项目、创建真实环境或运行仿真。

本轮两个新条目在CPU替身上检查：直接抽取旧_reward_powers/_reward_dof_acc、Lab ManagerTermBase及RewardManager.compute/reset，与待写函数/类合用。功率得到[2,-6,0] W，且特意让computed_effort与applied_effort不同以核对限幅后取数；连续两次策略步、仅重置一个环境、全量slice重置后的再次计算共4轮对照通过。修改当前速度不会同步修改上一速度缓存；未重置环境保持参考值。额外全量重置为非零速度只验证缓存跟随新参考的行为，当前任务的实际reset速度仍为0。

证据：`/tmp/tita_power_acc_20261018_iio621o7/result.json`及同目录terms_proposal.py/cfg_proposal.py/check.py。仅执行抽取代码和CPU替身；未导入任务、创建真实管理器/环境或运行仿真。该段为2026-10-18待写方案的历史检查；2026-10-19用户已保存两项实现，当前核对见下一段。

2026-10-19初读版本：对保存函数和10项函数/权重/导出作检查；抽取完整原生ManagerBase/RewardManager/ManagerTermBase类，用CPU配置与场景替身实际构造奖励管理器，确认类变实例、被登记、compute调用和reset传播。功率[2,-6,0] W；缓存独立、部分/全量reset通过。第10节待写step使用父类替身，4轮对照旧compute_reward，覆盖正/负普通合计、失败、超时、失败与超时同时发生、父类自动重置、手动重置标记及原始逐项日志；每步加速度只调用一次。报告：`/tmp/tita_reward_lifecycle_20261019_cz984jqo/result.json`。配置与场景为替身，未导入真实任务或运行仿真。

此前编辑期间功率函数曾被改为必填参数，原生管理器校验失败，使当时初读通过的结论失效；该问题现已按8.2节None接口修复。恢复默认值的旧方案、保存加速度类及第10节合成曾在CPU替身上通过4轮对照，历史证据：`/tmp/tita_reward_lifecycle_20261019_4ng2ex26/result.json`。当前文件核对见下一段；历史方案检查与真实环境运行分别记录。

本轮当前文件核对：`/tmp/tita_reward_sum_saved_20261019_1h4kb34t/result.json`。功率None修复、显式func=、slice注解与最终合成step均已保存；step AST与此前方案一致，不把源结构核对说成真实环境验证。

## 12. 对照源码

| 内容 | 本地源码 |
| --- | --- |
| Gym有效权重、目标高度 | [TITA rewards](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:100)、[父类rewards](/home/eaziel/Downloads/tita_rl-master/configs/legged_robot_config.py:163) |
| Gym准备、合成与非负裁剪 | [_prepare_reward_function](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:908)、[compute_reward](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:707) |
| Gym奖励公式与接触身体 | [_reward函数](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1153)、[asset配置](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:90) |
| Gym高度图采样 | [_get_heights](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1051) |
| Lab高度函数 | [base_height_l2](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:125) |
| Lab动作/历史接触函数 | [action_rate_l2](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:278)、[undesired_contacts](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:294) |
| Lab动作保存与重置 | [ActionManager.process_action](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py:380)、[reset](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/action_manager.py:358) |
| Lab权重与时序 | [RewardManager.compute](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:126)、[环境调用](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:250) |
| Lab类奖励与类项创建 | [is_terminated_term](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:43)、[ManagerBase](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/manager_base.py:400) |
| Lab机身状态与单位重力 | [通用奖励](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:111)、[PhysX projected_gravity_b](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py:1258) |
| Lab关节加速度更新频率 | [PhysX update](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py:132)、[joint_acc](/home/eaziel/IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py:1227) |
| 本项目导出机制 | [lazy_export](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/utils/module.py:84) |
| 执行器限幅后的力矩 | [ActuatorCollection.applied_effort](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_collection.py:176)、[字段说明](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_base.py:119) |
| 有状态条目与重置 | [ManagerTermBase](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/manager_base.py:28)、[类项创建](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/manager_base.py:400)、[RewardManager.reset](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/reward_manager.py:102)、[环境重置顺序](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:395) |

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> base_velocity，我们没有指定取第几个数据，他怎么知道track_ang_vel_z_exp就是wz呢？

**A：** track_ang_vel_z_exp内部明确写了get_command(command_name)[:,2]。配置选择名为base_velocity的命令项；当前UniformVelocityCommand正好以[vx,vy,wz]排列输出。多个command与自定义列顺序的追问见[04 Commands补充问答](04_commands迁移.md#补充问答)，其中保留原问题和源码链。

**Q：**

> 我记得，PPO叫近端策略优化，里头就有让连续两次策略输出不要太远，但这或许是PPO的数学架构，和这里的action rate是啥关系？

**A：** 你记得的是PPO希望一次训练更新后，策略相对更新前不要变化过猛。这里需要区分训练迭代与机器人控制时刻：

| 比较 | 固定/变化的对象 | 目的 |
| --- | --- | --- |
| PPO新旧策略 | 固定同一条样本的观测o_t与动作a_t，比较更新前后两个策略的概率分布 | 控制训练中的策略更新幅度 |
| action_rate | 在轨迹上比较相邻控制时刻的实际动作a_t与a_(t-1)；观测可以不同 | 在任务奖励中鼓励动作时序连续 |

PPO的核心比值为：

```text
rho_t(theta) = pi_theta(a_t | o_t) / pi_old(a_t | o_t)
```

分子分母用同一个o_t、同一个已经采集的a_t；改变的是网络参数版本。这里“概率”在连续动作下具体是概率密度。本地RSL-RL实现的关键代码为：

```python
ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
surrogate = -torch.squeeze(batch.advantages) * ratio
surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

clip_param=0.2时，优化目标使用[0.8,1.2]范围的裁剪版本。这个裁剪通过目标函数限制某些继续改变策略的收益；它没有直接给相邻动作差设置上限，也不保证实际所有概率比严格落在该范围。完整公式见[原PPO论文第3节](https://arxiv.org/pdf/1707.06347)。

反过来，当前action_rate计算sum((a_t-a_(t-1))²)。即使网络参数完全不更新，同一个固定策略也可能在两个不同观测下连续输出+0.8、-0.8；此时新旧策略相同，PPO概率比为1，但这个通道的动作变化指标为(-0.8-0.8)²=2.56。训练中的动作还可能从分布采样，因此也有探索带来的逐步变化。

本项目的数据流为：相邻动作→action_rate惩罚→合入环境奖励r_t→计算回报/优势→PPO根据这些优势更新策略。action_rate定义任务更偏好怎样的控制行为；PPO定义怎样用采样数据更新网络，两者同时使用。该奖励不会直接覆盖当前动作或强制滤波。

本地runner也是先用当前策略采一段轨迹，再compute_returns和update；当前生成配置num_steps_per_env=16。因此一次网络训练更新与一次0.01 s控制步并非同一个周期。本次仅追读必要消费端，未迁移或调参算法。

源码：[本地RSL-RL PPO](/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py:269)、[采样与更新](/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/lib/python3.12/site-packages/rsl_rl/runners/on_policy_runner.py:81)、[当前任务PPO配置](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/agents/rsl_rl_ppo_cfg.py:14)。

**Q：**

> 对那个action rate我还有个问题，因为物理上，观测是连续变化的，所以可以这样做，因为环境连续，要让策略对连续变化的观测产生的响应也是连续变化的，这意味着要求不能有很大的突变，对吧！

**A：** 这是动作平滑惩罚的主要动机。机器人每0.01 s接收一次新动作，希望正常运动时动作少来回跳，以减少抖动和执行器负担。当前计算为：

```text
action_rate_t = sum_j((a_t[j] - a_(t-1)[j])²)
reward_contribution = -0.01 * action_rate_t * 0.01
```

这里比较完整8维原始策略动作；函数没有除以dt，所以名称里的rate不代表严格计算了动作时间导数。腿动作随后映射为位置目标偏置，轮动作映射为前馈力矩；PD反馈、当前状态和限幅仍会影响最终力矩，不能把动作平滑直接等同于力矩平滑。

“希望少突变”是软偏好：策略需要在跟踪、稳定和动作变化代价之间权衡，仍可以为制动或防摔快速改动作。噪声、接触冲击、命令重采样也可能使相邻观测发生明显变化；探索时，策略还会从分布采样动作。

本项只在采到的轨迹上比较相邻动作，不给所有相近观测对应的输出规定连续性或变化上限，也不直接把当前动作改小。PPO的新旧策略比较见上一问。

**Q：**

> 而且，功率，这里怎么算？对应物理实体的话，我感觉很复杂啊！电机电功率，取决于很多东西，电机还不是纯功率器件，我的物理学和工程学打架了！物理学告诉我什么P=Fv啥的，以及P=UI（纯功率器件），这里的功率到底是啥？？

**A：** 旧Gym的powers以及当前Lab映射，计算的都是执行器在关节处传递的带符号机械功率。旧代码为：

```python
return torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1)
```

每个转动关节用力矩tau（N·m）乘角速度dq（rad/s），得到W；再对8个关节求和，返回每环境一个值(N,)。这是一个采样时刻的功率指标，没有积分得到整段能量，也没有计算电压、电流或电机温升。

三个公式都来自“单位时间传递多少能量”：

| 接口 | 功率 | 对应的传能过程 |
| --- | --- | --- |
| 平动机械接口 | P = F·v | 力作用下发生位移 |
| 转动机械接口 | P = tau*omega | 转矩作用下发生转角 |
| 电气接口 | p(t) = u(t)*i(t) | 电荷流过电压差 |

其中F·v是向量点积。电气公式可从dW=u*dQ和i=dQ/dt得到；它适用于电机，纯电阻并非使用UI的前提。电阻上的电压满足u_R=Ri，才能进一步把该电阻的发热功率写成i²R或u_R²/R。电机端电压还要平衡反电势和电感电压，不能把电机全部输入都解释为绕组发热。若电压、电流变化，应先计算同一时刻的u(t)i(t)，平均功率取这个乘积的时间平均；不普遍等于两个平均值的乘积。[OpenStax的电机例题](https://openstax.org/books/university-physics-volume-2/pages/9-5-electrical-energy-and-power)也分别计算电输入IV、机械输出Fv和损耗。

真实驱动的能量路径是：

```text
电池/直流母线 → 驱动器 → 电机 → 减速器 → 关节
                  每个环节都可能有损耗或暂时储存能量
```

以边界包住驱动器、电机和减速器，规定正电功率为从母线流入、正机械功率为向机器人输出，可写：

```text
电端输入功率 = 关节机械输出功率 + 损耗功率 + 内部储能变化率
```

这里把不可逆损耗单列，内部储能指磁场、转子动能等暂存项；不把同一份损耗发热再次计入储能。稳定驱动、内部储能变化可忽略时，才可用某个工作点的总效率eta近似P_elec=P_joint/eta。假设该工作点总效率0.8、机械输出100 W，则需电输入约125 W；这是假设算例，不是TITA的实测效率。效率随转速、力矩和温度等变化，不能给整个任务统一乘一个常数就声称获得真实电耗。[maxon公式手册](https://www.maxongroup.com/medias/sys_master/8802918268958.pdf?attachment=true)第39、46页分别给出电机效率和绕组/传动损耗关系。

电机模型能更直接地连接两边。对简化永磁直流电机，假定电感L恒定：

```text
u = Ri + L*(di/dt) + e       # 绕组电阻压降、电感电压、反电势
e = k_e*omega_m             # 电机转速产生反电势
tau_e = k_t*i               # 电流产生电磁转矩

u*i = R*i² + d(L*i²/2)/dt + e*i
```

在一致的SI单位和直流电机常数定义下k_e=k_t，因此e*i=tau_e*omega_m。这个推导说明电输入分到铜损、磁场储能和机电能量转换；电磁转矩tau_e还要驱动转子加速并克服机械损耗，之后才是轴端输出，再经过减速器到关节。[maxon电机建模说明](https://support.maxongroup.com/hc/en-us/articles/360013761160-Motor-data-and-simulation)解释了电流/转矩、转速/反电势常数及换相方式的影响。三相BLDC/PMSM需采用相应的相量或dq约定，不能直接把母线电流当成相电流套这个简化式。

最直接的反例是静止撑住机器人：关节转速为0，即使保持转矩很大，tau*omega仍为0；在没有机械锁止、靠通电维持转矩时，电流并非0，绕组仍有i²R发热。因此当前powers不能衡量静止出力的电耗。

此外，tau*omega为负说明执行器在机械侧吸收能量。它是否回到电池、暂存在母线电容、消耗在驱动器或制动电阻，要由实际驱动电路和控制方式决定。当前按关节相加会正负相抵，不能由“一个关节+100 W、另一个-100 W”推断电池净耗电为0。乘负奖励权重后，负机械功率反而产生正贡献；这是旧公式已有的性质，迁移时需明确保留。

**Q：**

> 而且robomaster这个比赛也有功率限制，这里限制电功率，整机总功率不得大于限制。
> 如果sim这里的功率，基本就能当作电功率，那我很想知道，什么原理！

**A：** 当前仿真输出不能直接当成供电端电功率。TITA资产目前使用IdealPDActuatorCfg，只配置PD增益和关节限矩；当前任务没有电机绕组电阻、转矩常数、电压、电流、驱动效率或电池模型。相同的关节机械功率可以对应不同电输入，现有信息不足以唯一确定。

RoboMaster的类比需要先指定测量边界。官方[2026高校系列赛通信协议V1.3.0](https://bbs-web-static.robomaster.com/ef4f084944e34393aa70378a4a405c681774586313231/RoboMaster%202026%20%E6%9C%BA%E7%94%B2%E5%A4%A7%E5%B8%88%E9%AB%98%E6%A0%A1%E7%B3%BB%E5%88%97%E8%B5%9B%E9%80%9A%E4%BF%A1%E5%8D%8F%E8%AE%AE%20V1.3.0%EF%BC%8820260327%EF%BC%89.pdf)第13页明确命名chassis_power_limit，并分别列出底盘、云台、发射机构供电接口。因此不直接把底盘功率上限推广成整机所有设备用电的总和；具体范围和检测机制应以参赛赛季适用规则核实，此处不引用功率限值或声称该版本为最新版。

还要区分功率指标与功率上限：我们当前的powers是带权奖励项，没有P_max，也不会保证某个W数不被超过。若以后研究电功率限制，需要额外定义测量接口、驱动/损耗模型、参数依据及限功率机制。当前迁移阶段继续保留旧机械功率公式，不在解释过程中改成电耗模型或更换奖励定义。

**Q：**

> 我想问，为啥“原生 RewardManager 自动调用它的 reset()”？？

**A：** 本地源码明确串起三步：类项创建后被放入RewardManager._class_term_cfgs；环境_reset_idx调用reward_manager.reset；该方法遍历列表执行term_cfg.func.reset(env_ids=env_ids)。类项继承ManagerTermBase，因而可以由框架统一管理其状态。缓存跟随重置后的速度，避免跨回合差分。原代码和对象归属见[rewards.py补充问答](../../code_explaination/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/rewards.py.md#补充问答)。

**Q：**

> 以及，__call__干啥用了？？python的类名，在这里，是不是可以看作func就是构造函数__init__()？

**A：** 类名指向类对象。配置func=类对象时尚未创建奖励实例；ManagerBase调用该类创建实例，过程中执行__init__，然后把实例存到管理器内部的term_cfg.func。之后func(env)调用同一实例的__call__，每策略步计算一次。类对象、初始化方法与实例是三个对象/阶段，不能把func与__init__画等号。[源码解释](../../code_explaination/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/rewards.py.md#补充问答)给出逐步代码；Python语义见[官方__init__](https://docs.python.org/3.12/reference/datamodel.html#object.__init__)和[__call__](https://docs.python.org/3.12/reference/datamodel.html#object.__call__)。

**Q：**

> 因为ruff插件警告让我写到函数体里，我不想看到黄下划线。

**A：** 已用已安装Ruff插件CLI和项目配置复现B008。它关注默认参数里的函数调用只在定义时执行一次，返回对象会被多次调用共享。采用8.2节None默认值、if分支中按需构造即可；把形参直接改成必填会改变调用接口。该方案通过项目现有lint与CPU参数/选择检查，不要求关闭规则或忽略黄线。[Ruff官方B008说明](https://docs.astral.sh/ruff/rules/function-call-in-default-argument/)。

此前追问的Python括号调用、登记对应哪句append，见[奖励源码末尾Q/A](../../code_explaination/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/rewards.py.md#补充问答)；多个reset的对象归属与自定义历史用途见[环境源码末尾Q/A](../../code_explaination/TITA_Lab/src/TITA_Lab/tasks/velocity/tita_velocity_env.py.md#补充问答)。原问题保留；当前第10节合成代码已保存，其计分含义见下面的新问答。

**Q：**

> 现在是新的step的问题。我没搞懂什么叫普通项？为啥要 * step_dt？active_terms指的就是在rewardcfg里写的项？
> 还有为啥要这个计算公式啊，你说的旧gym的那个，他为啥这样搞？

**A：** 本轮说的“普通项”，就是当前RewardsCfg里除termination以外的9项：两项跟踪、机身角速度、倾斜、高度、动作变化、小腿碰撞、功率和关节加速度。其中既有正的奖励，也有负权重的惩罚；这个称呼用于解释旧合成顺序，不是Lab额外定义的配置类。termination在正常步为0，非超时失败步原始值为1，当前权重-200、step_dt=0.01，产生-2的贡献。

active_terms是RewardManager收集的奖励项名字列表。当前按配置顺序为：

```python
[
    "tracking_lin_vel", "tracking_ang_vel", "ang_vel_xy", "orientation",
    "base_height", "action_rate", "collision", "powers", "dof_acc", "termination",
]
```

本地属性直接return self._term_names；准备时以self._term_names.append(term_name)保存名字。index("termination")当前返回9，据此选_step_reward第9列。它不是得分列表；每个环境各项的得分存在(N,10)缓存里。当前版本None配置不收集；weight=0的RewTerm仍可能列在active_terms中，但compute跳过并把对应列置0，不应把active这个名字理解成每个权重必定非零。

step_dt=0.0025*4=0.01 s，是两次策略动作之间的时间。原生计算为：

```python
value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
self._reward_buf += value
self._step_reward[:, term_idx] = value / dt
```

乘dt采用按经过时间累计得分的约定。例如某项raw*weight一直等于1：

| 策略周期 | 1 s内步数 | 每步贡献 | 1 s合计 |
| --- | --- | --- | --- |
| 0.01 s | 100 | 1*0.01=0.01 | 1 |
| 0.02 s | 50 | 1*0.02=0.02 | 1 |

若不乘dt，这两个合计会变成100和50，仅因调用频率不同就改变了累计量。可以把这种缩放理解为对加权得分率做矩形积分。它是框架/奖励设计约定，并非所有RL奖励都必须乘dt，也不保证改变控制频率后任务完全等价：相邻动作差等原始量可能随步长变化，折扣也受步数影响；终止这样的一次性事件也被框架统一乘dt，当前-200对应一次失败-2，改变dt后需要重新考虑事件权重。官方文档说明其用途是平衡奖励相对时间步长的尺度：[RewardManager](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/api/lab/isaaclab.managers.html#isaaclab.managers.RewardManager)。旧Gym也在_prepare_reward_function中先把权重乘self.dt，本迁移保留这个约定。

新的step为什么又出现一次*step_dt？因为上面的_step_reward保存的是value/dt，也就是raw*weight。父类返回的reward已经包含dt；我们只为从缓存取回失败贡献而乘回来，没有把整个总分重复缩放。失败时缓存列为-200，取回-200*0.01=-2，才能从已经含-2的reward里减去它。

旧任务最后采用：

```python
final_reward = max(普通项合计, 0) + 失败项
```

所有数值都是该环境本策略步、已经乘权重及dt的贡献。父类给的是普通项+失败项，所以先减去失败项得到普通合计，夹非负后再加回来。不能把每个负项分别夹掉，否则功率/碰撞等惩罚就失去作用；也不能把最终总分夹掉，否则可能抹去失败扣分。

| 普通项合计 | 失败贡献 | 默认直接相加 | 旧Gym合成 |
| --- | --- | --- | --- |
| -0.003 | 0 | -0.003 | 0 |
| -0.003 | -2 | -2.003 | -2 |
| 0.004 | -2 | -1.996 | -1.996 |

为什么旧任务这样设计？本地legged_robot_config.py:181对only_positive_rewards的注释明确写着“avoids early termination problems”；[上游legged_gym配置](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py)也保留相同说明。按这个注释理解，其动机是：若机器人继续运行每一步都净扣分，而结束回合可以省掉后续扣分，策略可能倾向于提前结束；夹住普通合计是在减轻这种动机，失败项最后追加则保留失败扣分。这是设计动机的解释，不是当前任务训练中已观察到的结果，也不是关于最优策略的保证。

代价是：普通合计-0.003和-0.03都会得到0即时分数，这一范围的区分信息会被丢掉；负项仍可在未把合计压到0之前降低得分。PPO本身允许正负奖励，没有要求必须采用这个公式。当前照搬是为保留旧任务的奖励定义，是否改变应作为后续独立设计/实验决定。本轮只核对与讲解当前保存step，未运行真实环境或训练。

[返回迁移入口](README.md) · [来源与覆盖索引](00_迁移映射索引.md)
