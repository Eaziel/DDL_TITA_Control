# 03 Actions：从八维策略动作到腿轮力矩

**目标与结果：** 让八个策略数值按旧规则变成六腿位置目标、两轮前馈力矩，再由控制器生成受限力矩。

整理：2026-10-27。动作、显式PD、电机因子及物理步延迟已写。process_actions重复更新裁剪历史的问题已由用户修正；新venv实际FIR/action-rate CPU复核通过，四环境短随机检查报告自行结束。当前证据见[模块核查](../audits/2026-10-27_saved_stack_sim62/README.md)，缺陷过程保留在[25](25_Lab_main与Sim62_入口和配置.md)。延迟完整数据流见12，当前整体状态见[项目记忆](../../PROJECT_MEMORY.md)。

## 1. Gym怎样计算

每次策略输出actions:(N,8)，一行对应一个环境。旧底层关节顺序为左右各四关节，policy接口还做重排。本项目显式采用左1/2/3、右1/2/3、左轮、右轮；旧checkpoint需要单独转换顺序，不能直接加载。

旧_compute_torques()先做滤波、缩放和延迟，再按当前关节状态计算力矩。先单独看名义控制公式：

先把当前策略输出记作 $a_t\in\mathbb R^8$，裁剪后为 $c_t$，两点FIR输出为 $u_t$：

$$
c_t=\operatorname{clip}(a_t,-100,100),\qquad u_t=0.2c_{t-1}+0.8c_t.
$$

暂不计延迟和电机随机因子，六腿与两轮的名义控制为：

$$
q_{des,j}=q_{default,j}+s_j u_{t,j},\qquad
\tau_{leg,j}=40(q_{des,j}-q_j)-\dot q_j,
$$

$$
\tau_{wheel,j}=5u_{t,j}-0.5\dot q_j,\qquad
\tau_{applied,j}=\operatorname{clip}(\tau_j,-\tau_{lim,j},\tau_{lim,j}).
$$

腿的 $s_j$ 对髋侧摆为0.25，其余为0.5 rad/action；腿/轮限矩分别60/15 N·m。角度与角速度来自当前物理状态，不从动作矩阵里取。

轮子5*a来自旧10*joint_pos_target，而其target在无延迟时为0.5*a；默认轮角为0。该分支没有减轮角，也就不构成轮位置反馈。

## 2. Lab如何分工

Actions把策略输出变成目标；执行器用目标和实际状态计算力矩。目标每策略步更新一次，显式执行器每2.5 ms物理步重新计算，所以一个10 ms策略周期内会算四次力矩。

### 2.1 ActionsCfg：位置目标与轮前馈

当前ActionsCfg选择TitaJointAction，两组依次接收前六维与后两维。每策略步先计算c_t=clip(a_t,±100)、u_t=0.2*c_(t-1)+0.8*c_t，再由父类JointAction计算scale*u_t+offset。位置模式使用默认角作为offset，力矩模式offset为0。

```python
@configclass
class ActionsCfg:
    """Nominal action mapping for TITA legs and wheels."""

    leg_position = mdp.TitaJointActionCfg(
        asset_name="robot",
        command_type="position",
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
        raw_clip=100.0,
        previous_weight=0.2,
    )

    wheel_effort = mdp.TitaJointActionCfg(
        asset_name="robot",
        command_type="effort",
        joint_names=["joint_left_leg_4", "joint_right_leg_4"],
        preserve_order=True,
        scale=5.0,
        raw_clip=100.0,
        previous_weight=0.2,
    )
```

`preserve_order=True`让选择顺序与列表一致。六腿的位置项加默认角；轮力矩项产生5*u_t前馈。这里的scale把网络数值映射为角度或力矩，不是PPO的数学要求。FIR只在process_actions中推进；apply_actions在四个物理步中重复提交同一目标，执行器则根据各物理步状态重新算力矩。

原始策略输入保存在ActionManager.action，供动作观测读取；action_rate_clipped_l2从两个动作项的裁剪缓存计算sum((c_t-c_(t-1))²)。例如raw从150变200时，两次裁剪都为100，动作变化惩罚为0。完整缓存与函数说明见[actions.py](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/actions.py.md)。

此前apply_actions末尾多了`return super().apply_actions()`，现已由用户删除。父接口没有可复用的提交实现，会抛NotImplementedError；目标setter已完成提交职责。process_actions中的super调用保留，它有实际的scale/offset实现。

### 2.2 Scene：选择任务执行器

assets/tita.py保留名义IdealPD配置，速度任务在Scene里把两组替换为TitaPDActuatorCfg：

```python
robot.actuators = {
    "legs": TitaPDActuatorCfg(
        joint_names_expr=["joint_.*_leg_[123]"],
        stiffness=40.0, damping=1.0,
        actuator_effort_limit=60.0, joint_effort_limit=60.0,
        joint_velocity_limit=25.0,
    ),
    "wheels": TitaPDActuatorCfg(
        joint_names_expr=["joint_.*_leg_4"],
        stiffness=0.0, damping=0.5,
        actuator_effort_limit=15.0, joint_effort_limit=15.0,
        joint_velocity_limit=20.0,
        scale_effort_with_kp=True,
    ),
}
```

上面是当前配置。PhysX关节速度上限已在资产与Scene两处的两组配置同时补齐，来由见[09 F2](09_阶段审查.md)。

## 3. 电机强度与Kp/Kd因子

Gym为每个环境、每个关节保存三个独立U(0.8,1.2)因子：m为总力矩输出比例，p/d为增益比例；创建时采样，回合重置保留。

对一个环境中的一个关节，令 $m,p,d$ 为三个无量纲随机因子：

$$
\tau_{leg}=m\,[40p(q_{des}-q)-d\dot q],\qquad
\tau_{wheel}=m\,[5p\,u-0.5d\dot q].
$$

最后仍按固定限矩裁剪。轮前馈也乘 $p$，是沿用旧TITA轮分支的规则；默认因子全1时回到名义控制。

m不是效率或额定力矩；它是旧模型对输出差异的简化描述。p和d用于控制律的随机化。把所有因子设为1，就退化到上述名义控制器。

### 3.1 actuators.py保存并使用因子

```python
class TitaPDActuator(IdealPDActuator):
    """Explicit PD with TITA's independent gain and motor-strength factors."""

    cfg: TitaPDActuatorCfg

    def __init__(self, cfg: TitaPDActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        # 无量纲输出比例，用于随机化电机输出力矩，实现：有些给力过大，有些给力不足。
        self.motor_strength = torch.ones_like(self.stiffness) 
        # 用于随机化，如：实际kp = kp * kp_factor = 40.0 * 1.1 = 44，目前先设置为1！
        self.kp_factor = torch.ones_like(self.stiffness) 
        # 同上
        self.kd_factor = torch.ones_like(self.damping)

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        error_pos = control_action.joint_positions - joint_pos
        error_vel = control_action.joint_velocities - joint_vel
        feedforward = control_action.joint_efforts
        if self.cfg.scale_effort_with_kp:
            feedforward = self.kp_factor * feedforward

        self.computed_effort = self.motor_strength * (
            self.kp_factor * self.stiffness * error_pos + self.kd_factor * self.damping * error_vel + feedforward
        )

        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None

        return control_action
```

```python
@configclass
class TitaPDActuatorCfg(IdealPDActuatorCfg):
    """Configure TITA's explicit PD and wheel feedforward calculation."""

    class_type: type[TitaPDActuator] = TitaPDActuator
    scale_effort_with_kp: bool = False
```

`ones_like`创建同形状、同设备但数值全1的新Tensor；腿组(N,6)、轮组(N,2)。三份因子独立存储。轮组开关只让前馈多乘p，阻尼仍乘d；这对应旧轮分支。ActuatorCollection传入各组目标和当前状态，compute返回限幅后的effort。

### 3.2 startup Event填入样本

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

```python
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
```

函数从robot.actuators取得两组执行器，再把采样结果写进它们的因子Tensor。`for factors, bounds in (...)`逐次解包“Tensor与范围”，切片赋值修改原存储。Event不计算当步力矩，后续compute会读取这些样本。

## 4. 各控制环节当前怎样对应

| 环节 | Gym规则 | 状态 |
| --- | --- | --- |
| 原始裁剪 | c_t=clip(a_t,±100) | 已保存，CPU计算通过 |
| 两点FIR | u_t=0.2*c_(t-1)+0.8*c_t | 已保存，原候选CPU计算通过；多余父调用已删 |
| action-rate | 两次裁剪动作之差平方和 | 已保存，func/导出/计算已核对 |
| 物理步延迟 | 每物理步保存三份目标；编号0/1/2对应2/1/0步延迟 | 已保存并修正笔误，当前CPU流程通过；见12 |

早期检查复现过多余父调用异常，用户随后已删除该调用。2026-10-20内存修正的原生ActionManager对照通过，含4环境、12步裁剪/FIR/奖励/部分重置检查，以及16步×4物理调用的实际Gym滤波和力矩方法对照。后者开启滤波、关闭延迟，最大误差1.91e-6 N·m。目标写入端为CPU替身，未运行PhysX。首次无延迟基线真实运行见[10](10_首次环境运行.md)，此前截止电机因子的审查保存在[09](09_阶段审查.md)。

## 5. 源码位置

Gym：[control与domain_rand](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:63)、[_compute_torques](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:647)。Lab：[官方JointAction](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py)、[执行器模型](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/actuators/actuator_pd.py)、[ActuatorCollection](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/actuators/actuator_collection.py)。

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> 为啥rl非要有个action scale？不仅这个，其他rl task也有，比如go2（我记得好像）

**A：** RL并不要求必须有一个独立命名的action_scale。任务需要约定网络输出怎样对应物理控制量，scale是常用的映射参数。当前六腿把动作映射成默认角附近的角度偏移，两轮映射成力矩前馈；例如a=0.2时，普通腿目标增加0.1 rad，轮前馈为1 N·m。

**Q：**

> 不知道推机器人算action吗？凭我的记忆和直觉，commandcfg类更合适？不太记得了。

**A：** 这里先按谁在改变什么区分：Actions是策略给机器人的控制信号；Commands是希望机器人完成的速度目标；推扰是外部直接改变机器人的实际状态或施力。旧_push_robots改实际水平根速度，对应interval Event。

**Q：**

> lag_timesteps是啥？就是策略延迟？

**A：** 旧lag_timesteps=3指定动作历史缓冲长度。缓冲在每个物理步推进，当前随机索引对应0、1、2个物理步，即0、2.5、5 ms的控制输入延迟。策略仍每10 ms输出一次动作。索引越靠近缓冲末尾，读取的样本越新。

**Q：** 而且，我没搞懂现在在做什么！和usd资产的api有关联吗？比如articulation root啥的？

**A：** 当前在迁移“控制响应的域随机化”：让不同环境的机器人具有不同的PD增益比例与力矩输出比例，策略在这些响应差异下训练。USD的刚体、关节、质量、惯量与碰撞描述机械系统；ArticulationRootAPI标记供物理引擎组织为articulation的机构范围。Lab的Articulation对象读这套机构的状态并提交控制量，TitaPDActuator负责由目标和实际状态计算关节力矩，Event负责给这个计算抽参数。本步改变控制计算中的Tensor，不改关节连接或Articulation Root。

```text
策略动作 → ActionsCfg生成位置目标/轮子前馈
                          ↓
当前q、dq → TitaPDActuator.compute → 限幅后的关节力矩
                          ↑                   ↓
               startup Event抽的三个因子   物理引擎推进机器人
```

**Q：** 电机力矩不是kp*pos+kd*vel+前馈力矩吗？这里我咋没搞懂这公式是啥？？电机强度是啥？？？

**A：** 把pos/vel写全，PD控制律是tau_cmd=Kp(q_des-q)+Kd(dq_des-dq)+tau_ff，位置/速度反馈乘的是误差。当前速度目标为0，因此速度反馈为-Kd*dq。这是控制器计算力矩指令的模型；电机内部电流、电磁与驱动动态没有在这条IdealPD公式中完整建模。

stiffness就是名义Kp，damping就是名义Kd。kp_factor=p、kd_factor=d是在名义值上乘的无量纲随机比例；motor_strength=m是在总力矩上乘的无量纲输出比例。旧Gym选择tau=m[p*Kp*error_pos+d*Kd*error_vel+tau_ff]，两轮还将前馈乘p，之后统一限幅。m=0.8表示该模型在限幅前输出为括号内计算值的80%；它是用于模拟输出差异的简化参数，不是电机额定力矩、效率或电功率。

以一个腿关节为例：Kp=40、Kd=1，位置误差0.1 rad，速度误差-1 rad/s，前馈0。名义值输出3 N·m；抽p=1.1、d=0.9、m=0.8后，输出0.8×(44×0.1-0.9)=2.8 N·m，最后仍受固定限矩约束。

**Q：** 而且咋又成explicit了？？我记得之前一直接触的是implicitactuatorcfg！

**A：** 当前assets/tita.py已经保存IdealPDActuatorCfg；Scene再选择继承IdealPD的TitaPDActuator。本轮继续使用这条显式路径。Implicit把目标和增益交给引擎内的drive求解；当前显式路径在Lab模型里每物理步读取q/dq算力矩、限幅，再提交引擎。旧Gym的_compute_torques本来就是这类显式算力矩的方式。这描述控制计算的执行方式，物理积分器属于另一层选择。两者都使用同一个机器人关节机构，不要求改Articulation Root。

**Q：** stiff不是kp吗？damping不是kd吗？为啥又来kp因子和kd因子？还是全1类似self.stiffness和damping？？这里的self.stiffness和damping shape是啥？？

**A：** 是，stiffness/damping是名义Kp/Kd，factor为每个环境/关节的随机倍数。配置中的40.0/1.0是标量；父类把它们展开成(N,J)张量，便于并行计算。腿组J=6，stiffness全40、damping全1；轮组J=2，stiffness全0、damping全0.5。

```python
self.motor_strength = torch.ones_like(self.stiffness)
self.kp_factor = torch.ones_like(self.stiffness)
self.kd_factor = torch.ones_like(self.damping)
```

ones_like使用参考Tensor的形状、dtype和device，填充值固定为1，不复制原数值。即使轮子的stiffness全0，ones_like(stiffness)仍全1。三次调用建立三份独立Tensor。全1是乘法的中性初值；startup再用随机数覆盖它们。

**Q：** 但我看到了最后“randomize_motors‘这一项。我好像理解kp和kd因子是啥了，是用来随机化的？

**A：** 对。配置决定范围与时机；Event填样本；执行器在每个物理步使用样本。范围相同不意味着三份样本相同，它们由三次独立采样产生。当前startup抽一次，回合reset保留。

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

startup Event是本函数的调用者。当前EventCfg.randomize_motors已提供asset_cfg与三组范围。Event填写的是调用参数，执行器对象已由Scene创建。
