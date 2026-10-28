# actuators.py：TITA每个物理步怎样把目标变成力矩

**阅读目标：** 读懂控制目标与当前关节状态怎样形成显式PD力矩，以及三个随机因子如何改变控制律。

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/`。
源码文件：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/actuators.py`。

更新：2026-10-20。实际保存类与旧Gym做CPU计算对照。用户首次真实运行发现A2：需在环境sim上设置use_newton_actuators=False，使用能构造自定义TitaPDActuator的Lab执行路径；默认True的原生模型转换不支持该自定义类。该开关已写入，用户4环境reset/20零动作步正常结束，训练未验证。运行记录见[10](../../../../../../docs/migration/10_首次环境运行.md)。

## 1. 当前执行器代码

```python
from __future__ import annotations

import torch

from isaaclab.actuators import IdealPDActuator, IdealPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


# Explicit：Lab 模型每个物理步读取状态，算力矩、限幅，再交给引擎
# Implicit：把目标和增益交给物理引擎，由引擎中的 drive 求解
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


@configclass
class TitaPDActuatorCfg(IdealPDActuatorCfg):
    """Configure TITA's explicit PD and wheel feedforward calculation."""

    class_type: type[TitaPDActuator] = TitaPDActuator
    scale_effort_with_kp: bool = False
```

## 2. 配置选择类，框架建立实例

```python
class_type: type[TitaPDActuator] = TitaPDActuator
```

TitaPDActuatorCfg继承IdealPDActuatorCfg，保留关节选择、增益、限矩等参数；增加一个轮子前馈开关。class_type保存TitaPDActuator这个类对象。框架根据选中的关节建立此类实例，不需要用户在环境step中手动new或调用compute。

本地真正的构造语句在[ActuatorCollection._build_groups](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/actuators/actuator_collection.py)：

```python
self._groups[actuator_name] = actuator_cfg.class_type(**actuator_kwargs)
```

actuator_kwargs由框架提供cfg、joint_names、joint_ids、num_envs、device及默认属性。新构造函数的*args、**kwargs把除cfg外收到的位置/关键字参数原样交给父类；不是另一份仿真配置。

## 3. 实例保存什么

```python
super().__init__(cfg, *args, **kwargs)
self.motor_strength = torch.ones_like(self.stiffness)
self.kp_factor = torch.ones_like(self.stiffness)
self.kd_factor = torch.ones_like(self.damping)
```

父类把stiffness、damping、限矩解析为(N,J)张量，腿组J=6，轮组J=2。ones_like分别分配三个新张量，继承形状/dtype/device，初始全1；它们没有共享同一份存储。stiffness和damping保持名义值，因子单独保存，便于以后检查与特权观测使用。

startup Event通过env.scene["robot"].actuators.values()取到这些实例并填因子。类没有重写reset；继承的IdealPDActuator.reset为空操作，因此自动回合重置会调用reset，但不会重抽或清空因子。这符合当前Gym在_init_buffers只抽一次的规则。

## 4. compute接收什么，怎样算

```python
error_pos = control_action.joint_positions - joint_pos
error_vel = control_action.joint_velocities - joint_vel
feedforward = control_action.joint_efforts
if self.cfg.scale_effort_with_kp:
    feedforward = self.kp_factor * feedforward
```

control_action是原生ArticulationActions容器，保存本组当前的关节目标：位置、速度、力矩前馈。joint_pos/joint_vel是本组当前实际状态。框架在每次集合compute中都从目标缓冲建立新容器，因此最终改写容器不会在下一次重复缩放原始目标。

当前ActionTerm给腿q_des=q_default+scale*u，给轮tau_ff=5u；速度目标保持0。腿的scale_effort_with_kp=False，轮为True；轮Kp名义值为0，因此位置误差不贡献力矩，p只作用于轮前馈。

```python
self.computed_effort = self.motor_strength * (
    self.kp_factor * self.stiffness * error_pos
    + self.kd_factor * self.damping * error_vel
    + feedforward
)
self.applied_effort = self._clip_effort(self.computed_effort)
```

全部乘法为(N,J)逐元素运算，同一环境不同关节可有不同系数。先组合位置反馈、速度反馈、前馈，再乘整份m，最后使用父类_clip_effort。computed_effort与applied_effort这两个原生字段照常供ActuatorCollection汇集；腿限60，轮限15 N·m。

```python
control_action.joint_efforts = self.applied_effort
control_action.joint_positions = None
control_action.joint_velocities = None
return control_action
```

返回容器只保留最终力矩，表示显式执行器已经完成PD；后续提交力矩。位置/速度目标仍保存在集合的target_command中，容器里的None不会清掉那些目标。

## 5. 谁调用它

每个物理步，环境让ActionTerm写目标，Scene调用机器人的write_data_to_sim。内部[ActuatorCollection.compute](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/actuators/actuator_collection.py)执行：

```python
control_action = actuator.compute(
    control_action,
    joint_pos=joint_pos.torch[:, joint_indices],
    joint_vel=joint_vel.torch[:, joint_indices],
)
```

当前策略每0.01 s给一次动作，物理步为0.0025 s，所以同一策略步内此方法运行4次。目标通常相同，实际角度/速度在变，力矩据此重算。

## 检查边界

当前文件的配置/CPU检查见[阶段审查](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/docs/migration/09_阶段审查.md)；未运行物理环境或训练。

## 补充问答

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
