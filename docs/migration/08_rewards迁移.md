# 08 Rewards：十项计算与最终分数

**目标与结果：** 从本步运动结果计算十项分数，并按旧顺序组成最终奖励。

整理：2026-10-26。十项奖励与最终合成已写，公式/管理器有CPU检查。已有环境短运行；奖励设计能否训练出稳定行为尚未验证。 当前证据与整体状态以[项目记忆](../../PROJECT_MEMORY.md)为准。

## 1. Gym怎样形成一个步的奖励

旧_prepare_reward_function把各非零权重乘策略步长dt=0.01。compute_reward先累加普通项，合计夹到非负，再加入失败项：

$$
r_t=\max\!\left(0,\Delta t\sum_{i\in\mathcal I}w_i f_i(t)\right)
+\Delta t\,w_{term} f_{term}(t),\qquad \Delta t=0.01\ \mathrm{s}.
$$

$\mathcal I$ 是除失败项外所有已启用奖励项；$f_i$ 是函数原始输出，$w_i$ 是配置权重。负权重惩罚也包含在这个普通项集合中。

普通项含正奖励和负惩罚；失败项独立后加。此规则来自旧only_positive_rewards，不是PPO必须使用非负奖励的要求。

## 2. Lab配置与各项公式

```python
@configclass
class RewardsCfg:
    """Reward terms for TITA velocity tracking."""

    tracking_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,  # 每项贡献 = 函数返回值 * weight * step_dt
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

    # 相邻策略步裁剪后的八维动作之差。
    action_rate = RewTerm(
        func=mdp.action_rate_clipped_l2,
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
    termination = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={"term_keys": "base_contact"})
```

| 项 | 原始值f_i | 权重 |
| --- | --- | --- |
| tracking_lin_vel | $\exp(-\lVert\mathbf v_{cmd,xy}-\mathbf v_{base,xy}\rVert_2^2/0.25)$ | 1 |
| tracking_ang_vel | $\exp(-(\omega_{cmd,z}-\omega_{base,z})^2/0.25)$ | 0.5 |
| ang_vel_xy | $\omega_x^2+\omega_y^2$ | -0.05 |
| orientation | $g_x^2+g_y^2$，g为机身系单位重力 | -1 |
| base_height | $(z_{base}-\overline z_{ground}-0.35)^2$ | -1 |
| action_rate | $\lVert c_t-c_{t-1}\rVert_2^2$，c为裁剪到±100的策略动作 | -0.01 |
| collision | $\sum_{b\in\mathcal L}\mathbf1\{\lVert\mathbf F_b\rVert_2>0.1\mathrm N\}$，$\mathcal L$为两小腿 | -1 |
| powers | $\sum_j\tau_{applied,j}\dot q_j$，带符号机械功率 | -2e-5 |
| dof_acc | $\sum_j((\dot q_{t,j}-\dot q_{t-1,j})/0.01)^2$ | -2.5e-7 |
| termination | base_contact发生且本步非超时 | -200 |

原生跟踪函数分母为std²，所以旧tracking_sigma=0.25对应std=0.5。command_name选择哪组命令张量；函数实现约定第三列为wz，不会根据数据自动猜列。

重力指标度量倾斜的横向分量；完全倒置时也可为0，沿用的是旧公式。高度公式在给定相同187个地面高度时对应；实际射线与高度表的采样差异仍存在，扫描节点修正见09 F1。

## 3. 小腿碰撞为什么补任务函数

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
        selected_forces_w,
        dim=-1,
    )
    return torch.sum(
        (force_magnitudes > threshold).float(),
        dim=1,
    )
```

旧Gym只看当前力；原生某些接触奖励会取历史峰值，所以此处单独读取当前张量。索引(N,B,3)选身体，再沿xyz求范数，阈值比较后按身体求和，得到(N,)计数。Lab读取聚合法向接触力，旧Gym读取合接触力，真实数值并不保证相同。

## 4. 功率是怎样的物理量

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

tau[N·m]*dq[rad/s]得到关节机械功率W。使用applied_effort表示模型限幅后的驱动力矩；正负可以抵消，保留旧Gym的带符号sum。它没有电机效率、绕组损耗、驱动损耗或整机电子设备耗电，因此不能当作电池电功率，也不是硬功率上限。

asset_cfg=None是可选接口；函数内部创建默认SceneEntityCfg("robot")，避免可变默认对象与Ruff警告。未指定joint_names时joint_ids为全选slice。

## 5. 策略步加速度为什么用类

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

旧_dof_acc以10 ms前后两次关节速度差分，原生joint_acc可能使用2.5 ms物理步差分，不能直接替换。因此类持有上一策略步的速度副本。

ManagerBase识别ManagerTermBase子类后执行term_cfg.func(cfg, env)，将配置中的类换成实例；RewardManager随后每步调用该实例的__call__。计算结束copy_保存下一步参考。环境reset先写新关节状态，再由RewardManager.reset调用该实例reset，清理被选环境的参考。未重置环境的缓存保留。

## 6. 最终合成放在环境step

```python
termination_index = self.reward_manager.active_terms.index("termination")
termination_reward = self.reward_manager._step_reward[:, termination_index] * self.step_dt
ordinary_reward = reward - termination_reward
reward.copy_(ordinary_reward.clamp_min(0.0) + termination_reward)
```

RewardManager已经将每项乘weight和dt。内部_step_reward保存的是每项贡献除以dt的值，故这里只在恢复失败贡献时再乘dt；总reward不能再整体乘一次dt。先从原合计减去失败项，对普通合计夹零，再加回失败项。

当前失败原始值为0或1，贡献为-200*0.01=-2。is_terminated_term还屏蔽time_out，即使接触与超时同一步发生也不扣失败项。父类reset保留本步_step_reward，故返回后还能取这份值。

此处依赖本地内部字段，升级需核对；默认无Recorder条目。若以后启用父类record_post_step，需调整最终奖励处理的位置，避免记录到合成前的总分。

## 7. 动作变化惩罚读取哪份动作

旧action_rate使用裁剪动作；当前已保存的action_rate_clipped_l2从两个动作项的clipped_actions与previous_clipped_actions计算平方差并求和，返回(N,)；权重和RewardManager乘step_dt的行为保留。自动reset时裁剪缓存保留，与旧post_physics_step末尾写回终止步动作相对应。最新CPU检查包含150→200但裁剪后均为100时惩罚为0，及部分reset缓存规则；实现解释见[actions.py说明](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/actions.py.md)。

## 8. 验证与来源

本轮实际构造原生RewardManager/TerminationManager，核对十项调用、加速度reset、超时屏蔽、_step_reward保留以及最终合成，详见[09审查](09_阶段审查.md)。给定相同输入的公式对照不证明真实接触力、地形或时序等价。

Gym：[权重](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:100)、[compute_reward](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:707)、[_reward方法](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1153)。Lab：[官方奖励](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/rewards.py)、[RewardManager](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/reward_manager.py)。

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
| action_rate | 在轨迹上比较相邻控制时刻的裁剪动作c_t与c_(t-1)；观测可以不同 | 在任务奖励中鼓励动作时序连续 |

PPO的核心比值为：

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid o_t)}{\pi_{old}(a_t\mid o_t)}.
$$

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

反过来，当前action_rate比较裁剪动作，计算sum((c_t-c_(t-1))²)。即使网络参数完全不更新，同一个固定策略也可能在两个不同观测下连续输出+0.8、-0.8；此时新旧策略相同，PPO概率比为1，但这个通道的动作变化指标为(-0.8-0.8)²=2.56。训练中的动作还可能从分布采样，因此也有探索带来的逐步变化。

本项目的数据流为：相邻动作→action_rate惩罚→合入环境奖励r_t→计算回报/优势→PPO根据这些优势更新策略。action_rate定义任务更偏好怎样的控制行为；PPO定义怎样用采样数据更新网络，两者同时使用。该奖励不会直接覆盖当前动作或强制滤波。

本地runner也是先用当前策略采一段轨迹，再compute_returns和update；当前生成配置num_steps_per_env=16。因此一次网络训练更新与一次0.01 s控制步并非同一个周期。本次仅追读必要消费端，未迁移或调参算法。

源码：[本地RSL-RL PPO](/home/eaziel/my_venvs/lab3_sim62/.venv/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py)、[采样与更新](/home/eaziel/my_venvs/lab3_sim62/.venv/lib/python3.12/site-packages/rsl_rl/runners/on_policy_runner.py)、[当前任务PPO配置](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/config/Tita/agents/rsl_rl_ppo_cfg.py:14)。

**Q：**

> 对那个action rate我还有个问题，因为物理上，观测是连续变化的，所以可以这样做，因为环境连续，要让策略对连续变化的观测产生的响应也是连续变化的，这意味着要求不能有很大的突变，对吧！

**A：** 这是动作平滑惩罚的主要动机。机器人每0.01 s接收一次新动作，希望正常运动时动作少来回跳，以减少抖动和执行器负担。当前计算为：

$$
f_{rate}(t)=\sum_j(c_{t,j}-c_{t-1,j})^2,\qquad
r_{rate}(t)=-0.01\,f_{rate}(t)\,0.01.
$$

这里 $c_t$ 明确指裁剪后的策略动作，与当前实现的取数一致。

这里比较完整8维裁剪后的策略动作；函数没有除以dt，所以名称里的rate不代表严格计算了动作时间导数。腿动作随后映射为位置目标偏置，轮动作映射为前馈力矩；PD反馈、当前状态和限幅仍会影响最终力矩，不能把动作平滑直接等同于力矩平滑。

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

**A：** 本地源码明确串起三步：类项创建后被放入RewardManager._class_term_cfgs；环境_reset_idx调用reward_manager.reset；该方法遍历列表执行term_cfg.func.reset(env_ids=env_ids)。类项继承ManagerTermBase，因而可以由框架统一管理其状态。缓存跟随重置后的速度，避免跨回合差分。原代码和对象归属见[rewards.py状态与reset说明](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/rewards.py.md)。

**Q：**

> 以及，__call__干啥用了？？python的类名，在这里，是不是可以看作func就是构造函数__init__()？

**A：** 类名指向类对象。配置func=类对象时尚未创建奖励实例；ManagerBase调用该类创建实例，过程中执行__init__，然后把实例存到管理器内部的term_cfg.func。之后func(env)调用同一实例的__call__，每策略步计算一次。类对象、初始化方法与实例是三个对象/阶段，不能把func与__init__画等号。[源码解释](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/rewards.py.md)给出逐步代码；Python语义见[官方__init__](https://docs.python.org/3.12/reference/datamodel.html#object.__init__)和[__call__](https://docs.python.org/3.12/reference/datamodel.html#object.__call__)。

**Q：**

> 因为ruff插件警告让我写到函数体里，我不想看到黄下划线。

**A：** 已用已安装Ruff插件CLI和项目配置复现B008。它关注默认参数里的函数调用只在定义时执行一次，返回对象会被多次调用共享。采用None默认值、if分支中按需构造即可，见对应源码解释；把形参直接改成必填会改变调用接口。该方案通过项目现有lint与CPU参数/选择检查，不要求关闭规则或忽略黄线。[Ruff官方B008说明](https://docs.astral.sh/ruff/rules/function-call-in-default-argument/)。

此前追问的Python括号调用、登记对应哪句append，见[奖励源码的类调用说明](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/rewards.py.md)；多个reset的对象归属与自定义历史用途见[环境源码的step/reset说明](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/tita_velocity_env.py.md)。原问题保留；当前第10节合成代码已保存，其计分含义见下面的新问答。

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

若不乘dt，这两个合计会变成100和50，仅因调用频率不同就改变了累计量。可以把这种缩放理解为对加权得分率做矩形积分。它是框架/奖励设计约定，并非所有RL奖励都必须乘dt，也不保证改变控制频率后任务完全等价：相邻动作差等原始量可能随步长变化，折扣也受步数影响；终止这样的一次性事件也被框架统一乘dt，当前-200对应一次失败-2，改变dt后需要重新考虑事件权重。官方文档说明其用途是平衡奖励相对时间步长的尺度：[RewardManager](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html#isaaclab.managers.RewardManager)。旧Gym也在_prepare_reward_function中先把权重乘self.dt，本迁移保留这个约定。

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
