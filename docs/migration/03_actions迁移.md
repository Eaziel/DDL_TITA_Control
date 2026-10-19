# 03 Actions：从八维策略动作到腿轮力矩

更新：2026-10-18。下文的基础动作和执行器已写入项目并静态核对；动作滤波、延迟和随机化尚未迁入，未运行仿真。

这一章要保留的计算是：**六个腿关节做位置 PD，两个轮子使用力矩前馈加速度阻尼。** 旧代码在P控制分支先计算PD，再覆盖两轮力矩；下面分别保留两种公式。

## 1. Gym 原来怎么做

### 1.1 actions 是什么

策略输出 `actions`，形状为 `(num_envs, 8)`。每行是一台机器人的动作，每列是一路控制信号；这些数值经过下面的动作处理和控制器计算得到关节力矩。

旧 `step()` 和 `_compute_torques()` 合起来的执行顺序是：

```text
策略输出八维动作
  → 保存原始动作，供下一份观测使用
  → reindex 重排，按 clip_actions=100 裁剪
  → 两点动作滤波
  → 缩放
  → 物理步延迟缓冲
  → 加默认角，计算腿 PD / 轮子前馈与阻尼
  → 乘电机强度因子，裁剪总力矩
  → 写入仿真
```

动作处理和力矩计算集中在同一个环境文件中。迁移时既要拆开职责，也要保留这些运算的先后位置。

### 1.2 先把基础力矩公式取出来

下面是旧 `_compute_torques()` 的相关节选，暂时省略滤波、延迟、随机化和最终限幅：

```python
actions_scaled = actions[:, :8] * self.cfg.control.action_scale
actions_scaled[:, [0, 4]] *= self.cfg.control.hip_scale_reduction

joint_pos_target = actions_scaled + self.default_dof_pos
torques = (
    self.p_gains * (joint_pos_target - self.dof_pos)
    - self.d_gains * self.dof_vel
)
```

`[:, :8]` 选所有环境的前八列。乘法生成缩放后的新张量；`[:, [0, 4]]` 再选其中两列，额外乘一次髋缩放。旧函数内部这两列是左右 `leg_1`，不能直接把下标移到新任务。

旧参数是 `action_scale=0.5`、`hip_scale_reduction=0.5`、`Kp=40`、`Kd=1`。以 `u` 表示进入缩放的有效动作：

```text
左右 leg_1：q_des = q_default + 0.25u
左右 leg_2/3：q_des = q_default + 0.5u
六个腿关节：tau = 40(q_des - q) - dq
```

随后旧代码覆盖两轮的力矩：

```python
torques[:, [3, 7]] = (
    self.kp_factor[:, [3, 7]] * 10 * joint_pos_target[:, [3, 7]]
    - 0.5 * self.kd_factor[:, [3, 7]] * self.dof_vel[:, [3, 7]]
)
```

令随机因子为 1，两轮默认角为 0，便得到：

```text
轮子：tau = 10 * 0.5u - 0.5dq = 5u - 0.5dq
```

这条公式没有减轮子当前转角 `q`。轮子可以持续旋转，动作提供驱动力矩，速度阻尼抑制转速。

## 2. Lab 中怎样写

Lab 把上述计算分成两段：**ActionTerm 把策略数字变成关节目标，执行器把目标与关节状态变成力矩。**

### 2.1 env_cfg.py：六腿位置目标与两轮力矩前馈

当前 [env_cfg.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py) 中已有：

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

`asset_name="robot"` 取 Scene 中的机器人。`joint_names` 选择关节，`preserve_order=True` 保持列表中的次序。

位置动作执行 `raw * scale + offset`。`use_default_offset=True` 让 offset 取资产保存的默认关节角，因此目标为q_default+s*u，参考点来自保存的默认角；当前角q进入后续PD误差。轮子动作则把 `5u` 写入力矩前馈目标。

ActionManager 按条目顺序分段，所以新任务的动作排列为：

```text
[左1, 左2, 左3, 右1, 右2, 右3, 左轮, 右轮]
 └──────── 前六维位置动作 ────────┘  └─ 后两维力矩动作 ─┘
```

### 2.2 tita.py：用显式执行器计算反馈力矩

当前 [assets/tita.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/assets/tita.py) 已导入 `IdealPDActuatorCfg`，并在 `TITA_CFG` 中设置：

```python
actuators={
    "legs": IdealPDActuatorCfg(
        joint_names_expr=["joint_.*_leg_[123]"],
        stiffness=40.0,
        damping=1.0,
        actuator_effort_limit=60.0,
        joint_effort_limit=60.0,
    ),
    "wheels": IdealPDActuatorCfg(
        joint_names_expr=["joint_.*_leg_4"],
        stiffness=0.0,
        damping=0.5,
        actuator_effort_limit=15.0,
        joint_effort_limit=15.0,
    ),
},
```

这是一段 `ArticulationCfg(...)` 的构造参数。`legs`、`wheels` 是执行器分组名，不必与 ActionsCfg 条目同名；它们通过选中的关节对应起来。

IdealPD 的计算是：

```text
tau = Kp(q_des - q) + Kd(dq_des - dq) + tau_ff
```

当前没有发送非零速度目标，因此：

| 分组 | ActionTerm 提供什么 | 执行器得到的力矩 |
| --- | --- | --- |
| 六腿 | `q_des=q_default+s*u`，力矩前馈为 0 | `40(q_des-q)-dq` |
| 两轮 | `tau_ff=5u`，Kp 为 0 | `5u-0.5dq` |

`actuator_effort_limit` 对显式模型算出的总力矩限幅，包含反馈与前馈；`joint_effort_limit` 是仿真求解器侧的关节上限。当前两层均设为腿 60、轮 15 N·m，与此前静态核对的资产限矩一致。

## 3. 为什么这样迁移

**腿和轮子使用两种动作项，是因为它们的旧力矩公式不同。** 全部配置为位置动作会让轮子的累计转角进入位置误差，改变原控制方法；只给轮子力矩动作又会缺少 `-0.5dq`，因此还要配套阻尼执行器。

选择显式 IdealPD，是为了对应旧代码每个物理步在 Python 中算力矩、再提交给引擎的过程。此前 USD 的 Drive 增益静态读值为 0；仅配置位置动作、继续继承这些零增益，不能得到原来的 40/1 PD。

`scale`把策略数字映射到物理目标。这里0.25/0.5对应关节角偏移，5对应轮子力矩前馈；同样大小的策略输出，经过不同scale会产生不同的物理变化和探索幅度。裁剪、滤波、延迟在第4节分别核对。

例如在不触发限幅时：

- 髋外展动作 `u=0.2`，默认角和当前角均为 0、速度为 0：目标角为 0.05 rad，力矩为 2 N·m。
- 轮子动作 `u=0.2`，轮速为 0.4 rad/s：前馈为 1 N·m，阻尼为 -0.2 N·m，总力矩为 0.8 N·m。

以上是公式示例。旧策略从Normal分布采样动作，该分布没有固定的[-1,1]边界；角度偏移的范围还取决于动作值与裁剪配置。

新排列由关节名称明确指定。旧 Gym 还有 `[4,5,6,7,0,1,2,3]` 重排，新任务不能照搬旧 `[0,4]` 髋索引或 `[3,7]` 轮索引。观测采用相同的新顺序，旧 checkpoint 能否复用仍需完整转换。

一次策略动作保持 10 ms，显式执行器每 2.5 ms 重算力矩。此时 action 不变，反馈状态仍在变化，力矩不会必然保持不变。

## 4. 旧动作路径中尚未迁入的部分

上面的 Lab 实现只对应基础控制公式。旧默认配置还开启以下行为，不能因基础公式对应就认为动作迁移完成。

### 4.1 原始动作裁剪

Gym 在缩放之前按 `clip_actions=100` 裁剪。Lab 标准 JointAction 的 `clip` 位于缩放、偏置之后，直接填同一个 100 会改变裁剪对象和范围。这部分仍待接入。

### 4.2 两点低通滤波

旧实现为：

```python
actons_filtered = self.last_actions * 0.2 + actions * 0.8
```

`last_actions` 保存上一策略步的重排、裁剪动作，**不是上一次滤波输出**。所以它是两点 FIR，不能直接替换成递归 EMA。

输入从 0 跳到 1，先输出 0.8，下次仍输入 1 就输出 1；输入持续正负交替时，幅度由 1 减为 0.6。它减弱动作快速变化，因而能减轻位置目标突变带来的力矩跳变，同时也延缓响应。它只滤动作，不保证反馈力矩完全平滑。当前 Lab 两个动作项尚未包含此滤波。

### 4.3 物理步延迟

旧 `lag_timesteps=3` 分配 `(N,3,8)` 缓冲。每次 `_compute_torques()` 都先追加动作，再按每环境的随机索引读取：

| 索引 | 读取的样本 | 当前对应延迟 |
| --- | --- | --- |
| 0 | 两个物理步之前 | 5 ms |
| 1 | 一个物理步之前 | 2.5 ms |
| 2 | 刚写入的样本 | 0 ms |

3 是缓冲长度，实际时延为 0/1/2 个物理步，不能写成三个策略步。索引在创建时按环境采样，八个关节共用；当前旧源码未见每次 reset 重采样索引。重置会清空缓冲。Lab 侧尚未实现，选用现成延迟执行器时也要核对这些语义。

### 4.4 增益与电机强度随机化

旧 Kp/Kd 因子和电机强度范围均为 `[0.8,1.2]`，但作用位置不同：腿分别缩放位置反馈与速度反馈；轮子 `kp_factor` 缩放前馈，`kd_factor` 缩放阻尼；最后 `motor_strength` 乘整份力矩，再进行限幅。

因此，轮子名义公式虽然可写成 `0.5*(10u-dq)`，使用固定目标轮速加一个随机 Kd 会把前馈与阻尼绑定到同一随机因子，不能复现原来独立的因子。

旧 `_init_buffers()` 的 motor_strength 采样和 `_compute_torques()` 的乘法没有在这两处检查 `randomize_motor` 开关。这是当前旧源码的实际行为，不能只把布尔值改 False 就宣称关闭。Lab 侧全部这些随机化仍待实现：Event 可以负责采样，动作或执行器负责使用采样值。

旧推扰通过修改实际基座速度产生外界扰动，迁移职责落在interval Event；完整随机化来源见00表。

## 5. 当前结果与对照源码

当前动作、执行器各覆盖六腿和两轮，无重复，总动作宽度为 8；scale、默认角偏置、增益和限矩做过静态核对。尚未实例化执行器或运行环境，滤波、延迟和随机化仍为待办。

| 要查什么 | Gym | Lab |
| --- | --- | --- |
| 基础参数与力矩 | [control](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:63)、[_compute_torques](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:647) | [ActionsCfg 官方例子](/home/eaziel/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:152)、[动作配置类](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/actions_cfg.py:30) |
| 缩放与目标写入 | [step](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:295) | [JointAction.process_actions](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:168)、[JointEffortAction](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py:252) |
| PD 与限矩 | [_compute_torques](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:647) | [IdealPDActuator.compute](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:319)、[限制字段](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_base_cfg.py:39) |
| 滤波、延迟、随机化 | [_low_pass_action_filter](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:643)、[domain_rand](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:123) | 当前工程未实现；完整随机化分工见 [00](00_迁移映射索引.md#3-domain_rand-迁到哪里) |

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> 为啥rl非要有个action scale？不仅这个，其他rl task也有，比如go2（我记得好像）

**A：** RL并不要求必须有一个独立命名的action_scale。任务需要约定网络输出怎样对应物理控制量，scale是常用的映射参数。当前六腿把动作映射成默认角附近的角度偏移，两轮映射成力矩前馈；第3节的0.2动作例子展示了其数值影响。

**Q：**

> 不知道推机器人算action吗？凭我的记忆和直觉，commandcfg类更合适？不太记得了。

**A：** 这里先按谁在改变什么区分：Actions是策略给机器人的控制信号；Commands是希望机器人完成的速度目标；推扰是外部直接改变机器人的实际状态或施力。旧_push_robots改实际水平根速度，对应interval Event。

**Q：**

> lag_timesteps是啥？就是策略延迟？

**A：** 旧lag_timesteps=3指定动作历史缓冲长度。缓冲在每个物理步推进，当前随机索引对应0、1、2个物理步，即0、2.5、5 ms的控制输入延迟。策略仍每10 ms输出一次动作。第4.3节列出了每个索引读取的样本。

[返回迁移入口](README.md) · [查看来源与覆盖表](00_迁移映射索引.md)
