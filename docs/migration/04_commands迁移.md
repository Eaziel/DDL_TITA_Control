# 04 Commands：速度目标、航向目标与重采样

更新：2026-10-18。当前 CommandsCfg 及环境登记已写入并静态核对；补充配置对象、运行时生成器及命令张量的对应。平移死区和命令更新时序仍有差异，未运行仿真。

Commands 回答的是：**这段时间希望机器人怎么运动？** 它给策略任务目标；Actions 才是策略为完成目标输出的八维关节控制信号。

## 1. Gym 原来怎么做

### 1.1 commands 保存四个目标量

旧 `self.commands` 的形状是 `(num_envs,4)`，每行对应一个环境：

| 列 | 内容 | 单位 |
| --- | --- | --- |
| 0 | 机身前向目标速度 vx | m/s |
| 1 | 机身侧向目标速度 vy | m/s |
| 2 | 目标偏航角速度 yaw_rate | rad/s |
| 3 | 世界系目标航向 heading_target | rad |

它保存目标，不保存机器人实际测得的速度。例如给一个环境设置 `vx=0.6`、`vy=0`、目标航向 90°，任务就是“沿机身前向走 0.6 m/s，同时转向 90°”。机身前向会随着朝向旋转，这不等于始终沿世界 X 轴运动。

### 1.2 _resample_commands 为指定环境换一组目标

旧配置给出：

```python
resampling_time = 10.0
heading_command = True
# ranges 中：
lin_vel_x = [-1.0, 1.0]
lin_vel_y = [-1.0, 1.0]
ang_vel_yaw = [-1.0, 1.0]
heading = [-3.14, 3.14]
```

`_resample_commands(self, env_ids)` 从这些范围中均匀抽取新值，写入 `env_ids` 指定的行。例如 `env_ids=[2,17]`，只给第 2、17 个环境换目标，其他环境保持原目标。调用方提供env_ids，函数为这些行抽取新的目标数值。

调用发生在两种情况：

1. 环境重置，为新回合分配任务。
2. 连续运行达到重采样周期，给相应环境换任务。

旧每步检查：

```python
env_ids = (
    self.episode_length_buf
    % int(self.cfg.commands.resampling_time / self.dt) == 0
).nonzero(as_tuple=False).flatten()
self._resample_commands(env_ids)
```

这里 `self.dt=0.01 s`，所以 `10 / 0.01=1000`：每个环境正常连续运行 1000 个策略步后换一次目标。10 s 是仿真时间，失败重置的环境可以提前换目标。

维持一段时间能让机器人经历加速、转向和跟踪；反复抽取目标则覆盖不同任务。这解释了这种设计的用途，但源码没有解释为什么恰好取 10 s，也没有证明它最优。

### 1.3 航向误差每步换算成目标转速

开启 heading 模式后，重采样抽取的是目标朝向；目标转速由旧回调每个策略步重新计算：

```python
forward = quat_apply(self.base_quat, self.forward_vec)
heading = torch.atan2(forward[:, 1], forward[:, 0])
self.commands[:, 2] = torch.clip(
    0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0
)
```

计算顺序是：取得当前航向 → 求最短航向误差 → 乘比例系数 0.5 → 限制到 ±1 rad/s。

例如目标为 90°，当前为 0°：误差 π/2 rad，目标转速约 0.785 rad/s。接近到 85° 时，目标转速减为约 0.0436 rad/s；到 90° 时归零。当前 170°、目标 -170° 时，`wrap_to_pi` 给出最短误差 +20°，避免要求绕 -340°。

这叫“航向外环”，因为它根据朝向反馈生成下一层的速度目标：

```text
目标航向、当前航向
  → 目标转速
  → 策略根据速度目标与自身观测输出关节动作
  → 关节控制器与物理仿真
  → 新的实际航向
```

0.5 是航向误差到目标转速的比例系数，不是关节 PD 的 Kp。目标朝向可以保持 10 s，但当前朝向一直变化，因此目标转速仍需每 0.01 s 更新。

### 1.4 目标如何参与训练

旧基础观测拼入 `commands[:, :3]`，策略因此知道要追踪什么。速度奖励再比较目标速度与实际机身速度，评价本次运动结果。

第四列航向不直接进入默认 Actor 的当前33维输入；它先通过上述公式变成第三列目标转速。Commands 本身也不会把机器人实际速度改成目标值。

## 2. Lab 中怎样写

当前 [env_cfg.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py) 中已有：

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

同一文件的 `VelocityEnvCfg` 中登记：

```python
commands: CommandsCfg = CommandsCfg()
```

`base_velocity` 是命令条目的名字。后续观测和奖励通过这个名字向 CommandManager 取目标：

```python
env.command_manager.get_command("base_velocity")
```

返回值为 `(N,3)`，分别是 vx、vy、yaw_rate。航向目标由运行时对象单独保存，不需要再对外返回第四列。

### 2.1 各参数怎样对应旧规则

| Lab 字段 | 作用与本任务取值 |
| --- | --- |
| `asset_name="robot"` | 从 Scene 取机器人，用其实际航向做反馈 |
| `resampling_time_range=(10,10)` | 每次保持时间固定为 10 s，重置也会重采样 |
| `heading_command=True` | 允许用目标航向误差生成目标转速 |
| `heading_control_stiffness=0.5` | 保留旧航向比例系数 |
| `rel_heading_envs=1.0` | 每次重采样时，以概率1将环境标记为航向模式，即全部使用航向反馈 |
| `rel_standing_envs=0.0` | 不额外抽取强制静止环境；命中静止标记时，三个速度目标会全部归零 |
| `ranges.lin_vel_x/y` | 机身前向、侧向目标范围，±1 m/s |
| `ranges.ang_vel_z` | 直接转速采样范围，也是航向反馈转速的限幅范围 |
| `ranges.heading` | 世界系目标航向范围，±3.14 rad |
| `debug_vis=True` | 显示目标与实际速度箭头，不参与目标计算 |

`rel_heading_envs=0.5` 表示每次抽样时每个环境有一半概率使用航向模式，并不保证恰好一半数量。这里取 1.0，才对应旧 TITA 全部使用 heading 模式的配置。

### 2.2 转速计算在哪里执行

配置中的 `UniformVelocityCommandCfg` 指向运行时 `UniformVelocityCommand`。Lab 的 `_update_command()` 实际执行：

```python
if self.cfg.heading_command:
    env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
    heading_error = math_utils.wrap_to_pi(
        self.heading_target[env_ids] - self.robot.data.heading_w.torch[env_ids]
    )
    self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * heading_error,
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
    )
```

因此你在自己的 cfg 中只看到系数和开关，实际运算已经由库类提供。`CommandManager.compute(dt=step_dt)` 每个策略步调用命令项：到期时重采样，随后更新航向反馈转速；静止标记在后面将相应环境的全部速度目标清零。

## 3. 为什么这样迁移

旧 `_resample_commands()` 的采样和计时，对应 Lab 的命令项与 CommandManager；旧回调中的航向计算，对应 `UniformVelocityCommand._update_command()`。这两部分已有现成实现，所以无需再写一个同名采样函数或复制整段回调。

旧post_physics_step在每个策略步主动调用_post_physics_step_callback，其中包含命令更新、高度读取与推扰。迁移后分别交给以下模块：

| 旧回调里的工作 | Lab 中由谁负责 | 当前状态 |
| --- | --- | --- |
| 到期抽取新目标、每步更新目标转速 | CommandManager / UniformVelocityCommand | 基础配置已接入 |
| 更新局部地面高度 | Scene 中的 RayCaster，MDP 再取数 | 传感器已配置，待运行 |
| 定时推机器人 | interval Event | 尚未迁入 |
| disturbance 施力 | 相应 Event | 旧默认关闭，当前保持关闭 |

这解释了为什么 Lab 代码比旧回调短：通用实现由框架提供，任务仍需指定参数，并核对没有被现成类覆盖的特殊规则。

## 4. 尚未对齐的两条规则

### 4.1 0.2 m/s 平移死区

旧采样末尾还有：

```python
self.commands[env_ids, :2] *= (
    torch.norm(self.commands[env_ids, :2], dim=1) > 0.2
).unsqueeze(1)
```

当 `(vx,vy)` 的模不大于 0.2 m/s 时，同时把 vx、vy 归零，yaw_rate 不受这条规则影响。当前标准命令项没有实现这一步，后续需补专用规则。

它不等于随机抽取 standing 环境：一个按采样速度大小判断，只清平移；另一个按概率抽样，并清三个速度分量。因此不能靠改 `rel_standing_envs` 对齐。

### 4.2 命令更新相对奖励的时序

旧 Gym 和 Lab 默认顺序不同：

```text
Gym：物理 → 更新命令 → 终止/奖励 → 重置 → 下一观测
Lab：物理 → 终止/奖励 → 重置 → 更新命令 → 下一观测
```

旧 Gym 在评分前更新命令，所以换目标的那一帧，评分目标可能不同于刚执行动作所依据的上一份观测。Lab 默认奖励先使用原命令，再给下一份观测更新目标。这是实际行为差异，参数相同不会自动消除，后续还要作明确处理。

当前回合已设置为20 s，两项速度跟踪奖励已按08章写入。10 s命令保持与回合结束可以在同一连续回合中出现；实际切换、奖励时序与跟踪表现尚待运行检查。

## 5. 对照源码与验证边界

本章配置和 `VelocityEnvCfg.commands` 登记做过静态核对，尚未运行命令项或验证跟踪。

| 要查什么 | Gym | Lab |
| --- | --- | --- |
| 参数与采样 | [commands 配置](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:76)、[_resample_commands](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:1103) | [官方 CommandsCfg](/home/eaziel/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:134)、[配置类](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/commands_cfg.py:35) |
| 航向反馈 | [_post_physics_step_callback](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:536) | [_resample_command / _update_command](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py:157) |
| 调用与时序 | [post_physics_step](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:409) | [CommandTerm.compute](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/command_manager.py:145)、[环境 step](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py:205) |
| 观测取目标 | [compute_observations](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:335) | [generated_commands](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py:775) |

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> base_velocity，我们没有指定取第几个数据，他怎么知道track_ang_vel_z_exp就是wz呢？

**A：** 列索引写在奖励函数内部。本地track_ang_vel_z_exp执行：

```python
ang_vel_error = torch.square(
    env.command_manager.get_command(command_name)[:, 2]
    - asset.data.root_ang_vel_b.torch[:, 2]
)
```

当前命令项由UniformVelocityCommand生成，列顺序为[vx_cmd,vy_cmd,wz_cmd]；函数用[:,2]取得目标wz，再与实际三维角速度[wx,wy,wz]的第三列比较。command_name选择命令项，函数中的索引选择该项的列。这一组合依赖双方相同的数据格式约定。

源码：[角速度奖励](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/rewards.py:345)。

<a id="23-base_velocity配置对象与实际命令张量"></a>

**Q：**

> 以及，base_velocity我得加点注释，这个玩意最终是啥内容，shape是啥，看代码我看不来，因为鬼知道那个类最后被咋处理了

**A：** 这里按三个对象顺着看：配置中的base_velocity是UniformVelocityCommandCfg对象；CommandManager根据它创建并以同名登记UniformVelocityCommand对象；运行时生成器维护速度目标张量。

```python
# UniformVelocityCommand.__init__
self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
self.heading_target = torch.zeros(self.num_envs, device=self.device)
```

生成器的command属性返回vel_command_b；CommandManager.get_command再返回该属性：

```python
# CommandManager.get_command
return self._terms[name].command
```

因此get_command("base_velocity")得到torch.Tensor，shape=(N,3)，每行[vx_cmd,vy_cmd,wz_cmd]；xy单位m/s，wz单位rad/s，均以机身系表达。目标航向另存于heading_target，shape=(N,)，当前由航向误差算出的wz写回第三列。

可加在现有base_velocity赋值上方的注释：

```python
# 速度命令生成器的配置，由CommandManager以名字"base_velocity"登记运行时生成器。
# get_command("base_velocity")返回torch.Tensor，shape=(num_envs, 3)。
# 每行[vx_cmd, vy_cmd, wz_cmd]：机身系前向/侧向目标线速度(m/s)、绕z轴目标角速度(rad/s)。
# heading_target单独保存在生成器中；当前wz_cmd由航向误差计算，输出仍为三列。
```

源码：[创建张量与command属性](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py:80)、[get_command](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/command_manager.py:368)、[创建命令项](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/command_manager.py:394)。

**Q：**

> 不是，那如果命令张量他第三列不是wz呢？？比如我有很多command

**A：** 那这个现成奖励就会读取错误的分量。当前实现只按[:,2]取数，不检查列名、单位或物理含义；形状允许时可能计算错误而不报错。

多个命令项各自维护自己的张量。例如另外自定义高度和目标位置命令后，可以分别读取：

```python
velocity = env.command_manager.get_command("base_velocity")    # (N,3): vx,vy,wz
height = env.command_manager.get_command("base_height")         # 假设自定义为(N,1)
position = env.command_manager.get_command("target_position")  # 假设自定义为(N,3)
```

CommandManager按名字取其中一项；增加其他命令项不会改变base_velocity的三列。若某个自定义命令本身改成[vx,vy,height,wz]，就应选择适配该布局的奖励函数，例如按[:,3]取目标wz。当前UniformVelocityCommand与track_ang_vel_z_exp的布局约定匹配。

[返回迁移入口](README.md) · [查看来源与覆盖表](00_迁移映射索引.md)
