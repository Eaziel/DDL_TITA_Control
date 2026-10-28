# 18 修正cost接线，检查真实环境输出

**阅读目标：** 本页保留四处接线错误如何修正，以及真实cost接口运行证明了什么；临时打印属于历史排查方法。

更新：2026-10-22。内容纯AI生成，由人类要求生成，未经人类修改。

四处修正均已由用户写入。用户PhysX/cuda:0、4环境/200零动作步真实运行通过，首步与第200步cost形状均(4,6)，没有断言错误，正常清理并回到shell。证据见[真实日志](../audits/2026-10-22_cost_first_run/user_log.txt)。以下第1—4节保留修正与检查方法，当前不用重复修正。此前原生CPU内存对照保留在[接线检查](../audits/2026-10-22_cost_wiring/result.json)，与本次真实证据分别看待。助手未修改TITA_Lab。

## 1. 管理器准备方法名

修正前cost_manager.py写的是：

```python
def _prepare_term(self) -> None:
```

应改为：

```python
def _prepare_terms(self) -> None:
```

父类声明并调用的是_prepare_terms，末尾带s。方法名不同就没有实现父类要求的抽象方法，创建管理器时直接报Can't instantiate abstract class，而不是仅仅漏算某项。其余方法体保留。

## 2. 分别创建k和d

修正前只有k_values且view成(1,1,6)，没有d_values。用户已按下列初始化段修正：

```python
# 六项cost的算法参数；k为(1,6)，d为(1,1,6)。
self.k_values = torch.tensor(
    [term.k for term in self._term_cfgs], device=env.device
).view(1, -1)
self.d_values = torch.tensor(
    [term.d for term in self._term_cfgs], device=env.device
).view(1, 1, -1)
```

这影响未来算法的接口；当前compute不读取k/d，不能只靠每步cost正常就认定这部分对齐。另给现有_episode_sums补关键注释：

```python
# 每台机器人在当前回合内各项cost的累计值，(N,6)；reset只清选中行。
self._episode_sums = torch.zeros_like(self.cost_buf)
```

## 3. 速度项应选择关节

修正前dof_vel_limits误写了sensor_cfg以及k=0.1。joint_velocity_limit_cost要求asset_cfg，读取机器人关节速度；旧Gym该项k=0.3。只替换CostsCfg中的这一项：

```python
dof_vel_limits = CostTerm(
    func=mdp.joint_velocity_limit_cost,
    k=0.3,
    params={
        "asset_cfg": SceneEntityCfg(
            "robot", joint_names=TITA_JOINT_NAMES, preserve_order=True
        ),
    },
)
```

本地Lab在准备函数参数时已复现：expects mandatory parameters ['asset_cfg'] ... received ['sensor_cfg']。传感器选择保留在feet_contact_forces和stumble两项。

## 4. 临时显示实际输出

官方zero_agent会调用env.step，但不接收并显示返回的cost。为本次运行，在TitaVelocityEnv.step现有extras["costs"]赋值之后、return之前临时加入：

```python
# 临时检查：每步返回N台机器人、六项cost，且全部有限、非负。
costs = extras["costs"]
assert costs.shape == (self.num_envs, 6)
assert torch.isfinite(costs).all(), "Non-finite cost detected."
assert (costs >= 0.0).all(), "Negative cost detected."

# 只打印首步和第200步，避免每一步都刷屏。
if self.common_step_counter in (1, 200):
    print("[COST] terms:", self.cost_manager.active_terms)
    print("[COST] shape:", tuple(costs.shape))
    print("[COST] env0:", costs[0].detach().cpu().tolist())
```

这些断言检查数值是否满足接口要求，不证明物理量与旧Gym完全等价。cost全零允许；后五项原始输出是0/1，乘dt后单步为0或0.01，第一项加速度cost是连续非负值。

激活lab3_sim62后，在Tita_Lab目录运行同一检查：

```bash
isaaclab zero_agent \
  --task TitaLab-Velocity-Tita \
  --num_envs 4 \
  --max_steps 200 \
  --device cuda:0 \
  --viz none \
  physics=isaacsim_physx
```

200步对应2秒仿真时间；检查环境构造、reset/step和六列真实输出。若期间发生自动reset，也会经过本次cost前采集/后重置流程。此次长度不覆盖15秒推扰或20秒超时，仍需后续长一点的运行。本轮未替用户启动模拟器。

完成标准：正常结束，首步/200步shape均为(4,6)，列名为acc_smoothness/dof_vel_limits/feet_contact_forces/pos_limit/stumble/torque_limit，无断言错误。之后依据用户日志决定下一步；尚不把接口检查当作已能训练或平衡。

## 5. 当时结果与后续长运行检查（未附完整日志）

首步环境0为[0.09465301036834717,0,0,0,0,0]：第一项加速度超额cost为正，其余五项为零。第200步六项为零，表示该步这些cost均为零。它们是已乘策略dt的输出。

本次200步共2秒，不能确认15秒推扰；日志也未记录自动reset次数。当时另提出过2200步的较长检查，但未收到完整日志，不记为完成。以后若需要覆盖15秒推扰，可以单独安排长运行；本次不要求重加历史打印。22秒总时长不保证一个回合存活到20秒，超时是否出现需看回合记录。

包装接口和actor已另行写入，见22—24。本次联调使用[26](26_actor与环境联调.md)，之后暂停新增内容；启动或接口成功仍不等于训练成功。

## 补充问答

**Q：** 能不能来个可视化的？还是说没必要，或者不能？因为0agent？不能有一个agent吗？

**A：** 可以。visualizer负责显示仿真，agent负责提供动作，两者分别选择。zero_agent表示输出零动作的agent，num_envs决定并行机器人数量。当前TITA的零动作对应腿默认位置目标、轮子零前馈力矩，显式PD仍然执行反馈；因此机器人仍受重力、接触与执行器力矩影响，可能移动或跌倒。

建议先开一台机器人观察出生姿态、接地和跌倒/reset。激活lab3_sim62后，在Tita_Lab目录运行：

```bash
isaaclab zero_agent \
  --task TitaLab-Velocity-Tita \
  --num_envs 1 \
  --device cuda:0 \
  --viz kit \
  physics=isaacsim_physx
```

该观察命令省略max_steps，关闭窗口或Ctrl+C结束；也可加--max_steps 2200做有界检查。num_envs=1时cost形状自然变为(1,6)。此命令按本地源码和官方文档核对，助手未运行GUI，用户尚未提供可视化结果。

若要看随机动作，把zero_agent替换为random_agent；本地实现逐步均匀采样[-1,1]，用于观察动作响应，不包含学会的平衡控制。play才从checkpoint读取已训练策略，当前TITA迁移尚未完成训练接入。

依据：[本地快速入门](/home/eaziel/IsaacLab-main/docs/source/isaaclab/quickstart.rst)、[窗口参数](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/app/app_launcher.py)、[零/随机动作策略](/home/eaziel/IsaacLab-main/source/isaaclab_rl/isaaclab_rl/entrypoints/simple_agents.py)、[官方快速入门](https://isaac-sim.github.io/IsaacLab/develop/source/setup/quickstart.html)。网页为develop，具体参数以本地读取的版本为准。

历史续记：2026-10-22曾出现Kit窗口中机器人不可见。随后用户写入asset跟随相机并报告可见，见[相机证据](../audits/2026-10-22_kit_visibility/README.md)；该故障已不再作为当前阻塞。200步cost日志与后续可见性报告分别保留，不把任一结果扩大为训练通过。
