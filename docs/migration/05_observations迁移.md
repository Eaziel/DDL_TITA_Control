# 05 Observations：当前33维、噪声和过去十帧

**目标与结果：** 每策略步得到一行33维当前观测，同时保存按旧到新排列的过去十帧。

整理：2026-10-26。policy33、history330及其他三组已写。此章集中讲当前帧、噪声和历史；scan/dynamics见13，六维与586布局见14。当前PPO模板仍只读policy33。 当前证据与整体状态以[项目记忆](../../PROJECT_MEMORY.md)为准。

## 1. Gym返回哪些数据

旧compute_observations()依次拼接：

| 当前帧列 | 内容 | 缩放 | 最终加性噪声半宽 |
| --- | --- | --- | --- |
| 0:3 | 机身角速度 | 0.25 | 0.2 |
| 3:6 | 机身系重力单位方向 | 1 | 0.05 |
| 6:9 | 目标[vx,vy,wz] | [2,2,0.25] | 0 |
| 9:17 | 八关节q-q_default，轮角先置0 | 1 | 0.01 |
| 17:25 | 八关节角速度 | 0.05 | 0.075 |
| 25:33 | 完整原始策略动作 | 1 | 0 |

当前帧共33维。旧完整观测还含187维地形高度、36维特权量和330维历史，共586维；当前Lab已按五组分别输出，586拼接函数已建立；本章聚焦policy和history，其他组见13、14章。旧模型的Actor、Critic、Cost对这些区段取数不同，后续迁模型时再接完整消费关系。

## 2. Lab配置当前帧

```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):  # 单环境33维。
        """Observations for policy group."""

        # observation terms (order preserved)
        # Lab API 先加噪声后缩放，而Gym先缩放观测后加噪声
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.8, n_max=0.8, operation="add"),  # 加噪声
            scale=0.25,
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05, operation="add"),  # 加噪声
            scale=1.0,
        )

        velocity_commands = ObsTerm(  # 不加噪声
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )

        # 八个关节的（当前角度-默认角度）。例如膝关节默认 −1.5 rad，当前 −1.4 rad，结果是 +0.1
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_zero_wheels,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=TITA_JOINT_NAMES,
                    preserve_order=True,
                ),
                "wheel_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=["joint_left_leg_4", "joint_right_leg_4"],
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01, operation="add"),  # 加噪声
            scale=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )
        # 八个关节的实际角速度，统一乘 0.05，对应旧任务的速度缩放
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=TITA_JOINT_NAMES,
                    preserve_order=True,
                ),
            },
            noise=Unoise(n_min=-1.5, n_max=1.5, operation="add"),  # 加噪声
            scale=0.05,
        )

        # 环境接收的完整八维动作，处于动作项缩放之前
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True  # 启用噪声
            self.concatenate_terms = True  # 告诉观测管理器：把这一组中的各个观测项合并！
            # 没有修改 concatenate_dim，所以使用默认值 -1，沿最后一维拼接。

    # 历史观测组
    @configclass
    class HistoryCfg(ObsGroup):  # 单环境330维。
        """Past proprioceptive observations."""

        proprio = ObsTerm(func=mdp.proprio_history)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()  # 把该观测组命名为 policy，因此最终通过 obs["policy"] 取得整组张量。
    history: HistoryCfg = HistoryCfg()  # obs["history"]
```

ObservationManager按字段顺序调用函数，得到(N,d)张量，再处理噪声和缩放，最后沿最后一维拼接。policy字段是运行时字典的键，PolicyCfg是配置类名。

## 3. 为什么噪声参数不同

Gym先缩放后加噪声；本地Lab按noise→clip→scale处理。要使最终噪声一致，Lab的噪声宽度必须除以缩放系数：

对本例未额外裁剪的项，原始变量为 $x$、缩放为 $s>0$、原始噪声半宽为 $b$：

$$
y=s(x+\epsilon),\quad\epsilon\sim\mathcal U(-b,b)
\quad\Longrightarrow\quad y=sx+\epsilon',\quad\epsilon'\sim\mathcal U(-sb,sb).
$$

角速度取 $s=0.25,b=0.8$，最终噪声半宽0.2；关节速度取 $s=0.05,b=1.5$，最终半宽0.075。若中间还有限幅，应使用完整顺序 $y=s\,\operatorname{clip}(x+\epsilon,l,u)$，不能把限幅随意移到噪声前面。

gravity和关节位置的scale为1，范围直接沿用。命令与动作不加噪声；轮角先置零后加位置噪声，所以最终轮角列含纯噪声。enable_corruption控制本组是否使用配置的noise。

## 4. 轮角屏蔽放在MDP

```python
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
```

`q-q_default`生成新张量，先用机器人原始joint_ids定位两轮并置零，再按明确的八关节顺序取列。这样不会修改资产本身的关节角。无限转动的轮相位在旧任务中被舍弃，轮速仍保留。

## 5. 过去十帧怎样对应Gym

Gym先把旧历史拼进输出，再把当前帧写进缓存。因此返回o_t时，历史是o_(t-10)…o_(t-1)，不含o_t；开局时按episode_length<=1填满，reset清对应环境。

Lab的内置history按自己的采样时机工作，还会按项存储。当前方案显式保存(N,10,33)整帧缓存，由环境控制更新：

```text
上次返回：policy=o20，history=10…19
下次step开头：将已有o20写入缓存 → 11…20
父类推进物理，生成o21
本次返回：policy=o21，history=11…20
```

| 代码 | 职责 |
| --- | --- |
| ProprioHistory | 保存、读取、更新、清理缓存 |
| TitaVelocityEnv.__init__ | 在父类建立ObservationManager之前准备缓存 |
| TitaVelocityEnv.step | 每策略步在父类step之前保存已有policy |
| TitaVelocityEnv._reset_idx | 父类重置后清相应缓存行 |
| mdp.proprio_history | 只读缓存并展开为(N,330) |
| ObservationsCfg.HistoryCfg | 登记只读观测函数 |

配置声明“用哪个实现”；状态对象保存真正的数据；环境安排更新时间。三者共同完成历史功能。缓存源码及Torch操作详见[observation_history.py](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/observation_history.py.md)，环境完整代码见[tita_velocity_env.py](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/tita_velocity_env.py.md)。

## 6. 初始化为什么也会调用观测函数

ObservationManager准备条目时执行term_cfg.func(env, **params)，用返回值的shape确定观测维度。这时mdp.proprio_history已经会被调用，所以缓存必须先存在。读取函数本身不更新缓存，重复读也不会多推进一帧。

本轮核验了40次历史转移、两次部分reset、只读重复调用与tuple整数索引。一般的首次reset没有旧Gym内部暖机step；旧自动reset后还会读到部分预计算缓存，Lab按新状态取数，详见09 F5。基础环境初始化已有运行日志；最新观测的真实数值和完整时序仍待验证。

## 7. 其他观测与训练接入的分工

[13](13_训练观测迁移.md)的scan187/dynamics30和[14](14_特权状态与观测布局.md)的state6及586拼接/末端裁剪均已保存，当前代码CPU检查通过。当前普通PPO的actor和critic都只选policy组；新布局函数尚未接适配器。裁剪放在拼接输出端，内部policy/history仍保留未裁剪值。

当前文件检查和证据集中见[09审查](09_阶段审查.md)。

## 8. 对照源码

Gym：[compute_observations](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:335)。Lab：[ObservationManager](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/observation_manager.py)、[官方MDP观测](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/observations.py)。

## 补充问答

问题保留用户原句；回答按对应代码整理。

<a id="33-enable_corruption-控制什么"></a>

**Q：**

> 回答一下enable_corruption是不是让noise这一项被考虑了？类似源码的if flag 加noise的计算？
> if so,那为啥api不叫enable noise？

**A：** 是。当前ObservationManager在准备观测项时处理：

```python
if not group_cfg.enable_corruption:
    term_cfg.noise = None
```

True保留已配置的noise，False移除它；命令和动作项本来没配noise，所以开启后也不会增加噪声。scale、clip、modifiers和history按各自配置处理。这个开关在初始化准备阶段生效。

corruption表示观测受到扰动。当前源码没有说明作者为什么选这个API名字，因此只能确认行为，不能据名字推导额外功能。

**Q：**

> 当前默认关节速度为零，两者目前数值相同。
> 这句话啥意思？？rel是啥含义？

**A：** rel是relative，表示相对默认参考值：joint_vel_rel=dq-dq_default；joint_vel=dq。当前dq_default=0，因此实际速度为2 rad/s时，两者都返回2。这里为零的是保存的参考值，实际运动仍由dq给出。位置同理，joint_pos_rel=q-q_default。

**Q：**

> 好了。轮子位置指的是轮子旋转角度？确实不需要这个，轮子无限转圈。是这样理解的吧！

**A：** 对，这里指轮轴转动关节的角度。当前任务沿用旧TITA的特征定义：屏蔽两轮转角、保留两轮转速。清零发生在观测的新张量中，随后仍会加位置噪声，所以最终轮角两列可能是纯噪声。是否需要轮子相位取决于任务；本次迁移保持旧任务选择。

**Q：**

> 我没搞懂。为啥会调用这个？

**A：** 你问的是父类初始化ObservationManager时为何执行proprio_history。调用入口就是自己登记的这一项：

```python
proprio = ObsTerm(func=mdp.proprio_history)
```

func保存函数对象。管理器准备各项时实际执行：

```python
obs_dims = tuple(term_cfg.func(self._env, **term_cfg.params).shape)
```

这里self是ObservationManager，self._env才是TitaVelocityEnv。处理history/proprio项时，保存的func就是mdp.proprio_history，它接收当前环境、读取缓存并返回(N,330)，管理器据此确定维度。因此该调用发生在父类构造期间，缓存需要先创建。类型注解torch.Tensor只声明返回类型，列数由真实返回值的.shape提供。

源码：[准备观测项](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/observation_manager.py)。
