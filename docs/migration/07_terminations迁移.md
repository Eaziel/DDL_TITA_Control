# 07 Terminations：一个回合什么时候结束

**目标与结果：** 每环境输出失败/超时标志，让框架在正确时刻结束并重置该回合。

整理：2026-10-26。接触失败与超时已写。四处额外碰撞关闭、恢复自碰撞后用户报告正常；完整训练未验证。已结束的诊断过程收在文末折叠区，不是当前操作步骤。 当前证据与整体状态以[项目记忆](../../PROJECT_MEMORY.md)为准。

## 1. Gym 原来怎样做

旧 `asset.terminate_after_contacts_on=["base"]` 选择机身，`check_termination()` 判断当前接触力：

```python
self.reset_buf = torch.any(
    torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
    dim=1,
)
self.time_out_buf = self.episode_length_buf > self.max_episode_length
self.reset_buf |= self.time_out_buf
```

每环境得到一个布尔值：机身接触力模大于1 N，或者回合超时，就重置该环境。轮子正常接触地面不会仅因这一条触发终止。

回合时长来自父类env配置的20 s，策略周期为0.01 s，最大步数为ceil(20/0.01)=2000。旧代码使用严格大于比较，Lab原生time_out使用大于等于；本次保留20 s参数、采用Lab标准边界，比较符的一策略步差异单独记录。

对第n个环境，令 $\mathcal B$ 为选中的机身刚体，当前失败判据是：

$$
\mathrm{failed}_n=\mathbf1\{\exists b\in\mathcal B:\lVert\mathbf F_{n,b}^w\rVert_2>1\ \mathrm N\}.
$$

超时单独由策略步数判断，随后done对两者取或；这样训练侧仍能区分失败与时间截断。

## 2. Lab 中怎样写

### 2.1 env_cfg.py 声明两条结束条件

当前env_cfg.py已将TerminationsCfg替换为以下两项，移除了Cartpole的cart_out_of_bounds：

```python
@configclass
class TerminationsCfg:
    """Conditions that end a TITA episode."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact_current,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link"]),
            "threshold": 1.0,
        },
    )
```

time_out是Lab已有函数；illegal_contact_current是下面新增的任务函数。VelocityEnvCfg已有 `terminations: TerminationsCfg = TerminationsCfg()`，保留即可。

同文件VelocityEnvCfg.__post_init__中的当前时长为：

```python
self.episode_length_s = 20.0
```

### 2.2 新增mdp/terminations.py计算当前接触终止

源码路径：`Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/terminations.py`。用户已创建，以下代码与当前保存实现的计算一致（排版与docstring略有不同）：

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def illegal_contact_current(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return whether selected bodies currently exceed a contact-force threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_w = contact_sensor.data.net_normal_forces_w.torch
    selected_forces_w = forces_w[:, sensor_cfg.body_ids, :]
    force_magnitudes = torch.linalg.vector_norm(selected_forces_w, dim=-1)
    return torch.any(force_magnitudes > threshold, dim=1)
```

SceneEntityCfg按传感器中的身体名字base_link选择机身。管理器先将身体名字解析为body_ids，函数按索引取对应的力。

按当前Scene静态推导，全部接触力形状(N,9,3)；选一个base后为(N,1,3)，求范数得到(N,1)，阈值比较并沿身体轴any，最后为(N,)布尔张量。实际身体绑定和传感器输出仍需运行验证。

### 2.3 在mdp/__init__.pyi导出

当前.pyi保留版权头，并登记以下导出：

```python
__all__ = [
    "joint_pos_target_l2",
    "joint_pos_rel_zero_wheels",
    "proprio_history",
    "illegal_contact_current",
]

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import joint_pos_rel_zero_wheels, proprio_history
from .rewards import joint_pos_target_l2
from .terminations import illegal_contact_current
```

本项目lazy_export读取.pyi，加入导出后cfg才能访问mdp.illegal_contact_current。

## 3. 为什么这样迁移

### 3.1 为什么不用现成illegal_contact

本地Lab的illegal_contact使用net_normal_forces_w_history，并取历史力模最大值；当前Scene保存3次物理步采样。旧Gym判断的是本次读取的当前力。直接换成现成项会扩大采样窗口，所以本次加一个任务内小函数，使用net_normal_forces_w当前帧。

仍沿用当前PhysX公开的聚合法向接触力接口；与旧接触报告分量、碰撞几何的差异继续保留在02/00章，不以这个函数宣称整个接触模型等价。传感器history_length=3保持原配置，可供后续其他计算使用。

### 3.2 超时分类与回合时长

func返回是否结束；TerminationTermCfg.time_out告诉管理器该项是时间截断，通常返回为truncated。机身接触项默认time_out=False，属于任务终止terminated。它们都会结束回合，但对终止惩罚和学习程序的末尾价值处理有区别。20 s时长来自episode_length_s。

### 3.3 终止与重置怎样连接

```text
每个策略步推进物理
  → TerminationManager计算两项并取逻辑或
  → 对结束的env_ids调用_reset_idx
  → 父类执行reset Events，放回关节与机身状态
  → 当前环境子类清相应历史缓存
```

Terminations决定何时结束，reset Events决定新回合从什么状态开始。不需要手写一段发现True后调用Event函数的循环。

## 4. 当前状态与检查

- EventCfg的两个TITA重置项和VelocityEnvCfg.events登记已与06章提案做AST比对，一致；基础环境的公开reset已成功执行，接触失败的实际触发仍待验证。
- 本章函数、导出、两条Terminations和20 s时长均已保存；配置、函数入口和.pyi导出静态检查通过。
- 2026-10-18直接抽取用户保存的函数做独立CPU检查：只给轮子大力不触发、机身向量范数超过1 N触发、恰好1 N不触发；结果为[False,True,False]，输出(3,)bool。报告在`/tmp/tita_termination_review_20261018/saved_code_result.json`，没有导入项目或验证真实传感器。此前提案检查另见2026-10-17项目历史。
- 当前十项奖励、startup/reset/interval事件与六项cost均已写入，四环境/200步真实cost接口通过；训练消费和NP3O尚未接入，尚无完整训练或复现结果。

## 5. 对照源码

| 内容 | 本地入口 |
| --- | --- |
| Gym结束判断 | [check_termination](/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py:699) |
| Gym机身选择与20 s | [TITA asset](/home/eaziel/Downloads/tita_rl-master/configs/tita_constraint_config.py:90)、[父类env](/home/eaziel/Downloads/tita_rl-master/configs/legged_robot_config.py:4) |
| Lab标准配置参考 | [官方velocity TerminationsCfg](/home/eaziel/IsaacLab-main/source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py) |
| Lab函数与窗口 | [time_out](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/terminations.py)、[illegal_contact](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/terminations.py) |
| 布尔结果分类与合并 | [TerminationManager.compute](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/managers/termination_manager.py) |
| 自动重置调用 | [ManagerBasedRLEnv.step](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/manager_based_rl_env.py) |

<details>
<summary>历史诊断：2026-10-22—23 的频繁reset与碰撞排查（已结束）</summary>

以下命令、代码和假设属于当时排查过程。当前四处碰撞覆盖已修正，不要求重新加入这些诊断。

## 6. 看见频繁重置后，怎样查原因

2026-10-22用户报告：跟随相机后能看到机器人，但似乎刚落地就重置。当前配置只有time_out与base_contact。旧Gym也是机身受力模超过1 N或20秒超时，因此自动重置本身是旧任务已有行为；这不证明当前每次重置的触发都正确。

上一条zero_agent命令没有平衡策略。零动作经过当前Actions后，腿目标为默认关节角，轮子前馈为零但仍有速度阻尼；有PD不等于能主动保持整个轮足机身平衡。旧simple_play.py:73加载tita_example_10000.pt，且在41–55行缩小地形并关闭多项随机化，它的演示条件与当前零动作训练配置不同。当前通用六类ROUGH地形也尚未与旧台阶地形对齐。

本次先保留终止条件，记录每次重置的原因和回合仿真时长。在TitaVelocityEnv._reset_idx的docstring之后、已有观测采样及super()._reset_idx之前加入以下临时诊断：

```python
        # 初次env.reset()时总步数为0，不作为失败回合；本次只观察第0台机器人。
        if self.common_step_counter > 0 and 0 in env_ids:
            # 第0台机器人本回合已经运行的策略步数；父类重置后会归零。
            steps = int(self.episode_length_buf[0].item())
            reasons = []  # 保存本次值为True的终止项名称。
            for name in self.termination_manager.active_terms:
                if self.termination_manager.get_term(name)[0].item():
                    reasons.append(name)
            print(
                f"[RESET] env=0 steps={steps} time={steps * self.step_dt:.3f}s "
                f"reasons={reasons}"
            )
```

active_terms是当前配置中的终止项名字；get_term(name)读取TerminationManager本策略步已经计算的(N,)布尔张量，索引[0]取第0台。这里不重新计算终止。episode_length_buf是各环境回合计数，step_dt=0.01 s；打印的时间属于仿真，不能从窗口快慢推算它。诊断放在父类重置之前，是为了读到结束回合的步数和状态。

例如输出`steps=34 time=0.340s reasons=['base_contact']`表示跑了34个策略步后机身接触项触发。若是time_out，当前应在2000步/20 s。若多数回合只有1–2步，优先检查出生时机身接触、初始几何及传感器绑定；如果先明显失衡倒下再触发，支持零动作无法平衡的解释。具体根因仍须结合实际日志和画面，不先把频繁失败解释为正常，更不通过删除终止项来掩盖它。

2026-10-23用户已写入并运行本段，提供29次重置，均为base_contact；其中10次1步，2次2步，最长31步。证据见[接触审查](../audits/2026-10-23_reset_contacts/result.json)。只读依据：ManagerBasedRLEnv.step先compute终止、再调用子类_reset_idx；父类_reset_idx最后清episode_length_buf；TerminationManager.get_term读取已有_term_dones。

## 7. 区分地形接触、自碰撞和读数不一致（2026-10-23，已运行）

base_contact当前并非“只判断机身与地面”。它读取net_normal_forces_w，只按传感器body_ids选机身，没有筛选碰撞对方。只要引擎报告机身碰撞，来自腿部的自碰撞也可能进入聚合力；腿通过关节传递的支撑力属于关节约束，不会仅因传力就变成这里的碰撞接触。

当前静态核实：tita.py没有覆盖enabled_self_collisions，原USD没有显式self-collision属性；已安装omni.usd.schema.physx的generatedSchema.usda:1232默认该属性为1。各joint的physics:collisionEnabled为未显式写入的默认False，相邻链接通常被过滤。没有运行时stage读数，不能断言任意base-leg组合都启用碰撞，也不能将自碰撞判为本次根因。

另外，本地PhysX ContactSensor.reset确实通过kernel清空当前力和历史；update每物理步读取PhysX接触，当前终止也不读取history。因此源码不支持直接断言“忘清历史”。底层旧接触报告、实际几何碰撞、传感器绑定仍需区分。

下一步仅增加诊断读数，保留终止规则和物理配置。在VelocitySceneCfg现有contact_forces下添加：

```python
    # 只监测机身；额外区分机身受到的地形接触力。
    base_contact_debug = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/base_link",
        update_period=0.0,  # 每个物理步更新。
        filter_prim_paths_expr=["/World/ground/terrain/mesh"],
    )
```

PhysX过滤要求每环境只监测一个源刚体，因此另设机身诊断项，不给已有的九刚体传感器直接加filter。地形路径来自TerrainImporter.import_mesh('terrain')及spawn_from_mesh，其碰撞Mesh位于/World/ground/terrain/mesh。

在已存在的_reset_idx诊断if块里、原[RESET]打印之后添加：

```python
            # 原终止项实际使用的传感器和身体索引。
            sensor_cfg = self.termination_manager.get_term_cfg("base_contact").params["sensor_cfg"]
            sensor = self.scene.sensors[sensor_cfg.name]
            trigger_forces = sensor.data.net_normal_forces_w.torch[0, sensor_cfg.body_ids, :]

            # 独立机身传感器：总法向力和仅来自地形的法向力。
            data = self.scene.sensors["base_contact_debug"].data
            total = data.net_normal_forces_w.torch[0, 0, :]          # shape=(3,)，单位N。
            terrain = data.normal_force_matrix_w.torch[0, 0, 0, :]  # shape=(3,)，第0个过滤对象是地形。
            other = total - terrain  # 先减力向量，再求模；表示地形以外的聚合法向力。
            print(
                f"[CONTACT] trigger={torch.linalg.vector_norm(trigger_forces, dim=-1).max().item():.3f}N "
                f"total={torch.linalg.vector_norm(total).item():.3f}N "
                f"terrain={torch.linalg.vector_norm(terrain).item():.3f}N "
                f"other={torch.linalg.vector_norm(other).item():.3f}N"
            )
```

新传感器net_normal_forces_w为(N,1,3)，normal_force_matrix_w为(N,1,1,3)：第一个1是只监测机身，第二个1是只有地形这一项过滤。filter增加分对象读数，不会把net_normal_forces_w改成只含地形的力，也不改变哪些碰撞被物理解算。

若trigger与total接近且terrain解释了total，优先查出生位置和地形碰撞；若total明显非零而terrain近零，先验证过滤路径在确实触地时能报力，再检查自碰撞等非地形来源；不能仅凭差值一次非零定案。若trigger明显超过阈值而独立total近零，优先查原传感器身体绑定、采样/缓存一致性。用户现已写入并提供运行日志，结果见下一节；助手未运行仿真。

建议原zero_agent单环境Kit命令加--max_steps 200，收集少量完整[RESET]/[CONTACT]配对。10 ms仅重力从静止下落约0.49 mm，若碰撞体确有明显间隙，需要解释即时触发；窗口也可能已显示重置后的新姿态，因此判断依据应取父类reset之前的接触数据。

## 8. 过滤接触结果与自碰撞对照（2026-10-23，已运行）

用户最新[接触日志](../audits/2026-10-23_reset_contacts/user_filtered_contact_excerpt.txt)有45组CONTACT与44组RESET。当前源码先打印RESET再打印CONTACT；附件开头的CONTACT是截断后的孤立行，不能与下一行RESET配对。

- trigger与独立total全部相同，打印精度为0.001 N。
- 41组terrain为0；4组terrain非零，其中3003.833 N和237.553 N两组与total一致、other=0，说明地形过滤能够报数。
- 43组other超过1 N，total范围29.120–8607.972 N；44条RESET均为base_contact，11次只有1步。

这支持“主要触发力不属于过滤到的地形接触”。当前一机器人加地形的配置下，自碰撞成为首要解释，但还没有碰撞对数据，也未排除底层接触报告陈旧；不写成已确认自碰撞根因。terrain力模比total大并非矛盾，total是多个力向量相加再求模，方向不同可能互相抵消。

下一次只改变自碰撞开关。在assets/tita.py现有PhysxArticulationCfg中增加一行：

```python
PhysxArticulationCfg(
    # 临时对照：关闭机器人内部碰撞，检查异常机身受力是否消失。
    enabled_self_collisions=False,
    solver_position_iteration_count=4,
    solver_velocity_iteration_count=0,
)
```

字段属于物理资产的articulation属性，控制物理引擎是否生成同一机器人内部链接间的碰撞；它不是接触传感器过滤，也不是屏蔽终止项。保持zero_agent单环境、当前其他配置及诊断，用相同策略步预算对照；随机化样本会随重置时机改变，不声称两个运行每回合抽到相同状态。

用户现已写入False并复跑：日志到第200步并正常清理，记录两次base_contact，持续62步/84步；两次other=0且terrain=total，没有记录单步重置。用户报告现象改善。这强烈支持此前异常来自自碰撞路径，尚未定位碰撞对或证明训练稳定。当前文件False已只读核实；助手未运行仿真。

旧Gym configs/tita_constraint_config.py:97的self_collisions=0注释明确0启用，legged_robot.py:240将其传给create_actor，因此False只用于诊断；后续目标仍是定位相撞链接，检查碰撞几何、初值及必要的成对过滤。NVIDIA[稳定性指南](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html#articulation-link-self-collision)也推荐先关闭自碰撞验证，再用碰撞几何和接触可视化定位。

## 9. 碰撞覆盖已修正，临时诊断可以撤除

旧URDF只有机身box、两小腿box、两轮cylinder共5处有效collision，左右leg_1/leg_2没有碰撞；原USD额外启用了这四处Mesh。用户已直接在源USD关闭四处Collision Enabled，重新开启自碰撞后报告正常。本次静态读取确认四处禁用、五处启用，证据见[修正后状态](../audits/2026-10-23_collision_model/after_user_usd_edit.json)。具体是哪一对接触仍未记录，新运行也未附完整日志；现象改善不等于训练通过。

现在建议删除Scene中的base_contact_debug和环境中的四类临时打印及cost断言，保留原正式接触传感器、终止规则、观测和cost采集时机。给用户的清理后step/reset与可选三个box几何方案见[19](19_碰撞模型对齐与诊断清理.md)。清理候选仅通过AST，尚未写入或运行。第6—8节代码用于保存诊断方法，不要求继续留在生产环境。

</details>

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> any是什么函数？

**A：** 这里用torch.any沿选中身体轴检查是否至少有一个True。当前函数把(N,B,3)力向量沿xyz求范数，得到(N,B)，再比较阈值并沿dim=1合并，返回(N,)布尔张量。dim、body_ids及完整数值例子集中在[终止函数的补充问答](../../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/terminations.py.md#补充问答)。

**Q：** 但我看到机器人疯狂的reset，希望这些reset不是异常的reset，真的很频繁，我有点担心开始训练的时候就出问题，因为看起来机器人刚落地就reset了！

**A：** 当前零动作没有平衡策略，机身接触导致回合结束是旧Gym已有规则，但不能仅凭这一点保证当前触发正确。先按第6节记录每回合步数、仿真时长与终止项；尤其区分先倒下再碰机身，与出生后1–2步就触发。当前地形也尚未对齐，训练前需要解释清楚实际触发原因。

**Q：** 我们代码的行为，base碰到腿，也算接触吗？我不知道是不是base碰到腿了，我感觉没接触啊！

**A：** 只要引擎启用并报告了该碰撞，就可能计入。当前函数选择接收力的身体base_link，没有选择碰撞对方；它并非只判断地面。机身与腿通过关节传力不算碰撞接触，直接相邻链接通常也默认过滤碰撞。静态文件没有显式覆盖自碰撞，已安装schema默认开启；实际运行时开关和碰撞对方尚待验证，第7节诊断用于区分来源。

**Q：** 而且我想知道zero agent啥意思？random是啥意思？

**A：** 在本任务中，两者是无需训练模型的动作生成器。zero_agent每策略步给(N,8)全零原始动作；当前Actions将其变成腿的默认角目标和轮子零前馈力矩，PD、轮阻尼、重力和碰撞仍生效。random_agent每策略步对每个动作独立采样[-1,1)，再交给同一动作处理；随机数不是最终力矩或关节角。两者均不学习、不更新网络参数，适合检查环境能否运行，不能据其表现推断训练好的平衡能力。依据当前isaaclab_rl/entrypoints/simple_agents.py及任务actions.py、actuators.py。

[返回迁移入口](README.md) · [查看来源与覆盖表](00_迁移映射索引.md)
