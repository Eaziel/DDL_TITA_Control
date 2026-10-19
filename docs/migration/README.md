# TITA：Isaac Gym → Isaac Lab 3 迁移说明

**内容纯 AI 生成，由人类要求生成，未经人类修改。**

更新：2026-10-19。本文档记录旧任务的计算方法、Lab对应代码和迁移理由。当前仍沿用用户生成的 manager-based 外部工程。

## 从哪里读

| 文档 | 解决什么问题 |
| --- | --- |
| [00 来源与覆盖索引](00_迁移映射索引.md) | 写一个Lab模块时，到Gym哪里找参数、计算和遗漏项 |
| [01 Physics](01_physics.md) | 物理步长、decimation、substeps及PhysX参数如何对应 |
| [02 Scene](02_scene迁移.md) | 地形、高度扫描、接触力是什么，怎样配置和读取 |
| [03 Actions](03_actions迁移.md) | 八维动作如何经过缩放、腿PD与轮子前馈，成为力矩 |
| [04 Commands](04_commands迁移.md) | 目标如何采样，航向误差如何变成目标转速 |
| [05 Observations](05_observations迁移.md) | 当前33维如何取数、加噪、拼接，以及过去十帧如何保存和输出 |
| [06 Events](06_events迁移.md) | 两项reset、两项质量/质心startup已保存；全局15 s推扰的函数/导出/配置方案待写 |
| [07 Terminations](07_terminations迁移.md) | 机身接触与20 s超时怎样结束回合；已写入，静态/局部CPU检查通过 |
| [08 Rewards](08_rewards迁移.md) | 10项及None功率接口/最终合成已写入；附普通项、时间缩放和裁剪动机问答，附类项生命周期与物理功率问答 |

**当前状态：基础观测/历史、TITA重置、结束条件、10项奖励及最终合成step均已写入。** 功率None接口、显式func=和slice注解已保存；step AST与此前CPU替身方案对应。当前文件核对与历史CPU/Ruff检查分别记录，尚未运行真实RL环境。

**当前下一步：每隔15 s扰动实际水平速度。** 基座质量/质心已保存并核对；新增任务函数将世界系vx/vy设为U(-1,1) m/s，通过全局interval Event调用。完整函数、.pyi导出、EventCfg配置及理由见[06第6节](06_events迁移.md#6-运动过程中每15秒扰动一次水平速度)。当前方案已有CPU替身/临时配置检查，尚未写入任务源码或运行真实环境。

00是查证与覆盖清单，不是固定教学顺序。每轮以一个完整功能为单位，按当前依赖推进。

## Gym 与 Lab 的组织有什么不同

旧 `envs/legged_robot.py` 同时负责创建仿真、创建机器人、推进物理、算力矩、更新命令、处理重置、计算观测/奖励/成本。参数则分散在继承配置与工具文件中。

Lab 把通用执行过程放入环境类和manager，任务选择具体功能并填写参数：

| 层次 | 本项目例子 | 负责什么 |
| --- | --- | --- |
| 配置 | `VelocityEnvCfg`、各个Cfg、`TITA_CFG` | 选择对象、函数/类实现和参数，把任务组合起来 |
| 执行 | `ManagerBasedRLEnv`、各manager、`InteractiveScene` | 创建对象，组织step/reset，调用具体计算 |
| 具体计算与状态 | 官方/自定义MDP、ActionTerm、执行器、`ProprioHistory` | 取数或算数；需要跨步数据时保存状态 |

例如 `ObsTerm(func=mdp.base_ang_vel, scale=0.25)` 声明怎样得到一项观测；ObservationManager 运行时调用函数并缩放。cfg因此既有数值，也有实现入口，但不会替尚未实现的功能自动生成算法。

TITA特有规则才补自定义代码。例如轮角屏蔽用一个MDP函数；当前历史方案用缓存类保存十帧，由 `TitaVelocityEnv` 控制更新与重置。这些文件共同完成观测的存储、更新与输出。

## 每章怎样写和维护

每项功能按同一条主线讲完：

1. **Gym原来怎样做：** 给有效配置、关键原代码，解释输入、计算和输出。
2. **Lab中怎样写：** 给具体代码与放置位置，标明已写入或待写方案。
3. **为什么这样迁移：** 把两边公式、数据顺序和执行时机对应起来，说明现成API覆盖什么、自定义部分补什么。
4. **还有什么差异：** 集中列未完成项及验证边界，最后附对照源码。

正文按功能连续讲完：旧计算、Lab代码、对应理由、差异与验证。链接用于查证源码和深入实现。

用户追问产生的补充放在主线与源码索引之后，以`Q：`、`A：`保存，Q保留用户原句。只有用户表示不知如何提问时，才整理问题并注明。正文保留理解当前功能必需的定义和计算；对话中途的澄清、编辑器提示、索引细节等进入问答，修改经过进入项目历史。要求保留的旧解释原文单独折叠，当前实现放在前面。

当前状态与完整覆盖缺口放在[00](00_迁移映射索引.md#2-覆盖状态与已知差异)，检查范围和证据按功能记在各章。[按源码解释](../../code_explaination/README.md)用于查Python对象、索引和具体文件，并保留要求归档的历史原文。本地项目记忆与协作历史不纳入Git提交。

用户负责写代码，助手在回复中直接给当前功能所需的代码、位置和连贯解释，并维护文档；不把先读文档作为继续学习的条件。助手不修改 `TITA_Lab/`。

用户2026-10-18补充：适当将关联条目合并推进，让每轮有更多新内容；回复侧重必要代码和关键解释，文档继续按完整主线归档。

## 版本与路径

| 对象 | 当前记录 |
| --- | --- |
| Gym参考副本 | `/home/eaziel/Downloads/tita_rl-master`，无`.git`，不能确定上游提交 |
| Isaac Lab | `/home/eaziel/IsaacLab`，`release/3.0.0`，`9ef05c8972634889968b332bfc4b80860789c3ff`；API按本地文件核对 |
| 目标配置 | [TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py) |
| 机器人配置 | [TITA_Lab/src/TITA_Lab/assets/tita.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/assets/tita.py) |
| 当前任务名 | `TitaLab-Velocity-Tita` |

用户报告 `scripts/import_assets.py` 已能运行、关节正常；助手未复跑。静态检查、独立CPU计算、仿真运行、训练复现分别记录，资产导入成功不代表RL任务已经通过。

