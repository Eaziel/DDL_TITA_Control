# 当前main／Sim6.2：代码与文档核对

日期：2026-10-28。内容纯AI生成，由人类要求生成，未经人类修改。

用户明确授权文档更新，源码始终只读。核对RL任务、MDP、执行器、cost管理器、观测布局、wrapper、actor、独立脚本、注册及配置；同时扫描根文档、docs、代码解释和研究文档。原始日志与历史原文保留为快照，不将旧运行转换成新版验证。

## 核对的接口

| 范围 | 本地当前依据 | 结论 |
| --- | --- | --- |
| 后端实例与事件 | SimulationContext.__init__、RandomizeTitaMaterial | 当前backend_manager实例检查与Kit后导入保留；asset_cfg在使用前定义 |
| 动作与执行器 | JointAction.process_actions、BaseArticulation目标setter、IdealPDActuator | 现有FIR、缩放、延迟提交与限矩API存在；正常actions.py没有重复copy_ |
| 观测/奖励/cost | height_scan、ContactSensorData、RewardManager.compute、ManagerBasedRLEnv.step/reset | 当前字段和时间缩放相符；明确法向字段与含摩擦总力不同 |
| 启动与actor脚本 | simple_agents.run、launch_simulation、resolve_task_config | 当前三阶段流程成立；旧草稿的三个包导入需改名 |
| RSL-RL/PPO | train_rsl_rl.py、RslRlVecEnvWrapper、当前rl_cfg | 仍使用官方PPO模板，policy33；不会自动使用TITA actor或NP3O约束 |
| 独立脚本 | AppLauncher兼容类、当前脚本和资产位置 | 兼容入口仍在但已弃用；非RL相对路径依赖工作目录 |

## 证据与边界

- [审查前源码哈希](source_before.json)：普通项目文件，不读取Git元数据。
- [具体API与依赖来源](api_and_package_check.json)：目标方法存在，依赖覆盖与本地main一致，可编辑来源路径存在。
- [收尾检查](review_result.json)：语法、链接、公式/问答保存和源码变化范围。
- [具体代码建议](../../当前代码检查与修改清单.md)、[actor参考](../../migration/26_actor与环境联调.md)。

本轮只执行静态读取、元数据读取和文档检查，没有导入项目运行逻辑、启动Kit、运行GPU交互、优化器或训练。Ruff可执行程序不存在，没有宣称Lint通过。此前实际配置导入、CPU数值与用户物理日志见[历史证据索引](../README.md)。

轮子穿地仍未定位；本轮不据此调整碰撞、去穿透速度或力口径。新旧Lab比较已由用户取消，旧环境不再作为当前运行入口。
