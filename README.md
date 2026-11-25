# DDL TITA Control

TITA轮腿机器人的Isaac Lab学习与迁移工程：先理解原任务的控制、观测与训练计算，再接入Lab环境和训练模块。工程使用IsaacLab-main、IsaacSim6.2及lab3_sim62 venv。

**当前进度（2026-11-14核对）：** 环境、模型、rollout、GAE与小批量已接通；用户完成critic与PPO奖励策略各8次更新检查，并生成[固定单批critic拟合图](docs/audits/2026-11-14_critic_plot/README.md)。六类cost已于本日完成单批策略更新检查；持续训练runner及学习后任务评估尚未接通。

| 入口 | 用途 |
| --- | --- |
| [文档入口](docs/README.md) | 当前阅读路线及专题入口 |
| [重要代码解释](docs/重要代码解释索引.md) | 按模块查Python、Torch、数学与数据流 |
| [迁移映射](docs/migration/00_迁移映射索引.md) | 从Lab功能找到原Gym参数和行为 |
| [项目记忆入口](PROJECT_MEMORY.md) | 按任务进入迁移状态、运行环境和独立专题 |
| [协作与讲解（agent必读）](project_memory/协作与讲解.md) | 理解用户的意图、学习基础，以及怎样给代码和解释 |
| [新助手接手](docs/AGENT_HANDOFF.md) | 项目目标、两条训练路线、当前调用链及数据约定 |
| [研究调研](graduate_research/README.md) | 研究问题、论文证据与候选验证方案 |
| [求职项目建议](career_planning/2026-11-08_TITA求职项目建设建议.md) | 职业方向的项目建设建议，尚非迁移实施计划 |

源码工程分别位于[Tita_Lab](Tita_Lab/)和[TITA_non_RL](TITA_non_RL/)，由用户维护。GAE和[小批量取数](docs/migration/32_更新前的小批量数据.md)已有实现与用户短检查；[critic更新检查](docs/migration/33_价值网络首次参数更新.md)已取得8次更新及参数变化的用户输出。策略奖励项检查和绘图也已有证据；六类cost约束策略也已完成单批检查，之后仍缺持续runner及训练配置。需要复用NP3O代码时选择独立实现，[来源调查](docs/migration/30_NP3O来源与接入边界.md)保留为参考，尚未选定依赖。

原任务参考：[DDTRobot/tita_rl](https://github.com/DDTRobot/tita_rl)。协作权限见[AGENTS.md](AGENTS.md)，安装和运行入口见[25](docs/migration/25_Lab_main与Sim62_入口和配置.md)。

本README、docs与code_explaination中的说明内容纯AI生成，由人类要求生成，未经人类修改；此声明不用于标记项目源码。
