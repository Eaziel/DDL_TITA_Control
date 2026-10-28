# DDL TITA Control

TITA 轮腿机器人的 Isaac Lab 学习与迁移工程。目标是在理解原任务计算方法的基础上，把本末科技的 Isaac Gym 环境与 NP3O 训练实现迁入现有 Lab 工程。

**当前状态（2026-10-28）：** 使用IsaacLab-main、Sim6.2和lab3_sim62 venv。环境、五组观测、六项cost、wrapper与actor/辅助损失已写入，完整NP3O尚未接通。新环境已有四环境短随机日志；轮子偶尔穿地仍待定位。

| 目录或文件 | 内容 |
| --- | --- |
| [Tita_Lab](Tita_Lab/) | Manager-based RL 外部工程，由用户维护源码 |
| [TITA_non_RL](TITA_non_RL/) | 独立 USD 资产与非 RL 脚本，由用户维护源码 |
| [文档入口](docs/README.md) | 当前阅读顺序、迁移说明和检查证据 |
| [源码解释](code_explaination/README.md) | 按源码文件学习 Python、数学与数据流 |
| [项目记忆](PROJECT_MEMORY.md) | 当前状态、关键约定与尚未验证的内容 |
| [研究调研](graduate_research/README.md) | 论文原文、研究问题与验证计划，独立于迁移主线 |

当前先学习 [官方训练流程与 wrapper](docs/训练流程与Wrapper入门.md)，再读 [wrapper 源码解释](code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/env_wrapper.py.md) 和 [actor 源码解释](code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/actor.py.md)。本次仅完成[actor与环境联调](docs/migration/26_actor与环境联调.md)，然后暂停新增模块、阅读文档。源码改动由用户执行，具体见[检查清单](docs/当前代码检查与修改清单.md)。

参考实现：[DDTRobot/tita_rl](https://github.com/DDTRobot/tita_rl)。本地源码和证据路径见项目记忆；协作边界见 [AGENTS.md](AGENTS.md)。

**本 README、docs/ 与 code_explaination/ 中的说明内容纯 AI 生成，由人类要求生成，未经人类修改。此声明不用于标记项目源码。**
