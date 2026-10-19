# DDL TITA Control

TITA 轮腿机器人的 Isaac Lab 学习与迁移工程。以本目录作为 Git 仓库根目录，同时维护 RL 工程和独立的非 RL 资产、脚本。

| 目录 | 内容 |
| --- | --- |
| [TITA_Lab](TITA_Lab/) | Isaac Lab manager-based RL 外部工程，包含机器人资产、任务配置、MDP 和训练配置 |
| [TITA_non_RL](TITA_non_RL/) | 独立维护的机器人 USD 资产和非 RL 脚本 |
| [docs](docs/README.md) | Gym → Lab 的迁移说明与源码映射 |
| [code_explaination](code_explaination/README.md) | 按源码文件归档的代码解释 |

迁移参考：[DDTRobot/tita_rl](https://github.com/DDTRobot/tita_rl)。当前目标为本地 Isaac Lab `release/3.0.0` 与 PhysX；具体实现、已知差异和检查范围见[迁移入口](docs/migration/README.md)。当前 RL 任务仍在迁移，尚未完成真实 RL 环境与训练验证。

文档生成说明：`docs/` 与 `code_explaination/` 中的说明文档**内容纯 AI 生成，由人类要求生成，未经人类修改**。本 README 同样由 AI 生成。该声明针对说明文档，不将项目源码标为纯 AI 生成。

本地项目记忆、助手配置、凭据目录、缓存与运行产物由根目录 `.gitignore` 排除。

USD资产约2 MB，使用普通Git存储；根目录`.gitattributes`统一管理属性，独立工程模板的LFS属性文件留在本机。
