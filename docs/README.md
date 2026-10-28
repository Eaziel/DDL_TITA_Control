# 项目文档入口

内容纯 AI 生成，由人类要求生成，未经人类修改。整理：2026-10-28。

这组文档解释原TITA怎样计算，以及当前Lab代码由谁调用、怎样组织数据。当前工程使用IsaacLab-main、IsaacSim6.2和共享venv；运行命令见[项目入口与运行](migration/25_Lab_main与Sim62_入口和配置.md)。本次只联调已经保存的actor和环境，之后暂停新增内容、阅读文档。

## 训练接口与 actor 的学习材料

| 顺序 | 阅读材料 | 读完应能说明什么 |
| --- | --- | --- |
| 1 | [从官方训练流程理解 Wrapper](训练流程与Wrapper入门.md) | 环境、包装器、runner、算法、PyTorch 各做什么；谁真正更新参数 |
| 2 | [env_wrapper.py 解释](../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/env_wrapper.py.md) | Lab 的五项返回怎样变成六项；`infos`、`None`、自动 reset 的含义 |
| 3 | [actor.py 解释](../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/actor.py.md) | 三条网络支路、两个历史窗口、每层计算与辅助损失怎样对应 |
| 按需 | [586 列布局解释](../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/observation_layout.py.md) | actor 的输入和十项监督目标分别从哪些列取得 |

actor 文档可分两次读：先读架构与前向（0—6 节），再读辅助学习和梯度（7—9 节）。API 索引供回查，章末 Q/A 保留实际提问，不要求一次背完。

## 其他材料怎样查

| 需要 | 入口 |
| --- | --- |
| 新版Lab＋完整Sim的公开安装教程 | [Ubuntu＋uv安装教程](新版IsaacLab与IsaacSim安装教程_Ubuntu_uv.md) |
| 当前代码需要改哪里 | [检查与修改清单](当前代码检查与修改清单.md) |
| 本次actor联调 | [26 actor与环境联调](migration/26_actor与环境联调.md) |
| 当前进度、约定和缺口 | [项目记忆](../PROJECT_MEMORY.md) |
| Gym 与 Lab 的功能对应 | [迁移目录](migration/README.md) |
| 找原参数和执行行为的代码位置 | [00 映射表](migration/00_迁移映射索引.md) |
| 按源码文件查对象、索引和数学 | [源码解释目录](../code_explaination/README.md) |
| 核对一次检查实际证明了什么 | [证据目录](audits/README.md) |
| 查以前的决定与故障 | [项目历史](PROJECT_HISTORY.md) |

迁移正文按“目标 → Gym 怎样做 → Lab 怎样对应 → 迁移理由与差异”组织。旧审查中的代码属于当时快照；当前状态以项目记忆和新证据为准。2026-10-15 的 [USD 检查](USD_CHECK_2026-10-15.md) 保留为历史资产快照，[协议说明](开源协议说明.md) 保留为当时问答。

按当前问题选择材料，无需通读全部迁移章、排查日志或研究文献。
