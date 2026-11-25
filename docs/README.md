# 项目文档入口

内容纯AI生成，由人类要求生成，未经人类修改。更新：2026-11-25。

当前已接通从采集到 critic 和六类 cost 约束策略的单批更新。用户把两种损失移到 `losses.py` 后，已经完成 4 环境、24 步的运行。进度与运行证据见[项目记忆](../PROJECT_MEMORY.md)。

## 当前先把一批更新读通

| 材料 | 从哪里开始，解决什么问题 |
| --- | --- |
| **先读 [check 第 1 节](../code_explaination/Tita_Lab/scripts/check_np3o_interface.py.md#1-从一条经历看完整次更新)** | 跟一条动作经历，看清采集、GAE 和两次网络更新怎样接起来 |
| [rollout 第 13 节](../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/rollout_storage.py.md#13-gae-把后几步结果算进当前动作) | 用两步重新算一遍，分清回报目标与优势 |
| [一批经历怎样完成策略更新](一批经历怎样完成策略更新.md) | 需要画整块算法时回看总览，第 7 节指向各文件的详情 |

11 月 25 日的单批运行完成了 critic 和策略各 8 次更新。记录包含六类 cost 的损失和合计梯度，以及对同一批数据的拟合图和拆分前后对照。[损失拆分检查](audits/2026-11-25_loss_extraction/README.md)

## 按当前疑问回查

| 需要理解什么 | 阅读入口 |
| --- | --- |
| 官方训练怎样串起环境、wrapper、runner和PPO | [训练流程与Wrapper](训练流程与Wrapper入门.md)，第6节是调用链，第9节对照当前自有check |
| 每个重要Py文件的用途 | [重要代码解释索引](重要代码解释索引.md)，本页只放链接与简要介绍 |
| 动作高斯分布、密度、熵与Torch接口 | [policy解释](../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/policy.py.md) |
| 观测怎样经过扫描/历史编码并预测价值 | [critic解释](../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/critic.py.md) |
| actor的网络、辅助预测与Barlow Twins | [actor解释](../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/actor.py.md) |
| 一次环境step/reset由谁调用各模块 | [环境主类解释](../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/tita_velocity_env.py.md) |
| 586列的顺序与各观测组 | [观测打包解释](../code_explaination/Tita_Lab/src/Tita_Lab/tasks/velocity/observation_layout.py.md) |

数学与API的详细解释以对应源码页为主。迁移章节记录原Gym做法、Lab落点和迁移理由；不将章末旧“下一步”当作当前执行计划。

## 状态、来源与运行

| 需要 | 入口 |
| --- | --- |
| 新对话接手 | [Agent说明](AGENT_HANDOFF.md)、[可复制续接提示](新对话续接提示词.md) |
| 新助手如何理解用户与讲解代码（必读） | [协作与讲解](../project_memory/协作与讲解.md)、[解释文档写法](../code_explaination/README.md#解释文档怎样写) |
| 当前进度、剩余工作与权限 | [项目记忆入口](../PROJECT_MEMORY.md)、[迁移状态](../project_memory/迁移状态.md)、[AGENTS](../AGENTS.md) |
| 当前检查结论 | [检查清单](当前代码检查与修改清单.md) |
| Gym参数和行为的来源 | [00映射索引](migration/00_迁移映射索引.md) |
| 按功能查看迁移阶段 | [迁移目录](migration/README.md) |
| 全部源码解释 | [代码解释目录](../code_explaination/README.md) |
| 安装及手动运行 | [25项目入口](migration/25_Lab_main与Sim62_入口和配置.md)、[Ubuntu＋uv教程](新版IsaacLab与IsaacSim安装教程_Ubuntu_uv.md) |
| 运行日志与静态检查各证明什么 | [证据目录](audits/README.md)，当前单批更新见[约束检查](audits/2026-11-14_constrained_policy_update/README.md)，超时GAE见[历史两次检查](audits/2026-11-04_gae/README.md) |
| 过去的决定、故障和修正 | [项目历史](PROJECT_HISTORY.md) |
| 历史USD检查与协议问题 | [USD快照](USD_CHECK_2026-10-15.md)、[协议说明](开源协议说明.md) |
| 研究与职业方向 | [研究调研](../graduate_research/README.md)、[求职项目建议](../career_planning/2026-11-08_TITA求职项目建设建议.md) |

历史记录保留当时日期与证据；研究/职业提案不代表当前迁移已接受或已实施的计划。
