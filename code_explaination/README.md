# 项目代码解释

内容纯AI生成，由人类要求生成，未经人类修改。导航更新：2026-11-25。

按源码相对路径镜像归档，每个源码文件一份主说明。源码路径写在页首，先贴相关代码，再按功能与执行顺序解释数学、对象、接口和张量变化。

## 解释文档怎样写

写作前读[AGENTS](../AGENTS.md)与必读的[协作与讲解](../project_memory/协作与讲解.md)，用户明确说过的基础也统一维护在后者。用户学过一部分RL理论，正在首次项目使用Torch；新API与对象机制不能因被归为“基础”就略去，已经写完／跑通也不等于已掌握。

正文先说明文件要计算什么、解决什么问题、由谁调用及结果在哪里使用；再沿实际计算过程分段贴源码，把数学、API与数据变化一起说明。数学应解释关系的含义、成立原因和选择理由；张量变形应说明前后shape、轴含义和经历对应关系。解释相关方法的输入、输出／副作用、所属对象和执行时机，不将库方法名的中文翻译当作原理。具体调用主体要明确，避免多个env、step、reset混称。

详细程度与连贯性参考[policy.py](Tita_Lab/src/Tita_Lab/learning/np3o/policy.py.md)关于概率、分布形状及方法的衔接，以及[critic.py](Tita_Lab/src/Tita_Lab/learning/np3o/critic.py.md)关于目标、编码、卷积和轴变化的衔接；按本文件内容组织，不机械套模板，不反复补用户已明确知道的整章RL入门。

源码片段保留英文功能docstring、中文解释性行内注释与有用类型提示；关键量不能只标一个shape而没有数学／任务含义。长推导和完整API讲解写在正文。用户主线必需的追问直接融入相应位置；独立延伸问题放后面，用真实Q／A，不编造用户提问。解释文档的完整性不代替对话中应给的代码、放置位置与关键来龙去脉。

## 当前阅读

第一次只读 [check 第 1 节](Tita_Lab/scripts/check_np3o_interface.py.md#1-从一条经历看完整次更新)。第 1 节跟一条动作经历，看后续 cost 怎样进入学习。遇到一处没接上，再查相关说明。

| 读到哪里 | 接着看 |
| --- | --- |
| cost 晚一拍出现，为什么会影响前一拍 | [rollout 第 13 节](Tita_Lab/src/Tita_Lab/learning/np3o/rollout_storage.py.md#13-gae-把后几步结果算进当前动作) |
| 误差怎样变成参数的梯度 | [check 第 12 节](Tita_Lab/scripts/check_np3o_interface.py.md#12-check_value_update从误差到自动求导再到参数更新) |
| 当前网络怎样算同一个旧动作的密度 | [policy.py](Tita_Lab/src/Tita_Lab/learning/np3o/policy.py.md) |
| 两种价值和动作均值分别怎样算 | [critic.py](Tita_Lab/src/Tita_Lab/learning/np3o/critic.py.md) · [actor.py](Tita_Lab/src/Tita_Lab/learning/np3o/actor.py.md) |
| 想画出整块更新的流程 | [一批经历怎样完成策略更新](../docs/一批经历怎样完成策略更新.md) |

## 按需查阅

- [重要代码解释索引](../docs/重要代码解释索引.md)：模块用途和讲解链接。
- [Tita_Lab完整索引](Tita_Lab/README.md)：资产、配置、环境、MDP与训练侧模块。
- [非RL独立脚本](TITA_non_RL/scripts/run_sim.py.md)：材质随机化、Torch/Warp和物理循环。
- [官方训练流程与Wrapper](../docs/训练流程与Wrapper入门.md)：配置、环境与runner怎样连接。
- [迁移记录](../docs/migration/README.md)：原Gym行为、Lab写法与迁移理由。
- [检查证据](../docs/audits/README.md)：静态、CPU对照与用户运行的各自范围。

主线必需的追问融入相关正文，延伸讨论保留真实问题的Q/A。要求保留的原文独立折叠，并标明历史状态；源码片段是否完整、是否为参考实现，以各页说明为准。权限和归档规则见[AGENTS.md](../AGENTS.md)。
