# 项目代码解释

**内容纯AI生成，由人类要求生成，未经人类修改。**

当前先读[训练流程](../docs/训练流程与Wrapper入门.md)、[wrapper](Tita_Lab/src/Tita_Lab/learning/np3o/env_wrapper.py.md)、[actor](Tita_Lab/src/Tita_Lab/learning/np3o/actor.py.md)。目标是能解释调用者、持有的数据、计算结果与更新时机。当前环境为IsaacLab-main＋Sim6.2；本次actor脚本见[26](../docs/migration/26_actor与环境联调.md)，完成后暂停新增内容。

按源码目录镜像归档，每个源码文件只有一份主说明：先贴对应代码，再解释对象、类型、索引和数据流。用户追问保留在末尾Q/A，要求保留的历史原文单独折叠。

- [Tita_Lab索引](Tita_Lab/README.md)
- [TITA_non_RL / run_sim.py](TITA_non_RL/scripts/run_sim.py.md)
- [迁移主线](../docs/migration/README.md)：Gym计算、Lab写法与迁移理由
- [检查证据](../docs/audits/README.md)：当前检查与早期修正

协作与格式以[AGENTS.md](../AGENTS.md)为准。正文解释当前实现；源码摘录、早期候选和运行结果分别标注；文档不替代仿真验证。
