# 动作网络前向候选：CPU旧实现对照

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-25。内容纯AI生成，由人类要求生成，未经人类修改。

候选proposal_actor.py拟供用户写入learning/np3o/actor.py。原始对照来自/home/eaziel/Downloads/tita_rl-master/modules/actor_critic.py的MlpBarlowTwinsActor，以及common_modules.py的两个MLP工厂；通过AST提取原始定义执行，不改旧源码、跳过包导入扩散。

运行：

```bash
/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/bin/python -B docs/audits/2026-10-25_actor_forward/check_forward.py
```

PyTorch 2.12.0+cu130，张量和网络均在CPU。result.json记录来源哈希、参数量和误差。相同前向子网权重，batch=1/4/64输出差0；相同标量目标的一次backward，输入与参数梯度均一致。检查历史0..9加当前99送入编码器成为1..9、99；输入未被改写。

没有优化器step、Sim/PhysX、真实环境采样、训练、辅助损失、价值/cost网络或旧checkpoint整体载入。旧类的obs_encoder与BatchNorm不属于本次动作前向，未迁入候选；不能把子集参数对照称为完整网络迁移成功。
