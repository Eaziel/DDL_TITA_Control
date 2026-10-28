# 当前actor与辅助损失候选检查

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-26。内容纯AI生成，由人类要求生成，未经人类修改。

check_actor.py加载用户当前actor.py（最终版本已修正prediction_dim拼写）、proposal_actor.py候选，以及通过AST提取的旧MlpBarlowTwinsActor、MLP工厂、off_diagonal。候选只在docs保存，含属性名统一、obs_encoder/BatchNorm和auxiliary_loss，不修改用户源码。

运行：

```bash
/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/bin/python -B docs/audits/2026-10-26_actor_auxiliary/check_actor.py
```

result.json记录实际来源哈希。当前actor同权重的batch1/4/64前向与旧源码差0，时间编号窗口正确；候选辅助损失batch4/32差0与1.907e-6。旧对角线原位平方/归约与候选非原位计算有float32舍入差异，采用rtol=atol=1e-5；实际参数梯度最大差0，输入梯度与BatchNorm缓冲也相同。保留过去十帧的辅助窗口，区别于动作forward的九历史+当前。

未运行优化器step、仿真、完整NP3O或训练。没有用此CPU结果声称机器人性能或完整算法迁移成功。
