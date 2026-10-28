# 当前已写actor与辅助损失：CPU对照

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-26。内容纯AI生成，由人类要求生成，未经人类修改。

本轮检查用户实际保存的TITA_Lab/src/TITA_Lab/learning/np3o/actor.py，而非仅运行候选。check_saved_actor.py执行读取时的完整源码快照，与旧MlpBarlowTwinsActor、MLP工厂和off_diagonal原定义对照，来源哈希见result.json。

运行：

```bash
/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/bin/python -B docs/audits/2026-10-26_saved_actor/check_saved_actor.py
```

检查时实际方法名为auxilary_loss，计划接口为auxiliary_loss。测试明确读取实际方法名以验证计算体，没有给类添加别名，也没有修改用户源文件。method_name_matches_planned_interface=false与arithmetic_checks_passed=true分别表示命名问题和计算结果，不把它们混成一个“全部通过”。

同权重结果：batch1/4/64动作前向最大差0；batch4/32辅助损失最大差约3.815e-6/0，float32比较容差rtol=atol=1e-5；所检参数与输入梯度最大差0，BatchNorm缓冲相同。两个实际编码窗口分别为九历史+当前、原样过去十帧，均核实。

没有执行optimizer.step、AppLauncher、Sim/PhysX或训练。用户继续自己阅读与提问，本轮不安排后续模块。解释见code_explaination中唯一actor.py.md，开头新增完整数学架构与编码器用途。
