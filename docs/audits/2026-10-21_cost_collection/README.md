# 六列cost检查

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

2026-10-21。当前五个cost函数直接读取用户源码；加速度类、收集器、cfg和环境钩子从[17](../../migration/17_cost收集与加速度.md)代码块读取，均未写入TITA_Lab。

结果见[result.json](result.json)及[current_five_result.json](current_five_result.json)。使用本地原生ManagerBase做名称解析和类实例化，复用原生ManagerBasedRLEnv.step/_reset_idx及原生各管理器，机器人状态、传感器与物理推进用CPU替身。候选构造中的super后初始化位置由本地源码核对，未运行真正环境构造。

两种compute_final_obs设置下各运行12策略步、48物理步，包含第2/5/9步部分reset、第3/6/9/12步推扰。六列每步对照从旧Gym源码抽取的方法，最大差0。检查初次reset不计步、终止步未被清掉、同一步重复compute不重复累计、reset后速度参考、非step部分reset、回合统计及返回副本不被后续compute覆盖。配置validate与候选Ruff通过；当前五函数另做64环境边界对照。

复现：

```bash
/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/bin/python -B docs/audits/2026-10-21_cost_collection/check.py
```

未运行AppLauncher、PhysX或训练。不证明真实接触力、实际平衡或策略效果；法向接触力与旧合接触力的口径差异保留。
