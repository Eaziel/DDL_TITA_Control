# 当前cost接线检查

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

日期：2026-10-22。使用用户当前六函数、CostTermCfg/CostsCfg、TitaVelocityEnv.step/_reset_idx与收集器源码，未修改TITA_Lab或TITA_non_RL。

1. 当前源码五文件AST、Ruff通过，配置validate通过；这些检查没有识别抽象方法名或func参数语义错误。
2. 原生Python/Lab复现_prepare_term少s导致抽象类无法实例化。
3. 只改方法名后，原生参数解析复现速度项收到sensor_cfg而要求asset_cfg。
4. 当前k形状与d缺失、速度k=0.1由源码与旧Gym配置对照确认。
5. 仅在内存修正上述四处后，复用原生ManagerBase及RL step/reset CPU夹具；两种final_obs各12策略步、48物理步，三次部分reset、四次推扰，六旧函数对照最大差0；重复读取、回合统计、输出副本及初次reset通过。

夹具的物理推进、机器人状态与传感器是CPU替身，真正的环境构造没有运行。没有启动AppLauncher、PhysX或训练。完整结果见[result.json](result.json)。

复现：

```bash
/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/bin/python -B docs/audits/2026-10-22_cost_wiring/check_current.py
```

脚本依赖2026-10-21_cost_collection/check.py中原生流程夹具，但将计算、配置和环境钩子改为当前用户文件，在内存修正指定错误后运行；不是复测文档候选。真实200步检查由用户按[18](../../migration/18_cost接线检查与首次运行.md)进行。
