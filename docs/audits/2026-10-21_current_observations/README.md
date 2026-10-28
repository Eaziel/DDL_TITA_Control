# 2026-10-21 当前观测代码复核

本页用途是保留该日期的检查范围与复现方法。文中的“当前/候选/待写”按检查当时理解；后续实现状态见[项目记忆](../../../PROJECT_MEMORY.md)。2026-10-26文档整理未重新执行本页检查。

内容纯AI生成，由人类要求生成，未经人类修改。

check_current.py复用2026-10-20两个CPU夹具，改为导入当前保存的actions、observations、events、环境reset、cfg和布局函数。历史候选与结果文件不变；用户两个源码目录未修改。

- state_result.json：4环境×12策略步×4物理调用，开/关final_obs，部分reset、推扰和重复读取；六维对照旧Gym表达式并转换轮序后误差0，586拼接和末端裁剪通过。其他四组为固定数据，TitaVelocityEnv.step的历史/奖励合成不在这项检查范围内。
- dynamics_result.json：8环境，原生ObservationManager/EventManager、实际TitaPD及CPU射线变换核；当前scan/dynamics配置、质量/COM差值、材质和因子重排通过。六维组在这一夹具中关闭，另项单独验证。
- config_result.json：当前完整cfg.validate与七文件AST通过。
- diagnostics.json：Ruff仅observations.py导入.actions/.events顺序I001，其余无诊断；没有自动修复源码。

物理状态、资产setter和射线命中值仍有CPU替身，未启动AppLauncher、PhysX、GPU传感器或训练。用户报告CLI显示观测维度，没有附最新真实运行日志。CPU通过不能替代物理数值验证。

复现：项目根目录用既有环境执行`/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/bin/python -B docs/audits/2026-10-21_current_observations/check_current.py`。脚本仅写本审查目录，使用/tmp的Warp缓存；不运行Git或仿真。
