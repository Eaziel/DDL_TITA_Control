# 新 main / Sim 6.2：首组配置核查

日期：2026-10-27。用户源码未修改；未运行 Git、安装、Kit、物理仿真或训练。

目标 Python：`/home/eaziel/my_venvs/lab3_sim62/.venv/bin/python`；Lab 源码：`/home/eaziel/IsaacLab-main`。本轮直接读取该 checkout 中的迁移、传感器/执行器和编码规范技能，并核对官方 main / Sim 6.2 文档。修改方案见[迁移说明 25](../../migration/25_Lab_main与Sim62_入口和配置.md)。

## 实际执行

1. 当前 `TITA_CFG` 绑定 `/World/Robot` 后调用 validate：通过，USD 文件路径存在。没有绑定 `prim_path` 的基础模板不能直接作为完整场景配置校验，这不是源码缺陷。
2. 导入当前任务并按注册加载配置：失败，真实错误为 `RayCasterCfg.__init__() got an unexpected keyword argument 'spawn'`。
3. 用 AST 仅在内存移除当前配置中 `RayCasterCfg` 的 `spawn` 关键字，按原包上下文执行并构造 `VelocityEnvCfg`、调用 validate：通过。数值仍为 dt=0.0025、decimation=4、20 s，材质类为 `PhysxRigidBodyMaterialCfg`。此时 physics 仍是预设集合，未创建或运行求解器。
4. 执行现有 `test_task_registrations()`：原预期因环境入口不符而失败；内存中将预期改为 `TitaVelocityEnv` 后通过。未写测试源文件。
5. 源码核查：新 `SimulationContext.backend_manager` 是实例，`physics_manager` 为弃用别名；`PhysxManager` 实现直接导入 Kit/Omni，需将实现类导入延后至事件构造。该候选尚未在真实 Kit 中实例化。

检查过程遇到 Warp 在工具沙箱中报告 CUDA error 100，不能据此判断宿主 GPU 不可用。Matplotlib 因用户配置目录不可写回退到临时缓存，另有依赖弃用提示。它们没有阻止候选配置 validate；也不构成 GPU 验收。

结果与源码哈希见 [result.json](result.json)。哈希仅核实记录中的普通文件内容，不读取 Git，不代表提交、远端状态或完整环境备份。

## 依据

- 新/旧本地 `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_cfg.py`：旧有 `spawn`，新无该字段。
- 新 `source/isaaclab_tasks/isaaclab_tasks/core/velocity/velocity_env_cfg.py:108`：直接跟踪已有刚体的官方配置。
- 新 `source/isaaclab/isaaclab/sim/simulation_context.py:190,297`：管理器实例及弃用别名。
- 新 `source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py:27`：Kit/Omni 运行依赖。
- [Ray Caster 官方说明](https://isaac-sim.github.io/IsaacLab/main/source/concepts/sensors/ray_caster.html)、[新版启动说明](https://isaac-sim.github.io/IsaacLab/main/source/how-to/launch_app.html)、[Sim 6.2 Core API 迁移](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/migration_guides/isaac_sim_6_0/core_api_to_core_experimental.html)。

以上只支持配置/API 层的结论。真实材质读写、机器人关节/碰撞、扫描值、reset/step、训练结果均待后续验证。
