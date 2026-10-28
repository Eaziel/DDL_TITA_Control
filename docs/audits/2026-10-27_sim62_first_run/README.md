# Sim 6.2：首次任务运行与事件接口核查

日期：2026-10-27。内容纯 AI 生成，由人类要求生成，未经人类修改。

**结果：用户已写入首组三处修改，新环境完成任务初始化、进入 zero-agent 循环并清理退出。** 用户报告 GUI 出现、表现与此前一致。依据为[用户日志片段](user_log_excerpt.txt)及本轮只读源码检查；助手没有运行仿真。

## 日志与当前文件

- PhysX / cuda:0，1环境；物理步0.0025 s、策略步0.01 s。
- Articulation、ContactSensor、RayCaster均选择PhysX，地形网格被读取。
- 五组观测33/330/187/30/6、动作8；命令/事件/奖励/终止/课程均建立。
- zero-agent入口在完成首次env.reset之后打印运行提示，最终有SimulationContext清理与应用关闭，片段无traceback。
- 没有打印已完成步数、reset次数、推扰或超时触发，不能把建议的200步当作日志已经证明的计数。

Fabric接口版本、命令箭头等警告保留原文，未诊断根因；未据正常退出认定所有警告无害。

新Python对当前实际文件检查：配置构造/validate通过，实际注册测试通过，events/observations AST通过。三处适配均已写入。检查过程工具沙箱仍报告Warp CUDA error100，不能据此否定用户日志中的真实GPU运行。

## 事件数据路径

读取新checkout的随机化技能、MDP实现和PhysX写入实现后，未找到本组额外必改项：

| 项目 | 当前配置与数据路径 |
| --- | --- |
| 材质 | startup、64候选；Sim6.2 tensor API仍读写(N,S,3)，三列为静摩擦、动摩擦、恢复系数 |
| 基座质量 | startup、base_link增加U(-1,3) kg；原生类调用set_masses_index，并按质量比缩放惯量 |
| 基座质心 | startup、各轴加U(-0.1,0.1) m；原生类保留质心朝向，通过set_coms_index写入 |
| 名义参数 | 新RL环境仍在ObservationManager构造之后执行startup，30维观测保存名义值的时机保留 |
| reset | 现有关节缩放和根状态采样配置继续对应原生接口 |

PhysX setter处理体/环境索引、更新缓存并提交tensor API。本轮没有实际参数回读的逐项数值对照。

依据：[main MDP API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.envs.mdp.html)，新Lab的envs/mdp/events.py与PhysX articulation.py，以及本机Sim6.2 omni.physics.tensors/api.py。

## 编辑器

只读发现用户extraPaths仍指旧Lab/Conda/tensor扩展，工作区仍默认Conda管理器。新SimulationContext.__init__明确声明self.backend_manager: PhysicsManager，补全问题先处理分析路径。

官方编辑器配置的只读发现函数找到12个新Lab包根、无旧路径，但未发现Sim扩展，因此没有把原样运行isaaclab --editor说成完整修复。没有执行生成器、没有改编辑器设置。建议用户在工作区设置已核实的新路径，选择新解释器并重载；说明见[官方编辑器文档](https://isaac-sim.github.io/IsaacLab/main/source/developer-tools/editor_setup.html)。

## 下一次运行

四环境、随机动作、2200策略步预算即22 s，env.seed=42。目标为非零动作、多环境reset和经过15 s interval时刻；随机策略的平衡能力不是本次判据。提前关窗口不能证明预算跑完，提前失败重置也使22 s总运行不保证单回合20 s超时。

该physics/seed/环境数组合已在新Python通过resolve_task_config和validate，输出LONG_RUN_CONFIG_OK PhysxCfg 42 4 0.01；没有实际长运行。命令见[25迁移说明](../../migration/25_Lab_main与Sim62_入口和配置.md)。记录与源码哈希见[result.json](result.json)。未改两个源码工程、Git元数据或环境依赖。
