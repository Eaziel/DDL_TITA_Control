# 采样策略短交互检查

内容纯AI生成，由人类要求生成，未经人类修改。记录：2026-11-01。

## 用户实际运行结果

证据为用户本轮提供的[原始终端片段](user_check_log.txt)，日志UTC时间为2026-10-31 16:19，即北京时间11月1日00:19。助手未代运行脚本。

| 项目 | 输出 |
| --- | --- |
| 设备 / 种子 / 环境数 | cuda:0 / 42 / 4 |
| 物理 / 策略步长 | 0.0025s / 0.01s |
| obs / actions / costs | (4,586) / (4,8) / (4,6) |
| mean / std参数 / log_prob | (4,8) / (8,) / (4,) |
| 首步总熵 | 11.3515，与8个初始标准差为1的独立正态分布一致 |
| 完成步数 | 200/200，每环境累计2秒 |
| 重置 / 超时计数 | 14 / 0；重置原因未逐次记录 |
| 退出 | SimulationContext cleared，Simulation App Shutting Down，无异常栈 |

本次shape、有限值及正标准差断言未触发。脚本不含参数更新，因此不构成NP3O训练验证，也不证明接触物理、15秒推扰、20秒超时或课程正确。

## 警告属于哪里

1. `rtx.hydra`：OpenUSD/RTX渲染链路。日志提示同时启用从Fabric读取物体变换和几何流送，可能使动态物体的显示更新不正确。此处Hydra与配置系统Hydra同名；警告不是任务配置解析错误。若画面正常，暂不据此改任务或安装；若物体消失、残影、显示不跟随运动，再针对渲染同步核查。不能把警告本身当作已发生故障。
2. `/Visuals/Command/velocity_current`、`velocity_goal`：Lab速度命令的实际/目标速度箭头。当前`env_cfg.py`的`base_velocity.debug_vis=True`启用，Lab的`commands_cfg.py`指定路径、`velocity_command.py`创建标记。`PointInstancer`通过引用原型模型绘制多个标记；警告指出USD和Fabric的原型数据不一致。`material:binding ... /proto0/Looks`指箭头显示材质，不是机器人/地面的摩擦或恢复系数。
3. `VtValue extraction`和`invalid data source`：USD/Fabric读取值或转换数组时的警告。两条本身未给对象路径；与箭头警告同期出现，可能属于同一可视化链路，但不能逐条确定根因。`USD Diagnostics ... muted`只说明更详细的USD诊断被隐藏。
4. 首行`IStageReaderWriter v0.16 against v0.14`：底层Fabric接口请求版本不一致的提示，具体由哪个插件请求无法从片段确认。此次继续运行至200步，故不是本次启动阻断；也没有证据把它归为旧Conda残留。若出现崩溃或数据更新异常，再取完整Kit日志定位，不据单行警告重装。

助手没有为消除警告修改源码、Kit设置或依赖。若用户希望隐藏速度箭头，可自行将`base_velocity`中的`debug_vis`改为`False`；只关闭该调试显示，不改变速度命令的生成或奖励计算，也不保证消除其他插件警告。

依据：[OpenUSD Hydra架构](https://openusd.org/25.11/api/_page__hydra__getting__started__guide.html)、[NVIDIA对PointInstancer警告的说明](https://github.com/isaac-sim/IsaacLab/issues/4119)、本地`/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py`及`commands_cfg.py`。上游issue曾报告特定3.x场景已解决，本次日志仍出现，不能据关闭状态认定本机已解决。
