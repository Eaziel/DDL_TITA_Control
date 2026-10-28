# 新venv：已保存动作、wrapper、actor的兼容性检查

日期：2026-10-27。内容纯AI生成，由人类要求生成，未经人类修改。

**结果：用户动作修正已保存；实际FIR、wrapper、actor计算在新Python/Torch中通过CPU检查。仅记录既有模块检查；当时提出的actor联调脚本已撤回。**

## 1. 用户运行证据

[日志](user_post_fir_log.txt)显示PhysX/cuda:0、4环境、seed42、随机动作入口、五观测组及八动作，末尾清理退出且无Ctrl+C/traceback。用户明确报告“自行结束，actions已经修改”，对应上一轮200步短命令；日志自身未打印步数。[记录](runtime_summary.json)区分用户报告与日志字段。没有15s推扰/20s超时的触发计数。

## 2. 当前文件的CPU执行

使用`/home/eaziel/my_venvs/lab3_sim62/.venv/bin/python`，Torch2.11.0+cu130，执行[check_saved_stack.py](check_saved_stack.py)：

| 范围 | 结果 | 边界 |
| --- | --- | --- |
| 当前FIR | 输入10/0/200/-200得到8/2/80/-60；四环境六腿/两轮目标与公式差0；action-rate恢复 | CPU缓存、实际父类与奖励，不是物理力矩回读 |
| 当前wrapper | 586列拼接/裁剪、reset/get、六返回、cost直接传递、done/timeout、日志副本、回合计数转发和close通过 | Gymnasium CPU状态夹具；使用实际类TitaNP3OEnvWrapper及num_cost字段 |
| 当前actor | batch1/4/64前向与旧网络差0；辅助batch4/32损失差约3.81e-6/0、参数梯度差0、BN缓冲和两历史窗口一致 | 无优化器更新、GPU网络或训练 |

actor检查复用2026-10-26的数值检查逻辑，但用本轮解释器读取实际actor源码，输出定位到本目录actor子目录；没有覆盖旧证据。[总结果](result.json)、[actor细节](actor/result.json)保留哈希和实际方法名`auxilary_loss`，该名字仍待用户统一为`auxiliary_loss`。

## 3. 传感器与时序核查

新技能、官方文档与实际源代码支持保留现有字段和时机：

- 接触读取仍用net_normal_forces_w.torch；默认update_period=0的接触传感器随物理步更新。Sim6.2的net_forces_w已包含摩擦，继续用显式法向字段保持当前任务判据。
- 高度扫描仍从pos_w与ray_hits_w取高度差；机身绑定、187点、0.01 s更新和20 m射线起点偏移继续沿用。
- 当前PhysX路径在每个物理步apply/write/step/scene.update，每策略步再加common_step_counter、算终止/奖励；reset和interval事件后算最终观测。六维项的按策略步缓存仍有对应调用位置。

以上是API/实现检查，没有逐点验证真实射线和接触力数值。[接触文档](https://isaac-sim.github.io/IsaacLab/main/source/concepts/sensors/contact_sensor.html)、[射线文档](https://isaac-sim.github.io/IsaacLab/main/source/concepts/sensors/ray_caster.html)。

## 4. 当时的脚本候选（已撤回）

[proposal_check_np3o_interface.py](proposal_check_np3o_interface.py)是给用户写入`TITA_Lab/scripts/check_np3o_interface.py`的候选，不是已经安装的入口。它启动真实任务，用现有wrapper与随机初始化actor做前向交互，检查有限值与shape，并打印steps/resets/timeouts。

其run_check函数已用实际wrapper/actor及CPU状态夹具验证：5/5步、模拟3次done/2次timeout；提前关闭条件返回3/5；最后一步注入NaN也能捕获。[候选检查](proposed_rollout_result.json)明确不是实际Kit/GPU环境运行。启动方式按新main原生simple_agents入口和launch_simulation实现组织，cfg组合已解析validate通过；真实main尚未执行。

助手没有修改两个源码工程、编辑器、依赖或Git；没有启动仿真或训练。工具沙箱的Warp CUDA error100不用于判断用户宿主GPU状态。该联调脚本建议已撤回，不属于当前已有代码检查的待办。
