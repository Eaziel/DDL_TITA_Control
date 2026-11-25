# 31 奖励与cost的回报目标、GAE优势

内容纯AI生成，由人类要求生成，未经人类修改。阶段记录整理：2026-11-08。

**已写入并完成两次用户短检查。** 普通与短回合检查均为4环境24/24步，四份GAE结果有限，正常退出；统计分别1/0和16/16次reset/timeout。运行摘录见[GAE证据](../audits/2026-11-04_gae/README.md)。本章对应11月4日GAE阶段；后续33／34已完成单批critic与奖励策略更新，完整NP3O训练尚未接通。

## 1. 这一步将数据用于什么

环境reward/cost公式计算单步结果，critic预测累计回报。rollout保存这些经历；本步利用实际结果和旧价值预测，形成critic学习目标与策略/约束更新需要的优势。

| 结果 | shape | 后续用途 |
| --- | --- | --- |
| reward_returns | (T,N,1) | 奖励critic拟合目标 |
| cost_returns | (T,N,6) | 六类cost critic各自的拟合目标 |
| reward_advantages | (T,N,1) | 奖励对应的策略更新信号 |
| cost_advantages | (T,N,6) | 六类约束对应的更新信号 |

计算目标不会自动修改网络。当前只保存原始优势，没有标准化、损失或优化器。

## 2. 完整数学与逐句代码在哪里

[rollout_storage.py第13节](../../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/rollout_storage.py.md)是主讲解：从一步自举目标为何成立、TD误差在比较什么，讲到多步估计、GAE加权关系、倒序递推和compute_returns全部语句；包含数值例子、shape、no_grad、切片和copy_。

[check 第 7、9 节](../../code_explaination/Tita_Lab/scripts/check_np3o_interface.py.md)说明数据怎样取得，包括动作前的 obs、step 返回的 next_obs、reset 前的 final_obs 及整批末尾的 last_values。两份说明都在末尾折叠附上完整源码。

## 3. Lab与训练侧怎样分工

| 部分 | 当前职责 |
| --- | --- |
| Lab的compute_final_obs开关 | 在同一步自动reset之前保留最终观测 |
| wrapper | 六项返回不变，infos额外保留terminated及final_obs |
| check | 选择超时且未真实终止的行，调用critic并填入timeout_*；采集结束后计算last_* |
| storage.add | 独立复制13项来源，不改原始reward/cost，不重复乘dt |
| storage.compute_returns | 使用旧预测、普通后续价值或纯超时最终价值，倒序计算四份结果 |

真实终止取后续价值0；纯超时取reset前最终状态价值。两种结束都停止跨回合累加优势，避免新回合进入旧回合目标。整批24步结束但回合未结束时，用last_values补足后续价值。

本次选择明确的最终状态价值；与本地RSL-RL按当前状态价值近似超时补偿的实现有区别。比较旧训练结果时保留这一口径记录，不宣称逐数复现。

## 4. 参数来源与独立参考

gamma=.99、lam=.95来自原任务继承的[配置](/home/eaziel/Downloads/tita_rl-master/configs/legged_robot_config.py:266)。它们控制折扣与多步估计，与物理dt和decimation不同。

GAE的独立理论参考为[Schulman等原论文第3节](https://arxiv.org/html/1506.02438v6#S3)；实现结构参考本地[RSL-RL PPO.compute_returns](/home/eaziel/my_venvs/lab3_sim62/.venv/lib/python3.12/site-packages/rsl_rl/algorithms/ppo.py:166)。本步没有使用旧TITA现成NP3O更新代码。

## 5. 保留的手动检查命令

激活lab3_sim62并进入Tita_Lab后，由用户执行：

```bash
python scripts/check_np3o_interface.py \
  --num_envs 4 --max_steps 24 --device cuda:0 --viz kit

python scripts/check_np3o_interface.py \
  --num_envs 4 --max_steps 24 --episode_length_s 0.05 \
  --device cuda:0 --viz kit
```

第二条将本次检查的回合长度覆盖为0.05s，方便触发时间上限；不传选项时仍保留20s任务配置。若提前关闭窗口，未采满会跳过GAE。完整输出和检查边界见运行记录。

## 6. 剩余工作

下一层是接NP3O损失与参数更新，再由runner组织多批采集、更新、日志与检查点。选定外部NP3O实现后再核实优势标准化、cost约束尺度、辅助损失和minibatch等接口，当前未确定依赖。

GAE与check已完成的短计算，不代替真实训练、逐元素参考对照或任务效果验证。后续[32小批量取数](32_更新前的小批量数据.md)已由用户写入并通过短检查；当前完整状态统一见项目记忆。
