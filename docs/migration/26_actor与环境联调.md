# 26 actor与环境的首次联调

内容纯AI生成，由人类要求生成，未经人类修改。阶段记录整理：2026-11-08。

## 1. 本章验证哪一段

这一阶段首次把Lab环境、TitaNP3OEnvWrapper和动作均值网络连起来：环境提供586列，actor读取当前33列与历史330列，输出8维动作，再交给环境执行。

此后27加入高斯动作分布，28加入价值网络，29加入存储，31加入GAE。当前check已包含这些后续功能，不能再用本阶段的早期actor-only脚本覆盖它。

当前完整脚本与逐句解释统一放在[check说明](../../code_explaination/Tita_Lab/scripts/check_np3o_interface.py.md)，本页删除已被替代的整份草稿，只保留阶段职责与结果入口。

## 2. 当前的调用关系

```python
actions = policy.act(obs=obs)
reward_values, cost_values = value_networks(obs)
next_obs, _, rewards, costs, dones, infos = env.step(actions=actions)
```

policy内部仍调用TitaHistoryActor计算均值，只是现在会根据均值与std采样动作；critic另行预测奖励和cost价值。后续storage保存相配数据，GAE构造学习信号。整条调用由run_check显式组织，未使用完整训练runner。

场景启动仍是解析环境配置、进入launch_simulation、gym.make、建立自有wrapper，最后调用run_check并清理。它不需要MDP的.pyi注册，也不会自动读取默认PPO agent配置。

## 3. 怎样查结果和运行

- Gaussian policy的200步记录见[27](27_动作分布迁移.md)。
- 奖励/cost价值网络的200步记录见[28](28_奖励与cost价值网络.md)。
- 最新普通与含超时24步记录见[GAE证据](../audits/2026-11-04_gae/README.md)。
- 当前手动命令见[25](25_Lab_main与Sim62_入口和配置.md)和[31](31_回报目标与GAE.md)。

当前网络没有更新参数，短联调验证数据和接口，不代表已经学会平衡。actor的Torch基础和辅助损失按需阅读[actor解释](../../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/actor.py.md)。
