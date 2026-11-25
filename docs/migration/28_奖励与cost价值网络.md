# 28 奖励与cost价值网络

内容纯AI生成，由人类要求生成，未经人类修改。整理：2026-11-02。

**状态：已写入并接check；用户11月2日提供[4环境200/200步输出](../audits/2026-11-02_value_networks/README.md)，奖励价值(4,1)、cost价值(4,6)，正常退出，尚未训练。** 以下保留接入步骤供理解，已写内容不用重复复制。

## 1. 本步完成什么

给同一份586列观测增加两个输出：奖励价值(N,1)、六类cost价值(N,6)。critic估计未来的折扣累计量；环境返回的是当步reward/cost。下一环再用交互数据构造目标和训练损失。

候选代码、数学含义和逐步shape见[critic.py解释](../../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/critic.py.md)。按该页完整代码新建Tita_Lab/src/Tita_Lab/learning/np3o/critic.py；它是普通训练模块，直接import，不修改MDP、.pyi、env_cfg或任务注册。

## 2. Gym怎样计算，为什么这样迁移

| 部分 | 原Gym | 候选Lab训练模块 |
| --- | --- | --- |
| 扫描 | 187→128→64→32，末尾两次ELU | 保留实际工厂输出的两次ELU |
| 36维额外状态/参数 | priv_encoder_dims=[]，直接通过 | obs[:,220:256] |
| 历史 | StateHistoryEncoder，十帧时间卷积→32维 | 保留33→30逐帧映射、两次Conv1d及30→32输出 |
| 拼接 | 当前33、额外36、扫描32、历史32 | 顺序相同，共133维 |
| 奖励价值 | 133→512→256→128→1 | reward_head，输出保持线性 |
| cost价值 | 133→512→256→128→6，再Softplus | cost_head，逐类输出非负预测 |

结构与参数来源见解释文档第8节。共享编码只计算一次并返回两个输出，是本项目的职责组织方式；它不把两个价值分支变成同一个预测，也不新增另一个RL算法。

## 3. check脚本的修改

脚本位置：Tita_Lab/scripts/check_np3o_interface.py。已有policy调用和环境step保持，以下是新增片段。

导入：

```python
from Tita_Lab.learning.np3o.critic import TitaValueNetworks
```

run_check中，在policy.eval()之后创建模型：

```python
value_networks = TitaValueNetworks().to(device=env.device)
value_networks.eval()
```

循环中，在entropy = policy.entropy之后、env.step之前添加：

```python
# 与本次采样动作使用同一份obs；这里只检查预测接口，不进行训练。
reward_values, cost_values = value_networks(obs)
assert reward_values.shape == (env.num_envs, 1)
assert cost_values.shape == (env.num_envs, env.num_cost)
assert torch.isfinite(reward_values).all(), "Non-finite reward value."
assert torch.isfinite(cost_values).all(), "Non-finite cost value."
assert (cost_values >= 0.0).all(), "Negative cost value."
```

已有if completed == 1的分支中，再添加一条输出：

```python
print(
    f"[VALUE] reward={tuple(reward_values.shape)} "
    f"cost={tuple(cost_values.shape)}"
)
```

这些计算仍位于已有torch.inference_mode()内，不反向传播。cost_values和env.step返回的costs是不同变量：前者预测未来累计量，后者记录刚完成一步的六类实际cost。

## 4. 用户运行

```bash
source "$HOME/my_venvs/lab3_sim62/.venv/bin/activate"
cd /mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab
python scripts/check_np3o_interface.py \
  --num_envs 4 --max_steps 200 --device cuda:0 --viz kit
```

用户已有输出确认[VALUE] reward=(4,1) cost=(4,6)，并完成200/200步。当前随机初始化值不用于评价机器人状态好坏；本步也不要求数值接近真实回报。

## 5. 验证边界与后续

本阶段已取得用户接口运行证据，助手的结构/语法核对不冒充仿真。后续29存储与31GAE也已完成短检查，详细学习目标见storage解释第13节；本章对应当时接口阶段；后续33已完成单批critic更新，35已生成拟合图，完整训练尚未接通。

