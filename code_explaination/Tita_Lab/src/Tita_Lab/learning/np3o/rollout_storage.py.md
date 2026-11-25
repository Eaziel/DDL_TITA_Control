# rollout_storage.py：把每一步经历整理成两种学习信号

内容由 AI 整理。2026-11-25 对照源码重写。

被解释文件：[Tita_Lab/src/Tita_Lab/learning/np3o/rollout_storage.py](../../../../../../Tita_Lab/src/Tita_Lab/learning/np3o/rollout_storage.py)。

先看第 1 节和第 13 节的两步例子，就能把“存经历、算优势、交数据”接起来。需要自己对照代码时，再读相应的小节。

## 1. storage 为什么要留下一整段经历

机器人的动作会改变后面的运动。假设 TITA 第 0 步追上了速度命令，这一步拿到奖励。机器人在第 1 步为了稳住身体，关节加速度才超出阈值。如果程序只留下第 0 步的奖励，就没法用第 1 步的 cost 评价刚才的动作。如果程序只保留最后一步，又会丢掉产生前一个动作的观测。

`TitaRolloutStorage` 先把每台机器人每一步的记录留住。采满后，`compute_returns` 按时间倒着看这批经历，计算回报目标和优势。两者都考虑了后续结果。回报目标给 critic 一个数值参考，优势给策略一个相对旧预测的评价。最后，`mini_batch_generator` 把多条完整记录一起交给更新函数。

## 2. 一批 24 步和一个回合有什么关系

```python
self.num_envs = num_envs
self.num_steps = num_steps
self.device = torch.device(device)
```

当前检查同时推进 4 个环境。`num_steps=24` 表示这批数据留 24 个时间位置。一次采集会给这 24 个位置之一填 4 条经历。一批采满后共有 96 条经历。

回合由机器人自己的开始和结束决定。一台机器人可以在这 24 步内跌倒并 reset，其他机器人继续走。有的回合也可能延续到这批之外。GAE 用结束标记识别回合边界。未来持续训练还会再采下一批，第二批可以从第一批留下的环境状态继续。

## 3. 一条经历要存哪些数

```python
entries = (
    ("observations", self.observations, observations),
    ("actions", self.actions, actions),
    ("rewards", self.rewards, rewards.unsqueeze(-1)),
    ("costs", self.costs, costs),
    ("dones", self.dones, dones.unsqueeze(-1)),
    ("time_outs", self.time_outs, time_outs.unsqueeze(-1)),
    ("reward_values", self.reward_values, reward_values),
    ("cost_values", self.cost_values, cost_values),
    ("log_probs", self.log_probs, log_probs.unsqueeze(-1)),
    ("action_mean", self.action_mean, action_mean),
    ("action_std", self.action_std, action_std),
    ("timeout_reward_values", self.timeout_reward_values, timeout_reward_values),
    ("timeout_cost_values", self.timeout_cost_values, timeout_cost_values),
)
```

以第 $t$ 拍为例，观测和动作要配对。`observations[t,n]` 是第 $n$ 台机器人在动作**之前**看到的东西，`actions[t,n]` 是用这份观测抽到的动作，`rewards[t,n]` 和 `costs[t,n]` 是环境执行这个动作后给出的本步结果。

| 存储字段 | 单条经历的内容 | 什么时候用 |
| --- | --- | --- |
| `observations`、`actions` | 586 列旧观测、8 列原始动作 | critic 重算价值，policy 对旧动作重算密度 |
| `rewards`、`costs` | 1 列当步奖励、6 列当步 cost | 算 TD 误差和 GAE |
| `dones`、`time_outs` | 是否结束、是否触发时间上限 | 判断未来价值来源和回合边界 |
| `reward_values`、`cost_values` | 动作前旧 critic 的两种价值预测 | 算回报目标与优势 |
| `log_probs` | 旧策略给整组动作的对数密度 | 作为 PPO 密度比的分母依据 |
| `action_mean`、`action_std` | 采集时的高斯参数，各 8 列 | 描述旧分布，当前损失用旧 `log_probs` |
| `timeout_reward_values`、`timeout_cost_values` | 纯超时后，reset 之前最后状态的旧价值 | 给纯超时步补后续价值，其他位置由 check 填 0 |

## 4. 时间、环境、特征三个轴分别指什么

奖励按 `(T,N,1)` 存，cost 按 `(T,N,6)` 存。$T$ 是本批推进的步数，$N$ 是环境数，末轴是每条经历要留几列。同一个 `[t,n]` 在不同张量中都指同一条经历。比如 `costs[5,2,0]` 是第 2 号环境、第 5 个时间位置、第一类 cost。代码索引从 0 开始。

`destination[self.step]` 先选一个时间位置，剩下 `(N,D)`。$D$ 随字段变。写入的来源必须和选出的目的切片形状一致。

这里还有一个容易混淆的“历史”。一条 `observations[t,n]` 的 586 个观测数中，已经包含由环境维护的过去十帧。那十帧供网络决策。storage 的时间轴则保留这一批的每一步经历，供程序在采样后计算和取数。

## 5. 程序提前分配哪些数据

```python
self.reward_returns = torch.zeros_like(self.reward_values)
self.cost_returns = torch.zeros_like(self.cost_values)
self.reward_advantages = torch.zeros_like(self.reward_values)
self.cost_advantages = torch.zeros_like(self.cost_values)
```

构造函数用 `torch.zeros` 分配固定大小的张量，用 `zeros_like` 建立形状、设备和数据类型相同的缓冲。`device` 与网络和环境保持一致，GPU 上的 rollout 可以直接交给同设备上的网络。

`reward_returns`、`cost_returns` 和两份优势也先分配空间。刚创建时的零只是占位值。调用方后来要看 `_returns_ready`，才能知道这批的学习信号是否写好。

## 6. 为什么标量奖励还要保留一列

环境给每台机器人一个奖励，返回形状 `(N,)`。`rewards.unsqueeze(-1)` 把每个数放到一行的一列，形状成为 `(N,1)`。比如四个数 `[r0,r1,r2,r3]` 会变成四行一列。`dones`、`time_outs`、`log_probs` 也做同样处理。critic 已经用一列输出奖励价值，所以 `reward_values` 直接保存。

形状统一后，四台机器人的奖励与四个价值预测可以一行对一行相加。若把 `(4,)` 直接和 `(4,1)` 相加，PyTorch 会按末轴对齐并广播成 `(4,4)`。那样每台机器人的预测会与四台机器人的奖励都组合一次。[PyTorch 广播规则](https://docs.pytorch.org/docs/2.11/notes/broadcasting.html)

## 7. add 怎样先核对再写入

```python
for name, destination, source in entries:
    expected = destination[self.step].shape
    if source.shape != expected:
        raise ValueError(f"Expected '{name}' with shape {tuple(expected)}, got {tuple(source.shape)}.")

for _, destination, source in entries:
    # 保存独立数值，不保留采样时的梯度计算图。
    destination[self.step].copy_(source.detach())

self.step += 1
```

`entries` 是一组条目。每条有字段名、整批目的张量和本步来源。第一个循环只是检查所有形状。这样如果第 13 个字段形状出错，前 12 个字段也还没有写入。第二个循环才实际复制。`self.step += 1` 放在复制之后，所以下次 `add` 会写下一个时间位置。

方法签名里的独立 `*` 要求调用方写出参数名，例如 `observations=obs`。字段多时，名称能提醒调用方正在传哪个时刻的量。

## 8. detach 和 copy_ 各自负责什么

`source.detach()` 取出同一份数值，但结果与原来的求导关系断开。返回张量仍可能共享原来的内存。`destination[self.step].copy_(...)` 再把这些数值写进提前分配的存储。以后环境缓冲更新、网络再前向，已复制的旧记录仍能被读取。

训练时还能对固定观测求出参数的梯度。用一个权重的例子就能看见，$y=wx$ 中的 $x$ 是固定输入，$y$ 仍然随 $w$ 变化。策略和 critic 在更新阶段重新前向，所以从新的损失可以求出当前网络参数的导数。[PyTorch detach](https://docs.pytorch.org/docs/2.11/generated/torch.Tensor.detach.html)

## 9. 为什么“旧”密度和“旧”价值都要留着

采集动作时，策略和 critic 都有当时的参数。GAE 用旧 critic 的估计与实际后果比较，才能形成相对于采集策略的优势。更新策略时，当前网络对同一个旧动作算一次新密度，再除以采集时保存的旧密度。若把分母也重算成当前密度，比值就失去了比较作用。

当前脚本先完成整批采集，再更新 critic 和策略。`storage` 保存的是旧结果的数值，后来的网络更新不会改写这些记录。

## 10. 调用方在什么时候换成 next_obs

```python
storage.add(
    observations=obs,
    actions=actions,
    rewards=rewards,
    costs=costs,
    dones=dones,
    time_outs=time_out_flags,
    reward_values=reward_values,
    cost_values=cost_values,
    log_probs=log_prob,
    action_mean=policy.action_mean,
    action_std=policy.action_std,
    timeout_reward_values=timeout_reward_values,
    timeout_cost_values=timeout_cost_values,
)
obs = next_obs
```

脚本用 `obs` 产生动作，环境执行动作后才得到 `next_obs`。`storage.add` 归档刚刚结束的这一拍，所以调用方先传旧 `obs`，再用 `obs = next_obs` 准备下一拍。如果某个环境已经 reset，`next_obs` 对应的那一行是新回合的观测。纯超时步需要的旧回合终点价值由 check 另行计算。[check 第 7 节](../../../../scripts/check_np3o_interface.py.md#7-下一份观测和-reset-前观测为什么要分开)

## 11. 哪一批数据已经可以交给更新函数

```python
@property
def is_full(self) -> bool:
    """Return whether the configured rollout has been collected."""
    return self.step == self.num_steps
```

```python
def clear(self) -> None:
    """Mark all stored transitions as consumed."""
    # 下次从第0步覆盖；只有[0:self.step]是当前有效数据。
    self.step = 0
    self._returns_ready = False
```

`is_full` 是属性。读取它时会执行方法中的比较。24 次采集成功后，`step == num_steps` 才成立。`compute_returns` 要求已采满，算完后把 `_returns_ready` 设为真。小批量生成器要求这两个条件同时满足。

持续训练的调用方用完一批后，可以用 `clear()` 把写入位置移回 0 并把目标标记成未就绪。旧内存会在下一批被覆盖。当前 check 只用一批。某台机器人 reset 时，记录仍留在本批，GAE 根据这条记录的 `done` 识别边界。

## 12. 普通和超时的历史检查

2026-11-04 两次检查各采 4 环境 24 步。普通运行发生 1 次 reset、0 次超时。短回合运行发生 16 次 reset、16 次超时。两个运行都算出了形状正确且有限的目标和优势。2026-11-14 的日志还记录了这批数据进入 critic 和约束策略各 8 次更新。检查日志给出运行结果，下面的手算单独解释逐步数值。[GAE 检查记录](../../../../../../docs/audits/2026-11-04_gae/README.md) · [约束策略检查记录](../../../../../../docs/audits/2026-11-14_constrained_policy_update/README.md)

## 13. GAE 把后几步结果算进当前动作

### 13.1 先用一步结果修正旧预测

假设机器人已经执行第 $t$ 步。程序知道这一步发生了多少奖励或 cost，也有旧 critic 对下一状态的预测。如果程序把两部分加起来，就有了一个用新经历修正过的累计量估计。用这个估计减去动作前的旧预测，就知道原先估高了还是估低了。

为了用同一组符号讲奖励和 cost，下面用 $q_t$ 指其中一种当步结果，比如一类 cost。$V_t$ 是旧 critic 对当前观测的预测，$B_t$ 是代码选择的后续价值。$\gamma$ 表示未来一拍的折扣。一步修正量叫 TD 误差，记作 $\delta_t$。

$$
\delta_t=q_t+\gamma B_t - V_t.
$$

普通步选择下一状态的旧价值作为 $B_t$。例如当前 cost 为 0，旧预测为 0.2，下一状态旧预测为 0.4，折扣是 0.9。一步目标是 $0+0.9\times0.4=0.36$，所以 TD 误差为 $0.36-0.2=0.16$。

### 13.2 后一拍的实际经验怎样接回来

一步目标用旧 critic 代替了下一状态之后的全部结果。但既然我们已经采到后一拍，就可以用后一拍的 TD 误差再修正一次。GAE 把后面每一步的修正带回来，越远权重越小。

记 $\widehat A_t$ 为当前优势，$\widehat A_{t+1}$ 为已经算好的后一拍优势。$\lambda$ 控制要拿进来多少后续修正。若还在同一回合，递推式为：

$$
\widehat A_t=\delta_t+\gamma\lambda\widehat A_{t+1}.
$$

当 $\lambda=0$ 时，优势只用一步 TD 误差。$\lambda$ 越大，实际采到的后续结果参与越多，估计较少依赖近处的旧预测，但也更容易随采样波动。与上式相同，把递推展开就得到 $\delta_t+\gamma\lambda\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots$。因此代码从最后一格倒推，省去每一步都重复展开。[GAE 原论文](https://arxiv.org/abs/1506.02438)

### 13.3 为什么典型的三种“到头了”处理不同

```python
for step in reversed(range(self.num_steps)):
    if step == self.num_steps - 1:
        next_reward_values = last_reward_values
        next_cost_values = last_cost_values
    else:
        next_reward_values = self.reward_values[step + 1]
        next_cost_values = self.cost_values[step + 1]

    # (N,1)：回合未结束为1；失败或超时reset时为0。
    continuation = (~self.dones[step]).to(dtype=self.rewards.dtype)
    time_out = self.time_outs[step].to(dtype=self.rewards.dtype)

    # 普通步使用下一状态价值；纯超时使用reset前最终状态价值。
    # 失败时两项均为0；timeout_*在所有非纯超时位置也为0。
    reward_bootstrap = continuation * next_reward_values + time_out * self.timeout_reward_values[step]
    cost_bootstrap = continuation * next_cost_values + time_out * self.timeout_cost_values[step]
```

| 本步结束后发生什么 | TD 误差中的后续价值 $B_t$ | 优势还能接到哪里 |
| --- | --- | --- |
| 回合继续，本批后面还有记录 | 下一条记录的旧价值 | 接上同一环境下一条的优势 |
| 回合继续，采集窗用完 | 采集完的 `last_*_values` | 本批外的优势用初始零值，TD 中仍补一个价值 |
| 仅因时间上限 reset | reset 前最终观测的旧价值 | 在这里断开，不把新回合的优势接进来 |
| 真实终止，例如失败 | 0 | 在这里断开 |

`continuation` 为 1 时回合继续，为 0 时回合结束。`time_out` 选择超时值，但 check 只在纯超时行填入非零来源。真实终止同时触发超时时，check 留下的超时值仍为零。**“允许补一个未来价值”和“允许接着用下一条记录”是两个判断。** 纯超时保留前者，切断后者。

`continuation` 是 `(N,1)`，cost 值是 `(N,6)`。PyTorch 按末轴广播，给同一台机器人的六列使用同一个结束标记。每台环境保留自己的边界。

### 13.4 用两步把目标和优势同时算出

仍用 check 第 1 节的例子。第 0 步没有 cost，第 1 步才有。假设第 1 步真实终止，$\gamma=0.9$，$\lambda=0.8$。

| 步 $t$ | 奖励 | 一类 cost | 旧奖励价值 | 旧 cost 价值 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1 | 0 | 1 | 0.2 |
| 1 | 2 | 1 | 1 | 0.4 |

从第 1 步倒过来。第 1 步的 cost TD 误差为 $1+0-0.4=0.6$，因为回合在这里结束，cost 优势也是 $0.6$。第 0 步的 cost TD 误差为 $0.16$。第 0 步仍在同一回合，所以 cost 优势为 $0.16+0.9\times0.8\times0.6=0.592$。后一拍的 cost 就这样进入第 0 步的评价。

```python
reward_delta = self.rewards[step] + gamma * reward_bootstrap - self.reward_values[step]
cost_delta = self.costs[step] + gamma * cost_bootstrap - self.cost_values[step]

# A_t = δ_t + γλ * A_{t+1}；reset处停止跨步递推。
reward_advantage = reward_delta + gamma * lam * continuation * reward_advantage
cost_advantage = cost_delta + gamma * lam * continuation * cost_advantage

self.reward_advantages[step].copy_(reward_advantage)
self.cost_advantages[step].copy_(cost_advantage)

# 供critic拟合的目标：旧价值 + 优势估计。
self.reward_returns[step].copy_(self.reward_values[step] + reward_advantage)
self.cost_returns[step].copy_(self.cost_values[step] + cost_advantage)
```

优势表示相对旧预测的修正。回报目标还要把旧预测加回来。第 0 步的 cost 目标为 $0.2+0.592=0.792$，第 1 步目标为 $0.4+0.6=1$。奖励按同样的规律算，两个优势依次为 $1.62$、$1$，两个目标依次为 $2.62$、$2$。

若把例子中的第 1 步 cost 改为 0，而旧预测仍是 0.4，该步的 cost 优势就是 $0-0.4=-0.4$。cost 本身非负，优势的正负表示它相对旧预测是增加还是减少。

还可以用两个成本回报估计理解 $0.792$。只看一步，目标为 $0.36$，余下结果用价值预测补上。看完整两步，目标为 $0+0.9\times1=0.9$。本例的 $\lambda$ 将两种估计按 $0.2$ 和 $0.8$ 混合，得到 $0.2\times0.36+0.8\times0.9=0.792$。因此，这个数叫 $\lambda$ 回报目标。

critic 读 `returns` 拟合累计量。policy 读 `advantages`，对照旧动作密度调整策略。`compute_returns` 用旧价值构造两份信号，`@torch.no_grad()` 关闭这一段的求导记录。函数把数值复制进 storage，随后更新函数读取，当前批里会反复用同一份目标与优势。[check 第 12 节](../../../../scripts/check_np3o_interface.py.md#12-check_value_update从误差到自动求导再到参数更新)

## 14. 小批量怎样一次取出整条记录

```python
flat_fields = {name: values.flatten(0, 1) for name, values in fields.items()}

for _ in range(num_learning_epochs):
    # 每轮重新排列全部记录的编号，每条记录恰好出现一次。
    indices = torch.randperm(batch_size, device=self.device)
    for start in range(0, batch_size, mini_batch_size):
        batch_indices = indices[start : start + mini_batch_size]
        # 同一组编号用于全部字段，yield每次交出一份小批量数据。
        yield {name: values[batch_indices] for name, values in flat_fields.items()}
```

### 14.1 先给每条经历编号

如果只有 3 步、2 个环境，`flatten(0, 1)` 会把前两轴合成 6 行，顺序是 `(步0, env0)`、`(步0, env1)`、`(步1, env0)`，直到最后一行。编号 $i=tN+n$，意味着每个时间位置先排完 $N$ 个环境，再排下一个时间位置。每个字段都用相同顺序展平。

### 14.2 打乱编号，再一起取行

每个 epoch 用 `torch.randperm` 打乱全部编号。生成器每次选出一组 `batch_indices`，再对字典里的每个字段都用这组编号选行。只要旧观测、旧动作、奖励目标等仍然同行，网络就可以把多条记录一起处理。

`yield` 交出一份字典后先停在原处。调用方的 `for` 处理完这份小批量，才来取下一份。96 条分 4 份，每份 24 条。2 个 epoch 重用同一批经历，给每个更新循环产生 8 份小批量。

### 14.3 哪些字段最后交给哪个网络

| 更新函数 | 读取的主要数据 | 用途 |
| --- | --- | --- |
| `check_value_update` | 旧观测、`reward_returns`、`cost_returns` | 用当前 critic 重新预测，再和固定目标比较 |
| `check_policy_update` | 旧观测、动作、对数密度、两类优势 | 给同一旧动作算新密度，构造奖励和约束损失 |

脚本另从整批的 `cost_returns[0]` 取环境均值，近似 reset 起点的累计 cost。`k`、`d` 从环境配置经 wrapper 交给策略更新。storage 在计算 GAE 或排列小批量时不需要这两个配置。[check 第 17 节](../../../../scripts/check_np3o_interface.py.md#17-六类-cost-怎样进入同一次策略更新)

## 完整源码

<details>
<summary>展开完整源码</summary>

```python
"""Rollout storage for TITA policy and value learning."""

from __future__ import annotations

from collections.abc import Iterator

import torch


class TitaRolloutStorage:
    """Store fixed-length batches of TITA environment transitions."""

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        num_obs: int = 586,
        num_actions: int = 8,
        num_costs: int = 6,
        device: str | torch.device = "cpu",
    ) -> None:
        """Allocate per-step observations, outcomes, and network predictions."""

        if min(num_envs, num_steps, num_obs, num_actions, num_costs) < 1:
            raise ValueError("Storage dimensions must be positive.")

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = torch.device(device)

        # 第0维是采集时间，第1维是机器人，第2维是对应特征。
        self.observations = torch.zeros(num_steps, num_envs, num_obs, device=self.device)
        self.actions = torch.zeros(num_steps, num_envs, num_actions, device=self.device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.costs = torch.zeros(num_steps, num_envs, num_costs, device=self.device)

        # dones表示回合结束；time_outs另外保留是否因时间上限结束。
        self.dones = torch.zeros(num_steps, num_envs, 1, dtype=torch.bool, device=self.device)
        self.time_outs = torch.zeros_like(self.dones)

        # 都是采集动作时的旧网络结果，以后更新网络时仍须保留这些数值。
        self.reward_values = torch.zeros_like(self.rewards)
        self.cost_values = torch.zeros_like(self.costs)
        self.log_probs = torch.zeros_like(self.rewards)
        self.action_mean = torch.zeros_like(self.actions)
        self.action_std = torch.zeros_like(self.actions)

        # 纯超时位置保存reset前最终状态的价值；其他位置为0。
        self.timeout_reward_values = torch.zeros_like(self.reward_values)
        self.timeout_cost_values = torch.zeros_like(self.cost_values)

        # 回报目标供critic拟合；优势供后续策略/约束更新使用。
        self.reward_returns = torch.zeros_like(self.reward_values)
        self.cost_returns = torch.zeros_like(self.cost_values)
        self.reward_advantages = torch.zeros_like(self.reward_values)
        self.cost_advantages = torch.zeros_like(self.cost_values)

        # 已保存的策略步数，同时也是下一次写入的时间索引。
        self.step = 0

        # 当前这批的回报目标与优势是否已经计算完成。
        self._returns_ready = False

    @property
    def is_full(self) -> bool:
        """Return whether the configured rollout has been collected."""
        return self.step == self.num_steps

    def add(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        time_outs: torch.Tensor,
        reward_values: torch.Tensor,
        cost_values: torch.Tensor,
        log_probs: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        timeout_reward_values: torch.Tensor,
        timeout_cost_values: torch.Tensor,
    ) -> None:
        """Copy one parallel environment step into persistent storage."""

        if self.step >= self.num_steps:
            raise RuntimeError("Rollout storage is full.")

        # 每项依次为：名称、保存整段数据的张量、当前这一步的数据。
        # 标量结果由(N,)增加一列成为(N,1)，与存储的一步切片一致。
        entries = (
            ("observations", self.observations, observations),
            ("actions", self.actions, actions),
            ("rewards", self.rewards, rewards.unsqueeze(-1)),
            ("costs", self.costs, costs),
            ("dones", self.dones, dones.unsqueeze(-1)),
            ("time_outs", self.time_outs, time_outs.unsqueeze(-1)),
            ("reward_values", self.reward_values, reward_values),
            ("cost_values", self.cost_values, cost_values),
            ("log_probs", self.log_probs, log_probs.unsqueeze(-1)),
            ("action_mean", self.action_mean, action_mean),
            ("action_std", self.action_std, action_std),
            ("timeout_reward_values", self.timeout_reward_values, timeout_reward_values),
            ("timeout_cost_values", self.timeout_cost_values, timeout_cost_values),
        )

        # 先检查所有形状，避免copy_的广播掩盖时间或环境维度错误。
        for name, destination, source in entries:
            expected = destination[self.step].shape
            if source.shape != expected:
                raise ValueError(f"Expected '{name}' with shape {tuple(expected)}, got {tuple(source.shape)}.")

        for _, destination, source in entries:
            # 保存独立数值，不保留采样时的梯度计算图。
            destination[self.step].copy_(source.detach())

        self.step += 1

    @torch.no_grad()
    def compute_returns(
        self,
        last_reward_values: torch.Tensor,
        last_cost_values: torch.Tensor,
        *,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        """Compute GAE advantages and value targets for a full rollout."""

        if not self.is_full:
            raise RuntimeError("Collect a full rollout before computing returns.")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
            raise ValueError("gamma and lam must be in [0, 1].")
        if last_reward_values.shape != self.reward_values[-1].shape:
            raise ValueError("last_reward_values must have shape (N, 1).")
        if last_cost_values.shape != self.cost_values[-1].shape:
            raise ValueError("last_cost_values must have shape (N, C).")

        # 从采集末尾倒推；初始时，本批之外没有继续累加的TD误差。
        reward_advantage = torch.zeros_like(last_reward_values)
        cost_advantage = torch.zeros_like(last_cost_values)

        self._returns_ready = False

        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_reward_values = last_reward_values
                next_cost_values = last_cost_values
            else:
                next_reward_values = self.reward_values[step + 1]
                next_cost_values = self.cost_values[step + 1]

            # (N,1)：回合未结束为1；失败或超时reset时为0。
            continuation = (~self.dones[step]).to(dtype=self.rewards.dtype)
            time_out = self.time_outs[step].to(dtype=self.rewards.dtype)

            # 普通步使用下一状态价值；纯超时使用reset前最终状态价值。
            # 失败时两项均为0；timeout_*在所有非纯超时位置也为0。
            reward_bootstrap = continuation * next_reward_values + time_out * self.timeout_reward_values[step]
            cost_bootstrap = continuation * next_cost_values + time_out * self.timeout_cost_values[step]

            # δ_t = 当步结果 + γ * 后续价值 - 当前旧价值。
            reward_delta = self.rewards[step] + gamma * reward_bootstrap - self.reward_values[step]
            cost_delta = self.costs[step] + gamma * cost_bootstrap - self.cost_values[step]

            # A_t = δ_t + γλ * A_{t+1}；reset处停止跨步递推。
            reward_advantage = reward_delta + gamma * lam * continuation * reward_advantage
            cost_advantage = cost_delta + gamma * lam * continuation * cost_advantage

            self.reward_advantages[step].copy_(reward_advantage)
            self.cost_advantages[step].copy_(cost_advantage)

            # 供critic拟合的目标：旧价值 + 优势估计。
            self.reward_returns[step].copy_(self.reward_values[step] + reward_advantage)
            self.cost_returns[step].copy_(self.cost_values[step] + cost_advantage)

        self._returns_ready = True

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_learning_epochs: int = 1,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield aligned mini-batches from a rollout with computed targets."""

        if not self.is_full or not self._returns_ready:
            raise RuntimeError("Collect a full rollout and compute returns first.")
        if num_mini_batches < 1 or num_learning_epochs < 1:
            raise ValueError("Mini-batch and epoch counts must be positive.")

        # 一批中的记录总数：策略步数T × 并行环境数N。
        batch_size = self.num_steps * self.num_envs
        if batch_size % num_mini_batches != 0:
            raise ValueError("Rollout size must be divisible by num_mini_batches.")
        mini_batch_size = batch_size // num_mini_batches

        # 每个字段的第(t,n)项都属于同一条经历，之后必须一起选行。
        fields = {
            "observations": self.observations,
            "actions": self.actions,
            "reward_values": self.reward_values,
            "cost_values": self.cost_values,
            "log_probs": self.log_probs,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "reward_returns": self.reward_returns,
            "cost_returns": self.cost_returns,
            "reward_advantages": self.reward_advantages,
            "cost_advantages": self.cost_advantages,
        }
        # (T,N,D) → (T*N,D)；586维观测中的10帧历史仍保留原来的顺序。
        flat_fields = {name: values.flatten(0, 1) for name, values in fields.items()}

        for _ in range(num_learning_epochs):
            # 每轮重新排列全部记录的编号，每条记录恰好出现一次。
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                batch_indices = indices[start : start + mini_batch_size]
                # 同一组编号用于全部字段，yield每次交出一份小批量数据。
                yield {name: values[batch_indices] for name, values in flat_fields.items()}

    def clear(self) -> None:
        """Mark all stored transitions as consumed."""
        # 下次从第0步覆盖；只有[0:self.step]是当前有效数据。
        self.step = 0
        self._returns_ready = False
```

</details>
