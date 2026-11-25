# policy.py：从八个动作均值到一次随机动作

内容由 AI 整理。2026-11-25 对照源码重写。

被解释文件：[Tita_Lab/src/Tita_Lab/learning/np3o/policy.py](../../../../../../Tita_Lab/src/Tita_Lab/learning/np3o/policy.py)。

第一遍读第 1 节，再看第 4 节怎样给旧动作算密度。标准差的梯度和熵可以在读策略损失时再回来看。

## 1. actor 的输出和机器人执行的动作为什么不同

假设现在只看一个关节。actor 根据观测算出动作均值 0.5。策略可以直接让关节每次执行 0.5，但采集阶段需要尝试附近的动作，才能看到哪种动作带来的结果更好。于是代码给动作增加随机变化。某次可能抽到 0.4，另一次可能抽到 0.8。

`TitaGaussianPolicy` 做三件事。它先让 actor 算动作均值，再用标准差决定动作通常离均值多远，最后根据得到的分布抽样或给一个已有的动作打分。训练中的“打分”是动作密度。环境收到的是抽出来的具体动作。

```python
def act(self, obs: torch.Tensor) -> torch.Tensor:
    """Sample actions for environment interaction."""
    self.update_distribution(obs)
    # a = mean + std * epsilon，epsilon的各分量独立服从N(0, 1)。
    return self._get_distribution().sample()  # (N, 8)。

def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
    """Return action means without sampling."""
    return self(obs)
```

`act` 先更新分布，再用 `sample()` 抽一个动作。`act_inference` 直接调用前向，返回均值。同一份观测、同一组参数，两条路径都会得到同一个均值，采样路径还会抽一组随机数。

## 2. 均值、标准差和密度各表示什么

```python
def __init__(self, init_noise_std: float = 1.0) -> None:
    """Initialize the action network and trainable standard deviations."""
    super().__init__()

    if init_noise_std <= 0.0:
        raise ValueError("init_noise_std must be positive.")

    self.actor = TitaHistoryActor()
    # 每个动作维度一个可学习标准差，shape=(8,)，所有环境共享。
    # 沿用旧TITA直接学习std的参数化方式。
    self.std = nn.Parameter(init_noise_std * torch.ones(8))
    # 保存最近一次根据观测建立的分布；创建对象时还没有对应观测。
    self._distribution: Normal | None = None
```

均值 $\mu$ 决定动作建议的中心，标准差 $\sigma$ 决定随机变化的尺度。代码为每个动作分量建立一个标准差，共 8 个。`nn.Parameter` 把这个张量加入模块的可学习参数。初值为 1 时，采集的随机变化从这个尺度开始。

用一维动作说明抽样。程序先从标准正态分布抽一个数 $\epsilon$，再把它乘以 $\sigma$ 并平移到 $\mu$ 附近。

$$
a=\mu+\sigma\epsilon,\qquad \epsilon\sim\mathcal N(0,1).
$$

这里 $a$ 是抽到的数。例如 $\mu=0.5$、$\sigma=0.2$、这次 $\epsilon=-0.5$，动作就等于 $0.5+0.2\times(-0.5)=0.4$。

密度回答动作附近的概率有多集中。连续变量恰好等于一个指定实数的概率为零，所以程序用“附近密度”描述分布，而一段区间的概率由区间上的面积给出。例如均值不变，标准差变小，中心附近更集中，离中心较远的动作就较少出现。密度本身可以大于 1，整个分布的面积仍等于 1。

## 3. 一份 586 列的观测怎样变成 8 个分布

```python
def forward(self, obs: torch.Tensor) -> torch.Tensor:
    """Return deterministic action means from packed observations."""
    if obs.ndim != 2 or obs.shape[1] != 586:
        raise ValueError(f"Expected observations with shape (N, 586), got {tuple(obs.shape)}.")

    # 586列中，前33列是当前观测，最后330列是过去十帧。
    proprio = obs[:, :33]
    history = obs[:, -330:].reshape(obs.shape[0], 10, 33)
    return self.actor(proprio, history)  # (N, 8)，动作均值。
```

`forward` 先确认输入是二维，且每行 586 个数。随后 `obs[:, :33]` 取所有行的前 33 列。`obs[:, -330:]` 取最后 330 列，再用 `reshape(N, 10, 33)` 还原每台机器人过去十帧的顺序。actor 用这两部分为每台机器人算出 8 个均值。[actor.py 说明](actor.py.md)

```python
def update_distribution(self, obs: torch.Tensor) -> None:
    """Build the Gaussian distribution for the given observations."""
    mean = self(obs)  # (N, 8)，调用本类forward。
    # (8,) -> (N, 8)：各环境使用同一组std，但各自具有不同的均值。
    scale = self.std.expand_as(mean)
    self._distribution = Normal(loc=mean, scale=scale, validate_args=True)
```

`self(obs)` 调用当前 `TitaGaussianPolicy` 对象。这个类继承 `nn.Module`，所以 PyTorch 的调用入口会执行本类的 `forward`。这样代码先拿到 `(N,8)` 的均值，再用 `expand_as(mean)` 将 `(8,)` 的标准差按行扩展成可对应的 `(N,8)`。扩展后，每行仍使用同一组标准差参数。

`Normal(loc=mean, scale=scale)` 给每一行的 8 个分量分别建立一维正态分布。`loc` 是均值，`scale` 是标准差。以 4 个环境为例，PyTorch 看到的是一张 `(4,8)` 的标量分布表。给定观测时，代码按独立分量抽样。`validate_args=True` 会检查标准差等参数的取值。

`_distribution` 是最近一次建立的分布。读分布之前需要先调用 `act` 或 `update_distribution`。`_get_distribution` 会在缓存为空时抛出错误。换了观测或更新了网络参数以后，调用者要重新建立分布。check 用 storage 单独保存采集时的密度、均值和标准差。

## 4. 为什么给 8 维动作算密度时要加 8 个数

```python
def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    """Return joint action log densities under the current distribution."""
    # 给定观测时，8个动作分量独立；联合密度的对数是8项之和。
    return self._get_distribution().log_prob(actions).sum(dim=-1)  # (N,)。
```

给定一份观测，一个动作包括全部 8 个分量。例如左轮和右轮的两个数要作为同一次动作来评价。代码假定各个动作分量在给定观测时独立，因此整组动作的联合密度是各分量密度的乘积。

设 $a_j$ 为第 $j$ 个动作分量，$p_j(a_j)$ 为对应的一维密度。代码保存对数，所以把 8 个密度的乘积改写成 8 个对数密度的和。

$$
\log p(a_1,\ldots,a_8)=\sum_{j=1}^8\log p_j(a_j).
$$

比如一个两维教学动作的两项对数密度分别是 $-1.2$ 和 $-0.8$。整组动作的对数密度就是 $-2.0$。代码中的 `.sum(dim=-1)` 加的是动作分量轴，因此 `(N,8)` 变成 `(N,)`，每台机器人保留一个结果。storage 保存时再补一列，得到 `(N,1)`。

采集时，`act` 先抽动作，然后 `get_actions_log_prob` 对这个动作算旧密度。更新时，check 用旧观测执行 `update_distribution`，再把**同一个旧动作**传给 `get_actions_log_prob`。这样新旧密度比较的对象完全相同。[check 的密度比与裁剪](../../../../scripts/check_np3o_interface.py.md#13-check_policy_update策略怎样学会调整动作分布)

## 5. 动作已经抽完了，标准差怎样学

抽一个动作时，当前 check 只需要数值。学习时，程序固定这条记录中的动作，再问“如果我把当前分布的均值或标准差改一点，这个旧动作的密度会怎样变？”对这个问题可以求导，所以均值网络和标准差都有学习信号。

以一个分量为例。动作 $a$ 取固定值，$\mu$ 和 $\sigma>0$ 来自当前策略。对数密度包含两部分。一部分来自分布高度的变化，一部分来自动作离中心的距离。

$$
\log p(a)
=-\log\sigma-\frac12\log(2\pi)-\frac{(a-\mu)^2}{2\sigma^2}.
$$

看均值的影响时，动作如果位于均值右边，把均值朝右移一点通常会提高密度。看标准差时，若固定动作恰好等于均值，增大标准差会降低中心的密度。若固定动作离均值很远，适度增大标准差反而可能提高密度。这就是为什么标准差可以通过当前策略损失学习。

对 $\sigma$ 求导可得 $-1/\sigma+(a-\mu)^2/\sigma^3$。这一项只说明单个动作的密度如何随 $\sigma$ 变，整批损失还要考虑优势、裁剪和约束。优化器拿到 `policy.parameters()` 后才会调整 `std`。因为当前实现直接学习标准差，脚本每次更新后检查 `std` 仍为有限正数。[PyTorch Normal](https://docs.pytorch.org/docs/2.11/distributions.html#normal)

## 6. mean、std 和 entropy 为什么还要单独读

```python
@property
def action_mean(self) -> torch.Tensor:
    """Return the current action means."""
    return self._get_distribution().mean  # (N, 8)。

@property
def action_std(self) -> torch.Tensor:
    """Return the current action standard deviations."""
    return self._get_distribution().stddev  # (N, 8)。

@property
def entropy(self) -> torch.Tensor:
    """Return the summed action entropy for each environment."""
    return self._get_distribution().entropy().sum(dim=-1)  # (N,)。
```

`@property` 让调用方写 `policy.action_mean` 就能执行读取逻辑。`action_mean` 和 `action_std` 各返回 `(N,8)`，用来记录当前分布。一个原本形状 `(8,)` 的 `policy.std` 参数为所有环境共享，和扩展后的 `action_std` 含义不同。

熵衡量整个分布有多分散。对一个标准差为 $\sigma$ 的正态分量，微分熵是 $\frac12\log(2\pi e\sigma^2)$。这里 $e$ 是自然对数的底。8 个独立分量的熵相加，所以每个环境得到一个数。初值全为 1 时，8 维总熵约为 11.3515，与检查输出对应。当前 check 打印熵，策略损失由奖励和 cost 两项组成。

要执行一条可重复的策略时，可以用 `act_inference` 直接取均值。取均值是如何使用已学分布的选择。这个动作是不是最优，要看优化结果和任务约束。`eval()` 只设置网络模式，调用 `act` 依然会抽样。

## 原来问过的问题

<details>
<summary>展开原问题和对应解释</summary>

**Q：我没搞懂，为什么要可学习的标准差？？**

**A：** 标准差越大，策略采到的动作一般离均值越远。策略损失依赖旧动作在当前分布下的密度，所以对标准差也能求导。第 5 节用一个动作说明。

**Q：当策略成型，训练完毕，推理时不应该还是随机变量（也是rl理论教的），而是最优策略，最优策略就是确定性deterministic的，这里取均值？？**

**A：** 取均值会得到一种确定的执行方式。带期望 cost 上限时，随机选择也可能更好。比如动作 A 给奖励 0、cost 0，动作 B 给奖励 2、cost 1。上限若是 0.5，一半时候选 B 可得平均奖励 1，而符合上限的确定性选择只有 A。

**Q：我没搞懂self(obs)是什么鬼，咋还有这种语法。**

**A：** `self` 指当前 policy 对象。PyTorch 的 `nn.Module` 支持直接调用对象，因此 `self(obs)` 会执行本类的前向计算。第 3 节对应这句。

**Q：这块，讲一下，这样换是为啥，和之前有啥区别。**

**A：** 早期脚本让环境执行 actor 的均值。当前策略先从高斯分布采样，再把抽到的动作交给环境，并记录这个动作的密度。第 1、4 节分别说明执行和取密度。

</details>

## 完整源码

<details>
<summary>展开完整源码</summary>

```python
"""Gaussian action policy for TITA."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from .actor import TitaHistoryActor


class TitaGaussianPolicy(nn.Module):
    """Sample joint actions from a history-based Gaussian policy."""

    def __init__(self, init_noise_std: float = 1.0) -> None:
        """Initialize the action network and trainable standard deviations."""
        super().__init__()

        if init_noise_std <= 0.0:
            raise ValueError("init_noise_std must be positive.")

        self.actor = TitaHistoryActor()
        # 每个动作维度一个可学习标准差，shape=(8,)，所有环境共享。
        # 沿用旧TITA直接学习std的参数化方式。
        self.std = nn.Parameter(init_noise_std * torch.ones(8))
        # 保存最近一次根据观测建立的分布；创建对象时还没有对应观测。
        self._distribution: Normal | None = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic action means from packed observations."""
        if obs.ndim != 2 or obs.shape[1] != 586:
            raise ValueError(f"Expected observations with shape (N, 586), got {tuple(obs.shape)}.")

        # 586列中，前33列是当前观测，最后330列是过去十帧。
        proprio = obs[:, :33]
        history = obs[:, -330:].reshape(obs.shape[0], 10, 33)
        return self.actor(proprio, history)  # (N, 8)，动作均值。

    def update_distribution(self, obs: torch.Tensor) -> None:
        """Build the Gaussian distribution for the given observations."""
        mean = self(obs)  # (N, 8)，调用本类forward。
        # (8,) -> (N, 8)：各环境使用同一组std，但各自具有不同的均值。
        scale = self.std.expand_as(mean)
        self._distribution = Normal(loc=mean, scale=scale, validate_args=True)

    def _get_distribution(self) -> Normal:
        """Return the most recently built action distribution."""
        if self._distribution is None:
            raise RuntimeError("Call update_distribution() or act() first.")
        return self._distribution

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample actions for environment interaction."""
        self.update_distribution(obs)
        # a = mean + std * epsilon，epsilon的各分量独立服从N(0, 1)。
        return self._get_distribution().sample()  # (N, 8)。

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action means without sampling."""
        return self(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Return joint action log densities under the current distribution."""
        # 给定观测时，8个动作分量独立；联合密度的对数是8项之和。
        return self._get_distribution().log_prob(actions).sum(dim=-1)  # (N,)。

    @property
    def action_mean(self) -> torch.Tensor:
        """Return the current action means."""
        return self._get_distribution().mean  # (N, 8)。

    @property
    def action_std(self) -> torch.Tensor:
        """Return the current action standard deviations."""
        return self._get_distribution().stddev  # (N, 8)。

    @property
    def entropy(self) -> torch.Tensor:
        """Return the summed action entropy for each environment."""
        return self._get_distribution().entropy().sum(dim=-1)  # (N,)。
```

</details>
