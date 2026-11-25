# critic.py：根据现在的观测估计接下来的累计结果

内容由 AI 整理。2026-11-25 对照源码重写。

被解释文件：[Tita_Lab/src/Tita_Lab/learning/np3o/critic.py](../../../../../../Tita_Lab/src/Tita_Lab/learning/np3o/critic.py)。

第一遍读第 1 节和第 6 节。先知道 critic 在预测什么，再看一行观测怎样汇合成两种输出。第 5 节单独讲时间卷积。

## 1. 为什么已经有奖励，还要让 critic 预测价值

环境能算出“机器人刚刚做得怎样”。例如机器人这一拍跟上了速度命令，环境就给出奖励。学习还想知道“从现在出发，按这个策略继续行动，后面大概能累计多少奖励和 cost”。未来的动作、接触和运动还没发生，critic 用当前能看到的信息给出估计。

输入给 critic 的每一行是一台机器人的观测。critic 每行输出 1 个奖励价值和 6 个 cost 价值。奖励价值估计折扣累计奖励。每个 cost 价值单独估计一种折扣累计 cost。例如，当前速度是否越限由环境判断，之后可能累计多少速度越限 cost 由 critic 预测。

采集时，check 把旧 critic 的预测留在 storage，供 GAE 修正。学习时，check 用同一个 critic 对旧观测重新预测，让它靠近 GAE 给的固定目标。两次传入的可以是同一行观测，预测的用途随阶段改变。

## 2. 586 个观测数分别来自哪里

```python
if obs.ndim != 2 or obs.shape[1] != 586:
    raise ValueError(
        f"Expected observations with shape (N, 586), got {tuple(obs.shape)}."
    )

batch_size = obs.shape[0]
proprio = obs[:, :33]  # (N,33)，当前观测。
priviliged = obs[:, 220:256]  # (N,36)，六维状态加30维动力学参数。
scan_features = self.scan_encoder(obs[:, 33:220])  # (N,32)。
```

环境在 critic 调用之前已经把观测拼好。$N$ 台机器人一起输入，张量形状是 `(N,586)`。每一行仍独立通过网络。下面的切片包含左端，右端不包含。

| 列区间 | 内容 | critic 怎样用 |
| --- | --- | --- |
| `0:33` | 当前观测，包括机身角速度、重力方向、速度命令、关节位置和速度、上一拍动作 | 33 列直接进入最后的拼接 |
| `33:220` | 地面高度扫描 | 187 列通过扫描网络变为 32 列 |
| `220:226` | 机身线速度、轮接触和控制延迟设置 | 和后面的动力学参数一起直接拼入 |
| `226:256` | 仿真随机化的动力学参数 | 同上 |
| `256:586` | 过去 10 帧，每帧 33 列 | 每帧先处理一遍，接着沿时间做卷积 |

`priviliged` 是源码里的局部变量名，指第 220 到 255 列这 36 个额外数。当前 33 列和扫描等字段已由环境处理。例如关节速度可能已经加噪声并缩放，critic 直接读这些数。[观测布局说明](../../tasks/velocity/observation_layout.py.md)

### 2.3 额外信息能帮 critic 知道什么

扫描点提供机器人周围的地面高度。环境用传感器参考高度减去地面交点高度，再减 0.4、裁到 `[-1,1]` 并乘 5。若参考高度为 0.6 m，某个地面交点是 0.1 m，输入就是 $5(0.6-0.4-0.1)=0.5$。

额外的六列是三维机身线速度乘 2、左右轮的接触标记各减 0.5、一个归一化的延迟编号。接触标记把当前拍和前一拍的接触判断做逻辑或。延迟编号 0、1、2 对应约 5、2.5、0 ms，环境用 0、0.5、1 表示。同样的动作在不同延迟下可能导致不同的后续结果，所以 critic 把这个设置也当作输入。

后面 30 列包括机身质量变化 1 列、质心偏移 3 列、摩擦和恢复系数各 1 列，以及 8 个关节的电机强度、Kp、Kd 倍率，共 $1+3+1+1+8+8+8=30$ 列。critic 可以用这些条件区分看起来相近的观测。actor 的动作路径读取当前观测与历史两部分。[观测计算](../../tasks/velocity/mdp/observations.py.md) · [动作延迟](../../tasks/velocity/mdp/actions.py.md)

## 3. 调用 critic 时，PyTorch 怎样用已有参数

调用方写 `value_networks(obs)`。`nn.Module` 的调用入口会执行本类的 `forward`。`__init__` 在创建对象时建立各层，`forward` 反复使用已有的权重。输入可以是采集时的 $N$ 行，也可以是更新时的 $B$ 条记录。第一轴只表示这次一起算多少行。

假设线性层的一行输入为 $[2,3]$，一个输出的权重为 $[0.5,-1]$，偏置为 $0.1$。输出为 $0.5\times2-1\times3+0.1=-1.9$。换一行输入，层仍使用相同权重。训练会用预测误差调整这些权重。

## 4. 扫描网络为什么把 187 列变成 32 列

```python
self.scan_encoder = nn.Sequential(
    nn.Linear(187, 128),
    nn.ELU(),
    nn.Linear(128, 64),
    nn.ELU(),
    nn.Linear(64, 32),
    nn.ELU(),
    nn.ELU()
)
```

三个线性层把每台机器人的 187 个扫描数变成 128、64，再变成 32 列。ELU 是逐元素的非线性函数，正输入直接通过，负输入平滑压缩。非线性让多层网络能表达比一次线性加权更复杂的关系。源码末尾连续两次 ELU，沿用了原网络的计算。

输出的 32 列是可学习的特征。哪些组合能帮助预测未来摔倒或速度跟踪，由价值损失在训练中决定。

## 5. 历史网络怎样沿时间看过去十帧

### 5.1 十帧先各自通过同一个线性层

```python
history = obs[:, 256:].reshape(batch_size, 10, 33)
frame_features = self.history_frame_encoder(
    history.reshape(batch_size * 10, 33)
)  # (N*10,30)。

# Conv1d要求顺序为：批量、通道、序列长度。
temporal_input = frame_features.reshape(batch_size, 10, 30).transpose(1, 2)
temporal_features = self.history_temporal_encoder(temporal_input)  # (N,30)。
history_features = self.history_output(temporal_features)  # (N,32)。
```

`reshape(batch_size, 10, 33)` 把每行末尾的 330 个数还原成十帧。网络把环境轴和帧轴合起来，得到 `(N*10,33)`。4 个环境共有 40 行。`history_frame_encoder` 对每行使用同一套 `Linear(33, 30)` 和 ELU。于是输出是 `(40,30)`，每行仍对应原来的一帧。

`reshape(batch_size,10,30)` 恢复帧轴。`transpose(1,2)` 交换帧轴和特征轴，把数据排成 `(N,30,10)`。`Conv1d` 约定最后一轴是滑动方向，所以代码把十帧放在最后。30 个通道指每帧的 30 个特征。

### 5.2 先用一条序列看卷积

假设一个特征在五帧中为 $[1,3,4,8,9]$。卷积窗口有两个权重 $[1,-1]$，每次前进两格。第一个窗口读取 $[1,3]$，结果为 $1-3=-2$。第二个窗口读取 $[4,8]$，结果为 $4-8=-4$。最后只剩一格，放不下完整窗口。相同权重在不同位置都能度量这个特征的变化。

如果一帧有 30 个特征，每个输出通道会给窗口内的 30 个特征和所有时间位置分别赋权，把结果加起来再加偏置。不同输出通道各有权重，因此能学习不同的时间变化。同一台机器人的时间轴上会依次应用相同权重。

### 5.3 两层卷积怎样把长度从 10 变成 3

```python
self.history_temporal_encoder = nn.Sequential(
    nn.Conv1d(30, 20, kernel_size=4, stride=2),
    nn.ELU(),
    nn.Conv1d(20, 10, kernel_size=2, stride=1),
    nn.ELU(),
    nn.Flatten(start_dim=1),  # (N,10,3) -> (N,30)。
)
self.history_output = nn.Sequential(
    nn.Linear(30, 32),
    nn.ELU(),
)
```

$L$ 是输入长度，$K$ 是窗口宽度，$S$ 是每次移动几格。当前卷积从 0 开始，窗口的最晚起点为 $L-K$。因此能放下的窗口数是 $\lfloor(L-K)/S\rfloor+1$。

| 操作 | 输入 | 输出及原因 |
| --- | --- | --- |
| 第一层 `Conv1d(30,20,kernel_size=4,stride=2)` | `(N,30,10)` | `(N,20,4)`，起点是 0、2、4、6 |
| 第二层 `Conv1d(20,10,kernel_size=2,stride=1)` | `(N,20,4)` | `(N,10,3)`，起点是上一层的 0、1、2 |
| `Flatten(start_dim=1)` | `(N,10,3)` | `(N,30)`，每台机器人的通道和时间合为一行 |
| `history_output` | `(N,30)` | `(N,32)`，`Linear(30,32)` 接 ELU |

critic 的历史支路使用原来的十帧，当前 33 个数稍后直接拼入。actor 的动作前向会用当前帧替换最老的历史帧，两者使用不同的窗口。[PyTorch Conv1d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Conv1d.html)

## 6. 四块信息怎样变成两种价值

```python
features = torch.cat(
    (proprio, priviliged, scan_features, history_features), dim=-1,
)  # (N,133)。

reward_values = self.reward_head(features)  # (N,1)，未来累计奖励估计。
cost_values = self.cost_head(features)  # (N,6)，六类未来累计cost估计。
return reward_values, cost_values
```

`torch.cat(..., dim=-1)` 沿列拼接当前观测 33 列、额外信息 36 列、扫描特征 32 列、历史特征 32 列。每行共有 $33+36+32+32=133$ 个数。奖励输出网络 `reward_head` 和 cost 输出网络 `cost_head` 都读取同一行的 133 个数。

```python
self.reward_head = _build_mlp(133, 1, (512, 256, 128))
self.cost_head = nn.Sequential(
    _build_mlp(133, 6, (512, 256, 128)),
    nn.Softplus(),  # 六类cost回报非负，保持旧网络的输出约束。
)
```

两条输出支路各有参数。奖励支路输出一列，允许为负。cost 支路输出六列，最后用 Softplus 约束到非负范围。若 $z$ 表示某列进入 Softplus 的值，默认转换的数学形式为 $\log(1+e^z)$。$z=0$ 时输出约 0.693，$z$ 很负时接近 0。这样，价值预测符合非负 cost 的取值范围。

`return reward_values, cost_values` 返回一个包含两个张量的元组。4 个环境得到 `(4,1)` 和 `(4,6)`。cost 列与环境配置同序。[costs.py 六列含义](../../tasks/velocity/mdp/costs.py.md#3-六列分别是什么按哪个顺序返回)

## 7. 预测误差怎样传回扫描和历史网络

### 7.1 两份误差怎样训练共享层

设一次小批量有 $B$ 条记录。对第 $i$ 条，critic 从观测算出当前奖励价值 $V^r_i$，storage 提供固定的奖励回报目标 $\widehat R_i$。若两者差得更远，损失就更大。奖励和 cost 分别用均方误差。

$$
L_r=\frac{1}{B}\sum_i(V_i^r-\widehat R_i)^2,\qquad
L_c=\frac{1}{6B}\sum_{i,j}(V_{i,j}^c-\widehat C_{i,j})^2.
$$

$j$ 指第几类 cost，$V^c_{i,j}$ 是当前预测，$\widehat C_{i,j}$ 是固定回报目标。训练端将 $L_r$ 和 $L_c$ 相加。两个输出都经过同一个扫描编码器和历史编码器，所以共享权重会收到两份误差的梯度贡献。各输出支路的末端权重由各自的误差训练。

这也解释了中间的 32 个扫描特征为什么可以学习。最终预测依赖中间层。PyTorch 用链式法则把误差传到能影响结果的参数，优化器再修改参数。各层一起为最终预测服务。[check 的手算与自动求导](../../../../scripts/check_np3o_interface.py.md#12-check_value_update从误差到自动求导再到参数更新)

### 7.2 本次检查已经验证到哪里

11 月 14 日的 check 对同一批 GAE 目标重新测量，发现 8 次更新后奖励和 cost 的整批 MSE 都下降。结果说明网络能按这些目标调整预测。要了解后续预测是否更准，需要用新经历比较。[单批拟合证据](../../../../../../docs/audits/2026-11-14_constrained_policy_update/README.md)

## 完整源码

<details>
<summary>展开完整源码</summary>

```python
"""Reward and cost value networks for TITA."""

from __future__ import annotations

import torch
from torch import nn

from .actor import _build_mlp


class TitaValueNetworks(nn.Module):
    """Estimate reward and cost returns from packed observations."""

    def __init__(self):
        """Initialize observation encoders and value heads."""

        super().__init__()

        # 扫描：187 -> 128 -> 64 -> 32。
        # 旧mlp_factory末层已有ELU，last_act=True又追加一次；保持原计算。
        self.scan_encoder = nn.Sequential(
            nn.Linear(187, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.ELU()
        )

        # 每帧33个数映射为30个特征；十帧共用同一套参数。
        self.history_frame_encoder = nn.Sequential(
            nn.Linear(33, 30),
            nn.ELU(),
        )

        # 沿时间轴处理历史：(N,30,10) -> (N,20,4) -> (N,10,3)。
        self.history_temporal_encoder = nn.Sequential(
            nn.Conv1d(30, 20, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv1d(20, 10, kernel_size=2, stride=1),
            nn.ELU(),
            nn.Flatten(start_dim=1),  # (N,10,3) -> (N,30)。
        )
        self.history_output = nn.Sequential(
            nn.Linear(30, 32),
            nn.ELU(),
        )

        # 当前33 + 仿真状态/参数36 + 扫描编码32 + 历史编码32 = 133。
        # 两个输出网络各自有参数，共享上面的观测编码器。
        self.reward_head = _build_mlp(133, 1, (512, 256, 128))
        self.cost_head = nn.Sequential(
            _build_mlp(133, 6, (512, 256, 128)),
            nn.Softplus(),  # 六类cost回报非负，保持旧网络的输出约束。
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reward and cost value estimates for each environment."""

        if obs.ndim != 2 or obs.shape[1] != 586:
            raise ValueError(
                f"Expected observations with shape (N, 586), got {tuple(obs.shape)}."
            )

        batch_size = obs.shape[0]
        proprio = obs[:, :33]  # (N,33)，当前观测。
        priviliged = obs[:, 220:256]  # (N,36)，六维状态加30维动力学参数。
        scan_features = self.scan_encoder(obs[:, 33:220])  # (N,32)。

        # 价值网络使用原样过去十帧，不追加当前帧，沿用旧critic。
        history = obs[:, 256:].reshape(batch_size, 10, 33)
        frame_features = self.history_frame_encoder(
            history.reshape(batch_size * 10, 33)
        )  # (N*10,30)。

        # Conv1d要求顺序为：批量、通道、序列长度。
        temporal_input = frame_features.reshape(batch_size, 10, 30).transpose(1, 2)
        temporal_features = self.history_temporal_encoder(temporal_input)  # (N,30)。
        history_features = self.history_output(temporal_features)  # (N,32)。

        # 拼接顺序与旧evaluate/evaluate_cost一致。
        features = torch.cat(
            (proprio, priviliged, scan_features, history_features), dim=-1,
        )  # (N,133)。

        reward_values = self.reward_head(features)  # (N,1)，未来累计奖励估计。
        cost_values = self.cost_head(features)  # (N,6)，六类未来累计cost估计。
        return reward_values, cost_values
```

</details>
