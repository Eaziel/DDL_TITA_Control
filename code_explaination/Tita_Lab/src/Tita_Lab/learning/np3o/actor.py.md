# actor.py：TITA 怎样用最近的运动决定动作均值

内容由 AI 整理。2026-11-25 对照源码重写。

被解释文件：[Tita_Lab/src/Tita_Lab/learning/np3o/actor.py](../../../../../../Tita_Lab/src/Tita_Lab/learning/np3o/actor.py)。

第一遍读第 1 节的动作计算。第 4、5 节的辅助学习可以等你需要理解 `auxiliary_loss` 时再看。

## 1. 为什么给 actor 看过去的观测

假设两台机器人这一拍看起来姿态差不多。一台正在往前倾，另一台刚开始往回摆。如果网络再看前几拍，就有更多依据判断运动趋势，并决定接下来的动作。过去的响应也提供线索，帮助网络适应延迟和动力学条件。

`TitaHistoryActor` 接收当前 33 个观测和过去十帧。历史编码器把一段变化压成 26 个特征，再把这些特征与当前的 33 个观测拼起来。动作网络根据两部分信息给出 8 个动作均值。随后 `policy.py` 用这些均值建立高斯分布。

```python
def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
    """Return the action means for a batch of observations."""

    # obs为(N,33)，obs_hist为(N,10,33)，后者不含当前帧。
    # 丢掉最老一帧，补入当前帧；得到最近九帧历史加当前帧。
    history_with_current = torch.cat(
        (obs_hist[:, 1:, :], obs.unsqueeze(1)), dim=1
    )
    batch_size = obs.shape[0]
    history_flat = history_with_current.reshape(batch_size, -1) # (N,330)。
    features = self.mlp_encoder(history_flat) # (N,26)。

    # 顺序沿用旧网络：26维编码在前，33维当前观测在后。
    actor_input = torch.cat((features, obs), dim=-1) ## (N,59)。
    return self.actor(actor_input) # (N,8)，动作分布的均值。
```

假设在第 $t$ 拍调用。`obs_hist` 装着从 $t-10$ 到 $t-1$ 的十帧，`obs` 是第 $t$ 拍。`obs_hist[:,1:,:]` 取所有样本、从第二帧开始的九帧和每帧全部特征。`obs.unsqueeze(1)` 把 `(N,33)` 变为 `(N,1,33)`。`cat(..., dim=1)` 把当前帧接到九帧后面，所以窗口改成 $t-9$ 到 $t$，还是十帧。

`reshape(batch_size,-1)` 留住 $N$ 行，让 PyTorch 算出每行的剩余长度 $10\times33=330$。历史编码器按每台机器人处理 330 个数，给出 `(N,26)`。下一次 `cat` 沿特征轴把 26 列和当前 33 列拼成 `(N,59)`。动作网络最后给每台机器人输出一行 8 个数。直接传入当前观测，能让动作输出使用那些不必经历史编码的当前信息。[policy.py 怎样抽动作](policy.py.md)

## 2. _build_mlp 怎样搭一串层

```python
def _build_mlp(
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        layer_norm: bool = False,
) -> nn.Sequential:
    """Build an ELU multilayer perceptron with a linear output."""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for index, hidden_dim in enumerate(hidden_dims):
        # 每层先做线性变换 Wx+b，再经过ELU非线性变换。
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        # 沿用旧工厂的准确位置：第一隐层不加LayerNorm。
        if layer_norm and index > 0:
            layers.append(nn.LayerNorm(hidden_dim))
        previous_dim = hidden_dim

    # 输出层不加激活函数，动作均值不在这里裁剪。
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)
```

`nn.Linear(previous_dim, hidden_dim)` 创建一层，并由 PyTorch 初始化权重和偏置。假设层要把一行的 2 个输入变成 3 个输出，它会创建 `(3,2)` 的权重矩阵和 3 个偏置。每个输出都是把两个输入按各自权重相加，再加偏置。比如一组权重是 `[0.5,-1]`，偏置是 0.1，输入为 `[2,3]`，这个输出就是 $0.5\times2-1\times3+0.1=-1.9$。所有行共用同一组权重。

`nn.ELU()` 再按元素变换线性结果。以默认参数为例，正输入直接通过，负输入按 $e^z-1$ 平滑压缩，$z$ 指这个元素进入 ELU 时的值。如果整条链都只有线性变换，多层仍能合成一次线性变换。非线性给网络增加了表达能力。

`enumerate(hidden_dims)` 同时提供隐层编号和该层宽度。`layers.extend(...)` 将 Linear 和 ELU 两个对象分别加进列表。`layers.append(...)` 只加一个对象。每轮最后用 `previous_dim = hidden_dim` 记录宽度，供下一层决定输入列数。列表里此时放的是层对象。真正的输入要等网络前向才传进来。

如果调用方选了 `layer_norm=True`，代码从第二个隐层开始，在 ELU 后放一个 LayerNorm。对当前二维数据，LayerNorm 分别在每条样本自己的特征列上算均值和方差，并用可学习参数缩放和平移。比如一行是 `[2,4]`，减去行均值 3 后变成 `[-1,1]`。除以标准差后再交给下一层。第 5 节的 BatchNorm 对同一列的多条样本计算统计，两者看的方向不同。

循环完后追加一个 Linear 作为输出层，所以输出仍可取正负值。`nn.Sequential(*layers)` 把列表中的层当作多个位置参数传给 Sequential，按列表顺序连接。签名里的独立 `*` 有另一个用途，它要求随后出现的 `layer_norm` 按名字传，例如 `layer_norm=True`。

## 3. 对象创建时得到哪些可学习的网络

```python
self.mlp_encoder = _build_mlp(
    num_proprio * history_length,
    encoded_dim,
    (512, 256, 128),
    layer_norm=True,
)
# 默认维度：26+33=59 -> 512 -> 256 -> 128 -> 8。
self.actor = _build_mlp(
    encoded_dim + num_proprio,
    num_actions,
    (512, 256, 128),
)

# 当前33维观测的另一条编码支路，只用于辅助学习损失。
self.obs_encoder = _build_mlp(
    num_proprio,
    self.latent_dim,
    (256, 128),
    layer_norm=True,
)
# 沿批量方向分别标准化16个特征列；不学习额外缩放和偏移。
self.bn = nn.BatchNorm1d(self.latent_dim, affine=False)
```

| 对象成员 | 默认宽度 | 负责什么 |
| --- | --- | --- |
| `mlp_encoder` | 330 → 512 → 256 → 128 → 26 | 读取十帧，形成历史特征 |
| `actor` | 59 → 512 → 256 → 128 → 8 | 用历史特征和当前观测算动作均值 |
| `obs_encoder` | 33 → 256 → 128 → 16 | 为辅助损失提供当前观测的特征 |

`TitaHistoryActor` 继承 `nn.Module`。`super().__init__()` 先初始化 PyTorch 用来管理子模块和参数的结构。把一层模块赋给 `self.xxx` 时，PyTorch 会登记它，之后 `parameters()` 才能找到这些层的参数。`actor(obs, obs_hist)` 会通过 `nn.Module` 的调用入口执行 `forward`。前向的意思就是把输入按定义好的层一步步计算到输出。

26 列在动作支路中全部使用。辅助训练另外把它们分成前 10 列与后 16 列，为历史编码器提供两个具体的学习目标。当前策略更新会经过历史编码器和动作网络。`auxiliary_loss` 已经写在类里，当前 check 的损失还没有调用这段辅助计算。因而不能靠“前 10 列”这个命名就判断它已经学会预测对应的状态。

## 4. 辅助学习为什么再看一次历史

```python
batch_size = obs.shape[0]
# obs=(N,33)，obs_hist=(N,10,33)，targets=(N,10)。
# 沿用旧损失：直接使用过去十帧，不追加当前帧。
predicted = self.mlp_encoder(obs_hist.reshape(batch_size, -1))
state_prediction = predicted[:, :self.prediction_dim] # (N,10)。
history_features = predicted[:, self.prediction_dim:] # (N,16)。
current_features = self.obs_encoder(obs)
```

如果只用奖励和 cost 训练，历史编码器从动作结果间接获得信号。辅助学习另外利用仿真能直接提供的一些量，帮助编码器从过去的运动推测当前状态。这样，编码器可以为动作选择保留有用的信息。

这次输入的是**原来的过去十帧**，即 $t-10$ 到 $t-1$。程序用 `mlp_encoder` 得到 26 列，前 10 列拿来预测当前的目标。后 16 列作为历史特征。另一个 `obs_encoder` 读当前 33 列，算出当前观测的 16 个特征。两条支路使用同一批样本的行，所以第 $i$ 行总是同一条经历。

| 量 | 形状 | 一行的含义 |
| --- | --- | --- |
| `obs` | `(B,33)` | 当前观测 |
| `obs_hist` | `(B,10,33)` | 这条观测之前的十帧 |
| `state_prediction`、`targets` | 各为 `(B,10)` | 历史推测的 10 个值与仿真提供的对应目标 |
| `history_features`、`current_features` | 各为 `(B,16)` | 从历史和从当前观测分别算出的特征 |

当前观测布局能从打包观测的 `220:230` 取这十项目标。前六列是线速度、车轮接触、延迟编号，后四列是质量变化和三维质心偏移。函数从调用方收 `targets`。反过来查看动作前向时，实际送给动作网络的是已有观测和历史，不需要调用者提供这些十项目标。[辅助损失迁移记录](../../../../../../docs/migration/24_actor辅助学习损失.md)

## 5. BatchNorm 和 Barlow Twins 怎样约束这 16 个特征

```python
history_norm = self.bn(history_features)
current_norm = self.bn(current_features)
# (16,N) @ (N,16) -> (16,16)，比较两条编码支路的特征相关性。
correlation = history_norm.T @ current_norm / batch_size

# 同编号特征的相关性接近1，不同编号特征的相关性接近0。
diagonal_loss = (torch.diagonal(correlation) - 1.0).square().sum()
off_diagonal_mask = ~torch.eye(
    self.latent_dim, dtype=torch.bool, device=correlation.device,
)
redundancy_loss = correlation[off_diagonal_mask].square().sum()
```

### 5.1 先对每个特征列用同一批样本做标准化

例如一个特征在四条经历中取 `[1,2,3,4]`，训练模式下 BatchNorm 用这四个数的均值和方差标准化这一列。特征放大十倍后，标准化仍把数拉回可比较的尺度。`BatchNorm1d(16, affine=False)` 会分别对 16 列计算统计。`affine=False` 表示这一层没有另外的可学习缩放和偏移。

源码用同一个 `self.bn` 分别处理历史特征和当前特征。训练时，两次调用分别用本次收到的那组样本算统计，也分别更新同一模块里的运行统计。评估模式会使用已有的运行统计。训练模式下，当输入形状为 `(B,16)`，至少需要两条样本才能执行这里的 BatchNorm。[PyTorch BatchNorm1d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm1d.html)

### 5.2 再比较两条支路的特征怎样一起变化

希望历史的第 1 个特征和当前观测的第 1 个特征，在同一批样本中随彼此一起变化。也希望历史的第 1 个特征不用和当前的许多其他特征都重复。在标准化之后，程序用乘积的平均看两列的同步程度。

令 $h_{b,i}$ 是第 $b$ 条记录的第 $i$ 个标准化历史特征，$v_{b,j}$ 是同一记录的第 $j$ 个标准化当前特征，$B$ 是记录数。相关矩阵的第 $i,j$ 格为：

$$
C_{ij}=\frac{1}{B}\sum_b h_{b,i}v_{b,j}.
$$

`history_norm.T` 把 `(B,16)` 变为 `(16,B)`，`@` 表示矩阵乘法。乘上 `(B,16)` 的 `current_norm`，就会对同一条记录相乘并沿 $B$ 条记录相加。除以 $B$ 后得到 `(16,16)`。对角线比较同编号的列，其他位置比较不同编号的列。

代码希望对角线接近 1，其他位置接近 0。`torch.diagonal` 取对角值。`torch.eye` 生成单位矩阵，转成布尔值后用 `~` 反转，得到“取所有非对角格”的索引。两个平方和分别惩罚对应列对不上、不同列过度重复。`device=correlation.device` 让布尔索引与相关矩阵处在同一设备。

举个两列例子，相关矩阵的对角值为 0.8、0.9，另外两格为 0.3、-0.2。对角损失为 $(0.8-1)^2+(0.9-1)^2=0.05$。其余两格的平方和是 $0.3^2+(-0.2)^2=0.13$。原 Barlow Twins 比较同一样本的两种输入视图。TITA 借用它的相关矩阵目标，比较同一经历的历史与当前观测。[Barlow Twins 原论文](https://proceedings.mlr.press/v139/zbontar21a.html)

### 5.3 最后把状态预测放进同一个辅助目标

```python
prediction_loss = F.mse_loss(state_prediction, target=targets)
return diagonal_loss + off_diag_weight * redundancy_loss + 0.01 * prediction_loss
```

`F.mse_loss` 将每条记录的 10 个预测与目标逐一比较，对所有平方误差求平均。函数把这个平均乘 0.01，把非对角损失乘默认的 0.005，再与对角损失相加。

调用者若把辅助损失用于训练，预测误差会通过 `state_prediction` 到达历史编码器。相关矩阵误差会同时到达历史编码器和当前观测编码器。策略损失沿动作均值的计算路径训练历史编码器和动作网络。每条路径最后都由调用者选择何时求导、用哪个优化器改参数。[check 如何求导与更新](../../../../scripts/check_np3o_interface.py.md#12-check_value_update从误差到自动求导再到参数更新)

## 原来问过的问题

<details>
<summary>展开原问题和对应解释</summary>

**Q：layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))这句没看懂，你注释是“每层先做线性变换 Wx+b，再经过ELU非线性变换。”，但是我们没给w和b啊，这句代码最后是一个代数式，还是什么？**

**A：** 构造 Linear 时，PyTorch 已经创建并初始化权重和偏置。这里把层对象放进列表。第 2 节用一行输入演算什么时候才实际使用权重。

**Q：LayerNorm是啥？nn.Module又是啥？`*layers`又是啥语法？？形参`*`又是啥鬼？？**

**A：** 第 2 节分别解释 LayerNorm、展开层列表的星号，以及要求具名参数的星号。第 3 节说明 Module 怎样保存层对象和参数。

**Q：forward指的是啥？我没学深度学习神经网络的前向传播和反向传播。这指的是前向传播这个概念吗？**

**A：** 是。前向用当前参数把输入算成输出。反向计算损失对参数的导数，优化器再修改参数。先看第 3 节的对象调用，再接 check 第 12 节。

**Q：什么叫批量方向？BatchNorm1d这个类也讲。**

**A：** 对 (B,16) 这种张量，批量方向是第 0 轴。第 5.1 节从同一特征的四个样本演示 BatchNorm。

**Q：整个架构指的是数学架构。我都不知道encoder用来干啥的！**

**A：** 第 1 节跟着十帧到 26 个历史特征，再到 8 个动作均值。第 4 节讲同一个编码器后来怎样收到额外学习信号。

**Q：Barlow Twins是啥玩意？**

**A：** 第 5.2 节用两列特征说明怎样计算对角和非对角损失。

**Q：self.mlp_encoder(obs_hist.reshape(batch_size, -1))，没看懂。**

**A：** 先把每台机器人的十帧展平成 330 个数，再把整理后的张量交给已有的历史编码器。编码器按行给出 26 个数。第 4 节对应这段代码。

**Q：device=correlation.device的correlation没有代码提示；correlation[off_diagonal_mask].square().sum()这里的square().sum()也没有代码提示。**

**A：** 补全取决于编辑器能否推断静态类型。给 correlation 加 torch.Tensor 类型标注，或把布尔索引结果先收进一个带 Tensor 类型的变量，能帮助编辑器识别后面的 square 和 sum。第 5.2 节说明这些运算实际接收的数。

</details>

## 完整源码

<details>
<summary>展开完整源码</summary>

```python
"""History-based action network for TITA."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _build_mlp(
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        layer_norm: bool = False,
) -> nn.Sequential:
    """Build an ELU multilayer perceptron with a linear output."""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for index, hidden_dim in enumerate(hidden_dims):
        # 每层先做线性变换 Wx+b，再经过ELU非线性变换。
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        # 沿用旧工厂的准确位置：第一隐层不加LayerNorm。
        if layer_norm and index > 0:
            layers.append(nn.LayerNorm(hidden_dim))
        previous_dim = hidden_dim

    # 输出层不加激活函数，动作均值不在这里裁剪。
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class TitaHistoryActor(nn.Module):
    """Predict action means from current and historical observations."""

    def __init__(
            self, 
            num_proprio: int = 33,
            history_length: int = 10,
            num_actions: int = 8,        
    ) -> None:
        """Initialize the history encoder and action network."""
        super().__init__()
        self.num_proprio = num_proprio
        self.history_length = history_length

        # 旧网络的26个中间输出：10个辅助预测量，加16个学习得到的特征。
        # 本步只迁前向；辅助预测的监督损失后续接入。
        self.prediction_dim = 10
        self.latent_dim = 16
        encoded_dim =  self.prediction_dim + self.latent_dim

        # 默认维度：330 -> 512 -> 256 -> 128 -> 26。
        self.mlp_encoder = _build_mlp(
            num_proprio * history_length,
            encoded_dim,
            (512, 256, 128),
            layer_norm=True,
        )
        # 默认维度：26+33=59 -> 512 -> 256 -> 128 -> 8。
        self.actor = _build_mlp(
            encoded_dim + num_proprio,
            num_actions,
            (512, 256, 128),
        )

        # 当前33维观测的另一条编码支路，只用于辅助学习损失。
        self.obs_encoder = _build_mlp(
            num_proprio,
            self.latent_dim,
            (256, 128),
            layer_norm=True,
        )
        # 沿批量方向分别标准化16个特征列；不学习额外缩放和偏移。
        self.bn = nn.BatchNorm1d(self.latent_dim, affine=False)


    def forward(self, obs: torch.Tensor, obs_hist: torch.Tensor) -> torch.Tensor:
        """Return the action means for a batch of observations."""

        # obs为(N,33)，obs_hist为(N,10,33)，后者不含当前帧。
        # 丢掉最老一帧，补入当前帧；得到最近九帧历史加当前帧。
        history_with_current = torch.cat(
            (obs_hist[:, 1:, :], obs.unsqueeze(1)), dim=1
        )
        batch_size = obs.shape[0]
        history_flat = history_with_current.reshape(batch_size, -1) # (N,330)。
        features = self.mlp_encoder(history_flat) # (N,26)。

        # 顺序沿用旧网络：26维编码在前，33维当前观测在后。
        actor_input = torch.cat((features, obs), dim=-1) ## (N,59)。
        return self.actor(actor_input) # (N,8)，动作分布的均值。

    def auxiliary_loss(
            self,
            obs: torch.Tensor,
            obs_hist: torch.Tensor,
            targets: torch.Tensor,
            off_diag_weight: float = 5e-3,
    ) -> torch.Tensor:
        """Compute state prediction and Barlow Twins losses."""

        batch_size = obs.shape[0]
        # obs=(N,33)，obs_hist=(N,10,33)，targets=(N,10)。
        # 沿用旧损失：直接使用过去十帧，不追加当前帧。
        predicted = self.mlp_encoder(obs_hist.reshape(batch_size, -1))
        state_prediction = predicted[:, :self.prediction_dim] # (N,10)。
        history_features = predicted[:, self.prediction_dim:] # (N,16)。
        current_features = self.obs_encoder(obs)

        # 训练模式下，BatchNorm需要至少两个样本。
        history_norm = self.bn(history_features)
        current_norm = self.bn(current_features)
        # (16,N) @ (N,16) -> (16,16)，比较两条编码支路的特征相关性。
        correlation = history_norm.T @ current_norm / batch_size

        # 同编号特征的相关性接近1，不同编号特征的相关性接近0。
        diagonal_loss = (torch.diagonal(correlation) - 1.0).square().sum()
        off_diagonal_mask = ~torch.eye(
            self.latent_dim, dtype=torch.bool, device=correlation.device,
        )
        redundancy_loss = correlation[off_diagonal_mask].square().sum()

        # 前10个输出拟合仿真提供的10个目标值，按全部元素平均平方误差。
        prediction_loss = F.mse_loss(state_prediction, target=targets)
        return diagonal_loss + off_diag_weight * redundancy_loss + 0.01 * prediction_loss
```

</details>
