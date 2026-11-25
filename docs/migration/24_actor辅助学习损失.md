# 24 动作网络的辅助学习损失

更新：2026-10-26。内容纯AI生成，由人类要求生成，未经人类修改。

状态：动作网络和辅助计算体已保存。2026-11-01将本章方法引用与当前actor.py的`auxiliary_loss`对齐；本轮不重跑既有数学与梯度检查，之前证据见[实际结果](../audits/2026-10-27_saved_stack_sim62/actor/result.json)。本章用于理解计算，当前迁移目标见[27动作分布](27_动作分布迁移.md)。

基础说明：[actor.py逐项解释](../../code_explaination/Tita_Lab/src/Tita_Lab/learning/np3o/actor.py.md)。它按Python语法、网络对象、数学计算、张量数据流逐项讲解当前actor及本章辅助代码。

## 1. 本次完成什么

让历史编码网络学习两件事：预测十个已有训练状态量，并让历史与当前观测的两路编码学习相互对应的16维特征。旧TITA用MlpBarlowTwinsActor.BarlowTwinsLoss实现，本次沿用其实际公式和层结构。

本次只让actor能够计算这项损失，暂不建立优化器。完整旧NP3O.update在启用imi_flag时，把它按imi_weight加入策略/约束/价值/熵组成的总损失，再backward与optimizer.step。本地代码位置：/home/eaziel/Downloads/tita_rl-master/algorithm/np3o.py:241—256。不能把本章辅助损失等同完整NP3O优化器。

## 2. 原来26个输出怎样使用

历史编码器已有输出(N,26)。旧代码将它划分为前10列和后16列：

| 部分 | 学习目标 | 用什么数据比较 |
| --- | --- | --- |
| 前10列 | 预测十项状态 | 训练观测中间36列的前10列 |
| 后16列 | 学习历史特征 | 当前33维观测经另一编码器产生的16维特征 |

当前布局中，十项目标是packed_obs[:,220:230]，形状(N,10)：六维privileged_state，加dynamics前四列。

| 列数 | 内容与当前变换 |
| --- | --- |
| 3 | 机身坐标系线速度乘2 |
| 2 | 左右轮过滤接触布尔值转float，再减0.5 |
| 1 | 延迟历史编号除以(L−1)，不是直接的毫秒数 |
| 1 | 基座质量相对名义值的增量，kg |
| 3 | 基座局部质心相对名义值的增量，m |

这些是已有观测返回的训练目标，不在损失函数里重新读取或计算物理状态。它们没有直接输入本章动作forward，而用于监督历史编码。

## 3. 已写入的成员与方法

文件：/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/learning/np3o/actor.py。

当前属性名为prediction_dim。以下是统一方法拼写后的实现参考，说明已写计算体的组织。

辅助计算使用的导入：

```python
from torch.nn import functional as F
```

TitaHistoryActor.__init__中的两个辅助成员：

```python
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

辅助方法与forward同级，当前代码如下：

```python
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
        state_prediction = predicted[:, : self.prediction_dim]  # (N,10)。
        history_features = predicted[:, self.prediction_dim :]  # (N,16)。
        current_features = self.obs_encoder(obs)  # (N,16)。

        # 训练模式下，BatchNorm需要至少两个样本。
        history_norm = self.bn(history_features)
        current_norm = self.bn(current_features)
        # (16,N) @ (N,16) -> (16,16)，比较两条编码支路的特征相关性。
        correlation = history_norm.T @ current_norm / batch_size

        # 同编号特征的相关性接近1，不同编号特征的相关性接近0。
        diagonal_loss = (torch.diagonal(correlation) - 1.0).square().sum()
        off_diagonal_mask = ~torch.eye(
            self.latent_dim, dtype=torch.bool, device=correlation.device
        )
        redundancy_loss = correlation[off_diagonal_mask].square().sum()

        # 前10个输出拟合仿真提供的10个目标值，按全部元素平均平方误差。
        prediction_loss = F.mse_loss(state_prediction, targets)
        return diagonal_loss + off_diag_weight * redundancy_loss + 0.01 * prediction_loss
```

这是现有类新增成员和方法，后续actor-critic对象会直接调用auxiliary_loss；本步不增加MDP导出或env_cfg配置。

## 4. 数据怎样进入这项损失

教学调用示意：

```python
current = packed_obs[:, :33]
history = packed_obs[:, -330:].reshape(packed_obs.shape[0], 10, 33)
targets = packed_obs[:, 220:230]
loss = actor.auxiliary_loss(current, history, targets)
```

loss是零维标量张量，保留求梯度所需的计算关系；返回loss本身不会修改网络参数。后续训练循环会把它加入总损失并执行反向传播。

这里有两个时间窗口，严格沿用旧代码：

| 方法 | 输入历史怎样交给mlp_encoder |
| --- | --- |
| 动作forward | 从过去十帧丢最老一帧，补当前，形成最近九帧+当前 |
| auxiliary_loss | 原样使用过去十帧，不补当前 |

本次两处CPU窗口检查均通过。不要为了让两种调用看起来一致而随意合并。

## 5. 三项损失的数学含义

第一项状态预测是平均平方误差：

$$
L_{pred}=\frac{1}{10N}\sum_{n=1}^{N}\sum_{j=1}^{10}(\hat s_{nj}-s_{nj})^2.
$$

F.mse_loss默认对所有元素的平方误差取平均。预测偏离目标越多，该项越大。[PyTorch MSE说明](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.mse_loss.html)

另外两项使用16维特征。BatchNorm1d在训练模式下，分别沿N个样本标准化每个特征列；affine=False表示不附加可学习缩放与偏移。它与此前LayerNorm沿每条样本的特征维做标准化不同。这里训练批量N至少为2。[PyTorch BatchNorm说明](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BatchNorm1d.html)

令两路标准化后的特征为H、O，均为(N,16)，计算跨两路的相关矩阵：

$$
C=H^{\mathsf T}O/N,\qquad C\in\mathbb R^{16\times16}.
$$

.T是二维转置，@是矩阵乘法。C的第i行、第j列比较历史支路第i个特征与当前支路第j个特征。

损失鼓励同编号特征相关性接近1：

$$
L_{diag}=\sum_i(C_{ii}-1)^2,
$$

并减少不同编号特征间的相关性：

$$
L_{off}=\sum_{i\ne j}C_{ij}^2.
$$

最后沿用旧组合：

$$
L_{aux}=L_{diag}+0.005L_{off}+0.01L_{pred}.
$$

目标是学习两路对应、不同分量较少重复的表示；损失设计本身不保证训练结果。torch.diagonal取对角线，square求元素平方，sum求和；eye建立对角True的布尔矩阵，~取反后选出非对角线。state_prediction[:,:10]与history_features[:,10:]读取的都是同一历史编码结果。

注意预测损失只更新参与该计算的历史编码器参数，相关性损失涉及历史和当前两路编码器；本项辅助损失没有调用self.actor动作输出子网络。动作子网络的更新还需后续策略损失。

## 6. 当前证据与早期候选

[实际源码检查](../audits/2026-10-26_saved_actor/README.md)记录当时计算体的数值：动作batch1/4/64与旧分支差0；辅助batch4/32值差约3.81e-6/0；所检参数和输入梯度差0，BN缓冲一致，两种历史窗口正确。当前方法名为auxiliary_loss；本轮只核对命名与调用。

[早期候选检查](../audits/2026-10-26_actor_auxiliary/README.md)保留当时方案的结果，不再用于声称当前实现尚未写入。float32归约存在微小舍入差，按记录中的rtol/atol核对。

两类检查均未执行optimizer.step、仿真或真实数据训练，不能据此推断平衡或完整NP3O效果。当前先读actor解释，不新增模块。
