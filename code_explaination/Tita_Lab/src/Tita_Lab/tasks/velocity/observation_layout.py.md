# observation_layout.py：把五组观测排成训练输入

**阅读目标：** 明确五组观测的顺序与列宽；输入命名字典，输出裁剪后的(N,586)，供wrapper和模型按固定区间取数。

被解释文件：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/observation_layout.py`。更新：2026-10-21。用户已创建；当前CPU流程检查通过，已由当前wrapper调用，完整训练循环尚未接通。

```python
from collections.abc import Mapping

import torch


def pack_tita_observations(
    observations: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Assemble the task's 586-feature training input with final clipping."""

    layout = (
        ("policy", 33),
        ("scan", 187),
        ("privileged_state", 6),
        ("dynamics", 30),
        ("history", 330),
    )

    blocks = []
    for name, size in layout:
        value = observations[name]
        if value.ndim != 2 or value.shape[1] != size:
            raise ValueError(f"Expected '{name}' with shape (N, {size}), got {tuple(value.shape)}.")
        blocks.append(value)

    # Internal observation/history tensors retain their unclipped values.
    return torch.cat(blocks, dim=-1).clamp(-100, 100)
```

layout规定五段的名称与列数，按这个顺序从observations字典取张量。先检查每段是二维且列数符合约定，再把整段张量放入blocks列表。循环结束后，torch.cat沿列轴拼成(N,586)，最后裁剪到±100。每一行仍对应同一个环境。

| 输出列（右端不含） | 字典键 | 列数 |
| --- | --- | --- |
| 0:33 | policy | 33 |
| 33:220 | scan | 187 |
| 220:226 | privileged_state | 6 |
| 226:256 | dynamics | 30 |
| 256:586 | history | 330 |

cat与非原地clamp产生输出，原字典及内部历史保留原值。6列与30列在这里相邻，构成旧输入中的36列区段。顺序由layout明确规定；不能把字典中各组的声明顺序当作训练布局。

定义函数不会自动调用它。当前自定义wrapper收到环境返回的obs后显式执行pack_tita_observations(obs)；random_agent和当前PPO模板不经过这一自定义接法。旧checkpoint还需要关节和轮序转换。

## 补充问答

**Q：** 我总是看不懂这里：blocks = []，for name, size in layout，value = observations[name]，然后检查和append。

**A：** 先看第一轮：layout取出("policy",33)，Python分别把两项交给name和size，即name="policy"、size=33。下一句成为value=observations["policy"]。假设4个并行环境，value是一个(4,33)张量。

```python
# 存放各组的完整张量，等循环结束后统一拼接。
blocks = []
for name, size in layout:
    # name是字典键；size是该组每个环境应有的列数。
    value = observations[name]
    # ndim是轴数：二维(N,D)应为2；shape[1]是列数D。
    # 任一条件不符合，便报错并停止，不能继续送给网络。
    if value.ndim != 2 or value.shape[1] != size:
        raise ValueError(f"Expected '{name}' with shape (N, {size}), got {tuple(value.shape)}.")
    # 把整个张量加入列表；此时尚未拼接，也不复制其中的数字。
    blocks.append(value)
```

第一轮ndim为2、shape[1]为33，两个错误条件都为False，所以不报错。append后列表只有一个张量；第二轮name="scan"、size=187，加入(4,187)张量。依此类推，最终列表内是五个张量：[(4,33)、(4,187)、(4,6)、(4,30)、(4,330)]。这里的括号只是展示各张量的shape，列表里实际保存的是张量对象。

最后一行cat才把五个张量的列排在一起，得到(4,586)。如果value是(4,32)，列数检查就会报错；如果是(4,2,33)，轴数检查先发现问题。or采用短路求值，第一项为True时不用再访问第二项。

当前检查不显式比较各组行数N；行数不一致时cat也会拒绝。此检查只验证结构，不验证每列数值是否代表正确物理量。

证据：[当前代码CPU检查](../../../../../../docs/audits/2026-10-21_current_observations/state_result.json)。未运行真实物理或训练。
