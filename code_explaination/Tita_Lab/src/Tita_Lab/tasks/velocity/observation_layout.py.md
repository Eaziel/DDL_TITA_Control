# observation_layout.py：五组观测怎样成为固定586列输入

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-02。

被解释文件：[observation_layout.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/observation_layout.py)。

本文件将Lab按名称返回的五组观测拼成模型约定的一张矩阵。读完应能跟着for循环找到每份数据，解释append与cat的区别，并根据列范围还原actor和critic实际读取了什么。

本文解释当前文件及本地 IsaacLab-main / Sim 6.2 调用顺序。只读检查与用户已有200步日志分别陈述，不把它们当成完整训练或接触物理验证。

## 1. 当前完整源码

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

## 2. Lab输出字典，模型接收矩阵

观测管理器输出类似：

```python
observations = {
    "policy": policy_tensor,
    "scan": scan_tensor,
    "privileged_state": state_tensor,
    "dynamics": dynamics_tensor,
    "history": history_tensor,
}
```

各值都是二维张量 `(N,D_group)`，一行对应一台机器人。这里变量名是示意，真正输入来自reset/step已经算好的观测。

`Mapping[str,torch.Tensor]`是参数类型注解，表示“字符串键映射到张量值”的对象；字典满足这个接口。函数只按键读取，不要求修改调用者的字典，也不要求输入必须是某个特定dict子类。

五个组保留名称便于环境配置，模型需要固定列号便于切片；这个函数承担名称到列布局的明确转换。

## 3. layout元组逐项规定名字和列数

`layout`是由五个二元组组成的元组。每个二元组形如 `("policy",33)`，把组名和预期单环境维数放在一起。最终矩阵的顺序取决于这份layout，输入字典的遍历顺序不会决定列顺序。

```python
blocks = []
for name, size in layout:
    value = observations[name]
    ...
    blocks.append(value)
```

第一次循环，name为 `"policy"`，size为33；`observations[name]`取得 `(N,33)`。第二次取scan的 `(N,187)`，依次继续。`for name,size in ...`是解包语法，并不声明两个并行循环。

`blocks=[]`建立空Python列表。`.append(value)`把整个张量对象加入列表，不逐元素复制、不把数值加到已有张量、不先展开成Python数字。循环后列表恰有五个元素，每个元素仍是一整块批量观测。

## 4. 形状检查在防哪类错误

```python
if value.ndim != 2 or value.shape[1] != size:
    raise ValueError(...)
```

`ndim`是轴的数量；`shape[1]`是第1轴长度，也就是列数。`or`具有短路规则：如果轴数已经不等于2，就不再访问可能不存在的第1轴。条件满足表示输入违反约定，raise终止当前调用并给错误信息。

例如scan误为 `(N,17,11)`，虽然总元素187，仍会因ndim为3而报错；函数不会暗自猜如何展开。history误为 `(N,329)`也会报错。

当前检查逐组确认二维与列数，并没有显式比较每组行数、device、dtype或数值是否有限。`cat`自身要求非拼接轴一致、设备兼容；上层check另检查有限值。准确理解各检查的边界，才能知道某个报错来自哪里。

错误字符串中的f前缀表示在运行时插入变量值；`tuple(value.shape)`把形状变成普通元组用于输出。它不改变张量数据。

## 5. cat只沿列拼接，同一机器人的数据留在同一行

```python
return torch.cat(blocks, dim=-1).clamp(-100, 100)
```

在二维张量中dim=−1与dim=1都表示列轴。第n行输出为五个第n行依次拼接：

$$
x_n=[o^{policy}_n,\ o^{scan}_n,\ o^{state}_n,\ o^{dynamics}_n,\ o^{history}_n]
\in\mathbb R^{586}.
$$

列数是 $33+187+6+30+330=586$。例如简化两组分别为 `[[1,2],[3,4]]` 与 `[[10],[20]]`，横向cat得到 `[[1,2,10],[3,4,20]]`；没有把两台机器人的观测接成一行，也没有按列求和。

`cat`返回新矩阵。后面的 `clamp`再次返回裁剪结果，每元素限制在[−100,100]：大于100取100、小于−100取−100。该处不使用带下划线的原位方法，不会把内部policy或history缓存一起改成裁剪值。

这层裁剪是训练输入的最终裁剪，组内noise/clip/scale已在观测管理器执行。改变最终裁剪时不应凭直觉回头修改历史内部存储；两个阶段有不同用途。[cat官方API](https://docs.pytorch.org/docs/2.11/generated/torch.cat.html)

## 6. 列号如何与当前网络对应

| 左闭右开切片 | 列数 | 内容 |
| --- | --- | --- |
| `0:33` | 33 | 当前policy |
| `33:220` | 187 | scan |
| `220:226` | 6 | privileged_state |
| `226:256` | 30 | dynamics |
| `256:586` | 330 | history |

actor动作支路由policy类取 `obs[:,:33]` 和 `obs[:,-330:]`，因此不直接得到scan和36维额外信息。critic读取当前33、扫描187、`220:256`的额外36维、最后330维历史，并分别处理。

历史330展开自 `(N,10,33)`，网络恢复这个形状时必须使用同一全帧顺序。数字586只检查总宽度，不能证明每个物理量已按正确顺序排好；例如调换两个同宽组仍可能形状正确。固定layout和网络切片共同定义了这个数据协议。

`env_wrapper.reset/step/get_observations`都会调用本函数，读取的是已经存在的观测结果。这个函数本身不再次采噪声、不更新接触记忆、不推进历史或物理状态。

## 源码与API依据

- [本项目wrapper的调用处](../../learning/np3o/env_wrapper.py.md)
- [本项目观测内容](mdp/observations.py.md)
- [本项目policy切片](../../learning/np3o/policy.py.md)
- [本项目critic切片](../../learning/np3o/critic.py.md)
- [PyTorch clamp](https://docs.pytorch.org/docs/2.11/generated/torch.clamp.html)
