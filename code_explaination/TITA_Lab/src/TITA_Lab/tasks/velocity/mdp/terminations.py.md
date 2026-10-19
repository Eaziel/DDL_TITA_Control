# mdp/terminations.py：从机身接触力得到结束标记

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/`。
源码文件：[terminations.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/terminations.py)。

更新：2026-10-18。函数、导出和配置已由用户保存，静态检查及保存函数的CPU检查通过；真实传感器与环境未运行。

## 1. 当前代码

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def illegal_contact_current(
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return whether selected bodies exceed a contact-force threshold."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # shape: (N, B_all, 3)
    #   N     = num_envs        环境数
    #   B_all = 传感器监测的所有 body 数
    #   3     = 力的 xyz 分量
    forces_w = contact_sensor.data.net_normal_forces_w.torch

    # shape: (N, B, 3)
    #   只保留配置里指定的那些 body，B = len(sensor_cfg.body_ids)
    selected_forces_w = forces_w[:, sensor_cfg.body_ids, :]

    # shape: (N, B)
    #   对最后一维（xyz）求模长 → 每个 body 一个“力的大小”
    force_magnitudes = torch.linalg.vector_norm(
        selected_forces_w, dim=-1
    )

    # shape: (N,)  bool
    #   每个环境一个判断：这个环境里“有没有任何一个 body”超阈值
    return torch.any(force_magnitudes > threshold, dim=1)
```

## 2. 数据按什么顺序流动

TerminationsCfg传入SceneEntityCfg("contact_forces", body_names=["base_link"])与threshold=1.0。管理器按名字解析身体索引，调用函数时带上这些参数。

| 步骤 | 结果形状 | 本步得到什么 |
| --- | --- | --- |
| 从scene.sensors取传感器 | ContactSensor对象 | 当前接触报告的数据入口 |
| net_normal_forces_w.torch | (N,M,3) | M个身体各自的世界系xyz法向力 |
| 按body_ids取身体 | (N,B,3) | B个选中身体的力，当前只选机身，B=1 |
| vector_norm沿dim=-1 | (N,B) | 每个选中身体的力模 |
| 与threshold比较 | (N,B)布尔 | 该身体是否超过1 N |
| any沿dim=1 | (N,)布尔 | 每个环境是否有选中身体超过阈值 |

函数把布尔张量交给TerminationManager，环境根据合并后的结束标记重置相应环境。

## 3. 为什么读取当前力

旧Gym判断本次读取的接触力；本地Lab现成illegal_contact从力历史取最大值。当前任务小函数读取net_normal_forces_w，保留Scene的3帧采样历史供其他用途。完整终止配置、20 s时长与超时分类见[07 Terminations](../../../../../../../docs/migration/07_terminations迁移.md)。

## 4. 检查范围

实际保存函数的CPU检查使用三环境、身体顺序轮/机身/轮、body_ids=[1]：仅轮子100 N时False，机身(0.8,0.8,0) N时True，机身恰好1 N时False。输出(3,)bool，结果[False,True,False]。

报告：`/tmp/tita_termination_review_20261018/saved_code_result.json`。实际身体绑定和物理读数尚待运行。

## 补充问答

问题保留用户原句；回答按对应代码整理。

**Q：**

> 小问题：dim这个参数，他没有代码提示，是没有吗？

**A：** 有。当前PyTorch 2.12的签名包含dim=None，保存函数的CPU检查也执行了dim=-1。函数通过编译扩展提供，静态提示可能不完整；未检查编辑器当前解释器，因此不能断言缺失提示的具体原因。

```text
linalg.vector_norm(x, ord=2, dim=None, keepdim=False, *, dtype=None, out=None) -> Tensor
```

[PyTorch 2.12文档](https://docs.pytorch.org/docs/2.12/generated/torch.linalg.vector_norm.html)。

**Q：**

> 第二个是， selected_forces_w = forces_w[:, sensor_cfg.body_ids, :]这句的body_ids，会是什么值？

**A：** 当前配置按名字选择base_link，SceneEntityCfg把它解析成传感器身体顺序中的索引。只选一个身体时，解析结果为[k]；具体k尚未从真实传感器读取。若举例k=2，则相当于：

```python
selected_forces_w = forces_w[:, [2], :]  # (N,1,3)
```

三个位置依次选择全部环境、机身、全部xyz分量。[2]保留身体维。同一组身体索引作用于所有环境。

源码：[SceneEntityCfg身体解析](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/managers/scene_entity_cfg.py:226)。

**Q：**

> 第三个，torch.linalg.vector_norm(，dim-1指的是，算最后一维的范数吗？就是[:, sensor_cfg.body_ids, :]的最后一个冒号取出来的？

**A：** 对，代码中的dim=-1指定最后一维，也就是保留下来的Fx/Fy/Fz。默认ord=2，计算sqrt(Fx²+Fy²+Fz²)，把(N,B,3)变成(N,B)。例如：

```python
selected_forces_w = torch.tensor([
    [[3.0, 4.0, 0.0]],
    [[0.0, 0.0, 0.5]],
])  # (2,1,3)

torch.linalg.vector_norm(selected_forces_w, dim=-1)
# tensor([[5.0], [0.5]])，shape=(2,1)
```

最后的冒号保留三个分量；dim=-1进一步指定沿这一轴计算。

**Q：**

> any是什么函数？

**A：** torch.any沿指定轴判断是否至少有一个True。延续上一例：

```python
force_magnitudes = torch.tensor([[5.0], [0.5]])
force_magnitudes > 1.0
# tensor([[True], [False]])，shape=(2,1)

torch.any(force_magnitudes > 1.0, dim=1)
# tensor([True, False])，shape=(2,)
```

dim=1对应选中的身体轴，每个环境单独判断：任一选中身体超阈值就True。当前B=1，最终就是每个环境的机身是否超阈值。[PyTorch 2.12文档](https://docs.pytorch.org/docs/2.12/generated/torch.any.html)。

