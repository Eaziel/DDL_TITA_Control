# mdp/events.py：给实际水平速度一次随机扰动

源码目录：`TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/`。
源码文件：[events.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/events.py)。更新：2026-10-19。**本轮查读确认函数已保存，导出和EventCfg接线尚未在本轮核对。** 下列源码省略版权头；此前候选的独立CPU与Ruff检查通过，未运行真实物理环境。

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def push_by_setting_xy_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
) -> None:
    """Replace world-frame horizontal root COM velocity with random samples."""
    asset: Articulation = env.scene[asset_cfg.name]
    selection = slice(None) if env_ids is None else env_ids
    velocity_w = asset.data.root_vel_w.torch[selection].clone()
    velocity_w[:, :2] = sample_uniform(
        velocity_range[0],
        velocity_range[1],
        (velocity_w.shape[0], 2),
        device=asset.device,
    )
    asset.write_root_velocity_to_sim_index(root_velocity=velocity_w, env_ids=env_ids)
```

## 1. 配置提供调用条件，函数执行物理状态修改

EventCfg.push_robot选择函数、范围、15 s间隔和全局计时；EventManager负责保存配置、计时并调用函数。这个普通函数只在被调用时读取速度、修改xy并写回，不持有跨步缓存，也不需要自己实现reset。

参数env是当前任务环境；env_ids选择本次处理哪些环境；velocity_range是两个水平分量共用的采样区间；asset_cfg.name指定Scene资产。返回类型None表示函数通过写入仿真产生作用，不返回奖励或观测。

`TYPE_CHECKING`导入用于类型提示，future annotations延后注解解析；运行时真正使用的是torch和sample_uniform。asset的Articulation注解帮助编辑器理解机器人对象，`env.scene[asset_cfg.name]`实际取得已经创建的对象。

## 2. 选择环境、复制数据

```python
selection = slice(None) if env_ids is None else env_ids
velocity_w = asset.data.root_vel_w.torch[selection].clone()
```

全局interval调用传env_ids=None。这里将None转换成slice(None)用于取张量的全部行，避免直接tensor[None]增加一维。若传入索引张量，则只取指定环境。

root_vel_w是根刚体质心的世界系速度，torch视图为(N,6)，每行[vx,vy,vz,ωx,ωy,ωz]。选中M个环境后得到(M,6)；全量时M=N。clone复制出独立数据用于修改，最后再通过写入API提交。

## 3. 两列覆盖，其余列保留

```python
velocity_w[:, :2] = sample_uniform(
    velocity_range[0],
    velocity_range[1],
    (velocity_w.shape[0], 2),
    device=asset.device,
)
```

sample_uniform接收下界、上界、输出shape和设备。这里返回(M,2)，各行和各分量独立均匀取样，赋给前两列vx/vy。其余四列保留原来的vz和三个角速度。

使用赋值意味着新vx直接等于抽到的数。旧vx=2、抽到0.3时，新vx=0.3；不是2.3。当前本地官方push_by_setting_velocity使用+=，因此本任务用这个小函数对应Gym的直接赋值。

## 4. 写回物理后端

```python
asset.write_root_velocity_to_sim_index(root_velocity=velocity_w, env_ids=env_ids)
```

PhysX接口接收(M,6)的所选环境数据；env_ids=None表示全量，否则按给定索引更新。本次写入改变实际速度，Commands中的目标速度不变。该方法针对根刚体质心，线速度单位m/s，角速度单位rad/s，均在世界系。

## 5. 如何接入与检查

通过任务`mdp/__init__.pyi`导出函数，再由EventCfg.push_robot引用。完整三处代码、旧Gym规则、global计时及执行顺序差异见[06第6节](../../../../../../../docs/migration/06_events迁移.md#6-运动过程中每15秒扰动一次水平速度)。

此前候选CPU检查覆盖全量None、部分环境、覆盖而非相加、其他四列保留，以及抽取的原生EventManager的全局计时/reset行为。证据：`/tmp/tita_push_events_20261019_9ju291ua/result.json`。函数现已保存，完整导出/配置接线待后续核对；尚未创建物理视图，未验证动态恢复效果。

## 补充问答

**Q：** 我想问了，“w”到底表示啥？？接触力有个w，这里还有w！

**A：** 接口名末尾的`_w`表示world，即世界坐标系。`root_vel_w`给出的线速度和角速度都沿世界坐标轴分解；`net_normal_forces_w`给出的力也沿世界xyz分解。对这些向量接口，`_b`表示body，即沿机身坐标轴表达，机身轴会随机器人转动。

例如假设世界+x向东、+y向北，机身+x为正前方。当机器人朝北以1 m/s前进，同一个线速度写成世界系分量是(0,1,0)，写成机身系分量是(1,0,0)。坐标变了，描述的仍是同一次运动。

变量名velocity_w沿用这个约定，提示它存的是世界系分量；名字本身不会完成坐标变换。此前写出的wx/wy/wz，则是用字母w代替希腊字母ω来表示角速度分量，更清楚的写法是ωx/ωy/ωz。后缀_w的world与角速度符号ω是两个含义。

本地定义：[root_vel_w是root_com_vel_w的别名](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:1376)、[六维质心速度及世界系定义](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:718)、[世界系法向接触力](/home/eaziel/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/base_contact_sensor_data.py:78)。
