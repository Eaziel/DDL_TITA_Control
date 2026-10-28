# commands.py：TITA平移速度命令死区

**阅读目标：** 理解配置类如何选中实现类，以及子类怎样复用原生采样并将过小平移目标归零。

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp`。
源码文件（已保存）：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/src/Tita_Lab/tasks/velocity/mdp/commands.py`。
更新：2026-10-25。以下为2026-10-25当前保存源码；导出、配置导入与class_type身份检查通过，未运行Sim。

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from isaaclab.utils import configclass


class TitaVelocityCommand(UniformVelocityCommand):
    """Velocity commands with the legacy TITA planar speed deadband."""

    cfg: TitaVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Sample velocity commands with a planar speed deadband."""
        super()._resample_command(env_ids)

        # 命令缓冲为(N, 3)，每行[vx, vy, wz]。
        # env_ids选出本次需要重新采样的M个环境。
        vel_xy = self.vel_command_b[env_ids, :2]  # (M, 2)，单位m/s。
        speed_xy = torch.linalg.vector_norm(vel_xy, dim=1)  # (M,)。

        # 只有速度模长大于阈值（默认0.2 m/s）才保留平移命令。
        moving: torch.Tensor = speed_xy > self.cfg.lin_vel_deadband  # (M,)，bool。

        # (M, 1)广播到两列，并把处理后的结果写回原始命令缓冲。
        self.vel_command_b[env_ids, :2] = vel_xy * moving.unsqueeze(1)


@configclass
class TitaVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for TITA velocity commands."""

    class_type: type[TitaVelocityCommand] = TitaVelocityCommand

    lin_vel_deadband: float = 0.2
    """Planar command speeds at or below this threshold are zeroed [m/s]."""
```

## 1. 两个类分别保存什么、执行什么

TitaVelocityCommandCfg继承UniformVelocityCommandCfg，继承asset_name、采样范围/周期、航向和可视化字段，新增lin_vel_deadband。class_type指向TitaVelocityCommand。CommandManager读取Cfg，再用class_type(cfg,env)构造真正保存命令张量的实例；配置类自身不存本轮随机目标。

TitaVelocityCommand沿用父类__init__创建的vel_command_b、heading_target、time_left等数据，只覆盖_resample_command。重采样时，父类填入随机目标，子类再修改新目标。原生reset、计时器和heading外环继续工作；new cfg的类型注解让代码工具知道self.cfg中有新增字段。

## 2. 一批环境的数据如何变化

vel_command_b是(N,3)，N为全部环境数，三列依次为vx、vy、wz。env_ids选择本次M行，vel_xy为(M,2)；vector_norm(dim=1)得到每行平移速度模长，shape=(M,)。与阈值比较得到bool moving；unsqueeze(1)扩为(M,1)，乘法广播到两列，最后写回vel_command_b所选行。

(0.12,0.09)的模长为0.15，被改成(0,0)；(0.15,0.15)模长约0.212，保留；(0.2,0)因使用严格大于而置零。其他环境的行不改变，第2列wz和heading_target沿用父类规则。批量索引得到临时张量后，本函数用显式索引赋值写回，避免只改副本。

## 3. 怎样让环境使用它

依次完成mdp/__init__.pyi的__all__和from .commands import TitaVelocityCommandCfg，再将CommandsCfg.base_velocity的构造类换为mdp.TitaVelocityCommandCfg，显式加lin_vel_deadband=0.2。完整配置见[21](../../../../../../../docs/migration/21_命令死区迁移.md)。保持条目名base_velocity和Ranges等已有参数。

## 4. 谁读取结果

CommandManager.get_command("base_velocity")返回生成器.command属性，而该属性返回vel_command_b。同一目标源随后由观测项读取并缩放，由追踪奖励读取作误差目标，由课程读取作期望位移判据。因此数值处理应放命令生成器内部。

Native父子CPU对照检查通过，包含真实原生构造/reset/compute和CPU状态替身；当前源码已写入且配置指向新类，未运行真实环境。证据见[本轮检查](../../../../../../../docs/audits/2026-10-25_command_deadband/README.md)。本轮不改Lab命令更新相对奖励的位置。

## 补充问答

**Q：TitaVelocityCommand这个最后到底咋被用了？这个类也没有被公开（被import啥的）！**

**A：** 配置里class_type = TitaVelocityCommand保存的是类对象。同一个commands.py被导入时，Python依次定义实现类和配置类；即使外部只导入Cfg，Cfg.class_type仍持有实现类。mdp/__init__.py执行lazy_export，额外解析相邻.pyi中的from .commands import TitaVelocityCommandCfg，让env_cfg可写mdp.TitaVelocityCommandCfg。

普通Python不会仅因为.pyi存在就把其中导入作为运行时导入执行；这里由Lab的lazy_export显式建立关联。CommandManager创建对象的实际语句是：

```python
term = term_cfg.class_type(term_cfg, self._env)
self._terms[term_name] = term
```

当term_name是base_velocity，term_cfg是该项TitaVelocityCommandCfg实例时，第一句相当于TitaVelocityCommand(term_cfg,env)。class_type右侧是类对象，后面的括号才构造实例；第二句将实例存入管理器字典。

环境每策略步调用command_manager.compute(dt=self.step_dt)，管理器逐项调用term.compute(dt)。实例继承的compute会倒计时，到期经_resample调用self._resample_command；self仍是TitaVelocityCommand实例，因此执行本项目覆盖的方法。reset也经相同_resample调用。生成器长期保存vel_command_b，观测、追踪奖励与课程通过get_command('base_velocity')读取。

依据：本地command_manager.py:412/417、:354、:145、:166、:368；manager_based_rl_env.py:112/287/427；utils/module.py:84。当前实际cfg导入、validate和class_type身份检查记录在docs/audits/2026-10-25_command_deadband/current_wiring.json。

**Q：而且什么叫消费者？**

**A：** 谁读取一份结果并拿它继续计算，就是这里所说的消费者。例如velocity_commands观测读取目标速度，追踪奖励读取目标与实际速度比较，课程读取目标计算期望移动距离。这个词不是Lab的类或API；解释时直接给出读取它的具体函数即可。
