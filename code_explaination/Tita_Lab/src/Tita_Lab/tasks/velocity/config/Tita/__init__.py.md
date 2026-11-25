# __init__.py：任务名怎样找到环境和配置

内容纯AI生成，由人类要求生成，未经人类修改。源码核对：2026-11-08。

对应源码：`Tita_Lab/src/Tita_Lab/tasks/velocity/config/Tita/__init__.py`。

## 1. 当前完整源码

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="TitaLab-Velocity-Tita",
    entry_point="Tita_Lab.tasks.velocity.tita_velocity_env:TitaVelocityEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:VelocityEnvCfg",
        "default_agent": "rsl_rl",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
```

## 2. 导入模块时发生登记，创建环境时才实例化

这个模块被import后，顶层gym.register执行，将任务ID和入口信息放入Gymnasium注册表。register本身不启动仿真、不创建机器人，也不构造训练网络。

之后gym.make("TitaLab-Velocity-Tita", cfg=cfg)按ID查到entry_point，再创建TitaVelocityEnv。字符串中的冒号分隔模块路径与对象名；它是入口字符串格式，不是Python继承语法。

## 3. 环境入口、配置入口、训练入口分别选什么

| 字段 | 作用 |
| --- | --- |
| id | 外部用来查找任务的名称 |
| entry_point | 创建哪一种环境对象：TitaVelocityEnv |
| env_cfg_entry_point | Lab加载哪份环境配置：VelocityEnvCfg |
| default_agent | 官方训练入口默认选择rsl_rl路线 |
| rsl_rl_cfg_entry_point | 这条路线加载哪份agent配置：PPORunnerCfg |

__name__是本注册模块的完整Python模块名，用f字符串拼出旁边env_cfg的路径；agents.__name__给出agents包的完整名。这里只保存入口文字，配置加载工具后续才解析它们。

disable_env_checker=True关闭Gymnasium默认环境检查器，不表示任务物理或训练正确；本项目另外用check脚本检查向量化张量接口。

## 4. 为什么登记了PPO，我们还能运行自有NP3O检查

官方训练入口加载上表agent配置，创建RSL wrapper与OnPolicyRunner。当前PPORunnerCfg仍是模板路线，只选择policy观测组。

自有check用agent_cfg_entry_point=None解析环境配置，没有加载PPORunnerCfg；它显式创建TitaNP3OEnvWrapper、Gaussian policy、critic和storage。保存了这些Py文件，不会自动替换注册的默认算法。

完整对象关系见[Wrapper入门](../../../../../../../../docs/训练流程与Wrapper入门.md)，实际环境扩展见[TitaVelocityEnv解释](../../tita_velocity_env.py.md)。本轮仅静态读取注册代码，没有实例化环境或训练。
