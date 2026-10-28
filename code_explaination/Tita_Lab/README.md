# Tita_Lab代码解释

按源码相对路径查阅。功能主线见[迁移说明](../../docs/migration/README.md)，当前检查结果见[审查索引](../../docs/audits/README.md)。

| 源码 | 解释 |
| --- | --- |
| [assets/tita.py](src/Tita_Lab/assets/tita.py.md) | 机器人资产与名义执行器 |
| [scripts/import_assets.py](scripts/import_assets.py.md) | 独立资产验证脚本 |
| [config/Tita/env_cfg.py](src/Tita_Lab/tasks/velocity/config/Tita/env_cfg.py.md) | 配置对象、manager与取数键 |
| [terrain_cfg.py](src/Tita_Lab/tasks/velocity/terrain_cfg.py.md) | 两套地形、生成难度与课程分工 |
| [actuators.py](src/Tita_Lab/tasks/velocity/actuators.py.md) | 显式PD、独立随机因子 |
| [tita_velocity_env.py](src/Tita_Lab/tasks/velocity/tita_velocity_env.py.md) | 历史更新、共享延迟编号、重置前采样与奖励合成 |
| [mdp/observations.py](src/Tita_Lab/tasks/velocity/mdp/observations.py.md) | 轮角屏蔽、历史读取、30维参数与六维采样 |
| [mdp/observation_history.py](src/Tita_Lab/tasks/velocity/mdp/observation_history.py.md) | 十帧缓存与Torch索引 |
| [mdp/events.py](src/Tita_Lab/tasks/velocity/mdp/events.py.md) | 材质、推扰、电机采样；Torch/Warp问答 |
| [mdp/rewards.py](src/Tita_Lab/tasks/velocity/mdp/rewards.py.md) | 接触、功率、加速度状态；类调用问答 |
| [mdp/terminations.py](src/Tita_Lab/tasks/velocity/mdp/terminations.py.md) | 当前接触终止 |
| [mdp/__init__.pyi](src/Tita_Lab/tasks/velocity/mdp/__init__.pyi.md) | 惰性导出 |
| [config/Tita/__init__.py](src/Tita_Lab/tasks/velocity/config/Tita/__init__.py.md) | 环境类与配置入口 |
| [mdp/actions.py](src/Tita_Lab/tasks/velocity/mdp/actions.py.md) | 裁剪/FIR、目标历史与物理步延迟、reset、super问答 |
| [observation_layout.py](src/Tita_Lab/tasks/velocity/observation_layout.py.md) | 五组取数、shape检查、586列拼接与最终裁剪 |
| [mdp/costs.py](src/Tita_Lab/tasks/velocity/mdp/costs.py.md) | 六项已写；原始计算、加速度状态与cost/CMDP问答 |
| [cost_manager.py](src/Tita_Lab/tasks/velocity/cost_manager.py.md) | 已写的任务收集器，六列缓冲、重复调用保护及reset |
| [learning/np3o/env_wrapper.py](src/Tita_Lab/learning/np3o/env_wrapper.py.md) | 已保存：586/六cost与训练接口；早期候选CPU证据和当前命名区别 |
| [learning/np3o/actor.py](src/Tita_Lab/learning/np3o/actor.py.md) | 完整数学架构、Python语法、层/API、shape、前向/反向与已写辅助损失 |
