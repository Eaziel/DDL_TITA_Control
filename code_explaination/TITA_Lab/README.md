# TITA_Lab 代码解释

对应源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/`。

更新：2026-10-18。任务为TitaLab-Velocity-Tita；当前33维/noise及历史完整接线已写入并静态核对。完整方案与理由见[05第5—6节](../../docs/migration/05_observations迁移.md#5-历史lab-中怎样接入)。本目录用于查单个Python文件：当前源码与职责在前，Q/A在后，旧原文单独折叠。reset、终止与跟踪/平稳/高度/动作变化/碰撞/失败共8项奖励也已写入并静态核对；RL环境尚未运行。

- [assets/tita.py：机器人配置模块](assets/tita.py.md)
- [scripts/import_assets.py：导入配置并验证生成](scripts/import_assets.py.md)
- [env_cfg.py：Physics、Scene、Actions、Commands、Observations 与 Gym 参数落点](src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py.md)
- [mdp/observations.py：轮角屏蔽函数](src/TITA_Lab/tasks/velocity/mdp/observations.py.md)
- [mdp/observation_history.py：10帧缓存、Torch API与对象状态，用户已写入并局部CPU核对，完整接线已写入](src/TITA_Lab/tasks/velocity/mdp/observation_history.py.md)
- [tita_velocity_env.py：当前step开头推进历史、重置时清理](src/TITA_Lab/tasks/velocity/tita_velocity_env.py.md)
- [config/TITA/__init__.py：已切换环境类；区分环境入口和配置入口](src/TITA_Lab/tasks/velocity/config/TITA/__init__.py.md)
- [mdp/__init__.pyi：自定义MDP函数导出](src/TITA_Lab/tasks/velocity/mdp/__init__.pyi.md)
- [mdp/terminations.py：当前接触力终止，已写入并局部检查](src/TITA_Lab/tasks/velocity/mdp/terminations.py.md)
- [mdp/rewards.py：当前帧小腿碰撞计数，已保存并局部核对](src/TITA_Lab/tasks/velocity/mdp/rewards.py.md)

后续按源码相对路径建立 `原文件名.md`；同一个源码文件的解释持续维护在同一份文档中。每份文档开头写明源码路径，每段解释先贴源码再说明。
