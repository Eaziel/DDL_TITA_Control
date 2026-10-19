# __init__.py：选择实际创建的环境类

日期：2026-10-18。状态：**用户已将entry_point改为TitaVelocityEnv；环境子类已存在，_reset_idx缩进现已修正。** 本页记录源码关系；历史更新、输出与重置接线均已保存，未运行Gym注册或实例化环境。

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/`。
源码文件：[__init__.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/__init__.py:15)。

## 对应代码与解释

```python
# 当前gym.register(...)中的环境类入口：
entry_point="TITA_Lab.tasks.velocity.tita_velocity_env:TitaVelocityEnv",
```

冒号左边是Python模块路径，右边是模块中的类名。当前入口与用户已创建的[TitaVelocityEnv](../../tita_velocity_env.py.md)对应。此前默认入口是isaaclab.envs:ManagerBasedRLEnv；新增子类后，Gym按新入口选择子类，子类仍沿用父类manager-based流程。

```python
# 注册项中的kwargs节选，其余字段省略：
kwargs = {
    "env_cfg_entry_point": f"{__name__}.env_cfg:VelocityEnvCfg",
}
```

两个入口回答不同问题：entry_point选择“创建哪一种环境对象”；env_cfg_entry_point供Lab任务加载工具取得“交给环境的配置对象”。[load_cfg_from_registry](/home/eaziel/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py:120)解析配置入口并实例化配置类；环境类接收这个cfg，建立scene和各manager。因此VelocityEnvCfg与TitaVelocityEnv是配合关系，不能因新增环境子类就把配置职责迁走。第二段展示kwargs中的环境配置入口，其余字段省略。

任务ID和agent配置沿用原值。当前注册入口、缓存类、MDP函数、history组以及环境step/reset共同完成历史接线。只读检查不代表任务可运行，未启动环境或训练。
