# 项目入口、安装与运行

内容纯AI生成，由人类要求生成，未经人类修改。

本文列出当前工程的入口和已有运行命令。任务使用IsaacLab-main、IsaacSim6.2及共享venv，Python包名为`Tita_Lab`。

## 1. 目录与注册

工程目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab`。

| 对象 | 当前值 |
| --- | --- |
| Python源码包 | `src/Tita_Lab` |
| 任务名 | `TitaLab-Velocity-Tita` |
| 环境类 | `Tita_Lab.tasks.velocity.tita_velocity_env:TitaVelocityEnv` |
| 环境配置 | `Tita_Lab.tasks.velocity.config.Tita.env_cfg:VelocityEnvCfg` |
| 任务注册文件 | `src/Tita_Lab/tasks/velocity/config/Tita/__init__.py` |
| 默认RL配置 | `rsl_rl`的PPO模板，完整NP3O尚未接通 |

TitaVelocityEnv建立观测历史和cost管理器，并扩展step/reset；动作延迟也读取该环境持有的编号。任务注册选择这个自定义类。Python包名和目录名在Linux上区分大小写。

## 2. 激活环境与安装本项目

```bash
source "$HOME/my_venvs/lab3_sim62/.venv/bin/activate"
```

安装本项目到这个共享环境：

```bash
uv --no-config pip install \
  --python "$HOME/my_venvs/lab3_sim62/.venv/bin/python" \
  --no-deps --no-sources \
  -e /mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab
```

Lab/Sim依赖由共享环境提供；这条命令只安装项目本身。editable安装会直接读取源码修改，普通编辑无需重复安装；工程目录移动后需要重新安装。

## 3. 查看安装与任务

```bash
python -I -c "import Tita_Lab; print(Tita_Lab.__file__)"
isaaclab list_envs --keyword TitaLab
```

打印的包路径应位于当前工程的`src/Tita_Lab/__init__.py`，任务入口应为表中的TitaVelocityEnv。任务能够列出、配置能够加载和真实物理运行是不同层次的检查。

## 4. 运行现有任务

```bash
isaaclab random_agent \
  --task TitaLab-Velocity-Tita \
  --num_envs 4 --device cuda:0 --max_steps 200 \
  --viz kit physics=isaacsim_physx env.seed=42
```

随机动作不需要checkpoint。200个策略步是每环境2秒仿真，运行到预算后退出；墙钟时间还包括初始化和渲染。倒地后触发base_contact重置符合当前任务规则。native日志不打印准确完成步数，实际结论结合运行方式、退出情况和日志记录。

零动作检查可把random_agent改为zero_agent。相机按现有KitVisualizerCfg跟随机器人。独立资产脚本的职责与接口见[import_assets.py说明](../../code_explaination/Tita_Lab/scripts/import_assets.py.md)。

## 5. 固定的计算约定

| 项目 | 当前配置 |
| --- | --- |
| PhysX物理步 | 0.0025 s |
| decimation / 策略步 | 4 / 0.01 s |
| 回合上限 | 20 s |
| 动作 | 六腿位置目标＋两轮力矩前馈，共8维 |
| 观测组 | policy33、history330、scan187、dynamics30、privileged_state6 |
| 训练包装输入 | 按既定顺序拼成586列 |
| cost | 每环境六列，由任务管理器收集 |

参数来源见[00映射表](00_迁移映射索引.md)，具体原理见01—08及12—24章。代码/CPU/真实运行分别记录在[证据索引](../audits/README.md)，注册和配置检查不能代替物理数值或训练验证。

## 6. 本次actor联调

已有wrapper和actor的调用脚本见[26](26_actor与环境联调.md)。由用户写入并运行，完成后暂停新增内容。现有代码建议见[检查清单](../当前代码检查与修改清单.md)；不再使用旧环境命令。

## 补充问答

**Q：if not isinstance(env.sim.backend_manager, PhysxManager)，backend_manager，没有代码提示！**

**A：** 该属性在[SimulationContext.__init__](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/sim/simulation_context.py)中声明，类型为PhysicsManager。先确认VS Code通过Python: Select Interpreter选择`/home/eaziel/my_venvs/lab3_sim62/.venv/bin/python`，分析路径指向当前Lab源码与Sim扩展，再重载窗口。PhysxManager实现依赖Kit，在事件构造函数内导入；这里判断的是后端实例类型。

**Q：我选哪个？uv吗？**

**A：** 选Enter Interpreter Path…，输入`/home/eaziel/my_venvs/lab3_sim62/.venv/bin/python`。列表中位于PythonRuntimes的uv项是基础Python；本项目依赖在lab3_sim62虚拟环境中。版本相同也应按venv入口路径选择。
