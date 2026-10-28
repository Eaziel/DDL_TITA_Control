# import_assets.py 早期排查原文

以下为早期独立资产排查的快照，保留当时过程和原文。当前入口已恢复物理循环，阅读[现行说明](../../code_explaination/Tita_Lab/scripts/import_assets.py.md)。

# import_assets.py：导入配置并验证机器人场景

**阅读目标：** 用独立脚本检查同一USD能否加载和参与物理；先读正常创建/推进流程，临时相机诊断仅作历史。

> 2026-10-20复核范围与证据见[09](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/docs/migration/09_阶段审查.md)。独立导入脚本可运行来自用户报告，本次未运行仿真。

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/scripts/`。
源码文件：[import_assets.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab/scripts/import_assets.py)。

更新：2026-10-22。用户此前报告此脚本可运行、关节正常；当前文件已引用独立TITA_CFG。下面1—3节解释原有动态验证流程，5节记录当前临时显示诊断及恢复方式；助手未复跑脚本。旧包发现与安装建议保留在末尾历史原文中。

## 1. 启动应用，再导入仿真配置

```python
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

CLI解析在前，AppLauncher创建应用。随后的仿真模块导入区包含：

```python
from Tita_Lab.assets.tita import TITA_CFG
```

这是工程的绝对包名：Tita_Lab是源码包，assets是子包，tita是模块，TITA_CFG是模块里的配置对象。src是工程布局目录。

## 2. Scene引用同一份资产配置

```python
tita_robot = TITA_CFG.replace(prim_path="{ENV_REGEX_NS}/TITA_Robot")
```

这一行位于TITARobotSceneCfg。replace生成指定路径的配置副本，资产参数继续来自tita.py。运行时按字段名字取对象：

```python
robot = scene["tita_robot"]
```

字段名tita_robot决定scene键；prim_path决定USD场景路径。

## 3. 设置起始状态和材质，再推进仿真

```python
joint_pos = robot.data.default_joint_pos.torch.clone()
joint_vel = robot.data.default_joint_vel.torch.clone()
joint_pos *= torch.empty_like(joint_pos).uniform_(0.5, 1.5)
joint_vel.zero_()
```

脚本从默认状态生成起始关节角并夹关节限位，设置每环境材质参数，写入实际关节状态，再同步执行器目标。随后在循环中写目标、推进sim、更新scene。脚本中的材质随机化张量与广播解释可参考[非RL脚本的原文说明](../../TITA_non_RL/scripts/run_sim.py.md)。

## 4. 当前验证范围

用户的运行报告覆盖这个独立脚本。RL任务通过env_cfg和manager创建自己的环境，当前RL迁移状态记录在[PROJECT_MEMORY](../../../PROJECT_MEMORY.md)。此前包不可发现和重复资产配置的建议已成为历史，后续是否需要安装以当前解释器与运行证据核实。

## 5. 2026-10-22：先确认静态机器人能否显示

用户要求用本脚本排查zero/random运行均不可见。以下静态诊断现已由用户写入源码；用户报告机器人能显示，助手没有启动GUI。

在main中保留Scene创建，将从原sim.reset()到run_simulator(sim, scene)的尾段替换为：

```python
sim.reset()
scene.update(sim.get_physics_dt())

robot: BaseArticulation = scene["tita_robot"]
# 第0台机器人的实际世界位置[x,y,z]，单位m；转到CPU仅用于打印和相机设置。
root_position = robot.data.root_pos_w.torch[0].detach().cpu().tolist()
print("[CHECK] robot world position:", root_position)
x, y, z = root_position
sim.set_camera_view(
    eye=(x + 1.5, y + 1.5, z + 1.0),
    target=(x, y, z),
)

# 第一轮只刷新画面，观察初始模型，不推进物理或运行随机化流程。
while simulation_app.is_running():
    sim.render()
```

原run_simulator函数保留，本轮入口不调用它。sim.render刷新画面；sim.step才推进物理，依据本地SimulationContext源码核对。这样能先观察初始资产，而不让动作、跌倒和随机化干扰。相机使用实际位置，不假定机器人在世界原点。

激活lab3_sim62后，在Tita_Lab目录运行：

```bash
python scripts/import_assets.py --num_envs 1 --device cuda:0 --viz kit
```

本脚本用argparse，命令不附加RL入口使用的physics=预设参数。SimulationCfg.physics=None在本地默认PhysxCfg。Stage路径是/World/envs/env_0/TITA_Robot，与RL入口的Robot不同。

先前备用方案是：若第一轮仍不可见，再在SimulationCfg中增加use_fabric=False。当前用户报告第一轮已可见，只读核实当前文件也没有增加该字段，因此无需执行这个备用对照，不能将显示恢复归因于关闭Fabric。

### 恢复已有物理循环

用户报告机器人保持原USD姿态、不动并有穿模。当前main只调用sim.render刷新画面，原run_simulator没有执行，因而没有运行其中的关节初值写入、驱动目标设置与sim.step。这轮结果证明静态模型能显示，尚不能判断碰撞是否有效。

下一步只将main末尾的render循环替换为：

```python
# 恢复关节初值、驱动目标和物理步进。
run_simulator(sim, scene)
```

保留前面的实际位置相机与现有SimulationCfg，运行同一命令。原函数会先写关节状态和驱动目标，再在循环中执行scene.write_data_to_sim、sim.step、scene.update。观察机器人在物理开始后能否持续显示、能否与地面发生碰撞；这个脚本没有平衡策略，倒下本身不代表失败。用户现已恢复该调用并报告“这次也正常”，助手只读核实调用存在，未观察窗口或测量接触过程。

### 动态运行中的CUDA Graph回退

用户本轮提供的[四行日志](../../../docs/audits/2026-10-22_kit_visibility/user_dynamic_excerpt.txt)依次为CUDA流依赖错误、无法读取DOF位置、Host Newton-actuator图捕获失败并回退eager。DOF位置在这里指关节位置。

本地SimulationCfg.use_newton_actuators默认True。当前资产使用IdealPDActuatorCfg，因此PhysX会通过适配器执行Newton原生执行器计算，物理后端仍是PhysX。该适配器尝试把重复的执行器计算录制成CUDA Graph，减少以后逐项提交GPU工作的开销。本次捕获期间读取PhysX关节状态触发legacy stream同步依赖限制；尚未追到闭源PhysX插件内部具体调用，不能据此归因于驱动安装或USD结构。

原生[physx_runtime.py](/home/eaziel/IsaacLab-main/source/isaaclab/isaaclab/actuators/newton/physx_runtime.py)包含明确回退：

```python
except Exception as exc:
    self._logger.warning("Host Newton-actuator CUDA graph capture failed; using eager execution: %s", exc)
    graphs = []
```

随后将native_actuator_graphs设为空元组；compute没有可重放的图时调用_run_native_actuator_kernels逐步执行。因此捕获失败后仍能继续运行；eager不表示转到CPU，实际计算设备仍由device决定。回退是框架设计的处理路径，不等于已验证全部数值正确或性能无影响。

先前给出的执行器回退旁支建议如下；用户随后明确要求继续排查RL不可见，该改动不作为当前排查的前置步骤：

```python
sim_cfg = sim_utils.SimulationCfg(
    device=args_cli.device,
    dt=0.005,
    # 使用Lab执行器计算路径，避开本次触发冲突的原生执行器图捕获。
    use_newton_actuators=False,
)
```

这个字段切换整个执行器计算路径，并非只关闭CUDA Graph，也不切换物理后端。当前本地API保留此Lab路径并标记deprecated；RL任务已因自定义TitaPD使用False，本次建议是让独立诊断脚本也采用这条现有路径。该建议尚待用户写入与复跑。官方CUDA限制见[CUDA Graphs：Prohibited and Unhandled Operations](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html#prohibited-and-unhandled-operations)。

**Q：** 啥反应没有，f是干啥的？

**A：** F是把Viewport相机聚焦到已选物体。需要先在Stage选择对象，再让Viewport取得键盘输入焦点。F无反应本身不能确定模型是否加载；当前改用脚本按实际位置设置相机来排查。

**Q：** RANDOM也没机器人！草！我怎么改我的那个IMPORT脚本，一步一步排查？TITA_Lab/scripts/import_assets.py

**A：** 先用本节的平地单机器人静态显示测试；若仍不可见，再单独关闭Fabric对照。每次只改变一个排查因素，不同时改资产结构、材质与控制器。

**Q：** 有机器人。但是穿模，无物理效果，机器人就是原始USD的姿态，不动。

**A：** 当前入口只刷新画面，尚未调用已有的初始化和物理循环。先恢复run_simulator(sim, scene)，再判断运动与碰撞；当前静态画面不能作为碰撞失效的证据。

**Q：** 有几个报错，不知道啥原因。这次也正常。

**A：** 本次日志显示执行器CUDA Graph捕获失败，随后框架回退逐步计算，所以运行可以继续。可在独立SimulationCfg加入use_newton_actuators=False，规避这条执行器路径；尚无证据将这几行解释为此前RL机器人不可见的原因。

## 历史解释原文

<details>
<summary>展开2026-10-15的原解释和建议，保留原文；状态属于当时</summary>

## 1. 从独立模块导入 TITA_CFG

### 当前源码位置

```python
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
```

### 解释与建议

当前脚本已有启动段，机器人配置涉及 Sim 相关模块，在 AppLauncher 启动之后的 import 区域添加：

```python
from TITA_Lab.assets.tita import TITA_CFG
```

- `TITA_Lab`：对应工程的 `src/TITA_Lab` 源码包。
- `assets`：该包中的 assets 子包。
- `tita`：assets/tita.py 模块。
- `TITA_CFG`：模块内定义的配置对象变量。

`src` 是源码布局目录，不写进这条 Python 导入语句。当前脚本位于 scripts/，因此使用工程的绝对包名导入；assets/tita.py 内部的 `from . import TITA_LAB_ASSETS_DIR` 则是该模块自身的包内相对导入。

## 2. 场景改用已导入的配置

### 当前源码

```python
tita_robot = TITA_HWBR_CFG.replace(
    prim_path="{ENV_REGEX_NS}/TITA_Robot"
)
```

### 解释与建议

当前脚本仍有重复的 TITA_HWBR_CFG = ArticulationCfg(...) 完整配置（本次读取的第 40～81 行）。导入独立配置后，删除脚本内这个重复定义，并将场景使用处改成：

```python
tita_robot = TITA_CFG.replace(
    prim_path="{ENV_REGEX_NS}/TITA_Robot"
)
```

这样机器人定义来自 assets/tita.py，脚本只负责生成场景与运行验证。Scene 配置成员名仍为 tita_robot，读取仍使用 scene["tita_robot"]。

## 3. src 布局下让 Python 找到工程包

### 项目配置

```toml
[build-system]
requires = ["uv_build>=0.12.6,<0.13"]
build-backend = "uv_build"

[project]
name = "TITA_Lab"
```

### 解释与建议

当前使用的 `/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/bin/python` 从 TITA_Lab 工程根目录检查时，find_spec("TITA_Lab") 为 None，发行包 TITA_Lab 和构建工具 uv_build 均未安装。仅创建 src/TITA_Lab 不会使任意 Python 进程自动找到该包。

建议由用户在现有 env_isaacSL-6.1.0 环境中、TITA_Lab 工程根目录执行：

```bash
python -m pip install -e . --no-deps
python scripts/import_assets.py
```

第一条将当前本地工程安装为可编辑项目，使 Python 识别其源码包；--no-deps 不安装该工程声明的运行依赖。构建工具 uv_build 当前缺失，pip 构建阶段可能需要下载它。第二条才会启动脚本中的 Sim；本次助手没有执行这两条命令。

可编辑安装后，修改 assets/tita.py 一般直接对后续新进程生效，无需每次重新安装。项目的 Python 包可发现性、配置对象实际导入成功、USD 加载及物理运行是不同验证层次；本次只做静态读取与包可发现性检查。

</details>
