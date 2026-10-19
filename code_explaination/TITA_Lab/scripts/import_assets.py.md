# import_assets.py：导入配置并验证机器人场景

源码目录：`/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/scripts/`。
源码文件：[import_assets.py](/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab/scripts/import_assets.py)。

更新：2026-10-18。用户此前报告此脚本可运行、关节正常；当前文件已引用独立TITA_CFG。以下按当前执行顺序解释；本次整理未复跑脚本。旧包发现与安装建议保留在末尾历史原文中。

## 1. 启动应用，再导入仿真配置

```python
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
```

CLI解析在前，AppLauncher创建应用。随后的仿真模块导入区包含：

```python
from TITA_Lab.assets.tita import TITA_CFG
```

这是工程的绝对包名：TITA_Lab是源码包，assets是子包，tita是模块，TITA_CFG是模块里的配置对象。src是工程布局目录。

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
