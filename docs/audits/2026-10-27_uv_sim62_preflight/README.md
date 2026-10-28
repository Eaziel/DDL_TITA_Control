# Ubuntu 改用 uv venv 与 Sim6.2：安装前核查

日期：2026-10-27，Asia/Shanghai。仅核查，没有卸载、安装、仿真、训练、Git操作或修改受保护工程。

## 结论

可以放弃 Conda，采用独立 uv Python＋共享 venv。现有 TITA 计算、历史缓存、cost、wrapper、actor 没有发现需要推倒重做的依据；用户已接受 API 与调用链适配。不能保证升级只涉及名称变化，也不能在新环境运行前确认所有物理行为保持一致。

建议先保留旧 `/home/eaziel/IsaacLab` 完整普通文件目录及依赖信息，再清理旧安装。尚未全面区分旧 Lab 中的用户文件与上游文件，不能支持无备份清空；新环境验收后删除旧环境最稳妥。若用户选择先删，需接受重装期间暂时不能运行。

已保存的[旧环境包元数据](old_environment_packages.json)不是完整 Conda 锁文件、不包含全部非Python依赖，也不是可直接恢复的环境备份。本轮没有创建源码备份或Git回滚点。

## 历史与当前安装

按用户授权搜索共享 CodexData 的 Windows11、Ubuntu_2404 两边 sessions/archived_sessions，只读相关消息，定位见[历史来源](history_sources.json)。Windows 用户明确希望 Python 本体在 PythonRuntimes、环境在 `my_venvs/环境名/.venv`、不同位置的机器人工程共享环境，激活后直接执行命令。用户日志确认安装 Python3.12.15，后续报告 Sim/Lab 安装完成；历史助手查过包元数据与启动器，不等于 TITA 新版 PhysX 验证。

| 对象 | 当前读取结果 |
| --- | --- |
| OS / 系统Python | Ubuntu24.04.5；`/usr/bin/python3` 为3.12.3；deb `3.12.3-1ubuntu0.17` |
| ROS | `/opt/ros/jazzy`；rclpy deb `7.1.12-1noble.20260912.162354`；ros2 shebang 为 `/usr/bin/python3` |
| Lab | `/home/eaziel/IsaacLab`，VERSION为3.0.0；当前配置固定Sim6.1.0.0 |
| 旧环境 | `/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0`，Sim6.1.0.0、Torch2.12.0+cu130、rsl-rl-lib5.5.1；Lab与TITA为editable |
| 其他环境 | `DL_study`存在，删全部Conda环境也会删其依赖；未审查其项目用途 |
| 用户成果 | TITA_Lab、TITA_non_RL在共享工作区，两者未发现符号链接；TITA USD使用自己的资产路径 |
| 资产设置 | Conda保存 `ISAACSIM_ASSET_ROOT=/home/eaziel/isaacsim_assets/Assets/Isaac/6.1`；环境删除后设置不会自动进入venv，资产文件本身在Miniconda外 |
| Shell | `.bashrc`有ROS自动source、Conda初始化、`goisaac-6.1`别名；重装后需要用户调整入口 |
| 驱动边界 | 内核模块595.91.07高于安装页最低595.58.03；工具nvidia-smi通信失败，不据此认定宿主驱动故障，也未验证GPU运行 |

## 已定位的适配

### 管理器由类改成实例

**2026-10-27后续核查补充：** 以下为早期表达式候选，不表示可以把实现类放在模块顶部导入。PhysxManager依赖Kit，完整修法在事件__init__内延后导入，见[新版首组适配](../../migration/25_Lab_main与Sim62_入口和配置.md)。

当前 `TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/events.py`：

```python
if env.sim.physics_manager.__name__ != "PhysxManager":
    raise ValueError("RandomizeTitaFriction currently supports Isaac Sim PhysX only.")
```

新main的SimulationContext建立 `backend_manager` 实例；旧 `physics_manager` 属性返回该实例。实例没有此处依赖的 `__name__`。本轮抽出实际表达式，用普通Python按新版对象形态复现AttributeError；不是物理运行。面向新版候选为：

```python
from isaaclab_physx.physics import PhysxManager

if not isinstance(env.sim.backend_manager, PhysxManager):
    raise ValueError("RandomizeTitaMaterial supports Isaac Sim PhysX only.")
```

由用户写入；摩擦/恢复系数公式无需随此判断修改。结果见[result.json](result.json)。

### 共享环境的同步入口

TITA_Lab/pyproject.toml 的uv overrides仍固定Torch2.12、torchvision0.27、Newton1.6；新main固定Torch2.11、torchvision0.26、Newton1.8.0.dev0、Sim6.2。不能直接在TITA目录 `uv sync` 并期待沿用新Lab那套依赖。

共享环境由新Lab源码的配置负责，在 `/home/eaziel/IsaacLab` 设对应 `UV_PROJECT_ENVIRONMENT` 后按官方流程同步，并选择 `isaacsim`、`rsl-rl` extras。再单独注册TITA：

```bash
uv pip install --python /home/eaziel/my_venvs/lab3_sim62/.venv/bin/python \
  --no-deps -e /mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/TITA_Lab
```

这样基础依赖由新Lab提供，不触发TITA旧依赖求解；不代表全部元数据兼容已获运行验证。以后若让TITA自己负责uv同步，须由用户更新其配置。保留Lab安装位置 `/home/eaziel/IsaacLab` 能避免当前本地sources路径失效。

日常激活后直接运行python/isaacsim/isaaclab即可。`uv sync`默认精确同步，会移除清单外包；另行editable注册的TITA需明确维护。`uv run`默认也会先检查/同步。不要从多个工程对同一环境随意同步不同依赖清单。

### 已有控制与数据接口

- 新main仍有 `sim.use_newton_actuators=False`，PhysX适配器保留Python执行器路径；ActuatorCollection仍调用各组 `compute(control_action, joint_pos, joint_vel)`，TitaPD公式与延迟有对应接入位置。
- IdealPD内部为 `_clip_effort` 增传joint_vel，但基类接受 `*args, **kwargs`；TitaPD当前只传effort不能据此判错。
- 普通PhysX decimation仍逐物理步apply/write/step/update，策略计数、termination/reward、reset、command/event、observation主要顺序仍有对应位置。读回刷新与reset实现有变化，须复测。
- RewardManager `_step_reward`仍为乘权重、未乘dt的每项值，合计再乘dt；TITA失败项取列乘step_dt的语义仍对应。归并实现改变，不能承诺浮点逐位相同。
- TITA当前接触cost、终止、奖励和特权观测显式使用 `net_normal_forces_w`，新main仍提供它。Sim6.2增加总力/摩擦力能力，不在升级时静默更改当前力定义；若以后改回旧Gym总力，单独验证。

## Python与ROS边界

```text
/usr/bin/python3                              Ubuntu/ROS系统Python
/home/eaziel/PythonRuntimes/...                uv独立Python3.12.x
/home/eaziel/my_venvs/lab3_sim62/.venv/         Sim/Lab共享环境
/home/eaziel/IsaacLab/                         新Lab源码
```

3.12.3与3.12.15/16是同3.12系列的补丁版，通常兼容，但不代表任意第三方二进制、路径和系统配置无条件兼容。保留系统Python及apt管理关系，uv另装新版Python并据此建venv；不要基于将删除的Conda Python建venv。ROS继续使用其系统解释器，不把uv独立Python当成apt ROS的已验证替代。

本轮Python官方页显示3.12.16已发布；uv可下载清单随uv版本而定，未验证当前uv是否已提供该构建。`uv python install 3.12`选择该uv支持的最新3.12补丁版，安装后核对实际版本。Ubuntu会回移安全补丁；不能只凭3.12.3字面判断补丁状态。本轮未升级系统。

## 新环境验收范围

官方空场景→TITA资产加载→五组观测33/330/187/30/6与8动作/6cost→非零动作、延迟、部分reset、15s推扰、20s超时、课程→actor/wrapper原CPU对照。完整NP3O训练仍未接完，不能作为这次已经通过的验收项。新环境运行前，原有证据保留其原版本标签。

## 来源与证据

本机与main目标文件52份比较、SHA256见[upstream_comparison.json](upstream_comparison.json)；[保护源码哈希](protected_source_hashes.json)记录本轮读取内容。main链接未锁定提交，哈希只标识本轮内容。本次未读Git元数据。

- [用户指定安装页](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html#installation-method-uv)、[main依赖](https://github.com/isaac-sim/IsaacLab/blob/main/pyproject.toml)
- [SimulationContext](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sim/simulation_context.py)、[接触数据](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor_data.py)
- [Sim6.2发布说明](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html)
- [uv Python](https://docs.astral.sh/uv/concepts/python-versions/)、[uv同步](https://docs.astral.sh/uv/concepts/projects/sync/)
- [ROS2 Jazzy Python说明原文](https://github.com/ros2/ros2_documentation/blob/jazzy/source/How-To-Guides/Using-Python-Packages.rst)
- [Python3.12.16](https://www.python.org/downloads/release/python-31216/)、[Ubuntu安全更新](https://documentation.ubuntu.com/security/security-updates/)
