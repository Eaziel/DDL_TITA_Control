# Ubuntu 安装新版 Isaac Lab 与完整 Isaac Sim：我的 uv 环境管理方式

整理日期：2026-10-27。本文以 **Ubuntu 24.04 x86_64、Bash、Isaac Lab main、Isaac Sim 6.2、Python 3.12** 为例，说明我采用的目录组织和安装步骤。main 会继续更新，后续安装时应核对仓库里的版本配置。

我采用官方的 uv 源码安装流程：由 uv 准备 Python 和虚拟环境，再按 Lab 的配置安装依赖。开始前，需要已有 Git、curl，以及满足[官方要求](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html)的 NVIDIA 显卡和驱动。

## 1. 我的目录偏好

**作者偏好：我把 Python 本体、虚拟环境、Lab 源码分开放。个人认为这样比较清晰，也方便多个机器人项目共用一套兼容的环境。**

```text
~/PythonRuntimes/                    uv 管理的 Python 本体
~/my_venvs/lab3_sim62/.venv/          Python 环境，安装 Torch、Sim 等包
~/IsaacLab-main/                     Lab main 源码
```

`~` 表示当前用户的主目录。目录名可以自行调整；环境建好后，我会保持这些路径稳定。

venv 不要求放在每个工程里面。激活这套环境后，位于其他目录的项目也可以使用它；共享环境里的依赖升级会同时影响这些项目。

我在普通系统终端中准备 uv 和独立的 Python 运行时，再创建本教程使用的 venv。系统 Python 与 ROS 环境保持独立。

## 2. 安装独立的 uv

我使用 uv 的独立安装器，无需先安装 Python，也无需 sudo：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

默认安装后，让当前终端加载它的路径，并检查版本：

```bash
source "$HOME/.local/bin/env"
uv --version
```

uv 是独立工具，通常位于 `~/.local/bin/uv`，不依附于某个 Conda 或 venv 环境。[uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)

## 3. 设置 Python 本体目录

我先指定 uv 把 Python 放到哪里：

```bash
export UV_PYTHON_INSTALL_DIR="$HOME/PythonRuntimes"
```

这句 `export` 对当前终端及其子进程生效。为了让以后新开的 Bash 终端也使用这个目录，我会打开：

```bash
nano ~/.bashrc
```

在末尾添加同一行并保存：

```bash
export UV_PYTHON_INSTALL_DIR="$HOME/PythonRuntimes"
```

随后安装 Python：

```bash
uv python install 3.12
uv python dir
uv python list --only-installed
```

第一句安装当前 uv 支持的最新 Python 3.12 补丁版；后两句查看安装目录和已有解释器。这里单独准备 Python 本体，还没有安装 Sim 或 Torch。[uv Python 管理说明](https://docs.astral.sh/uv/concepts/python-versions/)

我保留 Ubuntu 的 `/usr/bin/python3`，让系统和 ROS 2 继续使用原来的系统 Python。

## 4. 下载 Lab main 源码

```bash
cd "$HOME"
git clone https://github.com/isaac-sim/IsaacLab.git --branch main IsaacLab-main
cd "$HOME/IsaacLab-main"
```

我使用 `IsaacLab-main` 作为新目录名，便于与已有版本区分。后面的安装、同步命令都在这个源码目录中执行。

## 5. 创建共享环境，设置提示名称

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/my_venvs/lab3_sim62/.venv"

uv venv \
  --python 3.12 \
  --managed-python \
  --prompt lab3_sim62 \
  "$UV_PROJECT_ENVIRONMENT"
```

我把这里的参数分成四部分理解：

| 参数或变量 | 作用 |
| --- | --- |
| `UV_PROJECT_ENVIRONMENT` | 告诉 uv，这套 Lab 的环境放在哪里 |
| `--python 3.12` | 使用 Python 3.12 |
| `--managed-python` | 使用 uv 管理的 Python，与系统解释器分开 |
| `--prompt lab3_sim62` | 激活后显示 `(lab3_sim62)`，便于区分版本 |

官方流程可以让 `uv sync` 自动创建环境。我这里提前创建，是为了指定基础 Python 和提示名称。[uv 环境创建参数](https://docs.astral.sh/uv/reference/cli/#uv-venv)

两个环境变量的范围不同：

| 变量 | 我的设置方式 |
| --- | --- |
| `UV_PYTHON_INSTALL_DIR` | 写进 `.bashrc`，长期使用同一个 Python 本体目录 |
| `UV_PROJECT_ENVIRONMENT` | 只在这套 Lab 的终端设置，不写成所有项目的全局默认值 |

## 6. 安装 Lab、完整 Sim 和训练依赖

```bash
uv sync --extra isaacsim --extra rsl-rl
```

`uv sync` 按仓库中的 `pyproject.toml` 和 `uv.lock` 配置、同步环境。`--extra isaacsim` 加入完整 Sim，`--extra rsl-rl` 加入 RSL-RL 训练依赖。

**Sim 版本来自仓库配置，命令本身没有直接写版本号。** 本文整理时，main 的 `pyproject.toml` 包含：

```toml
"isaacsim[all,extscache]==6.2.0.0"
```

其中 `all,extscache` 表示 Sim 的完整主包组合及扩展缓存。可以直接查看[官方配置文件](https://github.com/isaac-sim/IsaacLab/blob/main/pyproject.toml)核对。

这一步也会安装配套 Torch，以及指向当前源码的 Lab 可编辑包。我不再额外执行 `pip install isaacsim`、手动安装另一套 Torch，或运行 `isaaclab.sh -i`。[Lab 官方 uv 安装流程](https://isaac-sim.github.io/IsaacLab/main/source/isaaclab/installation.html#installation-method-uv)

## 7. 检查版本，启动 Lab 和 Sim

先激活环境：

```bash
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
```

提示符应显示 `(lab3_sim62)`。我会先核对实际解释器和包版本：

```bash
python - <<'PY'
import sys
from importlib.metadata import version

print("Python 路径:", sys.executable)
print("Python 版本:", sys.version)
print("Isaac Sim:", version("isaacsim"))
print("Torch:", version("torch"))
PY
```

Python 路径应指向 `my_venvs/lab3_sim62/.venv/bin/python`，本文对应的 Sim 版本应为 `6.2.0.0`。

然后验证 Lab 空场景：

```bash
python scripts/tutorials/00_sim/create_empty.py --viz kit
```

关闭空场景后，再启动完整 Sim 界面：

```bash
isaacsim isaacsim.exp.full
```

首次启动若出现许可提示，阅读后按提示处理。空场景成功仅说明基础启动链路通过，自定义任务仍需单独验证。

## 8. 以后怎么用

新开终端后：

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/my_venvs/lab3_sim62/.venv"
source "$UV_PROJECT_ENVIRONMENT/bin/activate"
cd "$HOME/IsaacLab-main"
```

我习惯激活一次，然后直接运行命令：

```bash
# 打开 Sim
isaacsim

# 运行自己的脚本，替换为实际路径
python /path/to/my_script.py
```

中途切换工程目录，仍然使用刚才激活的环境。退出环境：

```bash
deactivate
```

**日常运行不需要反复执行 `uv sync`。** 我让这份 Lab 配置负责维护共享环境；其他工程使用这套环境运行。`uv sync` 默认会清理清单外的包，`uv run` 默认也会检查并在必要时同步，所以我不会从多个工程拿不同清单反复同步同一个共享环境。[uv 同步行为说明](https://docs.astral.sh/uv/concepts/projects/sync/)
