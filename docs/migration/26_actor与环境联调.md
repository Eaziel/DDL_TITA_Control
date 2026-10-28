# 26 把现有 actor 接到环境

内容纯AI生成，由人类要求生成，未经人类修改。核对：2026-10-28。

## 1. 这次要得到什么

让已经保存的环境、包装器和动作网络真正连起来：环境返回586列观测，取出当前33维和历史330维交给actor，actor输出8个动作，再由环境执行一步。重复200次，检查张量形状、有限值和回合结束计数。

网络随机初始化，本次不计算辅助损失、不执行反向传播或参数更新。它不会表现为训练好的平衡控制器。通过只说明这条交互路径在这次运行中接通。

本页提供**由用户写入的参考代码**，文档本身不证明脚本已经保存或运行成功。用户完成本次联调后暂停新增内容，阅读wrapper和actor说明。

## 2. 写入哪个文件

在项目中创建 `Tita_Lab/scripts/check_np3o_interface.py`。它直接导入已有的wrapper和actor，不新增MDP项，不修改 `.pyi`、`env_cfg.py` 或任务注册。

旧草稿的三个包名必须统一为 `Tita_Lab`；本页已经使用当前名字。

```python
"""Check the TITA actor and environment interface."""

import argparse

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config

import Tita_Lab.tasks  # noqa: F401
from Tita_Lab.learning.np3o.actor import TitaHistoryActor
from Tita_Lab.learning.np3o.env_wrapper import TitaNP3OEnvWrapper


def run_check(env: TitaNP3OEnvWrapper, max_steps: int) -> dict[str, int]:
    """Check actor outputs and count completed environment steps."""
    # 使用随机初始化的现有网络，设备与环境相同；eval只切换网络模式。
    actor = TitaHistoryActor().to(env.device)
    actor.eval()
    obs = env.reset()  # (N,586)，N是并行环境数。
    assert obs.shape == (env.num_envs, 586)
    assert torch.isfinite(obs).all(), "Non-finite observation."

    # completed统计step调用次数；后两项统计所有环境的回合结束次数。
    completed = resets = timeouts = 0
    with torch.inference_mode():
        for _ in range(max_steps):
            if not env.env.unwrapped.sim.is_running():
                break

            # 586列顺序：policy33、scan187、state6、dynamics30、history330。
            proprio = obs[:, :33]  # (N,33)，当前策略观测。
            history = obs[:, 256:586].reshape(env.num_envs, 10, 33)
            actions = actor(proprio, history)  # (N,8)，动作均值。
            assert actions.shape == (env.num_envs, 8)
            assert torch.isfinite(actions).all(), "Non-finite action."

            obs, _, rewards, costs, dones, infos = env.step(actions)
            assert obs.shape == (env.num_envs, 586)
            assert torch.isfinite(obs).all(), "Non-finite observation."
            assert costs.shape == (env.num_envs, env.num_cost)
            assert torch.isfinite(rewards).all(), "Non-finite reward."
            assert torch.isfinite(costs).all(), "Non-finite cost."

            completed += 1
            resets += int(dones.sum().item())
            # infos是本步的附加信息；time_outs表示因回合时长到达上限而结束。
            if "time_outs" in infos:
                timeouts += int(infos["time_outs"].sum().item())
            if completed == 1:
                print(f"[CHECK] obs={tuple(obs.shape)} actions={tuple(actions.shape)} costs={tuple(costs.shape)}")

    result = {"steps": completed, "requested": max_steps, "resets": resets, "timeouts": timeouts}
    print(f"[CHECK] {result}")
    return result


def main() -> None:
    """Launch the TITA task and check its actor interface."""
    parser = argparse.ArgumentParser(description="Check the saved TITA actor with a live environment.")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=200)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_steps < 1:
        parser.error("num_envs and max_steps must be positive.")

    task = "TitaLab-Velocity-Tita"
    cfg, _ = resolve_task_config(task, None, overrides=["physics=isaacsim_physx"])
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    torch.manual_seed(42)
    cfg.validate()

    # 启动接口负责Kit的生命周期；gym.make按已注册入口建立TitaVelocityEnv。
    with launch_simulation(cfg, args):
        native_env = gym.make(task, cfg=cfg)
        try:
            run_check(TitaNP3OEnvWrapper(native_env), args.max_steps)
        finally:
            native_env.close()


if __name__ == "__main__":
    main()
```

## 3. 数据怎样走完一圈

| 语句 | 输入与结果 |
| --- | --- |
| `env.reset()` | wrapper调用Lab重置，得到五组观测，再拼成 `(N,586)` |
| `obs[:, :33]` | 取当前33维观测 |
| `obs[:, 256:586].reshape(N,10,33)` | 取过去十帧，恢复时间轴 |
| `actor(proprio, history)` | 调用现有 `forward()`，返回 `(N,8)` |
| `env.step(actions)` | wrapper调用Lab；Lab内四次物理步后返回六项训练接口数据 |
| 下一次循环 | 使用刚返回的 `obs`，不重新创建网络或重算上一帧 |

`256 = 33 + 187 + 6 + 30`。动作网络只读取当前与历史观测；这里没有把全部586列直接塞进actor。环境自动重置结束的回合，脚本继续统计其余步数。

## 4. 怎样运行和判断结果

写好文件后由用户运行：

```bash
source "$HOME/my_venvs/lab3_sim62/.venv/bin/activate"
cd /mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR/Tita_Lab

python scripts/check_np3o_interface.py \
  --num_envs 4 --max_steps 200 --device cuda:0 --viz kit
```

首步应打印 `obs=(4,586)`、`actions=(4,8)`、`costs=(4,6)`，结尾 `steps` 应等于 `requested=200`，且没有断言或Traceback。手动提前关窗口时，`steps`可能小于200，按提前结束记录。

`resets`统计四个环境全部回合结束次数之和，可以大于单个环境的次数。`timeouts=0`允许：200策略步只有2秒仿真，当前回合上限为20秒。随机网络摔倒并重置也可能发生。轮子卡入地面的现象仍需单独核实，shape检查通过不能证明接触物理正确。

## 5. 与当前官方入口的对应

当前官方simple agent同样先解析任务配置，在 `launch_simulation()` 中建立环境，再通过 `sim.is_running()` 与步数预算控制循环。本页只把官方随机动作来源换成现有actor的前向结果。

依据：[启动API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.app.html)、[本地simple_agents.py](/home/eaziel/IsaacLab-main/source/isaaclab_rl/isaaclab_rl/entrypoints/simple_agents.py)、[本地配置解析](/home/eaziel/IsaacLab-main/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py)。本轮核对源码与脚本语法，没有运行Kit、GPU交互或训练。
