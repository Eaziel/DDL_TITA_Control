"""Check the TITA actor and environment interface."""

import argparse

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import resolve_task_config

import TITA_Lab.tasks  # noqa: F401
from TITA_Lab.learning.np3o.actor import TitaHistoryActor
from TITA_Lab.learning.np3o.env_wrapper import TitaNP3OEnvWrapper


def run_check(env: TitaNP3OEnvWrapper, max_steps: int) -> dict[str, int]:
    """Check actor outputs and count completed environment steps."""
    # 当前网络随机初始化；这里只验证已有模块能否连通。
    actor = TitaHistoryActor().to(env.device)
    actor.eval()
    obs = env.reset()
    assert obs.shape == (env.num_envs, 586)
    assert torch.isfinite(obs).all(), "Non-finite observation."
    completed = resets = timeouts = 0

    with torch.inference_mode():
        for _ in range(max_steps):
            if not env.env.unwrapped.sim.is_running():
                break
            # 586列：policy33、scan187、state6、dynamics30、history330。
            proprio = obs[:, :33]
            history = obs[:, 256:586].reshape(env.num_envs, 10, 33)
            actions = actor(proprio, history)  # (N, 8)，网络输出的动作均值。
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
            if "time_outs" in infos:
                timeouts += int(infos["time_outs"].sum().item())
            if completed == 1:
                print(f"[CHECK] obs={tuple(obs.shape)} actions={tuple(actions.shape)} costs={tuple(costs.shape)}")

    # steps是全局策略步数；resets/timeouts是各环境事件次数之和。
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
    # 新Lab启动接口负责启动与关闭Kit；环境仍由gym.make创建。
    with launch_simulation(cfg, args):
        native_env = gym.make(task, cfg=cfg)
        try:
            run_check(TitaNP3OEnvWrapper(native_env), args.max_steps)
        finally:
            native_env.close()


if __name__ == "__main__":
    main()
