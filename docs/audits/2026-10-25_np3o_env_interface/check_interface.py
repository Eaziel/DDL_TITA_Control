"""Check the proposed interface with CPU tensors and an environment fixture."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import gymnasium as gym
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PACK = ROOT / "TITA_Lab/src/TITA_Lab/tasks/velocity/observation_layout.py"
PROPOSAL = HERE / "proposal_env_wrapper.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 绕过tasks/__init__.py的仿真注册导入；拼接函数本身来自用户当前文件。
for name in ("TITA_Lab", "TITA_Lab.tasks", "TITA_Lab.tasks.velocity"):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
load_module("TITA_Lab.tasks.velocity.observation_layout", PACK)
Wrapper = load_module("candidate_np3o_env_wrapper", PROPOSAL).TitaNp3oEnvWrapper


class EnvFixture(gym.Env):
    def __init__(self):
        self.num_envs = 4
        self.device = "cpu"
        self.max_episode_length = 2000
        self.action_manager = types.SimpleNamespace(total_action_dim=8)
        self.episode_length_buf = torch.zeros(4, dtype=torch.long)
        self.cfg = types.SimpleNamespace(is_finite_horizon=False)
        self.cost_manager = types.SimpleNamespace(
            active_terms=["acc_smoothness", "dof_vel_limits", "feet_contact_forces",
                          "pos_limit", "stumble", "torque_limit"],
            k_values=torch.tensor([[0.1, 0.3, 0.1, 0.3, 0.1, 0.3]]),
            d_values=torch.zeros(1, 1, 6),
            cost_buf=torch.zeros(4, 6),
        )
        self.obs_buf = {}
        self.reset_calls = 0
        self.step_calls = 0

    def reset(self, *, seed=None, options=None):
        self.reset_calls += 1
        self.obs_buf = {
            name: torch.full((4, size), float(index))
            for index, (name, size) in enumerate(
                (("policy", 33), ("scan", 187), ("privileged_state", 6),
                 ("dynamics", 30), ("history", 330)), start=1
            )
        }
        self.obs_buf["policy"][0, 0] = 123.0
        return self.obs_buf, {}

    def step(self, actions):
        self.step_calls += 1
        self.received_actions = actions
        # 夹具令当前cost缓冲为零，但返回另一份结束步cost，检查取值来源。
        self.cost_manager.cost_buf.zero_()
        costs = torch.arange(24, dtype=torch.float32).view(4, 6) / 100
        self.extras = {"costs": costs, "log": {"Episode_Reward/test": torch.tensor(2.0)}}
        terminated = torch.tensor([False, True, False, True])
        truncated = torch.tensor([False, False, True, True])
        if self.step_calls > 1:
            terminated.zero_()
            truncated.zero_()
        return self.obs_buf, torch.arange(4, dtype=torch.float32), terminated, truncated, self.extras

    def close(self):
        self.closed = True


env = EnvFixture()
wrapped = Wrapper(env)
assert env.reset_calls == 0
obs = wrapped.reset()
assert env.reset_calls == 1 and obs.shape == (4, 586)
assert obs[0, 0] == 100 and env.obs_buf["policy"][0, 0] == 123
for start, stop, value in ((0, 33, 1), (33, 220, 2), (220, 226, 3),
                           (226, 256, 4), (256, 586, 5)):
    assert torch.all(obs[1, start:stop] == value)
assert torch.equal(wrapped.get_observations(), obs) and env.reset_calls == 1
assert wrapped.get_privileged_observations() is None
assert (wrapped.num_envs, wrapped.num_obs, wrapped.num_actions, wrapped.num_costs) == (4, 586, 8, 6)
assert wrapped.cost_k_values is env.cost_manager.k_values
assert wrapped.cost_d_values_tensor.shape == (1, 1, 6)

actions = torch.full((4, 8), 3.5)
result = wrapped.step(actions)
packed, privileged, rewards, costs, dones, infos = result
assert len(result) == 6 and packed.shape == (4, 586) and privileged is None
assert env.received_actions is actions and torch.all(actions == 3.5)
assert env.step_calls == 1 and env.reset_calls == 1
assert rewards.shape == (4,) and costs is env.extras["costs"]
assert torch.equal(costs, torch.arange(24).view(4, 6) / 100)
assert torch.equal(dones, torch.tensor([0, 1, 1, 1])) and dones.dtype == torch.long
assert torch.equal(infos["time_outs"], torch.tensor([False, False, True, True]))
assert infos["episode"] == env.extras["log"] and infos["episode"] is not env.extras["log"]
assert infos["depth"] is None
assert set(env.extras) == {"costs", "log"}
assert "episode" not in wrapped.step(actions)[-1]

env.cfg.is_finite_horizon = True
assert "time_outs" not in wrapped.step(actions)[-1]
lengths = torch.tensor([5, 6, 7, 8])
wrapped.episode_length_buf = lengths
assert wrapped.episode_length_buf is env.episode_length_buf is lengths
wrapped.close()
assert env.closed

result = {
    "date": "2026-10-25",
    "evidence": "CPU tensors and a synthetic environment; no simulator or optimizer",
    "proposal_sha256": hashlib.sha256(PROPOSAL.read_bytes()).hexdigest(),
    "current_pack_sha256": hashlib.sha256(PACK.read_bytes()).hexdigest(),
    "passed": [
        "five actual observation groups packed at correct offsets and clipped without source mutation",
        "constructor does not reset; reset and get_observations do not add a simulation step",
        "six-result step; rewards and pre-collected costs pass through without rescaling",
        "actions pass through without clipping or reordering",
        "failure, timeout, and simultaneous flags merged; infinite-horizon timeout retained",
        "finite-horizon timeout key omitted",
        "episode log alias only on done; native extras keys unchanged",
        "episode-length forwarding and close",
    ],
    "not_tested": [
        "native simulation and automatic-reset physics",
        "entire old runner compatibility or NP3O gradient update",
        "old checkpoint feature/joint order compatibility",
    ],
}
(HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
