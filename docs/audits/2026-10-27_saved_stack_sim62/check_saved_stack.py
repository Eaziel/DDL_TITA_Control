"""Check saved TITA action, wrapper, and actor code in the new Python environment."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import gymnasium as gym
import torch

from TITA_Lab.learning.np3o.env_wrapper import TitaNP3OEnvWrapper
from TITA_Lab.tasks.velocity.mdp import actions as action_module
from TITA_Lab.tasks.velocity.mdp import rewards as reward_module

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
torch.set_num_threads(2)

# 复用旧检查中的CPU缓存夹具，调用今天保存的实际动作类与奖励函数。
fir_check = ROOT / "docs/audits/2026-10-27_sim62_action_fir/check_fir.py"
helpers = [node for node in ast.parse(fir_check.read_text()).body if isinstance(node, ast.FunctionDef)]
scope = {"torch": torch, "reward_module": reward_module, "SimpleNamespace": types.SimpleNamespace}
exec(compile(ast.Module(body=helpers, type_ignores=[]), str(fir_check), "exec"), scope)
fir_trace = scope["trace"](action_module.TitaJointAction)
fir_grouped = scope["check_grouped"](action_module.TitaJointAction)
assert fir_trace == [8.0, 2.0, 80.0, -60.0]
assert fir_grouped["max_target_error"] == 0
assert not fir_grouped["all_action_rates_zero"]

# 只抽取旧脚本的环境夹具，包装器与拼接函数使用当前实际文件。
fixture_path = ROOT / "docs/audits/2026-10-25_np3o_env_interface/check_interface.py"
fixture_node = next(node for node in ast.parse(fixture_path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "EnvFixture")
fixture_scope = {"gym": gym, "torch": torch, "types": types, "__name__": "cpu_env_fixture"}
exec(compile(ast.Module(body=[fixture_node], type_ignores=[]), str(fixture_path), "exec"), fixture_scope)
EnvFixture = fixture_scope["EnvFixture"]
env = EnvFixture()
wrapped = TitaNP3OEnvWrapper(gym.Wrapper(env))
assert env.reset_calls == 0
obs = wrapped.reset()
assert obs.shape == (4, 586) and obs[0, 0] == 100
assert env.obs_buf["policy"][0, 0] == 123
for start, stop, value in ((0, 33, 1), (33, 220, 2), (220, 226, 3), (226, 256, 4), (256, 586, 5)):
    assert torch.all(obs[1, start:stop] == value)
assert torch.equal(wrapped.get_observations(), obs) and env.reset_calls == 1
assert wrapped.get_privileged_observations() is None
assert (wrapped.num_envs, wrapped.num_obs, wrapped.num_actions, wrapped.num_cost) == (4, 586, 8, 6)
assert wrapped.cost_k_values is env.cost_manager.k_values
assert wrapped.cost_d_values_tensor is env.cost_manager.d_values
actions = torch.full((4, 8), 3.5)
packed, privileged, rewards, costs, dones, infos = wrapped.step(actions)
assert packed.shape == (4, 586) and privileged is None
assert env.received_actions is actions
assert costs is env.extras["costs"] and rewards.shape == (4,)
assert torch.equal(dones, torch.tensor([0, 1, 1, 1]))
assert torch.equal(infos["time_outs"], torch.tensor([False, False, True, True]))
assert infos["episode"] == env.extras["log"] and infos["episode"] is not env.extras["log"]
assert set(env.extras) == {"costs", "log"}
assert infos["depth"] is None
assert "episode" not in wrapped.step(actions)[-1]
env.cfg.is_finite_horizon = True
assert "time_outs" not in wrapped.step(actions)[-1]
lengths = torch.tensor([5, 6, 7, 8])
wrapped.episode_length_buf = lengths
assert wrapped.episode_length_buf is env.episode_length_buf is lengths
wrapped.close()
assert env.closed

# 复用旧actor数值检查，将HERE/ROOT的输出定位改到本轮目录，保留旧证据。
actor_check = ROOT / "docs/audits/2026-10-26_saved_actor/check_saved_actor.py"
actor_tree = ast.parse(actor_check.read_text())
actor_dir = HERE / "actor"
actor_dir.mkdir(exist_ok=True)
for node in actor_tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in {"HERE", "ROOT"} for target in node.targets):
        name = node.targets[0].id
        value = actor_dir if name == "HERE" else ROOT
        node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(value))], keywords=[])
ast.fix_missing_locations(actor_tree)
exec(compile(actor_tree, str(actor_check), "exec"), {"__name__": "saved_actor_sim62_audit", "__file__": str(actor_check)})
actor_result = json.loads((actor_dir / "result.json").read_text())
actor_result["date"] = "2026-10-27"
actor_result["reused_check"] = str(actor_check)
(actor_dir / "result.json").write_text(json.dumps(actor_result, indent=2) + "\n")

source_names = [
    "TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/actions.py",
    "TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/observations.py",
    "TITA_Lab/src/TITA_Lab/learning/np3o/env_wrapper.py",
    "TITA_Lab/src/TITA_Lab/learning/np3o/actor.py",
    "TITA_Lab/src/TITA_Lab/tasks/velocity/observation_layout.py",
]
result = {
    "date": "2026-10-27", "python": sys.executable, "torch": torch.__version__,
    "boundary": "Actual saved code on CPU; wrapper environment state is synthetic; no Kit, physics, optimizer, or training.",
    "current_fir_trace": fir_trace, "current_fir_grouped": fir_grouped,
    "current_wrapper_checks_passed": True,
    "actual_wrapper_cost_attribute": "num_cost",
    "actor_checks_passed": actor_result["arithmetic_checks_passed"],
    "actual_auxiliary_method": actor_result["actual_method_name"],
    "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_names},
}
(HERE / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
