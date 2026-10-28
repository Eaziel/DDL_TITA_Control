"""Check documented cost proposals with native managers and native CPU step/reset flow."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TASK = ROOT / "TITA_Lab/src/TITA_Lab/tasks/velocity"
sys.path.insert(0, str(ROOT / "TITA_Lab/src"))

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass


def module(name, source, path):
    result = ModuleType(name)
    sys.modules[name] = result
    exec(compile(source, str(path), "exec"), result.__dict__)
    return result


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


document = ROOT / "docs/migration/17_cost收集与加速度.md"
blocks = re.findall(r"```python\n(.*?)\n```", document.read_text(), re.S)
assert len(blocks) == 4
cost_source = (TASK / "mdp/costs.py").read_text()
# 用户当前五函数不改动；在内存中追加本轮候选加速度类。
cost_source = cost_source.replace("from typing import TYPE_CHECKING", "from collections.abc import Sequence\nfrom typing import TYPE_CHECKING")
cost_source = cost_source.replace("import torch", "import torch\n\nfrom isaaclab.managers import ManagerTermBase", 1)
cost_source = cost_source.replace("from isaaclab.managers import SceneEntityCfg", "from isaaclab.managers import ManagerTermBaseCfg, SceneEntityCfg", 1)
cost_source += "\n\n" + blocks[0] + "\n"
C = module("tita_costs_checked_candidate", cost_source, document)
M = module("tita_cost_manager_checked_candidate", blocks[1], document)
from TITA_Lab.tasks.velocity.config.TITA import env_cfg as CURRENT_CFG
cfg_namespace = {
    "configclass": configclass, "CostTerm": M.CostTermCfg, "mdp": C,
    "SceneEntityCfg": SceneEntityCfg, "TITA_JOINT_NAMES": CURRENT_CFG.TITA_JOINT_NAMES,
}
exec(compile(blocks[2], str(document), "exec"), cfg_namespace)
CostsCfg = cfg_namespace["CostsCfg"]
cfg = CURRENT_CFG.VelocityEnvCfg()
cfg.costs = CostsCfg()
cfg.validate()

# 保留历史检查文件，内存中复用原生step/reset夹具并改为当前观测/动作实现。
fixture_path = ROOT / "docs/audits/2026-10-20_privileged_state/check_state.py"
fixture = fixture_path.read_text().split("# Extract the source Gym contact update")[0]
replacements = [
    ("A=load('tita_pending_lag_actions',ROOT/'docs/audits/2026-10-20_action_lag/proposal_actions.py')",
     "from TITA_Lab.tasks.velocity.mdp import actions as A"),
    ("S=load('tita_privileged_state_candidate',OUT/'proposal_state.py')",
     "from TITA_Lab.tasks.velocity.mdp import observations as S"),
    ("P=load('tita_push_capture_candidate',OUT/'proposal_push.py')",
     "from TITA_Lab.tasks.velocity.mdp import events as P"),
    ("G=load('tita_privileged_group_candidate',OUT/'proposal_cfg.py')",
     "from TITA_Lab.tasks.velocity.config.TITA import env_cfg as CC\nG=NS(make_group=lambda cls: CC.ObservationsCfg().privileged_state.copy())"),
    ("L=load('tita_layout_candidate',OUT/'proposal_layout.py')", "L=None"),
    ("self.proprio_history=NS(reset=lambda ids:None)",
     "self.proprio_history=NS(reset=lambda ids:None,read_then_push=lambda *args:None)"),
    ("self.reward_manager=RewardManager(NS(zero=RewardTermCfg(func=zero_reward,weight=1.)),self)",
     "self.reward_manager=RewardManager(NS(termination=RewardTermCfg(func=zero_reward,weight=1.)),self)"),
]
for old, new in replacements:
    assert old in fixture, old
    fixture = fixture.replace(old, new, 1)
F = {"__file__": str(fixture_path), "__name__": "cost_native_fixture"}
exec(compile(fixture, str(fixture_path), "exec"), F)
BaseFixture = F["CpuEnv"]
# 候选_reset_idx自身负责观测和cost钩子，父类直接调用原生reset。
BaseFixture._reset_idx = ManagerBasedRLEnv._reset_idx
N = F["N"]
proxy = F["proxy"]
old_asset_init = F["Asset"].__init__


def asset_init(self, env):
    old_asset_init(self, env)
    self.data.joint_pos = proxy(torch.zeros(N, 8))
    self.data.joint_vel = proxy(torch.zeros(N, 8))
    self.data.joint_vel_limits = proxy(torch.tensor([25.] * 6 + [20.] * 2).repeat(N, 1))
    self.data.joint_effort_limits = proxy(torch.tensor([60.] * 6 + [15.] * 2).repeat(N, 1))
    self.data.soft_joint_pos_limits = proxy(torch.tensor([-1., 1.]).repeat(N, 8, 1))
    self.actuators = NS(applied_effort=proxy(torch.zeros(N, 8)))


F["Asset"].__init__ = asset_init
old_reset = F["reset_velocity"]


def reset_velocity(env, env_ids):
    old_reset(env, env_ids)
    asset = env.scene["robot"]
    asset.data.joint_vel.torch[env_ids, :] = 0.0
    asset.data.joint_pos.torch[env_ids, :] = 0.0
    asset.actuators.applied_effort.torch[env_ids, :] = 0.0


F["reset_velocity"] = reset_velocity


def values(step):
    q = torch.zeros(N, 8)
    q[step % N, 0] = 1.2
    v = torch.full((N, 8), 20.0 if step % 2 else -20.0)
    v[0, 0] = 26.0
    tau = torch.zeros(N, 8)
    tau[1, -1] = 16.0
    forces = torch.zeros(N, 3, 3)
    forces[0, 2, 2] = 501.0
    forces[2, 0] = torch.tensor([101., 0., 20.])
    return q, v, tau, forces


old_sim_step = F["Sim"].step


def sim_step(self, render=False):
    old_sim_step(self, render)
    q, v, tau, forces = values((self.counter - 1) // 4 + 1)
    asset = self.env.scene["robot"]
    asset.data.joint_pos.torch.copy_(q)
    asset.data.joint_vel.torch.copy_(v)
    asset.actuators.applied_effort.torch.copy_(tau)
    self.env.scene["contact_forces"].data.net_normal_forces_w.torch.copy_(forces)


F["Sim"].step = sim_step
env_source = blocks[3].replace("from .cost_manager import TitaCostManager", "TitaCostManager = checked_manager.TitaCostManager")
env_source = env_source.replace("from .mdp.observation_history import ProprioHistory", "ProprioHistory = None")
env_source = env_source.replace("class TitaVelocityEnv(ManagerBasedRLEnv):", "class TitaVelocityEnv(BaseFixture):")
E = ModuleType("tita_cost_env_checked_candidate")
E.checked_manager = M
E.BaseFixture = BaseFixture
sys.modules[E.__name__] = E
exec(compile(env_source, str(document), "exec"), E.__dict__)


def cpu_init(self, final_obs=False):
    BaseFixture.__init__(self, final_obs)
    self.cfg.costs = CostsCfg()
    self.cost_manager = M.TitaCostManager(self.cfg.costs, self)


E.TitaVelocityEnv.__init__ = cpu_init

# 从旧源码抽取六个真实方法，不重新手写期望公式。
old_source = Path("/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py").read_text()
names = ["acc_smoothness", "dof_vel_limits", "feet_contact_forces", "pos_limit", "stumble", "torque_limit"]
nodes = [node for node in ast.walk(ast.parse(old_source))
         if isinstance(node, ast.FunctionDef) and node.name in {"_cost_" + name for name in names}]
old_functions = {"torch": torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "original_gym_costs", "exec"), old_functions)


def exercise(final_obs):
    env = E.TitaVelocityEnv(final_obs)
    manager = env.cost_manager
    assert manager.active_terms == names
    assert isinstance(manager._term_cfgs[0].func, ManagerTermBase)
    torch.testing.assert_close(manager.k_values, torch.tensor([[.1, .3, .1, .3, .1, .3]]))
    assert manager.d_values.shape == (1, 1, 6) and manager.d_values.eq(0).all()
    env._reset_idx(torch.arange(N))
    assert manager.cost_buf.eq(0).all() and manager._episode_sums.eq(0).all()
    env.obs_buf = env.observation_manager.compute()
    previous_v = torch.zeros(N, 8)
    expected_sums = torch.zeros(N, 6)
    snapshots = []
    maximum_error = 0.0
    for step in range(1, 13):
        q, v, tau, forces = values(step)
        # 对齐fixture物理顺序即可：六项都对关节求any/mean，不依赖排列。
        old = NS(dof_pos=q, dof_vel=v, last_dof_vel=previous_v,
                 dof_vel_limits=env.scene["robot"].data.joint_vel_limits.torch[0],
                 torque_limits=env.scene["robot"].data.joint_effort_limits.torch[0],
                 dof_pos_limits=env.scene["robot"].data.soft_joint_pos_limits.torch[0],
                 torques=tau, contact_forces=forces, feet_indices=[2, 0], dt=.01,
                 cfg=NS(rewards=NS(soft_dof_vel_limit=1., soft_torque_limit=1., max_contact_force=500.)))
        expected = torch.stack([old_functions["_cost_" + name](old) * .01 for name in names], dim=1)
        expected_sums += expected
        _, _, terminated, _, extras = env.step(torch.zeros(N, 8))
        torch.testing.assert_close(extras["costs"], expected)
        maximum_error = max(maximum_error, (extras["costs"] - expected).abs().max().item())
        selected = terminated.nonzero().flatten()
        previous_v = v.clone()
        if len(selected):
            for i, name in enumerate(names):
                torch.testing.assert_close(extras["log"]["Episode_Cost/" + name], expected_sums[selected, i].mean() / 20.)
            expected_sums[selected, :] = 0
            previous_v[selected, :] = 0
        torch.testing.assert_close(manager._episode_sums, expected_sums)
        for _ in range(3):
            torch.testing.assert_close(manager.compute(), expected)
            torch.testing.assert_close(manager._episode_sums, expected_sums)
        snapshots.append((extras["costs"], expected.clone()))
    for actual, expected in snapshots:
        torch.testing.assert_close(actual, expected)
    # 非step的部分reset：不重复累计，也不清掉其他环境累计。
    saved = manager.cost_buf.clone()
    before = manager._episode_sums.clone()
    env._reset_idx(torch.tensor([0]))
    torch.testing.assert_close(manager.cost_buf, saved)
    torch.testing.assert_close(manager._episode_sums[1:], before[1:])
    assert manager._term_cfgs[0].func._previous_joint_vel[0].eq(0).all()
    manager.reset(None)
    assert manager._episode_sums.eq(0).all()
    env.close()
    return {"final_obs_enabled": final_obs, "policy_steps": 12, "physics_steps": 48,
            "partial_resets": [2, 5, 9], "push_steps": env.push_steps,
            "maximum_error_against_six_gym_methods": maximum_error,
            "cold_reset_repeated_compute_partial_reset_and_snapshot_checks": True}


results = [exercise(False), exercise(True)]
# 对实际五函数再做原64环境边界检查；不改历史证据，结果写本目录。
contact_path = ROOT / "docs/audits/2026-10-21_contact_costs/check.py"
contact_source = contact_path.read_text()
contact_source = contact_source.replace("ns['wheel_contact_force_cost']", "current.wheel_contact_force_cost")
contact_source = contact_source.replace("ns['wheel_stumble_cost']", "current.wheel_stumble_cost")
contact_source = contact_source.replace("Current three joint functions plus two documented contact candidates, CPU state/sensor stand-ins", "Current five saved cost functions, CPU state/sensor stand-ins")
contact_source = contact_source.replace("(OUT / 'result.json').write_text", f"(Path({str(OUT / 'current_five_result.json')!r})).write_text")
exec(compile(contact_source, str(contact_path), "exec"), {"__file__": str(contact_path), "__name__": "current_five"})

ruff = "/home/eaziel/.vscode/extensions/charliermarsh.ruff-2026.88.0-linux-x64/bundled/libs/bin/ruff"
lint_results = []
for name, source in [("mdp/costs.py", cost_source), ("cost_manager.py", blocks[1]),
                     ("tita_velocity_env.py", blocks[3])]:
    run = subprocess.run([ruff, "check", "--no-cache", "--no-respect-gitignore", "--config",
                          str(ROOT / "TITA_Lab/pyproject.toml"), "--stdin-filename", str(TASK / name), "-"],
                         input=source, text=True, capture_output=True)
    lint_results.append({"file": name, "exit_code": run.returncode, "output": run.stdout + run.stderr})
    assert run.returncode == 0, lint_results[-1]
result = {"scope": "Five current cost functions plus documented acceleration/manager/cfg/env candidates",
          "configuration_validate": True, "cost_column_names": names, "runs": results,
          "ruff": lint_results, "simulation_or_training_started": False,
          "boundary": "Native ManagerBase entity resolution and native RL step/reset, CPU state and simulator stand-ins; proposed constructor placement checked against source, not real env construction"}
(OUT / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps(result, indent=2, ensure_ascii=False))
