"""Inspect saved fixes and the legacy policy interface without running simulation or training."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib
import io
import json
import math
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
GYM = Path('/home/eaziel/Downloads/tita_rl-master')
sys.path.insert(0, str(ROOT/'TITA_Lab/src'))
RESULT = {'boundary': 'CPU configuration and random-input neural forward only; no simulator, optimizer, or training'}

import TITA_Lab.tasks  # noqa: E402,F401
from TITA_Lab.assets.tita import TITA_CFG  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

cfg = parse_env_cfg('TitaLab-Velocity-Tita', device='cpu', num_envs=4,
                    overrides=['physics=isaacsim_physx'])
cfg.validate()
scanner = cfg.scene.height_scanner
assert scanner.prim_path == '{ENV_REGEX_NS}/Robot/Geometry/base_link'
assert scanner.spawn is None
for actuators in (TITA_CFG.actuators, cfg.scene.robot.actuators):
    assert actuators['legs'].joint_velocity_limit == 25.
    assert actuators['wheels'].joint_velocity_limit == 20.
body = TITA_CFG.spawn.rigid_props['/.*'][0]
assert body.linear_damping == 0. and body.angular_damping == 0.
assert body.max_linear_velocity == 1000.
assert math.isclose(body.max_angular_velocity, math.degrees(1000.))
action_path = ROOT/'TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/actions.py'
tree = ast.parse(action_path.read_text())
action_cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TitaJointAction')
apply = next(n for n in action_cls.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_actions')
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == 'apply_actions' for n in ast.walk(apply))
RESULT['saved_fixes'] = {'F1': True, 'F2': True, 'F3': True, 'A1': True,
                         'real_current_config_validate': True}
RESULT['saved_source_sha256'] = {
    str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (action_path, ROOT/'TITA_Lab/src/TITA_Lab/assets/tita.py',
              ROOT/'TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py')
}

# Skip the legacy modules/__init__ import fan-out; load its real network implementation.
# The three imported network files are unmodified PyTorch/NumPy modules.
package = types.ModuleType('modules')
package.__path__ = [str(GYM/'modules')]
sys.modules['modules'] = package
legacy = importlib.import_module('modules.actor_critic')
torch.set_num_threads(2)
torch.manual_seed(20261020)
with contextlib.redirect_stdout(io.StringIO()):
    policy = legacy.ActorCriticBarlowTwins(
        num_prop=33, num_scan=187, num_critic_obs=586, num_priv_latent=36,
        num_hist=10, num_actions=8, scan_encoder_dims=[128,64,32],
        actor_hidden_dims=[512,256,128], critic_hidden_dims=[512,256,128],
        activation='elu', init_noise_std=1., priv_encoder_dims=[],
        num_costs=6, teacher_act=True, imi_flag=True,
    )
obs = torch.randn(8, 586)
with torch.no_grad():
    actions = policy.act(obs)
    reward_value = policy.evaluate(obs)
    cost_values = policy.evaluate_cost(obs)
    auxiliary = policy.imitation_learning_loss(obs)
assert actions.shape == (8,8)
assert reward_value.shape == (8,1)
assert cost_values.shape == (8,6)
assert all(torch.isfinite(t).all() for t in (actions,reward_value,cost_values,auxiliary))
RESULT['legacy_network_forward'] = {
    'observation_shape': list(obs.shape), 'action_shape': list(actions.shape),
    'reward_value_shape': list(reward_value.shape), 'cost_value_shape': list(cost_values.shape),
    'observation_parts_in_order': [33,187,36,330],
    'barlow_auxiliary_loss_finite': True,
    'scope': 'Unmodified legacy network methods with random CPU input; no NP3O update or quality claim',
}

(OUT/'result.json').write_text(json.dumps(RESULT, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(RESULT, indent=2, ensure_ascii=False))
