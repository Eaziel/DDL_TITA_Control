"""Check current joint costs and proposed contact costs against old Gym expressions."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
task = ROOT / 'TITA_Lab/src/TITA_Lab/tasks/velocity'
spec = importlib.util.spec_from_file_location('current_cost_functions', task / 'mdp/costs.py')
current = importlib.util.module_from_spec(spec)
spec.loader.exec_module(current)

document = (ROOT / 'docs/migration/16_接触cost迁移.md').read_text()
proposal = re.search(r'```python\n(.*?)\n```', document, re.S).group(1)
ns = {'torch': torch}
exec(compile('from __future__ import annotations\n' + proposal, 'contact_costs_proposal.py', 'exec'), ns)

gym_source = Path('/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py').read_text()
names = {'_cost_pos_limit', '_cost_torque_limit', '_cost_dof_vel_limits',
         '_cost_feet_contact_forces', '_cost_stumble'}
old = {'torch': torch}
nodes = [n for n in ast.walk(ast.parse(gym_source)) if isinstance(n, ast.FunctionDef) and n.name in names]
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'original_gym_costs.py', 'exec'), old)

N = 64
torch.manual_seed(20261021)
q = torch.randn(N, 8)
v = torch.randn(N, 8) * 30
tau = torch.randn(N, 8) * 60
limits = torch.tensor([-1., 1.]).repeat(N, 8, 1)
vl = torch.tensor([25.] * 6 + [20.] * 2).repeat(N, 1)
tl = torch.tensor([60.] * 6 + [15.] * 2).repeat(N, 1)
proxy = lambda x: NS(torch=x)
robot = NS(data=NS(joint_pos=proxy(q), joint_vel=proxy(v), soft_joint_pos_limits=proxy(limits),
    joint_vel_limits=proxy(vl), joint_effort_limits=proxy(tl)), actuators=NS(applied_effort=proxy(tau)))
forces = torch.randn(N, 5, 3) * 300
# 精确边界、无接触、侧碰撞、负Fz及非选中身体的大力。
forces[:7] = 0
forces[0, 3] = torch.tensor([0., 0., 500.])
forces[1, 1] = torch.tensor([0., 0., 501.])
forces[2, 3] = torch.tensor([100., 0., 20.])
forces[3, 1] = torch.tensor([101., 0., 20.])
forces[4, 3] = torch.tensor([120., 0., -20.])
forces[6, 0] = torch.tensor([10000., 0., 0.])
sensor = NS(data=NS(net_normal_forces_w=proxy(forces)))
scene = {'robot': robot}
class Scene(dict):
    @property
    def sensors(self):
        return self
env = NS(scene=Scene(robot=robot, contact_forces=sensor))
asset_cfg = NS(name='robot', joint_ids=[7, 2, 0])
sensor_cfg = NS(name='contact_forces', body_ids=[3, 1])
legacy = NS(dof_pos=q[:, asset_cfg.joint_ids], dof_vel=v[:, asset_cfg.joint_ids],
    torques=tau[:, asset_cfg.joint_ids], dof_pos_limits=limits[0, asset_cfg.joint_ids],
    dof_vel_limits=vl[0, asset_cfg.joint_ids], torque_limits=tl[0, asset_cfg.joint_ids],
    contact_forces=forces, feet_indices=sensor_cfg.body_ids,
    cfg=NS(rewards=NS(soft_dof_vel_limit=1., soft_torque_limit=1., max_contact_force=500.)))
mapping = [
    (current.joint_position_limit_cost, '_cost_pos_limit', asset_cfg),
    (current.joint_effort_limit_cost, '_cost_torque_limit', asset_cfg),
    (current.joint_velocity_limit_cost, '_cost_dof_vel_limits', asset_cfg),
    (ns['wheel_contact_force_cost'], '_cost_feet_contact_forces', sensor_cfg),
    (ns['wheel_stumble_cost'], '_cost_stumble', sensor_cfg),
]
errors = {}
for func, name, selection in mapping:
    actual = func(env, selection)
    expected = old[name](legacy)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (N,) and actual.dtype == torch.float32
    errors[func.__name__] = (actual - expected).abs().max().item()
assert ns['wheel_contact_force_cost'](env, sensor_cfg)[:7].tolist() == [0., 1., 0., 0., 0., 0., 0.]
assert ns['wheel_stumble_cost'](env, sensor_cfg)[:7].tolist() == [0., 0., 0., 1., 1., 0., 0.]

pyi = ast.parse((task / 'mdp/__init__.pyi').read_text())
exports = [n.name for node in pyi.body if isinstance(node, ast.ImportFrom) and node.module == 'costs' for n in node.names]
missing = [name for name in exports if not hasattr(current, name)]
result = {
    'scope': 'Current three joint functions plus two documented contact candidates, CPU state/sensor stand-ins',
    'environments': N, 'max_errors_against_gym': errors,
    'force_threshold_equality_and_zero_and_unselected_bodies': True,
    'stumble_ratio_equality_negative_vertical_and_zero': True,
    'current_pyi_names_missing_from_current_cost_module': missing,
    'required_export_name': 'joint_effort_limit_cost',
    'physics_force_limit': 'Normal contact forces in Lab differ from total contact forces in old Gym; same vector inputs only',
    'simulation_or_training_started': False,
}
(OUT / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(result, indent=2, ensure_ascii=False))
