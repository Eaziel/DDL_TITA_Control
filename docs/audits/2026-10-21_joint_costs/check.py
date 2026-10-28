"""Compare the documented joint-cost proposal with the original Gym formulas."""

from __future__ import annotations

import ast
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
GYM = Path('/home/eaziel/Downloads/tita_rl-master')
document = (ROOT / 'docs/migration/15_cost迁移.md').read_text()
proposal = re.search(r'```python\n(.*?)\n```', document, re.S).group(1)
namespace = {}
exec(compile(proposal, 'documented_costs.py', 'exec'), namespace)

source = (GYM / 'envs/legged_robot.py').read_text()
names = ('_cost_pos_limit', '_cost_torque_limit', '_cost_dof_vel_limits')
functions = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name in names]
old_namespace = {'torch': torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])), 'gym_costs.py', 'exec'), old_namespace)

task_names = [f'joint_{side}_leg_{i}' for side in ('left', 'right') for i in (1, 2, 3)]
task_names += ['joint_left_leg_4', 'joint_right_leg_4']
joints = {j.attrib['name']: j.find('limit').attrib for j in ET.parse(
    GYM / 'resources/tita/urdf/tita_description.urdf').getroot().findall('joint') if j.find('limit') is not None}
hard = torch.tensor([[float(joints[name]['lower']), float(joints[name]['upper'])] for name in task_names])
center = hard.mean(-1)
width = hard[:, 1] - hard[:, 0]
soft = torch.stack((center - 0.45 * width, center + 0.45 * width), dim=-1)
vel_limits = torch.tensor([float(joints[name]['velocity']) for name in task_names])
torque_limits = torch.tensor([float(joints[name]['effort']) for name in task_names])

torch.manual_seed(20261021)
N = 128
pos = center + (torch.rand(N, 8) - 0.5) * width * 1.2
vel = (torch.rand(N, 8) * 2 - 1) * vel_limits * 1.3
torques = (torch.rand(N, 8) * 2 - 1) * torque_limits * 1.3
pos[0] = soft[:, 1]
vel[0] = vel_limits
torques[0] = torque_limits
pos[1] = center
vel[1] = 0
torques[1] = 0
proxy = lambda value: NS(torch=value)
asset = NS(data=NS(joint_pos=proxy(pos), joint_vel=proxy(vel),
    soft_joint_pos_limits=proxy(soft.repeat(N, 1, 1)),
    joint_vel_limits=proxy(vel_limits.repeat(N, 1)),
    joint_effort_limits=proxy(torque_limits.repeat(N, 1))),
    actuators=NS(applied_effort=proxy(torques)))
env = NS(scene={'robot': asset})
mapping = [('joint_position_limit_cost', '_cost_pos_limit'),
           ('joint_effort_limit_cost', '_cost_torque_limit'),
           ('joint_velocity_limit_cost', '_cost_dof_vel_limits')]
errors = {}
for ids in (slice(None), [7, 2, 0]):
    selection = NS(name='robot', joint_ids=ids)
    old = NS(dof_pos=pos[:, ids], dof_vel=vel[:, ids], torques=torques[:, ids],
        dof_pos_limits=soft[ids], dof_vel_limits=vel_limits[ids], torque_limits=torque_limits[ids],
        cfg=NS(rewards=NS(soft_dof_vel_limit=1., soft_torque_limit=1.)))
    for new_name, old_name in mapping:
        result = namespace[new_name](env, selection)
        expected = old_namespace[old_name](old)
        torch.testing.assert_close(result, expected)
        assert result.shape == (N,) and result.dtype == torch.float32
        assert result[0] == 0 and result[1] == 0
        errors[f'{new_name}/{ids}'] = (result - expected).abs().max().item()

# 超额需求经限幅后不触发旧力矩cost，避免无意改成computed_effort口径。
torques.copy_(torch.maximum(torch.minimum(torques, torque_limits), -torque_limits))
assert namespace['joint_effort_limit_cost'](env, NS(name='robot', joint_ids=slice(None))).eq(0).all()

sys.path.insert(0, str(ROOT / 'TITA_Lab/src'))
from TITA_Lab.tasks.velocity.config.TITA.env_cfg import VelocityEnvCfg

cfg = VelocityEnvCfg()
saved_factor = cfg.scene.robot.soft_joint_pos_limit_factor
cfg.scene.robot.soft_joint_pos_limit_factor = 0.9
cfg.validate()

# 不导入旧Gym运行时，抽取配置类字面值验证实际排序约定。
config_tree = ast.parse((GYM / 'configs/tita_constraint_config.py').read_text())
task_cls = next(n for n in config_tree.body if isinstance(n, ast.ClassDef) and n.name == 'TitaConstraintRoughCfg')
cost_cls = next(n for n in task_cls.body if isinstance(n, ast.ClassDef) and n.name == 'costs')
scales_cls = next(n for n in cost_cls.body if isinstance(n, ast.ClassDef) and n.name == 'scales')
values = {n.targets[0].id: ast.literal_eval(n.value) for n in scales_cls.body if isinstance(n, ast.Assign)}
order = sorted(values)
result = {
    'scope': 'Documented proposal only; user task source unmodified; no simulation or training',
    'envs': N,
    'all_joints_and_reordered_subset_match_original_gym': errors,
    'strict_equality_is_not_violation': True,
    'clipped_torque_with_ratio_one_is_zero': True,
    'saved_position_factor_before_proposal': saved_factor,
    'in_memory_cfg_with_factor_0_9_validate': True,
    'old_cost_order_from_dir': order,
    'old_k_values_in_that_order': [values[name] for name in order],
    'candidate_wired_into_environment': False,
}
(OUT / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
print(json.dumps(result, indent=2, ensure_ascii=False))
