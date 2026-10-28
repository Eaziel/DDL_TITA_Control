"""Check saved TITA action math and CLI configuration without starting a simulator."""

from __future__ import annotations

import ast
import copy
import importlib.metadata
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
GYM = Path('/home/eaziel/Downloads/tita_rl-master')
sys.path.insert(0, str(ROOT / 'TITA_Lab/src'))

import TITA_Lab.tasks  # noqa: E402, F401
from TITA_Lab.tasks.velocity.actuators import TitaPDActuator  # noqa: E402
from TITA_Lab.tasks.velocity.config.TITA import env_cfg as C  # noqa: E402
from TITA_Lab.tasks.velocity.mdp import actions as A, rewards as R  # noqa: E402
from isaaclab.managers import ActionManager  # noqa: E402
from isaaclab.utils.string import resolve_matching_names  # noqa: E402
from isaaclab.utils.types import ArticulationActions  # noqa: E402

torch.manual_seed(20261020)
RESULT = {'boundary': 'CPU math, native managers, and pre-launch CLI only; no AppLauncher/physics/training',
          'checks': {}, 'failures': {}}
N = 4
OLD_NAMES = ['joint_left_leg_1', 'joint_left_leg_2', 'joint_left_leg_3', 'joint_left_leg_4',
             'joint_right_leg_1', 'joint_right_leg_2', 'joint_right_leg_3', 'joint_right_leg_4']
PERM = [0, 1, 2, 4, 5, 6, 3, 7]
DEFAULTS = torch.tensor([0., .8, -1.5, 0., 0., .8, -1.5, 0.]).repeat(N, 1)


def check(name, fn):
    try:
        RESULT['checks'][name] = fn()
        print('PASS', name, flush=True)
    except Exception as exc:
        import traceback
        RESULT['failures'][name] = {'error': repr(exc), 'traceback': traceback.format_exc()}
        print('FAIL', name, repr(exc), flush=True)


class IndexProxy:
    def __init__(self, values):
        self.torch = torch.tensor(values, dtype=torch.int64)

    def __len__(self):
        return len(self.torch)


class Asset:
    num_joints = 8
    joint_names = OLD_NAMES

    def __init__(self):
        self.data = NS(default_joint_pos=NS(torch=DEFAULTS.clone()))
        self.position_target = DEFAULTS.clone()
        self.effort_target = torch.zeros(N, 8)
        self.setter_calls = 0

    def find_joints(self, names, preserve_order=False, as_proxy=False):
        ids, resolved = resolve_matching_names(names, self.joint_names, preserve_order=preserve_order)
        return (IndexProxy(ids) if as_proxy else ids), resolved

    def set_joint_position_target_index(self, target, joint_ids):
        self.position_target[:, joint_ids] = target
        self.setter_calls += 1

    def set_joint_effort_target_index(self, target, joint_ids):
        self.effort_target[:, joint_ids] = target
        self.setter_calls += 1


def make_env():
    asset = Asset()
    env = NS(num_envs=N, device='cpu', scene={'robot': asset}, sim=NS(is_playing=lambda: True))
    env.action_manager = ActionManager(C.ActionsCfg(), env)
    return env, asset


def saved_apply_failure():
    env, asset = make_env()
    env.action_manager.process_action(torch.zeros(N, 8))
    try:
        env.action_manager.apply_action()
    except NotImplementedError:
        assert asset.setter_calls == 1
        return {'reproduced': 'NotImplementedError from the inherited abstract apply_actions',
                'first_target_written_before_exception': True}
    raise AssertionError('Expected the saved super().apply_actions() to raise.')


check('saved_apply_actions_bug', saved_apply_failure)

# Delete only the trailing parent call in memory. Protected project source is never written.
tree = ast.parse((ROOT / 'TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/actions.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TitaJointAction')
method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'apply_actions'))
assert isinstance(method.body[-1], ast.Return)
method.body.pop()
namespace = dict(A.__dict__)
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
             '<in-memory apply_actions correction>', 'exec'), namespace)
candidate_apply = namespace['apply_actions']


def corrected_action_math():
    env, asset = make_env()
    manager = env.action_manager
    assert manager.active_terms == ['leg_position', 'wheel_effort']
    assert manager.action_term_dim == [6, 2]
    previous = torch.zeros(N, 8)
    max_target_error = 0.
    for step in range(12):
        raw = torch.randn(N, 8) * (150. if step % 3 == 0 else 1.)
        clipped = raw.clamp(-100., 100.)
        filtered = .2 * previous + .8 * clipped
        manager.process_action(raw)
        torch.testing.assert_close(manager.action, raw)
        torch.testing.assert_close(R.action_rate_clipped_l2(env), ((clipped-previous)**2).sum(1))
        expected_pos = DEFAULTS[:, PERM[:6]] + filtered[:, :6] * torch.tensor([.25, .5, .5, .25, .5, .5])
        expected_ff = 5. * filtered[:, 6:]
        for _ in range(4):
            manager.apply_action()
            torch.testing.assert_close(asset.position_target[:, PERM[:6]], expected_pos)
            torch.testing.assert_close(asset.effort_target[:, PERM[6:]], expected_ff)
        for name, sl in [('leg_position', slice(0, 6)), ('wheel_effort', slice(6, 8))]:
            term = manager.get_term(name)
            torch.testing.assert_close(term.raw_actions, raw[:, sl])
            torch.testing.assert_close(term.clipped_actions, clipped[:, sl])
            torch.testing.assert_close(term.previous_clipped_actions, previous[:, sl])
        max_target_error = max(max_target_error, float((asset.position_target[:, PERM[:6]]-expected_pos).abs().max()))
        previous = clipped.clone()
    saved_clipped = [manager.get_term(name).clipped_actions.clone() for name in manager.active_terms]
    manager.reset(torch.tensor([1, 3]))
    for name, saved in zip(manager.active_terms, saved_clipped, strict=True):
        torch.testing.assert_close(manager.get_term(name).clipped_actions, saved)
        assert manager.get_term(name).raw_actions[[1, 3]].count_nonzero() == 0
    manager.process_action(torch.full((N, 8), 150.))
    manager.process_action(torch.full((N, 8), 200.))
    assert R.action_rate_clipped_l2(env).count_nonzero() == 0
    return {'action_dim': 8, 'group_sizes': [6, 2], 'steps': 12, 'apply_per_step': 4,
            'max_target_error': max_target_error, 'raw_observation_and_clipped_reward_distinct': True,
            'clipped_history_preserved_on_partial_reset': True, 'saturated_action_rate_zero': True}


def corrected_torques_match_gym():
    env, asset = make_env()
    groups = {}
    for name, count in [('legs', 6), ('wheels', 2)]:
        groups[name] = TitaPDActuator(C.VelocitySceneCfg().robot.actuators[name].copy(),
            joint_names=[f'j{i}' for i in range(count)], joint_ids=slice(None),
            num_envs=N, device='cpu', stiffness=0., damping=0.)
        for attr in ('motor_strength', 'kp_factor', 'kd_factor'):
            getattr(groups[name], attr).uniform_(.8, 1.2)
    gym_tree = ast.parse((GYM/'envs/legged_robot.py').read_text())
    methods = [n for c in gym_tree.body if isinstance(c, ast.ClassDef) for n in c.body
               if isinstance(n, ast.FunctionDef) and n.name in ('_compute_torques', '_low_pass_action_filter')]
    legacy_ns = {'torch': torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])),
                 str(GYM/'envs/legged_robot.py'), 'exec'), legacy_ns)
    old = NS(cfg=NS(control=NS(use_filter=True, action_scale=.5, hip_scale_reduction=.5, control_type='P'),
                    domain_rand=NS(randomize_lag_timesteps=False, randomize_kpkd=True)),
             default_dof_pos=DEFAULTS, dof_pos=torch.zeros(N,8), dof_vel=torch.zeros(N,8),
             p_gains=torch.full((8,), 40.), d_gains=torch.ones(8),
             torque_limits=torch.tensor([60.,60.,60.,15.,60.,60.,60.,15.]), last_actions=torch.zeros(N,8))
    old._low_pass_action_filter = lambda x: legacy_ns['_low_pass_action_filter'](old, x)
    for attr in ('motor_strength', 'kp_factor', 'kd_factor'):
        value = torch.empty(N, 8)
        value[:, PERM[:6]] = getattr(groups['legs'], attr)
        value[:, PERM[6:]] = getattr(groups['wheels'], attr)
        setattr(old, attr, value)
    max_error = 0.
    for step in range(16):
        raw = torch.randn(N,8) * (150. if step % 4 == 0 else 1.)
        old_actions = torch.empty_like(raw)
        old_actions[:, PERM] = raw.clamp(-100.,100.)
        env.action_manager.process_action(raw)
        for _ in range(4):
            old.dof_pos = torch.randn(N,8)
            old.dof_vel = torch.randn(N,8)*20.
            expected = legacy_ns['_compute_torques'](old, old_actions)[:, PERM]
            env.action_manager.apply_action()
            actual = []
            for name, ids in [('legs', PERM[:6]), ('wheels', PERM[6:])]:
                out = groups[name].compute(ArticulationActions(
                    joint_positions=asset.position_target[:, ids].clone(),
                    joint_velocities=torch.zeros(N, len(ids)),
                    joint_efforts=asset.effort_target[:, ids].clone()),
                    old.dof_pos[:, ids], old.dof_vel[:, ids])
                actual.append(out.joint_efforts)
            error = (torch.cat(actual,1)-expected).abs().max()
            max_error = max(max_error, float(error))
            torch.testing.assert_close(torch.cat(actual,1), expected, atol=1e-5, rtol=2e-5)
        old.last_actions.copy_(old_actions)
    return {'steps': 16, 'physics_calls_per_step': 4, 'gym_filter_enabled': True,
            'lag_disabled': True, 'max_torque_error_Nm': max_error}


with patch.object(A.TitaJointAction, 'apply_actions', candidate_apply):
    check('candidate_action_processing_and_reward', corrected_action_math)
    check('candidate_torques_against_actual_gym_methods', corrected_torques_match_gym)


def cli_configuration_only():
    from isaaclab.cli import cli
    from isaaclab_rl.entrypoints import simple_agents

    entries = [str(e.value) for e in importlib.metadata.entry_points(group='isaaclab.tasks')
               if e.name == 'TITA_Lab']
    assert 'TITA_Lab.tasks' in entries, entries
    captured = {}

    class StopBeforeLaunch(Exception):
        pass

    def stop(env_cfg, args_cli):
        captured.update(physics=type(env_cfg.sim.physics).__name__, num_envs=env_cfg.scene.num_envs,
                        max_steps=args_cli.max_steps, visualizer=args_cli.visualizer,
                        registered_entry_point=True, configuration_validate_passed=True)
        raise StopBeforeLaunch

    argv = ['isaaclab', 'zero_agent', '--task', 'TitaLab-Velocity-Tita', '--num_envs', '4',
            '--max_steps', '20', '--device', 'cuda:0', '--visualizer', 'none', 'physics=isaacsim_physx']
    with patch.object(sys, 'argv', argv), patch.object(simple_agents, 'launch_simulation', stop):
        try:
            cli()
        except StopBeforeLaunch:
            pass
    assert captured['physics'] == 'PhysxCfg', captured
    return captured


check('official_zero_agent_cli_prelaunch', cli_configuration_only)
(OUT/'result.json').write_text(json.dumps(RESULT, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(RESULT, indent=2, ensure_ascii=False))
if RESULT['failures']:
    raise SystemExit(1)
