"""Compare proposed physical-step action latency against the actual Gym controller on CPU."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
GYM = Path('/home/eaziel/Downloads/tita_rl-master')
sys.path.insert(0, str(ROOT/'TITA_Lab/src'))

import TITA_Lab.tasks  # noqa: E402,F401
from TITA_Lab.tasks.velocity import tita_velocity_env as V  # noqa: E402
from TITA_Lab.tasks.velocity.actuators import TitaPDActuator  # noqa: E402
from TITA_Lab.tasks.velocity.config.TITA import env_cfg as C  # noqa: E402
from TITA_Lab.tasks.velocity.mdp import rewards as R  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import ActionManager, EventManager, EventTermCfg  # noqa: E402
from isaaclab.utils.string import resolve_matching_names  # noqa: E402
from isaaclab.utils.types import ArticulationActions  # noqa: E402


def load_candidate(name, filename):
    spec = importlib.util.spec_from_file_location(name, OUT/filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


A = load_candidate('tita_action_lag_candidate', 'proposal_actions.py')
E = load_candidate('tita_action_lag_event_candidate', 'proposal_event.py')
E.TitaJointAction = A.TitaJointAction
torch.manual_seed(20261020)
RESULT = {'scope': 'Proposals only; current TITA_Lab source is never written',
          'boundary': 'Native CPU action/event managers; asset writes/state and constructor parent are stand-ins; no simulator',
          'checks': {}, 'failures': {}}
PHYS_NAMES = ['joint_left_leg_1','joint_right_leg_1','joint_left_leg_2','joint_right_leg_2',
              'joint_left_leg_3','joint_right_leg_3','joint_left_leg_4','joint_right_leg_4']
GYM_TO_PHYS = [0,2,4,6,1,3,5,7]
PHYS_TO_GYM = [0,4,1,5,2,6,3,7]
POLICY_TO_GYM = [0,1,2,4,5,6,3,7]
DEFAULT_PHYS = torch.tensor([0.,0.,.8,.8,-1.5,-1.5,0.,0.])


def check(name, function):
    try:
        RESULT['checks'][name] = function()
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
    joint_names = PHYS_NAMES

    def __init__(self, count):
        self.data = NS(default_joint_pos=NS(torch=DEFAULT_PHYS.repeat(count,1)))
        self.position_target = self.data.default_joint_pos.torch.clone()
        self.effort_target = torch.zeros(count,8)

    def find_joints(self, names, preserve_order=False, as_proxy=False):
        ids, names = resolve_matching_names(names, self.joint_names, preserve_order=preserve_order)
        return (IndexProxy(ids) if as_proxy else ids), names

    def set_joint_position_target_index(self, target, joint_ids):
        self.position_target[:,joint_ids] = target

    def set_joint_effort_target_index(self, target, joint_ids):
        self.effort_target[:,joint_ids] = target


def action_cfg(slots):
    cfg = C.ActionsCfg()
    for name in ('leg_position','wheel_effort'):
        fields = vars(getattr(cfg,name)).copy()
        fields.pop('class_type')
        setattr(cfg,name,A.TitaJointActionCfg(**fields,lag_timesteps=slots))
    return cfg


def make_env(count=24, slots=3):
    asset = Asset(count)
    env = NS(num_envs=count,device='cpu',scene={'robot':asset},sim=NS(is_playing=lambda:True),
             action_lag_indices=torch.zeros(count,dtype=torch.long))
    env.action_manager = ActionManager(action_cfg(slots),env)
    env.event_manager = EventManager(NS(randomize_action_lag=EventTermCfg(
        func=E.randomize_tita_action_lag,mode='startup')),env)
    return env,asset


def ownership_and_startup():
    env,_ = make_env()
    shared = env.action_lag_indices
    for name in env.action_manager.active_terms:
        assert env.action_manager.get_term(name)._lag_indices is shared
    env.event_manager.apply(mode='startup')
    assert env.action_lag_indices is shared
    assert shared.min() >= 0 and shared.max() <= 2
    before = shared.clone()
    env.action_manager.reset(torch.tensor([1,3]))
    env.event_manager.reset(torch.tensor([1,3]))
    torch.testing.assert_close(shared,before)
    env.event_manager.apply(mode='startup',env_ids=torch.tensor([1,3]))
    assert torch.equal(shared[[0,2]],before[[0,2]])
    cfg = C.VelocityEnvCfg()
    cfg.actions = action_cfg(3)
    cfg.events.randomize_action_lag = EventTermCfg(func=E.randomize_tita_action_lag,mode='startup')
    cfg.validate()
    return {'shared_tensor_identity':True,'startup_bounds':[0,2],'partial_selection':True,
            'reset_preserves_indices':True,'candidate_config_validate':True}


check('ownership_startup_and_configuration',ownership_and_startup)


def startup_response():
    env,asset = make_env(count=3)
    env.action_lag_indices.copy_(torch.tensor([0,1,2]))
    env.action_manager.process_action(torch.ones(3,8))
    observed = []
    for physical_step in range(3):
        env.action_manager.apply_action()
        values = asset.effort_target[:,6].clone()
        observed.append(values.tolist())
        expected = torch.tensor([[0.,0.,4.],[0.,4.,4.],[4.,4.,4.]])[physical_step]
        torch.testing.assert_close(values,expected)
        target = DEFAULT_PHYS[:6] + (values/4.).unsqueeze(1)*torch.tensor([.2,.2,.4,.4,.4,.4])
        torch.testing.assert_close(asset.position_target[:,:6],target)
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        kept = term._lag_buffer[2].clone()
        term.reset((0,1))
        torch.testing.assert_close(term._lag_buffer[:2],term._initial_command[:2,None,:].repeat(1,3,1))
        torch.testing.assert_close(term._lag_buffer[2],kept)
        torch.testing.assert_close(term.clipped_actions,torch.ones_like(term.clipped_actions))
    return {'lag_indices':[0,1,2],'actual_delay_steps':[2,1,0],
            'first_three_wheel_efforts_Nm':observed,'tuple_partial_reset_axes':True,
            'fir_samples_preserved':True}


check('zero_offset_warmup_and_partial_reset',startup_response)

legacy_tree = ast.parse((GYM/'envs/legged_robot.py').read_text())
methods = [n for c in legacy_tree.body if isinstance(c,ast.ClassDef) for n in c.body
           if isinstance(n,ast.FunctionDef) and n.name in ('_compute_torques','_low_pass_action_filter')]
legacy = {'torch':torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),
             str(GYM/'envs/legged_robot.py'),'exec'),legacy)


def torque_comparison(slots):
    count = 24
    env,asset = make_env(count,slots)
    env.event_manager.apply(mode='startup')
    env.action_lag_indices[:3] = torch.tensor([0,1,2]) if slots == 3 else 0
    groups = {}
    for name,ids in [('legs',list(range(6))),('wheels',[6,7])]:
        groups[name] = TitaPDActuator(C.VelocitySceneCfg().robot.actuators[name].copy(),
            joint_names=[PHYS_NAMES[i] for i in ids],joint_ids=slice(None),
            num_envs=count,device='cpu',stiffness=0.,damping=0.)
        for attribute in ('motor_strength','kp_factor','kd_factor'):
            getattr(groups[name],attribute).uniform_(.8,1.2)
    old = NS(cfg=NS(control=NS(use_filter=True,action_scale=.5,hip_scale_reduction=.5,control_type='P'),
                    domain_rand=NS(randomize_lag_timesteps=slots>1,randomize_kpkd=True)),
        default_dof_pos=asset.data.default_joint_pos.torch[:,GYM_TO_PHYS],
        p_gains=torch.full((8,),40.),d_gains=torch.ones(8),
        torque_limits=torch.tensor([60.,60.,60.,15.,60.,60.,60.,15.]),
        last_actions=torch.zeros(count,8),lag_buffer=torch.zeros(count,slots,8),
        num_envs_indexes=list(range(count)),randomized_lag=env.action_lag_indices)
    old._low_pass_action_filter = lambda actions: legacy['_low_pass_action_filter'](old,actions)
    for attribute in ('motor_strength','kp_factor','kd_factor'):
        value = torch.cat([getattr(groups['legs'],attribute),getattr(groups['wheels'],attribute)],dim=1)
        setattr(old,attribute,value[:,GYM_TO_PHYS])
    max_torque_error = 0.
    max_target_error = 0.
    previous_clipped = torch.zeros(count,8)
    for step in range(20):
        raw = torch.randn(count,8)*(150. if step%5 == 0 else .5)
        clipped = raw.clamp(-100.,100.)
        old_actions = torch.empty_like(raw)
        old_actions[:,POLICY_TO_GYM] = clipped
        env.action_manager.process_action(raw)
        with patch.object(R,'TitaJointAction',A.TitaJointAction):
            torch.testing.assert_close(R.action_rate_clipped_l2(env),((clipped-previous_clipped)**2).sum(1))
        for _ in range(4):
            q = asset.data.default_joint_pos.torch+torch.randn(count,8)*.1
            velocity = torch.randn(count,8)*2.
            old.dof_pos = q[:,GYM_TO_PHYS]
            old.dof_vel = velocity[:,GYM_TO_PHYS]
            expected = legacy['_compute_torques'](old,old_actions)[:,PHYS_TO_GYM]
            env.action_manager.apply_action()
            actual = []
            for name,ids in [('legs',list(range(6))),('wheels',[6,7])]:
                output = groups[name].compute(ArticulationActions(
                    joint_positions=asset.position_target[:,ids].clone(),joint_velocities=torch.zeros(count,len(ids)),
                    joint_efforts=asset.effort_target[:,ids].clone()),q[:,ids],velocity[:,ids])
                actual.append(output.joint_efforts)
            actual = torch.cat(actual,dim=1)
            torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)
            max_torque_error = max(max_torque_error,float((actual-expected).abs().max()))
            if slots>1:
                old_target = old.lag_buffer[old.num_envs_indexes,old.randomized_lag,:]+old.default_dof_pos
                expected_pos = old_target[:,PHYS_TO_GYM[:6]]
                expected_effort = 10.*old_target[:,[3,7]]
                torch.testing.assert_close(asset.position_target[:,:6],expected_pos)
                torch.testing.assert_close(asset.effort_target[:,6:],expected_effort)
                max_target_error = max(max_target_error,float((asset.position_target[:,:6]-expected_pos).abs().max()))
        old.last_actions.copy_(old_actions)
        previous_clipped.copy_(clipped)
        if step in (5,12):
            indices = torch.tensor([1,3,7])
            kept = env.action_lag_indices.clone()
            env.action_manager.reset(indices)
            old.lag_buffer[indices,:,:] = 0.
            torch.testing.assert_close(env.action_lag_indices,kept)
    return {'num_envs':count,'policy_steps':20,'physical_calls_per_step':4,'history_slots':slots,
            'max_torque_error_Nm':max_torque_error,'max_position_target_error_rad':max_target_error,
            'real_physx_joint_order_used':True,'partial_auto_reset_steps':[5,12],
            'action_rate_unchanged':True,'gym_filter_enabled':True,'gym_lag_enabled':slots>1}


check('three_slot_gym_torque_comparison',lambda:torque_comparison(3))
check('one_slot_zero_latency_comparison',lambda:torque_comparison(1))


def environment_constructor_order():
    tree = ast.parse((ROOT/'TITA_Lab/src/TITA_Lab/tasks/velocity/tita_velocity_env.py').read_text())
    cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == 'TitaVelocityEnv')
    init = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == '__init__')
    init.body[0:0] = ast.parse('self.action_lag_indices = torch.zeros(cfg.scene.num_envs, dtype=torch.long, device=cfg.sim.device)').body
    namespace = dict(V.__dict__)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),
                 '<candidate environment constructor>','exec'),namespace)
    def parent(instance, cfg, **kwargs):
        assert instance.action_lag_indices.shape == (4,)
        assert instance.action_lag_indices.dtype == torch.long
        assert instance.proprio_history is not None
    cfg = C.VelocityEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = 'cpu'
    with patch.object(ManagerBasedRLEnv,'__init__',parent):
        namespace['TitaVelocityEnv'](cfg)
    return {'shared_indices_and_history_exist_before_parent':True,'shape':[4]}


check('environment_constructor_order',environment_constructor_order)
(OUT/'result.json').write_text(json.dumps(RESULT,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(RESULT,indent=2,ensure_ascii=False))
if RESULT['failures']:
    raise SystemExit(1)
