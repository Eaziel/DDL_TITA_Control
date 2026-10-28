"""Audit the saved TITA migration through motor-factor randomization on CPU."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.abc
import json
import re
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
import warp as wp
from pxr import Sdf, Usd, UsdPhysics

ROOT = Path('/mnt/D/BaiduSyncdisk/1ACrutial_Files/Ubuntu_2404_projects/isaaclab_ws/Tita_HWBR')
PKG = ROOT / 'TITA_Lab/src/TITA_Lab'
OUT = Path(__file__).parent
GYM = Path('/home/eaziel/Downloads/tita_rl-master')
LAB = Path('/home/eaziel/IsaacLab/source')
RESULT = {'scope': 'Saved implementation through motor factors; unfinished mdp/actions.py excluded',
          'boundary': 'CPU formulas/managers and in-memory USD only; no simulation or training',
          'checks': {}, 'failures': {}}

class ExcludeUnfinishedActions(importlib.abc.MetaPathFinder):
    """Reject imports outside the saved migration audit scope."""
    def find_spec(self, fullname, path, target=None):
        if fullname == 'TITA_Lab.tasks.velocity.mdp.actions':
            raise ImportError('Unfinished action module is excluded from this audit.')

sys.meta_path.insert(0, ExcludeUnfinishedActions())

# Bypass package discovery so the unfinished action module is never imported.
for name, rel in [('TITA_Lab', ''), ('TITA_Lab.tasks', 'tasks'),
                  ('TITA_Lab.tasks.velocity', 'tasks/velocity'),
                  ('TITA_Lab.tasks.velocity.config', 'tasks/velocity/config'),
                  ('TITA_Lab.tasks.velocity.config.TITA', 'tasks/velocity/config/TITA')]:
    m = types.ModuleType(name)
    m.__path__ = [str(PKG / rel)]
    sys.modules[name] = m

# Freeze the two configuration blocks affected by the user's ongoing FIR edits.
# The baseline comes from the pre-audit documentation, never from mdp/actions.py.
baseline_path = OUT / 'baseline_blocks.py'
if not baseline_path.exists():
    record=OUT/'documents_before/code_explaination/TITA_Lab/src/TITA_Lab/tasks/velocity/config/TITA/env_cfg.py.md'
    found={}
    for code in re.findall(r'```python\n(.*?)\n```',record.read_text().split('## 历史解释原文')[0],re.S):
        try: parsed=ast.parse(code)
        except SyntaxError:continue
        for item in parsed.body:
            if isinstance(item,ast.ClassDef) and item.name in ('ActionsCfg','RewardsCfg') and item.name not in found:
                found[item.name]=item
    baseline_path.write_text(ast.unparse(ast.Module(body=list(found.values()),type_ignores=[])))
baseline_nodes={n.name:n for n in ast.parse(baseline_path.read_text()).body}
mdp_name='TITA_Lab.tasks.velocity.mdp'
baseline_mdp=types.ModuleType(mdp_name);baseline_mdp.__path__=[str(PKG/'tasks/velocity/mdp')]
custom_modules={
 'joint_pos_rel_zero_wheels':'observations','proprio_history':'observations',
 'illegal_contact_current':'terminations','undesired_contacts_current':'rewards',
 'joint_power_signed':'rewards','JointAccelerationPenalty':'rewards',
 'push_by_setting_xy_velocity':'events','RandomizeTitaMaterial':'events',
 'randomize_tita_motor_factors':'events',
}
def baseline_mdp_getattr(name):
    if name in custom_modules:
        return getattr(importlib.import_module(mdp_name+'.'+custom_modules[name]),name)
    if name.startswith('TitaJoint') or name=='action_rate_clipped_l2':
        raise AttributeError('The unfinished action proposal is outside this audit.')
    return getattr(importlib.import_module('isaaclab.envs.mdp'),name)
baseline_mdp.__getattr__=baseline_mdp_getattr
sys.modules[mdp_name]=baseline_mdp
# The user may add imports and a new action-rate function to rewards.py while
# this audit runs. Execute only the four already completed reward definitions.
reward_path=PKG/'tasks/velocity/mdp/rewards.py'
reward_tree=ast.parse(reward_path.read_text())
reward_names={'joint_pos_target_l2','undesired_contacts_current','joint_power_signed','JointAccelerationPenalty'}
reward_tree.body=[n for n in reward_tree.body
    if (isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in reward_names)
    or isinstance(n,(ast.Import,ast.If))
    or (isinstance(n,ast.ImportFrom) and n.module!='actions')]
R=types.ModuleType(mdp_name+'.rewards');R.__file__=str(reward_path);R.__package__=mdp_name
sys.modules[R.__name__]=R
exec(compile(ast.fix_missing_locations(reward_tree),str(reward_path),'exec'),R.__dict__)
cfg_path=PKG/'tasks/velocity/config/TITA/env_cfg.py'
cfg_tree=ast.parse(cfg_path.read_text())
cfg_tree.body=[copy.deepcopy(baseline_nodes[n.name]) if isinstance(n,ast.ClassDef) and n.name in baseline_nodes else n
               for n in cfg_tree.body]
C=types.ModuleType('TITA_Lab.tasks.velocity.config.TITA.env_cfg')
C.__file__=str(cfg_path);C.__package__='TITA_Lab.tasks.velocity.config.TITA'
sys.modules[C.__name__]=C
exec(compile(ast.fix_missing_locations(cfg_tree),str(cfg_path),'exec'),C.__dict__)
RESULT['configuration_scope']='ActionsCfg/RewardsCfg reconstructed at the Kp cutoff; current in-progress FIR configuration and exports excluded'
A = importlib.import_module('TITA_Lab.tasks.velocity.actuators')
E = importlib.import_module('TITA_Lab.tasks.velocity.mdp.events')
R = importlib.import_module('TITA_Lab.tasks.velocity.mdp.rewards')
O = importlib.import_module('TITA_Lab.tasks.velocity.mdp.observations')
H = importlib.import_module('TITA_Lab.tasks.velocity.mdp.observation_history')
V = importlib.import_module('TITA_Lab.tasks.velocity.tita_velocity_env')
from isaaclab.actuators import ActuatorCollection
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp
from isaaclab.managers import ObservationManager, RewardManager, EventManager, TerminationManager, SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.types import ArticulationActions
from isaaclab.sensors.ray_caster.patterns import grid_pattern

def check(name, fn):
    try:
        RESULT['checks'][name] = fn()
        print('PASS', name, flush=True)
    except Exception as exc:
        import traceback
        RESULT['failures'][name] = {'error': repr(exc), 'traceback': traceback.format_exc()}
        print('FAIL', name, repr(exc), flush=True)

class Scene(dict):
    num_envs = 128
    env_origins = torch.zeros(128, 3)
    @property
    def sensors(self):
        return self

def proxy(t):
    return NS(torch=t)

N = 128
joint_names = C.TITA_JOINT_NAMES
body_names = ['base_link', 'left_leg_1', 'left_leg_2', 'left_leg_3', 'left_leg_4',
              'right_leg_1', 'right_leg_2', 'right_leg_3', 'right_leg_4']

class Asset:
    device = 'cpu'
    num_joints = 8
    num_bodies = 9
    joint_names = joint_names
    body_names = body_names
    def find_joints(self, names, preserve_order=False):
        return resolve_matching_names(names, self.joint_names, preserve_order=preserve_order)
    def find_bodies(self, names, preserve_order=False):
        return resolve_matching_names(names, self.body_names, preserve_order=preserve_order)
    def set_masses_index(self, masses, body_ids, env_ids):
        self.data.body_mass.torch[env_ids[:, None], body_ids] = masses
    def set_inertias_index(self, inertias, body_ids, env_ids):
        self.data.body_inertia.torch[env_ids[:, None], body_ids] = inertias
    def set_coms_index(self, coms, body_ids, env_ids):
        self.data.body_com_pose_b.torch[env_ids[:, None], body_ids] = coms
    def write_root_velocity_to_sim_index(self, root_velocity, env_ids=None):
        self.data.root_vel_w.torch[slice(None) if env_ids is None else env_ids] = root_velocity

asset = Asset()
asset.data = NS(joint_pos=proxy(torch.randn(N, 8)), default_joint_pos=proxy(torch.zeros(N, 8)),
                joint_vel=proxy(torch.randn(N, 8)), default_joint_vel=proxy(torch.zeros(N, 8)),
                root_ang_vel_b=proxy(torch.randn(N, 3)), root_lin_vel_b=proxy(torch.randn(N, 3)),
                projected_gravity_b=proxy(torch.randn(N, 3)), root_pos_w=proxy(torch.randn(N, 3)),
                root_vel_w=proxy(torch.randn(N, 6)),
                body_mass=proxy(torch.tensor([13.2]+[1.]*8).repeat(N, 1)),
                body_inertia=proxy(torch.eye(3).flatten().repeat(N, 9, 1)),
                body_com_pose_b=proxy(torch.tensor([0., 0., 0., 0., 0., 0., 1.]).repeat(N, 9, 1)))
asset.actuators = NS(applied_effort=proxy(torch.randn(N, 8)))
sensor = Asset()
sensor.data = NS(net_normal_forces_w=proxy(torch.randn(N, 9, 3)))
scene = Scene(robot=asset, contact_forces=sensor,
              height_scanner=NS(data=NS(ray_hits_w=proxy(torch.randn(N, 187, 3)))))
env = NS(num_envs=N, device='cpu', scene=scene, step_dt=.01, physics_dt=.0025,
         max_episode_length=2000, max_episode_length_s=20.,
         episode_length_buf=torch.zeros(N, dtype=torch.long),
         sim=NS(is_playing=lambda: True, physics_manager=type('PhysxManager', (), {})),
         action_manager=NS(action=torch.randn(N, 8), prev_action=torch.randn(N, 8)),
         proprio_history=H.ProprioHistory(N, 33, 10),
         command_manager=NS(get_command=lambda name: commands))
commands = torch.randn(N, 3)

def configs():
    cfg = C.VelocityEnvCfg()
    cfg.validate()
    assert cfg.sim.dt * cfg.decimation == .01
    assert cfg.sim.physics.default.__class__.__name__ == 'PhysxCfg'
    assert 'TITA_Lab.tasks.velocity.mdp.actions' not in sys.modules
    return {'validate': True, 'policy_dt': .01, 'reward_terms': list(vars(cfg.rewards)),
            'action_names': list(vars(cfg.actions)), 'unfinished_module_imported': False}
check('config_and_exports', configs)

def observations():
    manager = ObservationManager(C.ObservationsCfg(), env)
    before = asset.data.joint_pos.torch.clone()
    out = manager.compute()
    assert out['policy'].shape == (N, 33) and out['history'].shape == (N, 330)
    assert torch.equal(before, asset.data.joint_pos.torch)
    cfg = copy.deepcopy(C.ObservationsCfg())
    cfg.policy.enable_corruption = False
    manager = ObservationManager(cfg, env)
    out = manager.compute()['policy']
    q = asset.data.joint_pos.torch.clone(); q[:, 6:] = 0
    expected = torch.cat((asset.data.root_ang_vel_b.torch*.25,
        asset.data.projected_gravity_b.torch, commands*torch.tensor([2.,2.,.25]),
        q, asset.data.joint_vel.torch*.05, env.action_manager.action), dim=1)
    torch.testing.assert_close(out, expected)
    # Same random draws, so compare native manager noise with Gym's post-scale noise.
    noisy = ObservationManager(C.ObservationsCfg(), env)
    torch.manual_seed(1123); got = noisy.compute()['policy']
    torch.manual_seed(1123)
    noise_parts = [(torch.rand(N, k)*2-1)*s for k,s in [(3,.2),(3,.05),(8,.01),(8,.075)]]
    expected[:, :3] += noise_parts[0]; expected[:,3:6] += noise_parts[1]
    expected[:,9:17] += noise_parts[2]; expected[:,17:25] += noise_parts[3]
    torch.testing.assert_close(got, expected, atol=5e-7, rtol=1e-5)
    return {'policy_shape': [N,33], 'history_shape': [N,330],
            'noise_final_halfwidths': [.2,.05,.01,.075], 'source_joint_pos_unmodified': True}
check('native_observation_manager', observations)

def history():
    history = H.ProprioHistory(5, 33, 10)
    # Gym returns its old cache and then appends the current observation.
    gym_cache = torch.zeros(5,10,33)
    ep = torch.zeros(5, dtype=torch.long)
    reset_ids = torch.tensor([1,3])
    obs = torch.zeros(5,33)
    previous = obs.clone()
    for step in range(40):
        history.read_then_push(previous, ep)
        if step == 0:
            gym_cache = previous[:,None,:].repeat(1,10,1)
        ep += 1
        obs = torch.full((5,33), float(step+1))
        if step in (7,19):
            history.reset(reset_ids); gym_cache[reset_ids,:,:] = 0; ep[reset_ids] = 0
        torch.testing.assert_close(history.read(), gym_cache)
        torch.testing.assert_close(history.read(), gym_cache)
        gym_cache = torch.where((ep<=1)[:,None,None], obs[:,None,:].expand(-1,10,-1),
                               torch.cat((gym_cache[:,1:,:],obs[:,None,:]),1))
        previous = obs.clone()
    history.reset((0,2))
    assert not history.read()[[0,2]].any()
    return {'transitions': 40, 'partial_resets': 2, 'repeat_reads': True, 'tuple_reset': True}
check('history_timeline', history)

def rewards():
    tm = TerminationManager(C.TerminationsCfg(), env); env.termination_manager = tm
    rm = RewardManager(C.RewardsCfg(), env)
    assert len(rm.active_terms) == 10
    tm.compute(); result = rm.compute(.01)
    assert result.shape == (N,)
    cache = rm._step_reward.clone()
    ids = torch.tensor([1,3], dtype=torch.int32)
    rm.reset(ids)
    torch.testing.assert_close(rm._step_reward, cache)
    # Actual class manager owns reset; compare before/after reset velocity references.
    acc = rm.get_term_cfg('dof_acc').func
    torch.testing.assert_close(acc._previous_joint_vel[ids], asset.data.joint_vel.torch[ids])
    # Force collision+timeout simultaneously: timeout must suppress failure punishment.
    env.episode_length_buf[:] = 2000
    tm.compute()
    term = rm.get_term_cfg('termination').func
    assert not term(env, term_keys='base_contact').any()
    env.episode_length_buf.zero_()
    # Run the actual outer step implementation with the native manager results as parent output.
    dummy = object.__new__(V.TitaVelocityEnv)
    dummy.cfg = C.VelocityEnvCfg()
    dummy.scene = scene
    dummy._is_closed = True
    dummy.proprio_history = H.ProprioHistory(N,33,10)
    dummy.obs_buf = {'policy':torch.zeros(N,33)}
    dummy.episode_length_buf = env.episode_length_buf
    dummy.reward_manager = rm
    raw = torch.linspace(-4,2,N)
    ti = rm.active_terms.index('termination'); rm._step_reward[:,ti] = -200
    expected = (raw+2).clamp_min(0)-2
    with patch.object(ManagerBasedRLEnv, 'step', return_value=({},raw,torch.zeros(N),torch.zeros(N),{})):
        got = V.TitaVelocityEnv.step(dummy, torch.zeros(N,8))[1]
    torch.testing.assert_close(got, expected)
    return {'terms': rm.active_terms, 'native_manager_compute': True,
            'reset_keeps_step_reward': True, 'timeout_masks_failure': True,
            'final_positive_ordinary_sum': True}
check('native_reward_and_termination_managers', rewards)

def motors():
    groups = {}
    for name, cfg, count in [('legs', C.VelocitySceneCfg().robot.actuators['legs'],6),
                             ('wheels',C.VelocitySceneCfg().robot.actuators['wheels'],2)]:
        groups[name] = A.TitaPDActuator(cfg.copy(), joint_names=[f'j{i}' for i in range(count)],
            joint_ids=slice(None), num_envs=N, device='cpu', stiffness=0.,damping=0.)
    robot = NS(device='cpu',actuators=groups)
    en = NS(scene={'robot':robot})
    ecfg = SceneEntityCfg('robot')
    E.randomize_tita_motor_factors(en,None,ecfg,(.8,1.2),(.8,1.2),(.8,1.2))
    errors = []
    for name, a in groups.items():
        q = torch.randn_like(a.stiffness); qd = torch.randn_like(q); u = torch.randn_like(q)
        qp = torch.randn_like(q); qv = torch.zeros_like(q)
        ff = u*5 if name=='wheels' else torch.zeros_like(q)
        expected = a.motor_strength*(a.kp_factor*a.stiffness*(qp-q)
                    +a.kd_factor*a.damping*(qv-qd)+(a.kp_factor*ff if name=='wheels' else ff))
        lim = 15. if name=='wheels' else 60.
        out = a.compute(ArticulationActions(joint_positions=qp,joint_velocities=qv,joint_efforts=ff),q,qd)
        torch.testing.assert_close(out.joint_efforts,expected.clamp(-lim,lim))
        errors.append(float((out.joint_efforts-expected.clamp(-lim,lim)).abs().max()))
        old = a.kp_factor.clone(); a.reset(torch.tensor([1,3]))
        torch.testing.assert_close(a.kp_factor,old)
    gym_tree=ast.parse((GYM/'envs/legged_robot.py').read_text())
    method=next(n for cls in gym_tree.body if isinstance(cls,ast.ClassDef)
                for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_compute_torques')
    namespace={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),
                 str(GYM/'envs/legged_robot.py'),'exec'),namespace)
    perm=[0,1,2,4,5,6,3,7]
    action=torch.randn(N,8); q=torch.randn(N,8); qd=torch.randn(N,8)*20
    defaults=torch.tensor([0.,.8,-1.5,0.,0.,.8,-1.5,0.]).repeat(N,1)
    gs=NS(cfg=NS(control=NS(use_filter=False,action_scale=.5,hip_scale_reduction=.5,control_type='P'),
                 domain_rand=NS(randomize_lag_timesteps=False,randomize_kpkd=True)),
        default_dof_pos=defaults,dof_pos=q,dof_vel=qd,p_gains=torch.full((8,),40.),
        d_gains=torch.ones(8),torque_limits=torch.tensor([60.,60.,60.,15.,60.,60.,60.,15.]))
    for attr in ('motor_strength','kp_factor','kd_factor'):
        values=torch.empty(N,8)
        values[:,perm[:6]]=getattr(groups['legs'],attr)
        values[:,perm[6:]]=getattr(groups['wheels'],attr)
        setattr(gs,attr,values)
    legacy=namespace['_compute_torques'](gs,action)[:,perm]
    ap=action[:,perm]; qp=defaults[:,perm]+ap*torch.tensor([.25,.5,.5,.25,.5,.5,.5,.5])
    outputs=[]
    for name,sl in [('legs',slice(0,6)),('wheels',slice(6,8))]:
        ff=ap[:,sl]*5 if name=='wheels' else torch.zeros(N,6)
        out=groups[name].compute(ArticulationActions(joint_positions=qp[:,sl],
            joint_velocities=torch.zeros_like(ff),joint_efforts=ff),q[:,perm][:,sl],qd[:,perm][:,sl])
        outputs.append(out.joint_efforts)
    actual=torch.cat(outputs,1)
    torch.testing.assert_close(actual,legacy,atol=8e-6,rtol=2e-5)
    old = groups['legs'].kp_factor.clone()
    E.randomize_tita_motor_factors(en,torch.tensor([1,3]),ecfg,(1.,1.),(1.,1.),(1.,1.))
    assert torch.equal(old[0],groups['legs'].kp_factor[0])
    assert torch.equal(groups['legs'].kp_factor[1],torch.ones(6))
    return {'formula_max_error': max(errors),'gym_method_max_error':float((actual-legacy).abs().max()),
            'gym_filter_and_lag_disabled':True,'groups': [6,2], 'factors_keep_across_reset': True,
            'partial_selection': True}
check('motor_randomization_and_compute', motors)

def events():
    class View:
        def __init__(self): self.materials=torch.zeros(N,9,3)
        def get_material_properties(self): return wp.from_torch(self.materials)
        def set_material_properties(self, values, ids):
            self.materials[wp.to_torch(ids).long()] = wp.to_torch(values)[wp.to_torch(ids).long()]
    asset.root_view = View()
    cfg = C.EventCfg().randomize_material.copy()
    instance = E.RandomizeTitaMaterial(cfg,env)
    instance(env,None,**cfg.params)
    materials = asset.root_view.materials
    assert (materials[:,:,0]==materials[:,:,1]).all()
    assert (materials[:,1:,:]==materials[:,:1,:]).all()
    assert materials[:,:,0].min()>=.2 and materials[:,:,0].max()<=2.75
    assert materials[:,:,2].min()>=0 and materials[:,:,2].max()<=1
    saved = materials.clone(); instance(env,torch.tensor([1,3]),**cfg.params)
    torch.testing.assert_close(materials,saved)
    # Use the real native mass and CoM classes, with tensor-backed write endpoints.
    for key in ('add_base_mass','base_com'):
        term = getattr(C.EventCfg(),key).copy()
        term.params['asset_cfg'].resolve(scene)
        obj = term.func(cfg=term,env=env)
        obj(env,None,**term.params)
    assert asset.data.body_mass.torch[:,0].min()>=12.2
    assert asset.data.body_mass.torch[:,0].max()<=16.2
    assert (asset.data.body_mass.torch[:,1:]==1).all()
    assert asset.data.body_com_pose_b.torch[:,0,:3].abs().max()<=.1
    assert not asset.data.body_com_pose_b.torch[:,1:,:3].any()
    old=asset.data.root_vel_w.torch.clone()
    E.push_by_setting_xy_velocity(env,torch.tensor([1,3]),(-1.,1.),SceneEntityCfg('robot'))
    torch.testing.assert_close(asset.data.root_vel_w.torch[[1,3],2:],old[[1,3],2:])
    torch.testing.assert_close(asset.data.root_vel_w.torch[0],old[0])
    return {'material_shared_per_robot': True,'friction_range': [.2,2.75],
            'restitution_range':[0,1], 'native_mass_com_terms': True,'partial_push_preserves_other_columns':True}
check('material_mass_com_push',events)

def usd_and_paths():
    usd = PKG/'assets/data/robots/tita/usd/tita.usd'
    s=Usd.Stage.Open(str(usd))
    bodies=[p for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    cols=[p for p in s.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    tree=ET.parse(GYM/'resources/tita/urdf/tita_description.urdf')
    original_cols={link.attrib['name']:len(link.findall('collision')) for link in tree.findall('link')}
    root=s.GetPrimAtPath('/TITARobot/Physics/root_joint')
    assert not UsdPhysics.Joint(root).GetJointEnabledAttr().Get()
    physics_attrs={str(p.GetPath()):{a.GetName():str(a.Get()) for a in p.GetAttributes()
        if a.HasAuthoredValueOpinion() and ('velocity' in a.GetName().lower() or 'damping' in a.GetName().lower())}
        for p in bodies+[p for p in s.Traverse() if p.IsA(UsdPhysics.Joint)]}
    # Actual resolver with an in-memory stage, no PhysX simulation view.
    import isaaclab.sim.utils.queries as queries
    from isaaclab.sim.simulation_context import SimulationContext
    with patch.object(queries,'get_current_stage',return_value=s), \
         patch.object(SimulationContext,'instance',return_value=NS(get_clone_plan=lambda:None)):
        missing=queries.resolve_matching_prims_from_source('/TITARobot/Geometry/base_link/height_scanner',raise_if_no_matches=False)
        existing=queries.resolve_matching_prims_from_source('/TITARobot/Geometry/base_link')
        selected=queries.resolve_matching_prims_from_source('/TITARobot',predicate=lambda p:p.HasAPI(UsdPhysics.RigidBodyAPI))
    starts,_=grid_pattern(C.VelocitySceneCfg().height_scanner.pattern_cfg,'cpu')
    assert starts.shape==(187,3)
    return {'body_count':len(bodies),'usd_colliders':len(cols),'urdf_collision_elements':original_cols,
            'root_joint_disabled':True,'scanner_child_matches_before_spawn':len(missing),
            'base_matches':len(existing),'contact_descendant_candidates':len(selected),
            'height_rays':187,'authored_velocity_damping':physics_attrs}
check('usd_geometry_and_sensor_paths',usd_and_paths)

def raycaster_creation():
    from isaaclab.sensors.sensor_base import SensorBase
    from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
    from isaaclab.sim.simulation_context import SimulationContext
    import isaaclab.sim as sim_utils
    import isaaclab.sim.utils.queries as queries
    # Backend module imports Kit's carb. Execute unchanged method AST against the
    # native BaseRayCaster while replacing only the backend import dependency.
    path=LAB/'isaaclab_physx/isaaclab_physx/sensors/ray_caster/ray_caster.py'
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
           and n.name in ('_has_rigid_body_api','_physx_body_glob','_PhysXRayCasterMixin','RayCaster')]
    manager=NS(get_physics_sim_view=lambda:NS(create_rigid_body_view=lambda glob:NS(count=1)))
    namespace={'BaseRayCaster':BaseRayCaster,'sim_utils':sim_utils,'torch':torch,'wp':wp,
               'UsdPhysics':UsdPhysics,'PhysxManager':manager}
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    RayCaster=namespace['RayCaster']
    s=Usd.Stage.Open(str(PKG/'assets/data/robots/tita/usd/tita.usd'))
    cfg=C.VelocitySceneCfg().height_scanner.copy()
    cfg.prim_path='/TITARobot/Geometry/base_link/height_scanner'
    with patch.object(sim_utils,'get_current_stage',return_value=s), \
         patch.object(queries,'get_current_stage',return_value=s), \
         patch.object(SensorBase,'_register_callbacks'), \
         patch.object(SensorBase,'_clear_callbacks'), \
         patch.object(SensorBase,'set_debug_vis',return_value=False), \
         patch.object(SimulationContext,'instance',return_value=NS(get_clone_plan=lambda:None)):
        ray=RayCaster(cfg)
        created=bool(s.GetPrimAtPath(cfg.prim_path))
        assert not created
        try:
            ray._initialize_pose_tracking()
            raise AssertionError('Missing sensor path unexpectedly initialized')
        except RuntimeError as exc:
            missing_message=str(exc)
            assert 'No prim found' in missing_message
        ray.cfg.prim_path='/TITARobot/Geometry/base_link'
        ray.cfg.spawn=None
        ray._device='cpu'
        ray._initialize_pose_tracking()
        assert ray._view_count==1
        del ray
    return {'constructor_created_sensor_frame':created, 'existing_config_error':missing_message,
            'proposed_base_path_pose_tracking':True,
            'boundary':'Native method AST plus native BaseRayCaster; lifecycle callbacks and PhysX view replaced'}
check('raycaster_constructor_and_pose_tracking',raycaster_creation)

def reset_events():
    cfg=C.EventCfg()
    asset.data.default_joint_pos=proxy(torch.tensor([0.,.8,-1.5,0.,.8,-1.5,0.,0.]).repeat(N,1))
    low=torch.tensor([-.785398,-1.919862,-2.670354,-.785398,-1.919862,-2.670354,-999999.,-999999.])
    high=torch.tensor([.785398,3.490659,-.698132,.785398,3.490659,-.698132,999999.,999999.])
    asset.data.soft_joint_pos_limits=proxy(torch.stack((low,high),-1).repeat(N,1,1))
    asset.data.soft_joint_vel_limits=proxy(torch.tensor([25.]*6+[20.]*2).repeat(N,1))
    asset.data.default_root_pose=proxy(torch.tensor([0.,0.,.3,0.,0.,0.,1.]).repeat(N,1))
    asset.data.default_root_vel=proxy(torch.zeros(N,6))
    writes={}
    asset.write_joint_position_to_sim_index=lambda **kw:writes.update(joint_pos=kw['position'])
    asset.write_joint_velocity_to_sim_index=lambda **kw:writes.update(joint_vel=kw['velocity'])
    asset.write_root_pose_to_sim_index=lambda **kw:writes.update(root_pose=kw['root_pose'])
    ids=torch.tensor([0,3,19],dtype=torch.int32)
    manager=EventManager({'reset_robot_joints':cfg.reset_robot_joints,'reset_base':cfg.reset_base},env)
    manager.apply('reset',env_ids=ids,global_env_step_count=0)
    q=writes['joint_pos']; qd=writes['joint_vel']; pose=writes['root_pose']
    assert not qd.any() and not q[:,[0,3,6,7]].any()
    assert q[:,[1,4]].min()>=.4 and q[:,[1,4]].max()<=1.2
    assert q[:,[2,5]].min()>=-2.25 and q[:,[2,5]].max()<=-.75
    assert pose[:,:2].abs().max()<=1
    assert pose[:,2].min()>=.3 and pose[:,2].max()<=.5
    torch.testing.assert_close(pose[:,3:],torch.tensor([0.,0.,0.,1.]).repeat(len(ids),1))
    return {'native_event_manager':True,'ids':[0,3,19],'joint_ranges_preserved':True,
            'root_xyz_quaternion':True,'native_soft_clamp_inactive_for_current_ranges':True}
check('native_reset_events',reset_events)

def interval_schedule():
    cfg=C.EventCfg().push_robot.copy(); calls=[]; counter=0
    def spy(env,env_ids,velocity_range,asset_cfg):
        calls.append(counter)
    cfg.func=spy
    manager=EventManager({'push_robot':cfg},env)
    for counter in range(1,3002):
        if counter==800:manager.reset(torch.tensor([1,3]))
        manager.apply('interval',dt=.01)
    return {'trigger_steps':calls,'reset_step':800,'global_timer_survives_partial_reset':len(calls)==2}
check('native_interval_schedule',interval_schedule)

def engine_defaults():
    schemas=Path('/home/eaziel/miniconda3/envs/env_isaacSL-6.1.0/lib/python3.12/site-packages/isaacsim/extscache/omni.usd.schema.physx-110.3.2+110.3.0.lx64.r.cp312.u7f4/plugins/PhysxSchema/resources/generatedSchema.usda')
    layer=Sdf.Layer.FindOrOpen(str(schemas))
    out={}
    for cls,name in [('PhysxRigidBodyAPI','angularDamping'),('PhysxRigidBodyAPI','maxAngularVelocity'),
                     ('PhysxJointAPI','maxJointVelocity'),('PhysxArticulationAPI','enabledSelfCollisions')]:
        prim=layer.GetPrimAtPath('/'+cls)
        full=('physxRigidBody:' if cls=='PhysxRigidBodyAPI' else 'physxJoint:' if cls=='PhysxJointAPI' else 'physxArticulation:')+name
        out[full]=str(prim.properties[full].default)
    return out
check('local_physx_schema_defaults',engine_defaults)

def proposed_config_corrections():
    import math
    from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
    cfg=C.VelocityEnvCfg()
    cfg.scene.height_scanner.prim_path='{ENV_REGEX_NS}/Robot/Geometry/base_link'
    cfg.scene.height_scanner.spawn=None
    cfg.scene.robot.actuators['legs'].joint_velocity_limit=25.
    cfg.scene.robot.actuators['wheels'].joint_velocity_limit=20.
    cfg.scene.robot.spawn.rigid_props={'/.*':[PhysxRigidBodyCfg(linear_damping=0.,angular_damping=0.,
        max_linear_velocity=1000.,max_angular_velocity=math.degrees(1000.),max_depenetration_velocity=1.)]}
    cfg.validate()
    from TITA_Lab.assets.tita import TITA_CFG
    asset_cfg=TITA_CFG.replace(prim_path='/World/envs/env_0/Robot')
    asset_cfg.actuators['legs'].joint_velocity_limit=25.
    asset_cfg.actuators['wheels'].joint_velocity_limit=20.
    asset_cfg.validate()
    return {'configuration_validation':True,'angular_velocity_deg_s':math.degrees(1000.),
            'source_files_written':False}
check('proposed_config_corrections',proposed_config_corrections)

(OUT/'result.json').write_text(json.dumps(RESULT,ensure_ascii=False,indent=2))
print(json.dumps({'passed':len(RESULT['checks']),'failed':list(RESULT['failures'])},ensure_ascii=False))
