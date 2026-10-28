"""CPU parity checks for proposed terrain and dynamics observation groups."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
wp.config.kernel_cache_dir = '/tmp/tita_training_observations_warp_cache'
sys.path.insert(0, str(ROOT/'TITA_Lab/src'))

import TITA_Lab.tasks  # noqa: E402,F401
from TITA_Lab.tasks.velocity.actuators import TitaPDActuator  # noqa: E402
from TITA_Lab.tasks.velocity.config.TITA import env_cfg as C  # noqa: E402
from TITA_Lab.tasks.velocity.mdp.observation_history import ProprioHistory  # noqa: E402
from isaaclab.managers import EventManager, ObservationManager  # noqa: E402
from isaaclab.sensors.ray_caster import kernels  # noqa: E402
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster  # noqa: E402
from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData  # noqa: E402
from isaaclab.utils.string import resolve_matching_names  # noqa: E402


def load(name, filename):
    spec=importlib.util.spec_from_file_location(name, OUT/filename)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


P=load('tita_dynamics_proposal','proposal_observations.py')
G=load('tita_observation_cfg_proposal','proposal_cfg.py')
torch.manual_seed(20261020)
N=8
PHYS_NAMES=['joint_left_leg_1','joint_right_leg_1','joint_left_leg_2','joint_right_leg_2',
            'joint_left_leg_3','joint_right_leg_3','joint_left_leg_4','joint_right_leg_4']
GYM_TO_PHYS=[0,2,4,6,1,3,5,7]
TASK_TO_PHYS=[0,2,4,1,3,5,6,7]


def proxy(t):
    return NS(torch=t)


class MaterialView:
    def __init__(self):
        self.values=torch.tensor([1.,1.,0.]).repeat(N,9,1)
    def get_material_properties(self):
        return wp.from_torch(self.values)
    def set_material_properties(self, materials, indices):
        ids=wp.to_torch(indices).long()
        self.values[ids]=wp.to_torch(materials)[ids]


class Asset:
    device='cpu'
    num_joints=8
    num_bodies=9
    joint_names=PHYS_NAMES
    body_names=['base_link']+[f'link_{i}' for i in range(8)]
    def __init__(self):
        poses=torch.tensor([.02,-.01,.03,0.,0.,0.,1.]).repeat(N,9,1)
        self.data=NS(
            joint_pos=proxy(torch.randn(N,8)), default_joint_pos=proxy(torch.zeros(N,8)),
            joint_vel=proxy(torch.randn(N,8)), root_ang_vel_b=proxy(torch.randn(N,3)),
            projected_gravity_b=proxy(torch.randn(N,3)),
            body_mass=proxy(torch.tensor([13.2]+[1.]*8).repeat(N,1)),
            body_inertia=proxy(torch.eye(3).flatten().repeat(N,9,1)),
            body_com_pose_b=proxy(poses), body_com_pos_b=proxy(poses[:,:,:3]),
        )
        self.root_view=MaterialView()
        self.actuators={}
        for name, ids in [('legs',list(range(6))),('wheels',[6,7])]:
            self.actuators[name]=TitaPDActuator(
                C.VelocitySceneCfg().robot.actuators[name].copy(),
                joint_names=[PHYS_NAMES[i] for i in ids], joint_ids=torch.tensor(ids),
                num_envs=N, device='cpu', stiffness=0., damping=0.,
            )
    def find_joints(self, names, preserve_order=False, **kwargs):
        return resolve_matching_names(names,self.joint_names,preserve_order=preserve_order)
    def find_bodies(self, names, preserve_order=False):
        return resolve_matching_names(names,self.body_names,preserve_order=preserve_order)
    def set_masses_index(self, masses, body_ids, env_ids):
        self.data.body_mass.torch[env_ids[:,None],body_ids]=masses
    def set_inertias_index(self, inertias, body_ids, env_ids):
        self.data.body_inertia.torch[env_ids[:,None],body_ids]=inertias
    def set_coms_index(self, coms, body_ids, env_ids):
        self.data.body_com_pose_b.torch[env_ids[:,None],body_ids]=coms


class Scene(dict):
    num_envs=N
    @property
    def sensors(self):
        return self


asset=Asset()
scan_cfg=C.VelocitySceneCfg().height_scanner.copy()
sensor=NS(cfg=scan_cfg,_device='cpu',_view_count=N,_num_envs=N)
sensor._data=RayCasterData()
BaseRayCaster._initialize_rays_impl(sensor)
sensor._offset_pos_wp=wp.zeros(N,dtype=wp.vec3f,device='cpu')
quat=torch.zeros(N,4);quat[:,3]=1.
sensor._offset_quat_wp=wp.from_torch(quat,dtype=wp.quatf)
poses=torch.zeros(N,7);poses[:,2]=torch.linspace(.25,.75,N);poses[:,6]=1.
sensor._get_view_transforms_wp=lambda:wp.from_torch(poses).view(wp.transformf)
sensor._alignment_mode=kernels.ALIGNMENT_YAW
BaseRayCaster._update_ray_infos(sensor,wp.ones(N,dtype=wp.bool,device='cpu'))
torch.testing.assert_close(sensor._data.pos_w.torch,poses[:,:3])
torch.testing.assert_close(wp.to_torch(sensor._ray_starts_w)[:,:,2],poses[:,2,None]+20.+torch.zeros(N,187))
sensor.data=sensor._data
hits=sensor.data.ray_hits_w.torch
hits[:,:,2]=torch.linspace(-2.,2.,187).repeat(N,1)
scene=Scene(robot=asset,height_scanner=sensor)
commands=torch.randn(N,3)
env=NS(num_envs=N,device='cpu',scene=scene,
       sim=NS(is_playing=lambda:True,physics_manager=type('PhysxManager',(),{})),
       command_manager=NS(get_command=lambda name:commands),
       action_manager=NS(action=torch.randn(N,8)),
       proprio_history=ProprioHistory(N,33,10))

ecfg=C.EventCfg()
for name in ('reset_robot_joints','push_robot','reset_base'):
    setattr(ecfg,name,None)
env.event_manager=EventManager(ecfg,env)
ocfg=C.ObservationsCfg()
ocfg.policy.enable_corruption=False
ocfg.scan,ocfg.dynamics=G.make_groups(P.TitaDynamicsParameters,C.TITA_JOINT_NAMES)
manager=ObservationManager(ocfg,env)
before=manager.compute()
torch.testing.assert_close(before['dynamics'][:,:4],torch.zeros(N,4))
cfg=C.VelocityEnvCfg()
cfg.observations=ocfg
cfg.validate()

env.event_manager.apply(mode='startup')
obs=manager.compute()
assert {k:tuple(v.shape) for k,v in obs.items()}=={
    'policy':(N,33),'history':(N,330),'scan':(N,187),'dynamics':(N,30)}
expected_height=(poses[:,2,None]-.4-hits[:,:,2]).clamp(-1.,1.)*5.
torch.testing.assert_close(obs['scan'],expected_height)
assert obs['scan'].min()==-5. and obs['scan'].max()==5.

material=env.event_manager.get_term_cfg('randomize_material').func
mass=env.event_manager.get_term_cfg('add_base_mass').func
com=env.event_manager.get_term_cfg('base_com').func
delta_mass=asset.data.body_mass.torch[:,0:1]-mass.default_mass[:,0:1]
delta_com=asset.data.body_com_pos_b.torch[:,0]-com.default_com[:,0,:3]
factor_blocks=[]
for name in ('motor_strength','kp_factor','kd_factor'):
    values=torch.ones(N,8)
    for actuator in asset.actuators.values():
        values[:,actuator.joint_indices]=getattr(actuator,name)
    # Old Gym stores these in raw DOF order; undo that order before comparing.
    gym_values=values[:,GYM_TO_PHYS]
    old_to_task=[0,1,2,4,5,6,3,7]
    factor_blocks.append(gym_values[:,old_to_task])
expected_dynamics=torch.cat((delta_mass,delta_com,material.friction_coeffs[:,None],
                             material.restitution_coeffs[:,None],*factor_blocks),dim=1)
torch.testing.assert_close(obs['dynamics'],expected_dynamics)
torch.testing.assert_close(asset.root_view.values[:,:,0],material.friction_coeffs[:,None].repeat(1,9))
torch.testing.assert_close(asset.root_view.values[:,:,2],material.restitution_coeffs[:,None].repeat(1,9))

# Read-only computation and inherited reset keep nominal references stable.
kept=obs['dynamics'].clone()
repeated=manager.compute()['dynamics']
torch.testing.assert_close(repeated,kept)
manager.reset(torch.tensor([1,3]))
env.event_manager.reset(torch.tensor([1,3]))
torch.testing.assert_close(manager.compute()['dynamics'],kept)
assert torch.allclose(obs['dynamics'][:,0],delta_mass[:,0])
assert not torch.equal(obs['dynamics'][:,0],asset.data.body_mass.torch[:,0])

# Compare real grid generation with the source Gym height-point routine.
tree=ast.parse(Path('/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py').read_text())
method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body
            if isinstance(n,ast.FunctionDef) and n.name=='_init_height_points')
namespace={'torch':torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),'gym_height_points','exec'),namespace)
legacy=NS(device='cpu',num_envs=N,cfg=NS(terrain=NS(
    measured_points_x=[i/10. for i in range(-8,9)],
    measured_points_y=[i/10. for i in range(-5,6)])))
old_points=namespace['_init_height_points'](legacy)
local_starts=wp.to_torch(sensor.ray_starts.warp)
torch.testing.assert_close(local_starts[:,:,:2],old_points[:,:,:2],atol=1e-6,rtol=1e-6)
assert all(torch.isfinite(value).all() for value in obs.values())

result={
 'scope':'Proposed scan/dynamics groups only; no TITA_Lab source files modified',
 'boundary':'Native CPU ObservationManager/EventManager, real ray transform Warp kernel; asset properties/hits use stand-ins; no simulator or training',
 'checks':{
  'current_environment_with_candidate_groups_validate':True,
  'groups':{k:list(v.shape) for k,v in obs.items()},
  'grid_points_match_gym_order':True,
  'ray_starts_z_offset_m':20.,'sensor_pose_excludes_pattern_offset':True,
  'height_formula_clip_then_scale':True,'height_output_bounds':[-5.,5.],
  'nominal_captured_before_startup':True,'mass_and_com_are_deltas':True,
  'material_values_match_actual_event_writes':True,
  'motor_factors_match_after_explicit_joint_order_conversion':True,
  'repeated_reads_and_partial_reset_preserve_references':True,
  'all_candidate_outputs_finite_for_finite_fake_hits':True,
 },
 'remaining':'Not the full 36 privileged dimensions; base velocity/contact filter/lag still need the agreed layout and timing; final +/-100 clipping and NP3O adapter are not implemented',
}
(OUT/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
