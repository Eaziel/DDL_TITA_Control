"""Exercise the new observation on native RL step/reset control flow with CPU state stand-ins."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'TITA_Lab/src'))

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.managers import (ActionManager, EventManager, EventTermCfg, ObservationManager,
    ObservationGroupCfg, ObservationTermCfg, RewardManager, RewardTermCfg,
    SceneEntityCfg, TerminationManager, TerminationTermCfg)  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.string import resolve_matching_names  # noqa: E402


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


A=load('tita_pending_lag_actions',ROOT/'docs/audits/2026-10-20_action_lag/proposal_actions.py')
S=load('tita_privileged_state_candidate',OUT/'proposal_state.py')
S.TitaJointAction=A.TitaJointAction
P=load('tita_push_capture_candidate',OUT/'proposal_push.py')
G=load('tita_privileged_group_candidate',OUT/'proposal_cfg.py')
L=load('tita_layout_candidate',OUT/'proposal_layout.py')

N=4
PHYS_NAMES=['joint_left_leg_1','joint_right_leg_1','joint_left_leg_2','joint_right_leg_2',
            'joint_left_leg_3','joint_right_leg_3','joint_left_leg_4','joint_right_leg_4']
POLICY_NAMES=[PHYS_NAMES[i] for i in [0,2,4,1,3,5,6,7]]


def proxy(t):
    return NS(torch=t)


class IndexProxy:
    def __init__(self,ids):
        self.torch=torch.tensor(ids,dtype=torch.long)
    def __len__(self):
        return len(self.torch)


def zero_reward(env):
    return torch.zeros(N)


def failures(env):
    mask=torch.zeros(N,dtype=torch.bool)
    if env.common_step_counter in (2,5,9):
        mask[[1,3]]=True
    return mask


def reset_velocity(env, env_ids):
    env.scene['robot'].data.root_vel_w.torch[env_ids,:3]=100.


def constant_observation(env, key):
    return env.constants[key]


class Asset:
    num_joints=8
    joint_names=PHYS_NAMES
    device='cpu'
    def __init__(self, env):
        self.env=env
        vel=torch.zeros(N,6)
        self.data=NS(default_joint_pos=proxy(torch.zeros(N,8)),root_vel_w=proxy(vel),
                     root_lin_vel_b=proxy(vel[:,:3]))
    def find_joints(self,names,preserve_order=False,as_proxy=False):
        ids,names=resolve_matching_names(names,self.joint_names,preserve_order=preserve_order)
        return (IndexProxy(ids) if as_proxy else ids),names
    def set_joint_position_target_index(self,target,joint_ids):
        pass
    def set_joint_effort_target_index(self,target,joint_ids):
        pass
    def write_root_velocity_to_sim_index(self,root_velocity,env_ids=None):
        selection=slice(None) if env_ids is None else env_ids
        self.data.root_vel_w.torch[selection]=root_velocity
        self.env.push_steps.append(self.env.common_step_counter)


class Sensor:
    num_bodies=3
    body_names=['right_leg_4','base_link','left_leg_4']
    def __init__(self):
        self.data=NS(net_normal_forces_w=proxy(torch.zeros(N,3,3)))
    def find_bodies(self,names,preserve_order=False):
        return resolve_matching_names(names,self.body_names,preserve_order=preserve_order)


class Scene(dict):
    num_envs=N
    @property
    def sensors(self):
        return self
    def write_data_to_sim(self):
        pass
    def update(self,dt):
        pass
    def reset(self,env_ids):
        self['contact_forces'].data.net_normal_forces_w.torch[env_ids]=0.


def physical_values(step):
    rows=torch.arange(N,dtype=torch.float32)
    velocity=torch.stack((step+rows*.1,-.2*rows,torch.full((N,),.3)),dim=1)
    left=((step+torch.arange(N))%4<2)
    right=((step+torch.arange(N))%3==0)
    forces=torch.zeros(N,3,3)
    forces[:,2,2]=left.float()*2.
    forces[:,0,2]=right.float()*2.
    forces[:,:,0]=100.  # Horizontal force must not substitute for world Z threshold.
    return velocity,forces


class Sim:
    device='cpu'
    is_rendering=False
    def __init__(self,env):
        self.env=env
        self.counter=0
        self.render_context=NS(reset_scene_state_cadence=lambda:None)
    def is_playing(self):
        return True
    def step(self,render=False):
        self.counter+=1
        velocity,forces=physical_values((self.counter-1)//4+1)
        self.env.scene['robot'].data.root_vel_w.torch[:,:3]=velocity
        self.env.scene['contact_forces'].data.net_normal_forces_w.torch.copy_(forces)
    def consume_reset_request(self):
        return False


class CpuEnv(ManagerBasedRLEnv):
    def __init__(self,final_obs=False):
        self.cfg=NS(decimation=4,sim=NS(dt=.0025,render_interval=4),
                    compute_final_obs=final_obs,num_rerenders_on_reset=0,episode_length_s=20.)
        self.common_step_counter=0
        self._sim_step_counter=0
        self._physics_handles_decimation=False
        self.episode_length_buf=torch.zeros(N,dtype=torch.long)
        self.extras={}
        self.render_enabled=False
        self.has_rtx_sensors=False
        self.video_recorders=[]
        self.push_steps=[]
        self.reset_steps=[]
        self.scene=Scene(robot=Asset(self),contact_forces=Sensor())
        self.sim=Sim(self)
        self.action_lag_indices=torch.tensor([0,1,2,0],dtype=torch.long)
        self.command_manager=NS(compute=lambda **kw:None,reset=lambda *args,**kw:{})
        self.curriculum_manager=NS(compute=lambda **kw:None,reset=lambda *args,**kw:{})
        self.proprio_history=NS(reset=lambda ids:None)
        self.recorder_manager=NS(active_terms=[],reset=lambda *args,**kw:{})
        for name in ('record_pre_step','record_post_physics_decimation_step','record_pre_reset',
                     'record_post_reset'):
            setattr(self.recorder_manager,name,lambda *args,**kwargs:None)

        self.action_manager=ActionManager(NS(
            leg_position=A.TitaJointActionCfg(asset_name='robot',joint_names=POLICY_NAMES[:6],
                preserve_order=True,command_type='position',lag_timesteps=3),
            wheel_effort=A.TitaJointActionCfg(asset_name='robot',joint_names=POLICY_NAMES[6:],
                preserve_order=True,command_type='effort',lag_timesteps=3)),self)
        self.event_manager=EventManager(NS(
            reset_velocity=EventTermCfg(func=reset_velocity,mode='reset'),
            push=EventTermCfg(func=P.push_by_setting_xy_velocity,mode='interval',
                interval_range_s=(.03,.03),is_global_time=True,
                params={'velocity_range':(-1.,1.),'asset_cfg':SceneEntityCfg('robot')})),self)
        self.termination_manager=TerminationManager(NS(failure=TerminationTermCfg(func=failures)),self)
        self.reward_manager=RewardManager(NS(zero=RewardTermCfg(func=zero_reward,weight=1.)),self)
        self.constants={key:torch.full((N,size),float(i)) for i,(key,size) in enumerate(
            [('policy',33),('scan',187),('dynamics',30),('history',330)])}
        self.constants['policy'][:,0]=150.
        groups={}
        for key in self.constants:
            groups[key]=ObservationGroupCfg(concatenate_terms=True,enable_corruption=False)
            setattr(groups[key],'value',ObservationTermCfg(func=constant_observation,params={'key':key}))
        groups['privileged_state']=G.make_group(S.TitaPrivilegedState)
        self.observation_manager=ObservationManager(NS(**groups),self)

    def _reset_idx(self,env_ids):
        # Exact proposed hook followed by the actual native reset implementation.
        self.observation_manager.compute_group('privileged_state')
        self.reset_steps.append(self.common_step_counter)
        super()._reset_idx(env_ids)
        self.proprio_history.reset(env_ids)

    def close(self):
        pass


# Extract the source Gym contact update and the first six privileged features.
tree=ast.parse(Path('/home/eaziel/Downloads/tita_rl-master/envs/legged_robot.py').read_text())
robot_cls=next(c for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='LeggedRobot')
post=next(n for n in robot_cls.body if isinstance(n,ast.FunctionDef) and n.name=='post_physics_step')
updates=[]
for n in post.body:
    if isinstance(n,ast.Assign):
        target=n.targets[0]
        if (isinstance(target,ast.Name) and target.id=='contact') or (
            isinstance(target,ast.Attribute) and target.attr in ('contact_filt','last_contacts')):
            updates.append(n)
obsfn=next(n for n in robot_cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute_observations')
priv=next(n.value for n in obsfn.body if isinstance(n,ast.Assign)
          and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='priv_latent')
priv.args[0].elts=priv.args[0].elts[:3]
update_code=compile(ast.fix_missing_locations(ast.Module(body=updates,type_ignores=[])),'gym_contact_update','exec')
feature_code=compile(ast.fix_missing_locations(ast.Expression(body=priv)),'gym_privileged_six','eval')


def exercise(final_obs):
    env=CpuEnv(final_obs)
    cold=env.observation_manager.compute_group('privileged_state')
    torch.testing.assert_close(cold[:,:3],torch.zeros(N,3))
    torch.testing.assert_close(cold[:,3:5],torch.full((N,2),-.5))
    old=NS(feet_indices=[2,0],last_contacts=torch.zeros(N,2,dtype=torch.bool),
        obs_scales=NS(lin_vel=2.),reindex_feet=lambda t:t[:,[1,0]],
        randomized_lag_tensor=env.action_lag_indices.float()[:,None]/2.)
    max_error=0.
    for step in range(1,13):
        velocity,forces=physical_values(step)
        old.contact_forces=forces
        old.base_lin_vel=velocity
        local={'self':old,'torch':torch}
        exec(update_code,local)
        expected=eval(feature_code,local)[:,[0,1,2,4,3,5]] # right/left -> task left/right
        obs,_,_,_,_=env.step(torch.ones(N,8))
        torch.testing.assert_close(obs['privileged_state'],expected)
        max_error=max(max_error,(obs['privileged_state']-expected).abs().max().item())
        for _ in range(3):
            torch.testing.assert_close(env.observation_manager.compute_group('privileged_state'),expected)
        packed=L.pack_tita_observations(obs)
        assert packed.shape==(N,586)
        torch.testing.assert_close(packed[:,220:226],expected)
        torch.testing.assert_close(packed[:,226:256],obs['dynamics'])
        assert packed[:,0].eq(100.).all()
        assert obs['policy'][:,0].eq(150.).all()
        assert torch.isfinite(packed).all()
    assert env.reset_steps==[2,5,9]
    assert len(env.push_steps)>=3
    env.close()
    return {'policy_steps':12,'physics_steps':48,'partial_resets':env.reset_steps,
            'push_steps':env.push_steps,'max_error_against_gym_six_after_wheel_permutation':max_error,
            'repeated_reads':True,'final_obs_enabled':final_obs,
            'packing_shape':[N,586],'internal_policy_unclipped':True}


result={
 'scope':'Pending candidate composition, not a review of user source edits',
 'boundary':'Native ManagerBasedRLEnv.step/_reset_idx plus native managers on CPU; simulator/state/reset setters are stand-ins; no AppLauncher or training',
 'runs':[exercise(False),exercise(True)],
 'definitions':{'state_order':['vx*2','vy*2','vz*2','left_contact-.5','right_contact-.5','lag_slot/(L-1)'],
                'contact_threshold':'world normal Fz > 1 N; previous raw contact OR current raw contact',
                'cold_start':'zero velocity and false contacts until first completed policy step',
                'reset':'keep contact memory and sampled state, matching the old auto-reset rule'},
 'limits':['normal contact force differs from Gym total force',
           'task joint/contact order still needs conversion for old checkpoints',
           'initial/public reset differs from Gym reset that executes an extra zero-action step',
           'future state-changing events must preserve this sampling boundary'],
}
try:
    bad={name:torch.zeros(2,size) for name,size in [('policy',33),('scan',187),
        ('privileged_state',5),('dynamics',30),('history',330)]}
    L.pack_tita_observations(bad)
except ValueError:
    result['invalid_shape_rejected']=True
else:
    raise AssertionError('Expected bad shape rejection')
(OUT/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(result,indent=2,ensure_ascii=False))
