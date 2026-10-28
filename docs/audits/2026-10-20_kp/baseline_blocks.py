@configclass
class ActionsCfg:
    """Nominal action mapping for TITA legs and wheels."""
    leg_position = mdp.JointPositionActionCfg(asset_name='robot', joint_names=['joint_left_leg_1', 'joint_left_leg_2', 'joint_left_leg_3', 'joint_right_leg_1', 'joint_right_leg_2', 'joint_right_leg_3'], preserve_order=True, scale={'joint_.*_leg_1': 0.25, 'joint_.*_leg_[23]': 0.5}, use_default_offset=True)
    wheel_effort = mdp.JointEffortActionCfg(asset_name='robot', joint_names=['joint_left_leg_4', 'joint_right_leg_4'], preserve_order=True, scale=5.0)

@configclass
class RewardsCfg:
    """Reward terms for TITA velocity tracking."""
    tracking_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0, params={'command_name': 'base_velocity', 'std': 0.5})
    tracking_ang_vel = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5, params={'command_name': 'base_velocity', 'std': 0.5})
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-1.0, params={'target_height': 0.35, 'sensor_cfg': SceneEntityCfg('height_scanner')})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    collision = RewTerm(func=mdp.undesired_contacts_current, weight=-1.0, params={'sensor_cfg': SceneEntityCfg('contact_forces', body_names=['left_leg_3', 'right_leg_3']), 'threshold': 0.1})
    powers = RewTerm(func=mdp.joint_power_signed, weight=-2e-05)
    dof_acc = RewTerm(func=mdp.JointAccelerationPenalty, weight=-2.5e-07)
    termination = RewTerm(func=mdp.is_terminated_term, weight=-200.0, params={'term_keys': 'base_contact'})