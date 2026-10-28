# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "joint_pos_target_l2",
    "joint_pos_rel_zero_wheels",
    "proprio_history",
    "illegal_contact_current",
    "undesired_contacts_current",
    "joint_power_signed",
    "JointAccelerationPenalty",
    "push_by_setting_xy_velocity",
    "RandomizeTitaMaterial",
    "randomize_tita_motor_factors",
    "TitaJointActionCfg",
    "action_rate_clipped_l2",
    "randomize_tita_action_lag",
    "TitaDynamicsParameters",
    "TitaPrivilegedState",
    "joint_position_limit_cost",
    "joint_effort_limit_cost",
    "joint_velocity_limit_cost",
    "wheel_contact_force_cost",
    "wheel_stumble_cost",
    "JointAccelerationCost",
    "terrain_levels_vel",
    "TitaVelocityCommandCfg",
]

# Forward stable MDP terms lazily, then override with environment-specific terms below.
from isaaclab.envs.mdp import *  # noqa: F401, F403

from isaaclab_tasks.core.velocity.mdp.curriculums import terrain_levels_vel

from .actions import TitaJointActionCfg
from .commands import TitaVelocityCommandCfg
from .costs import (
    JointAccelerationCost,
    joint_effort_limit_cost,
    joint_position_limit_cost,
    joint_velocity_limit_cost,
    wheel_contact_force_cost,
    wheel_stumble_cost,
)
from .events import (
    RandomizeTitaMaterial,
    push_by_setting_xy_velocity,
    randomize_tita_action_lag,
    randomize_tita_motor_factors,
)
from .observations import TitaDynamicsParameters, TitaPrivilegedState, joint_pos_rel_zero_wheels, proprio_history
from .rewards import (
    JointAccelerationPenalty,
    action_rate_clipped_l2,
    joint_pos_target_l2,
    joint_power_signed,
    undesired_contacts_current,
)
from .terminations import illegal_contact_current
