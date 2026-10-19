# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import (
    ArticulationCfg,  # BaseArticulation：与其他后端（如Newton Mjwarp）共用的基类
)
from isaaclab_physx.sim.schemas import (
    PhysxArticulationCfg,
    PhysxCollisionCfg,
    PhysxRigidBodyCfg,
)

from . import TITA_LAB_ASSETS_DIR  # . 表示当前的 TITA_Lab.assets 包。这个路径常量已在它的 __init__.py 中定义

TITA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(TITA_LAB_ASSETS_DIR / "robots" / "tita" / "usd" / "tita.usd"),
        articulation_props={
            "/.*": [
                PhysxArticulationCfg(
                    solver_position_iteration_count=4,
                    solver_velocity_iteration_count=0,
                )
            ]
        },
        rigid_props={"/.*": [PhysxRigidBodyCfg(max_depenetration_velocity=1.0)]},
        collision_props={"/.*": [PhysxCollisionCfg(contact_offset=0.01, rest_offset=0.0)]},
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint_.*_leg_1": 0.0,
            "joint_.*_leg_2": 0.8,
            "joint_.*_leg_3": -1.5,
            "joint_.*_leg_4": 0.0,
        },
        pos=(0.0, 0.0, 0.3),
    ),
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=["joint_.*_leg_[123]"],
            stiffness=40.0,
            damping=1.0,
            actuator_effort_limit=60.0,
            joint_effort_limit=60.0,
        ),
        "wheels": IdealPDActuatorCfg(
            joint_names_expr=["joint_.*_leg_4"],
            stiffness=0.0,
            damping=0.5,
            actuator_effort_limit=15.0,
            joint_effort_limit=15.0,
        ),
    },
)
