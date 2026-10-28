"""Check the custom-actuator execution gate without starting a physics backend."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from pxr import Usd, UsdGeom

import TITA_Lab.tasks  # noqa: F401
from isaaclab.sim import SimulationContext
from isaaclab.sim.schemas.schemas_actuators import define_actuator_properties
from isaaclab_physx.assets.articulation.actuator_control import PhysxActuatorControl
from isaaclab_tasks.utils import parse_env_cfg

OUT = Path(__file__).resolve().parent
cfg = parse_env_cfg('TitaLab-Velocity-Tita', device='cpu', num_envs=4,
                    overrides=['physics=isaacsim_physx'])
groups = cfg.scene.robot.actuators
stage = Usd.Stage.CreateInMemory()
prim = UsdGeom.Xform.Define(stage, '/Robot').GetPrim()
result = {'boundary': 'Native authoring gate and control-selection methods with CPU stand-ins only; no simulation',
          'saved_use_newton_actuators': cfg.sim.use_newton_actuators,
          'physics_config': type(cfg.sim.physics).__name__}

for enabled in (True, False):
    sim_cfg = cfg.sim.copy()
    sim_cfg.use_newton_actuators = enabled
    observed = {}
    with patch.object(SimulationContext, 'instance', return_value=NS(cfg=sim_cfg)), \
         patch('isaaclab.sim.utils.queries.find_first_matching_prim', return_value=prim):
        try:
            define_actuator_properties('/Robot', groups, stage=stage)
            observed['authoring'] = 'returned normally'
        except ValueError as exc:
            observed['authoring'] = str(exc)
            assert enabled and 'TitaPDActuatorCfg' in str(exc)
    control = object.__new__(PhysxActuatorControl)
    control._articulation = NS(_sim_cfg=sim_cfg)
    try:
        native_groups = control.prepare_native_actuators(None, groups)
        observed['control_native_groups'] = sorted(native_groups)
        assert not enabled and native_groups == set()
    except ValueError as exc:
        observed['control_selection'] = str(exc)
        assert enabled and 'TitaPDActuatorCfg' in str(exc)
    if enabled:
        assert 'Newton-native actuator execution does not support' in observed['authoring']
    else:
        assert observed['authoring'] == 'returned normally'
    result[str(enabled)] = observed

candidate = cfg.copy()
candidate.sim.use_newton_actuators = False
candidate.validate()
result['candidate_validate_passed'] = True
(OUT/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(result, indent=2, ensure_ascii=False))
