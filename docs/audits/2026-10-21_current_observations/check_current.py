"""Check saved observation code with the previous native CPU fixtures."""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def run_fixture(folder, replacements):
    source_path = ROOT / "docs/audits" / folder
    source_path /= "check_state.py" if "privileged" in folder else "check_observations.py"
    code = source_path.read_text()
    # 复用旧夹具，改为导入用户当前代码；旧证据与候选文件保持不变。
    for old, new in replacements:
        if old not in code:
            raise AssertionError(f"Missing fixture anchor: {old}")
        code = code.replace(old, new, 1)
    namespace = {"__file__": str(source_path), "__name__": "__main__"}
    exec(compile(code, str(source_path), "exec"), namespace)


run_fixture("2026-10-20_privileged_state", [
    ("A=load('tita_pending_lag_actions',ROOT/'docs/audits/2026-10-20_action_lag/proposal_actions.py')",
     "from TITA_Lab.tasks.velocity.mdp import actions as A\n"
     "from TITA_Lab.tasks.velocity import tita_velocity_env as E\n"
     "from TITA_Lab.tasks.velocity.config.TITA import env_cfg as C"),
    ("S=load('tita_privileged_state_candidate',OUT/'proposal_state.py')",
     "from TITA_Lab.tasks.velocity.mdp import observations as S"),
    ("S.TitaJointAction=A.TitaJointAction", "assert S.TitaJointAction is A.TitaJointAction"),
    ("P=load('tita_push_capture_candidate',OUT/'proposal_push.py')",
     "from TITA_Lab.tasks.velocity.mdp import events as P"),
    ("G=load('tita_privileged_group_candidate',OUT/'proposal_cfg.py')",
     "G=NS(make_group=lambda cls: C.ObservationsCfg().privileged_state.copy())"),
    ("L=load('tita_layout_candidate',OUT/'proposal_layout.py')",
     "from TITA_Lab.tasks.velocity import observation_layout as L"),
    ("class CpuEnv(ManagerBasedRLEnv):",
     "class CpuEnv(E.TitaVelocityEnv):\n    step = ManagerBasedRLEnv.step"),
    ("        # Exact proposed hook followed by the actual native reset implementation.\n"
     "        self.observation_manager.compute_group('privileged_state')\n"
     "        self.reset_steps.append(self.common_step_counter)\n"
     "        super()._reset_idx(env_ids)\n"
     "        self.proprio_history.reset(env_ids)",
     "        self.reset_steps.append(self.common_step_counter)\n"
     "        E.TitaVelocityEnv._reset_idx(self, env_ids)"),
    ("'scope':'Pending candidate composition, not a review of user source edits'",
     "'scope':'Current saved action/state/push/reset/group/packing code; other groups are fixture constants'"),
    ("(OUT/'result.json').write_text", f"(Path({str(OUT / 'state_result.json')!r})).write_text"),
])

run_fixture("2026-10-20_training_observations", [
    ("P=load('tita_dynamics_proposal','proposal_observations.py')",
     "from TITA_Lab.tasks.velocity.mdp import observations as P"),
    ("G=load('tita_observation_cfg_proposal','proposal_cfg.py')",
     "G=NS(make_groups=lambda *args: (C.ObservationsCfg().scan.copy(), C.ObservationsCfg().dynamics.copy()))"),
    ("('reset_robot_joints','push_robot','reset_base')",
     "('reset_robot_joints','push_robot','reset_base','randomize_action_lag')"),
    ("ocfg=C.ObservationsCfg()", "ocfg=C.ObservationsCfg()\nocfg.privileged_state=None"),
    ("'scope':'Proposed scan/dynamics groups only; no TITA_Lab source files modified'",
     "'scope':'Current saved scan/dynamics groups and parameter class; six-state checked separately'"),
    ("'remaining':'Not the full 36 privileged dimensions; base velocity/contact filter/lag still need the agreed layout and timing; final +/-100 clipping and NP3O adapter are not implemented'",
     "'remaining':'State/packing checked separately; no real physics or training-adapter integration'"),
    ("(OUT/'result.json').write_text", f"(Path({str(OUT / 'dynamics_result.json')!r})).write_text"),
])

from TITA_Lab.tasks.velocity.config.TITA.env_cfg import VelocityEnvCfg

cfg = VelocityEnvCfg()
cfg.validate()
files = [
    "mdp/actions.py", "mdp/observations.py", "mdp/events.py", "mdp/__init__.pyi",
    "config/TITA/env_cfg.py", "tita_velocity_env.py", "observation_layout.py",
]
for name in files:
    path = ROOT / "TITA_Lab/src/TITA_Lab/tasks/velocity" / name
    ast.parse(path.read_text(), filename=str(path))
(OUT / "config_result.json").write_text(json.dumps({
    "configuration_validate": True,
    "ast_parse": files,
    "simulation_or_training_started": False,
}, indent=2) + "\n")
