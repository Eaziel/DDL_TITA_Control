"""Check current cost wiring and four corrections without modifying user source."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TASK = ROOT / "TITA_Lab/src/TITA_Lab/tasks/velocity"
sys.path.insert(0, str(ROOT / "TITA_Lab/src"))

from TITA_Lab.tasks.velocity import cost_manager as saved_manager
from TITA_Lab.tasks.velocity.config.TITA import env_cfg as saved_cfg

try:
    saved_manager.TitaCostManager(None, None)
except TypeError as exc:
    constructor_failure = str(exc)
else:
    raise AssertionError("Expected abstract _prepare_terms failure from saved code")

fixed_manager_source = (TASK / "cost_manager.py").read_text()
assert "def _prepare_term(self)" in fixed_manager_source
fixed_manager_source = fixed_manager_source.replace("def _prepare_term(self)", "def _prepare_terms(self)", 1)
anchor = "self.k_values = torch.tensor([term.k for term in self._term_cfgs], device=env.device).view(1, 1, -1)"
assert anchor in fixed_manager_source
fixed_manager_source = fixed_manager_source.replace(anchor,
    "self.k_values = torch.tensor([term.k for term in self._term_cfgs], device=env.device).view(1, -1)\n"
    "        self.d_values = torch.tensor([term.d for term in self._term_cfgs], device=env.device).view(1, 1, -1)", 1)

# 仅重建Manager子类，复用用户真正的CostTermCfg类型，防止类型身份被替身掩盖。
node = next(node for node in ast.parse(fixed_manager_source).body
            if isinstance(node, ast.ClassDef) and node.name == "TitaCostManager")
manager_ns = vars(saved_manager).copy()
exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
             "current_manager_with_corrections", "exec"), manager_ns)
corrected_manager_module = NS(TitaCostManager=manager_ns["TitaCostManager"], CostTermCfg=saved_manager.CostTermCfg)
assert not corrected_manager_module.TitaCostManager.__abstractmethods__


def corrected_costs_cfg():
    cfg = saved_cfg.CostsCfg()
    cfg.dof_vel_limits.k = 0.3
    cfg.dof_vel_limits.params = {
        "asset_cfg": saved_cfg.SceneEntityCfg("robot", joint_names=saved_cfg.TITA_JOINT_NAMES, preserve_order=True),
    }
    return cfg


# 复用上一轮原生RL流程夹具，但本轮三部分均读取当前用户文件，而非文档候选。
previous = ROOT / "docs/audits/2026-10-21_cost_collection/check.py"
source = previous.read_text()
start = source.index("cost_source = ")
end = source.index('C = module("tita_costs_checked_candidate"', start)
source = source[:start] + 'cost_source = (TASK / "mdp/costs.py").read_text()\n' + source[end:]
source = source.replace('M = module("tita_cost_manager_checked_candidate", blocks[1], document)',
                        'M = corrected_manager_module', 1)
start = source.index("cfg_namespace = ")
end = source.index("cfg = CURRENT_CFG.VelocityEnvCfg()", start)
source = source[:start] + 'CostsCfg = corrected_costs_cfg\n' + source[end:]
source = source.replace('env_source = blocks[3].replace(',
                        'env_source = (TASK / "tita_velocity_env.py").read_text().replace(', 1)
source = source.replace('("cost_manager.py", blocks[1])', '("cost_manager.py", fixed_manager_source)', 1)
source = source.replace('("tita_velocity_env.py", blocks[3])',
                        '("tita_velocity_env.py", (TASK / "tita_velocity_env.py").read_text())', 1)
source = source.replace('Five current cost functions plus documented acceleration/manager/cfg/env candidates',
                        'Current six cost functions, current manager/cfg/env with four in-memory corrections', 1)
namespace = dict(globals())
exec(compile(source, str(previous), "exec"), namespace)

# 实际调用通用解析，复现当前速度项params与函数签名不匹配。
fixture_env = namespace["BaseFixture"]()
try:
    corrected_manager_module.TitaCostManager(NS(dof_vel_limits=saved_cfg.CostsCfg().dof_vel_limits), fixture_env)
except ValueError as exc:
    parameter_failure = str(exc)
else:
    raise AssertionError("Expected native parameter resolution failure")
fixture_env.close()

result_path = OUT / "result.json"
result = json.loads(result_path.read_text())
result["saved_source_constructor_failure"] = constructor_failure
result["saved_velocity_parameter_failure"] = parameter_failure
result["saved_velocity_k"] = saved_cfg.CostsCfg().dof_vel_limits.k
result["in_memory_corrections"] = [
    "_prepare_term -> _prepare_terms",
    "k_values shape (1,6) and restore d_values (1,1,6)",
    "dof_vel_limits params sensor_cfg -> robot asset_cfg",
    "dof_vel_limits k 0.1 -> 0.3",
]
result["user_source_files_modified"] = False
result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({"constructor_failure": constructor_failure, "parameter_failure": parameter_failure,
                  "saved_k": result["saved_velocity_k"]}, indent=2, ensure_ascii=False))
