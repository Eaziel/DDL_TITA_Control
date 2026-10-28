"""Check saved action processing and an in-memory FIR correction on CPU."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

from TITA_Lab.tasks.velocity.mdp import actions as action_module
from TITA_Lab.tasks.velocity.mdp import rewards as reward_module

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "TITA_Lab/src/TITA_Lab/tasks/velocity/mdp/actions.py"
source = SOURCE.read_text()
tree = ast.parse(source)
action_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TitaJointAction")
method = next(node for node in action_class.body if isinstance(node, ast.FunctionDef) and node.name == "process_actions")
prefix = [ast.dump(node, include_attributes=False) for node in method.body[:2]]
repeated = [ast.dump(node, include_attributes=False) for node in method.body[3:5]]
assert prefix == repeated, "This audit expects the saved duplicate-copy version; inspect a changed source separately."

candidate_tree = copy.deepcopy(tree)
candidate_class = next(node for node in candidate_tree.body if isinstance(node, ast.ClassDef) and node.name == "TitaJointAction")
candidate_method = next(node for node in candidate_class.body if isinstance(node, ast.FunctionDef) and node.name == "process_actions")
# 只去掉最前面重复的两次copy_；原有英文说明随即成为函数docstring。
candidate_method.body = candidate_method.body[2:]
candidate = types.ModuleType("TITA_Lab.tasks.velocity.mdp._fir_audit_candidate")
candidate.__package__ = "TITA_Lab.tasks.velocity.mdp"
candidate.__file__ = str(SOURCE)
sys.modules[candidate.__name__] = candidate
exec(compile(candidate_tree, str(SOURCE), "exec"), candidate.__dict__)


def make_term(cls, count, scale, offset):
    """Create action buffers without constructing a simulator or robot."""
    term = object.__new__(cls)
    term.cfg = SimpleNamespace(raw_clip=100.0, previous_weight=0.2, clip=None)
    term._scale = torch.tensor(scale, dtype=torch.float32).view(1, -1)
    term._offset = torch.tensor(offset, dtype=torch.float32).view(1, -1).repeat(count, 1)
    term._raw_actions = torch.zeros_like(term._offset)
    term._processed_actions = torch.zeros_like(term._offset)
    term.clipped_actions = torch.zeros_like(term._offset)
    term.previous_clipped_actions = torch.zeros_like(term._offset)
    return term


def trace(cls):
    """Return one-channel outputs for a short deterministic sequence."""
    term = make_term(cls, 1, [1.0], [0.0])
    outputs = []
    for value in (10.0, 0.0, 200.0, -200.0):
        term.process_actions(torch.tensor([[value]]))
        outputs.append(term.processed_actions.item())
    return outputs


def check_grouped(cls):
    """Check native parent processing and the task's actual action-rate function."""
    terms = {
        "leg_position": make_term(cls, 4, [0.25, 0.5, 0.5] * 2, [0.0, 0.8, -1.5] * 2),
        "wheel_effort": make_term(cls, 4, [5.0, 5.0], [0.0, 0.0]),
    }
    env = SimpleNamespace(num_envs=4, device="cpu", action_manager=SimpleNamespace(get_term=terms.__getitem__))
    previous = torch.zeros(4, 8)
    generator = torch.Generator().manual_seed(20261027)
    max_target_error = max_rate_error = 0.0
    all_rates_zero = True
    saved_type = reward_module.TitaJointAction
    try:
        # 候选类在独立模块中构造；让真实奖励函数的类型检查指向本次受测类。
        reward_module.TitaJointAction = cls
        for _ in range(12):
            values = torch.randn(4, 8, generator=generator) * 150.0
            original = values.clone()
            clipped = values.clamp(-100.0, 100.0)
            filtered = 0.2 * previous + 0.8 * clipped
            for name, section in (("leg_position", slice(0, 6)), ("wheel_effort", slice(6, 8))):
                term = terms[name]
                term.process_actions(values[:, section])
                expected = filtered[:, section] * term._scale + term._offset
                max_target_error = max(max_target_error, (term.processed_actions - expected).abs().max().item())
                torch.testing.assert_close(term.raw_actions, values[:, section], rtol=0, atol=0)
            expected_rate = ((clipped - previous) ** 2).sum(dim=1)
            actual_rate = reward_module.action_rate_clipped_l2(env)
            max_rate_error = max(max_rate_error, (actual_rate - expected_rate).abs().max().item())
            all_rates_zero &= bool(torch.all(actual_rate == 0))
            torch.testing.assert_close(values, original, rtol=0, atol=0)
            previous = clipped.clone()
    finally:
        reward_module.TitaJointAction = saved_type
    return {"max_target_error": max_target_error, "max_action_rate_error": max_rate_error, "all_action_rates_zero": all_rates_zero}


current = check_grouped(action_module.TitaJointAction)
corrected = check_grouped(candidate.TitaJointAction)
assert current["max_target_error"] > 0 and current["all_action_rates_zero"]
assert corrected["max_target_error"] == 0
assert corrected["max_action_rate_error"] < 0.04
assert not corrected["all_action_rates_zero"]

result = {
    "scope": "CPU action buffers, actual new Lab JointAction parent, actual reward function; no environment construction or physics.",
    "python": sys.executable,
    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
    "source_method_before": ast.get_source_segment(source, method),
    "candidate_change": "Remove only the first two duplicated copy_ statements in memory.",
    "inputs": [10.0, 0.0, 200.0, -200.0],
    "expected_filtered": [8.0, 2.0, 80.0, -60.0],
    "current_filtered": trace(action_module.TitaJointAction),
    "candidate_filtered": trace(candidate.TitaJointAction),
    "grouped_current": current,
    "grouped_candidate": corrected,
    "source_unchanged": SOURCE.read_text() == source,
}
Path(__file__).with_name("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "source_method_before"}, ensure_ascii=False, indent=2))
