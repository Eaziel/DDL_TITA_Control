"""Check the saved actor and auxiliary-loss candidate against legacy code."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CURRENT = ROOT / "TITA_Lab/src/TITA_Lab/learning/np3o/actor.py"
PROPOSAL = HERE / "proposal_actor.py"
COMMON = Path("/home/eaziel/Downloads/tita_rl-master/modules/common_modules.py")
LEGACY = COMMON.with_name("actor_critic.py")
torch.set_num_threads(2)
torch.manual_seed(20261026)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scope = {"torch": torch, "nn": nn, "F": F}
for path, names in (
    (COMMON, {"mlp_factory", "mlp_layernorm_factory"}),
    (LEGACY, {"MlpBarlowTwinsActor", "off_diagonal"}),
):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)

reference = scope["MlpBarlowTwinsActor"](
    num_prop=33, num_hist=10, obs_encoder_dims=[256, 128],
    mlp_encoder_dims=[512, 256, 128], actor_dims=[512, 256, 128],
    latent_dim=16, num_actions=8, activation=nn.ELU(),
)
current = load(CURRENT, "saved_actor").TitaHistoryActor()
candidate = load(PROPOSAL, "auxiliary_actor").TitaHistoryActor()
current.load_state_dict({key: reference.state_dict()[key] for key in current.state_dict()})
candidate.load_state_dict(reference.state_dict())

forward_errors = {}
for batch_size in (1, 4, 64):
    obs = torch.randn(batch_size, 33)
    history = torch.randn(batch_size, 10, 33)
    with torch.no_grad():
        expected = reference(obs, history)
        actual = current(obs, history)
        proposed = candidate(obs, history)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(proposed, expected, rtol=0, atol=0)
    forward_errors[str(batch_size)] = (actual - expected).abs().max().item()

captures = []
hook = current.mlp_encoder.register_forward_pre_hook(
    lambda module, args: captures.append(args[0].detach().clone())
)
with torch.no_grad():
    current(torch.full((1, 33), 99.0), torch.arange(10.0).view(1, 10, 1).expand(1, 10, 33))
hook.remove()
assert torch.equal(captures[0].reshape(10, 33)[:, 0], torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 99.]))

loss_errors, grad_errors = {}, {}
for batch_size in (4, 32):
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    obs = torch.randn(batch_size, 33, requires_grad=True)
    history = torch.randn(batch_size, 10, 33, requires_grad=True)
    target = torch.randn(batch_size, 10)
    other_obs = obs.detach().clone().requires_grad_()
    other_history = history.detach().clone().requires_grad_()
    expected = reference.BarlowTwinsLoss(obs, history, target, 5e-3)
    actual = candidate.auxiliary_loss(other_obs, other_history, target)
    # 旧代码在strided对角线视图上原位平方并归约；候选使用非原位张量。
    # float32归约顺序可有舍入差异，按明确容差比较公式及梯度。
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(obs.grad, other_obs.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(history.grad, other_history.grad, rtol=1e-5, atol=1e-5)
    maximum = 0.0
    ref_params = dict(reference.named_parameters())
    for name, param in candidate.named_parameters():
        expected_grad = ref_params[name].grad
        if expected_grad is None:
            assert param.grad is None
        else:
            torch.testing.assert_close(param.grad, expected_grad, rtol=1e-5, atol=1e-5)
            maximum = max(maximum, (param.grad - expected_grad).abs().max().item())
    for name, buffer in candidate.bn.named_buffers():
        torch.testing.assert_close(buffer, dict(reference.bn.named_buffers())[name], rtol=0, atol=0)
    loss_errors[str(batch_size)] = (actual - expected).abs().item()
    grad_errors[str(batch_size)] = maximum

# 辅助损失沿用旧实现：送入编码器的仍是未经替换的过去十帧。
captures = []
hook = candidate.mlp_encoder.register_forward_pre_hook(
    lambda module, args: captures.append(args[0].detach().clone())
)
with torch.no_grad():
    candidate.auxiliary_loss(torch.randn(4, 33), torch.arange(10.0).view(1, 10, 1).expand(4, 10, 33), torch.randn(4, 10))
hook.remove()
assert torch.equal(captures[0].reshape(4, 10, 33)[0, :, 0], torch.arange(10.0))

result = {
    "date": "2026-10-26", "device": "cpu", "torch": torch.__version__,
    "scope": "saved actor forward; proposed auxiliary loss and gradients; no optimizer, simulation, or training",
    "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (CURRENT, PROPOSAL, COMMON, LEGACY)},
    "current_actor_forward_max_abs_error": forward_errors,
    "current_actor_history_window_correct": True,
    "current_actor_prediction_dim_name": "prediction_dim" if hasattr(current, "prediction_dim") else "predicton_dim (rename both uses before next candidate)",
    "candidate_auxiliary_loss_max_abs_error": loss_errors,
    "candidate_float32_comparison_tolerance": {"rtol": 1e-5, "atol": 1e-5},
    "candidate_parameter_gradient_max_abs_error": grad_errors,
    "candidate_input_gradients_and_batchnorm_buffers_equal": True,
    "candidate_auxiliary_history_is_unshifted": True,
    "candidate_not_installed_in_user_source": True,
}
(HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
