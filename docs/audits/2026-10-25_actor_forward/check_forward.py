"""Compare the proposed actor forward and gradients with the legacy actor."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
LEGACY_ROOT = Path("/home/eaziel/Downloads/tita_rl-master")
COMMON = LEGACY_ROOT / "modules/common_modules.py"
ACTOR = LEGACY_ROOT / "modules/actor_critic.py"
PROPOSAL = HERE / "proposal_actor.py"
torch.set_num_threads(2)
torch.manual_seed(20261025)

# 执行旧文件中的原始工厂与原始类，跳过整个包的无关导入。
namespace = {"torch": torch, "nn": nn}
for path, names in (
    (COMMON, {"mlp_factory", "mlp_layernorm_factory"}),
    (ACTOR, {"MlpBarlowTwinsActor"}),
):
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)

spec = importlib.util.spec_from_file_location("candidate_actor", PROPOSAL)
candidate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate_module)
candidate = candidate_module.TitaHistoryActor()
legacy = namespace["MlpBarlowTwinsActor"](
    num_prop=33,
    num_hist=10,
    obs_encoder_dims=[256, 128],
    mlp_encoder_dims=[512, 256, 128],
    actor_dims=[512, 256, 128],
    latent_dim=16,
    num_actions=8,
    activation=nn.ELU(),
)

# 只拷贝本次前向涉及的网络，旧辅助obs_encoder/BatchNorm留待后续迁移。
candidate.load_state_dict({key: legacy.state_dict()[key] for key in candidate.state_dict()})
errors = {}
for batch_size in (1, 4, 64):
    obs = torch.randn(batch_size, 33)
    history = torch.randn(batch_size, 10, 33)
    saved_obs, saved_history = obs.clone(), history.clone()
    with torch.no_grad():
        expected = legacy(obs, history)
        actual = candidate(obs, history)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (batch_size, 8)
    assert torch.equal(obs, saved_obs) and torch.equal(history, saved_history)
    errors[str(batch_size)] = (actual - expected).abs().max().item()

# 用时间编号核对窗口：原历史0..9，当前99，应输入1..9,99。
captured = []
hook = candidate.mlp_encoder.register_forward_pre_hook(
    lambda module, inputs: captured.append(inputs[0].detach().clone())
)
with torch.no_grad():
    candidate(torch.full((1, 33), 99.0), torch.arange(10.0).view(1, 10, 1).expand(1, 10, 33))
hook.remove()
assert torch.equal(captured[0].reshape(10, 33)[:, 0], torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 99.]))

# 同一组输入、参数和标量目标，比较前向涉及的参数/输入梯度；不更新参数。
obs_a = torch.randn(4, 33, requires_grad=True)
hist_a = torch.randn(4, 10, 33, requires_grad=True)
obs_b = obs_a.detach().clone().requires_grad_()
hist_b = hist_a.detach().clone().requires_grad_()
weights = torch.randn(4, 8)
(legacy(obs_a, hist_a) * weights).sum().backward()
(candidate(obs_b, hist_b) * weights).sum().backward()
torch.testing.assert_close(obs_a.grad, obs_b.grad, rtol=0, atol=0)
torch.testing.assert_close(hist_a.grad, hist_b.grad, rtol=0, atol=0)
legacy_params = dict(legacy.named_parameters())
max_grad_error = 0.0
for name, param in candidate.named_parameters():
    torch.testing.assert_close(param.grad, legacy_params[name].grad, rtol=0, atol=0)
    max_grad_error = max(max_grad_error, (param.grad - legacy_params[name].grad).abs().max().item())

result = {
    "date": "2026-10-25",
    "torch": torch.__version__,
    "device": "cpu",
    "scope": "actor forward subset and one backward comparison; no optimizer, simulator, or training",
    "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (COMMON, ACTOR, PROPOSAL)},
    "forward_max_abs_error_by_batch": errors,
    "parameter_gradient_max_abs_error": max_grad_error,
    "input_gradients_equal": True,
    "input_tensors_unchanged": True,
    "history_window": [1, 2, 3, 4, 5, 6, 7, 8, 9, 99],
    "parameter_count": sum(param.numel() for param in candidate.parameters()),
    "pending": ["auxiliary prediction and Barlow loss", "reward/cost critics", "Gaussian sampling", "NP3O and runner integration"],
}
(HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
