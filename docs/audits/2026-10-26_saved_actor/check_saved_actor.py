"""Compare the saved TITA actor and auxiliary loss with the legacy network."""

from __future__ import annotations

import ast
import hashlib
import json
import types
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CURRENT = ROOT / "TITA_Lab/src/TITA_Lab/learning/np3o/actor.py"
COMMON = Path("/home/eaziel/Downloads/tita_rl-master/modules/common_modules.py")
LEGACY = COMMON.with_name("actor_critic.py")
source_text = CURRENT.read_text()
torch.set_num_threads(2)
torch.manual_seed(20261026)

# 执行本次读取到的实际源码快照；不改方法名，不修改用户文件。
module = types.ModuleType("saved_actor_snapshot")
exec(compile(source_text, str(CURRENT), "exec"), module.__dict__)
current = module.TitaHistoryActor()
method_name = next(
    (name for name in ("auxiliary_loss", "auxilary_loss") if hasattr(current, name)),
    None,
)
assert method_name is not None, "No auxiliary-loss method found in the saved actor."
loss_method = getattr(current, method_name)

scope = {"torch": torch, "nn": nn, "F": F}
legacy_sources = {path: path.read_text() for path in (COMMON, LEGACY)}
for path, names in (
    (COMMON, {"mlp_factory", "mlp_layernorm_factory"}),
    (LEGACY, {"MlpBarlowTwinsActor", "off_diagonal"}),
):
    nodes = [n for n in ast.parse(legacy_sources[path]).body if getattr(n, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
reference = scope["MlpBarlowTwinsActor"](
    num_prop=33, num_hist=10, obs_encoder_dims=[256, 128],
    mlp_encoder_dims=[512, 256, 128], actor_dims=[512, 256, 128],
    latent_dim=16, num_actions=8, activation=nn.ELU(),
)
current.load_state_dict(reference.state_dict(), strict=True)

forward_errors = {}
for batch_size in (1, 4, 64):
    obs = torch.randn(batch_size, 33)
    history = torch.randn(batch_size, 10, 33)
    with torch.no_grad():
        expected = reference(obs, history)
        actual = current(obs, history)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (batch_size, 8)
    forward_errors[str(batch_size)] = (actual - expected).abs().max().item()

loss_errors, gradient_errors, input_errors = {}, {}, {}
for batch_size in (4, 32):
    reference.zero_grad(set_to_none=True)
    current.zero_grad(set_to_none=True)
    obs = torch.randn(batch_size, 33, requires_grad=True)
    history = torch.randn(batch_size, 10, 33, requires_grad=True)
    other_obs = obs.detach().clone().requires_grad_()
    other_history = history.detach().clone().requires_grad_()
    target = torch.randn(batch_size, 10)
    expected = reference.BarlowTwinsLoss(obs, history, target, 5e-3)
    actual = loss_method(other_obs, other_history, target)
    assert actual.ndim == 0 and torch.isfinite(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    expected.backward()
    actual.backward()
    torch.testing.assert_close(obs.grad, other_obs.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(history.grad, other_history.grad, rtol=1e-5, atol=1e-5)
    maximum = 0.0
    ref_params = dict(reference.named_parameters())
    for name, param in current.named_parameters():
        expected_grad = ref_params[name].grad
        if expected_grad is None:
            assert param.grad is None
        else:
            torch.testing.assert_close(param.grad, expected_grad, rtol=1e-5, atol=1e-5)
            maximum = max(maximum, (param.grad - expected_grad).abs().max().item())
    for name, buffer in current.bn.named_buffers():
        torch.testing.assert_close(buffer, dict(reference.bn.named_buffers())[name], rtol=0, atol=0)
    loss_errors[str(batch_size)] = (actual - expected).abs().item()
    gradient_errors[str(batch_size)] = maximum
    input_errors[str(batch_size)] = max(
        (obs.grad - other_obs.grad).abs().max().item(),
        (history.grad - other_history.grad).abs().max().item(),
    )

# 实际编码器输入：动作窗口为1..9、99，辅助窗口为原样0..9。
captures = []
hook = current.mlp_encoder.register_forward_pre_hook(
    lambda mod, args: captures.append(args[0].detach().clone())
)
with torch.no_grad():
    tagged = torch.arange(10.0).reshape(1, 10, 1).expand(4, 10, 33)
    current(torch.full((4, 33), 99.0), tagged)
    loss_method(torch.randn(4, 33), tagged, torch.randn(4, 10))
hook.remove()
assert torch.equal(captures[0].reshape(4, 10, 33)[0, :, 0], torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 99.]))
assert torch.equal(captures[1].reshape(4, 10, 33)[0, :, 0], torch.arange(10.0))

result = {
    "date": "2026-10-26", "torch": torch.__version__, "device": "cpu",
    "scope": "saved actor and saved auxiliary-loss body; random tensor forward/backward, no optimizer or simulator",
    "source_sha256": {str(CURRENT): hashlib.sha256(source_text.encode()).hexdigest(),
                      **{str(p): hashlib.sha256(s.encode()).hexdigest() for p, s in legacy_sources.items()}},
    "actual_method_name": method_name,
    "method_name_matches_planned_interface": method_name == "auxiliary_loss",
    "arithmetic_checks_passed": True,
    "forward_max_abs_error": forward_errors,
    "auxiliary_loss_max_abs_error": loss_errors,
    "parameter_gradient_max_abs_error": gradient_errors,
    "input_gradient_max_abs_error": input_errors,
    "batchnorm_buffers_equal": True,
    "both_history_windows_correct": True,
    "comparison_tolerance": {"rtol": 1e-5, "atol": 1e-5},
    "user_source_modified": False,
}
(HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
