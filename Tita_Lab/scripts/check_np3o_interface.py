# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the TITA actor and environment interface."""

import argparse
import csv
import json
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, cast

import gymnasium as gym
import torch
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from isaaclab.app import add_launcher_args, launch_simulation

from isaaclab_tasks.utils import resolve_task_config

import Tita_Lab.tasks  # noqa: F401
from Tita_Lab.learning.np3o.critic import TitaValueNetworks
from Tita_Lab.learning.np3o.env_wrapper import TitaNP3OEnvWrapper
from Tita_Lab.learning.np3o.losses import compute_policy_losses, compute_value_losses
from Tita_Lab.learning.np3o.policy import TitaGaussianPolicy
from Tita_Lab.learning.np3o.rollout_storage import TitaRolloutStorage
from Tita_Lab.tasks.velocity.observation_layout import pack_tita_observations

if TYPE_CHECKING:
    from Tita_Lab.tasks.velocity.config.Tita.env_cfg import VelocityEnvCfg
    from Tita_Lab.tasks.velocity.tita_velocity_env import TitaVelocityEnv


def run_check(env: TitaNP3OEnvWrapper, max_steps: int) -> dict[str, int]:
    """Check actor outputs and count completed environment steps."""

    # 当前网络随机初始化；这里只验证已有模块能否连通。
    policy = TitaGaussianPolicy(init_noise_std=1.0).to(device=env.device)
    policy.eval()
    value_networks = TitaValueNetworks().to(device=env.device)
    value_networks.eval()

    gamma, lam = 0.99, 0.95

    # 在inference_mode外分配存储，之后训练可以把这些普通张量作为输入。
    storage = TitaRolloutStorage(
        num_envs=env.num_envs,
        num_steps=max_steps,
        num_obs=env.num_obs,
        num_actions=env.num_actions,
        num_costs=env.num_cost,
        device=env.device,
    )

    obs = env.reset()
    assert obs.shape == (env.num_envs, 586)
    assert torch.isfinite(obs).all(), "Non-finite observation."
    completed = resets = timeouts = 0

    # 真正的TITA环境；外面的Gymnasium包装层已经去除。
    lab_env = cast("TitaVelocityEnv", env.env.unwrapped)

    with torch.inference_mode():
        for _ in range(max_steps):
            if not lab_env.sim.is_running():
                break
            # policy内部从586列中取当前33列和最后330列历史。
            actions = policy.act(obs=obs)  # (N,8)，这次交给环境的是采样结果。
            log_prob = policy.get_actions_log_prob(actions=actions)  # (N,)。
            entropy = policy.entropy  # (N,)。
            reward_values, cost_values = value_networks(obs)

            assert actions.shape == (env.num_envs, 8)
            assert policy.action_mean.shape == actions.shape
            assert policy.action_std.shape == actions.shape
            assert log_prob.shape == (env.num_envs,)
            assert entropy.shape == (env.num_envs,)
            assert torch.isfinite(actions).all(), "Non-finite action."
            assert torch.isfinite(log_prob).all(), "Non-finite action log density."
            assert torch.isfinite(entropy).all(), "Non-finite action entropy."
            assert (policy.action_std > 0.0).all(), "Non-positive action standard deviation."

            assert reward_values.shape == (env.num_envs, 1)
            assert cost_values.shape == (env.num_envs, env.num_cost)
            assert torch.isfinite(reward_values).all(), "Non-finite reward value."
            assert torch.isfinite(cost_values).all(), "Non-finite cost value."
            assert (cost_values >= 0.0).all(), "Negative cost value."

            next_obs, _, rewards, costs, dones, infos = env.step(actions=actions)
            time_out_flags: torch.Tensor = infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))

            # 默认没有超时价值补偿，分别为(N,1)/(N,6)。
            timeout_reward_values = torch.zeros_like(reward_values)
            timeout_cost_values = torch.zeros_like(cost_values)
            terminated: torch.Tensor = infos["terminated"]

            # 失败与时间上限同时触发时，按真实终止处理。
            bootstrap_timeouts = time_out_flags.to(dtype=torch.bool) & ~terminated

            if bool(bootstrap_timeouts.any()):
                if "final_obs" not in infos:
                    raise RuntimeError("Enable cfg.compute_final_obs for timeouts.")

                final_obs = pack_tita_observations(infos["final_obs"])
                final_reward_values, final_cost_values = value_networks(final_obs[bootstrap_timeouts])
                timeout_reward_values[bootstrap_timeouts] = final_reward_values
                timeout_cost_values[bootstrap_timeouts] = final_cost_values

            # obs仍是产生本次动作的旧观测；next_obs是环境推进后的新观测。
            storage.add(
                observations=obs,
                actions=actions,
                rewards=rewards,
                costs=costs,
                dones=dones,
                time_outs=time_out_flags,
                reward_values=reward_values,
                cost_values=cost_values,
                log_probs=log_prob,
                action_mean=policy.action_mean,
                action_std=policy.action_std,
                timeout_reward_values=timeout_reward_values,
                timeout_cost_values=timeout_cost_values,
            )
            obs = next_obs

            assert obs.shape == (env.num_envs, 586)
            assert torch.isfinite(obs).all(), "Non-finite observation."
            assert costs.shape == (env.num_envs, env.num_cost)
            assert torch.isfinite(rewards).all(), "Non-finite reward."
            assert torch.isfinite(costs).all(), "Non-finite cost."
            completed += 1
            resets += int(dones.sum().item())

            if "time_outs" in infos:
                # (N,)，每个环境是否在本步因达到回合时间上限而结束。
                timeouts += int(time_out_flags.sum().item())

            if completed == 1:
                print(
                    f"[CHECK] obs={tuple(obs.shape)} actions={tuple(actions.shape)} costs={tuple(costs.shape)}"
                    f"[POLICY] mean={tuple(policy.action_mean.shape)} "
                    f"std={tuple(policy.std.shape)} log_prob={tuple(log_prob.shape)} "
                    f"entropy={entropy[0].item():.4f}"
                    f"[VALUE] reward={tuple(reward_values.shape)} "
                    f"cost={tuple(cost_values.shape)}"
                )

    assert storage.step == completed
    print(
        f"[ROLLOUT] filled={storage.step}/{storage.num_steps} "
        f"obs={tuple(storage.observations.shape)} "
        f"actions={tuple(storage.actions.shape)} "
        f"costs={tuple(storage.costs.shape)}"
    )

    if storage.is_full:
        # 最后一个决策状态：用于估计采集结束后尚未观察到的回报。
        with torch.inference_mode():
            last_reward_values, last_cost_values = value_networks(obs)

        storage.compute_returns(
            last_reward_values,
            last_cost_values,
            gamma=gamma,
            lam=lam,
        )

        for values in (
            storage.reward_returns,
            storage.cost_returns,
            storage.reward_advantages,
            storage.cost_advantages,
        ):
            assert torch.isfinite(values).all(), "Non-finite GAE result."

        print(
            f"[GAE] reward_returns={tuple(storage.reward_returns.shape)} "
            f"cost_returns={tuple(storage.cost_returns.shape)} "
            f"reward_advantages={tuple(storage.reward_advantages.shape)} "
            f"cost_advantages={tuple(storage.cost_advantages.shape)}"
        )

        # 将本批经历分成4份；遍历2轮，一共交出8份小批量数据。
        num_mini_batches, num_learning_epochs = 4, 2
        batch_count = 0
        for batch in storage.mini_batch_generator(
            num_mini_batches=num_mini_batches,
            num_learning_epochs=num_learning_epochs,
        ):
            # B是本次交给更新代码的记录数，不是并行环境数。
            mini_batch_size = batch["observations"].shape[0]
            assert batch["observations"].shape == (mini_batch_size, 586)
            assert batch["actions"].shape == (mini_batch_size, 8)
            assert batch["reward_returns"].shape == (mini_batch_size, 1)
            assert batch["cost_returns"].shape == (mini_batch_size, env.num_cost)
            assert batch["log_probs"].shape == (mini_batch_size, 1)

            if batch_count == 0:
                print(
                    f"[MINIBATCH] obs={tuple(batch['observations'].shape)} "
                    f"actions={tuple(batch['actions'].shape)} "
                    f"reward_returns={tuple(batch['reward_returns'].shape)} "
                    f"cost_returns={tuple(batch['cost_returns'].shape)}"
                )
            batch_count += 1

        assert batch_count == num_mini_batches * num_learning_epochs
        print(f"[MINIBATCH] batches={batch_count} epochs={num_learning_epochs} per_epoch={num_mini_batches}")

        check_value_update(
            value_networks=value_networks,
            storage=storage,
            run_metadata={
                "task": "TitaLab-Velocity-Tita",
                "seed": lab_env.cfg.seed,
                "policy_step_dt_s": lab_env.step_dt,
                "episode_length_s": lab_env.cfg.episode_length_s,
                "gamma": gamma,
                "gae_lambda": lam,
                "cost_names": list(env.cost_names),
            },
        )
        check_policy_update(
            policy=policy,
            storage=storage,
            cost_k=env.cost_k_values,
            cost_d=env.cost_d_values_tensor,
            gamma=gamma,
        )

    else:
        print("[GAE] Skipped: rollout is incomplete.")

    # steps是全局策略步数；resets/timeouts是各环境事件次数之和。
    result = {"steps": completed, "requested": max_steps, "resets": resets, "timeouts": timeouts}
    print(f"[CHECK] {result}")
    return result


@torch.no_grad()
def measure_value_fit(
    value_networks: TitaValueNetworks,
    storage: TitaRolloutStorage,
    update: int,
) -> dict[str, float]:
    """Measure critic errors against the fixed full-rollout targets."""

    was_training = value_networks.training
    value_networks.eval()
    predictions: tuple[torch.Tensor, torch.Tensor] = value_networks(storage.observations.flatten(0, 1))
    value_networks.train(was_training)
    reward_values, cost_values = predictions

    # 每次都在同一批T*N条经历上测量，目标在所有更新间保持不变。
    reward_error: torch.Tensor = reward_values - storage.reward_returns.flatten(0, 1)
    cost_error: torch.Tensor = cost_values - storage.cost_returns.flatten(0, 1)
    row: dict[str, float] = {
        "update": float(update),
        "reward_mse": float(reward_error.square().mean().item()),
        "cost_mse": float(cost_error.square().mean().item()),
    }
    # CSV另存每类cost的MSE；类别顺序在metadata.json中记录。
    per_cost: list[float] = cost_error.square().mean(dim=0).cpu().tolist()
    row.update({f"cost_{index}_mse": float(value) for index, value in enumerate(per_cost)})
    return row


@torch.no_grad()
def save_value_plots(
    value_networks: TitaValueNetworks,
    storage: TitaRolloutStorage,
    history: list[dict[str, float]],
    run_metadata: dict[str, object],
) -> Path:
    """Save fixed-target fitting diagnostics and their source data."""

    created_at = datetime.now().astimezone()
    output_dir = (
        Path(__file__).resolve().parents[1] / "logs" / "interface_checks" / created_at.strftime("%Y%m%d_%H%M%S_%f")
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "value_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    value_networks.eval()
    predictions: tuple[torch.Tensor, torch.Tensor] = value_networks(storage.observations.flatten(0, 1))
    # 只把绘图所需的小数组传到CPU；不保存梯度图。
    target: torch.Tensor = storage.reward_returns.flatten().detach().cpu()
    before: torch.Tensor = storage.reward_values.flatten().detach().cpu()
    after: torch.Tensor = predictions[0].flatten().detach().cpu()
    with (output_dir / "reward_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step_index", "env_id", "gae_target", "before", "after"])
        for index in range(target.numel()):
            writer.writerow(
                [
                    index // storage.num_envs,
                    index % storage.num_envs,
                    float(target[index].item()),
                    float(before[index].item()),
                    float(after[index].item()),
                ]
            )

    metadata = dict(run_metadata)
    metadata.update(
        {
            "created_at": created_at.isoformat(),
            "num_envs": storage.num_envs,
            "rollout_steps": storage.num_steps,
            "samples": storage.num_steps * storage.num_envs,
            "optimizer_steps": int(history[-1]["update"]),
            "torch": str(torch.__version__),
            "matplotlib": version("matplotlib"),
            "metric_scope": "in-sample fitting to fixed GAE targets on one rollout",
            "cost_mse_reduction": "mean over samples and all cost columns",
        }
    )
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)

    # Agg直接保存文件，不额外建立交互式绘图窗口。
    with rc_context({"font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42}):
        figure = Figure(figsize=(11.4, 3.5), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 3)
        reward_ax: Axes = axes[0]
        cost_ax: Axes = axes[1]
        fit_ax: Axes = axes[2]
        updates = [int(row["update"]) for row in history]

        reward_ax.plot(
            updates,
            [row["reward_mse"] for row in history],
            color="#0072B2",
            marker="o",
            markersize=3,
            linewidth=1.4,
        )
        reward_ax.set(title="(a) Reward value fit", xlabel="Critic optimizer step", ylabel="Fixed-rollout MSE")
        reward_ax.set_xticks(updates)
        cost_ax.plot(
            updates,
            [row["cost_mse"] for row in history],
            color="#D55E00",
            marker="o",
            markersize=3,
            linewidth=1.4,
        )
        cost_ax.set(title="(b) Mean cost value fit", xlabel="Critic optimizer step", ylabel="Fixed-rollout MSE")
        cost_ax.set_xticks(updates)

        fit_ax.scatter(target.numpy(), before.numpy(), s=12, alpha=0.55, color="#999999", label="Before")
        fit_ax.scatter(target.numpy(), after.numpy(), s=12, alpha=0.65, color="#0072B2", label="After")
        bounds: torch.Tensor = torch.cat((target, before, after))
        lower = float(bounds.min().item())
        upper = float(bounds.max().item())
        padding = 0.05 * max(upper - lower, 1e-6)
        limits = (lower - padding, upper + padding)
        fit_ax.plot(limits, limits, color="black", linestyle="--", linewidth=1.0, label="y = x")
        fit_ax.set(
            title="(c) Reward predictions",
            xlabel="Fixed GAE target",
            ylabel="Predicted reward value",
            xlim=limits,
            ylim=limits,
        )
        fit_ax.set_aspect("equal", adjustable="box")
        fit_ax.legend(frameon=False, fontsize=8)
        for axis in (reward_ax, cost_ax, fit_ax):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(direction="in")
            axis.grid(alpha=0.18, linewidth=0.5)

        figure.suptitle(f"Single-rollout critic fitting | N={storage.num_envs}, T={storage.num_steps}")
        figure.savefig(output_dir / "critic_fit.png", dpi=300)
        figure.savefig(output_dir / "critic_fit.pdf")
        figure.clear()
    print(f"[PLOT] {output_dir}")
    return output_dir


def check_value_update(
    value_networks: TitaValueNetworks,
    storage: TitaRolloutStorage,
    run_metadata: dict[str, object],
) -> None:
    """Check critic optimization against fixed GAE return targets."""

    # 包含共享编码器、奖励输出网络和六类cost输出网络的全部参数。
    parameters = list(value_networks.parameters())
    parameters_before = [parameter.detach().clone() for parameter in parameters]

    # 仅用于本次更新检查；同一个Adam在下面8次更新间保留内部状态。
    optimizer = torch.optim.Adam(params=parameters, lr=3.0e-4)
    num_mini_batches, num_learning_epochs = 4, 2
    max_grad_norm = 1.0
    value_networks.train()

    reward_loss_sum: float = 0.0
    cost_loss_sum: float = 0.0
    update_count = 0
    history: list[dict[str, float]] = [measure_value_fit(value_networks, storage, update=0)]

    for batch in storage.mini_batch_generator(
        num_mini_batches=num_mini_batches,
        num_learning_epochs=num_learning_epochs,
    ):
        reward_loss, cost_loss = compute_value_losses(
            value_networks=value_networks,
            observations=batch["observations"],
            reward_returns=batch["reward_returns"],
            cost_returns=batch["cost_returns"],
        )
        loss: torch.Tensor = reward_loss + cost_loss
        assert loss.requires_grad, "Value loss has no gradient graph."
        assert torch.isfinite(loss).all(), "Non-finite value loss."

        # 清除上批梯度；链式求导；限制梯度范数；最后实际修改网络参数。
        optimizer.zero_grad(set_to_none=True)  # 清除上次梯度，参数值不变。
        loss.backward()  # 自动计算各参数的梯度。
        torch.nn.utils.clip_grad_norm_(  # 限制梯度过大的情况。
            parameters=parameters,
            max_norm=max_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()  # 根据梯度，实际修改参数。

        # item取出普通数值，统计时不保留梯度图。
        reward_loss_sum += float(reward_loss.item())
        cost_loss_sum += float(cost_loss.item())
        update_count += 1
        history.append(measure_value_fit(value_networks, storage, update=update_count))

    value_networks.eval()
    assert update_count == num_mini_batches * num_learning_epochs
    for parameter in parameters:
        assert torch.isfinite(parameter).all(), "Non-finite critic parameter."

    # 比较更新前后的独立副本，确认至少有一个参数实际发生变化。
    max_parameter_change = max(
        (parameter.detach() - before).abs().max().item()
        for parameter, before in zip(parameters, parameters_before, strict=False)
    )
    assert max_parameter_change > 0.0, "Critic parameters did not change."
    print(
        f"[VALUE_UPDATE] updates={update_count} "
        f"reward_loss={reward_loss_sum / update_count:.6f} "
        f"cost_loss={cost_loss_sum / update_count:.6f} "
        f"max_parameter_change={max_parameter_change:.3e}"
    )

    plot_metadata = dict(run_metadata)
    plot_metadata.update(
        {
            "optimizer": "Adam",
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "num_mini_batches": num_mini_batches,
            "num_learning_epochs": num_learning_epochs,
            "max_grad_norm": max_grad_norm,
        }
    )
    save_value_plots(value_networks, storage, history, plot_metadata)


def check_policy_update(
    policy: TitaGaussianPolicy,
    storage: TitaRolloutStorage,
    *,
    cost_k: torch.Tensor,
    cost_d: torch.Tensor,
    gamma: float,
) -> None:
    """Update policy parameters using reward and cost advantages."""

    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must be in [0, 1).")
    

    parameters = list(policy.parameters())
    parameters_before = [parameter.detach().clone() for parameter in parameters]
    optimizer = torch.optim.Adam(params=parameters, lr=3.0e-4)
    policy.train()

    # 用整批96条记录的统计量标准化奖励优势，保持原始存储不变。
    advantages: torch.Tensor = storage.reward_advantages
    advantage_mean: torch.Tensor = advantages.mean()
    advantage_std: torch.Tensor = advantages.std(correction=0).clamp_min(1e-8)

    # cost优势的形状是(T,N,C)：采集步数、环境数、cost种类数。
    # 分别计算每一种cost的平均值和标准差，得到C个数，当前C=6。
    cost_mean: torch.Tensor = storage.cost_advantages.mean(dim=(0, 1))
    cost_std: torch.Tensor = storage.cost_advantages.std(
        dim=(0, 1), correction=0,
    ).clamp_min(1e-8)

    # 每种cost各有一个惩罚系数k和允许上限d。
    # 去掉多余的维度，再使用与cost数据相同的设备和数值类型。
    k: torch.Tensor = cost_k.detach().reshape(-1).to(
        device=storage.device,
        dtype=storage.costs.dtype,
    )
    d: torch.Tensor = cost_d.detach().reshape(-1).to(storage.costs)
    assert k.shape == d.shape == cost_mean.shape
    assert torch.isfinite(k).all() and (k >= 0.0).all()
    assert torch.isfinite(d).all()

    # 取reset后第0步的cost回报目标，再对所有环境求平均。
    # 用它估计从初始状态出发的累计cost，其中包含旧critic的预测。
    initial_cost: torch.Tensor = storage.cost_returns[0].mean(dim=0)

    # cost优势减去了均值，这里要把均值补回来；
    # 再除以标准差，使约束项与标准化后的cost优势使用相同尺度。
    cost_offset: torch.Tensor = (
        (1.0 - gamma) * (initial_cost - d) + cost_mean
    ) / cost_std
    assert torch.isfinite(cost_offset).all()
    print(f"[COST_STATS] initial_cost={initial_cost.tolist()}")
    print(f"[COST_STATS] mean={cost_mean.tolist()} std={cost_std.tolist()}")

    clip_param: float = 0.2
    loss_sum: float = 0.0
    reward_loss_sum: float = 0.0
    cost_loss_sum: float = 0.0
    first_ratio_mean: float = 0.0
    first_cost_grad_norm: float = 0.0
    # 分别累加六类约束的违反量，最后用于打印平均值。
    violation_sum: torch.Tensor = torch.zeros_like(cost_mean)
    update_count: int = 0

    for batch in storage.mini_batch_generator(
        num_mini_batches=4,
        num_learning_epochs=2,
    ):
        # 重建当前分布，计算已保存旧动作的密度；这里不重新采样动作。
        policy.update_distribution(batch["observations"])
        new_log_probs: torch.Tensor = policy.get_actions_log_prob(
            batch["actions"],
        ).unsqueeze(-1)  # (B,1)，与保存的旧对数密度对齐。
        old_log_probs: torch.Tensor = batch["log_probs"]
        normalized_advantages: torch.Tensor = (batch["reward_advantages"] - advantage_mean) / advantage_std
        assert new_log_probs.shape == old_log_probs.shape == normalized_advantages.shape

        # 每一种cost使用前面算好的均值和标准差。
        # B是这个小批量的记录数；每条记录有C个cost优势，形状为(B,C)。
        normalized_cost_advantages: torch.Tensor = (
            batch["cost_advantages"] - cost_mean
        ) / cost_std
        assert normalized_cost_advantages.shape == (new_log_probs.shape[0], k.numel())

        # rho = 新策略对旧动作的密度 / 采集策略对该动作的密度。
        ratio: torch.Tensor = torch.exp(new_log_probs - old_log_probs)
        if update_count == 0:
            # 第一次更新前，当前策略应与采集策略一致，rho应接近1。
            assert torch.allclose(
                ratio,
                torch.ones_like(ratio),
                rtol=1e-4,
                atol=1e-4,
            ), "Policy changed before its first update."
            first_ratio_mean = float(ratio.detach().mean().item())

        reward_loss, cost_loss, violation = compute_policy_losses(
            ratio=ratio,
            normalized_advantages=normalized_advantages,
            normalized_cost_advantages=normalized_cost_advantages,
            cost_offset=cost_offset,
            k=k,
            clip_param=clip_param,
        )

        loss: torch.Tensor = reward_loss + cost_loss
        assert loss.requires_grad, "Policy loss has no gradient graph."
        assert torch.isfinite(loss).all(), "Non-finite policy loss."

        if update_count == 0:
            # 单独计算cost损失的梯度，检查cost项能否影响策略参数。
            # 这里只返回梯度；后面仍由总loss.backward()计算更新所用的梯度。
            cost_grads: tuple[torch.Tensor | None, ...] = torch.autograd.grad(
                outputs=cost_loss,
                inputs=parameters,
                retain_graph=True,
                allow_unused=True,
            )
            first_cost_grad_norm = sum(
                float(grad.detach().square().sum().item())
                for grad in cost_grads if grad is not None
            ) ** 0.5

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters=parameters,
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
        # 当前直接学习std，检查每次更新后仍为有限的正数。
        assert torch.isfinite(policy.std).all(), "Non-finite policy std."
        assert (policy.std > 0.0).all(), "Non-positive policy std."

        loss_sum += float(loss.item())
        reward_loss_sum += float(reward_loss.item())
        cost_loss_sum += float(cost_loss.item())
        violation_sum += violation.detach()
        update_count += 1

    policy.eval()
    assert update_count == 8
    for parameter in parameters:
        assert torch.isfinite(parameter).all(), "Non-finite policy parameter."

    max_parameter_change = max(
        float((parameter.detach() - before).abs().max().item())
        for parameter, before in zip(parameters, parameters_before, strict=False)
    )
    assert max_parameter_change > 0.0, "Policy parameters did not change."
    print(
        f"[CONSTRAINED_POLICY_UPDATE] updates={update_count} "
        f"loss={loss_sum / update_count:.6f} "
        f"reward_loss={reward_loss_sum / update_count:.6f} "
        f"cost_loss={cost_loss_sum / update_count:.6f} "
        f"first_ratio_mean={first_ratio_mean:.6f} "
        f"first_cost_grad_norm={first_cost_grad_norm:.3e} "
        f"max_parameter_change={max_parameter_change:.3e}"
    )
    print(f"[COST_VIOLATION] update_mean={(violation_sum / update_count).tolist()}")


def main() -> None:
    """Launch the TITA task and check its actor interface."""

    parser = argparse.ArgumentParser(description="Check the saved TITA actor with a live environment.")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--episode_length_s", type=float, default=None)
    add_launcher_args(parser)
    args = parser.parse_args()
    if args.episode_length_s is not None and args.episode_length_s <= 0.0:
        parser.error("episode_length_s must be positive.")
    if args.num_envs < 1 or args.max_steps < 1:
        parser.error("num_envs and max_steps must be positive.")

    task = "TitaLab-Velocity-Tita"
    cfg, _ = resolve_task_config(task_name=task, agent_cfg_entry_point=None, overrides=["physics=isaacsim_physx"])

    # 任务注册决定实际配置类型；这里将该类型告诉Pylance。
    cfg = cast("VelocityEnvCfg", cfg)

    cfg.scene.num_envs = args.num_envs
    cfg.compute_final_obs = True
    cfg.seed = 42
    torch.manual_seed(42)

    if args.episode_length_s is not None:
        cfg.episode_length_s = args.episode_length_s

    # 新Lab启动接口负责启动与关闭Kit；环境仍由gym.make创建。
    with launch_simulation(cfg, args):
        native_env = gym.make(task, cfg=cfg)
        try:
            run_check(TitaNP3OEnvWrapper(native_env), args.max_steps)
        finally:
            native_env.close()


if __name__ == "__main__":
    main()
