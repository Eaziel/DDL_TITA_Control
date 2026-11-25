"""Rollout storage for TITA policy and value learning."""

from __future__ import annotations

from collections.abc import Iterator

import torch


class TitaRolloutStorage:
    """Store fixed-length batches of TITA environment transitions."""

    def __init__(
        self,
        num_envs: int,
        num_steps: int,
        num_obs: int = 586,
        num_actions: int = 8,
        num_costs: int = 6,
        device: str | torch.device = "cpu",
    ) -> None:
        """Allocate per-step observations, outcomes, and network predictions."""

        if min(num_envs, num_steps, num_obs, num_actions, num_costs) < 1:
            raise ValueError("Storage dimensions must be positive.")

        self.num_envs = num_envs
        self.num_steps = num_steps
        self.device = torch.device(device)

        # 第0维是采集时间，第1维是机器人，第2维是对应特征。
        self.observations = torch.zeros(num_steps, num_envs, num_obs, device=self.device)
        self.actions = torch.zeros(num_steps, num_envs, num_actions, device=self.device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, device=self.device)
        self.costs = torch.zeros(num_steps, num_envs, num_costs, device=self.device)

        # dones表示回合结束；time_outs另外保留是否因时间上限结束。
        self.dones = torch.zeros(num_steps, num_envs, 1, dtype=torch.bool, device=self.device)
        self.time_outs = torch.zeros_like(self.dones)

        # 都是采集动作时的旧网络结果，以后更新网络时仍须保留这些数值。
        self.reward_values = torch.zeros_like(self.rewards)
        self.cost_values = torch.zeros_like(self.costs)
        self.log_probs = torch.zeros_like(self.rewards)
        self.action_mean = torch.zeros_like(self.actions)
        self.action_std = torch.zeros_like(self.actions)

        # 纯超时位置保存reset前最终状态的价值；其他位置为0。
        self.timeout_reward_values = torch.zeros_like(self.reward_values)
        self.timeout_cost_values = torch.zeros_like(self.cost_values)

        # 回报目标供critic拟合；优势供后续策略/约束更新使用。
        self.reward_returns = torch.zeros_like(self.reward_values)
        self.cost_returns = torch.zeros_like(self.cost_values)
        self.reward_advantages = torch.zeros_like(self.reward_values)
        self.cost_advantages = torch.zeros_like(self.cost_values)

        # 已保存的策略步数，同时也是下一次写入的时间索引。
        self.step = 0

        # 当前这批的回报目标与优势是否已经计算完成。
        self._returns_ready = False

    @property
    def is_full(self) -> bool:
        """Return whether the configured rollout has been collected."""
        return self.step == self.num_steps

    def add(
        self,
        *,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        costs: torch.Tensor,
        dones: torch.Tensor,
        time_outs: torch.Tensor,
        reward_values: torch.Tensor,
        cost_values: torch.Tensor,
        log_probs: torch.Tensor,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        timeout_reward_values: torch.Tensor,
        timeout_cost_values: torch.Tensor,
    ) -> None:
        """Copy one parallel environment step into persistent storage."""

        if self.step >= self.num_steps:
            raise RuntimeError("Rollout storage is full.")

        # 每项依次为：名称、保存整段数据的张量、当前这一步的数据。
        # 标量结果由(N,)增加一列成为(N,1)，与存储的一步切片一致。
        entries = (
            ("observations", self.observations, observations),
            ("actions", self.actions, actions),
            ("rewards", self.rewards, rewards.unsqueeze(-1)),
            ("costs", self.costs, costs),
            ("dones", self.dones, dones.unsqueeze(-1)),
            ("time_outs", self.time_outs, time_outs.unsqueeze(-1)),
            ("reward_values", self.reward_values, reward_values),
            ("cost_values", self.cost_values, cost_values),
            ("log_probs", self.log_probs, log_probs.unsqueeze(-1)),
            ("action_mean", self.action_mean, action_mean),
            ("action_std", self.action_std, action_std),
            ("timeout_reward_values", self.timeout_reward_values, timeout_reward_values),
            ("timeout_cost_values", self.timeout_cost_values, timeout_cost_values),
        )

        # 先检查所有形状，避免copy_的广播掩盖时间或环境维度错误。
        for name, destination, source in entries:
            expected = destination[self.step].shape
            if source.shape != expected:
                raise ValueError(f"Expected '{name}' with shape {tuple(expected)}, got {tuple(source.shape)}.")

        for _, destination, source in entries:
            # 保存独立数值，不保留采样时的梯度计算图。
            destination[self.step].copy_(source.detach())

        self.step += 1

    @torch.no_grad()
    def compute_returns(
        self,
        last_reward_values: torch.Tensor,
        last_cost_values: torch.Tensor,
        *,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> None:
        """Compute GAE advantages and value targets for a full rollout."""

        if not self.is_full:
            raise RuntimeError("Collect a full rollout before computing returns.")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
            raise ValueError("gamma and lam must be in [0, 1].")
        if last_reward_values.shape != self.reward_values[-1].shape:
            raise ValueError("last_reward_values must have shape (N, 1).")
        if last_cost_values.shape != self.cost_values[-1].shape:
            raise ValueError("last_cost_values must have shape (N, C).")

        # 从采集末尾倒推；初始时，本批之外没有继续累加的TD误差。
        reward_advantage = torch.zeros_like(last_reward_values)
        cost_advantage = torch.zeros_like(last_cost_values)

        self._returns_ready = False

        for step in reversed(range(self.num_steps)):
            if step == self.num_steps - 1:
                next_reward_values = last_reward_values
                next_cost_values = last_cost_values
            else:
                next_reward_values = self.reward_values[step + 1]
                next_cost_values = self.cost_values[step + 1]

            # (N,1)：回合未结束为1；失败或超时reset时为0。
            continuation = (~self.dones[step]).to(dtype=self.rewards.dtype)
            time_out = self.time_outs[step].to(dtype=self.rewards.dtype)

            # 普通步使用下一状态价值；纯超时使用reset前最终状态价值。
            # 失败时两项均为0；timeout_*在所有非纯超时位置也为0。
            reward_bootstrap = continuation * next_reward_values + time_out * self.timeout_reward_values[step]
            cost_bootstrap = continuation * next_cost_values + time_out * self.timeout_cost_values[step]

            # δ_t = 当步结果 + γ * 后续价值 - 当前旧价值。
            reward_delta = self.rewards[step] + gamma * reward_bootstrap - self.reward_values[step]
            cost_delta = self.costs[step] + gamma * cost_bootstrap - self.cost_values[step]

            # A_t = δ_t + γλ * A_{t+1}；reset处停止跨步递推。
            reward_advantage = reward_delta + gamma * lam * continuation * reward_advantage
            cost_advantage = cost_delta + gamma * lam * continuation * cost_advantage

            self.reward_advantages[step].copy_(reward_advantage)
            self.cost_advantages[step].copy_(cost_advantage)

            # 供critic拟合的目标：旧价值 + 优势估计。
            self.reward_returns[step].copy_(self.reward_values[step] + reward_advantage)
            self.cost_returns[step].copy_(self.cost_values[step] + cost_advantage)

        self._returns_ready = True

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_learning_epochs: int = 1,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield aligned mini-batches from a rollout with computed targets."""

        if not self.is_full or not self._returns_ready:
            raise RuntimeError("Collect a full rollout and compute returns first.")
        if num_mini_batches < 1 or num_learning_epochs < 1:
            raise ValueError("Mini-batch and epoch counts must be positive.")

        # 一批中的记录总数：策略步数T × 并行环境数N。
        batch_size = self.num_steps * self.num_envs
        if batch_size % num_mini_batches != 0:
            raise ValueError("Rollout size must be divisible by num_mini_batches.")
        mini_batch_size = batch_size // num_mini_batches

        # 每个字段的第(t,n)项都属于同一条经历，之后必须一起选行。
        fields = {
            "observations": self.observations,
            "actions": self.actions,
            "reward_values": self.reward_values,
            "cost_values": self.cost_values,
            "log_probs": self.log_probs,
            "action_mean": self.action_mean,
            "action_std": self.action_std,
            "reward_returns": self.reward_returns,
            "cost_returns": self.cost_returns,
            "reward_advantages": self.reward_advantages,
            "cost_advantages": self.cost_advantages,
        }
        # (T,N,D) → (T*N,D)；586维观测中的10帧历史仍保留原来的顺序。
        flat_fields = {name: values.flatten(0, 1) for name, values in fields.items()}

        for _ in range(num_learning_epochs):
            # 每轮重新排列全部记录的编号，每条记录恰好出现一次。
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                batch_indices = indices[start : start + mini_batch_size]
                # 同一组编号用于全部字段，yield每次交出一份小批量数据。
                yield {name: values[batch_indices] for name, values in flat_fields.items()}

    def clear(self) -> None:
        """Mark all stored transitions as consumed."""
        # 下次从第0步覆盖；只有[0:self.step]是当前有效数据。
        self.step = 0
        self._returns_ready = False
