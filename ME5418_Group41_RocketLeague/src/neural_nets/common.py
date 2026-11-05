"""Shared utilities for on-policy agents."""

import torch


class RolloutBuffer:
    """Fixed-size on-policy rollout storage with GAE.

    Tensors are stored as ``(rollout_length, num_envs, ...)``. Fill with :meth:`add`,
    call :meth:`compute_returns_and_advantages` once full, then iterate
    :meth:`get_minibatches`. Call :meth:`reset` before the next rollout.

    Args:
        rollout_length: Steps per env per rollout (T).
        num_envs: Number of parallel envs (N).
        obs_dim: Observation size.
        device: Torch device for the storage tensors.
    """

    def __init__(self, rollout_length, num_envs, obs_dim, device="cpu"):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = torch.device(device)

        shape = (rollout_length, num_envs)
        self.states = torch.zeros(*shape, obs_dim, device=self.device)
        self.actions = torch.zeros(shape, dtype=torch.long, device=self.device)
        self.log_probs = torch.zeros(shape, device=self.device)
        self.rewards = torch.zeros(shape, device=self.device)
        self.values = torch.zeros(shape, device=self.device)
        self.dones = torch.zeros(shape, device=self.device)  # 1.0 if the episode ended after step t
        self.advantages = torch.zeros(shape, device=self.device)
        self.returns = torch.zeros(shape, device=self.device)
        self.pos = 0

    def add(self, states, actions, log_probs, rewards, values, dones):
        """Store one step for all envs."""
        if self.is_full():
            raise RuntimeError("RolloutBuffer is full; call reset() first.")
        self.states[self.pos] = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        self.actions[self.pos] = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        self.log_probs[self.pos] = torch.as_tensor(log_probs, dtype=torch.float32, device=self.device)
        self.rewards[self.pos] = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        self.values[self.pos] = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        self.dones[self.pos] = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        self.pos += 1

    def is_full(self):
        return self.pos >= self.rollout_length

    def reset(self):
        self.pos = 0

    def compute_returns_and_advantages(self, last_values, gamma, gae_lambda):
        """Fill ``advantages`` and ``returns`` with GAE, bootstrapping from ``last_values`` = V(s_T).

        delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
        """
        last_values = torch.as_tensor(last_values, dtype=torch.float32, device=self.device)
        last_gae = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.rollout_length)):
            next_values = last_values if t == self.rollout_length - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * not_done * next_values - self.values[t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            self.advantages[t] = last_gae
        self.returns = self.advantages + self.values

    def get_minibatches(self, minibatch_size):
        """Yield shuffled ``(states, actions, old_log_probs, advantages, returns)`` minibatches."""
        batch_size = self.rollout_length * self.num_envs
        states = self.states.reshape(batch_size, self.obs_dim)
        actions = self.actions.reshape(batch_size)
        log_probs = self.log_probs.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        returns = self.returns.reshape(batch_size)

        perm = torch.randperm(batch_size, device=self.device)
        for start in range(0, batch_size, minibatch_size):
            idx = perm[start : start + minibatch_size]
            yield states[idx], actions[idx], log_probs[idx], advantages[idx], returns[idx]
