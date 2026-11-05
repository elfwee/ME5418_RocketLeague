"""Proximal Policy Optimisation"""

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from src.neural_nets.a2c_torch import ActorCritic
from src.neural_nets.common import RolloutBuffer


class PPO:
    """Proximal Policy Optimisation (clipped surrogate) for discrete actions.

    The caller collects a rollout into a :class:`RolloutBuffer` with :meth:`act`,
    calls ``buffer.compute_returns_and_advantages`` (bootstrapped from V(s_last))
    and passes the buffer to :meth:`update`, which runs ``update_epochs`` of
    shuffled minibatch :meth:`_backprop` steps.

    Loss: ``policy_loss + value_loss_coef * critic_loss - entropy_coef * entropy``, where
    ``policy_loss = -E[min(r * A, clip(r, 1 - eps, 1 + eps) * A)]`` and ``r = pi(a|s) / pi_old(a|s)``.

    Args:
        obs_dim: Observation size.
        n_actions: Number of discrete actions.
        hidden_size: Hidden layer width.
        learning_rate: Adam learning rate.
        discount_factor: Gamma (used for GAE).
        gae_lambda: GAE lambda.
        clip_coef: Surrogate clipping range epsilon.
        value_loss_coef: Critic loss weight.
        entropy_coef: Entropy bonus weight.
        max_norm: Gradient norm clip.
        update_epochs: Passes over the rollout per update.
        minibatch_size: Minibatch size per gradient step.
        trunk_type: ``"independent"`` or ``"shared"`` (see ActorCritic).
    """

    def __init__(
        self,
        obs_dim=52,
        n_actions=36,
        hidden_size=256,
        learning_rate=3e-4,
        discount_factor=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_norm=0.5,
        update_epochs=4,
        minibatch_size=64,
        trunk_type="independent",
    ):
        self.model = ActorCritic(obs_dim, n_actions, hidden_size, trunk_type)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_norm = max_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

    def _forwardpass(self, states):
        """Run the network; returns ``(logits, values)``."""
        return self.model(states)

    @torch.no_grad()
    def act(self, observations):
        """Sample actions; returns ``(actions, log_probs, values)``."""
        logits, values = self._forwardpass(observations)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions), values

    def _backprop(self, states, actions, old_log_probs, advantages, returns):
        """One clipped-surrogate gradient step on a minibatch. Returns ``(loss, approx_kl)``."""
        logits, values = self._forwardpass(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(new_log_probs - old_log_probs)  # r = pi(a|s) / pi_old(a|s)
        unclipped = -advantages * ratio
        clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
        actor_loss = torch.max(unclipped, clipped).mean()
        critic_loss = F.mse_loss(values, returns)
        entropy = dist.entropy().mean()
        loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
        self.optimizer.step()

        with torch.no_grad():
            approx_kl = (old_log_probs - new_log_probs).mean()
        return loss.item(), approx_kl.item()

    def update(self, buffer: RolloutBuffer):
        """Run ``update_epochs`` of minibatch updates over a filled buffer. Returns the mean loss."""
        losses = []
        for _ in range(self.update_epochs):
            for states, actions, old_log_probs, advantages, returns in buffer.get_minibatches(self.minibatch_size):
                loss, _ = self._backprop(states, actions, old_log_probs, advantages, returns)
                losses.append(loss)
        return sum(losses) / max(len(losses), 1)

    @torch.no_grad()
    def predict(self, observation):
        """Return the greedy action for a single observation."""
        logits, _ = self._forwardpass(torch.as_tensor(observation, dtype=torch.float32))
        return int(logits.argmax())


def test_ppo_torch(num_updates=300, num_envs=8, rollout_length=128):
    """Smoke test on a dummy env (constant reward, random obs, ~5% episode end)."""
    agent = PPO(obs_dim=52, n_actions=36)
    buffer = RolloutBuffer(rollout_length, num_envs, obs_dim=52)
    obs = torch.randn(num_envs, 52)
    for update in range(1, num_updates + 1):
        buffer.reset()
        while not buffer.is_full():
            states = obs
            actions, log_probs, values = agent.act(states)
            obs = torch.randn(num_envs, 52)
            rewards = torch.ones(num_envs)
            dones = (torch.rand(num_envs) < 0.05).float()
            buffer.add(states, actions, log_probs, rewards, values, dones)
        # Bootstrap from V(s_last), then compute GAE advantages / returns
        _, _, last_values = agent.act(obs)
        buffer.compute_returns_and_advantages(last_values, agent.discount_factor, agent.gae_lambda)
        loss = agent.update(buffer)
        if update % 30 == 0:
            print(f"  update: {update:5d}  loss: {loss:6.2f}  mean V: {buffer.values.mean().item():6.2f}")


if __name__ == "__main__":
    test_ppo_torch()
