"""
Proximal Policy Optimization (PPO) Actor-Critic Network, Rollout Buffer & Forward/Backward Pass Demo.
"""
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

# Ensure repository root is on sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.gym_config import GymConfig
from src.config.nn_config import PPOConfig
from src.gym_env.rocket_league_env import RocketLeagueEnv


class PPO(nn.Module):
    """Actor-Critic network for PPO over a discrete action space.

    Architecture (matched 1-to-1 with Dueling DQN in `dueling_ddqn_torch.py`):
        - Input Layer: 52-dimensional normalized state vector in [-1, 1].
        - Shared Feature Extractor: 2 fully-connected layers (256 units) with LayerNorm and ReLU.
        - Two Heads (128 hidden units each):
            1. Actor (Policy) Head pi(a | s): outputs logits over each discrete action a (36 actions).
            2. Critic (Value) Head V(s): estimates scalar value of being in state s.
    """

    def __init__(
        self,
        state_dim: int = PPOConfig.input_dim[0],
        action_dim: int = PPOConfig.output_dim,
        hidden_dim: int = PPOConfig.hidden_size,
    ):
        super().__init__()
        # BUILD NEURAL NETWORK
        # Shared feature backbone with LayerNorm to stabilize hidden activations across varying scales
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Actor stream pi(a | s): R^(hidden_dim) -> R^(action_dim) logits
        self.fc_actor = nn.Linear(hidden_dim, hidden_dim // 2)
        self.actor = nn.Linear(hidden_dim // 2, action_dim)

        # Critic stream V(s): R^(hidden_dim) -> R^1
        self.fc_critic = nn.Linear(hidden_dim, hidden_dim // 2)
        self.critic = nn.Linear(hidden_dim // 2, 1)

        # SET OPTIMISER
        self.optimiser = torch.optim.Adam(self.parameters(), lr=PPOConfig.learning_rate, eps=1e-5)

    @property
    def device(self) -> torch.device:
        return self.fc1.weight.device

    def _to_batch(self, obs) -> torch.Tensor:
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        return obs.unsqueeze(0) if obs.dim() == 1 else obs

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------
    def logits_and_value(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action logits and state value V(s)."""
        # Shared feature extraction
        x = F.relu(self.ln1(self.fc1(obs)))
        x = F.relu(self.ln2(self.fc2(x)))

        # Compute action logits for pi(a | s)
        logits = self.actor(F.relu(self.fc_actor(x)))

        # Compute state value V(s)
        value = self.critic(F.relu(self.fc_critic(x))).squeeze(-1)
        return logits, value

    def forward(self, obs, actions=None):
        """Returns (actions, log_prob, entropy, value). Samples actions if none are given."""
        logits, value = self.logits_and_value(self._to_batch(obs))
        dist = Categorical(logits=logits)
        if actions is None:
            actions = dist.sample()
        return actions, dist.log_prob(actions), dist.entropy(), value

    # ------------------------------------------------------------------
    # Agent-facing helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """Pick an action for a single observation. Returns plain Python values."""
        if deterministic:
            logits, _ = self.logits_and_value(self._to_batch(obs))
            return int(logits.argmax(dim=-1)[0]), 0.0, 0.0
        action, log_prob, _, value = self.forward(obs)
        return int(action[0]), float(log_prob[0]), float(value[0])

    @torch.no_grad()
    def get_value(self, obs):
        """State value for a single observation (used to bootstrap at the end of a rollout)."""
        _, value = self.logits_and_value(self._to_batch(obs))
        return float(value[0])

    # ------------------------------------------------------------------
    # Back propagation
    # ------------------------------------------------------------------
    def compute_loss(self, obs, actions, old_log_prob, advantages, returns):
        """PPO clipped surrogate loss on a minibatch. Returns (loss, stats)."""
        _, new_log_prob, entropy, value = self.forward(obs, actions)
        ratio = torch.exp(new_log_prob - old_log_prob)

        # Clipped surrogate objective: L^CLIP = E[min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t)]
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - PPOConfig.clip_coef, 1.0 + PPOConfig.clip_coef) * advantages
        policy_loss = -torch.min(unclipped, clipped).mean()

        value_loss = F.mse_loss(value, returns)
        entropy_mean = entropy.mean()

        # Total loss: policy loss + weighted value loss - weighted entropy bonus
        loss = policy_loss + PPOConfig.value_loss_coef * value_loss - PPOConfig.entropy_coef * entropy_mean

        with torch.no_grad():
            approx_kl = (old_log_prob - new_log_prob).mean()
        stats = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_mean.item(),
            "approx_kl": approx_kl.item(),
            "ratio": ratio.mean().item(),
        }
        return loss, stats

    def backward(self, obs, actions, old_log_prob, advantages, returns):
        """One gradient step on a minibatch using the PPO clipped loss."""
        loss, stats = self.compute_loss(obs, actions, old_log_prob, advantages, returns)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=PPOConfig.max_grad_norm)
        self.optimiser.step()
        return stats

    def update(
        self,
        buffer,
        epochs=PPOConfig.update_epochs,
        batch_size=PPOConfig.minibatch_size,
        target_kl=PPOConfig.target_kl,
    ):
        """Learn from a finished RolloutBuffer. The Agent calls this once per rollout."""
        stats = {}
        for _ in range(epochs):
            kls = []
            for mb in buffer.minibatches(batch_size):
                stats = self.backward(*mb)
                kls.append(stats["approx_kl"])
            if target_kl is not None and np.mean(kls) > target_kl:  # policy moved far enough; stop early
                break
        return stats


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = PPOConfig.discount_factor,
    gae_lambda: float = PPOConfig.gae_lambda,
):
    """Generalized Advantage Estimation over a rollout of length T.

    delta_t = r_t + gamma * (1 - done_t) * V(s_{t+1}) - V(s_t)
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        next_v = next_value if t == T - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * not_done * next_v - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class RolloutBuffer:
    """Stores one batch of experience and turns it into training data."""

    def __init__(self, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
        self.advantages = self.returns = None

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.values.append(value)

    def __len__(self):
        return len(self.rewards)

    def _tensor(self, data, dtype=torch.float32):
        return torch.as_tensor(np.array(data), dtype=dtype, device=self.device)

    def finish(self, last_value, gamma=PPOConfig.discount_factor, lam=PPOConfig.gae_lambda):
        """Compute advantages (GAE) and returns once the rollout is complete."""
        adv, self.returns = compute_gae(
            self._tensor(self.rewards),
            self._tensor(self.values),
            self._tensor(self.dones),
            torch.tensor(float(last_value), device=self.device),
            gamma,
            lam,
        )
        self.raw_advantages = adv
        self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

    def minibatches(self, batch_size):
        """Yield shuffled minibatches as tensors."""
        obs = self._tensor(self.obs)
        actions = self._tensor(self.actions, torch.long)
        log_probs = self._tensor(self.log_probs)
        idx = torch.randperm(len(self), device=self.device)
        for start in range(0, len(self), batch_size):
            mb = idx[start:start + batch_size]
            yield obs[mb], actions[mb], log_probs[mb], self.advantages[mb], self.returns[mb]


def run_ppo_network():
    """Assignment 2 Demo: Network Instantiation, Forward Pass & Backward Pass (Headless)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(GymConfig.gym_id, render_mode=None, bot_type=PPOConfig.bot_type, discrete_action=True)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    hidden_dim = PPOConfig.hidden_size
    rollout_steps = PPOConfig.demo_rollout_steps

    # 1. Create Actor-Critic Network
    ppo = PPO(num_states, num_actions, hidden_dim).to(device)
    total_params = sum(p.numel() for p in ppo.parameters() if p.requires_grad)

    print("=" * 65)
    print("1. NETWORK ARCHITECTURE (PPO ACTOR-CRITIC)")
    print("=" * 65)
    print(
        f"Device: {device} | State Dim: {num_states} | Action Dim: {num_actions} | "
        f"Hidden Dim: {hidden_dim} | Trainable Params: {total_params:,}"
    )
    print(ppo)

    # 2. Forward Pass with Environment Observation
    obs, info = env.reset()
    state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 52)

    logits, value = ppo.logits_and_value(state_tensor)  # Shapes: (1, 36), (1,)
    probs = F.softmax(logits, dim=-1)
    action_idx = probs.argmax(dim=-1).item()

    # Decode discrete action index (0..35) into (direction, jump, boost)
    dir_idx, jump_bit, boost_bit = action_idx // 4, (action_idx % 4) // 2, action_idx % 2

    print("\n" + "=" * 65)
    print("2. FORWARD PASS (INPUT -> OUTPUT)")
    print("=" * 65)
    print(f"Input State Shape : {state_tensor.shape}")
    print(f"Input State (first 8 features): {state_tensor[0, :8].detach().cpu().numpy().round(4)}")
    print(f"Output Logits Shape : {logits.shape}")
    print(f"Output pi(.|s) Sample (first 6 actions): {probs[0, :6].detach().cpu().numpy().round(4)}")
    print(f"Output V(s)         : {value.item():.6f}")
    print(
        f"Greedy Action     : {action_idx} -> dir={GymConfig.directions[dir_idx]}, "
        f"jump={bool(jump_bit)}, boost={bool(boost_bit)}"
    )

    # Collect a short on-policy rollout of transitions (s, a, log pi_old(a|s), r, done, V(s))
    buffer = RolloutBuffer(device)
    episodes_done = 0
    for _ in range(rollout_steps):
        action, log_prob, v = ppo.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.add(obs, action, log_prob, float(reward), done, v)
        episodes_done += int(done)
        obs = env.reset()[0] if done else next_obs

    # 3. Backward Pass (GAE, Clipped Surrogate Loss & Gradient Backpropagation)
    buffer.finish(ppo.get_value(obs))
    obs_t, actions_t, old_log_probs_t, advantages_t, returns_t = next(buffer.minibatches(len(buffer)))

    loss, stats = ppo.compute_loss(obs_t, actions_t, old_log_probs_t, advantages_t, returns_t)
    ppo.optimiser.zero_grad()
    loss.backward()

    print("\n" + "=" * 65)
    print("3. BACKWARD PASS (LOSS & GRADIENTS)")
    print("=" * 65)
    print(f"Rollout Steps       : {rollout_steps} | Total Reward: {sum(buffer.rewards):.4f} | Episodes Done: {episodes_done}")
    print(f"Mean Advantage (raw): {buffer.raw_advantages.mean().item():.6f}")
    print(f"Mean Return         : {buffer.returns.mean().item():.6f}")
    print(f"Mean Ratio          : {stats['ratio']:.6f} (should be ~1.0 before update)")
    print(f"Policy (Clip) Loss  : {stats['policy_loss']:.6f}")
    print(f"Value MSE Loss      : {stats['value_loss']:.6f}")
    print(f"Entropy             : {stats['entropy']:.6f}")
    print(f"Total Loss          : {stats['loss']:.6f}")

    print("\nLayer Gradient Norms (L2):")
    for name, param in ppo.named_parameters():
        if param.grad is not None:
            print(f"  {name:<22} | shape={str(list(param.shape)):<12} | grad_norm={param.grad.norm().item():.6f}")

    # Apply optimizer step and verify parameter update changes the policy ratio and value estimates
    nn.utils.clip_grad_norm_(ppo.parameters(), max_norm=PPOConfig.max_grad_norm)
    ppo.optimiser.step()
    with torch.no_grad():
        _, stats_after = ppo.compute_loss(obs_t, actions_t, old_log_probs_t, advantages_t, returns_t)
    print(f"\nMean Ratio          : {stats_after['ratio']:.6f} (after 1 optimizer step)")
    print(f"Approx KL           : {stats_after['approx_kl']:.6f}")
    print(f"Value MSE Loss      : {stats_after['value_loss']:.6f} (after 1 optimizer step)")
    print("=" * 65)

    env.close()


def test_nn(num_updates=100, rollout_len=PPOConfig.rollout_steps):
    """Minimal training loop: collect a rollout, compute GAE, run PPO updates."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo = PPO().to(device)
    buffer = RolloutBuffer(device)
    env = RocketLeagueEnv()
    obs, _ = env.reset()

    for update in range(num_updates):
        buffer.reset()

        # Play and record
        for _ in range(rollout_len):
            action, log_prob, value = ppo.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(obs, action, log_prob, reward, done, value)
            obs = next_obs
            if done:
                obs, _ = env.reset()

        # Compute advantages, then learn
        buffer.finish(ppo.get_value(obs))
        stats = ppo.update(buffer)
        print(f"update: {update}, policy_loss: {stats['policy_loss']:.2f}, value_loss: {stats['value_loss']:.2f}")


# Aliases for backward compatibility
run_network = run_ppo_network
test_ppo_torch = run_ppo_network


if __name__ == "__main__":
    run_ppo_network()
