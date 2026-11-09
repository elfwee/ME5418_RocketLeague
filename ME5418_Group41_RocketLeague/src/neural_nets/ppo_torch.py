"""
Proximal Policy Optimization (PPO) Actor-Critic Architecture & Forward/Backward Pass Demo.
"""
import math
import sys
from pathlib import Path
import gymnasium as gym
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

# Ensure repository root is on sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.gym_env.rocket_league_env import RocketLeagueEnv
from src.config.gym_config import GymConfig
from src.config.nn_config import PPOConfig


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO over a discrete action space.

    Architecture (matched 1-to-1 with Dueling DQN in `dueling_ddqn_torch.py`):
        - Input Layer: 52-dimensional normalized state vector in [-1, 1].
        - Shared Feature Extractor: 2 fully-connected layers (256 units) with LayerNorm and ReLU.
        - Two Heads (128 hidden units each):
            1. Actor (Policy) Head pi(a | s): outputs logits over each discrete action a (36 actions).
            2. Critic (Value) Head V(s): estimates scalar value of being in state s.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = PPOConfig.hidden_size):
        super(ActorCritic, self).__init__()

        def layer_init(layer: nn.Linear, std: float = math.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
            nn.init.orthogonal_(layer.weight, gain=std)
            nn.init.constant_(layer.bias, bias_const)
            return layer

        # Shared feature backbone with LayerNorm (hidden: gain = sqrt(2))
        self.fc1 = layer_init(nn.Linear(state_dim, hidden_dim))
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = layer_init(nn.Linear(hidden_dim, hidden_dim))
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Actor stream (hidden: gain = sqrt(2), policy output: gain = 0.01 for near-uniform exploration)
        self.fc_actor = layer_init(nn.Linear(hidden_dim, hidden_dim // 2))
        self.actor = layer_init(nn.Linear(hidden_dim // 2, action_dim), std=0.01)

        # Critic stream (hidden: gain = sqrt(2), value output: gain = 1.0)
        self.fc_critic = layer_init(nn.Linear(hidden_dim, hidden_dim // 2))
        self.critic = layer_init(nn.Linear(hidden_dim // 2, 1), std=1.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action logits and state value V(s) for input state tensor x."""
        # Shared feature extraction
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))

        # Compute action logits for pi(a | s)
        a = F.relu(self.fc_actor(x))
        logits = self.actor(a)

        # Compute state value V(s)
        v = F.relu(self.fc_critic(x))
        V = self.critic(v).squeeze(-1)
        return logits, V

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        """Sample (or evaluate a given) action, returning (action, log pi(a | s), entropy, V(s))."""
        logits, V = self(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), V


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


def run_ppo_network():
    """Assignment 2 Demo: Network Instantiation, Forward Pass & Backward Pass (Headless)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(GymConfig.gym_id, render_mode=None, bot_type=PPOConfig.bot_type, discrete_action=True)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    hidden_dim = PPOConfig.hidden_size

    # 1. Create Actor-Critic Network
    agent = ActorCritic(num_states, num_actions, hidden_dim).to(device)
    total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)

    optimizer = torch.optim.Adam(agent.parameters(), lr=PPOConfig.learning_rate, eps=1e-5)
    gamma = PPOConfig.discount_factor
    gae_lambda = PPOConfig.gae_lambda
    clip_coef = PPOConfig.clip_coef     # PPO surrogate clipping range epsilon
    vf_coef = PPOConfig.value_loss_coef  # Value loss weight
    ent_coef = PPOConfig.entropy_coef    # Entropy bonus weight
    rollout_steps = PPOConfig.demo_rollout_steps

    print("=" * 65)
    print("1. NETWORK ARCHITECTURE (PPO ACTOR-CRITIC)")
    print("=" * 65)
    print(
        f"Device: {device} | State Dim: {num_states} | Action Dim: {num_actions} | "
        f"Hidden Dim: {hidden_dim} | Trainable Params: {total_params:,}"
    )
    print(agent)

    # 2. Forward Pass with Environment Observation
    obs, info = env.reset()
    state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 52)

    logits, value = agent(state_tensor)  # Shapes: (1, 36), (1,)
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
    states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
    with torch.no_grad():
        for _ in range(rollout_steps):
            action, log_prob, _, v = agent.get_action_and_value(state_tensor)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state_tensor.squeeze(0))
            actions.append(action.squeeze(0))
            log_probs.append(log_prob.squeeze(0))
            values.append(v.squeeze(0))
            rewards.append(float(reward))
            dones.append(float(done))

            if done:
                next_state, _ = env.reset()
            state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        _, next_value = agent(state_tensor)

    states = torch.stack(states)                                                # Shape: (T, 52)
    actions = torch.stack(actions)                                              # Shape: (T,)
    old_log_probs = torch.stack(log_probs)                                      # Shape: (T,)
    values = torch.stack(values)                                                # Shape: (T,)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)         # Shape: (T,)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)             # Shape: (T,)

    # 3. Backward Pass (GAE, Clipped Surrogate Loss & Gradient Backpropagation)
    advantages, returns = compute_gae(rewards, values, dones, next_value.squeeze(0), gamma, gae_lambda)
    norm_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Re-evaluate rollout actions under the current policy
    _, new_log_probs, entropy, new_values = agent.get_action_and_value(states, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)

    # Clipped surrogate objective: L^CLIP = E[min(r_t * A_t, clip(r_t, 1 - eps, 1 + eps) * A_t)]
    pg_loss1 = -norm_advantages * ratio
    pg_loss2 = -norm_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    v_loss = F.mse_loss(new_values, returns)
    entropy_loss = entropy.mean()

    # Total loss: policy loss + weighted value loss - weighted entropy bonus
    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss
    optimizer.zero_grad()
    loss.backward()

    print("\n" + "=" * 65)
    print("3. BACKWARD PASS (LOSS & GRADIENTS)")
    print("=" * 65)
    print(f"Rollout Steps       : {rollout_steps} | Total Reward: {rewards.sum().item():.4f} | Episodes Done: {int(dones.sum().item())}")
    print(f"Mean Advantage (raw): {advantages.mean().item():.6f}")
    print(f"Mean Return         : {returns.mean().item():.6f}")
    print(f"Mean Ratio          : {ratio.mean().item():.6f} (should be ~1.0 before update)")
    print(f"Policy (Clip) Loss  : {pg_loss.item():.6f}")
    print(f"Value MSE Loss      : {v_loss.item():.6f}")
    print(f"Entropy             : {entropy_loss.item():.6f}")
    print(f"Total Loss          : {loss.item():.6f}")

    print("\nLayer Gradient Norms (L2):")
    for name, param in agent.named_parameters():
        if param.grad is not None:
            print(f"  {name:<22} | shape={str(list(param.shape)):<12} | grad_norm={param.grad.norm().item():.6f}")

    # Apply optimizer step and verify parameter update changes the policy ratio and value estimates
    nn.utils.clip_grad_norm_(agent.parameters(), max_norm=PPOConfig.max_grad_norm)
    optimizer.step()
    with torch.no_grad():
        _, log_probs_after, _, values_after = agent.get_action_and_value(states, actions)
        ratio_after = torch.exp(log_probs_after - old_log_probs)
        approx_kl = (old_log_probs - log_probs_after).mean()
    print(f"\nMean Ratio          : {ratio_after.mean().item():.6f} (after 1 optimizer step)")
    print(f"Approx KL           : {approx_kl.item():.6f}")
    print(f"Value MSE Loss      : {F.mse_loss(values_after, returns).item():.6f} (after 1 optimizer step)")
    print("=" * 65)

    env.close()


# Aliases for backward compatibility
run_network = run_ppo_network
test_ppo_torch = run_ppo_network


if __name__ == "__main__":
    run_ppo_network()
