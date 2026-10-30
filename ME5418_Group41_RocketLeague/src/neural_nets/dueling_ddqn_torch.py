"""
Dueling Deep Q-Network (DQN) Architecture & Forward/Backward Pass Demo.
"""
import sys
from pathlib import Path
import gymnasium as gym
import torch
from torch import nn
from torch.nn import functional as F

# Ensure repository root is on sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.gym_env.rocket_league_env import RocketLeagueEnv
from src.config.gym_config import GymConfig

class DQN(nn.Module):
    """Deep Q-Network supporting both Standard and Dueling DQN architectures.

    Architecture:
        - Input Layer: 52-dimensional normalized state vector in [-1, 1].
        - Shared Feature Extractor: 2 fully-connected layers with LayerNorm and ReLU.
        - Dueling Streams (when enable_dueling_dqn=True):
            1. State-Value Stream V(s): estimates scalar value of being in state s.
            2. Action-Advantage Stream A(s, a): estimates relative advantage of each action a.
            Combined via identifiability-stabilized aggregation:
                Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, enable_dueling_dqn: bool = True):
        super(DQN, self).__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        # Shared feature backbone with LayerNorm to stabilize hidden activations across varying scales
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        if self.enable_dueling_dqn:
            # State-Value stream V(s): R^(hidden_dim) -> R^1
            self.fc_value = nn.Linear(hidden_dim, hidden_dim // 2)
            self.value = nn.Linear(hidden_dim // 2, 1)

            # Action-Advantage stream A(s, a): R^(hidden_dim) -> R^(action_dim)
            self.fc_advantages = nn.Linear(hidden_dim, hidden_dim // 2)
            self.advantages = nn.Linear(hidden_dim // 2, action_dim)
        else:
            # Standard single-stream Q-value head: R^(hidden_dim) -> R^(action_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute action-value estimates Q(s, a) for input state tensor x."""
        # Shared feature extraction
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))

        if self.enable_dueling_dqn:
            # Compute state value V(s)
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Compute action advantages A(s, a)
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Combine streams: subtracting mean advantage ensures V(s) and A(s, a) are uniquely identifiable
            Q = V + A - torch.mean(A, dim=-1, keepdim=True)
            return Q
        else:
            return self.fc3(x)


def run_network(enable_dueling_dqn = True):
    """Assignment 2 Demo: Network Instantiation, Forward Pass & Backward Pass (Headless)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(GymConfig.gym_id, render_mode=None, bot_type="heuristic_bot", discrete_action=True)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n
    hidden_dim = 256

    # 1. Create Policy and Target Dueling DQN Networks
    policy_dqn = DQN(num_states, num_actions, hidden_dim, enable_dueling_dqn).to(device)
    target_dqn = DQN(num_states, num_actions, hidden_dim, enable_dueling_dqn).to(device)
    target_dqn.load_state_dict(policy_dqn.state_dict())
    target_dqn.eval()

    optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=1e-3)
    criterion = nn.SmoothL1Loss()  # Huber loss for Temporal-Difference (TD) error
    gamma = 0.99

    print("=" * 65)
    print("1. NETWORK ARCHITECTURE (DUELING DQN)")
    print("=" * 65)
    print(f"Device: {device} | State Dim: {num_states} | Action Dim: {num_actions} | Hidden Dim: {hidden_dim}")
    print(policy_dqn)

    # 2. Forward Pass with Environment Observation
    obs, info = env.reset()
    state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 52)

    q_values = policy_dqn(state_tensor)  # Shape: (1, 36)
    action_idx = q_values.argmax(dim=-1).item()

    # Decode discrete action index (0..35) into (direction, jump, boost)
    dir_idx, jump_bit, boost_bit = action_idx // 4, (action_idx % 4) // 2, action_idx % 2

    print("\n" + "=" * 65)
    print("2. FORWARD PASS (INPUT -> OUTPUT)")
    print("=" * 65)
    print(f"Input State Shape : {state_tensor.shape}")
    print(f"Input State (first 8 features): {state_tensor[0, :8].detach().cpu().numpy().round(4)}")
    print(f"Output Q(s, .) Shape: {q_values.shape}")
    print(f"Output Q(s, .) Sample (first 6 actions): {q_values[0, :6].detach().cpu().numpy().round(4)}")
    print(f"Selected Action   : {action_idx} -> dir={GymConfig.directions[dir_idx]}, jump={bool(jump_bit)}, boost={bool(boost_bit)}")

    # Step environment once with the selected action to observe transition (s, a, r, s', done)
    next_state, reward, terminated, truncated, _ = env.step(action_idx)
    done = terminated or truncated
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

    # 3. Backward Pass (TD Target, Loss Computation & Gradient Backpropagation)
    with torch.no_grad():
        # Double DQN target: select best next action using policy network, evaluate using target network
        next_best_action = policy_dqn(next_state_tensor).argmax(dim=-1, keepdim=True)
        next_q_value = target_dqn(next_state_tensor).gather(1, next_best_action).squeeze(1)
        q_target = torch.tensor([reward], dtype=torch.float32, device=device) + gamma * (1.0 - float(done)) * next_q_value

    # Predicted Q(s, a) for the executed action
    q_pred = q_values.gather(1, torch.tensor([[action_idx]], dtype=torch.long, device=device)).squeeze(1)

    # Compute Huber TD loss and backpropagate gradients
    loss = criterion(q_pred, q_target)
    optimizer.zero_grad()
    loss.backward()

    print("\n" + "=" * 65)
    print("3. BACKWARD PASS (LOSS & GRADIENTS)")
    print("=" * 65)
    print(f"Env Step Reward     : {reward:.4f} | Done: {done}")
    print(f"Predicted Q(s, a={action_idx}) : {q_pred.item():.6f} (before update)")
    print(f"Target Q-value (y)  : {q_target.item():.6f}")
    print(f"TD Huber Loss       : {loss.item():.6f}")

    print("\nLayer Gradient Norms (L2):")
    for name, param in policy_dqn.named_parameters():
        if param.grad is not None:
            print(f"  {name:<22} | shape={str(list(param.shape)):<12} | grad_norm={param.grad.norm().item():.6f}")

    # Apply optimizer step and verify parameter update changes Q(s, a)
    optimizer.step()
    with torch.no_grad():
        q_pred_after = policy_dqn(state_tensor)[0, action_idx].item()
    print(f"\nPredicted Q(s, a={action_idx}) : {q_pred_after:.6f} (after 1 optimizer step)")
    print("=" * 65)

    env.close()


if __name__ == "__main__":
    run_network()
