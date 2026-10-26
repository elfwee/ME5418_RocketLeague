import torch
from torch import nn
from torch.nn import functional as F
import gymnasium as gym
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rocket_league_env import RocketLeagueEnv

class DQN(nn.Module):
    """Deep Q-Network supporting both Standard and Dueling DQN architectures."""

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        if self.enable_dueling_dqn:
            # Value stream V(s)
            self.fc_value = nn.Linear(hidden_dim, hidden_dim // 2)
            self.value = nn.Linear(hidden_dim // 2, 1)

            # Advantage stream A(s, a)
            self.fc_advantages = nn.Linear(hidden_dim, hidden_dim // 2)
            self.advantages = nn.Linear(hidden_dim // 2, action_dim)
        else:
            self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))

        if self.enable_dueling_dqn:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Q(s, a) = V(s) + (A(s, a) - mean_a(A(s, a)))
            Q = V + A - torch.mean(A, dim=-1, keepdim=True)
            return Q
        else:
            return self.fc3(x)


if __name__ == "__main__":
    device = "cpu"
    env = gym.make("rocket-league-v1", render_mode="human", bot_type="heuristic_bot", discrete_action=True)

    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.n

    fcl_nodes = 256
    policy_dqn = DQN(num_states, num_actions, fcl_nodes, enable_dueling_dqn=True).to(device)

    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device)

    with torch.no_grad():
        action = policy_dqn(state.unsqueeze(0)).squeeze(0).argmax()
        action_idx = action.item()

    print(f"State shape: {state.shape}, State: {state}")
    print(f"Total actions: {num_actions}, Action index: {action_idx}")