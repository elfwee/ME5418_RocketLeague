"""
Dueling Double DQN (D3QN) Learning Agent — Training & Validation/Evaluation Pipeline.

Adapted from johnnycode8's PyTorch DQN tutorial (https://github.com/johnnycode8/dqn_pytorch)
and integrated with `src/neural_nets/dueling_ddqn_torch.py`, `src/config/nn_config.py`, and
`src/gym_env/rocket_league_env.py`.
"""

import argparse
from datetime import datetime, timedelta
import itertools
import os
from pathlib import Path
import random
import sys

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from torch import nn

# Ensure repository root is on sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.gym_config import GymConfig
from src.config.nn_config import DQNConfig
from src.gym_env.rocket_league_env import RocketLeagueEnv
from src.learning_agents.experience_replay import ReplayMemory
from src.neural_nets.dueling_ddqn_torch import DQN

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
RUNS_DIR = ROOT_DIR / DQNConfig.runs_dir
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib

device = "cuda" if torch.cuda.is_available() else "cpu"


class DuelingDDQNAgent:
    """Dueling Double DQN Learning Agent supporting both training and validation/evaluation.

    Key algorithmic components:
        1. Dueling Double DQN architecture (`DQN` from `src.neural_nets.dueling_ddqn_torch`):
           Decouples state-value stream V(s) and action-advantage stream A(s, a), and
           uses the online policy network to select argmax next actions while the target
           network evaluates them to mitigate maximization bias.
        2. Experience Replay (`ReplayMemory`):
           Stores `(state, action, next_state, reward, terminated, truncated)` transitions.
        3. Action Repeat with Edge-Triggered Jump Release (`_step_with_action_repeat`):
           Repeats chosen discrete action for `action_repeat` frames while clearing the jump
           bit on repeat frames > 0 so `Car._apply_jump` edge-triggering resets cleanly.
        4. Truncation-Aware Value Bootstrapping (`optimize`):
           Bootstraps V(s') on time-limit `truncated` episodes and only zeroes future value
           on true goal `terminated` states (`target_q = r + (1 - terminated) * gamma * next_q`).
    """

    def __init__(
        self,
        hyperparameter_set: str = DQNConfig.hyperparameter_set,
        enable_dueling_dqn: bool | None = None,
        enable_double_dqn: bool | None = None,
    ):
        self.hyperparameter_set = hyperparameter_set
        self.env_id = GymConfig.gym_id
        self.bot_type = DQNConfig.bot_type
        self.replay_memory_size = DQNConfig.buffer_size
        self.mini_batch_size = DQNConfig.batch_size
        self.learning_starts = DQNConfig.learning_start
        self.train_freq = DQNConfig.train_freq
        self.action_repeat = DQNConfig.action_repeat
        self.epsilon_init = DQNConfig.epsilon_init
        self.epsilon_decay = DQNConfig.epsilon_decay
        self.epsilon_min = DQNConfig.epsilon_min
        self.network_sync_rate = DQNConfig.target_update_interval
        self.learning_rate_a = DQNConfig.learning_rate
        self.discount_factor_g = DQNConfig.discount_factor
        self.fcl_nodes = DQNConfig.hidden_size
        self.max_grad_norm = DQNConfig.max_grad_norm

        self.enable_double_dqn = (
            enable_double_dqn if enable_double_dqn is not None else DQNConfig.enable_double_dqn
        )
        self.enable_dueling_dqn = (
            enable_dueling_dqn if enable_dueling_dqn is not None else DQNConfig.enable_dueling_dqn
        )

        # SmoothL1Loss (Huber loss) prevents exploding gradients from large +/-20 goal rewards
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"run_{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"model_{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"graph_{self.hyperparameter_set}.png")

    def optimize(self, mini_batch, policy_dqn: DQN, target_dqn: DQN) -> float:
        """Perform a single Double/Dueling DQN gradient step on a sampled minibatch."""
        state, action, next_state, reward, terminated, truncated = zip(*mini_batch)

        state = torch.stack(state)
        action = torch.stack(action)
        next_state = torch.stack(next_state)
        reward = torch.stack(reward)
        terminated = torch.tensor(terminated, dtype=torch.float32, device=device)

        with torch.no_grad():
            if self.enable_double_dqn:
                next_actions = policy_dqn(next_state).argmax(dim=1, keepdim=True)
                next_q = target_dqn(next_state).gather(dim=1, index=next_actions).squeeze(1)
            else:
                next_q = target_dqn(next_state).max(dim=1)[0]

            # Bootstrap on time-limit truncation; only zero out future value on true goal termination
            target_q = reward + (1.0 - terminated) * self.discount_factor_g * next_q

        current_q = policy_dqn(state).gather(dim=1, index=action.unsqueeze(dim=1)).squeeze(1)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    def _step_with_action_repeat(self, env: gym.Env, action_idx: int):
        """Repeat chosen action for `action_repeat` frames, releasing jump after frame 0 so
        edge-triggered jumping in Car._apply_jump can fire again on subsequent decisions."""
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        next_state = None

        # Clear jump bit on repeat frames > 0 so _prev_jump_action resets between decisions
        jump_bit = (action_idx % 4) // 2
        held_action_idx = (action_idx - 2) if jump_bit == 1 else action_idx

        for sub in range(max(1, self.action_repeat)):
            act = action_idx if sub == 0 else held_action_idx
            next_state, reward, terminated, truncated, info = env.step(act)
            total_reward += float(reward)
            if terminated or truncated or not env.unwrapped.isopen:
                break

        return next_state, total_reward, terminated, truncated, info

    def save_graph(self, rewards_per_episode: list[float], epsilon_history: list[float]):
        """Plot and save 100-episode rolling mean reward and epsilon decay curves."""
        import matplotlib.pyplot as plt

        mean_rewards = np.zeros(len(rewards_per_episode))
        for i in range(len(rewards_per_episode)):
            mean_rewards[i] = np.mean(rewards_per_episode[max(0, i - 99) : i + 1])

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.xlabel("Episodes")
        plt.ylabel("Mean Reward (Last 100 Episodes)")
        plt.plot(mean_rewards, label="Mean Reward")
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.xlabel("Episodes")
        plt.ylabel("Epsilon Decay")
        plt.plot(epsilon_history, label="Epsilon", color="orange")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.GRAPH_FILE)
        plt.close()

    def run(
        self,
        is_training: bool = True,
        render: bool = False,
        max_episode_steps: int | None = None,
        num_episodes: int | None = None,
        model_path: str | Path | None = None,
    ) -> dict[str, float]:
        """Run training (`is_training=True`) or validation/evaluation (`is_training=False`)."""
        make_kwargs = {}
        if max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = max_episode_steps

        env = gym.make(
            self.env_id,
            render_mode="human" if render else None,
            bot_type=self.bot_type,
            discrete_action=True,
            **make_kwargs,
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []
        goals_per_episode = []

        policy_dqn = DQN(
            num_states,
            num_actions,
            self.fcl_nodes,
            enable_dueling_dqn=self.enable_dueling_dqn,
        ).to(device)

        ckpt_file = str(model_path) if model_path is not None else self.MODEL_FILE

        if is_training:
            memory = ReplayMemory(maxlen=self.replay_memory_size)
            epsilon = self.epsilon_init

            target_dqn = DQN(
                num_states,
                num_actions,
                self.fcl_nodes,
                enable_dueling_dqn=self.enable_dueling_dqn,
            ).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            target_dqn.eval()

            step_count = 0
            epsilon_history = []
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            best_mean_reward = float("-inf")
            last_graph_update_time = datetime.now()
            print(
                f"[DuelingDDQN Train] Device={device} | Dueling={self.enable_dueling_dqn} | "
                f"Double={self.enable_double_dqn} | Bot={self.bot_type} | Model={ckpt_file}"
            )
        else:
            if os.path.exists(ckpt_file):
                state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)
                policy_dqn.load_state_dict(state_dict)
                print(f"[DuelingDDQN Eval] Loaded trained model weights from: {ckpt_file}")
            else:
                print(f"[DuelingDDQN Eval] Warning: Checkpoint '{ckpt_file}' not found; evaluating initial weights.")
            policy_dqn.eval()
            step_count = 0
            epsilon = 0.0

        episode_iter = range(num_episodes) if num_episodes is not None else itertools.count()

        try:
            for episode in episode_iter:
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device)

                if not env.unwrapped.isopen:
                    break

                terminated = False
                truncated = False
                episode_reward = 0.0
                scored_goal = 0

                while not (terminated or truncated):
                    if is_training and random.random() < epsilon:
                        action_idx = env.action_space.sample()
                        action = torch.tensor(action_idx, dtype=torch.int64, device=device)
                    else:
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(0)).squeeze(0).argmax()
                            action_idx = int(action.item())

                    next_state, reward, terminated, truncated, info = self._step_with_action_repeat(env, action_idx)

                    episode_reward += reward
                    if info.get("is_success", False) or (terminated and reward > 0.0):
                        scored_goal = 1

                    new_state = torch.tensor(next_state, dtype=torch.float32, device=device)
                    reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)

                    if is_training:
                        memory.append((state, action, new_state, reward_tensor, terminated, truncated))
                        step_count += 1

                        # Perform gradient update every `train_freq` decision steps inside the episode
                        if (
                            len(memory) >= max(self.learning_starts, self.mini_batch_size)
                            and step_count % self.train_freq == 0
                        ):
                            mini_batch = memory.sample(self.mini_batch_size)
                            self.optimize(mini_batch, policy_dqn, target_dqn)

                        # Sync target network every `network_sync_rate` decision steps
                        if step_count % self.network_sync_rate == 0:
                            target_dqn.load_state_dict(policy_dqn.state_dict())

                    state = new_state

                    if not env.unwrapped.isopen:
                        break

                rewards_per_episode.append(episode_reward)
                goals_per_episode.append(scored_goal)
                mean_reward_100 = float(np.mean(rewards_per_episode[-100:]))
                goal_rate_100 = float(np.mean(goals_per_episode[-100:])) * 100.0

                print(
                    f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
                    f"Mean(100) = {mean_reward_100:7.2f} | GoalRate(100) = {goal_rate_100:5.1f}% | "
                    f"Eps = {epsilon:.3f} | Steps = {step_count}"
                    if is_training
                    else f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
                    f"Goal = {bool(scored_goal)} | Terminated = {terminated} | Truncated = {truncated}"
                )

                if is_training:
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Save best model based on 100-episode rolling mean reward
                    should_save_best = (len(rewards_per_episode) >= 20 and mean_reward_100 > best_mean_reward) or (
                        num_episodes is not None and episode == num_episodes - 1 and best_mean_reward == float("-inf")
                    )
                    if should_save_best:
                        best_mean_reward = mean_reward_100
                        log_message = (
                            f"{datetime.now().strftime(DATE_FORMAT)} - New best 100-ep mean reward: "
                            f"{mean_reward_100:.2f} (Goal rate: {goal_rate_100:.1f}%, Ep reward: {episode_reward:.2f}) "
                            f"at episode {episode} (step {step_count}). Saving model."
                        )
                        print(log_message)
                        with open(self.LOG_FILE, "a") as log_file:
                            log_file.write(log_message + "\n")

                        torch.save(policy_dqn.state_dict(), ckpt_file)

                    current_time = datetime.now()
                    if current_time - last_graph_update_time > timedelta(seconds=10):
                        self.save_graph(rewards_per_episode, epsilon_history)
                        last_graph_update_time = current_time

                    if not env.unwrapped.isopen:
                        print("Exiting environment.")
                        break
        finally:
            env.close()

        return {
            "episodes": len(rewards_per_episode),
            "mean_reward": float(np.mean(rewards_per_episode)) if rewards_per_episode else 0.0,
            "goal_rate": float(np.mean(goals_per_episode)) * 100.0 if goals_per_episode else 0.0,
        }


# Backwards-compatible alias
Agent = DuelingDDQNAgent


def train_dueling_ddqn(
    num_episodes: int | None = None,
    max_episode_steps: int | None = None,
    enable_dueling_dqn: bool = True,
    enable_double_dqn: bool = True,
    hyperparameter_set: str = DQNConfig.hyperparameter_set,
) -> dict[str, float]:
    """Train the Dueling Double DQN agent."""
    agent = DuelingDDQNAgent(
        hyperparameter_set=hyperparameter_set,
        enable_dueling_dqn=enable_dueling_dqn,
        enable_double_dqn=enable_double_dqn,
    )
    return agent.run(
        is_training=True,
        render=False,
        max_episode_steps=max_episode_steps,
        num_episodes=num_episodes,
    )


def evaluate_dueling_ddqn(
    num_episodes: int | None = DQNConfig.eval_episodes,
    render: bool = False,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
    enable_dueling_dqn: bool = True,
    enable_double_dqn: bool = True,
    hyperparameter_set: str = DQNConfig.hyperparameter_set,
) -> dict[str, float]:
    """Evaluate/validate a trained Dueling Double DQN agent."""
    agent = DuelingDDQNAgent(
        hyperparameter_set=hyperparameter_set,
        enable_dueling_dqn=enable_dueling_dqn,
        enable_double_dqn=enable_double_dqn,
    )
    episodes = None if (render and num_episodes is None) else num_episodes
    return agent.run(
        is_training=False,
        render=render,
        max_episode_steps=max_episode_steps,
        num_episodes=episodes,
        model_path=model_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dueling Double DQN Agent for Rocket League")
    parser.add_argument(
        "--hyperparameter_set",
        type=str,
        default=DQNConfig.hyperparameter_set,
        help="Specify the hyperparameter set / run name",
    )
    parser.add_argument("--train", action="store_true", help="Run the agent in training mode")
    parser.add_argument("--eval", action="store_true", help="Run the agent in evaluation/validation mode")
    parser.add_argument("--render", action="store_true", help="Render the environment in a Pygame window")
    parser.add_argument("--episodes", type=int, default=None, help="Maximum number of episodes to run")
    parser.add_argument("--model_path", type=str, default=None, help="Optional path to model .pt checkpoint")
    args = parser.parse_args()

    if args.train:
        train_dueling_ddqn(
            num_episodes=args.episodes,
            hyperparameter_set=args.hyperparameter_set,
        )
    else:
        eval_eps = args.episodes if args.episodes is not None else (None if args.render else DQNConfig.eval_episodes)
        evaluate_dueling_ddqn(
            num_episodes=eval_eps,
            render=args.render,
            model_path=args.model_path,
            hyperparameter_set=args.hyperparameter_set,
        )
