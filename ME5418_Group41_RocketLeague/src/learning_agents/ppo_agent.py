"""
Proximal Policy Optimization (PPO) Learning Agent — Training & Validation/Evaluation Pipeline.

Adapted from CleanRL's `ppo.py` (https://github.com/vwxyzjn/cleanrl) for a controlled
comparison with `DuelingDDQNAgent` (`dueling_ddqn_agent.py`):
- Identical 150,949-parameter `LayerNorm + ReLU` two-stream network (`ActorCritic` in `src/neural_nets/ppo_torch.py`).
- Identical `action_repeat = 4` with edge-triggered jump release (`_step_with_action_repeat`).
- Identical core hyperparameters (`lr = 3e-4`, `gamma = 0.99`).
- On-policy rollout (`T = 4096`), reverse-time GAE (`lambda = 0.95`), batch-level advantage normalization,
  and `6` epochs of shuffled `256`-sample minibatch clipped surrogate updates.
- Identical logging, checkpointing (`runs/model_ppo.pt`), and 100-episode rolling metrics (`runs/graph_ppo.png`).
"""

import argparse
from datetime import datetime, timedelta
import itertools
import os
from pathlib import Path
import sys

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Ensure repository root is on sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.gym_config import GymConfig
from src.config.nn_config import PPOConfig
from src.gym_env.rocket_league_env import RocketLeagueEnv
from src.neural_nets.ppo_torch import ActorCritic

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
RUNS_DIR = ROOT_DIR / PPOConfig.runs_dir
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib

device = "cuda" if torch.cuda.is_available() else "cpu"


class PPOAgent:
    """Proximal Policy Optimization (PPO) Learning Agent supporting training and validation/evaluation.

    Key algorithmic components:
        1. Actor-Critic Architecture (`ActorCritic` from `src.neural_nets.ppo_torch`):
           Shared 2-layer `LayerNorm + ReLU` feature backbone (256 units) with 128-unit
           Actor (`pi(a|s)`) and Critic (`V(s)`) heads (150,949 parameters, matching Dueling DDQN).
        2. Action Repeat with Edge-Triggered Jump Release (`_step_with_action_repeat`):
           Repeats chosen discrete action for `action_repeat` frames while clearing the jump
           bit on repeat frames > 0 so `Car._apply_jump` edge-triggering resets cleanly.
        3. Truncation-Aware Generalized Advantage Estimation (`_compute_gae`):
           Terminal goal states (`terminated=True`) bootstrap to 0.0, whereas timeout states
           (`truncated=True`) bootstrap from the critic's value estimate `V(s_timeout)` before reset.
        4. Clipped Surrogate Objective (`optimize`):
           Normalizes advantages across the entire 4,096-step rollout batch and performs `update_epochs` (6)
           of shuffled 256-sample minibatch updates with ratio clipping `clip_coef`, value MSE loss,
           entropy bonus, and global gradient norm clipping.
    """

    def __init__(self, hyperparameter_set: str = PPOConfig.hyperparameter_set):
        self.hyperparameter_set = hyperparameter_set
        self.env_id = GymConfig.gym_id
        self.bot_type = PPOConfig.bot_type
        self.fcl_nodes = PPOConfig.hidden_size
        self.learning_rate = PPOConfig.learning_rate
        self.discount_factor_g = PPOConfig.discount_factor
        self.gae_lambda = PPOConfig.gae_lambda
        self.clip_coef = PPOConfig.clip_coef
        self.value_loss_coef = PPOConfig.value_loss_coef
        self.entropy_coef = PPOConfig.entropy_coef
        self.max_grad_norm = PPOConfig.max_grad_norm

        self.action_repeat = PPOConfig.action_repeat
        self.rollout_steps = PPOConfig.rollout_steps
        self.update_epochs = PPOConfig.update_epochs
        self.mini_batch_size = PPOConfig.minibatch_size

        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"run_{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"model_{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"graph_{self.hyperparameter_set}.png")

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        next_values: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE) with truncation-aware bootstrapping.

        Args:
            rewards: Shape (T,), step rewards.
            values: Shape (T,), critic estimates V(s_t).
            next_values: Shape (T,), bootstrap values V(s_{t+1}) (already 0.0 where `terminated=True`,
                         and `V(s_timeout)` where `truncated=True`).
            dones: Shape (T,), 1.0 where episode ended (`terminated or truncated`), resetting the GAE trace.
        """
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(T)):
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.discount_factor_g * next_values[t] - values[t]
            last_gae = delta + self.discount_factor_g * self.gae_lambda * not_done * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def optimize(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        policy_net: ActorCritic,
    ) -> tuple[float, float]:
        """Run `update_epochs` of shuffled minibatch PPO clipped surrogate updates."""
        batch_size = states.size(0)
        total_loss_sum = 0.0
        entropy_sum = 0.0
        num_mb = 0

        # Normalize advantages across the entire 4,096-sample rollout batch before minibatch sampling
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, self.mini_batch_size):
                idx = perm[start : start + self.mini_batch_size]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_adv = advantages[idx]

                _, new_log_probs, entropy, new_values = policy_net.get_action_and_value(mb_states, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.mse_loss(new_values, mb_returns)
                entropy_loss = entropy.mean()

                loss = pg_loss + self.value_loss_coef * v_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                total_loss_sum += float(loss.item())
                entropy_sum += float(entropy_loss.item())
                num_mb += 1

        denom = max(num_mb, 1)
        return total_loss_sum / denom, entropy_sum / denom

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

    def save_graph(self, rewards_per_episode: list[float], entropy_history: list[float]):
        """Plot and save 100-episode rolling mean reward and policy entropy curves."""
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
        plt.ylabel("Policy Entropy")
        plt.plot(entropy_history, label="Entropy", color="orange")
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
        """Run PPO training (`is_training=True`) or validation/evaluation (`is_training=False`)."""
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

        policy_net = ActorCritic(num_states, num_actions, self.fcl_nodes).to(device)
        ckpt_file = str(model_path) if model_path is not None else self.MODEL_FILE

        if is_training:
            self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.learning_rate, eps=1e-5)
            step_count = 0
            entropy_history = []
            current_entropy = float(np.log(num_actions))
            best_mean_reward = float("-inf")
            last_graph_update_time = datetime.now()

            # On-policy rollout storage lists
            buf_states, buf_actions, buf_log_probs = [], [], []
            buf_rewards, buf_values, buf_next_values, buf_dones = [], [], [], []

            print(
                f"[PPO Train] Device={device} | Rollout={self.rollout_steps} | Epochs={self.update_epochs} | "
                f"Minibatch={self.mini_batch_size} | Bot={self.bot_type} | Model={ckpt_file}"
            )
        else:
            if os.path.exists(ckpt_file):
                state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)
                policy_net.load_state_dict(state_dict)
                print(f"[PPO Eval] Loaded trained model weights from: {ckpt_file}")
            else:
                print(f"[PPO Eval] Warning: Checkpoint '{ckpt_file}' not found; evaluating initial weights.")
            policy_net.eval()
            step_count = 0
            current_entropy = 0.0

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
                    with torch.no_grad():
                        if is_training:
                            action, log_prob, _, value = policy_net.get_action_and_value(state.unsqueeze(0))
                            action_idx = int(action.item())
                        else:
                            logits, _ = policy_net(state.unsqueeze(0))
                            action = logits.squeeze(0).argmax()
                            action_idx = int(action.item())

                    next_state, reward, terminated, truncated, info = self._step_with_action_repeat(env, action_idx)
                    done = terminated or truncated

                    episode_reward += reward
                    if info.get("is_success", False) or (terminated and reward > 0.0):
                        scored_goal = 1

                    new_state = torch.tensor(next_state, dtype=torch.float32, device=device)

                    if is_training:
                        with torch.no_grad():
                            # Truncation-aware bootstrapping:
                            # - Terminal goal state (`terminated=True`) -> bootstrap 0.0
                            # - Non-terminal or timeout state (`terminated=False`) -> bootstrap V(new_state)
                            if terminated:
                                next_val = torch.tensor(0.0, dtype=torch.float32, device=device)
                            else:
                                _, next_val = policy_net(new_state.unsqueeze(0))
                                next_val = next_val.squeeze(0)

                        buf_states.append(state)
                        buf_actions.append(action.squeeze(0))
                        buf_log_probs.append(log_prob.squeeze(0))
                        buf_values.append(value.squeeze(0))
                        buf_next_values.append(next_val)
                        buf_rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
                        buf_dones.append(torch.tensor(float(done), dtype=torch.float32, device=device))

                        step_count += 1

                        # Perform PPO multi-epoch minibatch update once rollout horizon is reached
                        if len(buf_states) >= self.rollout_steps:
                            states_t = torch.stack(buf_states)
                            actions_t = torch.stack(buf_actions)
                            old_log_probs_t = torch.stack(buf_log_probs)
                            values_t = torch.stack(buf_values)
                            next_values_t = torch.stack(buf_next_values)
                            rewards_t = torch.stack(buf_rewards)
                            dones_t = torch.stack(buf_dones)

                            advantages_t, returns_t = self._compute_gae(
                                rewards_t, values_t, next_values_t, dones_t
                            )
                            _, current_entropy = self.optimize(
                                states_t,
                                actions_t,
                                old_log_probs_t,
                                returns_t,
                                advantages_t,
                                policy_net,
                            )

                            buf_states.clear()
                            buf_actions.clear()
                            buf_log_probs.clear()
                            buf_values.clear()
                            buf_next_values.clear()
                            buf_rewards.clear()
                            buf_dones.clear()

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
                    f"Ent = {current_entropy:.3f} | Steps = {step_count}"
                    if is_training
                    else f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
                    f"Goal = {bool(scored_goal)} | Terminated = {terminated} | Truncated = {truncated}"
                )

                if is_training:
                    entropy_history.append(current_entropy)

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

                        torch.save(policy_net.state_dict(), ckpt_file)

                    current_time = datetime.now()
                    if current_time - last_graph_update_time > timedelta(seconds=10):
                        self.save_graph(rewards_per_episode, entropy_history)
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


def train_ppo(
    num_episodes: int | None = None,
    max_episode_steps: int | None = None,
    hyperparameter_set: str = PPOConfig.hyperparameter_set,
) -> dict[str, float]:
    """Train the PPO agent."""
    agent = PPOAgent(hyperparameter_set=hyperparameter_set)
    return agent.run(
        is_training=True,
        render=False,
        max_episode_steps=max_episode_steps,
        num_episodes=num_episodes,
    )


def evaluate_ppo(
    num_episodes: int | None = PPOConfig.eval_episodes,
    render: bool = False,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
    hyperparameter_set: str = PPOConfig.hyperparameter_set,
) -> dict[str, float]:
    """Evaluate/validate a trained PPO agent."""
    agent = PPOAgent(hyperparameter_set=hyperparameter_set)
    episodes = None if (render and num_episodes is None) else num_episodes
    return agent.run(
        is_training=False,
        render=render,
        max_episode_steps=max_episode_steps,
        num_episodes=episodes,
        model_path=model_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Agent for Rocket League")
    parser.add_argument(
        "--hyperparameter_set",
        type=str,
        default=PPOConfig.hyperparameter_set,
        help="Specify the hyperparameter set / run name",
    )
    parser.add_argument("--train", action="store_true", help="Run the agent in training mode")
    parser.add_argument("--eval", action="store_true", help="Run the agent in evaluation/validation mode")
    parser.add_argument("--render", action="store_true", help="Render the environment in a Pygame window")
    parser.add_argument("--episodes", type=int, default=None, help="Maximum number of episodes to run")
    parser.add_argument("--model_path", type=str, default=None, help="Optional path to model .pt checkpoint")
    args = parser.parse_args()

    if args.train:
        train_ppo(
            num_episodes=args.episodes,
            hyperparameter_set=args.hyperparameter_set,
        )
    else:
        eval_eps = args.episodes if args.episodes is not None else (None if args.render else PPOConfig.eval_episodes)
        evaluate_ppo(
            num_episodes=eval_eps,
            render=args.render,
            model_path=args.model_path,
            hyperparameter_set=args.hyperparameter_set,
        )
