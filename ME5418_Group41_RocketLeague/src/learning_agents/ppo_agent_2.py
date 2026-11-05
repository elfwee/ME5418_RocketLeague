"""
PPO Learning Agent — Training & Validation/Evaluation Pipeline.

Mirrors `dueling_ddqn_agent.py`, built on the `PPO` algorithm class from
`src/neural_nets/ppo_torch_2.py` and the `RolloutBuffer` from `src/neural_nets/common.py`.
"""

import argparse
from datetime import datetime, timedelta
import itertools
import os
from pathlib import Path
import sys
from typing import Callable

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from torch.distributions import Categorical

# Ensure repository root is on sys.path when invoked directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.gym_config import GymConfig
from src.config.nn_config import PPOConfig
from src.gym_env.rocket_league_env import RocketLeagueEnv
from src.neural_nets.common import RolloutBuffer
from src.neural_nets.ppo_torch_2 import PPO

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
RUNS_DIR = ROOT_DIR / PPOConfig.runs_dir
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")  # Use a non-interactive backend for matplotlib

device = "cuda" if torch.cuda.is_available() else "cpu"


class PPOAgent:
    """PPO Learning Agent supporting both training and validation/evaluation.

    Key algorithmic components:
        1. PPO algorithm (`PPO` from `src.neural_nets.ppo_torch_2`):
           `ActorCritic` network (from `a2c_torch`), clipped surrogate objective, value MSE loss,
           entropy bonus and gradient norm clipping, run for `update_epochs` of shuffled minibatches.
        2. On-policy rollout storage (`RolloutBuffer` from `src.neural_nets.common`):
           Stores `rollout_steps` decisions and computes GAE advantages / returns.
        3. Action Repeat with Edge-Triggered Jump Release (`_step_with_action_repeat`):
           Repeats chosen discrete action for `action_repeat` frames while clearing the jump
           bit on repeat frames > 0 so `Car._apply_jump` edge-triggering resets cleanly.
        4. Truncation-Aware Value Bootstrapping (`run`):
           On time-limit `truncated` episodes, `gamma * V(s_timeout)` is folded into the final
           reward before storage, so only true goal `terminated` states bootstrap to 0.
    """

    def __init__(
        self,
        hyperparameter_set: str = PPOConfig.hyperparameter_set,
        yaml_path: str | Path | None = None,
    ):
        # Load from YAML if explicitly provided and exists; otherwise use PPOConfig defaults
        hyperparameters = {}
        if yaml_path is not None and Path(yaml_path).exists():
            import yaml

            with open(yaml_path, "r") as file:
                all_sets = yaml.safe_load(file) or {}
                hyperparameters = all_sets.get(hyperparameter_set, {})

        self.hyperparameter_set = hyperparameter_set
        self.env_id = GymConfig.gym_id
        self.bot = hyperparameters.get("bot", PPOConfig.bot_type == "heuristic_bot")
        self.fcl_nodes = hyperparameters.get("fcl_nodes", PPOConfig.hidden_size)
        self.trunk_type = hyperparameters.get("trunk_type", "independent")
        self.learning_rate = hyperparameters.get("learning_rate", PPOConfig.learning_rate)
        self.discount_factor_g = hyperparameters.get("discount_factor_g", PPOConfig.discount_factor)
        self.gae_lambda = hyperparameters.get("gae_lambda", PPOConfig.gae_lambda)
        self.clip_coef = hyperparameters.get("clip_coef", PPOConfig.clip_coef)
        self.value_loss_coef = hyperparameters.get("value_loss_coef", PPOConfig.value_loss_coef)
        self.entropy_coef = hyperparameters.get("entropy_coef", PPOConfig.entropy_coef)
        self.max_grad_norm = hyperparameters.get("max_grad_norm", PPOConfig.max_grad_norm)

        self.action_repeat = hyperparameters.get("action_repeat", PPOConfig.action_repeat)
        self.rollout_steps = hyperparameters.get("rollout_steps", PPOConfig.rollout_steps)
        self.update_epochs = hyperparameters.get("update_epochs", PPOConfig.update_epochs)
        self.mini_batch_size = hyperparameters.get("mini_batch_size", PPOConfig.minibatch_size)
        self.stop_on_reward = hyperparameters.get("stop_on_reward", PPOConfig.stop_on_reward)

        self.ppo = None  # persisted across training run() calls (e.g. curriculum stages)

        self.LOG_FILE = os.path.join(RUNS_DIR, f"run_{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"model_{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"graph_{self.hyperparameter_set}.png")

    def _build_ppo(self, num_states: int, num_actions: int) -> PPO:
        """Create a PPO algorithm instance from this agent's hyperparameters, on `device`."""
        ppo = PPO(
            obs_dim=num_states,
            n_actions=num_actions,
            hidden_size=self.fcl_nodes,
            learning_rate=self.learning_rate,
            discount_factor=self.discount_factor_g,
            gae_lambda=self.gae_lambda,
            clip_coef=self.clip_coef,
            value_loss_coef=self.value_loss_coef,
            entropy_coef=self.entropy_coef,
            max_norm=self.max_grad_norm,
            update_epochs=self.update_epochs,
            minibatch_size=self.mini_batch_size,
            trunk_type=self.trunk_type,
        )
        ppo.model.to(device)  # in-place, so the optimizer still tracks the same parameters
        return ppo

    @torch.no_grad()
    def _rollout_entropy(self, ppo: PPO, buffer: RolloutBuffer) -> float:
        """Mean policy entropy over the states in the rollout buffer (for logging/plotting)."""
        logits, _ = ppo.model(buffer.states.reshape(-1, buffer.obs_dim))
        return float(Categorical(logits=logits).entropy().mean().item())

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
        bot_type: str | None = None,
        stop_fn: Callable[[list[int]], bool] | None = None,
        env_kwargs: dict | None = None,
    ) -> dict[str, float]:
        """Run PPO training (`is_training=True`) or validation/evaluation (`is_training=False`).

        Args:
            bot_type: Overrides the configured opponent ("none" or "heuristic_bot") for this run.
            stop_fn: Training only. Called after every episode with the per-episode goal history
                     (1 = scored); returning True ends the run early.
            env_kwargs: Extra keyword arguments for the environment (e.g. `opponent_model_path` for self-play).
        """
        bot_type = bot_type if bot_type is not None else ("heuristic_bot" if self.bot else "none")
        make_kwargs = dict(env_kwargs or {})
        if max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = max_episode_steps

        env = gym.make(
            self.env_id,
            render_mode="human" if render else None,
            bot_type=bot_type,
            discrete_action=True,
            **make_kwargs,
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []
        goals_per_episode = []

        if is_training and self.ppo is not None:
            ppo = self.ppo  # continue training the same weights (e.g. next curriculum stage)
        else:
            ppo = self._build_ppo(num_states, num_actions)
            if is_training:
                self.ppo = ppo

        ckpt_file = str(model_path) if model_path is not None else self.MODEL_FILE

        if is_training:
            buffer = RolloutBuffer(self.rollout_steps, num_envs=1, obs_dim=num_states, device=device)
            ppo.model.train()

            step_count = 0
            entropy_history = []
            current_entropy = float(np.log(num_actions))
            best_mean_reward = float("-inf")
            last_graph_update_time = datetime.now()
            print(
                f"[PPO Train] Device={device} | Rollout={self.rollout_steps} | Epochs={self.update_epochs} | "
                f"Minibatch={self.mini_batch_size} | Bot={bot_type} | Model={ckpt_file}"
            )
        else:
            if os.path.exists(ckpt_file):
                state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)
                ppo.model.load_state_dict(state_dict)
                print(f"[PPO Eval] Loaded trained model weights from: {ckpt_file}")
            else:
                print(f"[PPO Eval] Warning: Checkpoint '{ckpt_file}' not found; evaluating initial weights.")
            ppo.model.eval()
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

                while (not (terminated or truncated)) and episode_reward < self.stop_on_reward:
                    if is_training:
                        action, log_prob, value = ppo.act(state.unsqueeze(0))
                        action_idx = int(action.item())
                    else:
                        action_idx = ppo.predict(state)

                    next_state, reward, terminated, truncated, info = self._step_with_action_repeat(env, action_idx)

                    episode_reward += reward
                    # Blue goal is scored when episode terminates with positive goal reward (or info["is_success"])
                    if info.get("is_success", False) or (terminated and reward > 0.0):
                        scored_goal = 1

                    new_state = torch.tensor(next_state, dtype=torch.float32, device=device)

                    if is_training:
                        stored_reward = reward
                        # Bootstrap on time-limit truncation; only zero out future value on true goal termination
                        if truncated and not terminated:
                            _, _, timeout_value = ppo.act(new_state.unsqueeze(0))
                            stored_reward += self.discount_factor_g * float(timeout_value.item())

                        buffer.add(
                            state.unsqueeze(0),
                            action,
                            log_prob,
                            torch.tensor([stored_reward]),
                            value,
                            torch.tensor([float(terminated or truncated)]),
                        )
                        step_count += 1

                        # Perform PPO multi-epoch minibatch update once the rollout buffer is full
                        if buffer.is_full():
                            _, _, last_value = ppo.act(new_state.unsqueeze(0))
                            buffer.compute_returns_and_advantages(last_value, self.discount_factor_g, self.gae_lambda)
                            ppo.update(buffer)
                            current_entropy = self._rollout_entropy(ppo, buffer)
                            buffer.reset()

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

                    # Save best model based on 100-episode rolling mean reward (after 20-episode warmup, or on final episode of short runs)
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

                        torch.save(ppo.model.state_dict(), ckpt_file)

                    current_time = datetime.now()
                    if current_time - last_graph_update_time > timedelta(seconds=10):
                        self.save_graph(rewards_per_episode, entropy_history)
                        last_graph_update_time = current_time

                    if not env.unwrapped.isopen:
                        print("Exiting environment.")
                        break

                    if stop_fn is not None and stop_fn(goals_per_episode):
                        print(f"[PPO Train] stop_fn triggered at episode {episode}.")
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
    # When rendering interactively without an explicit episode count, run until window is closed
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

    agent = PPOAgent(hyperparameter_set=args.hyperparameter_set)
    if args.train:
        agent.run(is_training=True, render=args.render, num_episodes=args.episodes, model_path=args.model_path)
    else:
        eval_eps = args.episodes if args.episodes is not None else (None if args.render else PPOConfig.eval_episodes)
        agent.run(is_training=False, render=args.render, num_episodes=eval_eps, model_path=args.model_path)
