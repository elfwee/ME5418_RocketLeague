"""A2C agent that trains on, logs and plays RocketLeagueEnv."""

import csv
import os
import time

import gymnasium as gym
import numpy as np
import torch

from src.config.gym_config import GymConfig
from src.gym_env.rocket_league_env import decode_action  # also registers the gym env
from src.neural_nets.a2c_torch import A2C

CSV_FIELDS = ["episode", "total_steps", "reward", "length", "outcome", "mean_loss"]
OUTCOME_CODE = {"orange": -1, "timeout": 0, "blue": 1}


class RocketLeagueAgent:
    """Trains an A2C policy on RocketLeagueEnv with n-step rollouts.

    Outputs go next to ``model_filepath``: ``metrics.csv``, ``tensorboard/``,
    ``gifs/`` and ``plots/``.

    Args:
        model_filepath: Where the model checkpoint is saved.
        visualise_n_episodes: Save a checkpoint and a GIF of a greedy episode every N episodes.
        bot_type: Opponent, ``"none"`` or ``"heuristic_bot"``.
        rollout_length: Steps per A2C update.
        seed: Seed for the first env reset.
        a2c_kwargs: Extra hyperparameters passed to :class:`A2C`.
    """

    def __init__(
        self,
        model_filepath,
        visualise_n_episodes=1_000,
        bot_type="none",
        rollout_length=16,
        seed=None,
        **a2c_kwargs,
    ) -> None:
        self.model_filepath = model_filepath
        self.run_dir = os.path.dirname(os.path.abspath(model_filepath))
        self.csv_path = os.path.join(self.run_dir, "metrics.csv")
        self.visualise_n_episodes = visualise_n_episodes
        self.bot_type = bot_type
        self.rollout_length = rollout_length

        self.gym = gym.make(GymConfig.gym_id, bot_type=bot_type)
        self.model = A2C(
            obs_dim=self.gym.observation_space.shape[0], n_actions=int(self.gym.action_space.n), **a2c_kwargs
        )
        self.writer = None
        self.episode = 0
        self.total_steps = 0
        if os.path.exists(self.csv_path):  # continue numbering from an earlier run in this dir
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                self.episode = int(rows[-1]["episode"])
                self.total_steps = int(rows[-1]["total_steps"])
        self._first_logged_episode = self.episode + 1
        self.reset(seed)

    def reset(self, seed=None):
        self.obs, _ = self.gym.reset(seed=seed)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_losses = []

    # ------------------------------------------------------------------ logging
    def tensorboard_logging(self, stats):
        if self.writer is None:
            if stats["episode"] != self._first_logged_episode:
                return  # tensorboard unavailable, already warned
            # Imported lazily: tensorboard pulls in TensorFlow if installed, and loading it
            # before torch's compiler stack (loaded when the optimizer is built) crashes with
            # a duplicate-LLVM-option error.
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                print("tensorboard not installed; skipping TensorBoard logging (pip install tensorboard)")
                return
            self.writer = SummaryWriter(os.path.join(self.run_dir, "tensorboard"))
        step = stats["episode"]
        self.writer.add_scalar("episode/reward", stats["reward"], step)
        self.writer.add_scalar("episode/length", stats["length"], step)
        self.writer.add_scalar("episode/outcome", OUTCOME_CODE[stats["outcome"]], step)
        if stats["mean_loss"] is not None:
            self.writer.add_scalar("train/loss", stats["mean_loss"], step)

    def csv_logging(self, stats):
        os.makedirs(self.run_dir, exist_ok=True)
        write_header = not os.path.exists(self.csv_path)
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(stats)

    def save_plots(self, window=100):
        """Plot KPIs from metrics.csv into plots/kpi.png."""
        if not os.path.exists(self.csv_path):
            return
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with open(self.csv_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return
        episodes = np.array([int(r["episode"]) for r in rows])
        rewards = np.array([float(r["reward"]) for r in rows])
        lengths = np.array([int(r["length"]) for r in rows])
        goals = np.array([r["outcome"] == "blue" for r in rows], dtype=float)
        losses = np.array([float(r["mean_loss"]) if r["mean_loss"] else np.nan for r in rows])

        def moving_average(x):
            w = min(window, len(x))
            return episodes[w - 1 :], np.convolve(x, np.ones(w) / w, mode="valid")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax = axes[0, 0]
        ax.plot(episodes, rewards, alpha=0.3, label="episode")
        ax.plot(*moving_average(rewards), label=f"{window}-ep mean")
        ax.set_title("Episode reward")
        ax.legend()
        ax = axes[0, 1]
        ax.plot(episodes, lengths, alpha=0.3)
        ax.plot(*moving_average(lengths))
        ax.set_title("Episode length (steps)")
        ax = axes[1, 0]
        ax.plot(*moving_average(goals))
        ax.set_ylim(0, 1)
        ax.set_title(f"Blue goal rate ({window}-ep mean)")
        ax = axes[1, 1]
        ax.plot(episodes, losses)
        ax.set_title("Mean A2C loss")
        for ax in axes.flat:
            ax.set_xlabel("episode")
            ax.grid(alpha=0.3)
        fig.tight_layout()

        plot_dir = os.path.join(self.run_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(os.path.join(plot_dir, "kpi.png"), dpi=120)
        plt.close(fig)

    # ----------------------------------------------------------------- training
    def _end_episode(self, outcome):
        self.episode += 1
        stats = {
            "episode": self.episode,
            "total_steps": self.total_steps,
            "reward": self.episode_reward,
            "length": self.episode_length,
            "outcome": outcome,
            "mean_loss": float(np.mean(self.episode_losses)) if self.episode_losses else None,
        }
        self.tensorboard_logging(stats)
        self.csv_logging(stats)
        print(
            f"episode {self.episode:6d} | steps {self.episode_length:5d} | "
            f"reward {self.episode_reward:8.2f} | {outcome}"
        )
        if self.episode % self.visualise_n_episodes == 0:
            self.model.save(self.model_filepath)
            self.save_visual_episode(self.episode)
        self.reset()

    def learn(self, number_of_episodes: int = 100_000, pretrain_model_filepath=None):
        if pretrain_model_filepath is not None:
            self.model.load(pretrain_model_filepath)
        self.reset()
        gamma = self.model.discount_factor
        target_episode = self.episode + number_of_episodes

        while self.episode < target_episode:
            states, actions, rewards, dones = [], [], [], []
            for _ in range(self.rollout_length):
                state = torch.as_tensor(self.obs, dtype=torch.float32)
                action, _ = self.model.act(state.unsqueeze(0))
                action = int(action.item())
                next_obs, reward, terminated, truncated, _ = self.gym.step(action)
                self.episode_reward += reward
                self.episode_length += 1
                self.total_steps += 1

                if truncated and not terminated:
                    # Time limit is not a true terminal state: bootstrap from V(s') here
                    _, next_value = self.model.act(torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0))
                    reward += gamma * float(next_value.item())

                states.append(state)
                actions.append(action)
                rewards.append(float(reward))
                dones.append(float(terminated or truncated))

                if terminated or truncated:
                    goal = self.gym.unwrapped.sim.goal_scored_this_step
                    self._end_episode(goal if goal is not None else "timeout")
                    if self.episode >= target_episode:
                        break
                else:
                    self.obs = next_obs

            # Bootstrap from V(s_last), then build n-step returns backwards
            _, discounted_return = self.model.act(torch.as_tensor(self.obs, dtype=torch.float32).unsqueeze(0))
            discounted_return = float(discounted_return.item())
            returns = []
            for t in reversed(range(len(rewards))):
                discounted_return = rewards[t] + gamma * (1.0 - dones[t]) * discounted_return
                returns.append(discounted_return)
            returns.reverse()
            loss = self.model._backprop(
                torch.stack(states),
                torch.as_tensor(actions, dtype=torch.long),
                torch.as_tensor(returns, dtype=torch.float32),
            )
            self.episode_losses.append(loss)

        self.model.save(self.model_filepath)
        self.save_plots()
        if self.writer is not None:
            self.writer.close()

    def save_visual_episode(self, episode):
        """Run one greedy episode offscreen and save it as a GIF."""
        gif_path = os.path.join(self.run_dir, "gifs", f"episode_{episode}.gif")
        env = gym.make(GymConfig.gym_id, bot_type=self.bot_type, gif_path=gif_path)
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            obs, _, terminated, truncated, _ = env.step(self.model.predict(obs))
        if not terminated:  # timed out without a goal: the env didn't save, so save what was buffered
            env.unwrapped.gif_saver.save(gif_path)
        env.close()

    # ------------------------------------------------------------------ playing
    def play(self, sim):
        """Greedy CarAction for the blue car in ``sim``."""
        obs = sim.get_state_norm(as_flat_array=True)
        return decode_action(self.model.predict(obs))


def train_agent(number_of_episodes=100_000, model_filepath=None, pretrain_model_filepath=None, **kwargs):
    if model_filepath is None:
        model_filepath = os.path.join("runs", f"a2c_{time.strftime('%Y%m%d_%H%M%S')}", "model.pt")
    agent = RocketLeagueAgent(model_filepath, **kwargs)
    agent.learn(number_of_episodes, pretrain_model_filepath)
    return agent


def play_agent(model_filepath, bot_type="none"):
    """Watch a trained model play in a window until it is closed."""
    env = gym.make(GymConfig.gym_id, render_mode="human", bot_type=bot_type)
    model = A2C(obs_dim=env.observation_space.shape[0], n_actions=int(env.action_space.n))
    model.load(model_filepath)
    obs, _ = env.reset()
    while env.unwrapped.isopen:
        obs, _, terminated, truncated, _ = env.step(model.predict(obs))
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
