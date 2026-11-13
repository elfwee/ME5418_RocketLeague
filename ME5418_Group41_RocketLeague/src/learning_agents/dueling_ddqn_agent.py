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
        self.epsilon_finetune = getattr(DQNConfig, "epsilon_finetune", 0.20)
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
        self.METRICS_FILE = os.path.join(RUNS_DIR, f"metrics_{self.hyperparameter_set}.csv")

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

    def _step_with_action_repeat(
        self,
        env: gym.Env,
        action_idx: int,
        action_orange_idx: int | None = None,
    ):
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

        if action_orange_idx is not None:
            jump_bit_orange = (action_orange_idx % 4) // 2
            held_action_orange_idx = (action_orange_idx - 2) if jump_bit_orange == 1 else action_orange_idx
        else:
            held_action_orange_idx = None

        repeat = max(1, self.action_repeat)
        unwrapped = env.unwrapped
        deferred = False
        for sub in range(repeat):
            deferred = sub < (repeat - 1)
            unwrapped._defer_obs = deferred
            act = action_idx if sub == 0 else held_action_idx
            if action_orange_idx is not None:
                act_orange = action_orange_idx if sub == 0 else held_action_orange_idx
                next_state, reward, terminated, truncated, info = env.step((act, act_orange))
            else:
                next_state, reward, terminated, truncated, info = env.step(act)
            total_reward += float(reward)
            if terminated or truncated or not unwrapped.isopen:
                break

        unwrapped._defer_obs = False
        if deferred:
            next_state = unwrapped._get_obs()
            if unwrapped.control_car == "both":
                info["obs_orange"] = unwrapped._get_obs("orange")

        return next_state, total_reward, terminated, truncated, info

    def _save_gif(self, frames: list[np.ndarray], path: str | Path, fps: int = 15):
        """Compile and save a list of RGB frames as an optimized animated GIF."""
        from PIL import Image

        if not frames:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Downscale by 50% (700x360) for smooth playback and lightweight GIF file size
        pil_images = [
            Image.fromarray(f).resize((700, 360), Image.Resampling.BILINEAR)
            for f in frames
        ]
        pil_images[0].save(
            str(path),
            save_all=True,
            append_images=pil_images[1:],
            duration=int(1000 / fps),
            loop=0,
            optimize=True,
        )
        print(f"[DQN Eval] Saved evaluation GIF ({len(pil_images)} frames, {fps} FPS) to: {path}")

    @staticmethod
    def _rolling_mean(values: list[float] | np.ndarray, window: int) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float64)
        n = len(arr)
        if n == 0:
            return arr
        csum = np.empty(n + 1, dtype=np.float64)
        csum[0] = 0.0
        np.cumsum(arr, out=csum[1:])
        idx = np.arange(n)
        starts = np.maximum(0, idx - (window - 1))
        counts = (idx - starts + 1).astype(np.float64)
        return (csum[idx + 1] - csum[starts]) / counts

    def save_graph(
        self,
        rewards_per_episode: list[float],
        epsilon_history: list[float],
        steps_per_episode: list[int] | None = None,
        goals_per_episode: list[int] | None = None,
        loss_history: list[float] | None = None,
        dpi: int = 150,
    ):
        """Plot and save a comprehensive 2x2 training dashboard.

        Subplots:
            1. Sample Efficiency: 100-episode rolling mean reward & episodic return vs. Environment Steps.
            2. Task Competency: 100-episode rolling goal conversion rate (%) vs. Environment Steps.
            3. Exploration Schedule: Epsilon decay (ε) vs. Environment Steps.
            4. Bellman Optimization Health: TD / Huber Loss (with 50-ep smoothing) vs. Environment Steps.
        """
        import matplotlib.pyplot as plt

        n = len(rewards_per_episode)
        if n == 0:
            return

        x = np.array(
            steps_per_episode if steps_per_episode is not None and len(steps_per_episode) == n else list(range(1, n + 1))
        )
        x_label = "Environment Steps" if steps_per_episode is not None else "Episodes"

        # 100-episode rolling mean reward (vectorized O(N))
        mean_rewards = self._rolling_mean(rewards_per_episode, 100)

        # 100-episode rolling goal rate (%) (vectorized O(N))
        has_goals = goals_per_episode is not None and len(goals_per_episode) == n
        goal_rates = self._rolling_mean(goals_per_episode, 100) * 100.0 if has_goals else np.zeros(n)

        fig, axs = plt.subplots(2, 2, figsize=(14, 9))

        # 1. Sample Efficiency
        axs[0, 0].plot(x, rewards_per_episode, color="#1f77b4", alpha=0.18, linewidth=0.8, label="Episodic Return")
        axs[0, 0].plot(x, mean_rewards, color="#1f77b4", linewidth=2.2, label="100-Ep Rolling Return")
        axs[0, 0].axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        axs[0, 0].set_title("Sample Efficiency (Return vs. Steps)", fontsize=11, fontweight="bold")
        axs[0, 0].set_xlabel(x_label)
        axs[0, 0].set_ylabel("Episodic Return")
        axs[0, 0].grid(True, alpha=0.3, linestyle="--")
        axs[0, 0].legend(loc="upper left", framealpha=0.8)

        # 2. Objective Goal Conversion Rate (%)
        if has_goals:
            axs[0, 1].plot(x, goal_rates, color="#2ca02c", linewidth=2.2, label="100-Ep Goal Rate (%)")
            axs[0, 1].axhline(50.0, color="#9467bd", linestyle=":", linewidth=1.0, alpha=0.7, label="50% Win Parity")
            axs[0, 1].set_ylim([-2, 105])
        axs[0, 1].set_title("Objective Competency (Goal Rate %)", fontsize=11, fontweight="bold")
        axs[0, 1].set_xlabel(x_label)
        axs[0, 1].set_ylabel("Goal Rate (%)")
        axs[0, 1].grid(True, alpha=0.3, linestyle="--")
        axs[0, 1].legend(loc="upper left", framealpha=0.8)

        # 3. Epsilon Decay
        eps_arr = np.array(epsilon_history) if epsilon_history is not None else np.array([])
        if len(eps_arr) == n:
            axs[1, 0].plot(x, eps_arr, color="#ff7f0e", linewidth=2.0, label="Epsilon (ε)")
        elif len(eps_arr) > 0:
            axs[1, 0].plot(eps_arr, color="#ff7f0e", linewidth=2.0, label="Epsilon (ε)")
        axs[1, 0].set_title("Exploration Schedule (Epsilon Decay)", fontsize=11, fontweight="bold")
        axs[1, 0].set_xlabel(x_label)
        axs[1, 0].set_ylabel("Epsilon (ε)")
        axs[1, 0].set_ylim([-0.02, 1.05])
        axs[1, 0].grid(True, alpha=0.3, linestyle="--")
        axs[1, 0].legend(loc="upper right", framealpha=0.8)

        # 4. TD Huber Loss
        if loss_history is not None and len(loss_history) == n:
            losses = np.array(loss_history)
            axs[1, 1].plot(x, losses, color="#d62728", alpha=0.25, linewidth=0.8, label="Raw TD Loss")
            smooth_loss = self._rolling_mean(losses, 50)
            axs[1, 1].plot(x, smooth_loss, color="#d62728", linewidth=2.0, label="50-Ep Smoothed Loss")
            axs[1, 1].legend(loc="upper right", framealpha=0.8)
        axs[1, 1].set_title("Bellman Optimization Health (TD Loss)", fontsize=11, fontweight="bold")
        axs[1, 1].set_xlabel(x_label)
        axs[1, 1].set_ylabel("Huber Loss")
        axs[1, 1].grid(True, alpha=0.3, linestyle="--")

        agent_name = "Dueling Double DQN" if (self.enable_dueling_dqn and self.enable_double_dqn) else "DQN"
        fig.suptitle(f"{agent_name} Training Dashboard ({self.hyperparameter_set})", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.GRAPH_FILE, dpi=dpi)
        plt.close()

    def run(
        self,
        is_training: bool = True,
        render: bool = False,
        max_episode_steps: int | None = None,
        num_episodes: int | None = None,
        model_path: str | Path | None = None,
        save_gif: bool = False,
        gif_path: str | Path | None = None,
        control_car: str = "blue",
    ) -> dict[str, float]:
        """Run training (`is_training=True`) or validation/evaluation (`is_training=False`)."""
        if control_car in ("bot_vs_bot", "bots", "both"):
            control_car = "both"

        make_kwargs = {}
        if max_episode_steps is not None:
            make_kwargs["max_episode_steps"] = max_episode_steps

        if render:
            render_mode = "human"
        elif save_gif:
            render_mode = "rgb_array"
        else:
            render_mode = None

        env = gym.make(
            self.env_id,
            render_mode=render_mode,
            bot_type=self.bot_type,
            discrete_action=True,
            control_car="blue" if is_training else control_car,
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
            loaded_ckpt = False
            if model_path is not None:
                if os.path.exists(ckpt_file):
                    state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)
                    policy_dqn.load_state_dict(state_dict)
                    loaded_ckpt = True
                    print(f"[DuelingDDQN Train] Loaded model weights from: {ckpt_file}")
                else:
                    print(f"[DuelingDDQN Train] Warning: Checkpoint '{ckpt_file}' not found; training from scratch.")
            else:
                print("[DuelingDDQN Train] Training from scratch; no model checkpoint specified.")

            memory = ReplayMemory(maxlen=self.replay_memory_size)
            epsilon = self.epsilon_finetune if loaded_ckpt else self.epsilon_init

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
            steps_history = []
            loss_history = []
            current_loss = 0.0
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

            best_mean_reward = float("-inf")
            last_graph_update_time = datetime.now()
            print(
                f"[DuelingDDQN Train] Device={device} | Dueling={self.enable_dueling_dqn} | "
                f"Double={self.enable_double_dqn} | Bot={self.bot_type} | Model={self.MODEL_FILE}"
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

        gif_frames = []
        gif_saved = False

        episode_iter = range(num_episodes) if num_episodes is not None else itertools.count()

        try:
            for episode in episode_iter:
                state, reset_info = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device)
                state_orange = (
                    torch.tensor(reset_info["obs_orange"], dtype=torch.float32, device=device)
                    if (not is_training and control_car == "both" and "obs_orange" in reset_info)
                    else None
                )

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

                    action_orange_idx = None
                    if not is_training and control_car == "both" and state_orange is not None:
                        with torch.no_grad():
                            action_orange = policy_dqn(state_orange.unsqueeze(0)).squeeze(0).argmax()
                            action_orange_idx = int(action_orange.item())

                    next_state, reward, terminated, truncated, info = self._step_with_action_repeat(
                        env, action_idx, action_orange_idx
                    )

                    if not is_training and save_gif and not gif_saved and len(gif_frames) < 1200:
                        frame = env.unwrapped.get_frame()
                        if frame is not None:
                            gif_frames.append(frame)

                    episode_reward += reward
                    if info.get("is_success", False) or (terminated and reward > 0.0):
                        scored_goal = 1

                    new_state = torch.tensor(next_state, dtype=torch.float32, device=device)
                    if not is_training and control_car == "both" and "obs_orange" in info:
                        state_orange = torch.tensor(info["obs_orange"], dtype=torch.float32, device=device)
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
                            current_loss = self.optimize(mini_batch, policy_dqn, target_dqn)

                        # Sync target network every `network_sync_rate` decision steps
                        if step_count % self.network_sync_rate == 0:
                            target_dqn.load_state_dict(policy_dqn.state_dict())

                    state = new_state

                    if not env.unwrapped.isopen:
                        break

                rewards_per_episode.append(episode_reward)
                goals_per_episode.append(scored_goal)

                if not is_training and save_gif and not gif_saved and len(gif_frames) > 0:
                    out_gif = (
                        gif_path if gif_path is not None else os.path.join(RUNS_DIR, f"eval_{self.hyperparameter_set}.gif")
                    )
                    self._save_gif(gif_frames, out_gif)
                    gif_saved = True
                mean_reward_100 = float(np.mean(rewards_per_episode[-100:]))
                goal_rate_100 = float(np.mean(goals_per_episode[-100:])) * 100.0
                score_blue = int(env.unwrapped.sim.score_blue)
                score_orange = int(env.unwrapped.sim.score_orange)

                print(
                    f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
                    f"Mean(100) = {mean_reward_100:7.2f} | GoalRate(100) = {goal_rate_100:5.1f}% | "
                    f"Eps = {epsilon:.3f} | Steps = {step_count}"
                    if is_training
                    else f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
                    f"Score(Blue-Orange) = {score_blue}-{score_orange} | "
                    f"Goal = {bool(scored_goal)} | Terminated = {terminated} | Truncated = {truncated}"
                )

                if is_training:
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)
                    steps_history.append(step_count)
                    loss_history.append(current_loss)

                    # Append metrics to CSV log for flexible offline plotting & evaluation
                    write_header = not os.path.exists(self.METRICS_FILE) or os.path.getsize(self.METRICS_FILE) == 0
                    with open(self.METRICS_FILE, "a", newline="") as f:
                        if write_header:
                            f.write("episode,step,reward,mean_reward_100,goal,goal_rate_100,epsilon,td_loss\n")
                        f.write(
                            f"{episode},{step_count},{episode_reward:.4f},{mean_reward_100:.4f},"
                            f"{scored_goal},{goal_rate_100:.2f},{epsilon:.4f},{current_loss:.6f}\n"
                        )

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

                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                    current_time = datetime.now()
                    if current_time - last_graph_update_time > timedelta(seconds=30):
                        self.save_graph(
                            rewards_per_episode,
                            epsilon_history,
                            steps_per_episode=steps_history,
                            goals_per_episode=goals_per_episode,
                            loss_history=loss_history,
                            dpi=100,
                        )
                        last_graph_update_time = current_time

                    if not env.unwrapped.isopen:
                        print("Exiting environment.")
                        break
        finally:
            final_score_blue = int(env.unwrapped.sim.score_blue)
            final_score_orange = int(env.unwrapped.sim.score_orange)
            if is_training and len(rewards_per_episode) > 0:
                self.save_graph(
                    rewards_per_episode,
                    epsilon_history,
                    steps_per_episode=steps_history,
                    goals_per_episode=goals_per_episode,
                    loss_history=loss_history,
                )
            env.close()

        return {
            "episodes": len(rewards_per_episode),
            "mean_reward": float(np.mean(rewards_per_episode)) if rewards_per_episode else 0.0,
            "goal_rate": float(np.mean(goals_per_episode)) * 100.0 if goals_per_episode else 0.0,
            "score_blue": final_score_blue,
            "score_orange": final_score_orange,
        }


# Backwards-compatible alias
Agent = DuelingDDQNAgent


def train_dueling_ddqn(
    num_episodes: int | None = None,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
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
        model_path=model_path,
    )


def evaluate_dueling_ddqn(
    num_episodes: int | None = DQNConfig.eval_episodes,
    render: bool = False,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
    enable_dueling_dqn: bool = True,
    enable_double_dqn: bool = True,
    hyperparameter_set: str = DQNConfig.hyperparameter_set,
    save_gif: bool = False,
    gif_path: str | Path | None = None,
    control_car: str = "blue",
) -> dict[str, float]:
    """Evaluate/validate a trained Dueling Double DQN agent."""
    agent = DuelingDDQNAgent(
        hyperparameter_set=hyperparameter_set,
        enable_dueling_dqn=enable_dueling_dqn,
        enable_double_dqn=enable_double_dqn,
    )
    episodes = None if (render and num_episodes is None and not save_gif) else num_episodes
    return agent.run(
        is_training=False,
        render=render,
        max_episode_steps=max_episode_steps,
        num_episodes=episodes,
        model_path=model_path,
        save_gif=save_gif,
        gif_path=gif_path,
        control_car=control_car,
    )


def evaluate_mixed_policies(
    blue_policy: str = "dueling_ddqn",
    orange_policy: str = "ppo",
    blue_model_path: str | Path | None = None,
    orange_model_path: str | Path | None = None,
    num_episodes: int | None = DQNConfig.eval_episodes,
    render: bool = False,
    max_episode_steps: int | None = None,
    save_gif: bool = False,
    gif_path: str | Path | None = None,
) -> dict[str, float]:
    """Evaluate two potentially different policies head-to-head (Blue policy vs. mirrored Orange policy).

    Supports any combination of ('dueling_ddqn', 'double_dqn', 'dqn', 'ppo') for Blue and Orange.
    """
    from src.config.nn_config import PPOConfig
    from src.neural_nets.ppo_torch import ActorCritic

    def _norm_policy_name(name: str) -> str:
        n = name.strip().lstrip("-").lower()
        if n in ("ddqn", "dueling_ddqn"):
            return "dueling_ddqn"
        if n in ("double_dqn", "dqn", "ppo"):
            return n
        raise ValueError(f"Unsupported policy type '{name}'. Choose from: dueling_ddqn, double_dqn, dqn, ppo.")

    blue_pol = _norm_policy_name(blue_policy)
    orange_pol = _norm_policy_name(orange_policy)

    make_kwargs = {}
    if max_episode_steps is not None:
        make_kwargs["max_episode_steps"] = max_episode_steps

    if render:
        render_mode = "human"
    elif save_gif:
        render_mode = "rgb_array"
    else:
        render_mode = None

    env = gym.make(
        GymConfig.gym_id,
        render_mode=render_mode,
        bot_type=DQNConfig.bot_type,
        discrete_action=True,
        control_car="both",
        **make_kwargs,
    )
    num_states = env.observation_space.shape[0]

    def _build_policy_selector(team_label: str, pol_type: str, ckpt_override: str | Path | None):
        if pol_type == "ppo":
            default_ckpt = os.path.join(ROOT_DIR, PPOConfig.runs_dir, "model_ppo.pt")
            ckpt_file = str(ckpt_override) if ckpt_override is not None else default_ckpt
            state_dict = None
            factored = getattr(PPOConfig, "factored_action", True)
            if os.path.exists(ckpt_file):
                state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)
                if "actor_dir.weight" in state_dict:
                    factored = True
                elif "actor_stream.2.weight" in state_dict:
                    factored = False
            action_dim = list(PPOConfig.action_dims) if factored else 36
            net = ActorCritic(num_states, action_dim, PPOConfig.hidden_size).to(device)
            if state_dict is not None:
                net.load_state_dict(state_dict)
                print(f"[Mix Eval ({team_label}: PPO)] Loaded trained weights from: {ckpt_file}")
            else:
                print(f"[Mix Eval ({team_label}: PPO)] Warning: Checkpoint '{ckpt_file}' not found; using initial weights.")
            net.eval()

            def _select(obs_t: torch.Tensor):
                with torch.no_grad():
                    logits, _ = net(obs_t.unsqueeze(0))
                    if net.is_factored:
                        return [int(l.squeeze(0).argmax().item()) for l in logits]
                    return int(logits.squeeze(0).argmax().item())

            return _select
        else:
            default_ckpt = os.path.join(ROOT_DIR, DQNConfig.runs_dir, f"model_{pol_type}.pt")
            ckpt_file = str(ckpt_override) if ckpt_override is not None else default_ckpt
            enable_dueling = pol_type == "dueling_ddqn"
            state_dict = None
            if os.path.exists(ckpt_file):
                state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)
                enable_dueling = "fc_value.weight" in state_dict
            net = DQN(num_states, 36, DQNConfig.hidden_size, enable_dueling_dqn=enable_dueling).to(device)
            if state_dict is not None:
                net.load_state_dict(state_dict)
                print(f"[Mix Eval ({team_label}: {pol_type})] Loaded trained weights from: {ckpt_file}")
            else:
                print(
                    f"[Mix Eval ({team_label}: {pol_type})] Warning: Checkpoint '{ckpt_file}' not found; using initial weights."
                )
            net.eval()

            def _select(obs_t: torch.Tensor):
                with torch.no_grad():
                    return int(net(obs_t.unsqueeze(0)).squeeze(0).argmax().item())

            return _select

    def _clear_jump(act):
        if isinstance(act, (list, tuple, np.ndarray)):
            return [int(act[0]), 0, int(act[2])]
        jump_bit = (int(act) % 4) // 2
        return (int(act) - 2) if jump_bit == 1 else int(act)

    select_blue = _build_policy_selector("Blue", blue_pol, blue_model_path)
    select_orange = _build_policy_selector("Orange", orange_pol, orange_model_path)

    episodes = None if (render and num_episodes is None and not save_gif) else num_episodes
    episode_iter = range(episodes) if episodes is not None else itertools.count()
    action_repeat = DQNConfig.action_repeat

    rewards_per_episode = []
    goals_per_episode = []
    gif_frames = []
    gif_saved = False
    helper_agent = DuelingDDQNAgent()

    try:
        for episode in episode_iter:
            state, reset_info = env.reset()
            state_b = torch.tensor(state, dtype=torch.float32, device=device)
            state_o = torch.tensor(reset_info["obs_orange"], dtype=torch.float32, device=device)

            if not env.unwrapped.isopen:
                break

            terminated = False
            truncated = False
            episode_reward = 0.0
            scored_goal = 0

            while not (terminated or truncated):
                act_b = select_blue(state_b)
                act_o = select_orange(state_o)
                held_b = _clear_jump(act_b)
                held_o = _clear_jump(act_o)

                reward_sum = 0.0
                repeat = max(1, action_repeat)
                unwrapped = env.unwrapped
                deferred = False
                for sub in range(repeat):
                    deferred = sub < (repeat - 1)
                    unwrapped._defer_obs = deferred
                    ab = act_b if sub == 0 else held_b
                    ao = act_o if sub == 0 else held_o
                    next_state, r, terminated, truncated, info = env.step((ab, ao))
                    reward_sum += float(r)
                    if terminated or truncated or not unwrapped.isopen:
                        break
                unwrapped._defer_obs = False
                if deferred:
                    next_state = unwrapped._get_obs()
                    info["obs_orange"] = unwrapped._get_obs("orange")

                if save_gif and not gif_saved and len(gif_frames) < 1200:
                    frame = env.unwrapped.get_frame()
                    if frame is not None:
                        gif_frames.append(frame)

                episode_reward += reward_sum
                if info.get("is_success", False) or (terminated and reward_sum > 0.0):
                    scored_goal = 1

                state_b = torch.tensor(next_state, dtype=torch.float32, device=device)
                state_o = torch.tensor(info["obs_orange"], dtype=torch.float32, device=device)

                if not env.unwrapped.isopen:
                    break

            rewards_per_episode.append(episode_reward)
            goals_per_episode.append(scored_goal)

            if save_gif and not gif_saved and len(gif_frames) > 0:
                out_gif = (
                    gif_path
                    if gif_path is not None
                    else os.path.join(RUNS_DIR, f"eval_mix_{blue_pol}_vs_{orange_pol}.gif")
                )
                helper_agent._save_gif(gif_frames, out_gif)
                gif_saved = True

            score_blue = int(env.unwrapped.sim.score_blue)
            score_orange = int(env.unwrapped.sim.score_orange)
            print(
                f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
                f"Score(Blue-Orange) = {score_blue}-{score_orange} | "
                f"Goal = {bool(scored_goal)} | Terminated = {terminated} | Truncated = {truncated}"
            )
    finally:
        final_score_blue = int(env.unwrapped.sim.score_blue)
        final_score_orange = int(env.unwrapped.sim.score_orange)
        env.close()

    return {
        "blue_policy": blue_pol,
        "orange_policy": orange_pol,
        "episodes": len(rewards_per_episode),
        "mean_reward": float(np.mean(rewards_per_episode)) if rewards_per_episode else 0.0,
        "goal_rate": float(np.mean(goals_per_episode)) * 100.0 if goals_per_episode else 0.0,
        "score_blue": final_score_blue,
        "score_orange": final_score_orange,
    }


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
    parser.add_argument("--gif", action="store_true", help="Record and save an animated GIF of the evaluation episode")
    parser.add_argument("--gif_path", type=str, default=None, help="Optional custom output path for the GIF")
    parser.add_argument("--blue", action="store_true", help="Policy controls Blue car vs HeuristicBot Orange (default)")
    parser.add_argument("--orange", action="store_true", help="Policy controls Orange car (mirrored) vs Human Blue")
    parser.add_argument(
        "--bot-vs-bot",
        "--bots",
        "--both",
        dest="bot_vs_bot",
        action="store_true",
        help="Policy controls both Blue and Orange cars",
    )
    args = parser.parse_args()

    if args.train:
        train_dueling_ddqn(
            num_episodes=args.episodes,
            model_path=args.model_path,
            hyperparameter_set=args.hyperparameter_set,
        )
    else:
        if args.orange:
            ctrl_car = "orange"
        elif args.bot_vs_bot:
            ctrl_car = "both"
        else:
            ctrl_car = "blue"
        eval_eps = args.episodes if args.episodes is not None else (None if args.render else DQNConfig.eval_episodes)
        evaluate_dueling_ddqn(
            num_episodes=eval_eps,
            render=args.render,
            model_path=args.model_path,
            hyperparameter_set=args.hyperparameter_set,
            save_gif=args.gif,
            gif_path=args.gif_path,
            control_car=ctrl_car,
        )
