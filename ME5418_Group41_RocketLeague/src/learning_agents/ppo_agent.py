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
        self.target_kl = getattr(PPOConfig, "target_kl", 0.02)

        self.action_repeat = PPOConfig.action_repeat
        self.rollout_steps = PPOConfig.rollout_steps
        self.update_epochs = PPOConfig.update_epochs
        self.mini_batch_size = PPOConfig.minibatch_size

        self.optimizer = None

        self.LOG_FILE = os.path.join(RUNS_DIR, f"run_{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"model_{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"graph_{self.hyperparameter_set}.png")
        self.METRICS_FILE = os.path.join(RUNS_DIR, f"metrics_{self.hyperparameter_set}.csv")

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
    ) -> tuple[float, float, float, float]:
        """Run `update_epochs` of shuffled minibatch PPO clipped surrogate updates.

        Returns:
            (mean_total_loss, mean_entropy, mean_value_loss, mean_approx_kl)
        """
        batch_size = states.size(0)
        total_loss_sum = 0.0
        entropy_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        num_mb = 0

        # Normalize advantages across the entire 4,096-sample rollout batch before minibatch sampling
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for epoch in range(self.update_epochs):
            perm = torch.randperm(batch_size, device=device)
            epoch_kl_sum = 0.0
            epoch_mb_count = 0

            for start in range(0, batch_size, self.mini_batch_size):
                idx = perm[start : start + self.mini_batch_size]
                mb_states = states[idx]
                mb_actions = actions[idx]
                mb_old_log_probs = old_log_probs[idx]
                mb_returns = returns[idx]
                mb_adv = advantages[idx]

                _, new_log_probs, entropy, new_values = policy_net.get_action_and_value(mb_states, mb_actions)
                logratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(logratio)

                # Non-negative, minimum-variance KL divergence approximation
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    kl_val = float(approx_kl.item())
                    epoch_kl_sum += kl_val
                    kl_sum += kl_val
                    epoch_mb_count += 1

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
                value_loss_sum += float(v_loss.item())
                num_mb += 1

            # Early stop if average KL divergence across this epoch exceeded target threshold
            mean_epoch_kl = epoch_kl_sum / max(epoch_mb_count, 1)
            if self.target_kl is not None and mean_epoch_kl > self.target_kl:
                break

        denom = max(num_mb, 1)
        return total_loss_sum / denom, entropy_sum / denom, value_loss_sum / denom, kl_sum / denom

    def _step_with_action_repeat(self, env: gym.Env, action, action_orange=None):
        """Repeat chosen action for `action_repeat` frames, releasing jump after frame 0 so
        edge-triggered jumping in Car._apply_jump can fire again on subsequent decisions."""
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        next_state = None

        # Clear jump bit on repeat frames > 0 so _prev_jump_action resets between decisions
        if isinstance(action, (list, tuple, np.ndarray)):
            held_action = [int(action[0]), 0, int(action[2])]
        else:
            jump_bit = (action % 4) // 2
            held_action = (action - 2) if jump_bit == 1 else action

        if action_orange is not None:
            if isinstance(action_orange, (list, tuple, np.ndarray)):
                held_action_orange = [int(action_orange[0]), 0, int(action_orange[2])]
            else:
                jump_bit_orange = (action_orange % 4) // 2
                held_action_orange = (action_orange - 2) if jump_bit_orange == 1 else action_orange
        else:
            held_action_orange = None

        repeat = max(1, self.action_repeat)
        unwrapped = env.unwrapped
        deferred = False
        for sub in range(repeat):
            deferred = sub < (repeat - 1)
            unwrapped._defer_obs = deferred
            act = action if sub == 0 else held_action
            if action_orange is not None:
                act_orange = action_orange if sub == 0 else held_action_orange
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
        print(f"[PPO Eval] Saved evaluation GIF ({len(pil_images)} frames, {fps} FPS) to: {path}")

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
        entropy_history: list[float],
        steps_per_episode: list[int] | None = None,
        goals_per_episode: list[int] | None = None,
        value_loss_history: list[float] | None = None,
        dpi: int = 150,
    ):
        """Plot and save a comprehensive 2x2 training dashboard.

        Subplots:
            1. Sample Efficiency: 100-episode rolling mean reward & episodic return vs. Environment Steps.
            2. Task Competency: 100-episode rolling goal conversion rate (%) vs. Environment Steps.
            3. Exploration Dynamics: Policy entropy H(pi) vs. Environment Steps.
            4. Critic Optimization Health: Critic value MSE loss (with 50-ep smoothing) vs. Environment Steps.
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

        # 3. Policy Entropy
        entropy_arr = np.array(entropy_history) if entropy_history is not None else np.array([])
        if len(entropy_arr) == n:
            axs[1, 0].plot(x, entropy_arr, color="#ff7f0e", linewidth=2.0, label="Policy Entropy H(pi)")
        elif len(entropy_arr) > 0:
            axs[1, 0].plot(entropy_arr, color="#ff7f0e", linewidth=2.0, label="Policy Entropy H(pi)")
        axs[1, 0].set_title("Exploration Dynamics (Policy Entropy)", fontsize=11, fontweight="bold")
        axs[1, 0].set_xlabel(x_label)
        axs[1, 0].set_ylabel("Entropy H(pi)")
        axs[1, 0].grid(True, alpha=0.3, linestyle="--")
        axs[1, 0].legend(loc="upper right", framealpha=0.8)

        # 4. Critic Value MSE Loss
        if value_loss_history is not None and len(value_loss_history) == n:
            v_losses = np.array(value_loss_history)
            axs[1, 1].plot(x, v_losses, color="#d62728", alpha=0.25, linewidth=0.8, label="Raw Value Loss")
            smooth_loss = self._rolling_mean(v_losses, 50)
            axs[1, 1].plot(x, smooth_loss, color="#d62728", linewidth=2.0, label="50-Ep Smoothed Loss")
            axs[1, 1].legend(loc="upper right", framealpha=0.8)
        axs[1, 1].set_title("Critic Optimization Health (Value Loss)", fontsize=11, fontweight="bold")
        axs[1, 1].set_xlabel(x_label)
        axs[1, 1].set_ylabel("Value MSE Loss")
        axs[1, 1].grid(True, alpha=0.3, linestyle="--")

        fig.suptitle(f"PPO Training Dashboard ({self.hyperparameter_set})", fontsize=13, fontweight="bold")
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
        random_spawn: bool = True,
    ) -> dict[str, float]:
        """Run PPO training (`is_training=True`) or validation/evaluation (`is_training=False`)."""
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

        ckpt_file = str(model_path) if model_path is not None else self.MODEL_FILE
        preloaded_state_dict = None
        factored = getattr(PPOConfig, "factored_action", False)
        hidden_dim = self.fcl_nodes
        if (not is_training or model_path is not None) and os.path.exists(ckpt_file):
            preloaded_state_dict = torch.load(ckpt_file, map_location=device, weights_only=True)
            hidden_dim = preloaded_state_dict["fc1.weight"].shape[0]
            if "actor_heads.0.weight" in preloaded_state_dict:
                factored = True
            elif "actor.weight" in preloaded_state_dict:
                factored = False

        env = gym.make(
            self.env_id,
            render_mode=render_mode,
            bot_type=self.bot_type,
            discrete_action=not factored,
            control_car="blue" if is_training else control_car,
            random_spawn=random_spawn,
            **make_kwargs,
        )

        num_states = env.observation_space.shape[0]
        if preloaded_state_dict is not None and "actor_heads.0.weight" in preloaded_state_dict:
            n_heads = len({k.split(".")[1] for k in preloaded_state_dict if k.startswith("actor_heads.")})
            action_dim: int | list[int] = [
                preloaded_state_dict[f"actor_heads.{i}.weight"].shape[0] for i in range(n_heads)
            ]
        elif preloaded_state_dict is not None and "actor.weight" in preloaded_state_dict:
            action_dim = preloaded_state_dict["actor.weight"].shape[0]
        else:
            action_dim = list(env.action_space.nvec) if factored else env.action_space.n

        rewards_per_episode = []
        goals_per_episode = []

        policy_net = ActorCritic(num_states, action_dim, hidden_dim).to(device)

        if is_training:
            if model_path is not None:
                if preloaded_state_dict is not None:
                    policy_net.load_state_dict(preloaded_state_dict)
                    print(f"[PPO Train] Loaded model weights from: {ckpt_file}")
                else:
                    print(f"[PPO Train] Warning: Checkpoint '{ckpt_file}' not found; training from scratch.")
            else:
                print("[PPO Train] Training from scratch; no model checkpoint specified.")
            self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.learning_rate, eps=1e-5)
            step_count = 0
            entropy_history = []
            steps_history = []
            value_loss_history = []
            current_v_loss = 0.0
            current_entropy = (
                float(sum(np.log(d) for d in action_dim))
                if isinstance(action_dim, list)
                else float(np.log(action_dim))
            )
            best_mean_reward = float("-inf")
            last_graph_update_time = datetime.now()

            # On-policy rollout storage lists
            buf_states, buf_actions, buf_log_probs = [], [], []
            buf_rewards, buf_values, buf_next_values, buf_dones = [], [], [], []

            print(
                f"[PPO Train] Device={device} | Rollout={self.rollout_steps} | Epochs={self.update_epochs} | "
                f"Minibatch={self.mini_batch_size} | Bot={self.bot_type} | Model={self.MODEL_FILE}"
            )
        else:
            if preloaded_state_dict is not None:
                policy_net.load_state_dict(preloaded_state_dict)
                print(f"[PPO Eval] Loaded trained model weights from: {ckpt_file}")
            else:
                print(f"[PPO Eval] Warning: Checkpoint '{ckpt_file}' not found; evaluating initial weights.")
            policy_net.eval()
            step_count = 0
            current_entropy = 0.0

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
                    action_orange_for_env = None
                    with torch.no_grad():
                        if is_training:
                            action, log_prob, _, value = policy_net.get_action_and_value(state.unsqueeze(0))
                            action_for_env = (
                                action.squeeze(0).cpu().numpy().tolist()
                                if policy_net.is_factored
                                else int(action.item())
                            )

                            # Backfill the previous step's next_value using this step's already-computed value
                            if (len(buf_next_values) > 0 and buf_next_values[-1] is None):
                                buf_next_values[-1] = value.squeeze(0)
                        else:
                            logits, _ = policy_net(state.unsqueeze(0))
                            if policy_net.is_factored:
                                action_for_env = [int(l.squeeze(0).argmax().item()) for l in logits]
                            else:
                                action_for_env = int(logits.squeeze(0).argmax().item())

                            if control_car == "both" and state_orange is not None:
                                logits_o, _ = policy_net(state_orange.unsqueeze(0))
                                if policy_net.is_factored:
                                    action_orange_for_env = [int(l.squeeze(0).argmax().item()) for l in logits_o]
                                else:
                                    action_orange_for_env = int(logits_o.squeeze(0).argmax().item())

                    next_state, reward, terminated, truncated, info = self._step_with_action_repeat(
                        env, action_for_env, action_orange_for_env
                    )
                    done = terminated or truncated

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

                    if is_training:
                        with torch.no_grad():
                            # Truncation-aware bootstrapping:
                            # - Terminal goal state (`terminated=True`) -> bootstrap 0.0
                            # - Timeout state (`truncated=True`) -> evaluate V(new_state) once before reset                   
                            # - Non-terminal (`done=False`) -> defer forward pass to the next step
                            if terminated:
                                next_val = torch.tensor(0.0, dtype=torch.float32, device=device)
                            elif truncated:
                                with torch.no_grad():
                                    _, next_val = policy_net(new_state.unsqueeze(0))
                                    next_val = next_val.squeeze(0)
                            else:
                                next_val = None

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
                            # if the rollout cut off mid-episode, bootstrap the tail step's next_value
                            if buf_next_values[-1] is None:
                                with torch.no_grad():
                                    _, tail_val = policy_net(new_state.unsqueeze(0))
                                    buf_next_values[-1] = tail_val.squeeze(0)

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
                            _, current_entropy, current_v_loss, _ = self.optimize(
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
                    f"Ent = {current_entropy:.3f} | Steps = {step_count}"
                    if is_training
                    else f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
                    f"Score(Blue-Orange) = {score_blue}-{score_orange} | "
                    f"Goal = {bool(scored_goal)} | Terminated = {terminated} | Truncated = {truncated}"
                )

                if is_training:
                    steps_history.append(step_count)
                    entropy_history.append(current_entropy)
                    value_loss_history.append(current_v_loss)

                    # Append metrics to CSV log for flexible offline plotting & evaluation
                    write_header = not os.path.exists(self.METRICS_FILE) or os.path.getsize(self.METRICS_FILE) == 0
                    with open(self.METRICS_FILE, "a", newline="") as f:
                        if write_header:
                            f.write("episode,step,reward,mean_reward_100,goal,goal_rate_100,entropy,value_loss\n")
                        f.write(
                            f"{episode},{step_count},{episode_reward:.4f},{mean_reward_100:.4f},"
                            f"{scored_goal},{goal_rate_100:.2f},{current_entropy:.4f},{current_v_loss:.6f}\n"
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

                        # torch.save(policy_net.state_dict(), ckpt_file)
                        torch.save(policy_net.state_dict(), self.MODEL_FILE)

                    current_time = datetime.now()
                    if current_time - last_graph_update_time > timedelta(seconds=30):
                        self.save_graph(
                            rewards_per_episode,
                            entropy_history,
                            steps_per_episode=steps_history,
                            goals_per_episode=goals_per_episode,
                            value_loss_history=value_loss_history,
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
                    entropy_history,
                    steps_per_episode=steps_history,
                    goals_per_episode=goals_per_episode,
                    value_loss_history=value_loss_history,
                )
            env.close()

        return {
            "episodes": len(rewards_per_episode),
            "mean_reward": float(np.mean(rewards_per_episode)) if rewards_per_episode else 0.0,
            "goal_rate": float(np.mean(goals_per_episode)) * 100.0 if goals_per_episode else 0.0,
            "score_blue": final_score_blue,
            "score_orange": final_score_orange,
        }


def train_ppo(
    num_episodes: int | None = None,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
    hyperparameter_set: str = PPOConfig.hyperparameter_set,
) -> dict[str, float]:
    """Train the PPO agent."""
    agent = PPOAgent(hyperparameter_set=hyperparameter_set)
    return agent.run(
        is_training=True,
        render=False,
        max_episode_steps=max_episode_steps,
        num_episodes=num_episodes,
        model_path=model_path,
    )


def evaluate_ppo(
    num_episodes: int | None = PPOConfig.eval_episodes,
    render: bool = False,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
    hyperparameter_set: str = PPOConfig.hyperparameter_set,
    save_gif: bool = False,
    gif_path: str | Path | None = None,
    control_car: str = "blue",
    random_spawn: bool = True,
) -> dict[str, float]:
    """Evaluate/validate a trained PPO agent."""
    agent = PPOAgent(hyperparameter_set=hyperparameter_set)
    episodes = (
        None
        if (render and num_episodes is None and not save_gif)
        else (num_episodes if num_episodes is not None else PPOConfig.eval_episodes)
    )
    return agent.run(
        is_training=False,
        render=render,
        max_episode_steps=max_episode_steps,
        num_episodes=episodes,
        model_path=model_path,
        save_gif=save_gif,
        gif_path=gif_path,
        control_car=control_car,
        random_spawn=random_spawn,
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
        train_ppo(
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
        eval_eps = args.episodes if args.episodes is not None else (None if args.render else PPOConfig.eval_episodes)
        evaluate_ppo(
            num_episodes=eval_eps,
            render=args.render,
            model_path=args.model_path,
            hyperparameter_set=args.hyperparameter_set,
            save_gif=args.gif,
            gif_path=args.gif_path,
            control_car=ctrl_car,
        )
