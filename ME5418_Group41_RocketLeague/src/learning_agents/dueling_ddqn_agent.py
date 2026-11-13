"""
Dueling Double DQN (D3QN) Learning Agent — Training & Validation/Evaluation Pipeline.

Adapted from johnnycode8's PyTorch DQN tutorial (https://github.com/johnnycode8/dqn_pytorch)
and integrated with `src/neural_nets/dueling_ddqn_torch.py`, `src/config/nn_config.py`, and
`src/gym_env/rocket_league_env.py`.

This file holds the DQN learning rule and its training loop. Environment creation, action repeat,
the evaluation loop, logging and plotting are shared with PPO (`env_factory`, `eval_loop`, `training_logger`, `dashboard`, `run_dirs`).
"""

import itertools
import os
import random
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn

from src.config.gym_config import GymConfig
from src.config.nn_config import DQNConfig
from src.learning_agents.dashboard import DashboardPanel, save_training_dashboard
from src.learning_agents.env_factory import device, make_env, normalize_control_car, obs_to_tensor, scored_goal
from src.learning_agents.eval_loop import resolve_eval_episodes, run_eval_episodes, summarize_episodes
from src.learning_agents.experience_replay import ReplayMemory
from src.learning_agents.run_dirs import latest_model_path, make_run_dir, save_periodic_checkpoint
from src.learning_agents.training_logger import TrainingLogger
from src.neural_nets.checkpoint import load_state_dict
from src.neural_nets.dueling_ddqn_torch import DQN
from src.neural_nets.policy import greedy_action


class DuelingDDQNAgent:
    """Dueling Double DQN Learning Agent supporting both training and validation/evaluation.

    Key algorithmic components:
        1. Dueling Double DQN architecture (`DQN` from `src.neural_nets.dueling_ddqn_torch`):
           Decouples state-value stream V(s) and action-advantage stream A(s, a), and
           uses the online policy network to select argmax next actions while the target
           network evaluates them to mitigate maximization bias.
        2. Experience Replay (`ReplayMemory`):
           Stores `(state, action, next_state, reward, terminated, truncated)` transitions.
        3. Action Repeat with Edge-Triggered Jump Release (`ActionRepeat` env wrapper):
           Repeats chosen discrete action for `action_repeat` frames while clearing the jump
           bit on repeat frames > 0 so `Car._apply_jump` edge-triggering resets cleanly.
        4. Truncation-Aware Value Bootstrapping (`optimize`):
           Bootstraps V(s') on time-limit `truncated` episodes and only zeroes future value
           on true goal `terminated` states (`target_q = r + (1 - terminated) * gamma * next_q`).

    One training decision step (`_train`):
        `_select_action` -> `env.step` (`ActionRepeat`) -> `_store_transition` -> `_maybe_optimize`
        -> `_sync_target_network` (every `target_update_interval` steps); `_decay_epsilon` once per episode.
    """

    def __init__(self, config: DQNConfig | None = None):
        self.cfg = config or DQNConfig()
        # SmoothL1Loss (Huber loss) prevents exploding gradients from large +/-20 goal rewards
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = None

    def _set_run_dir(self, run_dir: str | Path):
        """Point every training output (log, checkpoint, dashboard, metrics CSV) at `run_dir`."""
        self.LOG_FILE = os.path.join(run_dir, f"run_{self.cfg.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(run_dir, f"model_{self.cfg.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(run_dir, f"graph_{self.cfg.hyperparameter_set}.png")
        self.METRICS_FILE = os.path.join(run_dir, f"metrics_{self.cfg.hyperparameter_set}.csv")
        self.CHECKPOINT_DIR = os.path.join(run_dir, "checkpoints")

    # ----------------------------------------------------------------------------------------
    # Learning rule
    # ----------------------------------------------------------------------------------------

    def optimize(self, mini_batch, policy_dqn: DQN, target_dqn: DQN) -> float:
        """Perform a single Double/Dueling DQN gradient step on a sampled minibatch."""
        state, action, next_state, reward, terminated, truncated = zip(*mini_batch, strict=True)

        state = torch.stack(state)
        action = torch.stack(action)
        next_state = torch.stack(next_state)
        reward = torch.stack(reward)
        terminated = torch.tensor(terminated, dtype=torch.float32, device=device)

        with torch.no_grad():
            if self.cfg.enable_double_dqn:
                next_actions = policy_dqn(next_state).argmax(dim=1, keepdim=True)
                next_q = target_dqn(next_state).gather(dim=1, index=next_actions).squeeze(1)
            else:
                next_q = target_dqn(next_state).max(dim=1)[0]

            # Bootstrap on time-limit truncation; only zero out future value on true goal termination
            target_q = reward + (1.0 - terminated) * self.cfg.discount_factor * next_q

        current_q = policy_dqn(state).gather(dim=1, index=action.unsqueeze(dim=1)).squeeze(1)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=self.cfg.max_grad_norm)
        self.optimizer.step()
        return float(loss.item())

    # ----------------------------------------------------------------------------------------
    # Algorithm steps
    # ----------------------------------------------------------------------------------------

    def _build_network(self, num_states: int, num_actions: int) -> DQN:
        return DQN(num_states, num_actions, self.cfg.hidden_size, enable_dueling_dqn=self.cfg.enable_dueling_dqn).to(
            device
        )

    def _load_checkpoint(self, policy_dqn: DQN, ckpt_file: str) -> bool:
        """Load weights into `policy_dqn`; returns False when the checkpoint file does not exist."""
        state_dict = load_state_dict(ckpt_file, device)
        if state_dict is None:
            return False
        policy_dqn.load_state_dict(state_dict)
        return True

    def _greedy_action(self, policy_dqn: DQN, state: torch.Tensor) -> torch.Tensor:
        """argmax_a Q(s, a) as a 0-dim int64 tensor."""
        return torch.tensor(greedy_action(policy_dqn, state), dtype=torch.int64, device=device)

    def _select_action(self, env: gym.Env, policy_dqn: DQN, state: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Epsilon-greedy: a uniformly random action with probability `epsilon`, else the greedy one."""
        if random.random() < epsilon:
            return torch.tensor(env.action_space.sample(), dtype=torch.int64, device=device)
        return self._greedy_action(policy_dqn, state)

    def _store_transition(
        self,
        memory: ReplayMemory,
        state: torch.Tensor,
        action: torch.Tensor,
        new_state: torch.Tensor,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
        memory.append((state, action, new_state, reward_tensor, terminated, truncated))

    def _maybe_optimize(self, memory: ReplayMemory, step_count: int, policy_dqn: DQN, target_dqn: DQN) -> float | None:
        """Run one gradient step every `train_freq` decision steps once the replay memory has warmed
        up; returns the TD loss, or None when no update happened."""
        warmed_up = len(memory) >= max(self.cfg.learning_start, self.cfg.batch_size)
        if not warmed_up or step_count % self.cfg.train_freq != 0:
            return None
        mini_batch = memory.sample(self.cfg.batch_size)
        return self.optimize(mini_batch, policy_dqn, target_dqn)

    def _sync_target_network(self, policy_dqn: DQN, target_dqn: DQN):
        """Hard update: copy the online network's weights into the target network."""
        target_dqn.load_state_dict(policy_dqn.state_dict())

    def _decay_epsilon(self, epsilon: float) -> float:
        """Per-episode exponential decay, floored at `epsilon_min`."""
        return max(epsilon * self.cfg.epsilon_decay, self.cfg.epsilon_min)

    # ----------------------------------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------------------------------

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
        agent_name = "Dueling Double DQN" if (self.cfg.enable_dueling_dqn and self.cfg.enable_double_dqn) else "DQN"
        save_training_dashboard(
            self.GRAPH_FILE,
            f"{agent_name} Training Dashboard ({self.cfg.hyperparameter_set})",
            rewards_per_episode,
            steps_per_episode,
            goals_per_episode,
            exploration=DashboardPanel(
                values=epsilon_history,
                title="Exploration Schedule (Epsilon Decay)",
                ylabel="Epsilon (ε)",
                label="Epsilon (ε)",
                ylim=(-0.02, 1.05),
            ),
            loss=DashboardPanel(
                values=loss_history,
                title="Bellman Optimization Health (TD Loss)",
                ylabel="Huber Loss",
                label="Raw TD Loss",
            ),
            dpi=dpi,
        )

    # ----------------------------------------------------------------------------------------
    # Training / evaluation loops
    # ----------------------------------------------------------------------------------------

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
        """Run training (`is_training=True`) or validation/evaluation (`is_training=False`)."""
        control_car = normalize_control_car(control_car)
        env = make_env(
            bot_type=self.cfg.bot_type,
            discrete_action=True,
            control_car="blue" if is_training else control_car,
            render=render,
            save_gif=save_gif,
            random_spawn=random_spawn,
            max_episode_steps=max_episode_steps,
            env_id=GymConfig.gym_id,
            action_repeat=self.cfg.action_repeat,
        )
        try:
            policy_dqn = self._build_network(env.observation_space.shape[0], env.action_space.n)
            if is_training:
                run_dir = make_run_dir(self.cfg.hyperparameter_set)
                self._set_run_dir(run_dir)
                print(f"[DuelingDDQN Train] Saving outputs to: {run_dir}")
                return self._train(env, policy_dqn, num_episodes, model_path)

            # Evaluation defaults to the checkpoint of the most recent training run
            if model_path is None:
                model_path = latest_model_path(self.cfg.hyperparameter_set)
            if save_gif and gif_path is None:
                gif_path = Path(model_path).parent / f"eval_{self.cfg.hyperparameter_set}.gif"
            return self._evaluate(
                env, policy_dqn, num_episodes, model_path, control_car, gif_path if save_gif else None
            )
        finally:
            env.close()

    def _train(
        self,
        env: gym.Env,
        policy_dqn: DQN,
        num_episodes: int | None,
        model_path: str | Path | None,
    ) -> dict[str, float]:
        """Off-policy training loop: act epsilon-greedily, store transitions, learn from replay."""
        loaded_ckpt = False
        if model_path is None:
            print("[DuelingDDQN Train] Training from scratch; no model checkpoint specified.")
        elif self._load_checkpoint(policy_dqn, str(model_path)):
            loaded_ckpt = True
            print(f"[DuelingDDQN Train] Loaded model weights from: {model_path}")
        else:
            print(f"[DuelingDDQN Train] Warning: Checkpoint '{model_path}' not found; training from scratch.")

        memory = ReplayMemory(maxlen=self.cfg.buffer_size)
        epsilon = self.cfg.epsilon_finetune if loaded_ckpt else self.cfg.epsilon_init

        target_dqn = self._build_network(env.observation_space.shape[0], env.action_space.n)
        self._sync_target_network(policy_dqn, target_dqn)
        target_dqn.eval()

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.cfg.learning_rate)

        logger = TrainingLogger(self.LOG_FILE, self.METRICS_FILE, extra_columns=("epsilon", "td_loss"))
        epsilon_history = []
        loss_history = []
        step_count = 0
        current_loss = 0.0

        def save_graph(dpi: int = 150):
            self.save_graph(
                logger.rewards,
                epsilon_history,
                steps_per_episode=logger.steps,
                goals_per_episode=logger.goals,
                loss_history=loss_history,
                dpi=dpi,
            )

        print(
            f"[DuelingDDQN Train] Device={device} | Dueling={self.cfg.enable_dueling_dqn} | "
            f"Double={self.cfg.enable_double_dqn} | Bot={self.cfg.bot_type} | Model={self.MODEL_FILE}"
        )

        episode_iter = range(num_episodes) if num_episodes is not None else itertools.count()
        try:
            for episode in episode_iter:
                state, _ = env.reset()
                state = obs_to_tensor(state)

                if not env.unwrapped.isopen:
                    break

                terminated = False
                truncated = False
                episode_reward = 0.0
                goal = 0

                while not (terminated or truncated):
                    action = self._select_action(env, policy_dqn, state, epsilon)
                    next_state, reward, terminated, truncated, info = env.step(int(action.item()))

                    episode_reward += reward
                    if scored_goal(info, terminated, reward):
                        goal = 1

                    new_state = obs_to_tensor(next_state)
                    self._store_transition(memory, state, action, new_state, reward, terminated, truncated)
                    step_count += 1

                    # Perform gradient update every `train_freq` decision steps inside the episode
                    loss = self._maybe_optimize(memory, step_count, policy_dqn, target_dqn)
                    if loss is not None:
                        current_loss = loss

                    # Sync target network every `network_sync_rate` decision steps
                    if step_count % self.cfg.target_update_interval == 0:
                        self._sync_target_network(policy_dqn, target_dqn)

                    state = new_state

                    if not env.unwrapped.isopen:
                        break

                mean_reward_100, goal_rate_100 = logger.record_episode(episode_reward, goal)
                print(
                    f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
                    f"Mean(100) = {mean_reward_100:7.2f} | GoalRate(100) = {goal_rate_100:5.1f}% | "
                    f"Eps = {epsilon:.3f} | Steps = {step_count}"
                )

                epsilon = self._decay_epsilon(epsilon)
                epsilon_history.append(epsilon)
                loss_history.append(current_loss)
                logger.write_metrics(episode, step_count, extras=(epsilon, current_loss))

                # Save best model based on 100-episode rolling mean reward
                if logger.is_new_best(episode, num_episodes):
                    logger.log_new_best(episode, step_count)
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                # Periodic snapshot of the current model, independent of the best-model file
                save_periodic_checkpoint(
                    policy_dqn, self.CHECKPOINT_DIR, self.cfg.hyperparameter_set, episode, self.cfg.checkpoint_interval
                )

                if logger.graph_due():
                    save_graph(dpi=100)

                if not env.unwrapped.isopen:
                    print("Exiting environment.")
                    break
        finally:
            # Also runs on Ctrl-C, so an interrupted training run still gets its final dashboard
            if logger.rewards:
                save_graph()

        return summarize_episodes(env, logger.rewards, logger.goals)

    def _evaluate(
        self,
        env: gym.Env,
        policy_dqn: DQN,
        num_episodes: int | None,
        model_path: str | Path,
        control_car: str,
        gif_path: str | Path | None,
    ) -> dict[str, float]:
        """Greedy evaluation; with `control_car="both"` the same network also drives the orange car."""
        ckpt_file = str(model_path)
        if self._load_checkpoint(policy_dqn, ckpt_file):
            print(f"[DuelingDDQN Eval] Loaded trained model weights from: {ckpt_file}")
        else:
            print(f"[DuelingDDQN Eval] Warning: Checkpoint '{ckpt_file}' not found; evaluating initial weights.")
        policy_dqn.eval()

        def select(state: torch.Tensor) -> int:
            return greedy_action(policy_dqn, state)

        return run_eval_episodes(
            env,
            select_blue=select,
            select_orange=select if control_car == "both" else None,
            num_episodes=num_episodes,
            gif_path=gif_path,
            gif_tag="DQN",
        )


def _resolve_config(
    config: DQNConfig | None,
    enable_dueling_dqn: bool | None,
    enable_double_dqn: bool | None,
    hyperparameter_set: str | None,
) -> DQNConfig:
    """`config` (default `DQNConfig()`) with any explicitly given variant settings applied."""
    overrides = {
        "enable_dueling_dqn": enable_dueling_dqn,
        "enable_double_dqn": enable_double_dqn,
        "hyperparameter_set": hyperparameter_set,
    }
    return replace(config or DQNConfig(), **{k: v for k, v in overrides.items() if v is not None})


def train_dueling_ddqn(
    num_episodes: int | None = None,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
    enable_dueling_dqn: bool | None = None,
    enable_double_dqn: bool | None = None,
    hyperparameter_set: str | None = None,
    config: DQNConfig | None = None,
) -> dict[str, float]:
    """Train the Dueling Double DQN agent (or a plain / Double DQN variant via the flags)."""
    agent = DuelingDDQNAgent(_resolve_config(config, enable_dueling_dqn, enable_double_dqn, hyperparameter_set))
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
    enable_dueling_dqn: bool | None = None,
    enable_double_dqn: bool | None = None,
    hyperparameter_set: str | None = None,
    save_gif: bool = False,
    gif_path: str | Path | None = None,
    control_car: str = "blue",
    random_spawn: bool = True,
    config: DQNConfig | None = None,
) -> dict[str, float]:
    """Evaluate/validate a trained Dueling Double DQN agent."""
    agent = DuelingDDQNAgent(_resolve_config(config, enable_dueling_dqn, enable_double_dqn, hyperparameter_set))
    return agent.run(
        is_training=False,
        render=render,
        max_episode_steps=max_episode_steps,
        num_episodes=resolve_eval_episodes(num_episodes, render, save_gif, agent.cfg.eval_episodes),
        model_path=model_path,
        save_gif=save_gif,
        gif_path=gif_path,
        control_car=control_car,
        random_spawn=random_spawn,
    )
