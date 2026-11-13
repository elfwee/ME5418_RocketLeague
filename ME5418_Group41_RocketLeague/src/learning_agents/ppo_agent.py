"""
Proximal Policy Optimization (PPO) Learning Agent — Training & Validation/Evaluation Pipeline.

Adapted from CleanRL's `ppo.py` (https://github.com/vwxyzjn/cleanrl) for a controlled
comparison with `DuelingDDQNAgent` (`dueling_ddqn_agent.py`):
- Identical 150,949-parameter `LayerNorm + ReLU` two-stream network (`ActorCritic` in `src/neural_nets/ppo_torch.py`).
- Identical `action_repeat = 4` with edge-triggered jump release (`step_with_action_repeat`).
- Identical core hyperparameters (`lr = 3e-4`, `gamma = 0.99`).
- On-policy rollout (`T = 4096`), reverse-time GAE (`lambda = 0.95`), batch-level advantage normalization,
  and `6` epochs of shuffled `256`-sample minibatch clipped surrogate updates.
- Identical logging, checkpointing and 100-episode rolling metrics; each training run writes its
  outputs to its own `runs/YYYYMMDD_hhmm_ppo/` folder (`model_ppo.pt`, `graph_ppo.png`, ...).

This file holds the PPO learning rule and its training loop. Environment creation, action repeat,
the evaluation loop, logging and plotting are shared with DQN in `agent_common.py`.
"""

import itertools
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.config.gym_config import GymConfig
from src.config.nn_config import PPOConfig
from src.learning_agents.agent_common import (
    DashboardPanel,
    TrainingLogger,
    device,
    infer_ppo_arch,
    latest_model_path,
    load_state_dict_if_exists,
    make_env,
    make_run_dir,
    normalize_control_car,
    obs_to_tensor,
    resolve_eval_episodes,
    run_eval_episodes,
    save_periodic_checkpoint,
    save_training_dashboard,
    scored_goal,
    step_with_action_repeat,
    summarize_episodes,
)
from src.neural_nets.ppo_torch import ActorCritic


class RolloutBuffer:
    """On-policy storage for one rollout of decision steps.

    `next_value` is the bootstrap target V(s_{t+1}) for each step. For a step that does not end an
    episode it is left as None and filled in by `backfill_next_value` once the next step's forward
    pass has produced V(s_{t+1}), which saves one critic evaluation per step.
    """

    def __init__(self):
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.log_probs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.next_values: list[torch.Tensor | None] = []
        self.rewards: list[torch.Tensor] = []
        self.dones: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.states)

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        next_value: torch.Tensor | None,
        reward: float,
        done: bool,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.next_values.append(next_value)
        self.rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        self.dones.append(torch.tensor(float(done), dtype=torch.float32, device=device))

    def needs_next_value(self) -> bool:
        """True when the most recent step is still waiting for its bootstrap value."""
        return len(self.next_values) > 0 and self.next_values[-1] is None

    def backfill_next_value(self, value: torch.Tensor):
        """Give the most recent step its V(s_{t+1}), if it is still waiting for one."""
        if self.needs_next_value():
            self.next_values[-1] = value

    def as_tensors(self) -> tuple[torch.Tensor, ...]:
        """(states, actions, log_probs, values, next_values, rewards, dones), each stacked over time."""
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.log_probs),
            torch.stack(self.values),
            torch.stack(self.next_values),
            torch.stack(self.rewards),
            torch.stack(self.dones),
        )

    def clear(self):
        for buf in (
            self.states,
            self.actions,
            self.log_probs,
            self.values,
            self.next_values,
            self.rewards,
            self.dones,
        ):
            buf.clear()


class PPOAgent:
    """Proximal Policy Optimization (PPO) Learning Agent supporting training and validation/evaluation.

    Key algorithmic components:
        1. Actor-Critic Architecture (`ActorCritic` from `src.neural_nets.ppo_torch`):
           Shared 2-layer `LayerNorm + ReLU` feature backbone (256 units) with 128-unit
           Actor (`pi(a|s)`) and Critic (`V(s)`) heads (150,949 parameters, matching Dueling DDQN).
        2. Action Repeat with Edge-Triggered Jump Release (`step_with_action_repeat`):
           Repeats chosen discrete action for `action_repeat` frames while clearing the jump
           bit on repeat frames > 0 so `Car._apply_jump` edge-triggering resets cleanly.
        3. Truncation-Aware Generalized Advantage Estimation (`_compute_gae`):
           Terminal goal states (`terminated=True`) bootstrap to 0.0, whereas timeout states
           (`truncated=True`) bootstrap from the critic's value estimate `V(s_timeout)` before reset.
        4. Clipped Surrogate Objective (`optimize`):
           Normalizes advantages across the entire 4,096-step rollout batch and performs `update_epochs` (6)
           of shuffled 256-sample minibatch updates with ratio clipping `clip_coef`, value MSE loss,
           entropy bonus, and global gradient norm clipping.

    One training decision step (`_train`):
        `_sample_action` -> `step_with_action_repeat` -> `_bootstrap_value` -> `RolloutBuffer.add`
        -> `_update_from_rollout` (every `rollout_steps` steps: `_compute_gae` then `optimize`).
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

        self.checkpoint_interval = PPOConfig.checkpoint_interval

        self.optimizer = None

    def _set_run_dir(self, run_dir: str | Path):
        """Point every training output (log, checkpoint, dashboard, metrics CSV) at `run_dir`."""
        self.LOG_FILE = os.path.join(run_dir, f"run_{self.hyperparameter_set}.log")
        self.MODEL_FILE = os.path.join(run_dir, f"model_{self.hyperparameter_set}.pt")
        self.GRAPH_FILE = os.path.join(run_dir, f"graph_{self.hyperparameter_set}.png")
        self.METRICS_FILE = os.path.join(run_dir, f"metrics_{self.hyperparameter_set}.csv")
        self.CHECKPOINT_DIR = os.path.join(run_dir, "checkpoints")

    # ----------------------------------------------------------------------------------------
    # Learning rule
    # ----------------------------------------------------------------------------------------

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

        for _epoch in range(self.update_epochs):
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

    # ----------------------------------------------------------------------------------------
    # Algorithm steps
    # ----------------------------------------------------------------------------------------

    def _sample_action(
        self, policy_net: ActorCritic, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a ~ pi(.|s); returns (action, log pi(a|s), V(s)) for a single state."""
        with torch.no_grad():
            action, log_prob, _, value = policy_net.get_action_and_value(state.unsqueeze(0))
        return action.squeeze(0), log_prob.squeeze(0), value.squeeze(0)

    def _greedy_action(self, policy_net: ActorCritic, state: torch.Tensor) -> int | list[int]:
        """Most likely action as an env action: a Discrete index, or [direction, jump, boost] when factored."""
        with torch.no_grad():
            logits, _ = policy_net(state.unsqueeze(0))
        if policy_net.is_factored:
            return [int(l.squeeze(0).argmax().item()) for l in logits]
        return int(logits.squeeze(0).argmax().item())

    def _state_value(self, policy_net: ActorCritic, state: torch.Tensor) -> torch.Tensor:
        """Critic estimate V(s) for a single state."""
        with torch.no_grad():
            _, value = policy_net(state.unsqueeze(0))
        return value.squeeze(0)

    def _bootstrap_value(
        self, policy_net: ActorCritic, new_state: torch.Tensor, terminated: bool, truncated: bool
    ) -> torch.Tensor | None:
        """Truncation-aware bootstrap target V(s_{t+1}) for the step that just finished:
        - Terminal goal state (`terminated=True`) -> 0.0
        - Timeout state (`truncated=True`) -> evaluate V(new_state) once before reset
        - Non-terminal -> None; filled in from the next step's forward pass (see `RolloutBuffer`)
        """
        if terminated:
            return torch.tensor(0.0, dtype=torch.float32, device=device)
        if truncated:
            return self._state_value(policy_net, new_state)
        return None

    def _update_from_rollout(
        self, buffer: RolloutBuffer, policy_net: ActorCritic, last_state: torch.Tensor
    ) -> tuple[float, float]:
        """GAE + multi-epoch minibatch PPO update on a full rollout, then empty the buffer.
        Returns (mean_entropy, mean_value_loss)."""
        # If the rollout cut off mid-episode, bootstrap the tail step's next_value
        if buffer.needs_next_value():
            buffer.backfill_next_value(self._state_value(policy_net, last_state))

        states, actions, old_log_probs, values, next_values, rewards, dones = buffer.as_tensors()
        advantages, returns = self._compute_gae(rewards, values, next_values, dones)
        _, entropy, value_loss, _ = self.optimize(states, actions, old_log_probs, returns, advantages, policy_net)
        buffer.clear()
        return entropy, value_loss

    # ----------------------------------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------------------------------

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
        save_training_dashboard(
            self.GRAPH_FILE,
            f"PPO Training Dashboard ({self.hyperparameter_set})",
            rewards_per_episode,
            steps_per_episode,
            goals_per_episode,
            exploration=DashboardPanel(
                values=entropy_history,
                title="Exploration Dynamics (Policy Entropy)",
                ylabel="Entropy H(pi)",
                label="Policy Entropy H(pi)",
            ),
            loss=DashboardPanel(
                values=value_loss_history,
                title="Critic Optimization Health (Value Loss)",
                ylabel="Value MSE Loss",
                label="Raw Value Loss",
            ),
            dpi=dpi,
        )

    # ----------------------------------------------------------------------------------------
    # Training / evaluation loops
    # ----------------------------------------------------------------------------------------

    def _resolve_architecture(self, state_dict: dict | None) -> tuple[bool, int, int | list[int] | None]:
        """(factored, hidden_dim, action_dim) from a checkpoint if given, else from `PPOConfig`.
        `action_dim` is None when it should be taken from the env's action space."""
        factored = getattr(PPOConfig, "factored_action", False)
        if state_dict is None:
            return factored, self.fcl_nodes, None
        hidden_dim, action_dim = infer_ppo_arch(state_dict)
        if action_dim is not None:
            factored = isinstance(action_dim, list)
        return factored, hidden_dim, action_dim

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
        control_car = normalize_control_car(control_car)

        # Training only reads a checkpoint when one is given (fine-tuning); evaluation always tries
        # to, defaulting to the checkpoint of the most recent training run
        ckpt_file = str(model_path) if model_path is not None else str(latest_model_path(self.hyperparameter_set))
        state_dict = load_state_dict_if_exists(ckpt_file) if (not is_training or model_path is not None) else None
        factored, hidden_dim, action_dim = self._resolve_architecture(state_dict)

        env = make_env(
            bot_type=self.bot_type,
            discrete_action=not factored,
            control_car="blue" if is_training else control_car,
            render=render,
            save_gif=save_gif,
            random_spawn=random_spawn,
            max_episode_steps=max_episode_steps,
            env_id=self.env_id,
        )
        try:
            if action_dim is None:
                action_dim = list(env.action_space.nvec) if factored else env.action_space.n
            policy_net = ActorCritic(env.observation_space.shape[0], action_dim, hidden_dim).to(device)
            if state_dict is not None:
                policy_net.load_state_dict(state_dict)

            if is_training:
                if model_path is None:
                    print("[PPO Train] Training from scratch; no model checkpoint specified.")
                elif state_dict is not None:
                    print(f"[PPO Train] Loaded model weights from: {ckpt_file}")
                else:
                    print(f"[PPO Train] Warning: Checkpoint '{ckpt_file}' not found; training from scratch.")
                run_dir = make_run_dir(self.hyperparameter_set)
                self._set_run_dir(run_dir)
                print(f"[PPO Train] Saving outputs to: {run_dir}")
                return self._train(env, policy_net, num_episodes, action_dim)

            if state_dict is not None:
                print(f"[PPO Eval] Loaded trained model weights from: {ckpt_file}")
            else:
                print(f"[PPO Eval] Warning: Checkpoint '{ckpt_file}' not found; evaluating initial weights.")
            if save_gif and gif_path is None:
                gif_path = Path(ckpt_file).parent / f"eval_{self.hyperparameter_set}.gif"
            return self._evaluate(env, policy_net, num_episodes, control_car, gif_path if save_gif else None)
        finally:
            env.close()

    def _train(
        self,
        env: gym.Env,
        policy_net: ActorCritic,
        num_episodes: int | None,
        action_dim: int | list[int],
    ) -> dict[str, float]:
        """On-policy training loop: collect `rollout_steps` decision steps, then update on them."""
        self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.learning_rate, eps=1e-5)

        logger = TrainingLogger(self.LOG_FILE, self.METRICS_FILE, extra_columns=("entropy", "value_loss"))
        buffer = RolloutBuffer()
        entropy_history = []
        value_loss_history = []
        step_count = 0
        current_v_loss = 0.0
        # Until the first update, report the entropy of a uniform policy: sum over heads of log(n)
        current_entropy = (
            float(sum(np.log(d) for d in action_dim)) if isinstance(action_dim, list) else float(np.log(action_dim))
        )

        def save_graph(dpi: int = 150):
            self.save_graph(
                logger.rewards,
                entropy_history,
                steps_per_episode=logger.steps,
                goals_per_episode=logger.goals,
                value_loss_history=value_loss_history,
                dpi=dpi,
            )

        print(
            f"[PPO Train] Device={device} | Rollout={self.rollout_steps} | Epochs={self.update_epochs} | "
            f"Minibatch={self.mini_batch_size} | Bot={self.bot_type} | Model={self.MODEL_FILE}"
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
                    action, log_prob, value = self._sample_action(policy_net, state)
                    # Backfill the previous step's next_value using this step's already-computed value
                    buffer.backfill_next_value(value)

                    action_for_env = action.cpu().numpy().tolist() if policy_net.is_factored else int(action.item())
                    next_state, reward, terminated, truncated, info = step_with_action_repeat(
                        env, action_for_env, action_repeat=self.action_repeat
                    )

                    episode_reward += reward
                    if scored_goal(info, terminated, reward):
                        goal = 1

                    new_state = obs_to_tensor(next_state)
                    next_value = self._bootstrap_value(policy_net, new_state, terminated, truncated)
                    buffer.add(state, action, log_prob, value, next_value, reward, terminated or truncated)
                    step_count += 1

                    # Perform PPO multi-epoch minibatch update once rollout horizon is reached
                    if len(buffer) >= self.rollout_steps:
                        current_entropy, current_v_loss = self._update_from_rollout(buffer, policy_net, new_state)

                    state = new_state

                    if not env.unwrapped.isopen:
                        break

                mean_reward_100, goal_rate_100 = logger.record_episode(episode_reward, goal)
                print(
                    f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
                    f"Mean(100) = {mean_reward_100:7.2f} | GoalRate(100) = {goal_rate_100:5.1f}% | "
                    f"Ent = {current_entropy:.3f} | Steps = {step_count}"
                )

                entropy_history.append(current_entropy)
                value_loss_history.append(current_v_loss)
                logger.write_metrics(episode, step_count, extras=(current_entropy, current_v_loss))

                # Save best model based on 100-episode rolling mean reward
                if logger.is_new_best(episode, num_episodes):
                    logger.log_new_best(episode, step_count)
                    torch.save(policy_net.state_dict(), self.MODEL_FILE)

                # Periodic snapshot of the current model, independent of the best-model file
                save_periodic_checkpoint(
                    policy_net, self.CHECKPOINT_DIR, self.hyperparameter_set, episode, self.checkpoint_interval
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
        policy_net: ActorCritic,
        num_episodes: int | None,
        control_car: str,
        gif_path: str | Path | None,
    ) -> dict[str, float]:
        """Greedy evaluation; with `control_car="both"` the same network also drives the orange car."""
        policy_net.eval()

        def select(state: torch.Tensor) -> int | list[int]:
            return self._greedy_action(policy_net, state)

        return run_eval_episodes(
            env,
            select_blue=select,
            select_orange=select if control_car == "both" else None,
            action_repeat=self.action_repeat,
            num_episodes=num_episodes,
            gif_path=gif_path,
            gif_tag="PPO",
        )


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
    return agent.run(
        is_training=False,
        render=render,
        max_episode_steps=max_episode_steps,
        num_episodes=resolve_eval_episodes(num_episodes, render, save_gif, PPOConfig.eval_episodes),
        model_path=model_path,
        save_gif=save_gif,
        gif_path=gif_path,
        control_car=control_car,
        random_spawn=random_spawn,
    )
