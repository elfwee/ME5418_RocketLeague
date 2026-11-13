"""
Proximal Policy Optimization (PPO) Learning Agent: the PPO learning rule plus its `train` / `evaluate` entry points.

Adapted from CleanRL's `ppo.py` (https://github.com/vwxyzjn/cleanrl) for a controlled
comparison with `DuelingDDQNAgent` (`dueling_ddqn_agent.py`):
- Identical 150,949-parameter `LayerNorm + ReLU` two-stream network (`ActorCritic` in `src/neural_nets/ppo_torch.py`).
- Identical `action_repeat = 4` with edge-triggered jump release (`ActionRepeat` env wrapper).
- Identical core hyperparameters (`lr = 3e-4`, `gamma = 0.99`).
- On-policy rollout (`T = 4096`), reverse-time GAE (`lambda = 0.95`), batch-level advantage normalization,
  and `6` epochs of shuffled `256`-sample minibatch clipped surrogate updates.
- Identical episode loop, logging, checkpointing and 100-episode rolling metrics (`trainer.py`); each
  training run writes its outputs to its own `runs/YYYYMMDD_hhmm_ppo/` folder (`model_ppo.pt`, `graph_ppo.png`, ...).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.config.gym_config import GymConfig
from src.config.nn_config import PPOConfig
from src.learning_agents.dashboard import DashboardPanel
from src.learning_agents.env_factory import device, make_env, normalize_control_car
from src.learning_agents.eval_loop import resolve_eval_episodes, run_eval_episodes
from src.learning_agents.rollout_buffer import RolloutBuffer
from src.learning_agents.run_dirs import RunPaths, latest_model_path
from src.learning_agents.trainer import run_training
from src.neural_nets.checkpoint import infer_spec, load_state_dict
from src.neural_nets.policy import greedy_action
from src.neural_nets.ppo_torch import ActorCritic


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (GAE) with truncation-aware bootstrapping.

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}

    Args:
        rewards: Shape (T,), step rewards.
        values: Shape (T,), critic estimates V(s_t).
        next_values: Shape (T,), bootstrap values V(s_{t+1}) (already 0.0 where `terminated=True`,
                     and `V(s_timeout)` where `truncated=True`).
        dones: Shape (T,), 1.0 where episode ended (`terminated or truncated`), resetting the GAE trace.

    Returns:
        (advantages, returns), each of shape (T,).
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] - values[t]
        last_gae = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class PPOAgent:
    """Proximal Policy Optimization (PPO) Learning Agent supporting training and validation/evaluation.

    Key algorithmic components:
        1. Actor-Critic Architecture (`ActorCritic` from `src.neural_nets.ppo_torch`):
           Shared 2-layer `LayerNorm + ReLU` feature backbone (256 units) with 128-unit
           Actor (`pi(a|s)`) and Critic (`V(s)`) heads (150,949 parameters, matching Dueling DDQN).
        2. Action Repeat with Edge-Triggered Jump Release (`ActionRepeat` env wrapper):
           Repeats chosen discrete action for `action_repeat` frames while clearing the jump
           bit on repeat frames > 0 so `Car._apply_jump` edge-triggering resets cleanly.
        3. Truncation-Aware Generalized Advantage Estimation (`_compute_gae`):
           Terminal goal states (`terminated=True`) bootstrap to 0.0, whereas timeout states
           (`truncated=True`) bootstrap from the critic's value estimate `V(s_timeout)` before reset.
        4. Clipped Surrogate Objective (`optimize`):
           Normalizes advantages across the entire 4,096-step rollout batch and performs `update_epochs` (6)
           of shuffled 256-sample minibatch updates with ratio clipping `clip_coef`, value MSE loss,
           entropy bonus, and global gradient norm clipping.

    One training decision step (driven by `trainer.run_training`):
        `act` (`_sample_action`) -> `env.step` (`ActionRepeat`) -> `observe` (`_bootstrap_value`
        -> `RolloutBuffer.add` -> `_update_from_rollout` every `rollout_steps` steps: `_compute_gae`
        then `optimize`).
    """

    metric_columns = ("entropy", "value_loss")

    def __init__(self, config: PPOConfig | None = None):
        self.cfg = config or PPOConfig()
        self.optimizer = None

        # Training state, set up by `train`
        self.model: ActorCritic | None = None
        self.buffer = RolloutBuffer()
        self.current_entropy = 0.0
        self.current_v_loss = 0.0
        self._pending: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None  # (action, log pi, V(s))

    @property
    def dashboard_title(self) -> str:
        return f"PPO Training Dashboard ({self.cfg.hyperparameter_set})"

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
        """Truncation-aware GAE over the rollout (see `compute_gae`)."""
        return compute_gae(rewards, values, next_values, dones, self.cfg.discount_factor, self.cfg.gae_lambda)

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

        for _epoch in range(self.cfg.update_epochs):
            perm = torch.randperm(batch_size, device=device)
            epoch_kl_sum = 0.0
            epoch_mb_count = 0

            for start in range(0, batch_size, self.cfg.minibatch_size):
                idx = perm[start : start + self.cfg.minibatch_size]
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
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = F.mse_loss(new_values, mb_returns)
                entropy_loss = entropy.mean()

                loss = pg_loss + self.cfg.value_loss_coef * v_loss - self.cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=self.cfg.max_grad_norm)
                self.optimizer.step()

                total_loss_sum += float(loss.item())
                entropy_sum += float(entropy_loss.item())
                value_loss_sum += float(v_loss.item())
                num_mb += 1

            # Early stop if average KL divergence across this epoch exceeded target threshold
            mean_epoch_kl = epoch_kl_sum / max(epoch_mb_count, 1)
            if self.cfg.target_kl is not None and mean_epoch_kl > self.cfg.target_kl:
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
        return greedy_action(policy_net, state)

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
    # Trainer hooks (see `trainer.Learner`)
    # ----------------------------------------------------------------------------------------

    def act(self, state: torch.Tensor) -> int | list[int]:
        action, log_prob, value = self._sample_action(self.model, state)
        # Backfill the previous step's next_value using this step's already-computed value
        self.buffer.backfill_next_value(value)
        self._pending = (action, log_prob, value)
        return action.cpu().numpy().tolist() if self.model.is_factored else int(action.item())

    def observe(self, state: torch.Tensor, reward: float, next_state: torch.Tensor, terminated: bool, truncated: bool):
        action, log_prob, value = self._pending
        next_value = self._bootstrap_value(self.model, next_state, terminated, truncated)
        self.buffer.add(state, action, log_prob, value, next_value, reward, terminated or truncated)

        # Perform PPO multi-epoch minibatch update once rollout horizon is reached
        if len(self.buffer) >= self.cfg.rollout_steps:
            self.current_entropy, self.current_v_loss = self._update_from_rollout(self.buffer, self.model, next_state)

    def progress(self) -> str:
        return f"Ent = {self.current_entropy:.3f}"

    def end_episode(self) -> tuple[float, float]:
        return self.current_entropy, self.current_v_loss

    def dashboard_panels(self, history: tuple[list[float], list[float]]) -> tuple[DashboardPanel, DashboardPanel]:
        """Exploration Dynamics (policy entropy) and Critic Optimization Health (value loss)."""
        entropy_history, value_loss_history = history
        return (
            DashboardPanel(
                values=entropy_history,
                title="Exploration Dynamics (Policy Entropy)",
                ylabel="Entropy H(pi)",
                label="Policy Entropy H(pi)",
            ),
            DashboardPanel(
                values=value_loss_history,
                title="Critic Optimization Health (Value Loss)",
                ylabel="Value MSE Loss",
                label="Raw Value Loss",
            ),
        )

    # ----------------------------------------------------------------------------------------
    # Training / evaluation loops
    # ----------------------------------------------------------------------------------------

    def _resolve_architecture(self, state_dict: dict | None) -> tuple[bool, int, int | list[int] | None]:
        """(factored, hidden_dim, action_dim) from a checkpoint if given, else from `PPOConfig`.
        `action_dim` is None when it should be taken from the env's action space."""
        if state_dict is None:
            return self.cfg.factored_action, self.cfg.hidden_size, None
        spec = infer_spec(state_dict)
        factored = spec.factored if spec.action_dim is not None else self.cfg.factored_action
        return factored, spec.hidden_dim, spec.action_dim

    def _build(
        self,
        state_dict: dict | None,
        control_car: str,
        render: bool,
        save_gif: bool,
        random_spawn: bool,
        max_episode_steps: int | None,
    ):
        """The env (flat or factored actions to match the network) and an ActorCritic, loaded if `state_dict`."""
        factored, hidden_dim, action_dim = self._resolve_architecture(state_dict)
        env = make_env(
            bot_type=self.cfg.bot_type,
            discrete_action=not factored,
            control_car=control_car,
            render=render,
            save_gif=save_gif,
            random_spawn=random_spawn,
            max_episode_steps=max_episode_steps,
            env_id=GymConfig.gym_id,
            action_repeat=self.cfg.action_repeat,
        )
        if action_dim is None:
            action_dim = list(env.action_space.nvec) if factored else env.action_space.n
        policy_net = ActorCritic(env.observation_space.shape[0], action_dim, hidden_dim).to(device)
        if state_dict is not None:
            policy_net.load_state_dict(state_dict)
        return env, policy_net, action_dim

    def train(
        self,
        num_episodes: int | None = None,
        model_path: str | Path | None = None,
        max_episode_steps: int | None = None,
        random_spawn: bool = True,
    ) -> dict[str, float]:
        """On-policy training: collect `rollout_steps` decision steps, then update on them.

        `model_path` fine-tunes from a checkpoint. Outputs go to a new
        `runs/YYYYMMDD_hhmm_<hyperparameter_set>/` folder.
        """
        # Training only reads a checkpoint when one is given (fine-tuning)
        state_dict = load_state_dict(str(model_path), device) if model_path is not None else None
        env, self.model, action_dim = self._build(state_dict, "blue", False, False, random_spawn, max_episode_steps)
        try:
            if model_path is None:
                print("[PPO Train] Training from scratch; no model checkpoint specified.")
            elif state_dict is not None:
                print(f"[PPO Train] Loaded model weights from: {model_path}")
            else:
                print(f"[PPO Train] Warning: Checkpoint '{model_path}' not found; training from scratch.")
            paths = RunPaths.new(self.cfg.hyperparameter_set)
            print(f"[PPO Train] Saving outputs to: {paths.run_dir}")

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, eps=1e-5)
            self.buffer = RolloutBuffer()
            self.current_v_loss = 0.0
            # Until the first update, report the entropy of a uniform policy: sum over heads of log(n)
            self.current_entropy = (
                float(sum(np.log(d) for d in action_dim)) if isinstance(action_dim, list) else float(np.log(action_dim))
            )

            print(
                f"[PPO Train] Device={device} | Rollout={self.cfg.rollout_steps} | Epochs={self.cfg.update_epochs} | "
                f"Minibatch={self.cfg.minibatch_size} | Bot={self.cfg.bot_type} | Model={paths.model_file}"
            )
            return run_training(env, self, paths, num_episodes, self.cfg.checkpoint_interval)
        finally:
            env.close()

    def evaluate(
        self,
        num_episodes: int | None = None,
        model_path: str | Path | None = None,
        render: bool = False,
        save_gif: bool = False,
        gif_path: str | Path | None = None,
        control_car: str = "blue",
        random_spawn: bool = True,
        max_episode_steps: int | None = None,
    ) -> dict[str, float]:
        """Greedy evaluation; with `control_car="both"` the same network also drives the orange car.

        Defaults to the checkpoint of the most recent training run.
        """
        control_car = normalize_control_car(control_car)
        ckpt_file = str(model_path) if model_path is not None else str(latest_model_path(self.cfg.hyperparameter_set))
        state_dict = load_state_dict(ckpt_file, device)
        env, policy_net, _ = self._build(state_dict, control_car, render, save_gif, random_spawn, max_episode_steps)
        try:
            if state_dict is not None:
                print(f"[PPO Eval] Loaded trained model weights from: {ckpt_file}")
            else:
                print(f"[PPO Eval] Warning: Checkpoint '{ckpt_file}' not found; evaluating initial weights.")
            if save_gif and gif_path is None:
                gif_path = Path(ckpt_file).parent / f"eval_{self.cfg.hyperparameter_set}.gif"
            policy_net.eval()

            def select(state: torch.Tensor) -> int | list[int]:
                return self._greedy_action(policy_net, state)

            return run_eval_episodes(
                env,
                select_blue=select,
                select_orange=select if control_car == "both" else None,
                num_episodes=num_episodes,
                gif_path=gif_path if save_gif else None,
                gif_tag="PPO",
            )
        finally:
            env.close()


def _resolve_config(config: PPOConfig | None, hyperparameter_set: str | None) -> PPOConfig:
    """`config` (default `PPOConfig()`) with the run name overridden when one is given."""
    config = config or PPOConfig()
    return replace(config, hyperparameter_set=hyperparameter_set) if hyperparameter_set is not None else config


def train_ppo(
    num_episodes: int | None = None,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
    hyperparameter_set: str | None = None,
    config: PPOConfig | None = None,
) -> dict[str, float]:
    """Train the PPO agent."""
    agent = PPOAgent(_resolve_config(config, hyperparameter_set))
    return agent.train(num_episodes=num_episodes, model_path=model_path, max_episode_steps=max_episode_steps)


def evaluate_ppo(
    num_episodes: int | None = PPOConfig.eval_episodes,
    render: bool = False,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
    hyperparameter_set: str | None = None,
    save_gif: bool = False,
    gif_path: str | Path | None = None,
    control_car: str = "blue",
    random_spawn: bool = True,
    config: PPOConfig | None = None,
) -> dict[str, float]:
    """Evaluate/validate a trained PPO agent."""
    agent = PPOAgent(_resolve_config(config, hyperparameter_set))
    return agent.evaluate(
        render=render,
        max_episode_steps=max_episode_steps,
        num_episodes=resolve_eval_episodes(num_episodes, render, save_gif, agent.cfg.eval_episodes),
        model_path=model_path,
        save_gif=save_gif,
        gif_path=gif_path,
        control_car=control_car,
        random_spawn=random_spawn,
    )
