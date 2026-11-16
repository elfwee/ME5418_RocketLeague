"""
PPO v2 Learning Agent: separate-network Actor-Critic with factored MultiDiscrete heads and parallel envs.

Differs from `src/learning_agents/ppo_agent.py` in four ways:
1. Architecture (`SeparateActorCritic` in `src/neural_nets/ppo_v2_torch.py`): completely independent
   policy and value MLPs (no shared trunk parameters; 164,366 trainable parameters for width 256,
   NOT parameter-matched to Dueling DDQN's 150,949 parameters).
2. Action space: factored `MultiDiscrete([9, 2, 2])` heads (`pi_heads.0/1/2`).
3. Parallel rollouts: collects `rollout_steps` across `num_envs` vectorized environments in either
   `"subproc"` (one worker process per env) or `"sync"` mode via `step_with_autoreset`.
4. Periodic fixed-seed greedy evaluation: writes `eval_<name>.csv` in the run directory alongside
   standard `metrics_<name>.csv`, best/periodic checkpoints, and `graph_<name>.png`.
"""

import multiprocessing as mp
import os
import signal
from collections.abc import Callable
from dataclasses import replace
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.config.nn_config import PPOv2Config
from src.learning_agents.dashboard import metric_panel, save_training_dashboard
from src.learning_agents.env_factory import device, make_env, normalize_control_car, obs_to_tensor, scored_goal
from src.learning_agents.eval_loop import resolve_eval_episodes, run_eval_episodes
from src.learning_agents.ppo_agent import compute_gae
from src.learning_agents.run_dirs import RunPaths, latest_model_path, save_periodic_checkpoint
from src.learning_agents.training_logger import TrainingLogger
from src.neural_nets.checkpoint import infer_spec, load_state_dict
from src.neural_nets.policy import greedy_action
from src.neural_nets.ppo_v2_torch import SeparateActorCritic

EVAL_SEED_BASE = 10_000


def step_with_autoreset(env, action) -> tuple[np.ndarray, float, bool, bool, bool, np.ndarray | None]:
    """Step `env`, detect goal success, and auto-reset on episode end while keeping `final_obs`."""
    next_obs, reward, terminated, truncated, info = env.step(action)
    goal = scored_goal(info, bool(terminated), float(reward))
    if terminated or truncated:
        final_obs = np.asarray(next_obs, dtype=np.float32)
        next_obs, _ = env.reset()
    else:
        final_obs = None
    return (
        np.asarray(next_obs, dtype=np.float32),
        float(reward),
        bool(terminated),
        bool(truncated),
        goal,
        final_obs,
    )


def _worker_loop(
    conn: Connection,
    bot_type: str,
    random_spawn: bool,
    max_episode_steps: int | None,
    env_id: str,
    action_repeat: int,
    env_kwargs: dict | None = None,
):
    """Subprocess env worker: ignores SIGINT so parent cleans up workers cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    env = make_env(
        bot_type=bot_type,
        discrete_action=False,
        control_car="blue",
        render=False,
        save_gif=False,
        random_spawn=random_spawn,
        max_episode_steps=max_episode_steps,
        env_id=env_id,
        action_repeat=action_repeat,
        env_kwargs=env_kwargs,
    )
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                obs, _ = env.reset(seed=payload)
                conn.send(np.asarray(obs, dtype=np.float32))
            elif cmd == "step":
                conn.send(step_with_autoreset(env, payload))
            elif cmd == "close":
                conn.send(None)
                break
            else:
                raise ValueError(f"Unknown worker command: {cmd}")
    finally:
        env.close()
        conn.close()


class _SyncEnvPool:
    """In-process vectorized wrapper over a list of `num_envs` environments."""

    def __init__(
        self, cfg: PPOv2Config, random_spawn: bool, max_episode_steps: int | None, env_kwargs: dict | None = None
    ):
        self.envs = [
            make_env(
                bot_type=cfg.bot_type,
                discrete_action=False,
                control_car="blue",
                render=False,
                save_gif=False,
                random_spawn=random_spawn,
                max_episode_steps=max_episode_steps,
                env_id=cfg.env_id,
                action_repeat=cfg.action_repeat,
                env_kwargs=env_kwargs,
            )
            for _ in range(cfg.num_envs)
        ]

    def reset(self, base_seed: int | None = None) -> np.ndarray:
        obs_list = []
        for idx, env in enumerate(self.envs):
            seed = None if base_seed is None else base_seed + idx
            obs, _ = env.reset(seed=seed)
            obs_list.append(np.asarray(obs, dtype=np.float32))
        return np.stack(obs_list, axis=0)

    def step(self, actions: list[list[int]]) -> list[tuple[np.ndarray, float, bool, bool, bool, np.ndarray | None]]:
        return [step_with_autoreset(env, act) for env, act in zip(self.envs, actions, strict=True)]

    def close(self):
        for env in self.envs:
            env.close()
        self.envs.clear()


class _SubprocEnvPool:
    """Multiprocessing vectorized wrapper: one daemon process per environment."""

    def __init__(
        self, cfg: PPOv2Config, random_spawn: bool, max_episode_steps: int | None, env_kwargs: dict | None = None
    ):
        self._closed = False
        self._parent_conns: list[Connection] = []
        self._procs: list[mp.Process] = []
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        for _ in range(cfg.num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_loop,
                args=(
                    child_conn,
                    cfg.bot_type,
                    random_spawn,
                    max_episode_steps,
                    cfg.env_id,
                    cfg.action_repeat,
                    env_kwargs,
                ),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._procs.append(proc)

    def reset(self, base_seed: int | None = None) -> np.ndarray:
        for idx, conn in enumerate(self._parent_conns):
            seed = None if base_seed is None else base_seed + idx
            conn.send(("reset", seed))
        return np.stack([conn.recv() for conn in self._parent_conns], axis=0)

    def step(self, actions: list[list[int]]) -> list[tuple[np.ndarray, float, bool, bool, bool, np.ndarray | None]]:
        for conn, act in zip(self._parent_conns, actions, strict=True):
            conn.send(("step", act))
        return [conn.recv() for conn in self._parent_conns]

    def close(self):
        if self._closed:
            return
        self._closed = True
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for conn in self._parent_conns:
            try:
                conn.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                conn.close()
            except OSError:
                pass
        for proc in self._procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)


def _make_env_pool(
    cfg: PPOv2Config, random_spawn: bool, max_episode_steps: int | None, env_kwargs: dict | None = None
):
    if cfg.env_workers not in ("subproc", "sync"):
        raise ValueError(f"Invalid env_workers: '{cfg.env_workers}'. Must be 'subproc' or 'sync'.")
    if cfg.num_envs == 1 or cfg.env_workers == "sync":
        return _SyncEnvPool(cfg, random_spawn, max_episode_steps, env_kwargs=env_kwargs)
    return _SubprocEnvPool(cfg, random_spawn, max_episode_steps, env_kwargs=env_kwargs)


class PPOv2Agent:
    """Separate-network PPO Learning Agent with factored action heads and parallel rollouts."""

    metric_columns = ("entropy", "value_loss")

    def __init__(self, config: PPOv2Config | None = None):
        self.cfg = config or PPOv2Config()
        self.optimizer: torch.optim.Optimizer | None = None
        self.model: SeparateActorCritic | None = None
        self.current_entropy = float(sum(np.log(d) for d in self.cfg.action_dims))
        self.current_v_loss = 0.0

    @property
    def dashboard_title(self) -> str:
        return f"PPO v2 Training Dashboard ({self.cfg.hyperparameter_set})"

    def optimize(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        policy_net: SeparateActorCritic,
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

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    kl_val = float(approx_kl.item())
                    epoch_kl_sum += kl_val
                    kl_sum += kl_val
                    epoch_mb_count += 1

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

            mean_epoch_kl = epoch_kl_sum / max(epoch_mb_count, 1)
            if self.cfg.target_kl is not None and mean_epoch_kl > self.cfg.target_kl:
                break

        denom = max(num_mb, 1)
        return total_loss_sum / denom, entropy_sum / denom, value_loss_sum / denom, kl_sum / denom

    def _run_periodic_eval(
        self,
        eval_env,
        policy_net: SeparateActorCritic,
        paths: RunPaths,
        step_count: int,
    ) -> dict[str, float]:
        """Run `periodic_eval_episodes` greedy episodes on fixed seeds and append to `eval_<name>.csv`."""
        was_training = policy_net.training
        policy_net.eval()
        goals = 0
        concedes = 0
        timeouts = 0
        returns: list[float] = []
        lengths: list[int] = []
        try:
            for ep_idx in range(self.cfg.periodic_eval_episodes):
                obs, _ = eval_env.reset(seed=EVAL_SEED_BASE + ep_idx)
                terminated = False
                truncated = False
                ep_ret = 0.0
                ep_len = 0
                ep_goal = False
                while not (terminated or truncated):
                    act = greedy_action(policy_net, obs_to_tensor(obs))
                    obs, reward, terminated, truncated, info = eval_env.step(act)
                    ep_ret += float(reward)
                    ep_len += 1
                    if scored_goal(info, bool(terminated), float(reward)):
                        ep_goal = True
                if ep_goal:
                    goals += 1
                elif terminated:
                    concedes += 1
                else:
                    timeouts += 1
                returns.append(ep_ret)
                lengths.append(ep_len)
        finally:
            policy_net.train(was_training)

        n = max(len(returns), 1)
        goal_rate = (goals / n) * 100.0
        concede_rate = (concedes / n) * 100.0
        timeout_rate = (timeouts / n) * 100.0
        mean_return = float(np.mean(returns)) if returns else 0.0
        mean_length = float(np.mean(lengths)) if lengths else 0.0

        write_header = not paths.eval_file.exists() or os.path.getsize(paths.eval_file) == 0
        with open(paths.eval_file, "a", newline="", encoding="utf-8") as f:
            if write_header:
                f.write("step,goal_rate,concede_rate,timeout_rate,mean_return,mean_length\n")
            f.write(
                f"{step_count},{goal_rate:.2f},{concede_rate:.2f},{timeout_rate:.2f},"
                f"{mean_return:.4f},{mean_length:.2f}\n"
            )

        print(
            f"[PPO v2 Eval @ step {step_count}] Goal={goal_rate:.1f}% | Concede={concede_rate:.1f}% | "
            f"Timeout={timeout_rate:.1f}% | Return={mean_return:.2f} | Len={mean_length:.1f}"
        )
        return {
            "step": step_count,
            "goal_rate": goal_rate,
            "concede_rate": concede_rate,
            "timeout_rate": timeout_rate,
            "mean_return": mean_return,
            "mean_length": mean_length,
        }

    def train(
        self,
        num_episodes: int | None = None,
        model_path: str | Path | None = None,
        max_episode_steps: int | None = None,
        random_spawn: bool = True,
        stop_fn: Callable[[float], bool] | None = None,
        env_kwargs: dict | None = None,
        eval_env_id: str | None = None,
    ) -> dict[str, float]:
        """Parallel on-policy training across `num_envs` workers."""
        state_dict = load_state_dict(str(model_path), device) if model_path is not None else None
        spec_action_dims = list(self.cfg.action_dims)
        hidden_dim = self.cfg.hidden_size
        if state_dict is not None:
            spec = infer_spec(state_dict)
            if isinstance(spec.action_dim, list):
                spec_action_dims = spec.action_dim
            hidden_dim = spec.hidden_dim

        pool = _make_env_pool(self.cfg, random_spawn, max_episode_steps, env_kwargs=env_kwargs)
        eval_enabled = self.cfg.eval_interval_steps > 0 and self.cfg.periodic_eval_episodes > 0
        eval_env = (
            make_env(
                bot_type=self.cfg.bot_type,
                discrete_action=False,
                control_car="blue",
                render=False,
                save_gif=False,
                random_spawn=random_spawn,
                max_episode_steps=max_episode_steps,
                env_id=eval_env_id or self.cfg.env_id,
                action_repeat=self.cfg.action_repeat,
            )
            if eval_enabled
            else None
        )

        try:
            obs_np = pool.reset(base_seed=self.cfg.seed)
            obs_dim = int(obs_np.shape[1])
            self.model = SeparateActorCritic(obs_dim, spec_action_dims, hidden_dim).to(device)
            if model_path is None:
                print("[PPO v2 Train] Training from scratch; no model checkpoint specified.")
            elif state_dict is not None:
                self.model.load_state_dict(state_dict)
                print(f"[PPO v2 Train] Loaded model weights from: {model_path}")
            else:
                print(f"[PPO v2 Train] Warning: Checkpoint '{model_path}' not found; training from scratch.")

            paths = RunPaths.new(self.cfg.hyperparameter_set)
            print(f"[PPO v2 Train] Saving outputs to: {paths.run_dir}")
            print(
                f"[PPO v2 Train] Device={device} | Envs={self.cfg.num_envs} ({self.cfg.env_workers}) | "
                f"Rollout={self.cfg.rollout_steps}x{self.cfg.num_envs} | Epochs={self.cfg.update_epochs} | "
                f"Minibatch={self.cfg.minibatch_size} | Bot={self.cfg.bot_type} | Model={paths.model_file}"
            )

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate, eps=1e-5)
            self.current_entropy = float(sum(np.log(d) for d in spec_action_dims))
            self.current_v_loss = 0.0

            logger = TrainingLogger(str(paths.log_file), str(paths.metrics_file), extra_columns=self.metric_columns)
            history: tuple[list[float], list[float]] = ([], [])

            def save_graph(dpi: int = 150):
                exploration, loss = (metric_panel(c, v) for c, v in zip(self.metric_columns, history, strict=True))
                save_training_dashboard(
                    paths.graph_file,
                    self.dashboard_title,
                    logger.rewards,
                    logger.steps,
                    logger.goals,
                    exploration=exploration,
                    loss=loss,
                    dpi=dpi,
                )

            n_envs = self.cfg.num_envs
            T = self.cfg.rollout_steps
            ep_returns = np.zeros(n_envs, dtype=np.float64)
            ep_goals = np.zeros(n_envs, dtype=np.int64)
            total_steps = 0
            episodes_done = 0
            stopped_early = False
            next_eval_step = self.cfg.eval_interval_steps if eval_enabled else 0
            last_eval_step = -1

            obs_buf = torch.zeros((T, n_envs, obs_dim), dtype=torch.float32, device=device)
            act_buf = torch.zeros((T, n_envs, len(spec_action_dims)), dtype=torch.long, device=device)
            logp_buf = torch.zeros((T, n_envs), dtype=torch.float32, device=device)
            val_buf = torch.zeros((T, n_envs), dtype=torch.float32, device=device)
            rew_buf = torch.zeros((T, n_envs), dtype=torch.float32, device=device)
            done_buf = torch.zeros((T, n_envs), dtype=torch.float32, device=device)
            term_buf = torch.zeros((T, n_envs), dtype=torch.bool, device=device)
            trunc_buf = torch.zeros((T, n_envs), dtype=torch.bool, device=device)
            trunc_val_buf = torch.zeros((T, n_envs), dtype=torch.float32, device=device)

            try:
                while not stopped_early and (num_episodes is None or episodes_done < num_episodes):
                    steps_collected = 0
                    for t in range(T):
                        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
                        with torch.no_grad():
                            actions_t, logp_t, _, values_t = self.model.get_action_and_value(obs_t)

                        obs_buf[t] = obs_t
                        act_buf[t] = actions_t
                        logp_buf[t] = logp_t
                        val_buf[t] = values_t

                        results = pool.step(actions_t.cpu().numpy().tolist())
                        next_obs_list = []
                        steps_collected = t + 1
                        total_steps += n_envs

                        for env_idx, (next_obs, reward, terminated, truncated, goal, final_obs) in enumerate(results):
                            next_obs_list.append(next_obs)
                            rew_buf[t, env_idx] = reward * self.cfg.reward_scale
                            done = terminated or truncated
                            done_buf[t, env_idx] = float(done)
                            term_buf[t, env_idx] = terminated
                            if truncated and not terminated and final_obs is not None:
                                trunc_buf[t, env_idx] = True
                                with torch.no_grad():
                                    _, v_final = self.model(obs_to_tensor(final_obs).unsqueeze(0))
                                trunc_val_buf[t, env_idx] = v_final.squeeze(0)
                            else:
                                trunc_buf[t, env_idx] = False

                            ep_returns[env_idx] += reward
                            if goal:
                                ep_goals[env_idx] = 1

                            if done:
                                if not stopped_early and (num_episodes is None or episodes_done < num_episodes):
                                    episode_idx = episodes_done
                                    ep_ret = float(ep_returns[env_idx])
                                    ep_goal = int(ep_goals[env_idx])
                                    episodes_done += 1

                                    mean_r100, goal_r100 = logger.record_episode(ep_ret, ep_goal)
                                    print(
                                        f"Episode {episode_idx:4d}: Reward = {ep_ret:7.2f} | "
                                        f"Mean(100) = {mean_r100:7.2f} | GoalRate(100) = {goal_r100:5.1f}% | "
                                        f"Ent = {self.current_entropy:.3f} | Steps = {total_steps}"
                                    )
                                    extras = (self.current_entropy, self.current_v_loss)
                                    for values, value in zip(history, extras, strict=True):
                                        values.append(value)
                                    logger.write_metrics(episode_idx, total_steps, extras=extras)

                                    if logger.is_new_best(episode_idx, num_episodes):
                                        logger.log_new_best(episode_idx, total_steps)
                                        torch.save(self.model.state_dict(), paths.model_file)

                                    save_periodic_checkpoint(
                                        self.model,
                                        paths.checkpoint_dir,
                                        paths.model_name,
                                        episode_idx,
                                        self.cfg.checkpoint_interval,
                                    )
                                    if logger.graph_due():
                                        save_graph(dpi=100)

                                    if stop_fn is not None and stop_fn(ep_ret):
                                        stopped_early = True

                                ep_returns[env_idx] = 0.0
                                ep_goals[env_idx] = 0

                        obs_np = np.stack(next_obs_list, axis=0)

                        if eval_enabled and total_steps >= next_eval_step:
                            self._run_periodic_eval(eval_env, self.model, paths, total_steps)
                            last_eval_step = total_steps
                            while next_eval_step <= total_steps:
                                next_eval_step += self.cfg.eval_interval_steps

                        if stopped_early or (num_episodes is not None and episodes_done >= num_episodes):
                            break

                    # Compute bootstrap targets and run GAE + clipped surrogate PPO update
                    with torch.no_grad():
                        last_obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device)
                        _, last_val = self.model(last_obs_t)

                    s_slice = slice(0, steps_collected)
                    next_vals = torch.cat([val_buf[1:steps_collected], last_val.unsqueeze(0)], dim=0)
                    next_vals[term_buf[s_slice]] = 0.0
                    next_vals[trunc_buf[s_slice]] = trunc_val_buf[s_slice][trunc_buf[s_slice]]

                    adv, ret = compute_gae(
                        rew_buf[s_slice],
                        val_buf[s_slice],
                        next_vals,
                        done_buf[s_slice],
                        self.cfg.discount_factor,
                        self.cfg.gae_lambda,
                    )
                    _, ent, v_loss, _ = self.optimize(
                        obs_buf[s_slice].reshape(-1, obs_dim),
                        act_buf[s_slice].reshape(-1, len(spec_action_dims)),
                        logp_buf[s_slice].reshape(-1),
                        ret.reshape(-1),
                        adv.reshape(-1),
                        self.model,
                    )
                    self.current_entropy = ent
                    self.current_v_loss = v_loss / (self.cfg.reward_scale**2)

                if eval_enabled and total_steps > 0 and total_steps != last_eval_step:
                    self._run_periodic_eval(eval_env, self.model, paths, total_steps)
            finally:
                if logger.rewards:
                    save_graph()

            total_goals = int(sum(logger.goals))
            return {
                "episodes": len(logger.rewards),
                "mean_reward": float(np.mean(logger.rewards)) if logger.rewards else 0.0,
                "goal_rate": float(np.mean(logger.goals)) * 100.0 if logger.goals else 0.0,
                "score_blue": total_goals,
                "score_orange": len(logger.goals) - total_goals,
            }
        finally:
            pool.close()
            if eval_env is not None:
                eval_env.close()

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
        """Greedy evaluation; with `control_car="both"` the same network also drives the orange car."""
        control_car = normalize_control_car(control_car)
        ckpt_file = str(model_path) if model_path is not None else str(latest_model_path(self.cfg.hyperparameter_set))
        state_dict = load_state_dict(ckpt_file, device)

        action_dims = list(self.cfg.action_dims)
        hidden_dim = self.cfg.hidden_size
        if state_dict is not None:
            spec = infer_spec(state_dict)
            if isinstance(spec.action_dim, list):
                action_dims = spec.action_dim
            hidden_dim = spec.hidden_dim

        env = make_env(
            bot_type=self.cfg.bot_type,
            discrete_action=False,
            control_car=control_car,
            render=render,
            save_gif=save_gif,
            random_spawn=random_spawn,
            max_episode_steps=max_episode_steps,
            env_id=self.cfg.env_id,
            action_repeat=self.cfg.action_repeat,
        )
        try:
            policy_net = SeparateActorCritic(env.observation_space.shape[0], action_dims, hidden_dim).to(device)
            if state_dict is not None:
                policy_net.load_state_dict(state_dict)
                print(f"[PPO v2 Eval] Loaded trained model weights from: {ckpt_file}")
            else:
                print(f"[PPO v2 Eval] Warning: Checkpoint '{ckpt_file}' not found; evaluating initial weights.")
            if save_gif and gif_path is None:
                gif_path = Path(ckpt_file).parent / f"eval_{self.cfg.hyperparameter_set}.gif"
            policy_net.eval()

            def select(state: torch.Tensor) -> list[int]:
                return greedy_action(policy_net, state)

            return run_eval_episodes(
                env,
                select_blue=select,
                select_orange=select if control_car == "both" else None,
                num_episodes=num_episodes,
                gif_path=gif_path if save_gif else None,
                gif_tag="PPO v2",
            )
        finally:
            env.close()


def _resolve_config(
    config: PPOv2Config | None,
    hyperparameter_set: str | None,
    num_envs: int | None = None,
) -> PPOv2Config:
    cfg = config or PPOv2Config()
    if hyperparameter_set is not None:
        cfg = replace(cfg, hyperparameter_set=hyperparameter_set)
    if num_envs is not None:
        cfg = replace(cfg, num_envs=num_envs)
    return cfg


def train_ppo_v2(
    num_episodes: int | None = None,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
    hyperparameter_set: str | None = None,
    config: PPOv2Config | None = None,
    num_envs: int | None = None,
) -> dict[str, float]:
    """Train the PPO v2 agent."""
    agent = PPOv2Agent(_resolve_config(config, hyperparameter_set, num_envs=num_envs))
    return agent.train(num_episodes=num_episodes, model_path=model_path, max_episode_steps=max_episode_steps)


def evaluate_ppo_v2(
    num_episodes: int | None = PPOv2Config.eval_episodes,
    render: bool = False,
    max_episode_steps: int | None = None,
    model_path: str | Path | None = None,
    hyperparameter_set: str | None = None,
    save_gif: bool = False,
    gif_path: str | Path | None = None,
    control_car: str = "blue",
    random_spawn: bool = True,
    config: PPOv2Config | None = None,
) -> dict[str, float]:
    """Evaluate/validate a trained PPO v2 agent."""
    agent = PPOv2Agent(_resolve_config(config, hyperparameter_set))
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
