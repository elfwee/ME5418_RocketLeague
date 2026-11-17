"""
PPO v2 Learning Agent: separate-network Actor-Critic with factored MultiDiscrete heads and parallel envs.

Differs from `src/learning_agents/ppo_agent.py` in five ways:
1. Architecture (`SeparateActorCritic` in `src/neural_nets/ppo_v2_torch.py`): completely independent
   policy and value MLPs (no shared trunk parameters; 164,366 trainable parameters for width 256,
   NOT parameter-matched to Dueling DDQN's 150,949 parameters). With a shared trunk the critic's
   gradient alone moved the policy past the KL stop, so most of each trust region went to critic noise.
2. Action space: one flat `Discrete(36)` softmax by default (`pi_heads.0`, like the DQN), or factored
   `MultiDiscrete([9, 2, 2])` heads (`pi_heads.0/1/2`) with `factored_action=True`.
3. Parallel rollouts: collects `rollout_steps` across `num_envs` vectorized environments (default: one
   per logical CPU minus two) in either `"subproc"` (one worker process per env, each with its own random
   streams) or `"sync"` mode via `step_with_autoreset`, and updates on large batches (~8,192 samples
   whatever `num_envs` is, 1,024-sample minibatches) under a loose KL safety stop. The learner keeps
   `rollout_threads` torch threads while the workers step and `update_threads` during the update.
4. Evaluation-based model selection: every `eval_interval_steps` the greedy policy plays fixed-seed
   games against each evaluation opponent (no opponent, the heuristic bot, or frozen checkpoints),
   split into small (opponent, seed range) tasks that run in parallel on the env workers. Each evaluated model is saved to `checkpoints/`; the best score
   (win rate - loss rate) is saved as `model_<name>.pt`. Results go to `eval_<name>.csv` and
   `eval_<name>_opponents.csv`.
5. Training length in env steps: training stops after `eval_patience` evaluations without a new best
   score (once `min_train_steps` have been collected), at `max_train_steps`, or after `num_episodes`.
"""

import contextlib
import io
import math
import multiprocessing as mp
import os
import random
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.config.curriculum_learning import CurriculumConfig
from src.config.gym_config import GymConfig
from src.config.nn_config import PPOv2Config
from src.learning_agents.dashboard import metric_panel, save_training_dashboard
from src.learning_agents.env_factory import device, make_env, normalize_control_car, obs_to_tensor, scored_goal
from src.learning_agents.eval_loop import resolve_eval_episodes, run_eval_episodes
from src.learning_agents.ppo_agent import compute_gae
from src.learning_agents.run_dirs import RunPaths, latest_model_path, save_periodic_checkpoint
from src.learning_agents.training_logger import DATE_FORMAT, TrainingLogger
from src.neural_nets.checkpoint import infer_spec, load_state_dict
from src.neural_nets.policy import greedy_action
from src.neural_nets.ppo_v2_torch import SeparateActorCritic

EVAL_SEED_BASE = 10_000
EVAL_NO_OPPONENT = "none"  # evaluation opponent: empty arena
EVAL_HEURISTIC = "heuristic"  # evaluation opponent: the scripted HeuristicBot

# (opponent, first episode index, number of episodes): one evaluation task
EvalTask = tuple[str, int, int]
# (opponent, wins, losses, draws, summed return, summed decision steps)
EvalResult = tuple[str, int, int, int, float, int]


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


def _seed_process_rngs(seed: int | None):
    """Give this process its own Python / NumPy / torch random streams.

    CPython reseeds `random` in forked children, but NumPy and torch keep the parent's state, so
    without this every worker's stochastic self-play opponents (torch sampling) drew identical actions.
    """
    seed = int.from_bytes(os.urandom(4), "little") if seed is None else seed
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _make_eval_env(
    opponent: str, discrete_action: bool, random_spawn: bool, max_episode_steps: int | None, action_repeat: int
):
    """Env with a fixed Orange opponent: `EVAL_NO_OPPONENT`, `EVAL_HEURISTIC`, or a checkpoint path (greedy)."""
    common = dict(
        discrete_action=discrete_action,
        control_car="blue",
        random_spawn=random_spawn,
        max_episode_steps=max_episode_steps,
        action_repeat=action_repeat,
    )
    if opponent in (EVAL_NO_OPPONENT, EVAL_HEURISTIC):
        bot_type = "none" if opponent == EVAL_NO_OPPONENT else "heuristic_bot"
        return make_env(bot_type=bot_type, env_id=GymConfig.gym_id, **common)
    with contextlib.redirect_stdout(io.StringIO()):  # PolicyController announces every checkpoint it loads
        return make_env(
            bot_type="heuristic_bot",
            env_id=CurriculumConfig.env_id,
            env_kwargs={"opponent_pool": [opponent], "heuristic_prob": 0.0, "league_prob": 0.0},
            **common,
        )


def evaluate_policy(
    state_dict: dict,
    tasks: Sequence[EvalTask],
    random_spawn: bool,
    max_episode_steps: int | None,
    action_repeat: int,
) -> list[EvalResult]:
    """Greedy games of a `SeparateActorCritic`, one result per (opponent, first episode, count) task.
    Episode `i` always uses seed `EVAL_SEED_BASE + i`, so every evaluation (and every opponent) sees the
    same spawns."""
    spec = infer_spec(state_dict)
    net = SeparateActorCritic(spec.state_dim, spec.action_dim, spec.hidden_dim)
    net.load_state_dict(state_dict)
    net.eval()
    results = []
    for opponent, first_episode, episodes in tasks:
        env = _make_eval_env(opponent, not net.is_factored, random_spawn, max_episode_steps, action_repeat)
        try:
            wins = losses = draws = steps = 0
            total_return = 0.0
            for ep_idx in range(first_episode, first_episode + episodes):
                obs, _ = env.reset(seed=EVAL_SEED_BASE + ep_idx)
                terminated = truncated = goal = False
                while not (terminated or truncated):
                    obs, reward, terminated, truncated, info = env.step(greedy_action(net, obs_to_tensor(obs)))
                    total_return += float(reward)
                    steps += 1
                    goal = goal or scored_goal(info, bool(terminated), float(reward))
                if goal:
                    wins += 1
                elif terminated:
                    losses += 1
                else:
                    draws += 1
            results.append((opponent, wins, losses, draws, total_return, steps))
        finally:
            env.close()
    return results


def _worker_loop(
    conn: Connection,
    bot_type: str,
    random_spawn: bool,
    max_episode_steps: int | None,
    env_id: str,
    action_repeat: int,
    env_kwargs: dict | None = None,
    worker_seed: int | None = None,
    discrete_action: bool = True,
):
    """Subprocess env worker: ignores SIGINT so parent cleans up workers cleanly."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    _seed_process_rngs(worker_seed)
    env = make_env(
        bot_type=bot_type,
        discrete_action=discrete_action,
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
            elif cmd == "eval":
                conn.send(evaluate_policy(*payload))
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
        self,
        cfg: PPOv2Config,
        random_spawn: bool,
        max_episode_steps: int | None,
        env_kwargs: dict | None = None,
        discrete_action: bool = True,
    ):
        self.envs = [
            make_env(
                bot_type=cfg.bot_type,
                discrete_action=discrete_action,
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

    def evaluate(self, state_dict: dict, tasks: Sequence[EvalTask], *eval_args) -> list[EvalResult]:
        """`evaluate_policy` in this process."""
        return evaluate_policy(state_dict, tasks, *eval_args)

    def close(self):
        for env in self.envs:
            env.close()
        self.envs.clear()


class _SubprocEnvPool:
    """Multiprocessing vectorized wrapper: one daemon process per environment."""

    def __init__(
        self,
        cfg: PPOv2Config,
        random_spawn: bool,
        max_episode_steps: int | None,
        env_kwargs: dict | None = None,
        discrete_action: bool = True,
    ):
        self._closed = False
        self._parent_conns: list[Connection] = []
        self._procs: list[mp.Process] = []
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        for idx in range(cfg.num_envs):
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
                    None if cfg.seed is None else cfg.seed + 1 + idx,
                    discrete_action,
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

    def evaluate(self, state_dict: dict, tasks: Sequence[EvalTask], *eval_args) -> list[EvalResult]:
        """`evaluate_policy` with the tasks spread over the (otherwise idle) env workers."""
        n = len(self._parent_conns)
        jobs = [(conn, list(tasks[i::n])) for i, conn in enumerate(self._parent_conns) if tasks[i::n]]
        for conn, chunk in jobs:
            conn.send(("eval", (state_dict, chunk, *eval_args)))
        return [result for conn, _ in jobs for result in conn.recv()]

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
    cfg: PPOv2Config,
    random_spawn: bool,
    max_episode_steps: int | None,
    env_kwargs: dict | None = None,
    discrete_action: bool = True,
):
    if cfg.env_workers not in ("subproc", "sync"):
        raise ValueError(f"Invalid env_workers: '{cfg.env_workers}'. Must be 'subproc' or 'sync'.")
    pool_cls = _SyncEnvPool if cfg.num_envs == 1 or cfg.env_workers == "sync" else _SubprocEnvPool
    return pool_cls(cfg, random_spawn, max_episode_steps, env_kwargs=env_kwargs, discrete_action=discrete_action)


def _eval_tasks(opponents: Sequence[str], episodes: int, num_workers: int) -> list[EvalTask]:
    """Split `episodes` games per opponent into about `num_workers` similar-sized (opponent, start, count) tasks."""
    chunk = max(1, math.ceil(len(opponents) * episodes / max(num_workers, 1)))
    return [
        (opponent, start, min(chunk, episodes - start)) for opponent in opponents for start in range(0, episodes, chunk)
    ]


def _opponent_label(opponent: str) -> str:
    return opponent if opponent in (EVAL_NO_OPPONENT, EVAL_HEURISTIC) else Path(opponent).stem


class PPOv2Agent:
    """Separate-network PPO Learning Agent with factored action heads and parallel rollouts."""

    metric_columns = ("entropy", "value_loss")

    def __init__(self, config: PPOv2Config | None = None):
        self.cfg = config or PPOv2Config()
        self.optimizer: torch.optim.Optimizer | None = None
        self.model: SeparateActorCritic | None = None
        self.current_entropy = float(sum(np.log(d) for d in self.cfg.policy_action_dims))
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

    def _evaluate(
        self,
        pool,
        paths: RunPaths,
        step_count: int,
        opponents: Sequence[str],
        eval_args: tuple,
        best_score: float,
    ) -> float:
        """Greedy fixed-seed games against every opponent; appends to the eval CSVs and returns the
        score (win rate - loss rate over all games, in [-1, 1])."""
        start = time.perf_counter()
        state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        episodes = self.cfg.eval_episodes_per_opponent
        totals = {opponent: [opponent, 0, 0, 0, 0.0, 0] for opponent in opponents}
        for opponent, *counts in pool.evaluate(state_dict, _eval_tasks(opponents, episodes, self.cfg.num_envs), *eval_args):
            for i, count in enumerate(counts, start=1):
                totals[opponent][i] += count
        results = [tuple(totals[opponent]) for opponent in opponents]

        wins, losses, draws = (sum(r[i] for r in results) for i in (1, 2, 3))
        games = max(wins + losses + draws, 1)
        score = (wins - losses) / games
        mean_return = sum(r[4] for r in results) / games
        mean_length = sum(r[5] for r in results) / games

        write_header = not paths.eval_file.exists() or os.path.getsize(paths.eval_file) == 0
        with open(paths.eval_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write("step,score,goal_rate,concede_rate,timeout_rate,mean_return,mean_length,is_best\n")
            f.write(
                f"{step_count},{score:.4f},{100 * wins / games:.2f},{100 * losses / games:.2f},"
                f"{100 * draws / games:.2f},{mean_return:.4f},{mean_length:.2f},{int(score > best_score)}\n"
            )
        opponents_file = paths.eval_file.with_name(f"eval_{paths.model_name}_opponents.csv")
        write_header = not opponents_file.exists() or os.path.getsize(opponents_file) == 0
        with open(opponents_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write("step,opponent,games,wins,losses,draws,score\n")
            for opponent, w, l, d, _, _ in results:
                n = max(w + l + d, 1)
                f.write(f"{step_count},{_opponent_label(opponent)},{n},{w},{l},{d},{(w - l) / n:.4f}\n")

        per_opponent = sorted(((r[1] - r[2]) / max(r[1] + r[2] + r[3], 1), _opponent_label(r[0])) for r in results)
        hardest = ", ".join(f"{name} {s:+.2f}" for s, name in per_opponent[:3])
        print(
            f"[PPO v2 Eval @ step {step_count}] Score={score:+.3f} (best {max(score, best_score):+.3f}) | "
            f"Goal={100 * wins / games:.1f}% Concede={100 * losses / games:.1f}% Timeout={100 * draws / games:.1f}% | "
            f"{len(opponents)} opponent(s) x {episodes} games | hardest: {hardest} | "
            f"{time.perf_counter() - start:.1f}s"
        )
        return score

    def _log(self, paths: RunPaths, message: str):
        """Print `message` and append it to the run's `.log` file."""
        line = f"{datetime.now().strftime(DATE_FORMAT)} - {message}"
        print(line)
        with open(paths.log_file, "a") as log_file:
            log_file.write(line + "\n")

    def train(
        self,
        num_episodes: int | None = None,
        model_path: str | Path | None = None,
        max_episode_steps: int | None = None,
        random_spawn: bool = True,
        stop_fn: Callable[[float], bool] | None = None,
        env_kwargs: dict | None = None,
        eval_opponents: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """Parallel on-policy training across `num_envs` workers.

        `eval_opponents` lists the evaluation opponents (`"none"`, `"heuristic"` or checkpoint paths);
        by default the env's own opponent (`"heuristic"` when `bot_type="heuristic_bot"`, else `"none"`).
        `stop_fn(episode_return) -> bool` can end training early from outside.
        """
        state_dict = load_state_dict(str(model_path), device) if model_path is not None else None
        spec_action_dims = list(self.cfg.policy_action_dims)
        hidden_dim = self.cfg.hidden_size
        if state_dict is not None:
            spec = infer_spec(state_dict)
            if isinstance(spec.action_dim, list):
                spec_action_dims = spec.action_dim
            hidden_dim = spec.hidden_dim

        if eval_opponents is None:
            eval_opponents = [EVAL_HEURISTIC if self.cfg.bot_type == "heuristic_bot" else EVAL_NO_OPPONENT]
        eval_opponents = list(eval_opponents)
        eval_enabled = bool(eval_opponents) and self.cfg.eval_interval_steps > 0 and self.cfg.eval_episodes_per_opponent > 0
        eval_args = (random_spawn, max_episode_steps, self.cfg.action_repeat)
        factored = len(spec_action_dims) > 1  # a checkpoint's head layout wins over `factored_action`

        pool = _make_env_pool(
            self.cfg, random_spawn, max_episode_steps, env_kwargs=env_kwargs, discrete_action=not factored
        )
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(self.cfg.rollout_threads)
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
            paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"[PPO v2 Train] Saving outputs to: {paths.run_dir}")
            print(
                f"[PPO v2 Train] Device={device} | Envs={self.cfg.num_envs} ({self.cfg.env_workers}) | "
                f"Rollout={self.cfg.rollout_steps}x{self.cfg.num_envs} | Epochs={self.cfg.update_epochs} | "
                f"Minibatch={self.cfg.minibatch_size} | TargetKL={self.cfg.target_kl} | Bot={self.cfg.bot_type} | "
                f"Model={paths.model_file}"
            )
            if eval_enabled:
                print(
                    f"[PPO v2 Train] Evaluation every {self.cfg.eval_interval_steps} steps vs "
                    f"{len(eval_opponents)} opponent(s) [{', '.join(_opponent_label(o) for o in eval_opponents)}]; "
                    f"stop after {self.cfg.eval_patience} evaluations without a new best "
                    f"(min {self.cfg.min_train_steps} steps, max {self.cfg.max_train_steps} steps)"
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
            best_score = float("-inf")
            evals_since_best = 0
            next_eval_step = 0  # the starting weights are evaluated first: a stage keeps them unless it beats them
            last_eval_step = -1

            def run_eval():
                """Evaluate, checkpoint the evaluated model, and keep it as the best model if it scored best."""
                nonlocal best_score, evals_since_best, last_eval_step
                score = self._evaluate(pool, paths, total_steps, eval_opponents, eval_args, best_score)
                last_eval_step = total_steps
                torch.save(self.model.state_dict(), paths.checkpoint_dir / f"model_{paths.model_name}_step{total_steps:09d}.pt")
                if score > best_score:
                    best_score = score
                    evals_since_best = 0
                    torch.save(self.model.state_dict(), paths.model_file)
                    self._log(paths, f"New best evaluation score {score:+.4f} at step {total_steps}. Saving model.")
                else:
                    evals_since_best += 1

            def budget_reason() -> str | None:
                """Why training should stop now (env-step budget / evaluation plateau), or None."""
                if self.cfg.max_train_steps is not None and total_steps >= self.cfg.max_train_steps:
                    return f"reached max_train_steps={self.cfg.max_train_steps}"
                plateau = eval_enabled and self.cfg.eval_patience > 0 and evals_since_best >= self.cfg.eval_patience
                if plateau and total_steps >= self.cfg.min_train_steps:
                    return f"{evals_since_best} evaluations without a new best (best {best_score:+.4f})"
                return None

            obs_buf = torch.zeros((T, n_envs, obs_dim), dtype=torch.float32, device=device)
            act_shape = (T, n_envs, len(spec_action_dims)) if factored else (T, n_envs)
            act_buf = torch.zeros(act_shape, dtype=torch.long, device=device)
            logp_buf = torch.zeros((T, n_envs), dtype=torch.float32, device=device)
            val_buf = torch.zeros((T, n_envs), dtype=torch.float32, device=device)
            rew_buf = torch.zeros((T, n_envs), dtype=torch.float32, device=device)
            done_buf = torch.zeros((T, n_envs), dtype=torch.float32, device=device)
            term_buf = torch.zeros((T, n_envs), dtype=torch.bool, device=device)
            trunc_buf = torch.zeros((T, n_envs), dtype=torch.bool, device=device)
            trunc_val_buf = torch.zeros((T, n_envs), dtype=torch.float32, device=device)

            try:
                while not stopped_early and (num_episodes is None or episodes_done < num_episodes):
                    if eval_enabled and total_steps >= next_eval_step:
                        run_eval()
                        next_eval_step = total_steps + self.cfg.eval_interval_steps
                    reason = budget_reason()
                    if reason is not None:
                        self._log(paths, f"Stopping at step {total_steps}: {reason}.")
                        break

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

                                    # Without evaluation, fall back to the training-return best model and
                                    # episode-based periodic checkpoints
                                    if not eval_enabled:
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
                    torch.set_num_threads(self.cfg.update_threads)
                    _, ent, v_loss, _ = self.optimize(
                        obs_buf[s_slice].reshape(-1, obs_dim),
                        act_buf[s_slice].reshape(-1, len(spec_action_dims)) if factored else act_buf[s_slice].reshape(-1),
                        logp_buf[s_slice].reshape(-1),
                        ret.reshape(-1),
                        adv.reshape(-1),
                        self.model,
                    )
                    torch.set_num_threads(self.cfg.rollout_threads)
                    self.current_entropy = ent
                    self.current_v_loss = v_loss / (self.cfg.reward_scale**2)

                if eval_enabled and total_steps > 0 and total_steps != last_eval_step:
                    run_eval()
                if not paths.model_file.exists():
                    torch.save(self.model.state_dict(), paths.model_file)
            finally:
                if logger.rewards:
                    save_graph()

            total_goals = int(sum(logger.goals))
            return {
                "episodes": len(logger.rewards),
                "steps": total_steps,
                "mean_reward": float(np.mean(logger.rewards)) if logger.rewards else 0.0,
                "goal_rate": float(np.mean(logger.goals)) * 100.0 if logger.goals else 0.0,
                "best_eval_score": best_score,
                "score_blue": total_goals,
                "score_orange": len(logger.goals) - total_goals,
            }
        finally:
            pool.close()
            torch.set_num_threads(previous_threads)

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

        action_dims = list(self.cfg.policy_action_dims)
        hidden_dim = self.cfg.hidden_size
        if state_dict is not None:
            spec = infer_spec(state_dict)
            if isinstance(spec.action_dim, list):
                action_dims = spec.action_dim
            hidden_dim = spec.hidden_dim

        env = make_env(
            bot_type=self.cfg.bot_type,
            discrete_action=len(action_dims) == 1,
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

            def select(state: torch.Tensor) -> int | list[int]:
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
