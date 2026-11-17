"""
Curriculum learning: train one policy through a sequence of increasingly hard opponents.

Stages:
    1. "none"      - no opponent; learn to drive to the ball and score.
    2. "heuristic" - against the scripted HeuristicBot.
    3. "self"      - adversarial self-play against a mixed pool of frozen self checkpoints
                     (latest + past checkpoints), the HeuristicBot and, optionally, a frozen
                     "league" of other checkpoints (`--league`, e.g. Dueling DDQN models).

Each stage (and each self-play round) is one ordinary training run of `PPOAgent`, `PPOv2Agent`,
or `DuelingDDQNAgent`, started from the previous stage's best weights (or a supplied `--model_path`).
D3QN / PPO stages stop once the 100-episode rolling mean return has not improved for `patience`
episodes. PPO v2 stages instead evaluate the greedy policy every `PPOv2Config.eval_interval_steps`
against the stage's opponents (none / HeuristicBot / HeuristicBot + league + self pool) and keep the
best-scoring model; stages 1-2 stop after `PPOv2Config.eval_patience` evaluations without improvement,
self-play rounds run a fixed `self_play_steps` env steps.

The agents are steered through their config (`bot_type`, `env_id`, `hyperparameter_set`), and the
opponent pool / plateau stop is provided by `CurriculumEnv`, a subclass of `RocketLeagueEnv`
registered under its own gym id (PPO v2's parallel workers each build their own `CurriculumEnv`).

Usage:
    python -m src --curriculum --ppo|--ppo_v2|--dueling_ddqn|--double_dqn|--dqn [--model_path PATH] [--start_stage self]
        [--league DIR_OR_PT ...]
    python -m src.curriculum_learning.curriculum_learning --agent ppo --model_path PATH --start_stage self
"""

import argparse
import random
import shutil
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.config.curriculum_learning import CurriculumConfig
from src.config.gym_config import GymConfig
from src.config.nn_config import DQNConfig, PPOConfig, PPOv2Config
from src.gym_env.rocket_league_env import RocketLeagueEnv
from src.learning_agents.dueling_ddqn_agent import DuelingDDQNAgent
from src.learning_agents.env_factory import make_env
from src.learning_agents.ppo_agent import PPOAgent
from src.learning_agents.ppo_v2_agent import EVAL_HEURISTIC, EVAL_NO_OPPONENT, PPOv2Agent
from src.learning_agents.run_dirs import latest_model_path, make_run_dir
from src.neural_nets.ppo_torch import ActorCritic
from src.neural_nets.ppo_v2_torch import SeparateActorCritic
from src.sim.control.policy import PolicyController

CURRICULUM_ENV_ID = CurriculumConfig.env_id
if CURRICULUM_ENV_ID not in gym.envs.registry:
    gym.register(
        id=CURRICULUM_ENV_ID,
        entry_point="src.curriculum_learning.curriculum_learning:CurriculumEnv",
        max_episode_steps=GymConfig.episode_max_steps,
    )


def resolve_league(paths: Sequence[str | Path]) -> list[Path]:
    """League checkpoints from `.pt` files and/or folders of `.pt` files (sorted, duplicates removed)."""
    league: list[Path] = []
    for entry in map(Path, paths):
        if entry.is_dir():
            found = sorted(entry.glob("*.pt"))
            if not found:
                raise FileNotFoundError(f"League folder has no .pt checkpoints: {entry}")
            league.extend(found)
        elif entry.is_file():
            league.append(entry)
        else:
            raise FileNotFoundError(f"League checkpoint or folder not found: {entry}")
    return list(dict.fromkeys(p.resolve() for p in league))


class CurriculumEnv(RocketLeagueEnv):
    """`RocketLeagueEnv` with two curriculum hooks.

    - `opponent_path` / `opponent_pool` / `league_pool`: during self-play, each episode samples Orange's
      controller from the HeuristicBot (`heuristic_prob`), a frozen league checkpoint (`league_prob`,
      greedy, when a league is given) or a frozen self-play checkpoint (latest / past). Policies are
      `PolicyController`s, mirrored so a blue-trained model can play orange; self-play checkpoints can
      sample stochastically.
    - `on_episode_end(episode_return) -> bool`: called with each finished episode's return. When it
      returns True the env marks itself closed (`isopen = False`), which ends the agent's training
      loop through its existing "window closed" path (dashboard is still saved, run returns normally).

    The episode return is reported on the following `reset()` because time-limit truncation happens
    in gym's `TimeLimit` wrapper, outside this env.
    """

    def __init__(
        self,
        *args,
        on_episode_end=None,
        opponent_path: str | Path | None = None,
        opponent_pool: Sequence[str | Path] | None = None,
        opponent_action_repeat: int = PPOConfig.action_repeat,
        stochastic_opponent: bool = CurriculumConfig.stochastic_opponent,
        latest_self_prob: float = CurriculumConfig.latest_self_prob,
        past_self_prob: float = CurriculumConfig.past_self_prob,
        heuristic_prob: float = CurriculumConfig.heuristic_prob,
        league_pool: Sequence[str | Path] | None = None,
        league_prob: float = CurriculumConfig.league_prob,
        **kwargs,
    ):
        self.on_episode_end = on_episode_end
        self._episode_return = 0.0
        self._episode_steps = 0
        self._latest_self_prob = latest_self_prob
        self._past_self_prob = past_self_prob
        self._heuristic_prob = heuristic_prob
        self._league_prob = league_prob
        self._heuristic_opponent = None
        self._policy_opponents: list[PolicyController] = []
        self._league_opponents: list[PolicyController] = []
        super().__init__(*args, **kwargs)

        raw_pool = list(opponent_pool) if opponent_pool else ([opponent_path] if opponent_path is not None else [])
        # Keep unique checkpoint paths in chronological order with latest at the end
        unique_paths: list[Path] = []
        for p in raw_pool:
            path = Path(p)
            if path in unique_paths:
                unique_paths.remove(path)
            unique_paths.append(path)

        if unique_paths:
            if self.orange_control is None:
                raise ValueError("Self-play needs an orange car: create the env with bot_type='heuristic_bot'.")
            self._heuristic_opponent = self.orange_control
            self._policy_opponents = [
                PolicyController(
                    "orange",
                    self.sim,
                    p,
                    action_repeat=opponent_action_repeat,
                    stochastic=stochastic_opponent,
                )
                for p in unique_paths
            ]
            # League opponents always play greedily, as in the round robin they are compared in
            self._league_opponents = [
                PolicyController("orange", self.sim, p, action_repeat=opponent_action_repeat)
                for p in dict.fromkeys(Path(p) for p in league_pool or ())
            ]
            self._sample_opponent()

    def _sample_opponent(self) -> None:
        """Select Orange's opponent for the upcoming episode from the mixed pool."""
        if not self._policy_opponents:
            return
        draw = random.random()
        if self._heuristic_opponent is not None and draw < self._heuristic_prob:
            self.orange_control = self._heuristic_opponent
        elif self._league_opponents and draw < self._heuristic_prob + self._league_prob:
            self.orange_control = random.choice(self._league_opponents)
        elif len(self._policy_opponents) == 1:
            self.orange_control = self._policy_opponents[0]
        else:
            self_total = max(self._latest_self_prob + self._past_self_prob, 1e-8)
            latest_weight = self._latest_self_prob / self_total
            if random.random() < latest_weight:
                self.orange_control = self._policy_opponents[-1]
            else:
                self.orange_control = random.choice(self._policy_opponents[:-1])
        self.orange_control.reset()

    def step(self, action, action_orange=None):
        result = super().step(action, action_orange)
        self._episode_return += float(result[1])
        self._episode_steps += 1
        return result

    def reset(self, seed=None, options=None):
        finished = self._episode_steps > 0
        episode_return = self._episode_return
        self._episode_return = 0.0
        self._episode_steps = 0

        self._sample_opponent()
        obs, info = super().reset(seed=seed, options=options)
        if finished and self.on_episode_end is not None and self.on_episode_end(episode_return):
            self.isopen = False
        return obs, info


class Curriculum:
    MIN_EPISODES = CurriculumConfig.min_episodes  # warm-up before a stage can set a best / plateau
    BOT_TYPES = dict(CurriculumConfig.stages)  # stage -> env bot_type
    CONFIGS = {PPOAgent: PPOConfig, PPOv2Agent: PPOv2Config, DuelingDDQNAgent: DQNConfig}

    def __init__(
        self,
        agent,
        patience: int | None = None,
        self_play_rounds: int = CurriculumConfig.self_play_rounds,
        max_stage_episodes: int | None = CurriculumConfig.max_stage_episodes,
        model_name: str | None = None,
        config: PPOConfig | PPOv2Config | DQNConfig | None = None,
        model_path: str | Path | None = None,
        start_stage: str = "none",
        stochastic_opponent: bool = CurriculumConfig.stochastic_opponent,
        league: Sequence[str | Path] = (),
        self_play_steps: int = CurriculumConfig.ppo_v2_self_play_steps,
    ) -> None:
        all_stages = [stage for stage, _ in CurriculumConfig.stages]
        if start_stage not in all_stages:
            raise ValueError(f"Invalid start_stage '{start_stage}'. Must be one of {all_stages}.")
        self.stages = all_stages[all_stages.index(start_stage) :]
        self.agent = agent  # PPOAgent, PPOv2Agent or DuelingDDQNAgent (the class)
        is_ppo_v2 = issubclass(self.agent, PPOv2Agent)
        if is_ppo_v2 and patience is not None:
            raise ValueError("PPO v2 stages end on evaluation (PPOv2Config.eval_patience), not on --patience.")
        self.patience = CurriculumConfig.patience if patience is None else patience
        self.config = config if config is not None else self.CONFIGS[agent]()  # e.g. a DQN ablation variant
        self.self_play_rounds = self_play_rounds
        self.max_stage_episodes = max_stage_episodes
        self.model_name = model_name or self.config.hyperparameter_set
        self.model_path = Path(model_path) if model_path is not None else None
        self.start_stage = start_stage
        self.stochastic_opponent = stochastic_opponent
        self.league = resolve_league(league)
        self.self_play_steps = self_play_steps  # PPO v2 only: fixed env-step budget per self-play round
        self.highest_reward = [float("-inf"), 0]  # [highest_reward, episode]

        self.run_path: Path | None = None
        self.weights_path: Path | None = None
        self._opponent_pool: list[Path] = []

        # Per-stage state read by the env factory / episode callback
        self._stage_rewards: list[float] = []
        self._opponent_path: Path | None = None
        self._stage_opponent_pool: list[Path] = []
        self._opponent_action_repeat = self.config.action_repeat

        gym.envs.registry.pop(CURRICULUM_ENV_ID, None)
        entry_point = (
            "src.curriculum_learning.curriculum_learning:CurriculumEnv" if is_ppo_v2 else self._make_env
        )
        gym.register(
            id=CURRICULUM_ENV_ID,
            entry_point=entry_point,
            max_episode_steps=GymConfig.episode_max_steps,
        )

    # ----------------------------------------------------------------------------------------
    # Environment hooks
    # ----------------------------------------------------------------------------------------

    def _make_env(self, **kwargs) -> CurriculumEnv:
        """gym entry point: the agent's `make_env` kwargs plus the current stage's hooks."""
        return CurriculumEnv(
            on_episode_end=self._on_episode_end,
            opponent_path=self._opponent_path,
            opponent_pool=self._stage_opponent_pool,
            opponent_action_repeat=self._opponent_action_repeat,
            stochastic_opponent=self.stochastic_opponent,
            league_pool=self.league if self._stage_opponent_pool else None,
            **kwargs,
        )

    def _on_episode_end(self, episode_return: float) -> bool:
        self._stage_rewards.append(episode_return)
        episode = len(self._stage_rewards) - 1
        average_reward = float(np.mean(self._stage_rewards[-CurriculumConfig.reward_window :]))
        plateau = self.check_plateu(average_reward, episode)
        if plateau:
            print(
                f"[Curriculum] Plateau at episode {episode}: no new best 100-ep mean reward "
                f"for {self.patience} episodes (best {self.highest_reward[0]:.2f} at episode "
                f"{self.highest_reward[1]})."
            )
        return plateau

    # ----------------------------------------------------------------------------------------
    # Curriculum
    # ----------------------------------------------------------------------------------------

    def init_weights(self):
        """
            1. self.run_path = runs/YYYYMMDD_hhmm_<model_name>_curriculum
            2. mkdir self.run_path
            3. copy `--model_path` if supplied, else initialise fresh weights
            4. save .pt file in directory
        """
        self.run_path = make_run_dir(f"{self.model_name}_curriculum")
        self.weights_path = self.run_path / f"model_{self.model_name}_init.pt"

        if self.model_path is not None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"Input model checkpoint not found: {self.model_path}")
            shutil.copyfile(self.model_path, self.weights_path)
            print(f"[Curriculum] Loaded starting weights from {self.model_path} -> {self.weights_path}")
            return

        is_ppo_v2 = issubclass(self.agent, PPOv2Agent)
        is_ppo = issubclass(self.agent, PPOAgent)
        factored = (is_ppo_v2 or is_ppo) and self.config.factored_action
        env = make_env(bot_type="none", discrete_action=not factored, control_car="blue")
        try:
            obs_dim = env.observation_space.shape[0]
            if is_ppo_v2:
                net = SeparateActorCritic(obs_dim, list(self.config.policy_action_dims), self.config.hidden_size)
            elif is_ppo:
                action_dim = [int(n) for n in env.action_space.nvec] if factored else env.action_space.n
                net = ActorCritic(obs_dim, action_dim, self.config.hidden_size)
            else:
                net = self.agent(self.config)._build_network(obs_dim, env.action_space.n)
        finally:
            env.close()

        torch.save(net.state_dict(), self.weights_path)
        print(f"[Curriculum] Saved initial weights to: {self.weights_path}")

    def _eval_opponents(self, stage: str) -> list[str]:
        """PPO v2 evaluation opponents for `stage`: the empty arena, the heuristic bot, or (self-play)
        the heuristic bot + the league + every frozen self-play checkpoint so far."""
        if stage == "none":
            return [EVAL_NO_OPPONENT]
        if stage == "heuristic":
            return [EVAL_HEURISTIC]
        opponents = [str(p) for p in dict.fromkeys([*self.league, *self._stage_opponent_pool])]
        return [EVAL_HEURISTIC, *opponents]

    def check_plateu(self, average_reward, episode):
        """
        1. if average_reward > highest_reward[0] update highest_reward and return False
        2. if episode - highest_reward[1] > patience return True
        """
        if episode + 1 < self.MIN_EPISODES:
            return False
        if average_reward > self.highest_reward[0]:
            self.highest_reward = [average_reward, episode]
            return False
        return episode - self.highest_reward[1] > self.patience

    def _train_stage(
        self,
        stage: str,
        label: str,
        opponent_path: Path | None = None,
        opponent_pool: list[Path] | None = None,
        first: bool = False,
    ):
        """One training run of the agent for `stage`, starting from `self.weights_path`."""
        self.highest_reward = [float("-inf"), 0]
        self._stage_rewards = []
        self._opponent_path = opponent_path
        self._stage_opponent_pool = list(opponent_pool) if opponent_pool is not None else []

        cfg = replace(
            self.config,
            hyperparameter_set=f"{self.model_name}_{label}",
            bot_type=self.BOT_TYPES[stage],
            env_id=CURRICULUM_ENV_ID,
        )
        if first and self.model_path is None and isinstance(cfg, DQNConfig):
            # DQN drops to epsilon_finetune when given a checkpoint; the init weights are untrained
            cfg = replace(cfg, epsilon_finetune=cfg.epsilon_init)
        if stage == "self" and isinstance(cfg, PPOv2Config):
            # Fixed budget: the evaluation plateau is too noisy to end a round against strong opponents
            cfg = replace(cfg, min_train_steps=self.self_play_steps, max_train_steps=self.self_play_steps)
        self._opponent_action_repeat = cfg.action_repeat
        agent = self.agent(cfg)

        pool_desc = ""
        if self._stage_opponent_pool:
            names = ", ".join(p.name for p in self._stage_opponent_pool)
            pool_desc = f" vs pool=[{names}] + heuristic (stochastic={self.stochastic_opponent})"
            if self.league:
                pool_desc += f" + league of {len(self.league)}"
        elif opponent_path is not None:
            pool_desc = f" vs {opponent_path}"
        print(f"\n[Curriculum] ===== Stage '{label}'{pool_desc} | start weights: {self.weights_path} =====")
        if isinstance(agent, PPOv2Agent):
            env_kwargs = {}
            if self._stage_opponent_pool or opponent_path is not None:
                env_kwargs = {
                    "opponent_path": str(opponent_path) if opponent_path is not None else None,
                    "opponent_pool": [str(p) for p in self._stage_opponent_pool],
                    "opponent_action_repeat": cfg.action_repeat,
                    "stochastic_opponent": self.stochastic_opponent,
                    "league_pool": [str(p) for p in self.league],
                }
            summary = agent.train(
                num_episodes=self.max_stage_episodes,
                model_path=self.weights_path,
                env_kwargs=env_kwargs or None,
                eval_opponents=self._eval_opponents(stage),
            )
        else:
            summary = agent.train(num_episodes=self.max_stage_episodes, model_path=self.weights_path)

        # Each stage has its own hyperparameter_set, so its latest run folder is the one just trained
        model_file = latest_model_path(cfg.hyperparameter_set)
        if model_file.is_file():
            self.weights_path = model_file
        else:
            print(f"[Curriculum] Warning: stage '{label}' saved no model; keeping {self.weights_path}")

        best = (
            f"steps={summary['steps']} best_eval_score={summary['best_eval_score']:+.4f}"
            if "best_eval_score" in summary
            else f"best_mean_100={self.highest_reward[0]:.2f}"
        )
        message = (
            f"{label}: episodes={summary['episodes']} mean_reward={summary['mean_reward']:.2f} "
            f"goal_rate={summary['goal_rate']:.1f}% {best} weights={self.weights_path}"
        )
        print(f"[Curriculum] {message}")
        with open(self.run_path / "curriculum.log", "a") as log_file:
            log_file.write(message + "\n")

    def learn(self) -> Path:
        self.init_weights()
        stage_numbers = {stage: idx for idx, (stage, _) in enumerate(CurriculumConfig.stages, start=1)}
        for stage in self.stages:
            i = stage_numbers[stage]
            if stage != "self":
                self._train_stage(stage, f"s{i}_{stage}", first=(stage == self.stages[0]))
            else:
                for round_idx in range(1, self.self_play_rounds + 1):
                    if self.weights_path not in self._opponent_pool:
                        self._opponent_pool.append(self.weights_path)
                    self._train_stage(
                        stage,
                        f"s{i}_self_r{round_idx}",
                        opponent_path=self.weights_path,
                        opponent_pool=list(self._opponent_pool),
                        first=(stage == self.stages[0] and round_idx == 1),
                    )

        final_path = self.run_path / f"model_{self.model_name}.pt"
        shutil.copyfile(self.weights_path, final_path)
        print(f"\n[Curriculum] Done. Final model: {final_path}")
        return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum learning: none -> heuristic bot -> self-play")
    parser.add_argument("--agent", choices=("ppo", "ppo_v2", "dqn"), default="ppo", help="Learning agent to train")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help=f"Episodes without a new best before a stage ends (default: {CurriculumConfig.patience}; not for ppo_v2)",
    )
    parser.add_argument(
        "--self_play_rounds",
        type=int,
        default=CurriculumConfig.self_play_rounds,
        help="Self-play rounds (opponent pool updates after each)",
    )
    parser.add_argument(
        "--max_stage_episodes",
        type=int,
        default=CurriculumConfig.max_stage_episodes,
        help="Optional hard cap on episodes per stage / round",
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Parallel envs (only for --agent ppo_v2)")
    parser.add_argument("--model_name", type=str, default=None, help="Run / model name (default: agent's)")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Checkpoint to start from (defaults --start_stage to 'self')"
    )
    parser.add_argument(
        "--start_stage",
        choices=("none", "heuristic", "self"),
        default=None,
        help="Stage to start from (default: 'self' when --model_path is given, else 'none')",
    )
    parser.add_argument(
        "--league", nargs="+", default=(), help="Frozen .pt checkpoints and/or folders mixed into self-play"
    )
    parser.add_argument(
        "--self_play_steps",
        type=int,
        default=CurriculumConfig.ppo_v2_self_play_steps,
        help="Env steps per self-play round (ppo_v2 only)",
    )
    args = parser.parse_args()

    agent_cls = {"ppo": PPOAgent, "ppo_v2": PPOv2Agent, "dqn": DuelingDDQNAgent}[args.agent]
    cfg = replace(PPOv2Config(), num_envs=args.num_envs) if (args.agent == "ppo_v2" and args.num_envs is not None) else None
    Curriculum(
        agent_cls,
        patience=args.patience,
        self_play_rounds=args.self_play_rounds,
        max_stage_episodes=args.max_stage_episodes,
        model_name=args.model_name,
        config=cfg,
        model_path=args.model_path,
        start_stage=args.start_stage or ("self" if args.model_path is not None else "none"),
        league=args.league,
        self_play_steps=args.self_play_steps,
    ).learn()
