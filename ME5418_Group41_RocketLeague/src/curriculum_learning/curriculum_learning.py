"""
Curriculum learning: train one policy through a sequence of increasingly hard opponents.

Stages:
    1. "none"      - no opponent; learn to drive to the ball and score.
    2. "heuristic" - against the scripted HeuristicBot.
    3. "self"      - self-play against a frozen copy of the last best model. Run as
                     `self_play_rounds` rounds; after each round the opponent is refreshed to the
                     newest best model.

Each stage (and each self-play round) is one ordinary training run of `PPOAgent` or
`DuelingDDQNAgent`, started from the previous stage's best weights, and stopped once the
100-episode rolling mean return has not improved for `patience` episodes.

The agents are steered through their config (`bot_type`, `env_id`, `hyperparameter_set`), and the
opponent / plateau stop is provided by `CurriculumEnv`, a subclass of `RocketLeagueEnv` registered
under its own gym id.

Usage:
    python -m src --curriculum --ppo|--dueling_ddqn|--double_dqn|--dqn [--patience N] [--self_play_rounds N]
    python -m src.curriculum_learning.curriculum_learning --agent ppo
    python -m src.curriculum_learning.curriculum_learning --agent dqn --patience 500 --self_play_rounds 3
"""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.config.gym_config import GymConfig
from src.config.nn_config import DQNConfig, PPOConfig
from src.gym_env.rocket_league_env import RocketLeagueEnv
from src.learning_agents.dueling_ddqn_agent import DuelingDDQNAgent
from src.learning_agents.env_factory import make_env
from src.learning_agents.ppo_agent import PPOAgent
from src.learning_agents.run_dirs import latest_model_path, make_run_dir
from src.neural_nets.ppo_torch import ActorCritic
from src.sim.control.policy import PolicyController

CURRICULUM_ENV_ID = "rocketleague_curriculum_v1"


class CurriculumEnv(RocketLeagueEnv):
    """`RocketLeagueEnv` with two curriculum hooks.

    - `opponent_path`: the orange car is driven by a frozen policy loaded from this checkpoint
      (`PolicyController`, mirrored so a blue-trained model can play orange) instead of the heuristic bot.
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
        opponent_action_repeat: int = PPOConfig.action_repeat,
        **kwargs,
    ):
        self.on_episode_end = on_episode_end
        self._episode_return = 0.0
        self._episode_steps = 0
        super().__init__(*args, **kwargs)

        if opponent_path is not None:
            if self.orange_control is None:
                raise ValueError("Self-play needs an orange car: create the env with bot_type='heuristic_bot'.")
            self.orange_control = PolicyController(
                "orange",
                self.sim,
                opponent_path,
                action_repeat=opponent_action_repeat,
            )

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

        obs, info = super().reset(seed=seed, options=options)
        if finished and self.on_episode_end is not None and self.on_episode_end(episode_return):
            self.isopen = False
        return obs, info


class Curriculum:
    MIN_EPISODES = 20  # warm-up before a stage can set a best / plateau (matches TrainingLogger)
    BOT_TYPES = {"none": "none", "heuristic": "heuristic_bot", "self": "heuristic_bot"}
    CONFIGS = {PPOAgent: PPOConfig, DuelingDDQNAgent: DQNConfig}

    def __init__(
        self,
        agent,
        patience: int = 1000,
        self_play_rounds: int = 5,
        max_stage_episodes: int | None = None,
        model_name: str | None = None,
        config: PPOConfig | DQNConfig | None = None,
    ) -> None:
        self.stages = ["none", "heuristic", "self"]
        self.patience = patience  # number of episodes to train for after last best model
        self.agent = agent  # either DuelingDDQNAgent or PPOAgent (the class)
        self.config = config if config is not None else self.CONFIGS[agent]()  # e.g. a DQN ablation variant
        self.self_play_rounds = self_play_rounds
        self.max_stage_episodes = max_stage_episodes
        self.model_name = model_name or self.config.hyperparameter_set
        self.highest_reward = [float("-inf"), 0]  # [highest_reward, episode]

        self.run_path: Path | None = None
        self.weights_path: Path | None = None

        # Per-stage state read by the env factory / episode callback
        self._stage_rewards: list[float] = []
        self._opponent_path: Path | None = None
        self._opponent_action_repeat = self.config.action_repeat

        gym.envs.registry.pop(CURRICULUM_ENV_ID, None)
        gym.register(
            id=CURRICULUM_ENV_ID,
            entry_point=self._make_env,
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
            opponent_action_repeat=self._opponent_action_repeat,
            **kwargs,
        )

    def _on_episode_end(self, episode_return: float) -> bool:
        self._stage_rewards.append(episode_return)
        episode = len(self._stage_rewards) - 1
        average_reward = float(np.mean(self._stage_rewards[-100:]))
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
            3. initialise weights in the structure using the same weight initialiser as the agents
            4. save .pt file in directory
        """
        self.run_path = make_run_dir(f"{self.model_name}_curriculum")

        is_ppo = issubclass(self.agent, PPOAgent)
        factored = is_ppo and self.config.factored_action
        env = make_env(bot_type="none", discrete_action=not factored, control_car="blue")
        try:
            obs_dim = env.observation_space.shape[0]
            if is_ppo:
                action_dim = [int(n) for n in env.action_space.nvec] if factored else env.action_space.n
                net = ActorCritic(obs_dim, action_dim, self.config.hidden_size)
            else:
                net = self.agent(self.config)._build_network(obs_dim, env.action_space.n)
        finally:
            env.close()

        self.weights_path = self.run_path / f"model_{self.model_name}_init.pt"
        torch.save(net.state_dict(), self.weights_path)
        print(f"[Curriculum] Saved initial weights to: {self.weights_path}")

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

    def _train_stage(self, stage: str, label: str, opponent_path: Path | None = None, first: bool = False):
        """One training run of the agent for `stage`, starting from `self.weights_path`."""
        self.highest_reward = [float("-inf"), 0]
        self._stage_rewards = []
        self._opponent_path = opponent_path

        cfg = replace(
            self.config,
            hyperparameter_set=f"{self.model_name}_{label}",
            bot_type=self.BOT_TYPES[stage],
            env_id=CURRICULUM_ENV_ID,
        )
        if first and isinstance(cfg, DQNConfig):
            # DQN drops to epsilon_finetune when given a checkpoint; the init weights are untrained
            cfg = replace(cfg, epsilon_finetune=cfg.epsilon_init)
        self._opponent_action_repeat = cfg.action_repeat
        agent = self.agent(cfg)

        opponent = f" vs {opponent_path}" if opponent_path is not None else ""
        print(f"\n[Curriculum] ===== Stage '{label}'{opponent} | start weights: {self.weights_path} =====")
        summary = agent.train(num_episodes=self.max_stage_episodes, model_path=self.weights_path)

        # Each stage has its own hyperparameter_set, so its latest run folder is the one just trained
        model_file = latest_model_path(cfg.hyperparameter_set)
        if model_file.is_file():
            self.weights_path = model_file
        else:
            print(f"[Curriculum] Warning: stage '{label}' saved no model; keeping {self.weights_path}")

        message = (
            f"{label}: episodes={summary['episodes']} mean_reward={summary['mean_reward']:.2f} "
            f"goal_rate={summary['goal_rate']:.1f}% best_mean_100={self.highest_reward[0]:.2f} "
            f"weights={self.weights_path}"
        )
        print(f"[Curriculum] {message}")
        with open(self.run_path / "curriculum.log", "a") as log_file:
            log_file.write(message + "\n")

    def learn(self) -> Path:
        self.init_weights()
        for i, stage in enumerate(self.stages, start=1):
            if stage != "self":
                self._train_stage(stage, f"s{i}_{stage}", first=(i == 1))
            else:
                for round_idx in range(1, self.self_play_rounds + 1):
                    last_best_model = self.weights_path
                    self._train_stage(stage, f"s{i}_self_r{round_idx}", opponent_path=last_best_model)

        final_path = self.run_path / f"model_{self.model_name}.pt"
        shutil.copyfile(self.weights_path, final_path)
        print(f"\n[Curriculum] Done. Final model: {final_path}")
        return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curriculum learning: none -> heuristic bot -> self-play")
    parser.add_argument("--agent", choices=("ppo", "dqn"), default="ppo", help="Learning agent to train")
    parser.add_argument("--patience", type=int, default=1000, help="Episodes without a new best before a stage ends")
    parser.add_argument("--self_play_rounds", type=int, default=5, help="Self-play rounds (opponent refreshes)")
    parser.add_argument(
        "--max_stage_episodes", type=int, default=None, help="Optional hard cap on episodes per stage / round"
    )
    parser.add_argument("--model_name", type=str, default=None, help="Run / model name (default: agent's)")
    args = parser.parse_args()

    Curriculum(
        PPOAgent if args.agent == "ppo" else DuelingDDQNAgent,
        patience=args.patience,
        self_play_rounds=args.self_play_rounds,
        max_stage_episodes=args.max_stage_episodes,
        model_name=args.model_name,
    ).learn()
