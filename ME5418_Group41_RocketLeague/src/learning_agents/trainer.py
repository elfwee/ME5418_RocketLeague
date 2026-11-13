"""The training loop shared by every agent.

`run_training` owns the episode skeleton (reset, step, bookkeeping, best-model and periodic
checkpoints, dashboard); the agent only supplies its learning rule through the `Learner` hooks:

    action = learner.act(state)                     # choose an env action for this decision
    learner.observe(state, reward, next_state, terminated, truncated)   # learn from the step
    learner.progress()                              # e.g. "Eps = 0.951", printed per episode
    metrics = learner.end_episode()                 # per-episode update -> the two extra CSV columns
"""

import itertools
from typing import Protocol

import gymnasium as gym
import torch
from torch import nn

from src.learning_agents.dashboard import DashboardPanel, save_training_dashboard
from src.learning_agents.env_factory import obs_to_tensor, scored_goal
from src.learning_agents.eval_loop import summarize_episodes
from src.learning_agents.run_dirs import RunPaths, save_periodic_checkpoint
from src.learning_agents.training_logger import TrainingLogger


class Learner(Protocol):
    model: nn.Module  # weights saved as the best model and periodic checkpoints
    metric_columns: tuple[str, str]  # names of the two agent-specific CSV columns
    dashboard_title: str

    def act(self, state: torch.Tensor): ...

    def observe(
        self, state: torch.Tensor, reward: float, next_state: torch.Tensor, terminated: bool, truncated: bool
    ) -> None: ...

    def progress(self) -> str: ...

    def end_episode(self) -> tuple[float, float]: ...

    def dashboard_panels(self, history: tuple[list[float], list[float]]) -> tuple[DashboardPanel, DashboardPanel]: ...


def run_training(
    env: gym.Env,
    learner: Learner,
    paths: RunPaths,
    num_episodes: int | None,
    checkpoint_interval: int,
) -> dict[str, float]:
    """Train `learner` on `env` for `num_episodes` episodes (forever when None); returns the run summary."""
    logger = TrainingLogger(paths.log_file, paths.metrics_file, extra_columns=learner.metric_columns)
    history: tuple[list[float], list[float]] = ([], [])
    step_count = 0

    def save_graph(dpi: int = 150):
        exploration, loss = learner.dashboard_panels(history)
        save_training_dashboard(
            paths.graph_file,
            learner.dashboard_title,
            logger.rewards,
            logger.steps,
            logger.goals,
            exploration=exploration,
            loss=loss,
            dpi=dpi,
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
                action = learner.act(state)
                next_state, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                if scored_goal(info, terminated, reward):
                    goal = 1

                new_state = obs_to_tensor(next_state)
                learner.observe(state, reward, new_state, terminated, truncated)
                step_count += 1
                state = new_state

                if not env.unwrapped.isopen:
                    break

            mean_reward_100, goal_rate_100 = logger.record_episode(episode_reward, goal)
            print(
                f"Episode {episode:4d}: Reward = {episode_reward:7.2f} | "
                f"Mean(100) = {mean_reward_100:7.2f} | GoalRate(100) = {goal_rate_100:5.1f}% | "
                f"{learner.progress()} | Steps = {step_count}"
            )

            metrics = learner.end_episode()
            for values, value in zip(history, metrics, strict=True):
                values.append(value)
            logger.write_metrics(episode, step_count, extras=metrics)

            # Save best model based on 100-episode rolling mean reward
            if logger.is_new_best(episode, num_episodes):
                logger.log_new_best(episode, step_count)
                torch.save(learner.model.state_dict(), paths.model_file)

            # Periodic snapshot of the current model, independent of the best-model file
            save_periodic_checkpoint(
                learner.model, paths.checkpoint_dir, paths.model_name, episode, checkpoint_interval
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
