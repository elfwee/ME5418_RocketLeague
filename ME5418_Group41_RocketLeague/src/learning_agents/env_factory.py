"""Environment creation for training and evaluation, plus the tensor/device conventions the agents share."""

import gymnasium as gym
import torch

from src.config.gym_config import GymConfig
from src.gym_env.rocket_league_env import RocketLeagueEnv  # noqa: F401  (import registers the gym env id)
from src.gym_env.wrappers import ActionRepeat

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalize_control_car(control_car: str) -> str:
    """Map the bot-vs-bot aliases onto the env's `"both"` mode."""
    return "both" if control_car in ("bot_vs_bot", "bots", "both") else control_car


def make_env(
    bot_type: str,
    discrete_action: bool,
    control_car: str,
    render: bool = False,
    save_gif: bool = False,
    random_spawn: bool = True,
    max_episode_steps: int | None = None,
    env_id: str = GymConfig.gym_id,
    action_repeat: int = 1,
) -> gym.Env:
    """Create the Rocket League env; renders to a window, to RGB frames (for GIFs), or not at all.

    With `action_repeat > 1` the env is wrapped in `ActionRepeat`: each `step` holds the action for
    that many sim frames. `max_episode_steps` always counts sim frames.
    """
    if render:
        render_mode = "human"
    elif save_gif:
        render_mode = "rgb_array"
    else:
        render_mode = None

    make_kwargs = {}
    if max_episode_steps is not None:
        make_kwargs["max_episode_steps"] = max_episode_steps

    env = gym.make(
        env_id,
        render_mode=render_mode,
        bot_type=bot_type,
        discrete_action=discrete_action,
        control_car=control_car,
        random_spawn=random_spawn,
        **make_kwargs,
    )
    return ActionRepeat(env, action_repeat) if action_repeat > 1 else env


def obs_to_tensor(obs) -> torch.Tensor:
    return torch.tensor(obs, dtype=torch.float32, device=device)


def scored_goal(info: dict, terminated: bool, reward: float) -> bool:
    """True when the decision step ended with the controlled car scoring."""
    return bool(info.get("is_success", False) or (terminated and reward > 0.0))
