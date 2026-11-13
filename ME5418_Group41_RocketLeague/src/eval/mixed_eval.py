"""
Mixed-policy evaluation: pit two (possibly different) trained policies against each other.

Blue plays its own policy; Orange plays its policy through the mirrored observation the env
provides in `control_car="both"` mode. Any combination of Dueling DDQN / Double DQN / DQN / PPO
checkpoints works, since each network's architecture is inferred from its checkpoint.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from src.config.nn_config import DQNConfig, PPOConfig
from src.gym_env.action_codec import NUM_DISCRETE_ACTIONS
from src.learning_agents.agent_common import (
    device,
    latest_model_path,
    make_env,
    resolve_eval_episodes,
    run_eval_episodes,
)
from src.neural_nets.checkpoint import DQN_KIND, PPO, PolicySpec, build_network, infer_spec, load_state_dict
from src.neural_nets.policy import greedy_action
from src.paths import RUNS_DIR


def _normalize_policy_name(name: str) -> str:
    n = name.strip().lstrip("-").lower()
    if n in ("ddqn", "dueling_ddqn"):
        return "dueling_ddqn"
    if n in ("double_dqn", "dqn", "ppo"):
        return n
    raise ValueError(f"Unsupported policy type '{name}'. Choose from: dueling_ddqn, double_dqn, dqn, ppo.")


def _default_spec(policy_type: str, num_states: int) -> PolicySpec:
    """The architecture `policy_type` trains with, used when its checkpoint does not exist."""
    if policy_type == "ppo":
        action_dim = list(PPOConfig.action_dims) if PPOConfig.factored_action else NUM_DISCRETE_ACTIONS
        return PolicySpec(PPO, num_states, PPOConfig.hidden_size, action_dim)
    return PolicySpec(
        DQN_KIND, num_states, DQNConfig.hidden_size, NUM_DISCRETE_ACTIONS, dueling=policy_type == "dueling_ddqn"
    )


def _load_net(policy_type: str, num_states: int, ckpt_file: str) -> tuple[nn.Module, bool]:
    """Network matching the checkpoint (or the config defaults when it is missing); returns (net, loaded)."""
    spec = _default_spec(policy_type, num_states)
    state_dict = load_state_dict(ckpt_file, device)
    if state_dict is not None:
        ckpt_spec = infer_spec(state_dict)
        spec = replace(
            ckpt_spec, action_dim=ckpt_spec.action_dim if ckpt_spec.action_dim is not None else spec.action_dim
        )
    net = build_network(spec).to(device)
    if state_dict is not None:
        net.load_state_dict(state_dict)
    return net, state_dict is not None


def _build_policy_selector(
    team_label: str,
    policy_type: str,
    ckpt_override: str | Path | None,
    num_states: int,
) -> Callable[[torch.Tensor], int | list[int]]:
    """Load `policy_type`'s network and return a greedy `observation -> env action` function."""
    # Default to the checkpoint of this policy's most recent training run
    ckpt_file = str(ckpt_override) if ckpt_override is not None else str(latest_model_path(policy_type))
    net, loaded = _load_net(policy_type, num_states, ckpt_file)
    net.eval()

    tag = f"[Mix Eval ({team_label}: {'PPO' if policy_type == 'ppo' else policy_type})]"
    if loaded:
        print(f"{tag} Loaded trained weights from: {ckpt_file}")
    else:
        print(f"{tag} Warning: Checkpoint '{ckpt_file}' not found; using initial weights.")

    def select(obs: torch.Tensor) -> int | list[int]:
        return greedy_action(net, obs)

    return select


def evaluate_mixed_policies(
    blue_policy: str = "dueling_ddqn",
    orange_policy: str = "ppo",
    blue_model_path: str | Path | None = None,
    orange_model_path: str | Path | None = None,
    num_episodes: int | None = DQNConfig.eval_episodes,
    render: bool = False,
    max_episode_steps: int | None = None,
    save_gif: bool = False,
    gif_path: str | Path | None = None,
    random_spawn: bool = True,
) -> dict[str, float]:
    """Evaluate two potentially different policies head-to-head (Blue policy vs. mirrored Orange policy).

    Supports any combination of ('dueling_ddqn', 'double_dqn', 'dqn', 'ppo') for Blue and Orange.
    """
    blue_pol = _normalize_policy_name(blue_policy)
    orange_pol = _normalize_policy_name(orange_policy)

    env = make_env(
        bot_type=DQNConfig.bot_type,
        discrete_action=True,
        control_car="both",
        render=render,
        save_gif=save_gif,
        random_spawn=random_spawn,
        max_episode_steps=max_episode_steps,
        action_repeat=DQNConfig.action_repeat,
    )
    try:
        num_states = env.observation_space.shape[0]
        select_blue = _build_policy_selector("Blue", blue_pol, blue_model_path, num_states)
        select_orange = _build_policy_selector("Orange", orange_pol, orange_model_path, num_states)

        if save_gif and gif_path is None:
            gif_path = RUNS_DIR / f"eval_mix_{blue_pol}_vs_{orange_pol}.gif"

        metrics = run_eval_episodes(
            env,
            select_blue,
            select_orange,
            num_episodes=resolve_eval_episodes(num_episodes, render, save_gif, DQNConfig.eval_episodes),
            gif_path=gif_path if save_gif else None,
            gif_tag="DQN",
        )
    finally:
        env.close()

    return {"blue_policy": blue_pol, "orange_policy": orange_pol, **metrics}
