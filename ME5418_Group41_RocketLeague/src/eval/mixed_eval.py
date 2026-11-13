"""
Mixed-policy evaluation: pit two (possibly different) trained policies against each other.

Blue plays its own policy; Orange plays its policy through the mirrored observation the env
provides in `control_car="both"` mode. Any combination of Dueling DDQN / Double DQN / DQN / PPO
checkpoints works, since each network's architecture is inferred from its checkpoint.
"""

from collections.abc import Callable
from pathlib import Path

import torch

from src.config.nn_config import DQNConfig, PPOConfig
from src.learning_agents.agent_common import (
    device,
    infer_dqn_arch,
    infer_ppo_arch,
    latest_model_path,
    load_state_dict_if_exists,
    make_env,
    resolve_eval_episodes,
    run_eval_episodes,
)
from src.neural_nets.dueling_ddqn_torch import DQN
from src.neural_nets.ppo_torch import ActorCritic
from src.paths import RUNS_DIR

NUM_FLAT_ACTIONS = 36  # Discrete(36) = 9 directions x jump x boost


def _normalize_policy_name(name: str) -> str:
    n = name.strip().lstrip("-").lower()
    if n in ("ddqn", "dueling_ddqn"):
        return "dueling_ddqn"
    if n in ("double_dqn", "dqn", "ppo"):
        return n
    raise ValueError(f"Unsupported policy type '{name}'. Choose from: dueling_ddqn, double_dqn, dqn, ppo.")


def _load_ppo_net(num_states: int, ckpt_file: str) -> tuple[ActorCritic, bool]:
    """Build an `ActorCritic` matching the checkpoint (or the config defaults); returns (net, loaded)."""
    hidden_dim = PPOConfig.hidden_size
    factored = getattr(PPOConfig, "factored_action", True)
    action_dim: int | list[int] = list(PPOConfig.action_dims) if factored else NUM_FLAT_ACTIONS

    state_dict = load_state_dict_if_exists(ckpt_file)
    if state_dict is not None:
        hidden_dim, ckpt_action_dim = infer_ppo_arch(state_dict)
        if ckpt_action_dim is not None:
            action_dim = ckpt_action_dim

    net = ActorCritic(num_states, action_dim, hidden_dim).to(device)
    if state_dict is not None:
        net.load_state_dict(state_dict)
    return net, state_dict is not None


def _load_dqn_net(num_states: int, ckpt_file: str, policy_type: str) -> tuple[DQN, bool]:
    """Build a `DQN` matching the checkpoint (or the config defaults); returns (net, loaded)."""
    hidden_dim = DQNConfig.hidden_size
    action_dim = NUM_FLAT_ACTIONS
    dueling = policy_type == "dueling_ddqn"

    state_dict = load_state_dict_if_exists(ckpt_file)
    if state_dict is not None:
        hidden_dim, action_dim, dueling = infer_dqn_arch(state_dict)

    net = DQN(num_states, action_dim, hidden_dim, enable_dueling_dqn=dueling).to(device)
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
    is_ppo = policy_type == "ppo"
    # Default to the checkpoint of this policy's most recent training run
    ckpt_file = str(ckpt_override) if ckpt_override is not None else str(latest_model_path(policy_type))

    if is_ppo:
        net, loaded = _load_ppo_net(num_states, ckpt_file)
    else:
        net, loaded = _load_dqn_net(num_states, ckpt_file, policy_type)
    net.eval()

    tag = f"[Mix Eval ({team_label}: {'PPO' if is_ppo else policy_type})]"
    if loaded:
        print(f"{tag} Loaded trained weights from: {ckpt_file}")
    else:
        print(f"{tag} Warning: Checkpoint '{ckpt_file}' not found; using initial weights.")

    def select_ppo(obs: torch.Tensor):
        with torch.no_grad():
            logits, _ = net(obs.unsqueeze(0))
            if net.is_factored:
                return [int(l.squeeze(0).argmax().item()) for l in logits]
            return int(logits.squeeze(0).argmax().item())

    def select_dqn(obs: torch.Tensor):
        with torch.no_grad():
            return int(net(obs.unsqueeze(0)).squeeze(0).argmax().item())

    return select_ppo if is_ppo else select_dqn


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
            action_repeat=DQNConfig.action_repeat,
            num_episodes=resolve_eval_episodes(num_episodes, render, save_gif, DQNConfig.eval_episodes),
            gif_path=gif_path if save_gif else None,
            gif_tag="DQN",
        )
    finally:
        env.close()

    return {"blue_policy": blue_pol, "orange_policy": orange_pol, **metrics}
