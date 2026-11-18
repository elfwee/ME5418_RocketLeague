# curriculum_learning
"""Curriculum learning settings: stage order, opponents and when a stage ends.

Read defaults straight off the class (`CurriculumConfig.patience`), or build a variant with
`dataclasses.replace(CurriculumConfig(), patience=500)`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumConfig:
    # Stages, in training order, as (stage, env bot_type). "self" needs an orange car in the env
    # ("heuristic_bot"); the curriculum then swaps the bot for a frozen copy of the last best model.
    stages: tuple[tuple[str, str], ...] = (
        ("none", "none"),
        ("heuristic", "heuristic_bot"),
        ("self", "heuristic_bot"),
    )
    self_play_rounds: int = 2  # Self-play rounds; the opponent is refreshed to the newest best model after each

    # Mixed self-play opponent pool, used by PPO v2 and whenever `--league` is given. Without either,
    # D3QN / PPO self-play is unchanged: always the frozen copy of the last best model.
    heuristic_prob: float = 0.15  # Episodes against the scripted heuristic bot (prevents forgetting)
    # Episodes against a frozen league checkpoint (`--league`, e.g. Dueling DDQN models; always greedy)
    league_prob: float = 0.35
    # The remaining episodes face the agent's own frozen checkpoints, split latest : earlier
    latest_self_prob: float = 0.70
    past_self_prob: float = 0.15

    # Plateau detection (ends a stage / self-play round). PPO v2 stages end on evaluation instead
    # (`PPOv2Config.eval_patience`), so `patience` only applies to D3QN / PPO.
    patience: int = 3000  # Episodes without a new best rolling mean return before the stage ends
    # PPO v2 self-play rounds run a fixed number of env steps (an evaluation-plateau stop fired on noise
    # against strong opponents); the round still keeps its best-evaluated checkpoint.
    ppo_v2_self_play_steps: int = 20_000_000
    reward_window: int = 100  # Episodes in the rolling mean return (matches the agents' best-model metric)
    min_episodes: int = 20  # Warm-up before a stage can set a best / plateau (matches TrainingLogger)
    max_stage_episodes: int | None = None  # Hard cap on episodes per stage / round (None: until plateau)

    # Environment
    env_id: str = "rocketleague_curriculum_v1"  # Gym id the curriculum registers its env under
