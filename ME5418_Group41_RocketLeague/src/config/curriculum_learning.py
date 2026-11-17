# curriculum_learning
"""Curriculum learning settings: stage order, opponents and when a stage ends.

Read defaults straight off the class (`CurriculumConfig.patience`), or build a variant with
`dataclasses.replace(CurriculumConfig(), patience=500)`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumConfig:
    # Stages, in training order, as (stage, env bot_type). "self" needs an orange car in the env
    # ("heuristic_bot"); the curriculum mixes frozen self-play checkpoints with the heuristic bot.
    stages: tuple[tuple[str, str], ...] = (
        ("none", "none"),
        ("heuristic", "heuristic_bot"),
        ("self", "heuristic_bot"),
    )
    self_play_rounds: int = 2  # Self-play rounds; the opponent pool is updated with the newest best model after each

    # Adversarial self-play opponent sampling & policy stochasticity
    stochastic_opponent: bool = False  # Sample opponent actions stochastically rather than greedy argmax
    latest_self_prob: float = 0.70  # Probability of facing the latest self-play checkpoint
    past_self_prob: float = 0.15  # Probability of facing an earlier self-play checkpoint (when >= 2 exist)
    heuristic_prob: float = 0.15  # Probability of facing the scripted heuristic bot to prevent forgetting

    # Plateau detection (ends a stage / self-play round)
    patience: int = 3000  # Episodes without a new best rolling mean return before the stage ends (D3QN / PPO v1)
    ppo_v2_patience: int = 4000  # Higher patience for 8-env PPO v2 (parallel burst variance + separate critic)
    reward_window: int = 100  # Episodes in the rolling mean return (matches the agents' best-model metric)
    min_episodes: int = 20  # Warm-up before a stage can set a best / plateau (matches TrainingLogger)
    max_stage_episodes: int | None = None  # Hard cap on episodes per stage / round (None: until plateau)

    # Environment
    env_id: str = "rocketleague_curriculum_v1"  # Gym id the curriculum registers its env under
