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
    self_play_rounds: int = 5  # Self-play rounds; the opponent is refreshed to the newest best model after each

    # Plateau detection (ends a stage / self-play round)
    patience: int = 1000  # Episodes without a new best rolling mean return before the stage ends
    reward_window: int = 100  # Episodes in the rolling mean return (matches the agents' best-model metric)
    min_episodes: int = 20  # Warm-up before a stage can set a best / plateau (matches TrainingLogger)
    max_stage_episodes: int | None = None  # Hard cap on episodes per stage / round (None: until plateau)

    # Environment
    env_id: str = "rocketleague_curriculum_v1"  # Gym id the curriculum registers its env under
