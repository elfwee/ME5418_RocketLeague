"""
Head-to-Head: pit a trained blue agent against orange (blue vs orange).

Orange is the heuristic bot by default, or a trained agent if given a model. Either way it plays
through the mirrored perspective, so it only sees what a blue car would see. Run it with a Pygame
window (render=True) or headlessly as fast as possible for a quick score tally.

    python -m src --head_to_head [--headless] [--random_spawn] [--seed N] [--blue PATH] [--orange PATH]
"""

import random
import time
from pathlib import Path
from typing import Dict, Optional

from src.config.sim_config import SCREEN_WIDTH, SCREEN_HEIGHT, SIM_HZ
from src.learning_agents.agent_control import AgentControl
from src.sim.core.simulation import Simulation
from src.sim.core.control import CarControl

def play_head_to_head(
    blue_model: Optional[str | Path] = None,
    orange_model: Optional[str | Path] = None,
    render: bool = True,
    max_goals: Optional[int] = None,
    max_seconds: Optional[float] = 600.0,
    random_spawn: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Play blue agent vs orange agent until `max_goals` total goals or `max_seconds` of sim time
    (either may be None; headless needs at least one limit). `orange_model=None` plays the heuristic
    bot on orange. `random_spawn` randomises every kickoff;
    `seed` makes those spawns repeatable and has no effect without it. Returns the final score."""
    if not render and max_goals is None and max_seconds is None:
        raise ValueError("Headless head-to-head needs max_goals or max_seconds, otherwise it never ends.")

    if seed is not None:
        random.seed(seed)

    sim = Simulation(enable_orange=True, random_spawn=random_spawn)

    if orange_model is None:
        # orange = OrangePerspectiveBot(sim)
        orange = CarControl(control_type="heuristic_bot", team = "orange", sim=sim)
    else:
        orange = AgentControl(team="orange", sim=sim, model_path=orange_model)

    if blue_model is None:
        blue = CarControl(control_type="heuristic_bot", team = "blue", sim=sim)
    else:
        blue = AgentControl(team="blue", sim=sim, model_path=blue_model)

    print(f"[HeadToHead] Blue:   {'heuristic bot' if blue_model is None else Path(blue_model).name}")
    print(f"[HeadToHead] Orange: {'heuristic bot' if orange_model is None else Path(orange_model).name}")

    if render:
        import pygame
        from src.sim.visualization.renderer import Renderer
        from src.sim.visualization.event_handling import handle_event

        pygame.init()
        pygame.display.set_caption("Rocket League Sideswipe 2D - Agent Head-to-Head")
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        renderer = Renderer(screen)

    dt = 1.0 / SIM_HZ
    sim_time = 0.0
    wall_start = time.perf_counter()
    running = True
    try:
        while running:
            if render:
                running = handle_event(sim, renderer, blue, orange)
                if not running:
                    break

            sim.step(action=blue.get_action(), dt=dt, action_orange=orange.get_action())
            sim_time += dt

            if sim.goal_scored_this_step is not None:
                # Sim auto-resets kickoff positions; restart the agents' decision clocks to match.
                blue.reset()
                orange.reset()
                print(
                    f"[HeadToHead] {sim_time:7.1f}s  GOAL {sim.goal_scored_this_step.upper():<6}  "
                    f"Blue {sim.score_blue} - {sim.score_orange} Orange"
                )

            if max_goals is not None and sim.score_blue + sim.score_orange >= max_goals:
                break
            if max_seconds is not None and sim_time >= max_seconds:
                break

            if render:
                renderer.render(sim)
                pygame.display.flip()
                clock.tick(SIM_HZ)
    finally:
        if render:
            pygame.quit()

    result = {"blue": sim.score_blue, "orange": sim.score_orange, "seconds": round(sim_time, 2)}
    print(
        f"[HeadToHead] Final: Blue {result['blue']} - {result['orange']} Orange "
        f"({result['seconds']}s sim, {time.perf_counter() - wall_start:.1f}s wall)"
    )
    return result
