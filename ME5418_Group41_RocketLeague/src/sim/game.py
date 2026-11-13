"""Main entry point for interactive 2D Rocket League / Sideswipe simulation."""

import pygame

from src.config.sim_config import SCREEN_HEIGHT, SCREEN_WIDTH, SIM_HZ
from src.sim.control import HeuristicController, HumanController
from src.sim.core.actions import CarAction
from src.sim.core.simulation import Simulation
from src.sim.visualization.event_handling import handle_event
from src.sim.visualization.renderer import Renderer


def play_interactive(enable_orange: bool = True, control_car: str = "blue"):
    """Launch the Pygame interactive sandbox with 2D vector controls and Octane physics.

    Args:
        enable_orange: Whether to spawn the Orange car in the arena.
        control_car: Which car the human player controls:
            - 'blue': Human controls Blue car (WASD/Arrows), Orange is Heuristic Bot.
            - 'orange': Human controls Orange car (WASD/Arrows), Blue is Heuristic Bot.
            - 'none' / 'bot_vs_bot': Both cars are controlled by Heuristic Bots.
    """
    pygame.init()
    pygame.joystick.init()

    # Initialize gamepad if connected
    gamepad = None
    if pygame.joystick.get_count() > 0:
        gamepad = pygame.joystick.Joystick(0)
        gamepad.init()
        print(f"Gamepad detected: {gamepad.get_name()}")

    pygame.display.set_caption("Rocket League Sideswipe 2D - Suspension Physics Sandbox")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    sim = Simulation(enable_orange=enable_orange)

    # Configure controllers based on control_car preference
    if control_car == "blue":
        blue_control = HumanController()
        orange_control = HeuristicController("orange", sim) if enable_orange else None
        print("Player control: BLUE car (WASD/Arrows/Space/Shift) | Opponent: ORANGE Heuristic Bot")
    elif control_car == "orange":
        blue_control = HeuristicController("blue", sim)
        orange_control = HumanController() if enable_orange else None
        print("Player control: ORANGE car (WASD/Arrows/Space/Shift) | Opponent: BLUE Heuristic Bot")
    else:
        blue_control = HeuristicController("blue", sim)
        orange_control = HeuristicController("orange", sim) if enable_orange else None
        print("Bot vs Bot match: BLUE Heuristic Bot vs ORANGE Heuristic Bot")

    renderer = Renderer(screen)

    running = True
    while running:
        dt = 1.0 / SIM_HZ

        running = handle_event(sim, renderer, blue_control, orange_control)

        # --- Input Mapping: 2D Direction Vector ---
        blue_action = blue_control.get_action()
        orange_action = orange_control.get_action() if orange_control is not None else CarAction()

        # --- Simulation Step ---
        sim.step(action=blue_action, dt=dt, action_orange=orange_action)
        if sim.goal_scored_this_step is not None:
            blue_control.reset()
            if orange_control is not None:
                orange_control.reset()

        # --- Render ---
        renderer.render(sim)
        pygame.display.flip()
        clock.tick(SIM_HZ)

    pygame.quit()


def play_headless(steps: int = 300, enable_orange: bool = True):
    """Run headless simulation loop for automated benchmark or verification."""
    print(f"Running headless simulation for {steps} steps (Orange car: {enable_orange})...")
    sim = Simulation(enable_orange=enable_orange)
    orange_control = HeuristicController("orange", sim) if enable_orange else None
    action = CarAction(dir_x=1.0, dir_y=0.2, boost=True)

    for step in range(steps):
        orange_action = orange_control.get_action() if orange_control is not None else CarAction()
        sim.step(action, 1.0 / SIM_HZ, action_orange=orange_action)
        if sim.goal_scored_this_step is not None and orange_control is not None:
            orange_control.reset()
        if step % 60 == 0:
            state = sim.get_state()
            blue_pos = state["blue_car"]["position"]
            ball_pos = state["ball"]["position"]
            line = (
                f"Step {step:03d} | Blue Pos: ({blue_pos[0]:.2f}, {blue_pos[1]:.2f}) | "
                f"Ball Pos: ({ball_pos[0]:.2f}, {ball_pos[1]:.2f}) | "
                f"Ball Spin: {state['ball']['angular_velocity']:+5.1f} rad/s"
            )
            if "car_orange" in state:
                orange_pos = state["car_orange"]["position"]
                bot_state = (
                    orange_control.bot.current_state
                    if (orange_control is not None and orange_control.bot is not None)
                    else "IDLE"
                )
                line += f" | Orange Pos: ({orange_pos[0]:.2f}, {orange_pos[1]:.2f}) [{bot_state}]"
            print(line)

    print("Headless simulation benchmark completed successfully!")
