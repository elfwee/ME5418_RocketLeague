"""Main entry point for interactive 2D Rocket League / Sideswipe simulation."""

import os
import sys

import pygame
from src.config.sim_config import ENABLE_ORANGE_CAR, SCREEN_HEIGHT, SCREEN_WIDTH, SIM_HZ
from src.sim.core.actions import CarAction
from src.sim.core.control import CarControl
from src.sim.core.simulation import Simulation
from src.sim.visualization.event_handling import handle_event
from src.sim.visualization.renderer import Renderer


def play_interactive(
    enable_orange: bool = True,
    control_car: str = "blue",
    blue_agent: str | None = None,
    orange_agent: str | None = None,
    blue_model: str | None = None,
    orange_model: str | None = None,
):
    """Launch the Pygame interactive sandbox with 2D vector controls and Octane physics.

    Args:
        enable_orange: Whether to spawn the Orange car in the arena.
        control_car: Which car the human player controls (used if blue_agent / orange_agent not specified):
            - 'blue': Human controls Blue car (WASD/Arrows), Orange is Heuristic Bot.
            - 'orange': Human controls Orange car (WASD/Arrows), Blue is Heuristic Bot.
            - 'none' / 'bot_vs_bot': Both cars are controlled by Heuristic Bots.
        blue_agent: Explicit controller for Blue ('human', 'heuristic_bot', 'ppo', 'dueling_ddqn').
        orange_agent: Explicit controller for Orange ('human', 'heuristic_bot', 'ppo', 'dueling_ddqn').
        blue_model: Optional path to .pt model checkpoint for Blue car.
        orange_model: Optional path to .pt model checkpoint for Orange car.
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

    # If an orange agent is explicitly specified, ensure Orange is enabled
    if orange_agent is not None:
        enable_orange = True

    sim = Simulation(enable_orange=enable_orange)

    # Configure controllers
    if blue_agent is not None or orange_agent is not None:
        b_type = (blue_agent or ("human" if control_car == "blue" else "heuristic_bot")).lower()
        o_type = (orange_agent or "heuristic_bot").lower()

        blue_control = CarControl(control_type=b_type, team="blue", sim=sim, model_path=blue_model)
        orange_control = (
            CarControl(control_type=o_type, team="orange", sim=sim, model_path=orange_model)
            if enable_orange
            else None
        )
        print(f"Match Setup: BLUE ({b_type.upper()}) vs ORANGE ({o_type.upper() if enable_orange else 'DISABLED'})")
    elif control_car == "blue":
        blue_control = CarControl(control_type="human", team="blue")
        orange_control = (
            CarControl(control_type="heuristic_bot", team="orange", sim=sim)
            if enable_orange
            else None
        )
        print("Player control: BLUE car (WASD/Arrows/Space/Shift) | Opponent: ORANGE Heuristic Bot")
    elif control_car == "orange":
        blue_control = CarControl(control_type="heuristic_bot", team="blue", sim=sim)
        orange_control = (
            CarControl(control_type="human", team="orange")
            if enable_orange
            else None
        )
        print("Player control: ORANGE car (WASD/Arrows/Space/Shift) | Opponent: BLUE Heuristic Bot")
    else:
        blue_control = CarControl(control_type="heuristic_bot", team="blue", sim=sim)
        orange_control = (
            CarControl(control_type="heuristic_bot", team="orange", sim=sim)
            if enable_orange
            else None
        )
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
        sim.get_state_norm(as_flat_array=True)

        # --- Render ---
        renderer.render(sim)
        pygame.display.flip()
        clock.tick(SIM_HZ)

    pygame.quit()


def play_headless(
    steps: int = 300,
    enable_orange: bool = True,
    blue_agent: str | None = None,
    orange_agent: str | None = None,
    blue_model: str | None = None,
    orange_model: str | None = None,
):
    """Run headless simulation loop for automated benchmark or verification."""
    print(f"Running headless simulation for {steps} steps (Orange car: {enable_orange})...")
    sim = Simulation(enable_orange=enable_orange)

    b_type = (blue_agent or "heuristic_bot").lower()
    o_type = (orange_agent or "heuristic_bot").lower()

    blue_control = CarControl(control_type=b_type, team="blue", sim=sim, model_path=blue_model)
    orange_control = (
        CarControl(control_type=o_type, team="orange", sim=sim, model_path=orange_model)
        if enable_orange
        else None
    )

    for step in range(steps):
        blue_action = blue_control.get_action()
        orange_action = orange_control.get_action() if orange_control is not None else CarAction()
        sim.step(blue_action, 1.0 / SIM_HZ, action_orange=orange_action)
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
                line += f" | Orange Pos: ({orange_pos[0]:.2f}, {orange_pos[1]:.2f})"
            print(line)

    print("Headless simulation benchmark completed successfully!")
