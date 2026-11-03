"""Main entry point for interactive 2D Rocket League / Sideswipe simulation."""

import sys
import os

import pygame
from src.config.sim_config import SCREEN_WIDTH, SCREEN_HEIGHT, SIM_HZ, ENABLE_ORANGE_CAR
from src.sim.core.actions import CarAction
from src.sim.core.control import CarControl
from src.sim.core.simulation import Simulation
from src.sim.visualization.renderer import Renderer
from src.sim.visualization.event_handling import handle_event


def play_interactive(enable_orange: bool = True):
    """Launch the Pygame interactive sandbox with 2D vector controls and Octane physics."""
    blue_control = CarControl(control_type="human", team="blue")
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
    blue_control = CarControl(control_type="heuristic_bot", team="blue", sim=sim)
    orange_control = CarControl(control_type="heuristic_bot", team="orange", sim=sim)

    renderer = Renderer(screen)

    running = True
    while running:
        dt = 1.0 / SIM_HZ

        running = handle_event(sim, renderer)

        # --- Input Mapping: 2D Direction Vector ---
        blue_action = blue_control.get_action()
        orange_action = orange_control.get_action()

        # --- Simulation Step ---
        sim.step(action=blue_action, dt=dt, action_orange=orange_action)
        sim.get_state_norm(as_flat_array=True)

        # --- Render ---
        renderer.render(sim)
        pygame.display.flip()
        clock.tick(SIM_HZ)

    pygame.quit()


def play_headless(steps: int = 300, enable_orange: bool = True):
    """Run headless simulation loop for automated benchmark or verification."""
    print(f"Running headless simulation for {steps} steps (Orange car: {enable_orange})...")
    sim = Simulation(enable_orange=enable_orange)
    action = CarAction(dir_x=1.0, dir_y=0.2, boost=True)

    for step in range(steps):
        sim.step(action, 1.0 / SIM_HZ)
        if step % 60 == 0:
            state = sim.get_state()
            line = (
                f"Step {step:03d} | Blue Pos: ({state['car']['position'][0]:.2f}, {state['car']['position'][1]:.2f}) | "
                f"Ball Pos: ({state['ball']['position'][0]:.2f}, {state['ball']['position'][1]:.2f}) | "
                f"Ball Spin: {state['ball']['angular_velocity']:+5.1f} rad/s"
            )
            if "car_orange" in state:
                line += f" | Orange Pos: ({state['car_orange']['position'][0]:.2f}, {state['car_orange']['position'][1]:.2f}) [{state['car_orange']['bot_state']}]"
            print(line)

    print("Headless simulation benchmark completed successfully!")
