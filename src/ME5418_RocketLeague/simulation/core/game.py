"""Main entry point for interactive 2D Rocket League / Sideswipe simulation."""
# import sys
# import os

# Ensure repository root is on sys.path when invoked directly
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from ME5418_RocketLeague.config.sim_config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, SIM_HZ, ENABLE_ORANGE_CAR
)
from ME5418_RocketLeague.simulation.core.control import CarControl
from ME5418_RocketLeague.simulation.core.simulation import Simulation
from ME5418_RocketLeague.visualization.renderer import Renderer

class Game():
    def __init__(self):
        pass

    def run_interactive(self, enable_orange: bool = True):
        """Launch the Pygame interactive sandbox with 2D vector controls and Octane physics."""
        pygame.init()
        pygame.display.set_caption("Rocket League Sideswipe 2D - Suspension Physics Sandbox")
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()

        sim = Simulation(enable_orange=enable_orange)
        orange_control = CarControl(control_type="human", team="orange")
        blue_control = CarControl(control_type="none", team="blue", sim=sim)
        renderer = Renderer(screen)

        running = True
        while running:
            dt = 1.0 / SIM_HZ

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        sim.reset()
                    elif event.key == pygame.K_b:
                        # Toggle Orange bot on/off
                        sim.set_orange_enabled(not sim.enable_orange)
                    elif event.key == pygame.K_l:
                        # Toggle 45-degree boundary raycasts on/off
                        renderer.show_raycasts = not renderer.show_raycasts
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        renderer.handle_click(event.pos)

            # --- Simulation Step ---
            sim.step(action_blue=blue_control.get_action(), 
                     action_orange=orange_control.get_action(),
                     dt=dt)

            # --- Render ---
            renderer.render(sim)
            pygame.display.flip()
            clock.tick(SIM_HZ)

        pygame.quit()


    def run_headless(self, steps: int = 300, enable_orange: bool = True):
        """Run headless simulation loop for automated benchmark or verification."""
        print(f"Running headless simulation for {steps} steps (Orange car: {enable_orange})...")
        sim = Simulation(enable_orange=enable_orange)
        orange_control = CarControl(control_type="level3", team="orange", sim=sim)
        blue_control = CarControl(control_type="level3", team="blue", sim=sim)

        for step in range(steps):
            sim.step(action_blue=blue_control.get_action(), action_orange=orange_control.get_action(), dt=1.0 / SIM_HZ)
            if step % 60 == 0:
                state = sim.get_state()
                line = (
                    f"Step {step:03d} | Blue Pos: ({state['car']['position'][0]:.2f}, {state['car']['position'][1]:.2f}) | "
                    f"Ball Pos: ({state['ball']['position'][0]:.2f}, {state['ball']['position'][1]:.2f}) | "
                    f"Ball Spin: {state['ball']['angular_velocity']:+5.1f} rad/s"
                )
                if "car_orange" in state:
                    line += f" | Orange Pos: ({state['car_orange']['position'][0]:.2f}, {state['car_orange']['position'][1]:.2f})"
                print(line)

        print("Headless simulation benchmark completed successfully!")

    def run_demo(self):
        """Run visualised game"""
        pygame.init()
        pygame.display.set_caption("Rocket League Sideswipe 2D - Suspension Physics Sandbox")
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        clock = pygame.time.Clock()
        renderer = Renderer(screen)

        sim = Simulation(enable_orange=True)
        orange_control = CarControl(control_type="level3", team="orange", sim=sim)
        blue_control = CarControl(control_type="level3", team="blue", sim=sim)

        running = True
        while running:
            dt = 1.0 / SIM_HZ

            # --- Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        sim.reset()
                    elif event.key == pygame.K_b:
                        # Toggle Orange bot on/off
                        sim.set_orange_enabled(not sim.enable_orange)
                    elif event.key == pygame.K_l:
                        # Toggle 45-degree boundary raycasts on/off
                        renderer.show_raycasts = not renderer.show_raycasts
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        renderer.handle_click(event.pos)

            # --- Simulation Step ---
            sim.step(action_blue=blue_control.get_action(), 
                     action_orange=orange_control.get_action(),
                     dt=dt)

            # --- Render ---
            renderer.render(sim)
            pygame.display.flip()
            clock.tick(SIM_HZ)

        pygame.quit()

if __name__ == '__main__':
    game = Game
    enable_orange = ENABLE_ORANGE_CAR
    if "--no-orange" in sys.argv:
        enable_orange = False
    elif "--orange" in sys.argv:
        enable_orange = True

    if "--headless" in sys.argv:
        game.run_headless(enable_orange=enable_orange)
    else:
        game.run_interactive(enable_orange=enable_orange)
