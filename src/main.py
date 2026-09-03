"""Main entry point for interactive 2D Rocket League / Sideswipe simulation."""
import sys
import os
import math
import pygame
from src.config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, SIM_HZ
)
from src.core.actions import CarAction
from src.core.simulation import Simulation
from src.visualization.renderer import Renderer


def run_interactive():
    """Launch the Pygame interactive sandbox with 2D vector controls and Octane physics."""
    pygame.init()
    pygame.joystick.init()

    # Initialize gamepad if connected
    gamepad = None
    if pygame.joystick.get_count() > 0:
        gamepad = pygame.joystick.Joystick(0)
        gamepad.init()
        print(f"Gamepad detected: {gamepad.get_name()}")

    pygame.display.set_caption("Rocket League Sideswipe 2D - Refined Physics Sandbox (V2)")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()

    sim = Simulation()
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

        # --- Input Mapping: 2D Direction Vector ---
        action = CarAction()
        keys = pygame.key.get_pressed()

        dir_x = 0.0
        dir_y = 0.0

        # Keyboard directional input: WASD or Arrow Keys
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dir_x += 1.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dir_x -= 1.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dir_y += 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dir_y -= 1.0

        # Gamepad analog stick input (if connected)
        if gamepad is not None:
            gx = gamepad.get_axis(0)
            gy = -gamepad.get_axis(1)  # Invert stick Y so +Y is up
            if abs(gx) > 0.15 or abs(gy) > 0.15:
                dir_x = gx
                dir_y = gy

        action.dir_x = dir_x
        action.dir_y = dir_y
        action.clamp()

        # Jump: Space bar or Gamepad Button A (button 0)
        action.jump = bool(keys[pygame.K_SPACE] or (gamepad and gamepad.get_button(0)))

        # Rocket Boost: Shift, O, J, or Gamepad Button B / Trigger (buttons 1, 5)
        action.boost = bool(
            keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or
            keys[pygame.K_o] or keys[pygame.K_j] or
            (gamepad and (gamepad.get_button(1) or gamepad.get_button(5)))
        )

        # --- Simulation Step ---
        sim.step(action, dt)

        # --- Render ---
        renderer.render(sim)
        pygame.display.flip()
        clock.tick(SIM_HZ)

    pygame.quit()


def run_headless(steps: int = 300):
    """Run headless simulation loop for automated benchmark or verification."""
    print(f"Running headless simulation for {steps} steps...")
    sim = Simulation()
    action = CarAction(dir_x=1.0, dir_y=0.2, boost=True)

    for step in range(steps):
        sim.step(action, 1.0 / SIM_HZ)
        if step % 60 == 0:
            state = sim.get_state()
            print(f"Step {step:03d} | Car Pos: ({state['car']['position'][0]:.2f}, {state['car']['position'][1]:.2f}) | "
                  f"Ball Pos: ({state['ball']['position'][0]:.2f}, {state['ball']['position'][1]:.2f}) | "
                  f"Boost: {state['car']['boost']:.1f}%")

    print("Headless simulation benchmark completed successfully!")


if __name__ == '__main__':
    if "--headless" in sys.argv:
        run_headless()
    else:
        run_interactive()
