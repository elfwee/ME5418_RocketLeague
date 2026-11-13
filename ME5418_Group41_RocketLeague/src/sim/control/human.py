"""Keyboard control (WASD / arrows, Space, Shift / O / J)."""

import pygame

from src.sim.core.actions import CarAction


class HumanController:
    """Reads the keyboard each frame; idles when no pygame window is open."""

    def __init__(self):
        self.action = CarAction()

    def get_action(self) -> CarAction:
        if not pygame.get_init() or not pygame.display.get_init():
            self.action = CarAction(dir_x=0.0, dir_y=0.0, jump=False, boost=False)
            return self.action

        keys = pygame.key.get_pressed()

        # Keyboard directional input: WASD or Arrow Keys
        dir_x = 0.0
        dir_y = 0.0
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            dir_x += 1.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            dir_x -= 1.0
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            dir_y += 1.0
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            dir_y -= 1.0
        self.action.dir_x = dir_x
        self.action.dir_y = dir_y

        # Jump: Space bar
        self.action.jump = bool(keys[pygame.K_SPACE])

        # Rocket Boost: Shift, O, or J
        self.action.boost = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] or keys[pygame.K_o] or keys[pygame.K_j])
        return self.action

    def reset(self):
        self.action = CarAction(dir_x=0, dir_y=0, jump=False, boost=False)
