"""Render clean game frames (no HUD) of two checkpoints playing, for slide backdrops."""
import os, sys
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.insert(0, os.getcwd())
import pygame
import src.config.render_config as rc
SCALE = float(os.environ.get("RENDER_SCALE", "1"))
rc.PPM *= SCALE; rc.SCREEN_WIDTH = int(rc.SCREEN_WIDTH * SCALE); rc.SCREEN_HEIGHT = int(rc.SCREEN_HEIGHT * SCALE)
from src.config.render_config import SCREEN_HEIGHT, SCREEN_WIDTH
from src.config.sim_config import SIM_HZ
from src.sim.control.policy import PolicyController
from src.sim.core.simulation import Simulation
from src.sim.visualization.renderer import Renderer

blue_path, orange_path, out_dir, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
os.makedirs(out_dir, exist_ok=True)
pygame.init()
pygame.display.set_mode((1, 1))
screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
r = Renderer(screen)
sim = Simulation(enable_orange=True, random_spawn=True, seed=seed)
blue = PolicyController("blue", sim, blue_path)
orange = PolicyController("orange", sim, orange_path)
dt = 1.0 / SIM_HZ
n = 0
for i in range(int(SIM_HZ * 40)):
    sim.step(action=blue.get_action(), dt=dt, action_orange=orange.get_action())
    if sim.goal_scored_this_step is not None:
        blue.reset(); orange.reset()
        print("goal", sim.goal_scored_this_step, i / SIM_HZ)
    if i % (SIM_HZ // 2) == 0:
        screen.fill((15, 18, 25))
        r._draw_arena(sim); r._draw_ball(sim); r._draw_car(sim.blue_car); r._draw_car(sim.orange_car)
        pygame.image.save(screen, f"{out_dir}/f{n:03d}.png"); n += 1
print("frames", n, "score", sim.score_blue, sim.score_orange)
