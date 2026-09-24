"""Unit tests verifying random non-overlapping spawns for cars and ball."""
import unittest
import math
from src.core.simulation import Simulation
from src.core.actions import CarAction
from src.config import BALL_RADIUS, CAR_WIDTH, CORNER_RADIUS, ARENA_SEGMENT_RADIUS
from src.rocket_league_env import RocketLeagueEnv
import gymnasium as gym


class TestRandomSpawns(unittest.TestCase):
    """Verify random spawns of cars and ball according to arena and team boundary rules."""

    def test_random_spawns_two_cars_and_ball_bounds_and_no_overlap(self):
        """Verify 500 random resets adhere to field boundaries, non-overlap, and goal post clearance."""
        sim = Simulation(enable_orange=True, random_spawn=True)

        for _ in range(500):
            sim.reset(reset_scores=True)

            bx_car, by_car = sim.car.position
            ox_car, oy_car = sim.car_orange.position
            bx, by = sim.ball.position

            # 1. Blue Car: on the ground in the left half of field, not in goal post
            self.assertGreaterEqual(bx_car, 7.8, f"Blue car X ({bx_car:.2f}) too close to left fillet/goal")
            self.assertLessEqual(bx_car, 16.8, f"Blue car X ({bx_car:.2f}) crossed into right half")
            self.assertAlmostEqual(by_car, sim.spawn_y, delta=0.05, msg="Blue car must be on the ground at ride height")
            self.assertEqual(sim.car.facing_x, 1, "Blue car must face Right (+1)")
            self.assertAlmostEqual(sim.car.body.angle, 0.0, delta=0.05, msg="Blue car must be horizontal")

            # 2. Orange Car: on the ground in the right half of field, not in goal post
            self.assertGreaterEqual(ox_car, 18.2, f"Orange car X ({ox_car:.2f}) crossed into left half")
            self.assertLessEqual(ox_car, 27.2, f"Orange car X ({ox_car:.2f}) too close to right fillet/goal")
            self.assertAlmostEqual(oy_car, sim.spawn_y, delta=0.05, msg="Orange car must be on the ground at ride height")
            self.assertEqual(sim.car_orange.facing_x, -1, "Orange car must face Left (-1)")
            self.assertAlmostEqual(sim.car_orange.body.angle, math.pi, delta=0.05, msg="Orange car must face Left (pi)")

            # Cars must not overlap each other
            self.assertGreaterEqual(ox_car - bx_car, 1.4, "Blue and Orange cars must not overlap")

            # 3. Ball: randomly within the boundary of the whole field, not inside goal post
            # Clearance from goal lines (x_left = 4.5, x_right = 30.5)
            self.assertGreater(bx - BALL_RADIUS, sim.arena.x_left, f"Ball ({bx:.2f}) clipped inside Left goal")
            self.assertLess(bx + BALL_RADIUS, sim.arena.x_right, f"Ball ({bx:.2f}) clipped inside Right goal")
            # Vertical clearance
            self.assertGreater(by - BALL_RADIUS, sim.arena.y_floor, f"Ball Y ({by:.2f}) clipped below floor")
            self.assertLess(by + BALL_RADIUS, sim.arena.y_ceil, f"Ball Y ({by:.2f}) clipped above ceiling")

            # Corner fillet clearance
            r_eff = BALL_RADIUS + ARENA_SEGMENT_RADIUS + 0.05
            max_corner_dist = CORNER_RADIUS - r_eff
            if bx < 7.0 and by < 4.0:
                self.assertLessEqual(math.hypot(bx - 7.0, by - 4.0), max_corner_dist, "Ball clipped bottom-left fillet")
            if bx < 7.0 and by > 14.0:
                self.assertLessEqual(math.hypot(bx - 7.0, by - 14.0), max_corner_dist, "Ball clipped top-left fillet")
            if bx > 28.0 and by < 4.0:
                self.assertLessEqual(math.hypot(bx - 28.0, by - 4.0), max_corner_dist, "Ball clipped bottom-right fillet")
            if bx > 28.0 and by > 14.0:
                self.assertLessEqual(math.hypot(bx - 28.0, by - 14.0), max_corner_dist, "Ball clipped top-right fillet")

            # 4. Ball and cars must not spawn overlapping each other (clearance >= 2.2m)
            dist_blue = math.hypot(bx - bx_car, by - by_car)
            dist_orange = math.hypot(bx - ox_car, by - oy_car)
            self.assertGreaterEqual(dist_blue, 2.2, f"Ball overlapped Blue car: dist={dist_blue:.2f}m")
            self.assertGreaterEqual(dist_orange, 2.2, f"Ball overlapped Orange car: dist={dist_orange:.2f}m")

    def test_random_spawns_single_car_mode(self):
        """Verify random spawning works seamlessly when Orange car is disabled."""
        sim = Simulation(enable_orange=False, random_spawn=True)

        for _ in range(200):
            sim.reset(reset_scores=True)

            bx_car, by_car = sim.car.position
            bx, by = sim.ball.position

            self.assertIsNone(sim.car_orange)
            self.assertGreaterEqual(bx_car, 7.8)
            self.assertLessEqual(bx_car, 16.8)
            self.assertAlmostEqual(by_car, sim.spawn_y, delta=0.05)

            dist_blue = math.hypot(bx - bx_car, by - by_car)
            self.assertGreaterEqual(dist_blue, 2.2, f"Ball overlapped Blue car in solo mode: dist={dist_blue:.2f}m")

    def test_fixed_kickoff_mode_when_random_spawn_false(self):
        """Verify that passing random_spawn=False restores standard deterministic kickoff spots."""
        sim = Simulation(enable_orange=True, random_spawn=False)

        # Kickoff round 0: defensive spot
        self.assertAlmostEqual(sim.car.position[0], 8.5, places=2)
        self.assertAlmostEqual(sim.car_orange.position[0], 26.5, places=2)
        self.assertAlmostEqual(sim.ball.position[0], sim.center_x, places=2)
        self.assertAlmostEqual(sim.ball.position[1], sim.ball_spawn_y, places=2)

        # Kickoff round 1: attack spot
        sim.reset(random_spawn=False)
        self.assertAlmostEqual(sim.car.position[0], 13.5, places=2)
        self.assertAlmostEqual(sim.car_orange.position[0], 21.5, places=2)
        self.assertAlmostEqual(sim.ball.position[0], sim.center_x, places=2)
        self.assertAlmostEqual(sim.ball.position[1], sim.ball_spawn_y, places=2)

    def test_gym_environment_reset_random_spawn(self):
        """Verify Gymnasium RocketLeagueEnv resets with random spawns by default and fixed on request."""
        env = gym.make("rocket-league-v1", render_mode=None, bot_type="bot_level_3")

        # 1. Default reset is random
        for _ in range(50):
            obs, _ = env.reset()
            sim = env.unwrapped.sim
            bx_car, by_car = sim.car.position
            ox_car, oy_car = sim.car_orange.position
            bx, by = sim.ball.position

            self.assertGreaterEqual(bx_car, 7.8)
            self.assertLessEqual(bx_car, 16.8)
            self.assertGreaterEqual(ox_car, 18.2)
            self.assertLessEqual(ox_car, 27.2)
            self.assertGreaterEqual(math.hypot(bx - bx_car, by - by_car), 2.2)
            self.assertGreaterEqual(math.hypot(bx - ox_car, by - oy_car), 2.2)

        # 2. Options can request fixed kickoff
        obs, _ = env.reset(options={"random_spawn": False})
        sim = env.unwrapped.sim
        self.assertAlmostEqual(sim.ball.position[0], sim.center_x, places=2)
        self.assertAlmostEqual(sim.ball.position[1], sim.ball_spawn_y, places=2)
        env.close()


if __name__ == "__main__":
    unittest.main()
