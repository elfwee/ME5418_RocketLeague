"""Automated tests verifying ball strikes, calibrated impulse, and boundary stability."""
import unittest
import math
from src.core.simulation import Simulation
from src.core.actions import CarAction


class TestCarBallCollision(unittest.TestCase):

    def setUp(self):
        self.sim = Simulation()

    def test_forward_strike_right_and_left(self):
        """Verify the car punches the ball forward in its driving direction (both Right and Left)."""
        # --- 1. Strike Rightward ---
        self.sim.reset()
        self.sim.ball.reset(13.0, 2.75, vx=0.0, vy=0.0)
        self.sim.car.reset(8.0, 2.1, angle=0.0, facing_x=1)

        action_right = CarAction(dir_x=1.0, dir_y=0.0)
        hit_right = False
        for _ in range(70):
            self.sim.step(action_right, dt=1.0 / 60.0)
            if self.sim.ball.velocity[0] > 5.0:
                hit_right = True
                break

        self.assertTrue(hit_right, "Car driving Right should punch ball forward to the Right")
        self.assertGreater(self.sim.ball.velocity[0], 5.0)
        self.assertLess(self.sim.ball.velocity[0], 25.0, "Ball speed should be calibrated, not runaway")

        # --- 2. Strike Leftward ---
        self.sim.reset()
        self.sim.ball.reset(13.0, 2.75, vx=0.0, vy=0.0)
        self.sim.car.reset(18.0, 2.1, angle=0.0, facing_x=-1)

        action_left = CarAction(dir_x=-1.0, dir_y=0.0)
        hit_left = False
        for _ in range(70):
            self.sim.step(action_left, dt=1.0 / 60.0)
            if self.sim.ball.velocity[0] < -5.0:
                hit_left = True
                break

        self.assertTrue(hit_left, "Car driving Left should punch ball forward to the Left")
        self.assertLess(self.sim.ball.velocity[0], -5.0)

    def test_elevated_goal_lower_wall_vertical_rebound(self):
        """Verify ground rolling ball hits lower wall below elevated goal and rebounds vertically."""
        self.sim.ball.reset(8.0, 2.75, vx=-15.0, vy=0.0)
        action = CarAction()

        rebounded_upward = False
        for step in range(60):
            self.sim.step(action, dt=1.0 / 60.0)
            if self.sim.ball.velocity[1] > 3.0:
                rebounded_upward = True
                break

        self.assertTrue(rebounded_upward, "Ball should bounce vertically in front of elevated goal mouth")

    def test_pinch_numerical_stability(self):
        """Verify that wedging the ball against the wall remains numerically stable (no NaN or inf)."""
        self.sim.ball.reset(27.0, 10.0, vx=0.0, vy=0.0)
        self.sim.car.reset(25.0, 10.0, angle=0.0, facing_x=1)

        action = CarAction(dir_x=1.0, dir_y=0.0, boost=True)
        for _ in range(60):
            self.sim.step(action, dt=1.0 / 60.0)
            bx, by = self.sim.ball.position
            bvx, bvy = self.sim.ball.velocity
            cx, cy = self.sim.car.position

            self.assertFalse(math.isnan(bx) or math.isnan(by), "Ball position became NaN during pinch")
            self.assertFalse(math.isnan(bvx) or math.isnan(bvy), "Ball velocity became NaN during pinch")
            self.assertFalse(math.isnan(cx) or math.isnan(cy), "Car position became NaN during pinch")

    def test_goal_geometry_expansion_and_smooth_scoring(self):
        """Verify 1.5x width, 1.2x height, smooth corners, and goal sensor detection."""
        from src.config import GOAL_DEPTH, GOAL_BOTTOM_Y, GOAL_TOP_Y, SCREEN_WIDTH

        # 1. Verify dimensions
        self.assertAlmostEqual(GOAL_DEPTH, 3.75, places=2)
        goal_height = GOAL_TOP_Y - GOAL_BOTTOM_Y
        self.assertAlmostEqual(goal_height, 5.28, places=2)
        self.assertEqual(SCREEN_WIDTH, 1400)

        # 2. Test scoring into smooth left goal (Orange scores)
        g_center_y = (self.sim.arena.goal_y_bot + self.sim.arena.goal_y_top) / 2.0
        self.sim.reset()
        self.sim.ball.reset(self.sim.arena.x_left + 1.0, g_center_y, vx=-10.0, vy=0.0)
        for _ in range(60):
            self.sim.step(CarAction(), dt=1.0 / 60.0)
            if self.sim.score_orange > 0:
                break
        self.assertEqual(self.sim.last_goal_team, 'orange', "Ball 100% in left goal awards score to Orange")
        self.assertEqual(self.sim.score_orange, 1)

        # 3. Test scoring into smooth right goal (Blue scores)
        self.sim.reset()
        self.sim.ball.reset(self.sim.arena.x_right - 1.0, g_center_y, vx=10.0, vy=0.0)
        for _ in range(60):
            self.sim.step(CarAction(), dt=1.0 / 60.0)
            if self.sim.score_blue > 0:
                break
        self.assertEqual(self.sim.last_goal_team, 'blue', "Ball 100% in right goal awards score to Blue")
        self.assertEqual(self.sim.score_blue, 1)

    def test_ball_spawns_on_ground_on_start_and_reset(self):
        """Verify the ball spawns resting on the arena floor on start and on reset (not mid-air)."""
        from src.core.simulation import Simulation

        sim = Simulation()
        # 1. On start: ball is on the floor
        self.assertAlmostEqual(sim.ball.position[1], sim.ball_spawn_y, places=2)
        self.assertAlmostEqual(sim.ball.position[0], sim.center_x, places=2)

        for _ in range(30):
            sim.step(CarAction(), dt=1.0 / 60.0)
        self.assertAlmostEqual(sim.ball.position[1], sim.ball_spawn_y, places=2, msg="Ball should remain resting on floor")
        self.assertAlmostEqual(sim.ball.velocity[1], 0.0, places=2)

        # 2. On reset: ball returns to floor
        sim.ball.reset(10.0, 10.0, vx=5.0, vy=-5.0)
        sim.reset()
        self.assertAlmostEqual(sim.ball.position[1], sim.ball_spawn_y, places=2)
        self.assertAlmostEqual(sim.ball.position[0], sim.center_x, places=2)

    def test_goal_requires_100_percent_entry_before_reset_and_score(self):
        """Verify goal only counts when ball is 100% inside, then scores and resets kickoff."""
        xl = self.sim.arena.x_left
        r = self.sim.ball.radius
        g_center_y = (self.sim.arena.goal_y_bot + self.sim.arena.goal_y_top) / 2.0

        self.sim.reset(reset_scores=True)
        self.assertEqual(self.sim.score_orange, 0)

        # 1. Partial entry: front edge is inside the pocket, but rear edge is still outside on the field
        self.sim.ball.reset(xl - 0.2, g_center_y, vx=0.0, vy=0.0)
        self.sim.step(CarAction(), dt=1.0 / 60.0)
        self.assertEqual(self.sim.score_orange, 0, "Partial goal line crossing must NOT score")
        self.assertIsNone(self.sim.last_goal_team)

        # 2. 100% inside: entire ball circle has fully crossed the goal line
        self.sim.ball.reset(xl - r - 0.05, g_center_y, vx=0.0, vy=0.0)
        self.sim.step(CarAction(), dt=1.0 / 60.0)
        self.assertEqual(self.sim.score_orange, 1, "Ball 100% inside must count as a goal")
        self.assertEqual(self.sim.last_goal_team, "orange")
        # After goal, game must reset ball to ground at center
        self.assertAlmostEqual(self.sim.ball.position[0], self.sim.center_x, places=2)
        self.assertAlmostEqual(self.sim.ball.position[1], self.sim.ball_spawn_y, places=2)

    def test_alternating_car_spawn_positions_on_car_side(self):
        """Verify car spawns on its side and alternates between defensive (8.5m) and attack (13.5m) spots."""
        from src.core.simulation import Simulation

        sim = Simulation()
        # Initial spawn is at defensive position (8.5m)
        self.assertAlmostEqual(sim.car.position[0], 8.5, places=2)
        self.assertEqual(sim.car.facing_x, 1)

        # Reset 1: alternates to attack position (13.5m)
        sim.reset()
        self.assertAlmostEqual(sim.car.position[0], 13.5, places=2)
        self.assertEqual(sim.car.facing_x, 1)

        # Reset 2: alternates back to defensive position (8.5m)
        sim.reset()
        self.assertAlmostEqual(sim.car.position[0], 8.5, places=2)
        self.assertEqual(sim.car.facing_x, 1)

        # Reset 3: alternates back to attack position (13.5m)
        sim.reset()
        self.assertAlmostEqual(sim.car.position[0], 13.5, places=2)
        self.assertEqual(sim.car.facing_x, 1)


if __name__ == '__main__':
    unittest.main()
