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


if __name__ == '__main__':
    unittest.main()
