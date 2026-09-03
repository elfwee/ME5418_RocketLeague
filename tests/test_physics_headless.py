"""Headless unit tests verifying bidirectional facing, wheelie physics, gravity, and turtle recovery."""
import unittest
import math
from src.core.simulation import Simulation
from src.core.actions import CarAction
from src.config import GRAVITY, CAR_MAX_BOOST


class TestPhysicsHeadless(unittest.TestCase):

    def setUp(self):
        self.sim = Simulation()

    def test_ball_freefall_gravity(self):
        """Verify the ball accelerates downwards under heavy gravity (28 m/s^2)."""
        initial_y = self.sim.ball.position[1]
        action = CarAction()

        for _ in range(30):
            self.sim.step(action, dt=1.0 / 60.0)

        curr_y = self.sim.ball.position[1]
        vy = self.sim.ball.velocity[1]

        self.assertLess(curr_y, initial_y, "Ball should lose altitude during freefall")
        self.assertLess(vy, -5.0, "Ball vertical velocity should be briskly negative under heavy gravity")

    def test_ball_floor_bounce(self):
        """Verify the ball bounces off the floor and rebounds upward."""
        self.sim.ball.reset(16.0, 3.0, vx=0.0, vy=-10.0)
        action = CarAction()

        bounced = False
        for _ in range(60):
            self.sim.step(action, dt=1.0 / 60.0)
            if self.sim.ball.velocity[1] > 2.0:
                bounced = True
                break

        self.assertTrue(bounced, "Ball should rebound upward after contacting floor")

    def test_car_ground_driving(self):
        """Verify car drives forward along the ground when 2D direction (dir_x=1.0) is applied."""
        self.sim.reset()
        for _ in range(30):
            self.sim.step(CarAction(), dt=1.0 / 60.0)

        self.assertTrue(self.sim.car.is_grounded, "Car should be resting on the ground")

        start_x = self.sim.car.position[0]
        drive_action = CarAction(dir_x=1.0, dir_y=0.0)

        for _ in range(60):
            self.sim.step(drive_action, dt=1.0 / 60.0)

        end_x = self.sim.car.position[0]
        vx = self.sim.car.velocity[0]

        self.assertGreater(end_x, start_x + 1.0, "Car should have driven forward horizontally")
        self.assertGreater(vx, 2.0, "Car forward speed should be positive")

    def test_bidirectional_facing(self):
        """Verify car dynamically flips facing direction between Left and Right."""
        self.sim.reset()
        self.assertEqual(self.sim.car.facing_x, 1, "Default facing direction should be Right (+1)")
        self.assertGreater(self.sim.car.forward_vector[0], 0.0, "Forward vector should point Right")

        # Command Left
        left_action = CarAction(dir_x=-1.0, dir_y=0.0)
        for _ in range(10):
            self.sim.step(left_action, dt=1.0 / 60.0)

        self.assertEqual(self.sim.car.facing_x, -1, "Car should face Left (-1) after left input")
        self.assertLess(self.sim.car.forward_vector[0], 0.0, "Forward vector should point Left")
        self.assertLess(self.sim.car.nose_position[0], self.sim.car.position[0], "Nose position should be to the left of center")

        # Command Right
        right_action = CarAction(dir_x=1.0, dir_y=0.0)
        for _ in range(10):
            self.sim.step(right_action, dt=1.0 / 60.0)

        self.assertEqual(self.sim.car.facing_x, 1, "Car should face Right (+1) after right input")
        self.assertGreater(self.sim.car.forward_vector[0], 0.0, "Forward vector should point Right")

    def test_ground_wheelie_pitch_up_no_flight(self):
        """Verify holding UP on ground rears nose up into wheelie without taking off into the air."""
        self.sim.reset()
        for _ in range(25):
            self.sim.step(CarAction(), dt=1.0 / 60.0)

        self.assertTrue(self.sim.car.is_grounded)
        up_action = CarAction(dir_x=0.0, dir_y=1.0, boost=False)

        # Hold UP for 60 steps (1 full second)
        for _ in range(60):
            self.sim.step(up_action, dt=1.0 / 60.0)

        pitch_deg = abs(math.degrees(self.sim.car.body.angle))
        car_y = self.sim.car.position[1]
        car_vy = self.sim.car.velocity[1]

        self.assertGreater(pitch_deg, 65.0, "Pressing UP on ground should lift nose into a wheelie")
        self.assertLess(car_y, 3.2, "Car must remain on the ground during a wheelie without booster")
        self.assertLess(abs(car_vy), 1.0, "Vertical velocity must remain near zero (no infinite flight)")

    def test_ground_wheelie_facing_left_nose_up(self):
        """Verify holding UP while facing Left lifts the nose UP (not the tail)."""
        self.sim.reset()
        self.sim.car.reset(15.0, 2.1, angle=0.0, facing_x=-1)
        for _ in range(25):
            self.sim.step(CarAction(), dt=1.0 / 60.0)

        up_action = CarAction(dir_x=0.0, dir_y=1.0, boost=False)
        for _ in range(70):
            self.sim.step(up_action, dt=1.0 / 60.0)

        nose_y = self.sim.car.nose_position[1]
        car_y = self.sim.car.position[1]
        tail_y = self.sim.car.tail_position[1]
        fwd_y = self.sim.car.forward_vector[1]

        self.assertGreater(nose_y, car_y, "Nose must point UP above car center")
        self.assertLess(tail_y, car_y, "Tail must point DOWN near the ground")
        self.assertGreater(fwd_y, 0.9, "Forward vector must point upwards")

    def test_rwd_pitch_cutoff_at_90_degrees(self):
        """Verify that at 90-degree vertical pitch, ground drive traction cannot lift car into sky."""
        self.sim.reset()
        for _ in range(25):
            self.sim.step(CarAction(), dt=1.0 / 60.0)

        self.sim.car.body.angle = math.pi / 2.0
        self.sim.car.body.angular_velocity = 0.0

        action = CarAction(dir_x=0.0, dir_y=1.0, boost=False)
        for _ in range(40):
            self.sim.step(action, dt=1.0 / 60.0)

        vy = self.sim.car.velocity[1]
        y_pos = self.sim.car.position[1]

        self.assertLess(vy, 1.0, "Car should not gain vertical flight from ground drive at 90 deg pitch")
        self.assertLess(y_pos, 3.5, "Car should remain on ground without boost")

    def test_car_boost_rocket_flight(self):
        """Verify rocket boost thrust propels car upward like a rocket when pitched at 90 degrees."""
        self.sim.reset()
        self.sim.car.body.angle = math.pi / 2.0
        self.sim.car.body.angular_velocity = 0.0

        boost_action = CarAction(dir_x=0.0, dir_y=1.0, boost=True)
        for _ in range(30):
            self.sim.step(boost_action, dt=1.0 / 60.0)

        vy = self.sim.car.velocity[1]
        y_pos = self.sim.car.position[1]

        self.assertGreater(vy, 4.0, "Boost thrust must accelerate car upward against gravity")
        self.assertGreater(y_pos, 3.0, "Car should take off into the air under boost")

    def test_auto_right_turtle_recovery_preserves_facing(self):
        """Verify turtle recovery inverts car upright while preserving facing direction (Left and Right)."""
        # Test 1: Facing Right turtled
        self.sim.reset()
        self.sim.car.reset(16.0, 2.3, angle=math.pi, facing_x=1)
        for _ in range(20):
            self.sim.step(CarAction(), dt=1.0 / 60.0)

        self.assertEqual(self.sim.car.facing_x, 1, "Should maintain facing Right")
        self.assertGreater(math.cos(self.sim.car.body.angle), 0.8, "Car should invert upright facing Right")

        # Test 2: Facing Left turtled
        self.sim.reset()
        self.sim.car.reset(16.0, 2.3, angle=0.0, facing_x=-1)
        for _ in range(20):
            self.sim.step(CarAction(), dt=1.0 / 60.0)

        self.assertEqual(self.sim.car.facing_x, -1, "Should maintain facing Left")
        self.assertLess(math.cos(self.sim.car.body.angle), -0.8, "Car should invert upright facing Left")

    def test_resultant_vector_with_gravity_and_thrust(self):
        """Verify airborne resultant motion reflects vector summation of forward boost and downward gravity."""
        self.sim.reset()
        # Spawn airborne at (16.0, 10.0) facing Right
        self.sim.car.reset(16.0, 10.0, angle=0.0, facing_x=1)
        self.sim.car.boost_amount = 100.0

        # Fire horizontal boost along +X (no vertical input)
        action = CarAction(dir_x=1.0, dir_y=0.0, boost=True)
        for _ in range(40):
            self.sim.step(action, dt=1.0 / 60.0)

        vx, vy = self.sim.car.velocity

        # Forward velocity should be strongly positive from thrust
        self.assertGreater(vx, 8.0, "Horizontal boost thrust should drive car forward along +X")
        # Vertical velocity should be strongly negative from gravity pulling down
        self.assertLess(vy, -8.0, "Gravity vector should continuously accelerate car downward along -Y")
        # Resultant angle should point down-right
        resultant_angle = math.atan2(vy, vx)
        self.assertLess(resultant_angle, 0.0, "Resultant velocity vector should tilt downward toward gravity")

    def test_wheelie_no_turtle_glitch(self):
        """Verify continuous vertical wheelie does not trigger false turtle resets or glitches."""
        self.sim.reset()
        for _ in range(25):
            self.sim.step(CarAction(), dt=1.0 / 60.0)

        up_action = CarAction(dir_x=0.0, dir_y=1.0)
        for step in range(120):  # 2 full seconds of holding UP
            self.sim.step(up_action, dt=1.0 / 60.0)
            # Turtled time must remain 0.0 throughout the entire wheelie
            self.assertEqual(self.sim.car.turtled_time, 0.0, f"Turtle timer falsely triggered at step {step}")

        pitch_deg = abs(math.degrees(self.sim.car.body.angle))
        self.assertGreater(pitch_deg, 70.0, "Car should remain in a vertical wheelie")
        self.assertLess(self.sim.car.position[1], 3.2, "Car should remain on the floor without taking off")

    def test_boost_up_right_and_up_left_alignment(self):
        """Verify that when boost is used with Up-Right or Up-Left, the Blue vector (heading) rotates to it."""
        # 1. Test Up-Right with Boost from ground
        self.sim.reset()
        for _ in range(25):
            self.sim.step(CarAction(), dt=1.0 / 60.0)

        action_ur = CarAction(dir_x=1.0, dir_y=1.0, boost=True)
        for _ in range(40):
            self.sim.step(action_ur, dt=1.0 / 60.0)

        fwd_ur = self.sim.car.forward_vector
        self.assertGreater(fwd_ur[0], 0.55, "Forward X should point toward +X (Right)")
        self.assertGreater(fwd_ur[1], 0.55, "Forward Y should point toward +Y (Up)")
        self.assertGreater(self.sim.car.position[1], 2.4, "Car should take off into the air diagonally")

        # 2. Test Up-Left with Boost from ground
        self.sim.reset()
        self.sim.car.reset(16.0, 2.1, angle=0.0, facing_x=-1)
        for _ in range(25):
            self.sim.step(CarAction(), dt=1.0 / 60.0)

        action_ul = CarAction(dir_x=-1.0, dir_y=1.0, boost=True)
        for _ in range(45):
            self.sim.step(action_ul, dt=1.0 / 60.0)

        fwd_ul = self.sim.car.forward_vector
        self.assertLess(fwd_ul[0], -0.55, "Forward X should point toward -X (Left)")
        self.assertGreater(fwd_ul[1], 0.55, "Forward Y should point toward +Y (Up)")
        self.assertGreater(self.sim.car.position[1], 2.4, "Car should take off into the air diagonally")

    def test_grounded_regardless_of_angle(self):
        """Verify that when touching the ground, the car is considered grounded regardless of angle."""
        for name, deg in [('flat', 0), ('wheelie_85', 85), ('stoppie_m85', -85),
                          ('vertical_90', 90), ('vertical_m90', -90), ('upside_down_180', 180)]:
            with self.subTest(angle=name):
                self.sim.reset()
                self.sim.car._update_recovery = lambda action, dt: None
                self.sim.car.reset(16.0, 2.5, angle=math.radians(deg), facing_x=1)
                for _ in range(40):
                    self.sim.step(CarAction(), dt=1.0 / 60.0)
                self.assertTrue(self.sim.car.is_grounded, f"Car should be grounded at {name} ({deg} deg)")

    def test_boost_recovery_and_boost_usage_on_ground(self):
        """Verify boost recovers when grounded, but recovery stops when boost is used on ground."""
        self.sim.reset()
        for _ in range(25):
            self.sim.step(CarAction(), dt=1.0 / 60.0)

        # 1. Boost recovery occurs while grounded and not boosting
        self.sim.car.boost_amount = 50.0
        self.sim.step(CarAction(boost=False), dt=1.0 / 60.0)
        self.assertGreater(self.sim.car.boost_amount, 50.0, "Boost should recover while grounded and idle")

        # 2. When boost is used while grounded, recovery must NOT happen (boost must drain)
        self.sim.car.boost_amount = 50.0
        self.sim.step(CarAction(dir_x=1.0, boost=True), dt=1.0 / 60.0)
        self.assertLess(self.sim.car.boost_amount, 50.0, "Boost must drain when used while grounded (no recovery)")

    def test_movement_when_boost_depleted_and_still_pressed(self):
        """Verify that when boost is at 0% and boost is still pressed, car moves at normal speed without boost effects."""
        # 1. Ground movement with diagonal aim and boost held at 0% boost
        self.sim.reset()
        self.sim.car.reset(10.0, self.sim.spawn_y, angle=0.0)
        self.sim.car.boost_amount = 0.0
        action = CarAction(dir_x=1.0, dir_y=1.0, boost=True)
        for _ in range(40):
            self.sim.step(action, dt=1.0 / 60.0)

        self.assertGreater(self.sim.car.velocity[0], 5.0, "Car should move forward on ground when boost held at 0%")
        self.assertFalse(self.sim.car.is_boosting, "Boosting effects should be inactive at 0% boost")

        # 2. Air movement when boost held at 0% boost
        self.sim.reset()
        self.sim.car.reset(10.0, 10.0, angle=0.0)
        self.sim.car.boost_amount = 0.0
        action_air = CarAction(dir_x=1.0, dir_y=0.0, boost=True)
        for _ in range(40):
            self.sim.step(action_air, dt=1.0 / 60.0)

        self.assertGreater(self.sim.car.velocity[0], 5.0, "Car should move forward in air when boost held at 0%")
        self.assertFalse(self.sim.car.is_boosting, "Boosting effects should be inactive at 0% boost")

    def test_diagonal_boost_climb_at_135_and_45_degrees(self):
        """Verify that boosting diagonally at +135 deg and +45 deg climbs high into the air."""
        # 1. Test +135 degrees (Up-Left)
        self.sim.reset()
        self.sim.car.reset(20.0, 5.0, angle=math.radians(135.0), facing_x=-1)
        action_135 = CarAction(dir_x=math.cos(math.radians(135.0)), dir_y=math.sin(math.radians(135.0)), boost=True)
        for _ in range(40):
            self.sim.step(action_135, dt=1.0 / 60.0)

        self.assertGreater(self.sim.car.position[1], 8.0, "Car should climb high at 135 deg")
        self.assertGreater(self.sim.car.velocity[1], 8.0, "Vertical velocity should be strongly positive at 135 deg")
        self.assertLess(self.sim.car.velocity[0], -8.0, "Horizontal velocity should be strongly negative (Left)")

        # 2. Test +45 degrees (Up-Right)
        self.sim.reset()
        self.sim.car.reset(10.0, 5.0, angle=math.radians(45.0), facing_x=1)
        action_45 = CarAction(dir_x=math.cos(math.radians(45.0)), dir_y=math.sin(math.radians(45.0)), boost=True)
        for _ in range(40):
            self.sim.step(action_45, dt=1.0 / 60.0)

        self.assertGreater(self.sim.car.position[1], 8.0, "Car should climb high at 45 deg")
        self.assertGreater(self.sim.car.velocity[1], 8.0, "Vertical velocity should be strongly positive at 45 deg")
        self.assertGreater(self.sim.car.velocity[0], 8.0, "Horizontal velocity should be strongly positive (Right)")


if __name__ == '__main__':
    unittest.main()
