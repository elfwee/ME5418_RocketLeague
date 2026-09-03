"""Quantitative physics tests.

These lock in the invariants that the raycast-suspension refactor fixed. Each one compares
the simulation against a closed-form expectation rather than against hand-tuned numbers, so
a regression in the force model shows up as a physical error instead of a "feel" change.
"""
import math
import unittest

from src.config import (
    CAR_DRIVE_ACCEL, CAR_MAX_GROUND_SPEED, CAR_JUMP_SPEED,
    CAR_BOOST_ACCEL, CAR_MASS, BALL_MASS, CAR_BALL_RESTITUTION,
    GRAVITY_MAG, PHYSICS_SUBSTEPS, SIM_HZ,
)
from src.core.actions import CarAction
from src.core.simulation import Simulation

DT = 1.0 / SIM_HZ

# The flat floor segment only spans x in [5.5, 26.5]; outside that the car sits on a
# corner fillet and legitimately tilts to follow the contour. Start well inside it.
FLAT_START_X = 9.5


def settle(sim, frames=90):
    """Let the car come to rest on its suspension."""
    for _ in range(frames):
        sim.step(CarAction(), DT)


class TestGroundLocomotion(unittest.TestCase):
    """Drive force must not be cancelled by chassis friction dragging on the floor."""

    def test_acceleration_matches_configured_value(self):
        sim = Simulation()
        sim.car.reset(FLAT_START_X, sim.spawn_y)
        settle(sim)

        samples = []
        for _ in range(20):
            v0 = sim.car.velocity[0]
            sim.step(CarAction(dir_x=1.0), DT)
            samples.append((sim.car.velocity[0] - v0) / DT)

        mean_accel = sum(samples) / len(samples)
        self.assertAlmostEqual(
            mean_accel, CAR_DRIVE_ACCEL, delta=1.0,
            msg=f"Ground acceleration {mean_accel:.2f} should match CAR_DRIVE_ACCEL"
        )

    def test_reaches_configured_top_speed(self):
        sim = Simulation()
        sim.car.reset(FLAT_START_X, sim.spawn_y)
        settle(sim)

        for _ in range(60):
            sim.step(CarAction(dir_x=1.0), DT)

        self.assertLess(sim.car.position[0], 25.0, "Test must stay on the flat floor section")
        self.assertAlmostEqual(
            sim.car.velocity[0], CAR_MAX_GROUND_SPEED, delta=0.15,
            msg="Car must actually reach CAR_MAX_GROUND_SPEED on flat ground"
        )

    def test_partial_throttle_is_proportional(self):
        sim = Simulation()
        sim.car.reset(FLAT_START_X, sim.spawn_y)
        settle(sim)

        for _ in range(120):
            sim.step(CarAction(dir_x=0.5), DT)

        self.assertAlmostEqual(sim.car.velocity[0], 0.5 * CAR_MAX_GROUND_SPEED, delta=0.3)

    def test_car_rests_perfectly_still(self):
        sim = Simulation()
        settle(sim, 120)

        heights = []
        for _ in range(120):
            sim.step(CarAction(), DT)
            heights.append(sim.car.position[1])

        self.assertLess(max(heights) - min(heights), 1e-3, "Parked car must not jitter")
        self.assertLess(abs(sim.car.velocity[1]), 1e-6)
        self.assertTrue(sim.car.is_grounded)
        self.assertEqual(sim.car.wheel_contact_count, 2, "Both axles should find the floor")


class TestVerticalMotion(unittest.TestCase):
    """Nothing may secretly damp or add vertical velocity while the car is grounded."""

    def test_jump_delivers_exact_impulse(self):
        sim = Simulation()
        settle(sim)

        sim.step(CarAction(jump=True), DT)
        expected = CAR_JUMP_SPEED - GRAVITY_MAG * DT
        self.assertAlmostEqual(
            sim.car.velocity[1], expected, delta=0.05,
            msg="Jump impulse must not be eaten by downforce or spurious damping"
        )

    def test_jump_apex_matches_ballistics(self):
        sim = Simulation()
        settle(sim)
        y0 = sim.car.position[1]

        apex = y0
        for _ in range(200):
            sim.step(CarAction(jump=True), DT)
            apex = max(apex, sim.car.position[1])

        ideal = CAR_JUMP_SPEED ** 2 / (2.0 * GRAVITY_MAG)
        self.assertAlmostEqual(apex - y0, ideal, delta=0.1 * ideal)

    def test_freefall_is_pure_gravity(self):
        sim = Simulation()
        sim.car.reset(16.0, 13.0)
        sim.ball.reset(4.0, 13.0)

        sim.step(CarAction(), DT)
        v0 = sim.car.velocity[1]
        sim.step(CarAction(), DT)
        accel = (sim.car.velocity[1] - v0) / DT

        self.assertAlmostEqual(accel, -GRAVITY_MAG, delta=0.05)

    def test_boost_thrust_and_gravity_sum_exactly(self):
        sim = Simulation()
        sim.car.reset(16.0, 12.0, angle=0.0)
        sim.ball.reset(4.0, 12.0)

        v0 = sim.car.velocity
        sim.step(CarAction(dir_x=1.0, boost=True), DT)
        v1 = sim.car.velocity

        self.assertAlmostEqual((v1[0] - v0[0]) / DT, CAR_BOOST_ACCEL, delta=0.1)
        self.assertAlmostEqual((v1[1] - v0[1]) / DT, -GRAVITY_MAG, delta=0.1)


class TestAttitudeController(unittest.TestCase):
    """The heading controller must converge without overshoot or ringing."""

    def _converge(self, target_deg):
        sim = Simulation()
        sim.car.reset(16.0, 12.5)
        sim.ball.reset(4.0, 12.5)

        t = math.radians(target_deg)
        action = CarAction(dir_x=math.cos(t), dir_y=math.sin(t))
        errors = []
        for _ in range(140):
            sim.car.update(action, 1.0 / 240.0)
            sim.space.step(1.0 / 240.0)
            fx, fy = sim.car.forward_vector
            errors.append((math.degrees(math.atan2(fy, fx)) - target_deg + 180.0) % 360.0 - 180.0)
        return sim, errors

    def test_heading_converges_to_input_vector(self):
        for target in (30.0, 60.0, 90.0, -45.0, -90.0):
            with self.subTest(target=target):
                sim, errors = self._converge(target)
                self.assertFalse(sim.car.is_grounded, "Test must stay airborne")
                self.assertLess(abs(errors[-1]), 0.5,
                                f"Heading should settle on the commanded {target} deg vector")

    def test_heading_sweep_left_hemisphere(self):
        """Verify continuous sweep across +135 deg -> 180 deg -> -135 deg and back."""
        sim = Simulation()
        sim.car.reset(16.0, 12.5, angle=math.radians(135.0))
        sim.ball.reset(4.0, 12.5)

        # Forward sweep: +135 -> 180 -> -135
        for target_deg in (135.0, 180.0, -135.0):
            t = math.radians(target_deg)
            action = CarAction(dir_x=math.cos(t), dir_y=math.sin(t))
            sim.car.body.position = (16.0, 12.5)
            sim.car.body.velocity = (0.0, 0.0)
            for _ in range(140):
                sim.car.update(action, 1.0 / 240.0)
                sim.space.step(1.0 / 240.0)
            fx, fy = sim.car.forward_vector
            actual = (math.degrees(math.atan2(fy, fx)) + 180.0) % 360.0 - 180.0
            err = abs((actual - target_deg + 180.0) % 360.0 - 180.0)
            self.assertLess(err, 0.5, f"Sweep to {target_deg}° settled at {actual:.1f}°, err={err:.2f}°")
            self.assertEqual(sim.car.facing_x, -1, f"Should display Left animation at {target_deg}°")

        # Reverse sweep: -135 -> 180 -> +135
        for target_deg in (-135.0, 180.0, 135.0):
            t = math.radians(target_deg)
            action = CarAction(dir_x=math.cos(t), dir_y=math.sin(t))
            sim.car.body.position = (16.0, 12.5)
            sim.car.body.velocity = (0.0, 0.0)
            for _ in range(140):
                sim.car.update(action, 1.0 / 240.0)
                sim.space.step(1.0 / 240.0)
            fx, fy = sim.car.forward_vector
            actual = (math.degrees(math.atan2(fy, fx)) + 180.0) % 360.0 - 180.0
            err = abs((actual - target_deg + 180.0) % 360.0 - 180.0)
            self.assertLess(err, 0.5, f"Reverse sweep to {target_deg}° settled at {actual:.1f}°, err={err:.2f}°")
            self.assertEqual(sim.car.facing_x, -1, f"Should display Left animation at {target_deg}°")

    def test_no_overshoot_or_ringing(self):
        _, errors = self._converge(60.0)
        # Approaching 60 deg from 0 means the error starts negative and must never
        # cross meaningfully above zero.
        self.assertLess(max(errors), 0.5, "Attitude controller overshot its target")

        tail = errors[-40:]
        reversals = sum(
            1 for i in range(1, len(tail) - 1)
            if (tail[i] - tail[i - 1]) * (tail[i + 1] - tail[i]) < 0
            and abs(tail[i + 1] - tail[i]) > 1e-4
        )
        self.assertLessEqual(reversals, 1, "Attitude controller is ringing around its target")

    def test_air_spin_damping_is_framerate_independent(self):
        results = []
        for hz in (120.0, 240.0, 480.0):
            sim = Simulation()
            sim.car.reset(16.0, 12.0)
            sim.ball.reset(4.0, 12.0)
            sim.car.body.angular_velocity = 6.0

            steps = int(0.5 * hz)
            for _ in range(steps):
                sim.car.update(CarAction(), 1.0 / hz)
                sim.space.step(1.0 / hz)
            results.append(sim.car.body.angular_velocity)

        self.assertLess(max(results) - min(results), 0.05,
                        f"Spin decay must not depend on the sub-step size: {results}")


class TestBallInteraction(unittest.TestCase):
    """Car/ball contact must transfer momentum, never manufacture it."""

    def test_strike_speed_respects_momentum_transfer(self):
        sim = Simulation()
        sim.car.reset(FLAT_START_X, sim.spawn_y)
        sim.ball.reset(22.0, 2.81)

        best_ball = 0.0
        car_at_impact = 0.0
        for _ in range(300):
            prev = sim.ball.velocity[0]
            sim.step(CarAction(dir_x=1.0), DT)
            if car_at_impact == 0.0 and abs(sim.ball.velocity[0] - prev) > 0.5:
                car_at_impact = sim.car.velocity[0]
            best_ball = max(best_ball, sim.ball.velocity[0])

        # Head-on rigid-body limit for a car of CAR_MASS hitting a ball of BALL_MASS.
        ceiling = ((1.0 + CAR_BALL_RESTITUTION) * CAR_MASS / (CAR_MASS + BALL_MASS)) * car_at_impact
        self.assertGreater(best_ball, 5.0, "A full-speed drive should still hit the ball hard")
        self.assertLessEqual(best_ball, ceiling * 1.10,
                             "Ball gained more speed than momentum transfer allows")

    def test_sustained_contact_does_not_pump_energy(self):
        """Pushing the ball must not repeatedly re-launch it above the car's own speed."""
        sim = Simulation()
        sim.car.reset(FLAT_START_X, sim.spawn_y)
        sim.ball.reset(12.6, 2.81)

        for _ in range(90):
            sim.step(CarAction(dir_x=1.0), DT)
            ball_speed = math.hypot(*sim.ball.velocity)
            car_speed = math.hypot(*sim.car.velocity)
            ceiling = ((1.0 + CAR_BALL_RESTITUTION) * CAR_MASS / (CAR_MASS + BALL_MASS))
            self.assertLessEqual(
                ball_speed, max(2.0, car_speed * ceiling) + 1.0,
                "Sustained car/ball contact is injecting energy every sub-step"
            )

    def test_ball_dropped_on_parked_car_does_not_pop(self):
        sim = Simulation()
        settle(sim, 30)
        drop_height = 5.0
        sim.ball.reset(sim.car.position[0], drop_height)

        fastest = 0.0
        for _ in range(180):
            sim.step(CarAction(), DT)
            fastest = max(fastest, math.hypot(*sim.ball.velocity))

        impact_speed = math.sqrt(2.0 * GRAVITY_MAG * drop_height)
        self.assertLess(fastest, impact_speed,
                        "A parked car must not launch a ball that merely fell on it")


class TestRobustness(unittest.TestCase):
    """The simulation must stay finite, in-bounds and reproducible."""

    def _chaotic_run(self):
        sim = Simulation()
        trace = []
        for i in range(900):
            action = CarAction(
                dir_x=math.sin(i * 0.13),
                dir_y=math.sin(i * 0.07),
                jump=(i % 23 == 0),
                boost=(i % 17 < 9),
            )
            sim.step(action, DT)
            trace.append((sim.car.position, sim.ball.position))
        return sim, trace

    def test_no_nan_or_escape_under_chaotic_input(self):
        sim, trace = self._chaotic_run()
        for (cx, cy), (bx, by) in trace:
            for value in (cx, cy, bx, by):
                self.assertFalse(math.isnan(value) or math.isinf(value))
            self.assertTrue(-2.0 < cx < 34.0 and -2.0 < cy < 20.0, f"Car escaped: {(cx, cy)}")
            self.assertTrue(-2.0 < bx < 34.0 and -2.0 < by < 20.0, f"Ball escaped: {(bx, by)}")

    def test_simulation_is_deterministic(self):
        _, first = self._chaotic_run()
        _, second = self._chaotic_run()
        self.assertEqual(first, second, "Identical inputs must produce identical trajectories")

    def test_car_can_climb_and_leave_a_corner_fillet(self):
        """The old controller forced the body flat and pinned the car in the corner."""
        sim = Simulation()
        settle(sim)

        for _ in range(300):
            sim.step(CarAction(dir_x=1.0), DT)
        pinned_x = sim.car.position[0]
        self.assertGreater(math.degrees(sim.car.angle), 15.0,
                           "Car should follow the corner fillet contour instead of staying flat")

        for _ in range(120):
            sim.step(CarAction(dir_x=-1.0), DT)
        self.assertLess(sim.car.position[0], pinned_x - 5.0, "Car must be able to drive back out")


class TestSubStepInvariance(unittest.TestCase):
    """Behaviour must not change when the sub-step count changes."""

    def test_drive_distance_is_stable_across_substep_counts(self):
        import src.core.simulation as sim_module

        original = sim_module.PHYSICS_SUBSTEPS
        distances = []
        try:
            for substeps in (2, 4, 8):
                sim_module.PHYSICS_SUBSTEPS = substeps
                sim = Simulation()
                sim.car.reset(FLAT_START_X, sim.spawn_y)
                settle(sim, 60)
                start = sim.car.position[0]
                for _ in range(60):
                    sim.step(CarAction(dir_x=1.0), DT)
                distances.append(sim.car.position[0] - start)
        finally:
            sim_module.PHYSICS_SUBSTEPS = original

        spread = max(distances) - min(distances)
        self.assertLess(spread, 0.25, f"Sub-step count changes the trajectory: {distances}")
        self.assertEqual(original, PHYSICS_SUBSTEPS)


if __name__ == '__main__':
    unittest.main()
