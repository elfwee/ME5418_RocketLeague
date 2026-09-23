"""Unit tests verifying the Boundary Surface Alignment Mechanic across all arena boundaries.

Requirements from changes.md:
- When the vehicle is very close to or touching a game boundary, its rotation toward that surface
  becomes constrained.
- If the player attempts to pitch the vehicle so that its nose or rear points directly into the boundary,
  the vehicle is smoothly rotated/aligned so that its wheel side faces the boundary (d = -n, u = n),
  allowing both wheels to sit against the surface.
- Pitching away from the boundary (e.g. wheelie on floor, diving away from ceiling) remains permitted.
- In mid-air far from boundaries, unrestricted 360-degree aerial rotation resumes.
- Smooth torque-driven angular transitions (no visual snapping).
"""
import unittest
import math
from src.core.simulation import Simulation
from src.core.actions import CarAction
from src.config import (
    CAR_AIR_ANGULAR_SPEED,
    BOUNDARY_ALIGN_DIST,
    BOUNDARY_PITCH_THRESHOLD,
)


class TestSurfaceAlignment(unittest.TestCase):
    """Test suite for Boundary Surface Alignment Mechanic across floor, ceiling, walls, corners, and mid-air."""

    def setUp(self):
        self.sim = Simulation(enable_orange=False)
        self.car = self.sim.car

    # ------------------------------------------------------------------ #
    # 1. Floor Tests
    # ------------------------------------------------------------------ #

    def test_floor_pitch_down_flattens_car_facing_right(self):
        """Holding Down near the floor flattens the car horizontally (wheels on floor), not diving nose-down."""
        self.car.reset(17.5, self.sim.spawn_y, angle=0.0, facing_x=1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        # Commanded pitch Down into the floor
        act = CarAction(dir_x=0.0, dir_y=-1.0)
        target = self.car._target_angle(act)
        # For floor (n = [0, 1]) and facing right (+1): target angle should be 0.0 (flat)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target, 0.0, places=3,
                               msg="Pitching down near floor must redirect to horizontal wheel alignment (0 rad)")

        # Run several frames: car should remain flat, not tilted nose-down (-90 deg)
        for _ in range(30):
            self.sim.step(act, 1.0 / 60.0)

        self.assertAlmostEqual(self.car.body.angle, 0.0, delta=0.15,
                               msg="Chassis must remain flat on floor rather than diving nose-first into ground")
        self.assertAlmostEqual(self.car.up_vector[1], 1.0, delta=0.1,
                               msg="Car roof must point upward (+Y) with wheels on floor")

    def test_floor_pitch_down_flattens_car_facing_left(self):
        """Holding Down near floor while facing left flattens car at pi rad (wheels on floor)."""
        self.car.reset(17.5, self.sim.spawn_y, angle=math.pi, facing_x=-1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        act = CarAction(dir_x=0.0, dir_y=-1.0)
        target = self.car._target_angle(act)
        # For floor (n = [0, 1]) and facing left (-1): target angle should be pi (flat facing left)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(abs(target), math.pi, places=3,
                               msg="Pitching down near floor facing left must redirect to pi rad")

    def test_floor_pitch_up_allows_wheelie(self):
        """Holding Up on the floor allows the nose to rear up into a wheelie stance (not blocked)."""
        self.car.reset(17.5, self.sim.spawn_y, angle=0.0, facing_x=1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        act = CarAction(dir_x=0.0, dir_y=1.0)
        target = self.car._target_angle(act)
        # Pitching UP is pointing away from floor (dot > 0), so target must be pi/2 (+90 deg)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target, math.pi / 2.0, places=3,
                               msg="Pitching UP on floor must be permitted to allow wheelies and takeoffs")

        # Step simulation to observe nose lifting into wheelie
        for _ in range(25):
            self.sim.step(act, 1.0 / 60.0)

        self.assertGreater(self.car.body.angle, 0.3,
                           msg="Car nose must rear up when player holds UP on the ground")

    # ------------------------------------------------------------------ #
    # 2. Ceiling Tests
    # ------------------------------------------------------------------ #

    def test_ceiling_pitch_up_aligns_upside_down(self):
        """Pitching Up near the ceiling aligns car upside-down with wheels facing the ceiling."""
        # Ceiling is at y = 16.5; place car at y = 15.8 (within BOUNDARY_ALIGN_DIST = 1.0m)
        self.car.reset(17.5, 15.8, angle=0.0, facing_x=1)
        self.car.body.velocity = (0.0, 0.0)
        self.sim.step(CarAction(), 1.0 / 60.0)

        # Commanded pitch UP into the ceiling
        act = CarAction(dir_x=0.0, dir_y=1.0)
        target = self.car._target_angle(act)
        # Ceiling inward normal n = (0, -1). Facing right (+1).
        # atan2(-1 * 0, 1 * (-1)) = atan2(0, -1) = pi rad (upside-down)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(abs(target), math.pi, places=3,
                               msg="Pitching UP near ceiling must redirect to upside-down wheel alignment (pi rad)")

        # Run attitude control steps: car rotates smoothly upside-down so wheels contact ceiling
        for _ in range(50):
            # Counter gravity for clean test of attitude convergence
            self.car.body.position = (17.5, 15.8)
            self.car.body.velocity = (0.0, 0.0)
            self.sim.step(act, 1.0 / 60.0)

        self.assertAlmostEqual(abs(self.car.body.angle), math.pi, delta=0.25,
                               msg="Car must smoothly rotate upside down near ceiling")
        # Roof (up_vector) points down into arena (u_y = -1), meaning wheels face ceiling (d_y = +1)
        self.assertAlmostEqual(self.car.up_vector[1], -1.0, delta=0.2,
                               msg="Car roof must face down into arena, wheels against ceiling")

    def test_ceiling_pitch_down_dives_away(self):
        """Pitching Down near the ceiling dives away into the arena (not blocked)."""
        self.car.reset(17.5, 15.8, angle=math.pi, facing_x=1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        # Commanded pitch DOWN away from ceiling
        act = CarAction(dir_x=0.0, dir_y=-1.0)
        target = self.car._target_angle(act)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target, -math.pi / 2.0, places=3,
                               msg="Pitching DOWN away from ceiling must be permitted (-pi/2 rad)")

    # ------------------------------------------------------------------ #
    # 3. Right Wall Tests
    # ------------------------------------------------------------------ #

    def test_right_wall_pitch_right_aligns_vertically(self):
        """Pitching Right near the right wall aligns car vertically with wheels against the wall."""
        # Right wall is at x = 30.5; place car at x = 29.8, y = 13.0 (above goal)
        self.car.reset(29.8, 13.0, angle=0.0, facing_x=1)
        self.car.body.velocity = (0.0, 0.0)
        self.sim.step(CarAction(), 1.0 / 60.0)

        # Commanded pitch RIGHT into the wall
        act = CarAction(dir_x=1.0, dir_y=0.0)
        target = self.car._target_angle(act)
        # Right wall inward normal n = (-1, 0). Facing right (+1).
        # atan2(-1 * (-1), 1 * 0) = atan2(1, 0) = pi/2 rad (+90 deg, nose up, wheels right)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target, math.pi / 2.0, places=3,
                               msg="Pitching RIGHT into right wall must redirect to vertical wheel alignment (pi/2 rad)")

        # Run attitude steps
        for _ in range(40):
            self.car.body.velocity = (0.0, 0.0)
            self.sim.step(act, 1.0 / 60.0)

        self.assertAlmostEqual(self.car.body.angle, math.pi / 2.0, delta=0.25,
                               msg="Chassis must align vertically along the right wall")
        # Roof points left into arena (u_x = -1), wheels point right against wall
        self.assertAlmostEqual(self.car.up_vector[0], -1.0, delta=0.2,
                               msg="Roof must point left into arena, wheels facing right wall")

    def test_right_wall_pitch_left_pitches_away(self):
        """Pitching Left near the right wall points away into the arena (not blocked)."""
        self.car.reset(29.8, 13.0, angle=math.pi / 2.0, facing_x=1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        # Commanded pitch LEFT away from right wall into arena
        act = CarAction(dir_x=-1.0, dir_y=0.0)
        target = self.car._target_angle(act)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(abs(target), math.pi, places=3,
                               msg="Pitching LEFT away from right wall must be permitted (pi rad)")

    # ------------------------------------------------------------------ #
    # 4. Left Wall Tests
    # ------------------------------------------------------------------ #

    def test_left_wall_pitch_left_aligns_vertically(self):
        """Pitching Left near the left wall aligns car vertically with wheels against the left wall."""
        # Left wall is at x = 4.5; place car at x = 5.2, y = 13.0 (above goal)
        self.car.reset(5.2, 13.0, angle=math.pi, facing_x=-1)
        self.car.body.velocity = (0.0, 0.0)
        self.sim.step(CarAction(), 1.0 / 60.0)

        # Commanded pitch LEFT into the left wall
        act = CarAction(dir_x=-1.0, dir_y=0.0)
        target = self.car._target_angle(act)
        # Left wall inward normal n = (1, 0). Facing left (-1).
        # atan2(-(-1) * 1, -1 * 0) = atan2(1, 0) = pi/2 rad (+90 deg, nose up, wheels left)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target, math.pi / 2.0, places=3,
                               msg="Pitching LEFT into left wall must redirect to vertical wheel alignment (pi/2 rad)")

        # Run attitude steps
        for _ in range(40):
            self.car.body.velocity = (0.0, 0.0)
            self.sim.step(act, 1.0 / 60.0)

        self.assertAlmostEqual(self.car.body.angle, math.pi / 2.0, delta=0.25,
                               msg="Chassis must align vertically along the left wall")
        # Roof points right into arena (u_x = +1), wheels point left against wall
        self.assertAlmostEqual(self.car.up_vector[0], 1.0, delta=0.2,
                               msg="Roof must point right into arena, wheels facing left wall")

    def test_left_wall_pitch_right_pitches_away(self):
        """Pitching Right near the left wall points away into the arena (not blocked)."""
        self.car.reset(5.2, 13.0, angle=math.pi / 2.0, facing_x=-1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        # Commanded pitch RIGHT away from left wall into arena
        act = CarAction(dir_x=1.0, dir_y=0.0)
        target = self.car._target_angle(act)
        self.assertIsNotNone(target)
        self.assertAlmostEqual(target, 0.0, places=3,
                               msg="Pitching RIGHT away from left wall must be permitted (0 rad)")

    # ------------------------------------------------------------------ #
    # 5. Corner Fillet Tests
    # ------------------------------------------------------------------ #

    def test_corner_fillet_diagonal_alignment(self):
        """Near a curved corner fillet, steering into the corner smoothly aligns wheels with curved normal."""
        # Bottom-Left fillet arc is between (8.0, 1.5) and (4.5, 5.0).
        # Place car near fillet midpoint (x = 5.8, y = 2.8)
        self.car.reset(5.8, 2.8, angle=0.0, facing_x=1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        # Inward normal of bottom-left fillet points toward arena center (nx > 0, ny > 0, ~45 deg)
        boundary = self.car._detect_nearby_boundary()
        self.assertIsNotNone(boundary, "Boundary must be detected near corner fillet")
        dist, (nx, ny) = boundary
        self.assertGreater(nx, 0.2, "Bottom-left fillet inward normal nx must be positive")
        self.assertGreater(ny, 0.2, "Bottom-left fillet inward normal ny must be positive")

        # Steering Down-Left into the fillet corner
        act = CarAction(dir_x=-1.0, dir_y=-1.0)
        target = self.car._target_angle(act)
        expected_target = math.atan2(-self.car.facing_x * nx, self.car.facing_x * ny)
        self.assertAlmostEqual(target, expected_target, places=3,
                               msg="Steering into corner fillet must align wheels tangent to curve")

    # ------------------------------------------------------------------ #
    # 6. Mid-Air Unrestricted Rotation
    # ------------------------------------------------------------------ #

    def test_midair_unrestricted_rotation(self):
        """In mid-air far from boundaries, full 360-degree aerial control is completely unrestricted."""
        # Field center: x = 17.5, y = 9.0 (at least 6.5m from any wall/surface)
        self.car.reset(17.5, 9.0, angle=0.0, facing_x=1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        self.assertIsNone(self.car._detect_nearby_boundary(),
                          "No boundary should be detected at midfield center")

        # Test various aerial angles
        test_directions = [
            (1.0, 0.0, 0.0),                     # 0 deg (Right)
            (1.0, 1.0, math.pi / 4.0),           # +45 deg (Up-Right)
            (0.0, 1.0, math.pi / 2.0),           # +90 deg (Up)
            (-1.0, 1.0, 3.0 * math.pi / 4.0),    # +135 deg (Up-Left)
            (-1.0, 0.0, math.pi),                # 180 deg (Left)
            (-1.0, -1.0, -3.0 * math.pi / 4.0),  # -135 deg (Down-Left)
            (0.0, -1.0, -math.pi / 2.0),         # -90 deg (Down)
            (1.0, -1.0, -math.pi / 4.0),         # -45 deg (Down-Right)
        ]

        for dx, dy, expected_rad in test_directions:
            act = CarAction(dir_x=dx, dir_y=dy)
            target = self.car._target_angle(act)
            self.assertIsNotNone(target)
            diff = abs((target - expected_rad + math.pi) % (2.0 * math.pi) - math.pi)
            self.assertAlmostEqual(diff, 0.0, places=3,
                                   msg=f"Mid-air angle for ({dx}, {dy}) must match unconstrained input")

    def test_midair_idle_spin_decay(self):
        """In mid-air with idle input, target angle is None so spin decays smoothly."""
        self.car.reset(17.5, 9.0, angle=0.0, facing_x=1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        act = CarAction(dir_x=0.0, dir_y=0.0)
        target = self.car._target_angle(act)
        self.assertIsNone(target, "In open mid-air with idle input, target angle must be None for spin decay")

    # ------------------------------------------------------------------ #
    # 7. Turtle (Upside-Down) Instant Recovery Tests
    # ------------------------------------------------------------------ #

    def test_instant_turtle_recovery_flat(self):
        """Inverted car (180 deg) on the floor immediately triggers recovery hop-flip without delay."""
        self.car.reset(17.5, 2.0, angle=math.pi, facing_x=1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        # Recovery flip must activate immediately on inverted ground contact
        self.assertGreater(self.car._recovery_time, 0.0, "Recovery flip must trigger immediately (delay = 0)")

        # Run 35 physics frames (~0.58s): car must be fully upright on both wheels
        for _ in range(35):
            self.sim.step(CarAction(), 1.0 / 60.0)

        self.assertTrue(self.car.both_wheels_grounded, "Car must be settled on both wheels")
        self.assertGreater(self.car.up_vector[1], 0.8, "Car roof must be pointing upward (+Y)")
        self.assertAlmostEqual(self.car.body.angle, 0.0, delta=0.25, msg="Car must be flat upright (0 rad)")

    def test_instant_turtle_recovery_oblique_angles(self):
        """Inverted car at oblique angles (110°, 135°, 225°, 250°) recovers immediately."""
        for deg in [110, 135, 225, 250]:
            with self.subTest(angle=deg):
                self.car.reset(17.5, 2.0, angle=math.radians(deg), facing_x=1)
                self.sim.step(CarAction(), 1.0 / 60.0)

                self.assertGreater(self.car._recovery_time, 0.0,
                                   f"Recovery must trigger immediately at {deg} deg")

                for _ in range(35):
                    self.sim.step(CarAction(), 1.0 / 60.0)

                self.assertTrue(self.car.both_wheels_grounded, f"Car must settle on both wheels from {deg} deg")
                self.assertGreater(self.car.up_vector[1], 0.8, f"Car roof must point up after recovery from {deg} deg")

    def test_instant_turtle_recovery_facing_left(self):
        """Inverted car facing left recovers immediately to flat facing left (pi rad)."""
        # When facing left (-1), inverted stance has body angle near 0 (roof pointing down)
        self.car.reset(17.5, 2.0, angle=0.01, facing_x=-1)
        self.sim.step(CarAction(), 1.0 / 60.0)

        self.assertGreater(self.car._recovery_time, 0.0, "Recovery must trigger immediately for facing_x = -1")

        for _ in range(35):
            self.sim.step(CarAction(), 1.0 / 60.0)

        self.assertTrue(self.car.both_wheels_grounded, "Car facing left must settle on both wheels")
        self.assertGreater(self.car.up_vector[1], 0.8, "Car roof must point upward (+Y)")
        self.assertAlmostEqual(abs(self.car.body.angle), math.pi, delta=0.25,
                               msg="Car facing left must be flat at pi rad")

    def test_instant_turtle_recovery_with_boost(self):
        """Inverted car recovers immediately even if player is holding boost."""
        self.car.reset(17.5, 2.0, angle=math.pi, facing_x=1)
        self.sim.step(CarAction(boost=True), 1.0 / 60.0)

        self.assertGreater(self.car._recovery_time, 0.0, "Holding boost must not block turtle recovery flip")

    # ------------------------------------------------------------------ #
    # 8. Aerial Descent Landing Tests (Prevent Nose-Down Vertically)
    # ------------------------------------------------------------------ #

    def test_aerial_fall_nose_down_idle_auto_levels(self):
        """Vehicle falling from aerial with nose pointing straight down (-90°) auto-levels with idle input."""
        self.sim.ball.body.position = (25.0, 10.0)
        self.car.reset(12.0, 5.0, angle=-math.pi / 2.0, facing_x=1)
        self.car.body.velocity = (0.0, -10.0)

        for _ in range(35):
            self.sim.step(CarAction(), 1.0 / 60.0)

        self.assertTrue(self.car.is_grounded, "Car must touch down on ground")
        self.assertTrue(self.car.both_wheels_grounded, "Car must settle flat on both wheels (not nose-down)")
        self.assertLess(self.car.pitch_degrees, 15.0, "Car pitch must be nearly flat (< 15 deg)")
        self.assertGreater(self.car.forward_vector[1], -0.2, "Car nose must not point vertically into ground")

    def test_aerial_fall_nose_down_holding_forward_auto_levels(self):
        """Vehicle falling nose-down (-90°) while holding forward auto-levels onto wheels upon landing."""
        self.sim.ball.body.position = (25.0, 10.0)
        self.car.reset(12.0, 5.0, angle=-math.pi / 2.0, facing_x=1)
        self.car.body.velocity = (0.0, -10.0)

        for _ in range(35):
            self.sim.step(CarAction(dir_x=1.0), 1.0 / 60.0)

        self.assertTrue(self.car.both_wheels_grounded, "Car must settle flat on both wheels when holding forward")
        self.assertLess(self.car.pitch_degrees, 15.0, "Car pitch must be nearly flat (< 15 deg)")

    def test_aerial_fall_fast_facing_left_auto_levels(self):
        """Fast aerial fall (-14 m/s) with facing_x = -1 auto-levels wheels to ground."""
        self.sim.ball.body.position = (25.0, 10.0)
        self.car.reset(17.5, 6.0, angle=-math.pi / 2.0, facing_x=-1)
        self.car.body.velocity = (0.0, -14.0)

        for _ in range(40):
            self.sim.step(CarAction(dir_x=-1.0), 1.0 / 60.0)

        self.assertTrue(self.car.both_wheels_grounded, "Car facing left must land and settle on both wheels")
        self.assertLess(self.car.pitch_degrees, 15.0, "Pitch must level out to floor (< 15 deg)")

    # ------------------------------------------------------------------ #
    # 9. State Space Telemetry Tests
    # ------------------------------------------------------------------ #

    def test_is_turtling_in_get_state(self):
        """Verify is_turtling and is_turtling_recovery correctly report in get_state and get_state_norm."""
        # 1. Normal kickoff: both flags False
        state = self.sim.get_state()
        state_norm = self.sim.get_state_norm()
        self.assertIn("is_turtling", state["car"])
        self.assertIn("is_turtling_recovery", state["car"])
        self.assertFalse(state["car"]["is_turtling"])
        self.assertFalse(state["car"]["is_turtling_recovery"])
        self.assertEqual(state_norm["car"]["is_turtling"], 0.0)
        self.assertEqual(state_norm["car"]["is_turtling_recovery"], 0.0)

        # 2. Inverted car on floor: is_turtling becomes True
        self.car.reset(17.5, 2.0, angle=math.pi, facing_x=1)
        self.car._recovery_time = 0.0
        self.car._sense_ground()

        self.assertTrue(self.car.is_turtled)
        self.assertTrue(self.car.is_turtling)
        state_inv = self.sim.get_state()
        self.assertTrue(state_inv["car"]["is_turtling"])

        # 3. During recovery flip: is_turtling_recovery becomes True
        self.sim.step(CarAction(), 1.0 / 60.0)
        state_rec = self.sim.get_state()
        state_rec_norm = self.sim.get_state_norm()
        self.assertTrue(state_rec["car"]["is_turtling_recovery"])
        self.assertTrue(state_rec["car"]["is_turtling"])
        self.assertEqual(state_rec_norm["car"]["is_turtling_recovery"], 1.0)
        self.assertEqual(state_rec_norm["car"]["is_turtling"], 1.0)


if __name__ == '__main__':
    unittest.main()

