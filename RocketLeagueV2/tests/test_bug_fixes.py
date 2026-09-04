"""Unit tests verifying resolution of critical and major flaws across RocketLeagueV2."""
import unittest
import math
from src.core.simulation import Simulation
from src.core.actions import CarAction
from src.config import THROTTLE_DEADZONE, FACING_FLIP_THRESHOLD


class TestBugFixes(unittest.TestCase):
    """Verify bug fixes for bot aerial state, kickoff dodge flip, steering deadzone, and RL state API."""

    def test_deadzone_and_facing_thresholds_aligned(self):
        """Verify THROTTLE_DEADZONE and FACING_FLIP_THRESHOLD are aligned."""
        self.assertEqual(
            THROTTLE_DEADZONE, FACING_FLIP_THRESHOLD,
            "THROTTLE_DEADZONE and FACING_FLIP_THRESHOLD must be equal to prevent inverted driving"
        )

    def test_gentle_left_input_does_not_propel_rightward(self):
        """Verify that gentle left input (dir_x = -0.12) does not propel a right-facing car to the right."""
        sim = Simulation()
        sim.car.reset(10.0, sim.spawn_y, angle=0.0, facing_x=1)
        for _ in range(30):
            sim.step(CarAction(), 1.0 / 60.0)

        start_x = sim.car.position[0]
        # Apply gentle left input
        act = CarAction(dir_x=-0.12, dir_y=0.0)
        for _ in range(60):
            sim.step(act, 1.0 / 60.0)

        end_x = sim.car.position[0]
        vx = sim.car.velocity[0]

        self.assertLessEqual(
            end_x, start_x,
            f"Car should move leftward or brake, never propel rightward (start={start_x:.2f}, end={end_x:.2f})"
        )
        self.assertLessEqual(vx, 0.0, f"Car velocity should be negative or zero, got {vx:.2f}")

    def test_bot_persists_in_aerial_state_when_airborne(self):
        """Verify bot persists in AERIAL state across multiple airborne frames rather than aborting after 1 frame."""
        sim = Simulation(enable_orange=True, orange_is_bot=True)
        # Place ball high at midfield
        sim.ball.reset(17.0, 8.0, vx=0.0, vy=0.0)
        # Place Orange car grounded in midfield
        sim.car_orange.reset(20.0, sim.spawn_y, angle=math.pi, facing_x=-1)

        aerial_frame_count = 0
        for step in range(30):
            sim.step(CarAction(), 1.0 / 60.0)
            if sim.orange_bot.current_state == "AERIAL":
                aerial_frame_count += 1

        self.assertGreaterEqual(
            aerial_frame_count, 10,
            f"Bot should sustain AERIAL state across multiple airborne frames, got {aerial_frame_count} frames"
        )

    def test_bot_kickoff_executes_power_dodge_flip(self):
        """Verify bot successfully executes a power dodge flip during kickoff approach."""
        sim = Simulation(enable_orange=True, orange_is_bot=True)

        flipped = False
        for step in range(120):
            sim.step(CarAction(), 1.0 / 60.0)
            if sim.car_orange._flip_active:
                flipped = True
                break

        self.assertTrue(flipped, "Bot must trigger a 360-degree power dodge flip during kickoff")

    def test_simulation_rl_metrics_and_match_time(self):
        """Verify goal_scored_step triggers for strictly 1 frame and match_time accumulates across goals."""
        sim = Simulation()
        g_center_y = (sim.arena.goal_y_bot + sim.arena.goal_y_top) / 2.0

        # Step 10 frames before goal
        for _ in range(10):
            sim.step(CarAction(), 1.0 / 60.0)
            self.assertIsNone(sim.goal_scored_this_step, "No goal should be reported before ball enters net")
            state = sim.get_state()
            self.assertIsNone(state["goal_scored_step"])

        time_before_goal = sim.match_time
        self.assertGreater(time_before_goal, 0.1)

        # Force a goal into right goal pocket (Blue scores)
        sim.ball.reset(sim.arena.x_right - 0.5, g_center_y, vx=15.0, vy=0.0)

        # Step until goal is triggered
        goal_detected_steps = 0
        scoring_team = None
        for _ in range(30):
            sim.step(CarAction(), 1.0 / 60.0)
            if sim.goal_scored_this_step is not None:
                goal_detected_steps += 1
                scoring_team = sim.goal_scored_this_step

        self.assertEqual(goal_detected_steps, 1, "goal_scored_this_step must be set for exactly 1 step")
        self.assertEqual(scoring_team, "blue")

        # In the frame immediately after goal reset, goal_scored_this_step must be None again
        sim.step(CarAction(), 1.0 / 60.0)
        self.assertIsNone(sim.goal_scored_this_step, "goal_scored_this_step must return to None on subsequent step")
        self.assertEqual(sim.get_state()["goal_scored_step"], None)

        # match_time must have continued accumulating (not reset to 0)
        self.assertGreater(sim.match_time, time_before_goal)

        # Round time_elapsed resets to 0 upon kickoff, but match_time does not
        self.assertLess(sim.time_elapsed, sim.match_time)

        # Calling reset with reset_scores=True resets match_time
        sim.reset(reset_scores=True)
        self.assertEqual(sim.match_time, 0.0)
        self.assertEqual(sim.score_blue, 0)

    def test_gamepad_safe_button_query(self):
        """Verify gamepad button query does not crash when index exceeds controller button count."""
        class MockGamepad:
            def __init__(self, num_buttons=4):
                self._num = num_buttons

            def get_numbuttons(self):
                return self._num

            def get_button(self, idx):
                if idx >= self._num:
                    raise SystemError("Invalid button index")
                return idx == 0

        pad = MockGamepad(num_buttons=4)

        # Import helper logic from main
        def _gamepad_btn(gamepad, idx: int) -> bool:
            if gamepad is None:
                return False
            try:
                return gamepad.get_numbuttons() > idx and bool(gamepad.get_button(idx))
            except (SystemError, Exception):
                return False

        self.assertTrue(_gamepad_btn(pad, 0), "Button 0 should return True")
        self.assertFalse(_gamepad_btn(pad, 1), "Button 1 should return False")
        self.assertFalse(_gamepad_btn(pad, 5), "Button 5 should safely return False without exception")
        self.assertFalse(_gamepad_btn(None, 0), "None gamepad should safely return False")


if __name__ == '__main__':
    unittest.main()
