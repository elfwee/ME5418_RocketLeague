"""Unit test suite for Heuristic Bot AI, Orange car toggles, and state machine transitions."""
import unittest
import math
from src.core.simulation import Simulation
from src.core.actions import CarAction
from src.ai.heuristic_bot import HeuristicBot
from src.config import CAR_SPAWN_X_DEFENSIVE, CAR_SPAWN_X_ATTACK


class TestHeuristicBot(unittest.TestCase):
    """Verify Orange car presence, AI state transitions, and defensive/offensive mechanics."""

    def setUp(self):
        self.sim = Simulation(enable_orange=True, orange_is_bot=True)

    def test_enable_disable_flag(self):
        """Verify enable_orange flag creates car and dynamic toggle cleanly adds/removes it."""
        self.assertIsNotNone(self.sim.car_orange)
        self.assertIsNotNone(self.sim.orange_bot)
        self.assertEqual(self.sim.car_orange.team, "orange")
        self.assertEqual(self.sim.car_orange.facing_x, -1)

        # Dynamic disable
        self.sim.set_orange_enabled(False)
        self.assertIsNone(self.sim.car_orange)
        self.assertIsNone(self.sim.orange_bot)
        self.assertFalse(self.sim.enable_orange)

        # Verify simulation steps without Orange car
        self.sim.step(CarAction(dir_x=1.0), 1.0 / 60.0)

        # Dynamic re-enable
        self.sim.set_orange_enabled(True, is_bot=True)
        self.assertIsNotNone(self.sim.car_orange)
        self.assertIsNotNone(self.sim.orange_bot)
        self.assertTrue(self.sim.enable_orange)

    def test_kickoff_state_and_challenge(self):
        """Verify bot enters KICKOFF state at match start and charges toward center with boost."""
        car_o = self.sim.car_orange
        bot = self.sim.orange_bot

        state = bot.determine_state(self.sim, car_o)
        self.assertEqual(state, "KICKOFF")

        action = bot.compute_action(self.sim)
        self.assertEqual(action.dir_x, -1.0, "Bot should drive toward midfield (dir_x = -1.0)")
        self.assertTrue(action.boost, "Bot should boost toward kickoff")

        # Advance 20 steps
        init_x = car_o.position[0]
        for _ in range(20):
            self.sim.step(CarAction(), 1.0 / 60.0)

        self.assertLess(car_o.position[0], init_x, "Orange car should move toward midfield on kickoff")

    def test_anti_own_goal_rotation(self):
        """Verify bot rotates back toward defensive side when caught on the wrong side of the ball."""
        car_o = self.sim.car_orange
        bot = self.sim.orange_bot

        # Place ball in center and place Orange car to the left of the ball (wrong side!)
        self.sim.ball.reset(16.0, 2.5)
        car_o.reset(12.0, self.sim.spawn_y, angle=0.0, facing_x=1)

        state = bot.determine_state(self.sim, car_o)
        self.assertEqual(state, "ROTATE_BACK", "Bot should recognize it is on the wrong side")

        action = bot.compute_action(self.sim)
        self.assertEqual(action.dir_x, 1.0, "Bot should drive back toward its defensive half (dir_x = +1.0)")

    def test_defend_state_and_save(self):
        """Verify bot enters DEFEND mode when the ball enters the defensive zone heading for net."""
        car_o = self.sim.car_orange
        bot = self.sim.orange_bot

        # Ball in Orange defensive half moving towards Orange goal
        self.sim.ball.reset(25.0, 2.5)
        self.sim.ball.body.velocity = (8.0, 0.0)
        car_o.reset(28.0, self.sim.spawn_y, angle=math.pi, facing_x=-1)

        state = bot.determine_state(self.sim, car_o)
        self.assertEqual(state, "DEFEND", "Bot should defend when ball is threatening its goal")

    def test_attack_strike_orientation(self):
        """Verify bot drives towards Blue net when attacking."""
        car_o = self.sim.car_orange
        bot = self.sim.orange_bot

        # Ball at midfield, Orange car safely behind the ball
        self.sim.ball.reset(16.0, 2.5)
        self.sim.ball.body.velocity = (0.0, 0.0)
        car_o.reset(20.0, self.sim.spawn_y, angle=math.pi, facing_x=-1)

        state = bot.determine_state(self.sim, car_o)
        self.assertEqual(state, "ATTACK")

        action = bot.compute_action(self.sim)
        self.assertEqual(action.dir_x, -1.0, "Bot should attack toward Blue goal (dir_x = -1.0)")

    def test_car_car_collision(self):
        """Verify that Blue and Orange cars physically collide and bounce."""
        blue = self.sim.car
        orange = self.sim.car_orange

        # Place cars facing each other on collision course
        blue.reset(14.0, self.sim.spawn_y, angle=0.0, facing_x=1)
        blue.body.velocity = (10.0, 0.0)

        orange.reset(16.5, self.sim.spawn_y, angle=math.pi, facing_x=-1)
        orange.body.velocity = (-10.0, 0.0)

        # Step until collision
        for _ in range(25):
            self.sim.step(CarAction(), 1.0 / 60.0, action_orange=CarAction())

        # Blue should have bounced backward (v_x < 0) or slowed significantly, Orange bounced forward (v_x > 0)
        self.assertLess(blue.velocity[0], 2.0, "Blue car should bounce backward upon head-on collision")
        self.assertGreater(orange.velocity[0], -2.0, "Orange car should bounce backward upon head-on collision")

    def test_no_freeze_in_rotate_back_or_corner(self):
        """Verify Orange bot does not get stuck frozen when ball is behind it or in the corner."""
        car_o = self.sim.car_orange
        bot = self.sim.orange_bot

        # Ball in Orange corner (x = 28.0), Orange positioned at x = 27.0
        self.sim.ball.reset(28.0, 2.5)
        car_o.reset(27.0, self.sim.spawn_y, angle=math.pi, facing_x=-1)

        # Step 80 frames without human Blue car moving
        for _ in range(80):
            self.sim.step(CarAction(), 1.0 / 60.0)

        # Orange car must be actively moving / clearing the ball, not frozen at x = 27.0
        self.assertNotEqual(car_o.position[0], 27.0, "Orange car must not freeze at x = 27.0")
        self.assertNotEqual(bot.current_state, "ROTATE_BACK", "Bot should not stay trapped in ROTATE_BACK in defensive zone")

    def test_attack_at_blue_goal_no_jerking(self):
        """Verify Orange bot does not oscillate, jerk, or enter ROTATE_BACK at Blue goal."""
        car_o = self.sim.car_orange
        bot = self.sim.orange_bot

        # Ball right in front of Blue goal, Orange attacking behind it
        self.sim.ball.reset(5.5, 2.5)
        car_o.reset(6.5, self.sim.spawn_y, angle=math.pi, facing_x=-1)

        dir_changes = 0
        prev_dir = None

        for _ in range(30):
            act = bot.compute_action(self.sim)
            state = bot.current_state
            self.assertEqual(state, "ATTACK", "State should remain ATTACK when attacking enemy goal")
            if prev_dir is not None and act.dir_x != prev_dir:
                dir_changes += 1
            prev_dir = act.dir_x
            self.sim.step(CarAction(), 1.0 / 60.0)

        self.assertEqual(dir_changes, 0, "Bot should cleanly drive toward net with 0 jerks/oscillations")

    def test_defend_moves_toward_own_goal_when_ball_is_behind_it(self):
        """Verify Orange bot moves toward its own goal when defending, never toward Blue goal."""
        car_o = self.sim.car_orange
        bot = self.sim.orange_bot

        # Ball in defensive half (x = 24.0), Orange at midfield (x = 16.0)
        self.sim.ball.reset(24.0, 2.5)
        car_o.reset(16.0, self.sim.spawn_y, angle=math.pi, facing_x=-1)

        state = bot.determine_state(self.sim, car_o)
        self.assertEqual(state, "DEFEND")

        act = bot.compute_action(self.sim)
        self.assertEqual(act.dir_x, 1.0, "Orange bot must drive RIGHT toward ball/own goal, not toward Blue goal")

        # Step 30 frames
        for _ in range(30):
            self.sim.step(CarAction(), 1.0 / 60.0)

        # Orange car should have moved right toward its defensive zone
        self.assertGreater(car_o.position[0], 17.5, "Orange car must advance toward its defensive half")


if __name__ == '__main__':
    unittest.main()
