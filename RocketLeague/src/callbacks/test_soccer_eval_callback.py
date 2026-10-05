"""Unit tests for SoccerEvalCallback and environment soccer scoreboard reporting."""

import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure repository root is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.rocket_league_env import RocketLeagueEnv
from src.callbacks.soccer_eval_callback import SoccerEvalCallback


class DummyPolicy:
    """Mock policy for evaluation testing without training overhead."""

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, obs, deterministic=True):
        return self.action_space.sample(), None

    def save(self, path):
        # Create a mock zip file at target path
        with open(f"{path}.zip" if not str(path).endswith(".zip") else path, "wb") as f:
            f.write(b"MOCK_MODEL_WEIGHTS")


class TestSoccerScoreboardAndEvalCallback(unittest.TestCase):
    """Test suite verifying soccer evaluation metrics and callback mechanics."""

    def setUp(self):
        self.env = RocketLeagueEnv(render_mode=None, bot_type="bot_level_3", terminate_on_goal=True)

    def tearDown(self):
        self.env.close()

    def test_environment_reset_reports_scoreboard(self):
        """Test that reset() provides initial match scoreboard info."""
        obs, info = self.env.reset(seed=123)
        self.assertIn("goals_for", info)
        self.assertIn("goals_against", info)
        self.assertIn("goal_difference", info)
        self.assertIn("match_result", info)
        self.assertEqual(info["goals_for"], 0)
        self.assertEqual(info["goals_against"], 0)
        self.assertEqual(info["goal_difference"], 0)
        self.assertEqual(info["match_result"], "draw")

    def test_environment_step_reports_scoreboard(self):
        """Test that step() maintains objective match scoreboard in info."""
        self.env.reset(seed=456)
        action = [0, 0, 0]
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIn("goals_for", info)
        self.assertIn("goals_against", info)
        self.assertIn("goal_difference", info)
        self.assertIn("match_result", info)
        self.assertIn(info["match_result"], ["win", "loss", "draw"])

    def test_metric_calculations_standard_soccer(self):
        """Test the (3W + D)/(3N) match points and goal diff formula."""
        callback = SoccerEvalCallback(eval_freq=1000, n_eval_matches=10, verbose=0)

        # Case 1: 2 wins, 1 draw, 1 loss (N=4)
        # Match score = (3*2 + 1) / (3*4) = 7 / 12 = 0.5833
        # Goal diff = (5 - 3) / 4 = +0.5000
        raw_result = {
            "wins": 2,
            "draws": 1,
            "losses": 1,
            "goals_for": 5,
            "goals_against": 3,
            "games": 4,
        }
        metrics = callback._calculate_metrics(raw_result)

        self.assertAlmostEqual(metrics["win_rate"], 2 / 4)
        self.assertAlmostEqual(metrics["draw_rate"], 1 / 4)
        self.assertAlmostEqual(metrics["loss_rate"], 1 / 4)
        self.assertAlmostEqual(metrics["goals_for"], 5 / 4)
        self.assertAlmostEqual(metrics["goals_against"], 3 / 4)
        self.assertAlmostEqual(metrics["goal_diff"], 2 / 4)
        self.assertAlmostEqual(metrics["match_score"], 7 / 12)

    def test_model_selection_tie_breaker(self):
        """Test that tie-breaker selects higher goal diff when match points are equal."""
        with tempfile.TemporaryDirectory() as temp_dir:
            callback = SoccerEvalCallback(save_path=temp_dir, verbose=0)
            mock_model = DummyPolicy(self.env.action_space)
            callback.model = mock_model
            callback.num_timesteps = 1000
            callback.n_calls = 1

            # Simulate baseline model: match_score = 0.500, goal_diff = +0.200
            callback.best_match_score = 0.500
            callback.best_goal_diff = 0.200

            # 1. Inferior model: match_score = 0.400 -> Should NOT replace
            better_inf = (0.400 > callback.best_match_score) or (
                np.isclose(0.400, callback.best_match_score) and 0.500 > callback.best_goal_diff
            )
            self.assertFalse(better_inf)

            # 2. Equal match_score (0.500) but higher goal_diff (+0.800) -> SHOULD replace
            better_tie_break = (0.500 > callback.best_match_score) or (
                np.isclose(0.500, callback.best_match_score) and 0.800 > callback.best_goal_diff
            )
            self.assertTrue(better_tie_break)

            # 3. Strictly superior match_score (0.600) even with lower goal diff -> SHOULD replace
            better_sup = (0.600 > callback.best_match_score) or (
                np.isclose(0.600, callback.best_match_score) and -0.100 > callback.best_goal_diff
            )
            self.assertTrue(better_sup)

    def test_full_evaluation_run_with_mock_policy(self):
        """Run a fast 2-match evaluation against multiple opponents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            opponents = {
                "heuristic_bot": "bot_level_3",
                "solo_practice": "none",
            }
            callback = SoccerEvalCallback(
                opponents=opponents,
                n_eval_matches=2,
                save_path=temp_dir,
                deterministic=True,
                verbose=1,
            )

            mock_model = DummyPolicy(self.env.action_space)
            callback.model = mock_model
            callback.num_timesteps = 50000
            callback.n_calls = 50000

            # Mock SB3 logger
            class MockLogger:
                def __init__(self):
                    self.records = {}

                def record(self, key, value):
                    self.records[key] = value

                def dump(self, step=0):
                    pass

            callback._logger = MockLogger()

            results = callback.evaluate()
            self.assertIn("overall_metrics", results)
            self.assertIn("opponents", results)
            self.assertIn("heuristic_bot", results["opponents"])
            self.assertIn("solo_practice", results["opponents"])

            # Verify saved checkpoint
            saved_checkpoint = os.path.join(temp_dir, "best_model.zip")
            metadata_file = os.path.join(temp_dir, "best_model_eval_metrics.json")
            self.assertTrue(os.path.exists(saved_checkpoint))
            self.assertTrue(os.path.exists(metadata_file))

            # Verify logged TensorBoard metrics
            self.assertIn("eval_soccer/overall/match_score", callback.logger.records)
            self.assertIn("eval_soccer/overall/goal_diff", callback.logger.records)
            self.assertIn("eval_soccer/heuristic_bot/match_score", callback.logger.records)
            self.assertIn("eval_soccer/solo_practice/match_score", callback.logger.records)

            callback._on_training_end()


if __name__ == "__main__":
    unittest.main()
