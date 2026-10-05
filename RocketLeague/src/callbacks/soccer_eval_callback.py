"""Independent soccer evaluation callback for Stable-Baselines3.

Evaluates policies against frozen/fixed opponents on objective soccer outcomes:
- Match points (primary metric): (3 * wins + draws) / (3 * games) in [0, 1]
- Goal differential (tie-breaker): (goals_for - goals_against) / games
- Independent of PPO shaped rewards (avoids reward-hacking model selection)
"""

import os
import sys
import json
from typing import Dict, Any, Optional, Union, List, Callable
import numpy as np

# Ensure repository root is on sys.path
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from stable_baselines3.common.callbacks import BaseCallback


class SoccerEvalCallback(BaseCallback):
    """Independent soccer evaluation callback for Stable-Baselines3.

    Does NOT use PPO shaped episodic reward to select the best model.
    Instead, periodically evaluates policy against frozen/fixed opponents for
    N matches and tracks objective soccer outcomes:
      - Match Score (primary metric): (3 * wins + draws) / (3 * total_matches) in [0, 1]
      - Goal Differential (tie-breaker): (goals_for - goals_against) / total_matches
      - Win / Draw / Loss rates
      - Goals For / Goals Against per game

    Logs metrics to SB3 Logger / TensorBoard under:
      - eval_soccer/{opponent_name}/...
      - eval_soccer/overall/...

    Automatically saves the best-performing checkpoint (`best_model.zip`).
    """

    def __init__(
        self,
        eval_env_factory: Optional[Callable[[Any], Any]] = None,
        opponents: Optional[Dict[str, Any]] = None,
        eval_freq: int = 50_000,
        n_eval_matches: int = 50,
        save_path: str = "./best_soccer_model",
        deterministic: bool = True,
        eval_seeds: Optional[Union[int, List[int], bool]] = None,
        verbose: int = 1,
    ):
        """Initialize the soccer evaluation callback.

        Args:
            eval_env_factory: Callable taking an opponent spec and returning an environment.
                              If None, a default factory creating `RocketLeagueEnv` is used.
            opponents: Dict mapping opponent_name -> opponent_spec (e.g. {"heuristic_bot": "bot_level_3"}).
                       If None, defaults to {"heuristic_bot": "bot_level_3"}.
            eval_freq: Evaluate every `eval_freq` callback rollout steps (on_step calls).
            n_eval_matches: Number of matches to simulate per opponent in each evaluation run.
            save_path: Directory path where `best_model.zip` and evaluation metadata will be saved.
            deterministic: Whether to use deterministic actions in `model.predict()`.
            eval_seeds: Seed or list of seeds for deterministic match evaluation.
                        Default uses fixed reproducible seeds [10000, 10001, ...] for each match.
            verbose: Verbosity level (0 = silent, 1 = table and save notices).
        """
        super().__init__(verbose)

        if eval_env_factory is None:
            self.eval_env_factory = self._default_eval_env_factory
        else:
            self.eval_env_factory = eval_env_factory

        if opponents is None:
            self.opponents = {"heuristic_bot": "bot_level_3"}
        else:
            self.opponents = opponents

        self.eval_freq = eval_freq
        self.n_eval_matches = n_eval_matches
        self.save_path = save_path
        self.deterministic = deterministic
        self.verbose = verbose

        # Configurable deterministic evaluation seeds for fair comparison across checkpoints
        if eval_seeds is None or eval_seeds is True:
            self.eval_seeds = [10_000 + i for i in range(self.n_eval_matches)]
        elif isinstance(eval_seeds, int):
            self.eval_seeds = [eval_seeds + i for i in range(self.n_eval_matches)]
        elif isinstance(eval_seeds, (list, tuple)):
            self.eval_seeds = list(eval_seeds)
        else:
            self.eval_seeds = None

        self.best_match_score = -np.inf
        self.best_goal_diff = -np.inf
        self.last_eval_metrics: Dict[str, Any] = {}
        self._cached_envs: Dict[str, Any] = {}
        self._custom_logger = None

        os.makedirs(self.save_path, exist_ok=True)

    def _default_eval_env_factory(self, opponent: Any) -> Any:
        """Create a headless RocketLeagueEnv configured for the specified opponent."""
        from src.rocket_league_env import RocketLeagueEnv
        if isinstance(opponent, str):
            return RocketLeagueEnv(render_mode=None, bot_type=opponent, terminate_on_goal=True)
        elif isinstance(opponent, dict):
            return RocketLeagueEnv(render_mode=None, **opponent)
        return RocketLeagueEnv(render_mode=None, bot_type="bot_level_3", terminate_on_goal=True)

    def _get_eval_env(self, opponent_name: str, opponent: Any) -> Any:
        """Retrieve cached environment for opponent or instantiate a new one."""
        if opponent_name not in self._cached_envs:
            self._cached_envs[opponent_name] = self.eval_env_factory(opponent)
        return self._cached_envs[opponent_name]

    def _evaluate_opponent(self, opponent_name: str, opponent: Any, n_matches: int) -> Dict[str, int]:
        """Simulate n_matches against a specific frozen opponent."""
        env = self._get_eval_env(opponent_name, opponent)

        result = {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "games": 0,
        }

        for match_idx in range(n_matches):
            seed = self.eval_seeds[match_idx] if (self.eval_seeds and match_idx < len(self.eval_seeds)) else None
            obs, info = env.reset(seed=seed)

            terminated = False
            truncated = False

            while not (terminated or truncated):
                action, _ = self.model.predict(
                    obs,
                    deterministic=self.deterministic,
                )
                obs, reward, terminated, truncated, info = env.step(action)

            # Scoreboard comes from objective game state, NOT shaped PPO reward
            if "goals_for" in info and "goals_against" in info:
                goals_for = int(info["goals_for"])
                goals_against = int(info["goals_against"])
            else:
                # Fallback to unwrapped sim state if info was filtered by external wrapper
                sim = getattr(getattr(env, "unwrapped", env), "sim", None)
                goals_for = int(sim.score_blue) if sim is not None else 0
                goals_against = int(sim.score_orange) if sim is not None else 0

            result["goals_for"] += goals_for
            result["goals_against"] += goals_against
            result["games"] += 1

            if goals_for > goals_against:
                result["wins"] += 1
            elif goals_for < goals_against:
                result["losses"] += 1
            else:
                result["draws"] += 1

        return result

    def _calculate_metrics(self, result: Dict[str, int]) -> Dict[str, float]:
        """Compute standard soccer performance rates and normalized match points."""
        n = max(result["games"], 1)
        wins = result["wins"]
        draws = result["draws"]
        losses = result["losses"]
        gf = result["goals_for"]
        ga = result["goals_against"]

        return {
            "win_rate": float(wins / n),
            "draw_rate": float(draws / n),
            "loss_rate": float(losses / n),
            "goals_for": float(gf / n),
            "goals_against": float(ga / n),
            "goal_diff": float((gf - ga) / n),
            # Normalized soccer match points: 3 for win, 1 for draw, 0 for loss
            "match_score": float((3 * wins + draws) / (3 * n)),
        }

    @property
    def logger(self):
        """Safely retrieve logger from _custom_logger, _logger, or model, or None if unavailable."""
        if getattr(self, "_custom_logger", None) is not None:
            return self._custom_logger
        if getattr(self, "_logger", None) is not None:
            return self._logger
        if getattr(self, "model", None) is not None and hasattr(self.model, "logger"):
            return self.model.logger
        return None

    @logger.setter
    def logger(self, val):
        self._custom_logger = val
        self._logger = val

    def _log_result(self, prefix: str, result: Dict[str, int], dump: bool = False):
        """Record evaluation metrics and raw match totals to the SB3 logger / TensorBoard."""
        logger = self.logger
        if logger is None:
            return

        metrics = self._calculate_metrics(result)

        for name, value in metrics.items():
            logger.record(f"{prefix}/{name}", value)

        logger.record(f"{prefix}/wins", result["wins"])
        logger.record(f"{prefix}/draws", result["draws"])
        logger.record(f"{prefix}/losses", result["losses"])
        logger.record(f"{prefix}/total_goals_for", result["goals_for"])
        logger.record(f"{prefix}/total_goals_against", result["goals_against"])

        if dump:
            logger.dump(self.num_timesteps)

    def evaluate(self) -> Dict[str, Any]:
        """Public method to run full multi-opponent evaluation and update best checkpoint.

        Can be called manually or automatically from `_on_step`.
        """
        overall = {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "games": 0,
        }

        opponent_results: Dict[str, Dict[str, Any]] = {}

        # --------------------------------
        # 1. Evaluate each frozen opponent
        # --------------------------------
        for opponent_name, opponent in self.opponents.items():
            result = self._evaluate_opponent(
                opponent_name,
                opponent,
                self.n_eval_matches,
            )
            metrics = self._calculate_metrics(result)
            self._log_result(f"eval_soccer/{opponent_name}", result)

            opponent_results[opponent_name] = {
                "raw": result,
                "metrics": metrics,
            }

            for key in overall:
                overall[key] += result[key]

        # --------------------------------
        # 2. Overall performance
        # --------------------------------
        overall_metrics = self._calculate_metrics(overall)
        self._log_result("eval_soccer/overall", overall, dump=True)

        match_score = overall_metrics["match_score"]
        goal_diff = overall_metrics["goal_diff"]

        # --------------------------------
        # 3. Model selection & checkpointing
        # --------------------------------
        better = (
            match_score > self.best_match_score
            or (
                np.isclose(match_score, self.best_match_score, atol=1e-5)
                and goal_diff > self.best_goal_diff
            )
        )

        saved = False
        saved_path = None
        if better:
            prev_best_score = self.best_match_score
            prev_best_diff = self.best_goal_diff

            self.best_match_score = match_score
            self.best_goal_diff = goal_diff

            model_save_path = os.path.join(self.save_path, "best_model")
            self.model.save(model_save_path)
            saved = True
            saved_path = f"{model_save_path}.zip"

            # Save detailed evaluation metadata alongside checkpoint
            metadata_path = os.path.join(self.save_path, "best_model_eval_metrics.json")
            metadata = {
                "timestep": int(self.num_timesteps),
                "n_calls": int(self.n_calls),
                "match_score": float(match_score),
                "goal_diff": float(goal_diff),
                "previous_best_match_score": float(prev_best_score) if np.isfinite(prev_best_score) else None,
                "previous_best_goal_diff": float(prev_best_diff) if np.isfinite(prev_best_diff) else None,
                "overall": overall_metrics,
                "opponents": opponent_results,
            }
            try:
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: could not write evaluation metadata: {e}")

        # --------------------------------
        # 4. Console report
        # --------------------------------
        if self.verbose >= 1:
            self._print_evaluation_summary(opponent_results, overall_metrics, overall, saved, saved_path)

        self.last_eval_metrics = {
            "overall_metrics": overall_metrics,
            "overall_raw": overall,
            "opponents": opponent_results,
            "is_best": saved,
        }

        return self.last_eval_metrics

    def _print_evaluation_summary(
        self,
        opponent_results: Dict[str, Dict[str, Any]],
        overall_metrics: Dict[str, float],
        overall_raw: Dict[str, int],
        is_new_best: bool,
        saved_path: Optional[str] = None,
    ):
        """Format and print an informative evaluation scoreboard to stdout."""
        header = f"\n{'='*75}\n🏆 SOCCER EVALUATION @ TIMESTEP {self.num_timesteps:,}\n{'='*75}"
        cols = f"{'Opponent':<18} | {'Score':<7} | {'Win %':<7} | {'W / D / L':<12} | {'GF / GA':<9} | {'Diff/G':<7}"
        sep = f"{'-'*19}+{'-'*9}+{'-'*9}+{'-'*14}+{'-'*11}+{'-'*8}"

        rows = [header, cols, sep]
        for name, data in opponent_results.items():
            m = data["metrics"]
            r = data["raw"]
            wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
            gf_ga = f"{r['goals_for']}/{r['goals_against']}"
            diff_str = f"{m['goal_diff']:+.2f}"
            rows.append(
                f"{name:<18} | {m['match_score']:.3f}   | {m['win_rate']*100:5.1f}% | {wdl:<12} | {gf_ga:<9} | {diff_str:<7}"
            )

        rows.append(sep)
        total_wdl = f"{overall_raw['wins']}/{overall_raw['draws']}/{overall_raw['losses']}"
        total_gf_ga = f"{overall_raw['goals_for']}/{overall_raw['goals_against']}"
        overall_diff = f"{overall_metrics['goal_diff']:+.2f}"
        rows.append(
            f"{'OVERALL':<18} | {overall_metrics['match_score']:.3f}   | {overall_metrics['win_rate']*100:5.1f}% | {total_wdl:<12} | {total_gf_ga:<9} | {overall_diff:<7}"
        )
        rows.append("="*75)

        if is_new_best:
            rows.append(f"⭐ NEW BEST SOCCER MODEL SAVED!")
            rows.append(f"   Match Score: {overall_metrics['match_score']:.3f} (3W+D)/(3N) | Goal Diff: {overall_metrics['goal_diff']:+.3f}")
            if saved_path:
                rows.append(f"   Checkpoint:  {saved_path}")
            rows.append("="*75)

        print("\n".join(rows))

    def _on_step(self) -> bool:
        """Triggered on each step by Stable-Baselines3."""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.evaluate()
        return True

    def _on_training_end(self) -> None:
        """Clean up and close any cached evaluation environments."""
        for env in self._cached_envs.values():
            try:
                env.close()
            except Exception:
                pass
        self._cached_envs.clear()
