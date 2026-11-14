"""
Run a head-to-head match with the policy paths set below.

Edit the settings, then run from the repo root:

    python run_head_to_head.py

Set a model path to None to play the heuristic bot on that side.
"""

from src.sim.ai import heuristic_bot
from src.eval.head_to_head import play_head_to_head

heuristic_bot = None
DDQN_none = "./runs/20261114_0855_dueling_ddqn_s1_none/checkpoints/model_dueling_ddqn_s1_none_ep024000.pt"
PPO_none = "./runs/20261114_0855_ppo_s1_none/checkpoints/model_ppo_s1_none_ep011000.pt"
DDQN_s2 = "./runs/20261114_0953_dueling_ddqn_s2_heuristic/checkpoints/model_dueling_ddqn_s2_heuristic_ep007000.pt"
PPO_s2 = "./runs/20261114_0953_ppo_s2_heuristic/checkpoints/model_ppo_s2_heuristic_ep006500.pt"
DDQN_s3_r1 = "./runs/20261114_1015_dueling_ddqn_s3_self_r1/checkpoints/model_dueling_ddqn_s3_self_r1_ep021500.pt"
PPO_s3_r1 = "./runs/20261114_1035_ppo_s3_self_r1/checkpoints/model_ppo_s3_self_r1_ep017500.pt"
DDQN_s3_r2 = "./runs/20261114_1102_dueling_ddqn_s3_self_r2/checkpoints/model_dueling_ddqn_s3_self_r2_ep012000.pt"
PPO_s3_r2 = "./runs/20261114_0447_ppo_s3_self_r2/checkpoints/model_ppo_s3_self_r2_ep001000.pt"

# ---- Settings -----------------------------------------------------------------
BLUE_MODEL = None
ORANGE_MODEL = DDQN_s3_r2

RENDER = False # False = headless, runs as fast as possible
MAX_GOALS = None  # stop after this many total goals (None = no limit)
MAX_SECONDS = 600.0  # stop after this much sim time (None = no limit)
RANDOM_SPAWN = False  # randomise every kickoff
SEED = None  # makes random spawns repeatable
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    play_head_to_head(
        blue_model=BLUE_MODEL,
        orange_model=ORANGE_MODEL,
        render=RENDER,
        max_goals=MAX_GOALS,
        max_seconds=MAX_SECONDS,
        random_spawn=RANDOM_SPAWN,
        seed=SEED,
    )
