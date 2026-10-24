from stable_baselines3 import A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from ME5418_RocketLeague.config.gym_config import GymConfig
# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym

# # # # # # if GymConfig.gym_id not in gym.envs.registry:
# # # # #     gym.register(
# # # #         id=GymConfig.gym_id,
# # #         entry_point="ME5418_RocketLeague.gym_environment.bentjh_gym:RocketLeagueSideSwipeGym",
# #         max_episode_steps=GymConfig.max_steps
#     )

class TestNN():
    def __init__(self):
        pass

def check_environment():
    from gymnasium.utils.env_checker import check_env

    gym_env = gym.make(id=GymConfig.gym_id, render_mode=None, disable_env_checker=True)
    check_env(gym_env.unwrapped)
    print("Environment check passed. gym.make successful")

class SB3ActorCritic():
    def __init__(self):
        self.file_name = "a2c_v4"
        pass
    def train(self):
        train_env = gym.make(id=GymConfig.gym_id, 
                             render_mode=GymConfig.train_render_mode, 
                             adversary_type=GymConfig.train_bot_type)
        train_env = gym.wrappers.NormalizeReward(train_env, gamma=0.99, epsilon=1e-8)

        model = A2C("MlpPolicy", train_env)
        model.learn(10_000)
        model.save(f"./ME5418_RocketLeague/gym_environment/{self.file_name}")
        train_env.close()

    def test(self):
        test_env = gym.make(id=GymConfig.gym_id, 
                       render_mode=GymConfig.test_render_mode, 
                    #    render_mode=None, 
                       adversary_type=GymConfig.test_bot_type)
        # print("Metadata:", test_env.unwrapped.metadata)
        # print("Render Mode:", test_env.unwrapped.render_mode) 
        loaded_model = A2C.load(f"./ME5418_RocketLeague/gym_environment/{self.file_name}")
        observed_state, info = test_env.reset()

        done = False
        action = test_env.action_space.sample()
        while test_env.unwrapped.is_open:
            action, _states = loaded_model.predict(observed_state, deterministic=True)
            observed_state, reward, terminated, truncated, info = test_env.step(action)
            test_env.render()
            print(f"{reward:0.1f}")
        test_env.close()

            # if not test_env.unwrapped.is_open:
                # break
            # if terminated or truncated:
                # test_env.reset()
                # pass
            # count += 1

def test_net(args = None):
    actor_critic = SB3ActorCritic()
    # actor_critic.train()
    actor_critic.test()