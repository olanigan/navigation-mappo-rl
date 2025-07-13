import yaml
import pickle
import numpy as np
import time
import os
from stable_baselines3 import PPO
from nav.environment import Environment
import supersuit as ss

model_path = "./best_model/best_model"
config_path = "configs/basic_env.yaml"

config = yaml.safe_load(open(config_path))
env = Environment(config, render_mode="human")

env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")

model = PPO.load(model_path)
obs, _ = env.reset()
episode_reward = 0
episode_length = 0
done = False

while not done:
    # Use the trained model to predict actions
    action, _states = model.predict(obs, deterministic=True)

    # Step the environment
    obs, reward, terminated, truncated, _ = env.step(action)

    # Handle both single value and array rewards
    if isinstance(reward, (list, np.ndarray)):
        episode_reward += reward[0]
    else:
        episode_reward += reward

    episode_length += 1

    # Small delay to make it watchable
    time.sleep(1 / 60)

    # Break if episode is done
    if terminated or truncated:
        break

print(episode_reward)
