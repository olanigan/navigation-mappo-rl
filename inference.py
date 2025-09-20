import yaml
from nav.environment import Environment
import supersuit as ss
from moviepy import ImageSequenceClip
import os
import time
import random
import numpy as np
import json


def make_eval_env(config, history_length):
    eval_env = Environment(config, render_mode="rgb_array")
    eval_env = ss.frame_stack_v1(eval_env, history_length)
    eval_env = ss.black_death_v3(eval_env)
    eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
    return eval_env


def inference(env, model, video_path=None, num_episodes=5, mode="rgb_array"):
    """
    Run inference with a trained model.

    Args:
        model: A PPO model
        config_path: Path to the config YAML file
        video_path: Optional path to save video. If None, no video is saved.

    Returns:
        tuple: (episode_reward, episode_length)
    """

    all_episode_rewards = []
    all_episode_lengths = []
    frames = []
    log = []

    for i in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        while True:
            action = model.predict(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]

            obs, reward, terminated, truncated, info = env.step(action)

            # state_value = model.predict_state_value(info[0]["global_state"])
            # print(state_value, reward)

            episode_reward += reward[0]
            episode_length += 1

            episode_done = (terminated | truncated).all()

            if episode_done:
                break

            if mode == "human":
                time.sleep(1 / 30)

            if video_path and mode == "rgb_array":
                frame = env.render()
                frames.append(frame)

            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
    env.close()
    del env

    if video_path and len(frames) > 0:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(video_path)
    return np.mean(all_episode_rewards), np.mean(all_episode_lengths)


if __name__ == "__main__":
    # from rl.ppo import PPO
    from rl.mappo import MAPPO
    import sys

    model_id = sys.argv[1]

    if len(sys.argv) > 2:
        env_path = sys.argv[2]
    else:
        env_path = f"./models/{model_id}/env.yaml"
        if not os.path.exists(env_path):
            raise FileNotFoundError(f"Environment config file not found at {env_path}")

    mode = "human"  # "rgb_array"  # "human"
    model_path = (
        f"./models/{model_id}/best_model"  # "./models/mm1/best_model/best_model.zip"
    )

    print(f"Loading model from {model_path}")
    print(f"Loading environment config from {env_path}")

    config = yaml.safe_load(open(env_path))
    config["terminal_strategy"] = "individual"
    video_path = None  # "videos/inference_hallway_4.mp4"
    env = Environment(config, render_mode=mode)
    model = MAPPO.load_model(model_path, env)

    env = ss.frame_stack_v1(env, 4)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    episode_reward, episode_length = inference(env, model, video_path, num_episodes=1)
    print(f"Episode reward: {episode_reward}, Episode length: {episode_length}")
