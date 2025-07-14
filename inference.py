import yaml
from nav.environment import Environment
import supersuit as ss
from moviepy import ImageSequenceClip
import os
import time
import random
import numpy as np


def make_eval_env(config, history_length):
    eval_env = Environment(config, render_mode="rgb_array")
    eval_env = ss.frame_stack_v1(eval_env, history_length)
    # eval_env = ss.black_death_v3(eval_env)
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

    for i in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            action = model.predict(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward[0]
            episode_length += 1

            if mode == "human":
                time.sleep(1 / 15)

            if video_path:
                frame = env.render()
                frames.append(frame)

            if any(terminated) or any(truncated):
                break

            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
    env.close()
    del env
    # Create video if path provided and frames were collected
    if video_path and frames:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(video_path)

    return np.mean(all_episode_rewards), np.mean(all_episode_lengths)


if __name__ == "__main__":
    from rl.ppo import PPO

    mode = "none"  # "human"
    model_path = "./models/test8/best_model"  # "./models/mm1/best_model/best_model.zip"
    model = PPO.load_model(model_path)
    config = yaml.safe_load(open("configs/basic_env.yaml"))
    video_path = None  # "videos/inference_demo.mp4"
    env = Environment(config, render_mode=mode)

    env = ss.frame_stack_v1(env, 4)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    episode_reward, episode_length = inference(env, model, video_path, num_episodes=5)
    print(f"Episode reward: {episode_reward}, Episode length: {episode_length}")
