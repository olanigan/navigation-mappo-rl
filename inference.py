import yaml
from stable_baselines3 import PPO
from nav.environment import Environment
import supersuit as ss
from stable_baselines3.common.vec_env import VecVideoRecorder
from moviepy import ImageSequenceClip
import os
import time


def inference(model, config, video_path=None, mode="rgb_array"):
    """
    Run inference with a trained model.

    Args:
        model: Either a PPO model object or a string path to a saved model
        config_path: Path to the config YAML file
        video_path: Optional path to save video. If None, no video is saved.

    Returns:
        tuple: (episode_reward, episode_length)
    """
    # Load model if path is provided
    if isinstance(model, str):
        model = PPO.load(model)

    env = Environment(config, render_mode=mode)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    obs, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    frames = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        # if episode_length > 100:
        #     action = [[-1, 1]]
        # else:
        #     action = [[1, 1]]
        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward[0]
        episode_length += 1

        # if mode == "human":
        #    time.sleep(1 / 30)

        if mode == "rgb_array":
            video_path = "temp.mp4" if video_path is None else video_path

        if video_path:
            frame = env.render()
            frames.append(frame)

        if terminated or truncated:
            break

    env.close()
    del env

    # Create video if path provided and frames were collected
    if video_path and frames:
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        clip = ImageSequenceClip(frames, fps=30)
        clip.write_videofile(video_path)

    return episode_reward, episode_length


if __name__ == "__main__":
    mode = "none"
    model_path = "./models/m3/best_model/best_model.zip"
    config = yaml.safe_load(open("configs/basic_env.yaml"))
    video_path = None  # "videos/inference_demo.mp4"
    episode_reward, episode_length = inference(
        model_path, config, video_path, mode=mode
    )
    print(f"Episode reward: {episode_reward}, Episode length: {episode_length}")
