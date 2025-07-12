from nav.environment import Environment

import yaml
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import VecMonitor
import os
from networks.actor_critic_network import ObservationEncoder
from callbacks import InferenceCallback

import sys

model_id = sys.argv[1]
os.makedirs(f"models/{model_id}", exist_ok=True)
os.makedirs(f"models/{model_id}/best_model", exist_ok=True)
os.makedirs(f"models/{model_id}/checkpoints", exist_ok=True)
os.makedirs(f"videos/{model_id}", exist_ok=True)
os.makedirs(f"logs/{model_id}", exist_ok=True)

config = yaml.safe_load(open("configs/basic_env.yaml"))

env = Environment(config)

agent_states_dim = env.agent_states_dim
lidar_dim = env.lidar_dim
env = ss.frame_stack_v1(env, 3)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 8, base_class="stable_baselines3")

# Add monitoring to track episode rewards
env = VecMonitor(env)

policy_kwargs = dict(
    features_extractor_class=ObservationEncoder,
    features_extractor_kwargs=dict(
        agent_states_dim=agent_states_dim,
        lidar_dim=lidar_dim,
        history_length=3,
    ),
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    batch_size=128,
    device="mps",
    policy_kwargs=policy_kwargs,
    tensorboard_log=f"./logs/{model_id}",
    learning_rate=5e-4,
    n_steps=1000,
    n_epochs=4,
    normalize_advantage=False,
    ent_coef=0.02,
)

eval_env = Environment(config)
eval_env = ss.frame_stack_v1(eval_env, 3)
eval_env = ss.pettingzoo_env_to_vec_env_v1(eval_env)
eval_env = ss.concat_vec_envs_v1(eval_env, 1, base_class="stable_baselines3")

# Create multiple callbacks
callbacks = [
    # Regular model checkpointing
    CheckpointCallback(
        save_freq=10000,
        save_path=f"models/{model_id}/checkpoints/",
        name_prefix="ppo_model",
    ),
    EvalCallback(
        eval_env,
        best_model_save_path=f"models/{model_id}/best_model/",
        log_path=f"logs/{model_id}",
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    ),
    InferenceCallback(
        config_path="configs/basic_env.yaml",
        inference_interval=25000,
        save_videos=True,
        video_dir=f"videos/{model_id}/training_progress",
        verbose=1,
    ),
]

# Train with all callbacks
model.learn(total_timesteps=1e7, callback=CallbackList(callbacks))
