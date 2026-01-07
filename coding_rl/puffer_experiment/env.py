import gymnasium as gym
import numpy as np
import pufferlib.emulation
from ..problems import PROBLEMS

class PufferCodingEnv(pufferlib.emulation.GymnasiumPufferEnv):
    def __init__(self, env=None):
        self.env = SimpleCodingEnv()
        super().__init__(env=self.env)

class SimpleCodingEnv(gym.Env):
    def __init__(self):
        self.problems = PROBLEMS
        self.current_problem_idx = 0

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0, high=len(self.problems), shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_problem_idx = np.random.randint(0, len(self.problems))
        return np.array([self.current_problem_idx], dtype=np.float32), {}

    def step(self, action):
        reward = 1.0 if action == 0 else 0.0
        self.current_problem_idx = np.random.randint(0, len(self.problems))
        obs = np.array([self.current_problem_idx], dtype=np.float32)

        return obs, reward, False, False, {}
