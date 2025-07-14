import yaml
import numpy as np
from nav.environment import Environment
import supersuit as ss


def test_single_env():
    """Test termination with single environment"""
    print("=== Testing Single Environment ===")
    config = yaml.safe_load(open("configs/basic_env.yaml"))
    env = Environment(config)

    obs, _ = env.reset()
    print(f"Initial agents: {env.agents}")

    for step in range(100):
        # Take random actions
        actions = {}
        for agent_id in env.agents:
            actions[agent_id] = env.action_space(agent_id).sample()

        obs, rewards, terminations, truncations, _ = env.step(actions)

        if any(terminations.values()) or any(truncations.values()):
            print(f"Step {step}: Terminations: {terminations}")
            print(f"Step {step}: Truncations: {truncations}")
            break

    env.close()


def test_vectorized_env():
    """Test termination with vectorized environment"""
    print("\n=== Testing Vectorized Environment (concat_vec_envs_v1) ===")
    config = yaml.safe_load(open("configs/basic_env.yaml"))
    env = Environment(config)

    # Apply SuperSuit wrappers
    env = ss.frame_stack_v1(env, 4)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8)

    obs, _ = env.reset()
    print(f"Vectorized environment created with {env.num_envs} environments")

    for step in range(100):
        # Take random actions
        actions = env.action_space.sample()

        obs, rewards, terminated, truncated, _ = env.step(actions)

        print(f"Step {step}: Terminated: {terminated}, Type: {type(terminated)}")
        print(f"Step {step}: Truncated: {truncated}, Type: {type(truncated)}")

        if np.any(terminated) or np.any(truncated):
            print(f"Step {step}: Some environment terminated!")
            break

        if step > 20:  # Limit output
            break

    env.close()


if __name__ == "__main__":
    test_single_env()
    test_vectorized_env()
