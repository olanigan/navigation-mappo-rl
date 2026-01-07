import pufferlib
import pufferlib.vector
import time
import gymnasium as gym
from .env import PufferCodingEnv

def make_env(*args, **kwargs):
    return PufferCodingEnv()

def run_puffer_demo():
    print("Initializing PufferLib Vectorization Demo...")

    # 1. Create a Vectorized Environment
    # PufferLib can run these in parallel processes or threads extremely efficiently.
    num_envs = 100
    print(f"Spinning up {num_envs} parallel coding environments...")

    # Wrap the environment creation function
    # PufferLib's GymnasiumSerial/Multiprocessing wrappers provide the vector interface
    vec_env = pufferlib.vector.Serial(
        env_creators=[make_env] * num_envs,
        env_args=[[]] * num_envs,
        env_kwargs=[{}] * num_envs,
        num_envs=num_envs
    )

    print("Environments ready. Starting rollout...")
    start_time = time.time()

    # 2. Reset
    obs, _ = vec_env.reset()

    total_steps = 10000
    steps_per_batch = num_envs
    batches = total_steps // steps_per_batch

    for i in range(batches):
        # 3. Random Actions (simulating a policy)
        actions = [vec_env.single_action_space.sample() for _ in range(num_envs)]

        # 4. Step (Execute 'code' in 100 envs at once)
        obs, rewards, terminals, truncations, infos = vec_env.step(actions)

        if i % 10 == 0:
            print(f"Batch {i}/{batches}: Collected {steps_per_batch} steps. Mean Reward: {sum(rewards)/len(rewards):.2f}")

    end_time = time.time()
    duration = end_time - start_time
    sps = total_steps / duration

    print(f"\n--- Performance Report ---")
    print(f"Total Steps: {total_steps}")
    print(f"Time Taken: {duration:.4f}s")
    print(f"Steps Per Second: {sps:.2f}")
    print("Note: Real code execution would be slower, but PufferLib ensures overhead is minimal.")

    vec_env.close()

if __name__ == "__main__":
    run_puffer_demo()
