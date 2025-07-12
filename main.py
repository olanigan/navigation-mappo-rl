import time
import yaml
import arcade
import argparse
import os
import numpy as np
from nav.config_models import EnvConfig, Vector2
from nav.environment import Environment, DELTA_T
from nav.live_renderer import SimulationWindow


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Navigation Simulation")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record the simulation to movies/footage.mp4",
    )
    args = parser.parse_args()

    # Create movies directory if recording
    if args.record:
        os.makedirs("movies", exist_ok=True)
        print("🎬 Recording enabled - will save to movies/footage.mp4")

    # 1. Load Configuration
    config_file = "configs/basic_env.yaml"
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
    env_config = EnvConfig(**config_data)

    # 2. Create the Model (Environment)
    environment = Environment(env_config)

    # 3. Reset environment to get initial state
    observations, infos = environment.reset()
    print(f"🤖 Initialized {len(environment.agents)} agents: {environment.agents}")

    # 4. Create the View (Renderer) with locked 30 FPS
    target_fps = 30  # Set your desired FPS here
    window = SimulationWindow(target_fps=target_fps, record=args.record)

    def update(dt):
        # Only process lidar every other step for performance
        if environment.num_steps % 2 == 0:
            lidar_observation = environment.get_lidar_observation()

            for idx, agent_id in enumerate(environment.agents):
                agent = environment.agents_dict[agent_id]
                if idx < len(lidar_observation):
                    _ = agent.get_action(lidar_observation[idx])

        # Get actions for all agents
        actions = {}

        # For demo purposes, control the first agent with mouse cursor
        if environment.agents:
            first_agent_id = environment.agents[0]
            first_agent = environment.agents_dict[first_agent_id]
            agent_pos = first_agent.pos
            cursor_x, cursor_y = window.cursor_pos

            vel_x = cursor_x - agent_pos[0]
            vel_y = cursor_y - agent_pos[1]
            actions[first_agent_id] = np.array([vel_x, vel_y])

            # Other agents get zero velocity (stationary)
            for agent_id in environment.agents[1:]:
                actions[agent_id] = np.array([0, 0])

        # Step the environment
        observations, rewards, terminations, truncations, infos = environment.step(
            actions
        )

        # Render the current state
        render_state = environment.get_render_state()
        window.render(render_state)

        # Check if simulation is done
        if (
            all(terminations.values())
            or all(truncations.values())
            or not environment.agents
        ):
            if args.record:
                print("🎬 Simulation finished - finalizing recording...")
                window.stop_recording()
            print("🏁 Simulation completed!")
            arcade.exit()

    # 5. Schedule the update loop with target FPS
    update_interval = 1.0 / window.target_fps
    arcade.schedule(update, update_interval)

    # Print usage information
    print(f"🎮 Starting simulation at {target_fps} FPS")
    if args.record:
        print("🎬 RECORDING ACTIVE - video will be saved to movies/footage.mp4")
    print("📖 Controls:")
    print("   Mouse - Control first agent movement")
    print("   ESC - Quick exit")
    if args.record:
        print("   R - Stop recording manually")
    print(
        "   Close window - End simulation"
        + (" and save recording" if args.record else "")
    )

    # 6. Run the application
    arcade.run()


if __name__ == "__main__":
    main()
