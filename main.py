import time
import yaml
import arcade
import argparse
import os
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

    # 3. Create the View (Renderer) with locked 30 FPS
    target_fps = 30  # Set your desired FPS here
    window = SimulationWindow(target_fps=target_fps, record=args.record)

    def update(dt):

        if environment.num_steps % 2 == 0:
            lidar_observation = environment.get_lidar_observation()

            for idx, agent in enumerate(environment.agents):
                _ = agent.get_action(lidar_observation[idx])

        vel = Vector2(x=0, y=0)
        if environment.agents:
            agent_pos = environment.agents[0].pos
            cursor_x, cursor_y = window.cursor_pos

            vel_x = cursor_x - agent_pos[0]
            vel_y = cursor_y - agent_pos[1]
            vel = Vector2(x=vel_x, y=vel_y)

        _, rewards, done, _, _ = environment.step([vel])
        render_state = environment.get_render_state()
        window.render(render_state)
        if all(done.values()):
            if args.record:
                print("🎬 Simulation finished - finalizing recording...")
                window.stop_recording()
            arcade.exit()

    # 4. Schedule the update loop with target FPS
    update_interval = 1.0 / window.target_fps
    arcade.schedule(update, update_interval)

    # Print usage information
    print(f"🎮 Starting simulation at {target_fps} FPS")
    if args.record:
        print("🎬 RECORDING ACTIVE - video will be saved to movies/footage.mp4")
    print("📖 Controls:")
    print("   Mouse - Control agent movement")
    print("   ESC - Quick exit")
    if args.record:
        print("   R - Stop recording manually")
    print(
        "   Close window - End simulation"
        + (" and save recording" if args.record else "")
    )

    # 5. Run the application
    arcade.run()


if __name__ == "__main__":
    main()
