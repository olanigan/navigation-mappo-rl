import time
import yaml
import arcade
from nav.config_models import EnvConfig, Vector2
from nav.environment import Environment, DELTA_T
from nav.live_renderer import SimulationWindow


def main():
    # 1. Load Configuration
    config_file = "configs/basic_env.yaml"
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
    env_config = EnvConfig(**config_data)

    # 2. Create the Model (Environment)
    environment = Environment(env_config)

    # 3. Create the View (Renderer) with locked 30 FPS
    target_fps = 30  # Set your desired FPS here
    window = SimulationWindow(target_fps=target_fps)

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

        done = environment.step([vel])
        render_state = environment.get_render_state()
        window.render(render_state)
        if done:
            arcade.exit()

    # 4. Schedule the update loop with target FPS
    update_interval = 1.0 / window.target_fps
    arcade.schedule(update, update_interval)
    # 5. Run the application
    arcade.run()


if __name__ == "__main__":
    main()
