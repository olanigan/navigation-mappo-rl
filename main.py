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

    # 3. Create the View (Renderer)
    window = SimulationWindow()

    def update(dt):
        # Controller logic: step the environment, get the state, and send it to the renderer
        state = environment.get_render_state()

        vel = Vector2(x=0, y=0)
        if state.agents:
            agent_pos = state.agents[0].position
            cursor_x, cursor_y = window.cursor_pos

            vel_x = cursor_x - agent_pos[0]
            vel_y = cursor_y - agent_pos[1]
            vel = Vector2(x=vel_x, y=vel_y)

        done = environment.step([vel])
        state = environment.get_render_state()
        window.render(state)
        if done:
            arcade.exit()

    # 4. Schedule the update loop
    arcade.schedule(update, DELTA_T)

    # 5. Run the application
    arcade.run()


if __name__ == "__main__":
    main()
