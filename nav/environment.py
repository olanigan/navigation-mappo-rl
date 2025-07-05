from .obstacles import PolygonBoundary
from .utils import *
from .config_models import *
from .renderer_models import RenderState, AgentState, ObstacleState, BoundaryState
from .obstacles import ObstacleFactory
import yaml

DELTA_T = 1 / 60


class Agent:
    def __init__(self, agent_config: AgentConfig):
        self.config = agent_config
        self.pos = self.config.start_pos.to_numpy()
        self.radius = self.config.radius
        self.current_speed = 0.1
        self.goal_pos = self.config.goal_pos.to_numpy()
        _, self.direction = convert_to_polar(self.goal_pos - self.pos)
        self.response_time = 0.5
        self.active = True

    def update_pos(self, delta_t: float = 1 / 30):
        if not self.active:
            return
        self.pos = self.pos + (self.direction * self.current_speed * delta_t)

    def apply_target_velocity(self, target_velocity: Vector2):
        if not self.active:
            self.current_speed = 0
            return
        current_velocity = self.current_speed * self.direction
        force = target_velocity.to_numpy() - current_velocity
        new_velocity = current_velocity + force * (self.response_time * DELTA_T)
        self.current_speed, self.direction = convert_to_polar(new_velocity)
        self.current_speed = max(self.current_speed, self.config.max_speed)


class Environment:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.boundary = PolygonBoundary(config.boundary)
        self.agents = [Agent(agent_config) for agent_config in config.agents]
        self.obstacles = [
            ObstacleFactory.create(obstacle) for obstacle in config.obstacles
        ]

    def step(self, actions):
        assert len(actions) == len(self.agents)
        for agent, action in zip(self.agents, actions):
            agent.apply_target_velocity(action)
        for obs in self.obstacles:
            obs.update(DELTA_T)
        for agent in self.agents:
            agent.update_pos(DELTA_T)
            if self.boundary.violating_boundary(agent):
                agent.active = False
            for obs in self.obstacles:
                if obs.check_collision(center=agent.pos, radius=agent.radius):
                    agent.active = False
        done = all([(not agent.active) for agent in self.agents])

        return done

    def get_render_state(self) -> RenderState:
        agent_states = [
            AgentState(
                position=(agent.pos[0], agent.pos[1]),
                radius=agent.radius,
                color=agent.config.agent_col,
                velocity=(
                    agent.current_speed * agent.direction[0],
                    agent.current_speed * agent.direction[1],
                ),
            )
            for agent in self.agents
        ]

        obstacle_states = []
        for obs in self.obstacles:
            obstacle_states.append(obs.get_current_state())

        boundary_state = BoundaryState(
            vertices=[(v[0], v[1]) for v in self.boundary.vertices]
        )

        return RenderState(
            agents=agent_states, obstacles=obstacle_states, boundary=boundary_state
        )


if __name__ == "__main__":
    config_file = "configs/basic_env.yaml"

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    env_config = EnvConfig(**config_data)
    env = Environment(env_config)
    env.step()
