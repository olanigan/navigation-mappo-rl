from .obstacles import PolygonBoundary, Obstacle
from .utils import *
from .config_models import *
from .renderer_models import RenderState, AgentState, ObstacleState, BoundaryState
from .obstacles import ObstacleFactory
import yaml
from typing import List
from .ray_intersection import (
    batch_ray_intersection_detailed,
    create_lidar_rays,
    RayIntersectionOutput,
)
from collections import deque

DELTA_T = 1 / 60


class Agent:
    def __init__(self, agent_config: AgentConfig):
        self.config = agent_config
        self.pos = self.config.start_pos.to_numpy()
        self.radius = self.config.radius
        self.current_speed = 0.1
        self.goal_rectangle = self.config.goal_rectangle
        self.goal_pos = self.config.goal_rectangle.center.to_numpy()
        _, self.direction = convert_to_polar(self.goal_pos - self.pos)
        self.response_time = 0.5
        self.active = True
        self.lidar_observation_history = deque(maxlen=4)
        self.last_raw_lidar_observation = None
        self.goal_reached = False

    def has_reached_goal(self):
        return circle_rectangle_intersection(
            self.pos,
            self.radius,
            self.goal_rectangle.center.to_numpy(),
            self.goal_rectangle.width,
            self.goal_rectangle.height,
            self.goal_rectangle.rotation,
        )

    def process_lidar_observation(self, lidar_observation: list[RayIntersectionOutput]):
        rays = np.zeros((len(lidar_observation), 3))
        for i in range(len(lidar_observation)):
            ray_data = lidar_observation[i]
            if not ray_data.intersects:
                continue

            if ray_data.intersecting_with == "obstacle":
                rays[i, 0] = self.config.max_range - ray_data.t
            elif ray_data.intersecting_with == "boundary":
                rays[i, 1] = self.config.max_range - ray_data.t
            elif ray_data.intersecting_with == "agent":
                rays[i, 2] = self.config.max_range - ray_data.t

        return rays

    def get_action(self, lidar_observation: list[RayIntersectionOutput]):
        self.last_raw_lidar_observation = lidar_observation
        processed_lidar_observation = self.process_lidar_observation(lidar_observation)
        self.lidar_observation_history.append(processed_lidar_observation)
        return None  # TODO: Implement action selection

    def update_pos(self, delta_t: float = 1 / 30):

        if self.goal_reached:
            return

        if not self.active:
            return
        self.pos = self.pos + (self.direction * self.current_speed * delta_t)
        self.goal_reached = self.has_reached_goal()

    def apply_target_velocity(self, target_velocity: Vector2):
        if not self.active:
            self.current_speed = 0
            return

        current_velocity = self.current_speed * self.direction
        force = target_velocity.to_numpy() - current_velocity
        new_velocity = current_velocity + force * (self.response_time * DELTA_T)

        # Handle zero velocity case to avoid division by zero
        velocity_magnitude = np.linalg.norm(new_velocity)
        if velocity_magnitude > 1e-10:  # Small threshold to avoid numerical issues
            self.current_speed = velocity_magnitude
            self.direction = (
                new_velocity / velocity_magnitude
            )  # Normalize to unit vector
        else:
            self.current_speed = 0.0
            # Keep previous direction when speed is zero

        # Clamp speed to maximum (was incorrectly using max instead of min)
        self.current_speed = min(self.current_speed, self.config.max_speed)


class Environment:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.boundary = PolygonBoundary(config.boundary)
        self.agents = [Agent(agent_config) for agent_config in config.agents]
        self.obstacles = [
            ObstacleFactory.create(obstacle) for obstacle in config.obstacles
        ]
        self.num_steps = 0

    def step(self, actions):
        assert len(actions) == len(self.agents)
        self.num_steps += 1
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
        done = all(
            [(not agent.active) or (agent.goal_reached) for agent in self.agents]
        )

        return done

    def get_lidar_observation(self):
        all_rays = []
        goal_rectangles = []
        rays_per_agent = []

        for agent in self.agents:
            rays = create_lidar_rays(
                agent.pos,
                agent.direction,
                self.config.num_rays,
                agent.config.max_range,
                agent.config.fov_degrees,
            )
            all_rays.extend(rays)
            goal_rectangles.append(agent.goal_rectangle)
            rays_per_agent.append(self.config.num_rays)

        all_rays = np.array(all_rays)
        result = batch_ray_intersection_detailed(
            all_rays,
            self.obstacles,
            [self.config.boundary],
            goal_rectangles=goal_rectangles,
            rays_per_agent=rays_per_agent,
        )

        result = np.reshape(result, (len(self.agents), self.config.num_rays))

        return result

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
                direction=(agent.direction[0], agent.direction[1]),
                lidar_observation=agent.last_raw_lidar_observation,
                fov_degrees=agent.config.fov_degrees,
                max_range=agent.config.max_range,
                goal_rectangle=agent.goal_rectangle,
                goal_reached=agent.goal_reached,
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
            agents=agent_states,
            obstacles=obstacle_states,
            boundary=boundary_state,
        )


if __name__ == "__main__":
    config_file = "configs/basic_env.yaml"

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    env_config = EnvConfig(**config_data)
    env = Environment(env_config)
    env.step()
