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
        self.start_pos = self.pos.copy()
        self.radius = self.config.radius
        self.current_speed = 0.1
        self.goal_pos = self.config.goal_pos.to_numpy()
        _, self.direction = convert_to_polar(self.goal_pos - self.pos)
        self.response_time = 0.5
        self.active = True
        self.lidar_observation_history = deque(maxlen=4)
        self.last_raw_lidar_observation = None
        self.goal_reached = False

    def has_reached_goal(self, goal_threshold: float = 0.02):
        return np.linalg.norm(self.goal_pos - self.pos) < goal_threshold

    def get_state_dict(self):
        original_distance_to_goal = np.linalg.norm(self.goal_pos - self.start_pos)
        current_distance_to_goal = np.linalg.norm(self.goal_pos - self.pos)
        progress = (
            original_distance_to_goal - current_distance_to_goal
        ) / original_distance_to_goal

        goal_vector = self.goal_pos - self.pos
        goal_vector = goal_vector / np.linalg.norm(goal_vector)
        cosine_angle = goal_vector.dot(self.direction)

        speed_ratio = self.current_speed / self.config.max_speed

        return {
            "state_vector": [
                progress,  # progress towards goal 0-1
                cosine_angle,  # cosine of angle between goal vector and direction vector
                speed_ratio,  # ratio of current speed to max speed
                speed_ratio * cosine_angle,
                current_distance_to_goal,
            ],
        }

    def get_action(self, lidar_observation: np.ndarray):
        self.lidar_observation_history.append(lidar_observation)
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


COLLIDING_WITH_TYPES = Literal["obstacle", "boundary", "agent"]


class CollisionData(BaseModel):
    is_colliding: bool = False
    colliding_with: Optional[COLLIDING_WITH_TYPES] = None


class Environment:
    def __init__(self, config: EnvConfig):
        self.config = config
        self.boundary = PolygonBoundary(config.boundary)
        self.agents = [Agent(agent_config) for agent_config in config.agents]
        self.obstacles = [
            ObstacleFactory.create(obstacle) for obstacle in config.obstacles
        ]
        self.num_steps = 0

    def calculate_reward(self, agent: Agent, collision_data: CollisionData):
        if agent.goal_reached:
            return 5

        if collision_data.is_colliding:
            return -1

        goal_reward = agent.direction.dot(agent.goal_pos - agent.pos)
        scale_goal_reward_with_speed = goal_reward * (
            agent.current_speed / agent.config.max_speed
        )

        stay_alive_reward = -0.1

        return scale_goal_reward_with_speed + stay_alive_reward

    def step(self, actions):
        assert len(actions) == len(self.agents)
        self.num_steps += 1
        for agent, action in zip(self.agents, actions):
            agent.apply_target_velocity(action)
        for obs in self.obstacles:
            obs.update(DELTA_T)

        collision_datas = []
        for agent in self.agents:
            this_agent_collision_data = CollisionData()
            agent.update_pos(DELTA_T)
            if self.boundary.violating_boundary(agent):
                agent.active = False
                this_agent_collision_data.is_colliding = True
                this_agent_collision_data.colliding_with = "boundary"

            for obs in self.obstacles:
                if obs.check_collision(center=agent.pos, radius=agent.radius):
                    agent.active = False
                    this_agent_collision_data.is_colliding = True
                    this_agent_collision_data.colliding_with = "obstacle"

            # TODO: Check for collisions with other agents

            collision_datas.append(this_agent_collision_data)

        rewards = {
            idx: self.calculate_reward(agent, collision_data)
            for idx, (agent, collision_data) in enumerate(
                zip(self.agents, collision_datas)
            )
        }

        terminations = {
            idx: (not agent.active) or (agent.goal_reached)
            for idx, agent in enumerate(self.agents)
        }

        truncations = {idx: False for idx in range(len(self.agents))}

        lidar_observations = self.get_lidar_observation()

        for agent, lidar_observation in zip(self.agents, lidar_observations):
            agent.last_raw_lidar_observation = lidar_observation

        processed_lidar_observations = [
            self.process_lidar_observation(agent.config.max_range, lidar_observation)
            for agent, lidar_observation in zip(self.agents, lidar_observations)
        ]

        agent_state_dicts = [agent.get_state_dict() for agent in self.agents]

        next_states = {
            idx: {
                **agent_state_dict,
                "lidar": processed_lidar_observation,
            }
            for idx, (agent_state_dict, processed_lidar_observation) in enumerate(
                zip(agent_state_dicts, processed_lidar_observations)
            )
        }

        return next_states, rewards, terminations, truncations, {}

    def process_lidar_observation(
        self, max_range, lidar_observation: list[RayIntersectionOutput]
    ):
        rays = np.zeros((len(lidar_observation), 3))
        for i in range(len(lidar_observation)):
            ray_data = lidar_observation[i]
            if not ray_data.intersects:
                continue

            if ray_data.intersecting_with == "obstacle":
                rays[i, 0] = max_range - ray_data.t
            elif ray_data.intersecting_with == "boundary":
                rays[i, 1] = max_range - ray_data.t
            elif ray_data.intersecting_with == "agent":
                rays[i, 2] = max_range - ray_data.t

        return rays

    def get_lidar_observation(self):
        all_rays = []
        goals = []
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

            goals.append(
                Circle(
                    center=Vector2(x=agent.goal_pos[0], y=agent.goal_pos[1]),
                    radius=self.config.goal_threshold,
                )
            )
            rays_per_agent.append(self.config.num_rays)

        all_rays = np.array(all_rays)
        result = batch_ray_intersection_detailed(
            all_rays,
            self.obstacles,
            [self.config.boundary],
            goals=goals,
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
                goals=Circle(
                    center=Vector2(x=agent.goal_pos[0], y=agent.goal_pos[1]),
                    radius=self.config.goal_threshold,
                ),
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
