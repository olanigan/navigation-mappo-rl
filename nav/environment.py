import pettingzoo
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from .obstacles import PolygonBoundary, Obstacle
from .utils import sample_point_in_rectangle, convert_to_polar
from .config_models import *
from .renderer_models import RenderState, AgentState, ObstacleState, BoundaryState
from .obstacles import ObstacleFactory
import yaml
from typing import List, Dict, Any, Literal, Optional
from pydantic import BaseModel
from .ray_intersection import (
    batch_ray_intersection_detailed,
    create_lidar_rays,
    RayIntersectionOutput,
)
from collections import deque
from .live_renderer import SimulationWindow

DELTA_T = 1 / 60

COLLIDING_WITH_TYPES = Literal["obstacle", "boundary", "agent"]


class CollisionData(BaseModel):
    is_colliding: bool = False
    colliding_with: Optional[COLLIDING_WITH_TYPES] = None


class Agent:
    def __init__(self, agent_config: AgentConfig, goal_threshold: float = 0.02):
        self.config = agent_config
        self.pos = self.config.start_pos.center.to_numpy()
        self.start_pos = self.pos.copy()
        self.radius = self.config.radius
        self.current_speed = 0.1
        self.goal_pos = self.config.goal_pos.to_numpy()
        _, self.direction = convert_to_polar(self.goal_pos - self.pos)
        self.response_time = 2
        self.active = True
        self.lidar_observation_history = deque(maxlen=4)
        self.last_raw_lidar_observation = None
        self.goal_reached = False
        self.recent_speeds = deque(maxlen=20)
        self.old_pos = self.pos.copy()
        self.goal_threshold = goal_threshold

    def has_reached_goal(self):
        return np.linalg.norm(self.goal_pos - self.pos) < (
            self.goal_threshold + self.radius
        )

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
                goal_vector[0],
                goal_vector[1],
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
        self.old_pos = self.pos.copy()
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
        self.recent_speeds.append(self.current_speed)


class Environment(pettingzoo.ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "navigation_v0"}

    def __init__(
        self, config: dict[str, Any] | EnvConfig, render_mode: Optional[str] = None
    ):
        if isinstance(config, dict):
            config = EnvConfig(**config)

        self.config = config
        self.boundary = PolygonBoundary(config.boundary)
        self.agents_dict = {
            f"agent_{i}": Agent(agent_config, goal_threshold=self.config.goal_threshold)
            for i, agent_config in enumerate(config.agents)
        }
        self.obstacles = [
            ObstacleFactory.create(obstacle) for obstacle in config.obstacles
        ]
        self.num_steps = 0

        # PettingZoo required attributes
        self.possible_agents = list(self.agents_dict.keys())
        self.agents = self.possible_agents.copy()

        # Define observation and action spaces
        self._setup_spaces()

        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        if render_mode == "human" or render_mode == "rgb_array":
            # Use headless mode for rgb_array to avoid showing window
            headless = render_mode == "rgb_array"
            self.window = SimulationWindow(
                target_fps=30, record=False, headless=headless
            )

    def _setup_spaces(self):
        """Setup observation and action spaces for all agents."""
        # Action space: 2D velocity vector (vx, vy)
        max_speed = max(agent.config.max_speed for agent in self.agents_dict.values())
        self._action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space: state vector + lidar readings
        # State vector: [progress, cosine_angle, speed_ratio, speed_ratio * cosine_angle, distance_to_goal]
        state_dim = 5
        lidar_dim = (
            self.config.num_rays * 3
        )  # 3 channels per ray (obstacle, boundary, agent)

        obs_low = np.concatenate(
            [
                np.array([-1.0, -1.0, 0.0, -1.0, 0.0]),  # state vector bounds
                np.zeros(lidar_dim),  # lidar readings are non-negative
            ]
        )
        obs_high = np.concatenate(
            [
                np.array(
                    [1.0, 1.0, 1.0, 1.0, 1]
                ),  # state vector bounds, externally guaranteed distance max = 1
                np.full(lidar_dim, max_speed),  # max lidar range
            ]
        )

        self._observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    @property
    def agent_states_dim(self):
        return 7

    @property
    def lidar_dim(self):
        return self.config.num_rays * 3

    @property
    def observation_spaces(self):
        """Returns observation spaces for all agents."""
        return {agent: self._observation_space for agent in self.agents}

    @property
    def action_spaces(self):
        """Returns action spaces for all agents."""
        return {agent: self._action_space for agent in self.agents}

    def observation_space(self, agent):
        """Returns observation space for a specific agent."""
        return self._observation_space

    def action_space(self, agent):
        """Returns action space for a specific agent."""
        return self._action_space

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        # Reset all agents to initial positions
        for agent in self.agents_dict.values():

            if random.random() < 0.5:
                agent.pos = sample_point_in_rectangle(agent.config.start_pos)
            else:
                agent.pos = agent.config.start_pos.center.to_numpy()

            agent.current_speed = 0.1
            _, agent.direction = convert_to_polar(agent.goal_pos - agent.pos)
            agent.active = True
            agent.goal_reached = False
            agent.lidar_observation_history.clear()
            agent.last_raw_lidar_observation = None
            agent.recent_speeds.clear()
            agent.old_pos = agent.pos.copy()

        # Reset obstacles
        for obs in self.obstacles:
            if hasattr(obs, "reset"):
                obs.reset()

        self.num_steps = 0
        self.agents = self.possible_agents.copy()

        # Get initial observations
        observations = self._get_observations()
        infos = {agent: {} for agent in self.agents}

        # Render initial state if render mode is set
        if self.render_mode == "human":
            self.render()

        return observations, infos

    def _get_observations(self):
        """Get observations for all agents."""
        lidar_observations = self.get_lidar_observation()

        # Update agent lidar data
        for agent_id, lidar_observation in zip(self.agents, lidar_observations):
            self.agents_dict[agent_id].last_raw_lidar_observation = lidar_observation

        processed_lidar_observations = [
            self.process_lidar_observation(
                self.agents_dict[agent_id].config.max_range, lidar_observation
            )
            for agent_id, lidar_observation in zip(self.agents, lidar_observations)
        ]

        observations = {}
        for agent_id in self.agents:
            agent = self.agents_dict[agent_id]
            agent_idx = list(self.agents_dict.keys()).index(agent_id)

            state_dict = agent.get_state_dict()
            state_vector = np.array(state_dict["state_vector"], dtype=np.float32)
            lidar_vector = (
                processed_lidar_observations[agent_idx].flatten().astype(np.float32)
            )

            observations[agent_id] = np.concatenate([state_vector, lidar_vector])

        return observations

    def calculate_reward(self, agent: Agent, collision_data: CollisionData):
        if agent.goal_reached:
            return 10

        if collision_data.is_colliding:
            return -10

        goal_reward = agent.direction.dot(agent.goal_pos - agent.pos)
        scale_goal_reward_with_speed = goal_reward * (
            agent.current_speed / agent.config.max_speed
        )

        stay_alive_reward = -0.1
        scale_goal_reward_with_speed *= 0.5

        return scale_goal_reward_with_speed + stay_alive_reward

    def transition(self, actions: dict[str, np.ndarray] | np.ndarray):

        # Apply actions to agents
        for agent_id, action in zip(self.agents, actions):
            agent = self.agents_dict[agent_id]
            target_velocity = Vector2(
                x=action.x * agent.config.max_speed,
                y=action.y * agent.config.max_speed,
            )
            agent.apply_target_velocity(target_velocity)

        # Update obstacles
        for obs in self.obstacles:
            obs.update(DELTA_T)

        # Update agent positions and check collisions
        collision_datas = []
        for agent_id in self.agents:
            agent = self.agents_dict[agent_id]
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

            # Calculate terminations and truncations
            terminations = {}
            truncations = {}
            for agent_id in self.agents:
                agent = self.agents_dict[agent_id]
                terminations[agent_id] = (
                    (not agent.active)
                    or agent.goal_reached
                    or (
                        len(agent.recent_speeds) > 0
                        and np.mean(agent.recent_speeds) < 0.01
                    )
                )
                truncations[agent_id] = (
                    False if self.num_steps < self.config.max_time else True
                )

        return collision_datas, terminations, truncations

    def step(self, actions: dict[str, np.ndarray] | np.ndarray):
        """Execute one step of the environment."""
        # Handle both dictionary format (original) and vectorized format (after SuperSuit wrapping)
        if isinstance(actions, dict):
            # Original PettingZoo format: {"agent_0": [vx, vy]}
            processed_actions = []
            for agent_id in self.agents:
                if agent_id in actions:
                    action = actions[agent_id]
                    processed_actions.append(
                        Vector2(x=float(action[0]), y=float(action[1]))
                    )
                else:
                    processed_actions.append(Vector2(x=0.0, y=0.0))
        else:
            # Vectorized format: np.array([vx, vy]) for single agent or np.array([vx1, vy1, vx2, vy2, ...]) for multiple agents
            processed_actions = []
            if len(self.agents) == 1:
                # Single agent case
                if len(actions) >= 2:
                    processed_actions.append(
                        Vector2(x=float(actions[0]), y=float(actions[1]))
                    )
                else:
                    processed_actions.append(Vector2(x=0.0, y=0.0))
            else:
                # Multiple agents case - actions are flattened [vx1, vy1, vx2, vy2, ...]
                for i, agent_id in enumerate(self.agents):
                    if i * 2 + 1 < len(actions):
                        processed_actions.append(
                            Vector2(
                                x=float(actions[i * 2]), y=float(actions[i * 2 + 1])
                            )
                        )
                    else:
                        processed_actions.append(Vector2(x=0.0, y=0.0))

        self.num_steps += 1

        for i in range(self.config.repeat_steps):
            collision_datas, terminations, truncations = self.transition(
                processed_actions
            )

            if any(terminations.values()):
                break

            if any(truncations.values()):
                break

        # Calculate rewards
        rewards = {}
        for agent_id, collision_data in zip(self.agents, collision_datas):
            agent = self.agents_dict[agent_id]
            rewards[agent_id] = self.calculate_reward(agent, collision_data)
        # Get next observations
        observations = self._get_observations()

        # Remove terminated agents
        self.agents = [
            agent_id for agent_id in self.agents if not terminations[agent_id]
        ]

        infos = {agent_id: {} for agent_id in self.possible_agents}

        # Auto-render if render mode is set to human
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def state(self):
        """Returns the global state for centralized critics."""
        # Combine all agent states and environment information
        all_agent_states = []
        for agent in self.agents_dict.values():
            state_dict = agent.get_state_dict()
            agent_state = [
                agent.pos[0],
                agent.pos[1],  # position
                agent.direction[0],
                agent.direction[1],  # direction
                agent.current_speed,  # speed
                agent.goal_pos[0],
                agent.goal_pos[1],  # goal position
                float(agent.active),  # active status
                float(agent.goal_reached),  # goal reached status
            ]
            all_agent_states.extend(agent_state)

        # Add obstacle states
        obstacle_states = []
        for obs in self.obstacles:
            if hasattr(obs, "pos"):
                obstacle_states.extend([obs.pos[0], obs.pos[1]])

        # Add environment info
        env_states = [
            self.num_steps,
            len(self.agents),  # number of active agents
        ]

        return np.array(
            all_agent_states + obstacle_states + env_states, dtype=np.float32
        )

    def close(self):
        """Close the environment."""
        if self.window is not None:
            self.window.close()
            self.window = None

    def render(self):
        render_state = self.get_render_state()
        if self.render_mode == "human":
            self.window.render(render_state)
        elif self.render_mode == "rgb_array":
            self.window.render(render_state)
            return self.window.get_rgb_array()

        return render_state

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

        for agent_id in self.agents:
            agent = self.agents_dict[agent_id]
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

        if not all_rays:
            return []

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
        agent_states = []
        for agent_id in self.agents:
            agent = self.agents_dict[agent_id]
            agent_states.append(
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
            )

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

    # Test the environment
    observations, infos = env.reset()
    print(f"Initial observations: {list(observations.keys())}")
    print(f"Observation space: {env.observation_space('agent_0')}")
    print(f"Action space: {env.action_space('agent_0')}")

    # Take a random action
    actions = {}
    for agent_id in env.agents:
        actions[agent_id] = env.action_space(agent_id).sample()

    next_obs, rewards, terminations, truncations, infos = env.step(actions)
    print(f"Rewards: {rewards}")
    print(f"Terminations: {terminations}")
