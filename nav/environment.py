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
    def __init__(
        self,
        agent_config: AgentConfig,
        goal_threshold: float = 0.02,
        use_global_information: bool = False,
    ):
        self.config = agent_config
        self.pos = self.config.start_pos.center.to_numpy()
        self.start_pos = self.pos.copy()
        self.radius = self.config.radius
        self.group_encoding = self.config.group_encoding
        self.current_speed = 0.1
        self.goal_sample_area = self.config.goal_pos
        self.goal_pos = self.config.goal_pos.center.to_numpy()
        _, self.direction = convert_to_polar(self.goal_pos - self.pos)
        self.response_time = 10
        self.active = True
        self.lidar_observation_history = deque(maxlen=4)
        self.last_raw_lidar_observation = None
        self.goal_reached = False
        self.recent_speeds = deque(maxlen=20)
        self.old_pos = self.pos.copy()
        self.goal_threshold = goal_threshold
        self.last_reward = 0
        self.use_global_information = use_global_information

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

        if self.use_global_information:
            global_information = [
                self.goal_pos[0] - self.pos[0],
                self.goal_pos[1] - self.pos[1],
                self.pos[0],
                self.pos[1],
                self.direction[0],
                self.direction[1],
                self.goal_pos[0],
                self.goal_pos[1],
            ]
        else:
            global_information = []
        return {
            "state_vector": [
                *self.group_encoding,
                *global_information,
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

        # Transform action from LCS (Y-axis = goal direction) to Global coordinates
        goal_vector = self.goal_pos - self.pos
        dist = np.linalg.norm(goal_vector)
        if dist > 1e-10:
            # Normalized direction to goal (New Y-axis)
            u_x = goal_vector[0] / dist
            u_y = goal_vector[1] / dist

            # Action components
            a_x = target_velocity.x
            a_y = target_velocity.y

            # Transform to global: v_global = a_x * Right + a_y * Forward
            # Right vector corresponds to (u_y, -u_x)
            global_vx = a_x * u_y + a_y * u_x
            global_vy = -a_x * u_x + a_y * u_y

            target_velocity_global = np.array([global_vx, global_vy])
        else:
            # If already at goal, keep existing behavior or zero out
            target_velocity_global = target_velocity.to_numpy()

        current_velocity = self.current_speed * self.direction
        force = target_velocity_global - current_velocity
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
    metadata = {"render_modes": ["human", "rgb_array", "none"], "name": "navigation_v0"}

    def __init__(
        self,
        config: dict[str, Any] | EnvConfig,
        render_mode: Optional[str] = None,
    ):
        if isinstance(config, dict):
            config = EnvConfig(**config)

        self.config = config
        self.state_image_size = config.state_image_size
        self.boundary = PolygonBoundary(config.boundary)
        self.terminal_strategy = config.terminal_strategy
        self.use_group_encoding = config.use_group_encoding
        self.num_groups = len(config.agents)  # rename config.agents to config.groups

        agent_configs = self.preprocess_agent_configs(
            config.agents, config.num_agents_per_group
        )

        self.agents_dict = {
            f"agent_{i}": Agent(
                agent_config,
                goal_threshold=self.config.goal_threshold,
                use_global_information=self.config.use_global_information,
            )
            for i, agent_config in enumerate(agent_configs)
        }

        self.state_dim = len(
            next(iter(self.agents_dict.values())).get_state_dict()["state_vector"]
        )
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
                target_fps=30, record=True, headless=headless
            )

    def preprocess_agent_configs(
        self, agent_configs: List[AgentConfig], num_agents_per_group: int
    ):
        """Preprocess agent configs to create multiple agents per group."""
        processed_configs = []
        num_groups = len(agent_configs)

        for group_idx, agent_config in enumerate(agent_configs):
            start_rect = agent_config.start_pos
            width = start_rect.width
            middle = start_rect.center.x
            spacing = width / num_agents_per_group

            for i in range(num_agents_per_group):
                new_agent_config = agent_config.model_copy()

                # Place agents in pattern: middle, middle-1*spacing, middle+1*spacing, middle-2*spacing, middle+2*spacing, etc.
                if i == 0:
                    x_offset = 0  # First agent at middle
                elif i % 2 == 1:  # Odd indices go left
                    x_offset = -((i + 1) // 2) * spacing
                else:  # Even indices > 0 go right
                    x_offset = (i // 2) * spacing

                this_center = Vector2(
                    x=middle + x_offset,
                    y=start_rect.center.y,
                )
                new_rect = Rectangle(
                    center=this_center,
                    width=(width / num_agents_per_group - new_agent_config.radius * 2),
                    height=start_rect.height,
                )
                new_agent_config.start_pos = new_rect

                if self.config.use_group_encoding:
                    ohe_encoding = [0 for _ in range(num_groups)]
                    ohe_encoding[group_idx] = 1
                    new_agent_config.group_encoding = ohe_encoding
                else:
                    new_agent_config.group_encoding = []
                processed_configs.append(new_agent_config)
        return processed_configs

    def _setup_spaces(self):
        """Setup observation and action spaces for all agents."""
        # Action space: 2D velocity vector (vx, vy)
        self._state_space = spaces.Box(
            low=0,
            high=1,
            shape=(11, self.state_image_size, self.state_image_size),
            dtype=np.float32,
        )
        max_speed = max(agent.config.max_speed for agent in self.agents_dict.values())
        self._action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space: state vector + lidar readings
        # State vector: [progress, cosine_angle, speed_ratio, speed_ratio * cosine_angle, distance_to_goal]
        state_dim = self.agent_states_dim
        lidar_dim = (
            self.config.num_rays * 3
        )  # 3 channels per ray (obstacle, boundary, agent)

        obs_low = np.concatenate(
            [
                np.array([-1] * state_dim),  # state vector bounds
                np.zeros(lidar_dim),  # lidar readings are non-negative
            ]
        )
        obs_high = np.concatenate(
            [
                np.array(
                    [1] * state_dim
                ),  # state vector bounds, externally guaranteed distance max = 1
                np.full(lidar_dim, max_speed),  # max lidar range
            ]
        )

        self._observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

    @property
    def agent_states_dim(self):
        return self.state_dim

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

    def state_space(self):
        """Returns state space for the environment."""
        return self._state_space

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        # Reset all agents to initial positions
        for agent in self.agents_dict.values():

            agent.pos = sample_point_in_rectangle(agent.config.start_pos)
            agent.goal_pos = sample_point_in_rectangle(agent.goal_sample_area)
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
        self.num_dead_agents = 0

        # Get initial observations
        observations = self._get_observations()

        state = self.state()
        infos = {
            agent: {
                "global_state": state,
            }
            for agent in self.agents
        }

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
        for agent_idx, agent_id in enumerate(self.agents):
            agent = self.agents_dict[agent_id]
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

        stay_alive_reward = -0.05
        scale_goal_reward_with_speed *= 0.25

        return scale_goal_reward_with_speed + stay_alive_reward

    def calculate_group_reward(self):
        all_goals_reached = all(
            [agent.goal_reached for agent in self.agents_dict.values()]
        )
        if all_goals_reached:
            return 10
        return 0

    def transition(self, actions: dict[str, np.ndarray] | np.ndarray):

        terminations = {}
        truncations = {}
        collision_datas = []

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
            for other_agent_id in self.agents:
                if other_agent_id == agent_id:
                    continue
                other_agent = self.agents_dict[other_agent_id]
                if np.linalg.norm(other_agent.pos - agent.pos) < (
                    agent.radius + other_agent.radius
                ):
                    agent.active = False
                    this_agent_collision_data.is_colliding = True
                    this_agent_collision_data.colliding_with = "agent"

            collision_datas.append(this_agent_collision_data)

        for agent_id in self.agents:
            agent = self.agents_dict[agent_id]

            # Individual termination conditions
            individual_termination = (
                (not agent.active)
                or agent.goal_reached
                or (
                    len(agent.recent_speeds) > 0 and np.mean(agent.recent_speeds) < 0.01
                )
            )

            # Apply terminal strategy
            if self.terminal_strategy == "group":
                # In group mode, if any agent terminates, all terminate
                terminations[agent_id] = individual_termination or (
                    self.num_dead_agents > 0
                )
            else:
                # In individual mode, each agent terminates independently
                terminations[agent_id] = individual_termination

            terminations[agent_id] = np.bool_(terminations[agent_id])
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
            if collision_data.is_colliding:
                self.num_dead_agents += 1

        for agent_id in self.agents_dict.keys():
            agent = self.agents_dict[agent_id]
            if agent_id in rewards:
                agent.last_reward = rewards[agent_id]
        # Get next observations
        observations = self._get_observations()
        state = self.state()
        infos = {
            agent: {
                "global_state": state,
            }
            for agent in self.agents_dict.keys()
        }

        # Remove terminated agents
        self.agents = [
            agent_id for agent_id in self.agents if not terminations[agent_id]
        ]
        # Auto-render if render mode is set to human
        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def state(self):
        """Returns the global state for centralized critics."""

        # Create multi-channel occupancy grid
        # Channel 0: Obstacles
        # Channel 1: Agent positions
        # Channel 2: Goal X difference (goal_x - agent_x)
        # Channel 3: Goal Y difference (goal_y - agent_y)
        # Channel 4: Agent direction X
        # Channel 5: Agent direction Y
        # Channel 6: Agent speed (magnitude)
        # Channel 7: Goal reached status
        # Channel 8: Agent collision status
        # Channel 9: Agent count per cell
        # Channel 10: Boundaries

        state_grid = self._create_multi_channel_grid(self.state_image_size)
        # Flatten and return
        return state_grid.astype(np.float32)

    def _get_structured_state(self):
        """Original structured state approach (fallback)"""
        # Combine all agent states and environment information
        all_agent_states = []
        for agent in self.agents_dict.values():
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

        # Add basic obstacle states (limited)
        obstacle_states = []
        for obs in self.obstacles:
            if hasattr(obs, "center"):
                obstacle_states.extend([obs.center[0], obs.center[1]])

        # Add environment info
        env_states = [
            self.num_steps,
            len(self.agents),  # number of active agents
        ]

        return np.array(
            all_agent_states + obstacle_states + env_states, dtype=np.float32
        )

    def _create_multi_channel_grid(self, grid_size):
        """Create a multi-channel occupancy grid with different features"""
        # Get environment bounds
        boundary_vertices = np.array(self.boundary.vertices)
        min_x, min_y = boundary_vertices.min(axis=0)
        max_x, max_y = boundary_vertices.max(axis=0)

        # Add small padding to avoid edge issues
        padding = 0.1
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding

        # Create coordinate grids
        x_coords = np.linspace(min_x, max_x, grid_size)
        y_coords = np.linspace(min_y, max_y, grid_size)

        # Initialize 11-channel grid
        num_channels = 11
        grid = np.zeros((num_channels, grid_size, grid_size), dtype=np.float32)

        # Channel 0: Obstacles
        grid[0, :, :] = self._fill_obstacle_channel(x_coords, y_coords)

        # Channel 1: Agent positions
        grid[1, :, :] = self._fill_agent_positions_channel(x_coords, y_coords)

        # Channel 2: Goal X difference
        grid[2, :, :] = self._fill_goal_x_difference_channel(x_coords, y_coords)

        # Channel 3: Goal Y difference
        grid[3, :, :] = self._fill_goal_y_difference_channel(x_coords, y_coords)

        # Channel 4: Agent direction X
        grid[4, :, :] = self._fill_agent_direction_x_channel(x_coords, y_coords)

        # Channel 5: Agent direction Y
        grid[5, :, :] = self._fill_agent_direction_y_channel(x_coords, y_coords)

        # Channel 6: Agent speed
        grid[6, :, :] = self._fill_agent_speed_channel(x_coords, y_coords)

        # Channel 7: Goal reached status
        grid[7, :, :] = self._fill_goal_reached_channel(x_coords, y_coords)

        # Channel 8: Agent collision status
        grid[8, :, :] = self._fill_collision_status_channel(x_coords, y_coords)

        # Channel 9: Agent count per cell
        grid[9, :, :] = self._fill_agent_count_channel(x_coords, y_coords)

        # Channel 10: Boundaries
        grid[10, :, :] = self._fill_boundary_channel(x_coords, y_coords)

        return grid

    def _fill_obstacle_channel(self, x_coords, y_coords):
        """Fill channel with obstacle information"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                point = np.array([x, y])

                # Check if point is inside any obstacle
                for obs in self.obstacles:
                    if obs.check_collision(point, radius=0.01):
                        channel[i, j] = 1.0
                        break

        return channel

    def _fill_agent_positions_channel(self, x_coords, y_coords):
        """Fill channel with agent presence at exact grid positions"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for agent in self.agents_dict.values():
            if not agent.active:
                continue

            # Find the grid cell where the agent is located
            agent_x, agent_y = agent.pos

            # Find closest grid indices to agent position
            x_idx = np.argmin(np.abs(x_coords - agent_x))
            y_idx = np.argmin(np.abs(y_coords - agent_y))

            # Set presence to 1.0 at agent location
            channel[y_idx, x_idx] = 1.0

        return channel

    def _fill_goal_x_difference_channel(self, x_coords, y_coords):
        """Fill channel with signed X difference (goal_x - agent_x) at agent's position"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for agent in self.agents_dict.values():
            if not agent.active or agent.goal_reached:
                continue

            # Find the grid cell where the agent is located
            agent_x, agent_y = agent.pos

            # Find closest grid indices to agent position
            x_idx = np.argmin(np.abs(x_coords - agent_x))
            y_idx = np.argmin(np.abs(y_coords - agent_y))

            # Calculate signed X difference (goal_x - agent_x)
            goal_x, goal_y = agent.goal_pos
            x_diff = goal_x - agent_x

            # Normalize by environment width for consistent scaling
            env_width = x_coords.max() - x_coords.min()
            normalized_x_diff = x_diff / env_width

            # Store the normalized difference at the agent's grid position
            # Range: [-1, 1] where -1 = goal is far left, +1 = goal is far right
            channel[y_idx, x_idx] = np.clip(normalized_x_diff, -1.0, 1.0)

        return channel

    def _fill_goal_y_difference_channel(self, x_coords, y_coords):
        """Fill channel with signed Y difference (goal_y - agent_y) at agent's position"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for agent in self.agents_dict.values():
            if not agent.active or agent.goal_reached:
                continue

            # Find the grid cell where the agent is located
            agent_x, agent_y = agent.pos

            # Find closest grid indices to agent position
            x_idx = np.argmin(np.abs(x_coords - agent_x))
            y_idx = np.argmin(np.abs(y_coords - agent_y))

            # Calculate signed Y difference (goal_y - agent_y)
            goal_x, goal_y = agent.goal_pos
            y_diff = goal_y - agent_y

            # Normalize by environment height for consistent scaling
            env_height = y_coords.max() - y_coords.min()
            normalized_y_diff = y_diff / env_height

            # Store the normalized difference at the agent's grid position
            # Range: [-1, 1] where -1 = goal is far down, +1 = goal is far up
            channel[y_idx, x_idx] = np.clip(normalized_y_diff, -1.0, 1.0)

        return channel

    def _fill_agent_direction_x_channel(self, x_coords, y_coords):
        """Fill channel with signed X direction at agent's position"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for agent in self.agents_dict.values():
            if not agent.active:
                continue

            # Find the grid cell where the agent is located
            agent_x, agent_y = agent.pos

            # Find closest grid indices to agent position
            x_idx = np.argmin(np.abs(x_coords - agent_x))
            y_idx = np.argmin(np.abs(y_coords - agent_y))

            # Store the X component of direction vector
            # agent.direction is already a unit vector, so values are in [-1, 1]
            channel[y_idx, x_idx] = agent.direction[0]

        return channel

    def _fill_agent_direction_y_channel(self, x_coords, y_coords):
        """Fill channel with signed Y direction at agent's position"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for agent in self.agents_dict.values():
            if not agent.active:
                continue

            # Find the grid cell where the agent is located
            agent_x, agent_y = agent.pos

            # Find closest grid indices to agent position
            x_idx = np.argmin(np.abs(x_coords - agent_x))
            y_idx = np.argmin(np.abs(y_coords - agent_y))

            # Store the Y component of direction vector
            # agent.direction is already a unit vector, so values are in [-1, 1]
            channel[y_idx, x_idx] = agent.direction[1]

        return channel

    def _fill_agent_speed_channel(self, x_coords, y_coords):
        """Fill channel with normalized speed at agent's position"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for agent in self.agents_dict.values():
            if not agent.active:
                continue

            # Find the grid cell where the agent is located
            agent_x, agent_y = agent.pos

            # Find closest grid indices to agent position
            x_idx = np.argmin(np.abs(x_coords - agent_x))
            y_idx = np.argmin(np.abs(y_coords - agent_y))

            # Store the normalized speed
            normalized_speed = agent.current_speed / agent.config.max_speed
            channel[y_idx, x_idx] = np.clip(normalized_speed, 0.0, 1.0)

        return channel

    def _fill_goal_reached_channel(self, x_coords, y_coords):
        """Fill channel with goal reached status at agent's position"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for agent in self.agents_dict.values():
            if not agent.active:
                continue

            # Find the grid cell where the agent is located
            agent_x, agent_y = agent.pos

            # Find closest grid indices to agent position
            x_idx = np.argmin(np.abs(x_coords - agent_x))
            y_idx = np.argmin(np.abs(y_coords - agent_y))

            # Store goal reached status (reuse existing agent.goal_reached)
            channel[y_idx, x_idx] = 1.0 if agent.goal_reached else 0.0

        return channel

    def _fill_collision_status_channel(self, x_coords, y_coords):
        """Fill channel with collision status at agent's position"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for agent in self.agents_dict.values():
            if not agent.active:
                continue

            # Find the grid cell where the agent is located
            agent_x, agent_y = agent.pos

            # Find closest grid indices to agent position
            x_idx = np.argmin(np.abs(x_coords - agent_x))
            y_idx = np.argmin(np.abs(y_coords - agent_y))

            # Check if agent is colliding (reuse existing collision detection logic)
            is_colliding = self._check_agent_collision(agent)
            channel[y_idx, x_idx] = 1.0 if is_colliding else 0.0

        return channel

    def _fill_agent_count_channel(self, x_coords, y_coords):
        """Fill channel with count of agents per grid cell"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for agent in self.agents_dict.values():
            if not agent.active:
                continue

            # Find the grid cell where the agent is located
            agent_x, agent_y = agent.pos

            # Find closest grid indices to agent position
            x_idx = np.argmin(np.abs(x_coords - agent_x))
            y_idx = np.argmin(np.abs(y_coords - agent_y))

            # Increment count for this cell
            channel[y_idx, x_idx] += 1.0

        # Normalize by maximum reasonable agent count (e.g., 10 agents per cell)
        # This keeps values in [0, 1] range for consistency
        max_agents_per_cell = 10.0
        channel = np.clip(channel / max_agents_per_cell, 0.0, 1.0)

        return channel

    def _check_agent_collision(self, agent):
        """Check if an agent is currently colliding with anything"""
        # Check boundary collision
        if self.boundary.violating_boundary(agent):
            return True

        # Check obstacle collision
        for obs in self.obstacles:
            if obs.check_collision(center=agent.pos, radius=agent.radius):
                return True

        # Check collision with other agents
        for other_agent_id, other_agent in self.agents_dict.items():
            if other_agent == agent or not other_agent.active:
                continue

            distance = np.linalg.norm(other_agent.pos - agent.pos)
            if distance < (agent.radius + other_agent.radius):
                return True

        return False

    def _fill_boundary_channel(self, x_coords, y_coords):
        """Fill channel with boundary information"""
        channel = np.zeros((len(y_coords), len(x_coords)), dtype=np.float32)

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                point = np.array([x, y])

                # Check distance to boundary walls
                min_dist = float("inf")
                for wall in self.boundary.walls:
                    p1, p2 = wall

                    # Distance from point to line segment
                    line_vec = p2 - p1
                    point_vec = point - p1
                    line_len_sq = np.sum(line_vec**2)

                    if line_len_sq == 0:
                        dist = np.linalg.norm(point - p1)
                    else:
                        t = np.clip(np.dot(point_vec, line_vec) / line_len_sq, 0, 1)
                        closest_point = p1 + t * line_vec
                        dist = np.linalg.norm(point - closest_point)

                    min_dist = min(min_dist, dist)

                # Convert distance to intensity (closer = higher intensity)
                max_boundary_dist = (
                    0.2  # Distance at which boundary effect becomes negligible
                )
                if min_dist < max_boundary_dist:
                    channel[i, j] = 1.0 - (min_dist / max_boundary_dist)

        return channel

    def close(self):
        """Close the environment."""
        if self.window is not None:
            self.window.on_close()
            self.window.close()
            self.window = None

    def render(self):
        render_state = self.get_render_state()
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            self.window.render(render_state)
            return self.window.get_rgb_array()
        else:
            return render_state

    def process_lidar_observation(
        self, max_range, lidar_observation: list[RayIntersectionOutput]
    ):
        rays = np.zeros((3, len(lidar_observation)))
        for i in range(len(lidar_observation)):
            ray_data = lidar_observation[i]
            if not ray_data.intersects:
                continue

            if ray_data.intersecting_with == "obstacle":
                rays[0, i] = max_range - ray_data.t
            elif ray_data.intersecting_with == "boundary":
                rays[1, i] = max_range - ray_data.t
            elif ray_data.intersecting_with == "agent":
                rays[2, i] = max_range - ray_data.t

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
            agents=[
                Circle(
                    center=Vector2(
                        x=self.agents_dict[agent].pos[0],
                        y=self.agents_dict[agent].pos[1],
                    ),
                    radius=self.agents_dict[agent].radius,
                )
                for agent in self.agents
            ],
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
                    last_reward=agent.last_reward,
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
