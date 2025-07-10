import arcade
import time
from typing import List
from .renderer_models import RenderState, AgentState

# --- Theme Colors ---
BACKGROUND_COLOR = arcade.color.BLACK
BOUNDARY_COLOR = arcade.color.WHITE
ARENA_COLOR = arcade.color.DIM_GRAY
OBSTACLE_COLOR = arcade.color.LIGHT_SALMON_PINK
OBSTACLE_BORDER_COLOR = arcade.color.BROWN
AGENT_COLOR = arcade.color.AERO_BLUE
AGENT_BORDER_COLOR = arcade.color.BLUE
AGENT_ARROW_COLOR = arcade.color.DARK_BLUE
AGENT_GOAL_COLOR = arcade.color.GREEN

# --- Ray Colors ---
RAY_OBSTACLE_HIT_COLOR = arcade.color.CRIMSON
RAY_BOUNDARY_HIT_COLOR = arcade.color.DARK_ORANGE
RAY_GOAL_HIT_COLOR = arcade.color.LIME_GREEN
RAY_MISS_COLOR = arcade.color.LIGHT_STEEL_BLUE
RAY_INTERSECTION_MARKER_COLOR = arcade.color.WHITE


class SimulationWindow(arcade.Window):
    """
    Renders a simulation state using Arcade.
    """

    def __init__(
        self, window_width: int = 800, window_height: int = 800, target_fps: int = 30
    ):
        super().__init__(window_width, window_height, "Navigation Simulation")
        arcade.set_background_color(BACKGROUND_COLOR)
        self.current_state = None
        self.cursor_pos = (0.0, 0.0)

        # Set target FPS
        self.target_fps = target_fps
        # self.set_vsync(False)  # Disable VSync to allow custom FPS control
        self.set_update_rate(1 / target_fps)

        # FPS tracking
        self.frame_count = 0

    def on_mouse_motion(self, x, y, dx, dy):
        self.cursor_pos = (x / self.width, y / self.height)

    def render(self, state: RenderState):
        """
        Receives a new state and schedules a redraw.
        """
        self.current_state = state
        self.on_draw()

    def on_draw(self):
        """
        The main rendering loop.
        """
        self.clear()

        # Safety check - don't draw if no state is available
        if not self.current_state:
            return

        # --- Draw Boundary ---
        scaled_vertices = [
            (v[0] * self.width, v[1] * self.height)
            for v in self.current_state.boundary.vertices
        ]
        arcade.draw_polygon_outline(scaled_vertices, BOUNDARY_COLOR, line_width=10)
        arcade.draw_polygon_filled(scaled_vertices, ARENA_COLOR)

        # --- Draw Obstacles ---
        for obstacle in self.current_state.obstacles:
            if obstacle.type == "rectangle":
                x = obstacle.center.x * self.width
                y = obstacle.center.y * self.height
                width = obstacle.width * self.width
                height = obstacle.height * self.height
                rotation = (
                    -obstacle.rotation
                )  # Rotation in degrees (arcade expects degrees)
                rect = arcade.XYWH(x, y, width, height)

                arcade.draw_rect_filled(rect, OBSTACLE_COLOR, rotation)
                arcade.draw_rect_outline(rect, OBSTACLE_BORDER_COLOR, 4, rotation)

        # --- Draw Rays (behind agents) ---
        for agent in self.current_state.agents:
            self._draw_agent_rays(agent)

        # --- Draw Agents (on top of rays) ---
        for agent in self.current_state.agents:
            x = agent.position[0] * self.width
            y = agent.position[1] * self.height
            vX = agent.velocity[0] / 5 * self.width
            vY = agent.velocity[1] / 5 * self.height
            radius = agent.radius * self.width
            color = getattr(arcade.color, agent.color.upper())

            # Draw goal rectangle first (behind agent)
            goal_x = agent.goal_rectangle.center.x * self.width
            goal_y = agent.goal_rectangle.center.y * self.height
            goal_width = agent.goal_rectangle.width * self.width
            goal_height = agent.goal_rectangle.height * self.height
            goal_rotation = (
                -agent.goal_rectangle.rotation
            )  # Arcade expects negative rotation
            goal_rect = arcade.XYWH(goal_x, goal_y, goal_width, goal_height)

            # Ensure proper color formatting
            goal_fill_color = (*AGENT_GOAL_COLOR[:3], 100)  # RGB + alpha
            goal_outline_color = AGENT_GOAL_COLOR[:3]  # RGB only for outline

            arcade.draw_rect_filled(
                goal_rect, goal_fill_color, goal_rotation
            )  # Semi-transparent fill
            arcade.draw_rect_outline(
                goal_rect, goal_outline_color, 3, goal_rotation
            )  # Solid outline

            # Draw agent circle
            arcade.draw_circle_filled(x, y, radius, AGENT_COLOR)
            arcade.draw_circle_outline(x, y, radius, AGENT_BORDER_COLOR, 1)

            # Draw velocity arrow on top
            arcade.draw_line(x, y, x + vX, y + vY, AGENT_ARROW_COLOR, 4)

    def _draw_agent_rays(self, agent: AgentState):
        """
        Draw rays for a single agent with different colors based on what they hit.
        """
        import math

        total_rays = len(agent.lidar_observation)
        if total_rays == 0:
            return

        fov_degrees = agent.fov_degrees
        max_range = agent.max_range

        # Get agent's current facing direction from explicit direction field
        agent_facing_angle = math.atan2(agent.direction[1], agent.direction[0])

        # Calculate ray directions based on FOV relative to agent's facing direction
        start_angle = agent_facing_angle - math.radians(fov_degrees / 2)
        angle_increment = (
            math.radians(fov_degrees) / (total_rays - 1) if total_rays > 1 else 0
        )

        for i, ray_result in enumerate(agent.lidar_observation):
            # Calculate ray angle and direction relative to agent's facing direction
            ray_angle_rad = start_angle + i * angle_increment

            # Ray direction accounting for agent's facing direction
            ray_dir_x = math.cos(ray_angle_rad)
            ray_dir_y = math.sin(ray_angle_rad)

            # Scale ray origin to screen coordinates
            start_x = agent.position[0] * self.width
            start_y = agent.position[1] * self.height

            # Determine ray color based on what it hit
            if ray_result.intersecting_with == "obstacle":
                ray_color = RAY_OBSTACLE_HIT_COLOR
                line_width = 2
                alpha = 200
            elif ray_result.intersecting_with == "boundary":
                ray_color = RAY_BOUNDARY_HIT_COLOR
                line_width = 2
                alpha = 180
            elif ray_result.intersecting_with == "goal":
                ray_color = RAY_GOAL_HIT_COLOR
                line_width = 3
                alpha = 255
            else:
                ray_color = RAY_MISS_COLOR
                line_width = 1
                alpha = 100

            # Calculate end point
            if ray_result.intersection:
                # Ray hit something - draw to intersection point
                end_x = ray_result.intersection.x * self.width
                end_y = ray_result.intersection.y * self.height
            else:
                # Ray missed - draw full length
                end_x = start_x + ray_dir_x * max_range * self.width
                end_y = start_y + ray_dir_y * max_range * self.height

            # Draw the ray line
            if ray_result.intersecting_with is None:
                # Draw dashed line for misses
                self._draw_dashed_line(
                    start_x, start_y, end_x, end_y, ray_color, line_width, alpha
                )
            else:
                # Draw solid line for hits
                color_with_alpha = (*ray_color[:3], alpha)  # Ensure RGB + alpha
                arcade.draw_line(
                    start_x, start_y, end_x, end_y, color_with_alpha, line_width
                )

            # # Draw intersection marker
            if ray_result.intersection:
                marker_x = ray_result.intersection.x * self.width
                marker_y = ray_result.intersection.y * self.height

                marker_size = 2
                arcade.draw_circle_filled(
                    marker_x,
                    marker_y,
                    marker_size,
                    RAY_INTERSECTION_MARKER_COLOR,
                    1,
                )

    def _draw_dashed_line(
        self, start_x, start_y, end_x, end_y, color, line_width, alpha
    ):
        """
        Draw a dashed line by drawing small segments.
        """
        import math

        # Calculate line length and direction
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.sqrt(dx * dx + dy * dy)

        if length < 1:
            return

        # Normalize direction
        dx /= length
        dy /= length

        # Draw dashed segments
        dash_length = 8
        gap_length = 4
        current_pos = 0

        while current_pos < length:
            # Start of dash
            dash_start_x = start_x + dx * current_pos
            dash_start_y = start_y + dy * current_pos

            # End of dash
            dash_end_pos = min(current_pos + dash_length, length)
            dash_end_x = start_x + dx * dash_end_pos
            dash_end_y = start_y + dy * dash_end_pos
            # Draw dash segment
            color_with_alpha = (*color[:3], alpha)  # Ensure RGB + alpha
            arcade.draw_line(
                dash_start_x,
                dash_start_y,
                dash_end_x,
                dash_end_y,
                color_with_alpha,
                line_width,
            )

            # Move to next dash
            current_pos += dash_length + gap_length
