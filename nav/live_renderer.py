import arcade
from typing import List
from .renderer_models import RenderState

# --- Theme Colors ---
BACKGROUND_COLOR = arcade.color.BLACK
BOUNDARY_COLOR = arcade.color.WHITE
ARENA_COLOR = arcade.color.DIM_GRAY
OBSTACLE_COLOR = arcade.color.LIGHT_SALMON_PINK
OBSTACLE_BORDER_COLOR = arcade.color.BROWN
AGENT_COLOR = arcade.color.AERO_BLUE
AGENT_BORDER_COLOR = arcade.color.BLUE
AGENT_ARROW_COLOR = arcade.color.DARK_BLUE


class SimulationWindow(arcade.Window):
    """
    Renders a simulation state using Arcade.
    """

    def __init__(self, window_width: int = 800, window_height: int = 800):
        super().__init__(window_width, window_height, "Navigation Simulation")
        arcade.set_background_color(BACKGROUND_COLOR)
        self.current_state = None
        self.cursor_pos = (0.0, 0.0)

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
        if not self.current_state:
            return

        # --- Draw Boundary ---
        scaled_vertices = [
            (v[0] * self.width, v[1] * self.height)
            for v in self.current_state.boundary.vertices
        ]
        arcade.draw_polygon_outline(scaled_vertices, BOUNDARY_COLOR, line_width=10)
        arcade.draw_polygon_filled(scaled_vertices, ARENA_COLOR)

        # --- Draw Agents ---
        for agent in self.current_state.agents:
            x = agent.position[0] * self.width
            y = agent.position[1] * self.height
            vX = agent.velocity[0] / 5 * self.width
            vY = agent.velocity[1] / 5 * self.height
            radius = agent.radius * self.width
            color = getattr(arcade.color, agent.color.upper())
            arcade.draw_circle_filled(x, y, radius, AGENT_COLOR)
            arcade.draw_circle_outline(x, y, radius, AGENT_BORDER_COLOR, 1)
            arcade.draw_line(x, y, x + vX, y + vY, AGENT_ARROW_COLOR, 4)
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
