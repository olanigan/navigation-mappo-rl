import numpy as np
import random
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator

PI = np.pi


class Vector2(BaseModel):
    x: float
    y: float

    def to_numpy(self):
        return np.array([self.x, self.y])


class Line(BaseModel):
    start: Vector2
    end: Vector2


class Rectangle(BaseModel):
    type: Literal["rectangle"] = "rectangle"
    center: Vector2
    width: float
    height: float
    rotation: float = 0  # Rotation in degrees


class AgentConfig(BaseModel):
    start_pos: Rectangle
    goal_pos: Rectangle
    radius: float = 0.02
    max_speed: float
    spawn_time: float
    agent_col: Literal[
        "blue",
        "green",
        "yellow",
        "red",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "white",
        "black",
    ] = "blue"
    max_range: float = 0.25
    fov_degrees: float = 210.0
    group_encoding: List[float] = []


class Circle(BaseModel):
    type: Literal["circle"] = "circle"
    center: Vector2
    radius: float


class ObstacleSchedule(BaseModel):
    speed: Optional[float] = None
    direction: Optional[Vector2] = None
    spawn_time: Optional[float] = None
    angular_speed: Optional[float] = None  # Angular speed in degrees per second
    rotating_up: Optional[bool] = None
    boundary_x_min: Optional[float] = None
    boundary_x_max: Optional[float] = None


class ObstacleConfig(BaseModel):
    shape: Union[Rectangle, Circle]
    schedule: Optional[ObstacleSchedule] = None
    noise: Optional[float] = None


class PolygonBoundaryConfig(BaseModel):
    type: Literal["polygon"] = "polygon"
    vertices: List[Vector2]


class EnvConfig(BaseModel):
    boundary: PolygonBoundaryConfig
    obstacles: List[ObstacleConfig] = []
    agents: List[AgentConfig]
    max_time: int
    num_rays: int = 60
    goal_threshold: float = 0.02
    repeat_steps: int = 2
    num_agents_per_group: int = 1
    terminal_strategy: Literal["individual", "group"] = "individual"
    use_group_encoding: bool = False
    state_image_size: int = 32
    use_global_information: bool = False


if __name__ == "__main__":
    vector2 = Vector2(x=0.4, y=0.6)
    print(vector2.x)
    print(vector2.model_dump())

    agent = AgentConfig(
        start_pos=Rectangle(center=Vector2(x=0, y=0), width=0.2, height=0.02),
        goal_pos=Vector2(x=1, y=0),
        max_speed=0.5,
        spawn_time=0,
    )
