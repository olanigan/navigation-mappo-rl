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


class AgentConfig(BaseModel):
    start_pos: Vector2
    goal_pos: Vector2
    radius: float = 0.02
    max_speed: float
    preferred_speed: float
    spawn_time: float
    agent_col: Literal["blue", "green", "yellow", "red"] = "blue"


class Rectangle(BaseModel):
    type: Literal["rectangle"] = "rectangle"
    center: Vector2
    width: float
    height: float
    rotation: float = 0  # Rotation in degrees


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


class PolygonBoundaryConfig(BaseModel):
    type: Literal["polygon"] = "polygon"
    vertices: List[Vector2]


class EnvConfig(BaseModel):
    boundary: PolygonBoundaryConfig
    obstacles: List[ObstacleConfig]
    agents: List[AgentConfig]
    max_time: int


if __name__ == "__main__":
    vector2 = Vector2(x=0.4, y=0.6)
    print(vector2.x)
    print(vector2.model_dump())

    agent = AgentConfig(
        start_pos=Vector2(x=0, y=0),
        goal_pos=Vector2(x=1, y=0),
        max_speed=0.5,
        preferred_speed=0.2,
        spawn_time=0,
    )
