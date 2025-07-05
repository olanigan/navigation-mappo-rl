import random
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator

class Vector2(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)

class AgentConfig(BaseModel):
    start_pos: Vector2
    goal_pos: Vector2
    max_speed: float
    preferred_speed: float
    spawn_time: float
    agent_col: Optional[Literal["blue", "green", "yellow", "red"]] = None

    @model_validator(mode='after')
    def set_default_agent_col(self) -> 'AgentConfig':
        if self.agent_col is None:
            self.agent_col = random.choice(["blue", "green", "yellow", "red"])
        return self

class RectangleObstacle(BaseModel):
    center: Vector2
    width: float
    height: float

class CircleObstacle(BaseModel):
    center: Vector2
    radius: float

class LineObstacle(BaseModel):
    start: Vector2
    end: Vector2

class MovingObstacle(BaseModel):
    obstacle: Union[RectangleObstacle, LineObstacle]
    speed: float
    direction: Vector2
    spawn_time: float
    angular_speed: float
    rotating_up: bool
    oscillation_time: float

class EnvConfig(BaseModel):
    obstacles: List[Union[LineObstacle, CircleObstacle, RectangleObstacle, MovingObstacle]]
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

    print(agent)

