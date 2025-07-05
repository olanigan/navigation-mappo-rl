from pydantic import BaseModel
from typing import List, Tuple, Dict, Union
from .config_models import *


class AgentState(BaseModel):
    position: Tuple[float, float]
    radius: float
    color: str
    velocity: Tuple[float, float]


class ObstacleState(BaseModel):
    shape: str  # "rectangle", "circle"
    properties: Dict


class BoundaryState(BaseModel):
    vertices: List[Tuple[float, float]]


class RenderState(BaseModel):
    """
    A complete, serializable snapshot of the environment for a single frame.
    This object is renderer-agnostic.
    """

    agents: List[AgentState]
    obstacles: List[Union[Rectangle, Circle]]
    boundary: BoundaryState
