from typing import Tuple
import numpy as np


def convert_to_polar(a: np.ndarray) -> Tuple[float, np.ndarray]:
    magnitude = float(np.linalg.norm(a))
    unit_vector = a / magnitude
    return magnitude, unit_vector
