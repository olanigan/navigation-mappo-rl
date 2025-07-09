from .config_models import *
from .obstacles import RectangleObstacle, CircleObstacle, Obstacle
import numpy as np


class Line(BaseModel):
    p1: Vector2
    p2: Vector2


class Ray(BaseModel):
    origin: Vector2
    direction: Vector2
    length: float


IntersectingWith = Literal["obstacle", "boundary", "agent"]


class RayIntersectionOutput(BaseModel):
    intersects: bool
    intersection: Optional[Vector2] = None
    t: Optional[float] = None
    intersecting_with: Optional[IntersectingWith] = None


NoHit = RayIntersectionOutput(intersects=False)


def ray_circle_intersection(ray: Ray, circle: Circle) -> RayIntersectionOutput:
    """
    Compute intersection between a ray and a circle using quadratic formula.
    Returns the closest intersection point along the ray (t >= 0 and t <= ray.length).
    """
    # Convert to numpy arrays for easier computation
    origin = ray.origin.to_numpy()
    direction = ray.direction.to_numpy()
    center = circle.center.to_numpy()
    radius = circle.radius

    # Vector from ray origin to circle center
    oc = origin - center

    # Quadratic equation coefficients: at² + bt + c = 0
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - radius * radius

    # Calculate discriminant
    discriminant = b * b - 4 * a * c

    # No intersection if discriminant is negative
    if discriminant < 0:
        return NoHit

    # Calculate the two possible intersection points
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2.0 * a)
    t2 = (-b + sqrt_discriminant) / (2.0 * a)

    # We want the closest intersection point that's in front of the ray (t >= 0) and within ray length (t <= ray.length)
    if t1 >= 0 and t1 <= ray.length:
        t = t1
    elif t2 >= 0 and t2 <= ray.length:
        t = t2
    else:
        # Both intersections are either behind the ray origin or beyond ray length
        return NoHit

    # Calculate intersection point
    intersection_point = origin + t * direction

    return RayIntersectionOutput(
        intersects=True,
        intersection=Vector2(x=intersection_point[0], y=intersection_point[1]),
        t=t,
    )


def ray_rectangle_intersection(ray: Ray, rectangle: Rectangle) -> RayIntersectionOutput:
    """
    Compute intersection between a ray and a rotated rectangle.
    Transform ray to rectangle's local coordinate system and check against edges.
    """
    # Convert to numpy arrays
    origin = ray.origin.to_numpy()
    direction = ray.direction.to_numpy()
    center = rectangle.center.to_numpy()

    # Transform ray to rectangle's local coordinate system
    # Translate to rectangle center
    local_origin = origin - center

    # Rotate by negative rotation to undo rectangle's rotation
    theta = -np.radians(rectangle.rotation)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    local_origin = rotation_matrix @ local_origin
    local_direction = rotation_matrix @ direction

    # Rectangle bounds in local coordinate system
    half_width = rectangle.width / 2
    half_height = rectangle.height / 2

    # Check intersection with all four edges of the rectangle
    closest_t = float("inf")
    closest_intersection = None

    # Define the four edges of the rectangle
    edges = [
        # Bottom edge
        Line(
            p1=Vector2(x=-half_width, y=-half_height),
            p2=Vector2(x=half_width, y=-half_height),
        ),
        # Right edge
        Line(
            p1=Vector2(x=half_width, y=-half_height),
            p2=Vector2(x=half_width, y=half_height),
        ),
        # Top edge
        Line(
            p1=Vector2(x=half_width, y=half_height),
            p2=Vector2(x=-half_width, y=half_height),
        ),
        # Left edge
        Line(
            p1=Vector2(x=-half_width, y=half_height),
            p2=Vector2(x=-half_width, y=-half_height),
        ),
    ]

    # Create ray in local coordinates (preserve the length)
    local_ray = Ray(
        origin=Vector2(x=local_origin[0], y=local_origin[1]),
        direction=Vector2(x=local_direction[0], y=local_direction[1]),
        length=ray.length,
    )

    # Check intersection with each edge
    for edge in edges:
        result = ray_line_intersection(local_ray, edge)
        if (
            result.intersects
            and result.t is not None
            and result.t < closest_t
            and result.t >= 0
            and result.t <= ray.length
        ):
            closest_t = result.t
            closest_intersection = result.intersection

    if closest_intersection is None:
        return NoHit

    # Transform intersection point back to world coordinates
    local_intersection = np.array([closest_intersection.x, closest_intersection.y])

    # Rotate back to world coordinates
    inverse_rotation_matrix = np.array(
        [[cos_theta, sin_theta], [-sin_theta, cos_theta]]
    )

    world_intersection = inverse_rotation_matrix @ local_intersection + center

    return RayIntersectionOutput(
        intersects=True,
        intersection=Vector2(x=world_intersection[0], y=world_intersection[1]),
        t=closest_t,
    )


def ray_line_intersection(ray: Ray, line: Line) -> RayIntersectionOutput:
    """
    Compute intersection between a ray and a line segment.
    Uses parametric line equations and solves for intersection.
    """
    # Convert to numpy arrays
    ray_origin = ray.origin.to_numpy()
    ray_direction = ray.direction.to_numpy()
    line_p1 = line.p1.to_numpy()
    line_p2 = line.p2.to_numpy()

    # Line segment vector
    line_direction = line_p2 - line_p1

    # Vector from ray origin to line start
    origin_to_line = line_p1 - ray_origin

    # Solve the system:
    # ray_origin + t * ray_direction = line_p1 + s * line_direction
    # This gives us: t * ray_direction - s * line_direction = origin_to_line

    # Create coefficient matrix
    # [ray_direction.x, -line_direction.x] [t] = [origin_to_line.x]
    # [ray_direction.y, -line_direction.y] [s]   [origin_to_line.y]

    denominator = ray_direction[0] * (-line_direction[1]) - ray_direction[1] * (
        -line_direction[0]
    )
    denominator = (
        ray_direction[0] * line_direction[1] - ray_direction[1] * line_direction[0]
    )

    # Lines are parallel
    if abs(denominator) < 1e-10:
        return NoHit

    # Solve for t and s using Cramer's rule
    t = (
        origin_to_line[0] * line_direction[1] - origin_to_line[1] * line_direction[0]
    ) / denominator
    s = (
        origin_to_line[0] * ray_direction[1] - origin_to_line[1] * ray_direction[0]
    ) / denominator

    # Check if intersection is valid:
    # t >= 0 (intersection is in front of ray)
    # t <= ray.length (intersection is within ray length)
    # 0 <= s <= 1 (intersection is within line segment)
    if t >= 0 and t <= ray.length and 0 <= s <= 1:
        # Calculate intersection point
        intersection_point = ray_origin + t * ray_direction

        return RayIntersectionOutput(
            intersects=True,
            intersection=Vector2(x=intersection_point[0], y=intersection_point[1]),
            t=t,
        )

    return NoHit


def ray_obstacle_intersection(ray: Ray, obstacle: Obstacle) -> RayIntersectionOutput:
    obstacle_type = obstacle.config.shape
    if isinstance(obstacle_type, Rectangle):
        return ray_rectangle_intersection(ray, obstacle_type)
    elif isinstance(obstacle_type, Circle):
        return ray_circle_intersection(ray, obstacle_type)
    else:
        raise ValueError(f"Unknown obstacle type: {obstacle_type}")


def ray_boundary_intersection(
    ray: Ray, boundary: PolygonBoundaryConfig
) -> RayIntersectionOutput:
    """
    Find the closest intersection between a ray and polygon boundary walls.
    """
    # Create a PolygonBoundary object to get the walls
    from .obstacles import PolygonBoundary

    polygon = PolygonBoundary(boundary)

    closest_t = float("inf")
    closest_intersection = None

    for wall in polygon.walls:
        p1, p2 = wall
        # Convert numpy arrays back to Vector2 for consistency
        line = Line(p1=Vector2(x=p1[0], y=p1[1]), p2=Vector2(x=p2[0], y=p2[1]))

        result = ray_line_intersection(ray, line)
        if (
            result.intersects
            and result.t is not None
            and result.t < closest_t
            and result.t <= ray.length
        ):
            closest_t = result.t
            closest_intersection = result.intersection

    if closest_intersection is None:
        return NoHit

    return RayIntersectionOutput(
        intersects=True, intersection=closest_intersection, t=closest_t
    )


def ray_intersection(
    ray: Ray,
    obstacles: List[Obstacle],
    boundaries: List[PolygonBoundaryConfig],
) -> RayIntersectionOutput:
    """
    Find the closest intersection between a ray and obstacles and boundaries.
    The effective maximum distance is the minimum of ray.length and max_t.
    """

    closest_t = float("inf")
    closest_intersection = None

    for obstacle in obstacles:
        result = ray_obstacle_intersection(ray, obstacle)
        if result.intersects and result.t is not None and result.t < closest_t:
            closest_t = result.t
            closest_intersection = result.intersection

    for boundary in boundaries:
        result = ray_boundary_intersection(ray, boundary)
        if result.intersects and result.t is not None and result.t < closest_t:
            closest_t = result.t
            closest_intersection = result.intersection

    if closest_intersection is None:
        return NoHit

    return RayIntersectionOutput(
        intersects=True, intersection=closest_intersection, t=closest_t
    )
