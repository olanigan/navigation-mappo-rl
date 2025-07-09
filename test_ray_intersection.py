import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from nav.ray_intersection import *
from nav.config_models import *
from nav.obstacles import PolygonBoundary, CircleObstacle, RectangleObstacle


def plot_ray(ax, ray: Ray, color="blue", linewidth=2):
    """Plot a ray as an arrow using the ray's actual length"""
    origin = ray.origin.to_numpy()
    direction = ray.direction.to_numpy()

    # Normalize direction and scale by ray's length
    direction_norm = direction / np.linalg.norm(direction)
    end_point = origin + direction_norm * ray.length

    ax.arrow(
        origin[0],
        origin[1],
        direction_norm[0] * ray.length,
        direction_norm[1] * ray.length,
        head_width=0.03,
        head_length=0.05,
        fc=color,
        ec=color,
        linewidth=linewidth,
    )


def plot_circle(ax, circle: Circle, color="red", alpha=0.3):
    """Plot a circle"""
    circle_patch = patches.Circle(
        (circle.center.x, circle.center.y),
        circle.radius,
        facecolor=color,
        edgecolor="darkred",
        alpha=alpha,
        linewidth=2,
    )
    ax.add_patch(circle_patch)


def plot_rectangle(ax, rectangle: Rectangle, color="green", alpha=0.3):
    """Plot a rectangle (with rotation support)"""
    # Create rectangle patch
    rect = patches.Rectangle(
        (-rectangle.width / 2, -rectangle.height / 2),
        rectangle.width,
        rectangle.height,
        facecolor=color,
        edgecolor="darkgreen",
        alpha=alpha,
        linewidth=2,
    )

    # Apply rotation and translation
    transform = (
        plt.matplotlib.transforms.Affine2D()
        .rotate_deg(rectangle.rotation)
        .translate(rectangle.center.x, rectangle.center.y)
        + ax.transData
    )
    rect.set_transform(transform)

    ax.add_patch(rect)


def plot_line(ax, line: Line, color="purple", linewidth=3):
    """Plot a line segment"""
    ax.plot(
        [line.p1.x, line.p2.x],
        [line.p1.y, line.p2.y],
        color=color,
        linewidth=linewidth,
        label="Line segment",
    )


def plot_intersection_point(ax, intersection: Vector2, color="red", size=100):
    """Plot an intersection point"""
    ax.scatter(
        intersection.x,
        intersection.y,
        color=color,
        s=size,
        marker="x",
        linewidth=4,
        label="Intersection point",
    )


def setup_plot(title: str, xlim=(-1, 2), ylim=(-1, 2)):
    """Setup a matplotlib plot with grid and labels"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    return fig, ax


def test_ray_circle_intersection_hit():
    """Test ray-circle intersection with a hit case"""
    print("Testing ray-circle intersection (HIT case)")

    # Create test data
    ray = Ray(origin=Vector2(x=0, y=0), direction=Vector2(x=1, y=0.5), length=2.5)

    circle = Circle(center=Vector2(x=1.5, y=0.75), radius=0.3)

    # Compute intersection
    result = ray_circle_intersection(ray, circle)

    # Plot
    fig, ax = setup_plot("Ray-Circle Intersection (HIT)")
    plot_ray(ax, ray, color="blue")
    plot_circle(ax, circle, color="red", alpha=0.3)

    if result.intersects and result.intersection:
        plot_intersection_point(ax, result.intersection, color="red")
        print(
            f"Intersection found at: ({result.intersection.x:.3f}, {result.intersection.y:.3f})"
        )
        print(f"Distance along ray: {result.t:.3f}")
        print(f"Ray length: {ray.length}")
    else:
        print("No intersection found")

    ax.legend()
    plt.show()
    return result


def test_ray_circle_intersection_miss():
    """Test ray-circle intersection with a miss case"""
    print("Testing ray-circle intersection (MISS case)")

    # Create test data
    ray = Ray(origin=Vector2(x=0, y=0), direction=Vector2(x=1, y=0), length=2.5)

    circle = Circle(center=Vector2(x=1.5, y=1), radius=0.3)

    # Compute intersection
    result = ray_circle_intersection(ray, circle)

    # Plot
    fig, ax = setup_plot("Ray-Circle Intersection (MISS)")
    plot_ray(ax, ray, color="blue")
    plot_circle(ax, circle, color="red", alpha=0.3)

    if result.intersects and result.intersection:
        plot_intersection_point(ax, result.intersection, color="red")
        print(
            f"Intersection found at: ({result.intersection.x:.3f}, {result.intersection.y:.3f})"
        )
    else:
        print("No intersection found (as expected)")

    ax.legend()
    plt.show()
    return result


def test_ray_rectangle_intersection_hit():
    """Test ray-rectangle intersection with a hit case"""
    print("Testing ray-rectangle intersection (HIT case)")

    # Create test data
    ray = Ray(origin=Vector2(x=0, y=0), direction=Vector2(x=1, y=0.3), length=2.5)

    rectangle = Rectangle(
        center=Vector2(x=1.5, y=0.4),
        width=0.6,
        height=0.4,
        rotation=30,  # 30 degree rotation
    )

    # Compute intersection
    result = ray_rectangle_intersection(ray, rectangle)

    # Plot
    fig, ax = setup_plot("Ray-Rectangle Intersection (HIT)")
    plot_ray(ax, ray, color="blue")
    plot_rectangle(ax, rectangle, color="green", alpha=0.3)

    if result.intersects and result.intersection:
        plot_intersection_point(ax, result.intersection, color="red")
        print(
            f"Intersection found at: ({result.intersection.x:.3f}, {result.intersection.y:.3f})"
        )
        print(f"Distance along ray: {result.t:.3f}")
        print(f"Ray length: {ray.length}")
    else:
        print("No intersection found")

    ax.legend()
    plt.show()
    return result


def test_ray_rectangle_intersection_miss():
    """Test ray-rectangle intersection with a miss case"""
    print("Testing ray-rectangle intersection (MISS case)")

    # Create test data
    ray = Ray(origin=Vector2(x=0, y=0), direction=Vector2(x=1, y=0), length=2.5)

    rectangle = Rectangle(
        center=Vector2(x=1.5, y=1), width=0.4, height=0.3, rotation=45
    )

    # Compute intersection
    result = ray_rectangle_intersection(ray, rectangle)

    # Plot
    fig, ax = setup_plot("Ray-Rectangle Intersection (MISS)")
    plot_ray(ax, ray, color="blue")
    plot_rectangle(ax, rectangle, color="green", alpha=0.3)

    if result.intersects and result.intersection:
        plot_intersection_point(ax, result.intersection, color="red")
        print(
            f"Intersection found at: ({result.intersection.x:.3f}, {result.intersection.y:.3f})"
        )
    else:
        print("No intersection found (as expected)")

    ax.legend()
    plt.show()
    return result


def test_ray_line_intersection_hit():
    """Test ray-line intersection with a hit case"""
    print("Testing ray-line intersection (HIT case)")

    # Create test data
    ray = Ray(origin=Vector2(x=0, y=0), direction=Vector2(x=1, y=0.5), length=2.5)

    line = Line(p1=Vector2(x=0.8, y=0.2), p2=Vector2(x=1.5, y=0.8))

    # Compute intersection
    result = ray_line_intersection(ray, line)

    # Plot
    fig, ax = setup_plot("Ray-Line Intersection (HIT)")
    plot_ray(ax, ray, color="blue")
    plot_line(ax, line, color="purple")

    if result.intersects and result.intersection:
        plot_intersection_point(ax, result.intersection, color="red")
        print(
            f"Intersection found at: ({result.intersection.x:.3f}, {result.intersection.y:.3f})"
        )
        print(f"Distance along ray: {result.t:.3f}")
        print(f"Ray length: {ray.length}")
    else:
        print("No intersection found")

    ax.legend()
    plt.show()
    return result


def test_ray_line_intersection_miss():
    """Test ray-line intersection with a miss case"""
    print("Testing ray-line intersection (MISS case)")

    # Create test data
    ray = Ray(origin=Vector2(x=0, y=0), direction=Vector2(x=1, y=0), length=2.5)

    line = Line(p1=Vector2(x=0.5, y=0.5), p2=Vector2(x=1.5, y=1.0))

    # Compute intersection
    result = ray_line_intersection(ray, line)

    # Plot
    fig, ax = setup_plot("Ray-Line Intersection (MISS)")
    plot_ray(ax, ray, color="blue")
    plot_line(ax, line, color="purple")

    if result.intersects and result.intersection:
        plot_intersection_point(ax, result.intersection, color="red")
        print(
            f"Intersection found at: ({result.intersection.x:.3f}, {result.intersection.y:.3f})"
        )
    else:
        print("No intersection found (as expected)")

    ax.legend()
    plt.show()
    return result


def test_ray_boundary_intersection():
    """Test ray-boundary intersection with a polygon"""
    print("Testing ray-boundary intersection")

    # Create test data
    ray = Ray(origin=Vector2(x=0.2, y=0.2), direction=Vector2(x=1, y=0.8), length=3.0)

    # Create a pentagon boundary
    boundary = PolygonBoundaryConfig(
        vertices=[
            Vector2(x=0, y=0),
            Vector2(x=2, y=0),
            Vector2(x=2.5, y=1),
            Vector2(x=1, y=2),
            Vector2(x=-0.5, y=1),
        ]
    )

    # Compute intersection
    result = ray_boundary_intersection(ray, boundary)

    # Plot
    fig, ax = setup_plot("Ray-Boundary Intersection", xlim=(-1, 3), ylim=(-0.5, 2.5))
    plot_ray(ax, ray, color="blue")

    # Plot boundary
    boundary_obj = PolygonBoundary(boundary)
    for wall in boundary_obj.walls:
        p1, p2 = wall
        line = Line(p1=Vector2(x=p1[0], y=p1[1]), p2=Vector2(x=p2[0], y=p2[1]))
        plot_line(ax, line, color="orange")

    # Plot boundary vertices
    for vertex in boundary.vertices:
        ax.scatter(vertex.x, vertex.y, color="orange", s=50, marker="o")

    if result.intersects and result.intersection:
        plot_intersection_point(ax, result.intersection, color="red")
        print(
            f"Intersection found at: ({result.intersection.x:.3f}, {result.intersection.y:.3f})"
        )
        print(f"Distance along ray: {result.t:.3f}")
        print(f"Ray length: {ray.length}")
    else:
        print("No intersection found")

    ax.legend()
    plt.show()
    return result


def test_multiple_intersections():
    """Test a ray that intersects multiple objects"""
    print("Testing multiple intersections")

    # Create test data
    ray = Ray(origin=Vector2(x=0, y=0.5), direction=Vector2(x=1, y=0), length=2.5)

    # Create multiple objects
    circle1 = Circle(center=Vector2(x=0.8, y=0.5), radius=0.2)
    circle2 = Circle(center=Vector2(x=1.8, y=0.5), radius=0.15)

    rectangle = Rectangle(
        center=Vector2(x=1.3, y=0.5), width=0.3, height=0.4, rotation=0
    )

    line = Line(p1=Vector2(x=2.2, y=0.2), p2=Vector2(x=2.2, y=0.8))

    # Compute intersections
    result_circle1 = ray_circle_intersection(ray, circle1)
    result_circle2 = ray_circle_intersection(ray, circle2)
    result_rectangle = ray_rectangle_intersection(ray, rectangle)
    result_line = ray_line_intersection(ray, line)

    # Plot
    fig, ax = setup_plot("Multiple Intersections", xlim=(-0.2, 2.5), ylim=(0, 1))
    plot_ray(ax, ray, color="blue")

    # Plot objects
    plot_circle(ax, circle1, color="red", alpha=0.3)
    plot_circle(ax, circle2, color="red", alpha=0.3)
    plot_rectangle(ax, rectangle, color="green", alpha=0.3)
    plot_line(ax, line, color="purple")

    # Plot intersections
    intersections = []
    if result_circle1.intersects and result_circle1.intersection:
        plot_intersection_point(ax, result_circle1.intersection, color="red")
        intersections.append(
            ("Circle 1", result_circle1.intersection, result_circle1.t)
        )

    if result_circle2.intersects and result_circle2.intersection:
        plot_intersection_point(ax, result_circle2.intersection, color="red")
        intersections.append(
            ("Circle 2", result_circle2.intersection, result_circle2.t)
        )

    if result_rectangle.intersects and result_rectangle.intersection:
        plot_intersection_point(ax, result_rectangle.intersection, color="red")
        intersections.append(
            ("Rectangle", result_rectangle.intersection, result_rectangle.t)
        )

    if result_line.intersects and result_line.intersection:
        plot_intersection_point(ax, result_line.intersection, color="red")
        intersections.append(("Line", result_line.intersection, result_line.t))

    # Print results sorted by distance
    intersections.sort(key=lambda x: x[2])
    print("Intersections found (sorted by distance):")
    for name, point, distance in intersections:
        print(f"  {name}: ({point.x:.3f}, {point.y:.3f}) at t={distance:.3f}")

    print(f"Ray length: {ray.length}")

    ax.legend()
    plt.show()
    return intersections


def test_ray_intersection_with_length():
    """Test the main ray_intersection function with ray length limit"""
    print("Testing ray_intersection with ray length limit")

    # Create test data
    ray = Ray(
        origin=Vector2(x=0, y=0.5), direction=Vector2(x=1, y=0), length=1.0
    )  # Ray length = 1.0

    # Create obstacles at various distances
    # Close obstacle (within ray.length)
    obstacle1_config = ObstacleConfig(
        shape=Circle(center=Vector2(x=0.8, y=0.5), radius=0.15)
    )
    obstacle1 = CircleObstacle(obstacle1_config)

    # Medium distance obstacle (close to ray.length limit)
    obstacle2_config = ObstacleConfig(
        shape=Rectangle(
            center=Vector2(x=1.3, y=0.5), width=0.25, height=0.3, rotation=20
        )
    )
    obstacle2 = RectangleObstacle(obstacle2_config)

    # Far obstacle (beyond ray.length - should be ignored)
    obstacle3_config = ObstacleConfig(
        shape=Circle(center=Vector2(x=1.9, y=0.5), radius=0.2)
    )
    obstacle3 = CircleObstacle(obstacle3_config)

    obstacles = [obstacle1, obstacle2, obstacle3]

    # Create boundary (beyond ray.length)
    boundary = PolygonBoundaryConfig(
        vertices=[
            Vector2(x=2.5, y=0.2),
            Vector2(x=3.0, y=0.2),
            Vector2(x=3.0, y=0.8),
            Vector2(x=2.5, y=0.8),
        ]
    )
    boundaries = [boundary]

    # Compute intersection
    result = ray_intersection(ray, obstacles, boundaries)

    # Also compute individual intersections for comparison
    individual_results = []
    for i, obstacle in enumerate(obstacles):
        individual_result = ray_obstacle_intersection(ray, obstacle)
        if individual_result.intersects:
            individual_results.append((f"Obstacle {i+1}", individual_result))

    boundary_result = ray_boundary_intersection(ray, boundary)
    if boundary_result.intersects:
        individual_results.append(("Boundary", boundary_result))

    # Plot
    fig, ax = setup_plot(
        "Ray Intersection with Length Limit", xlim=(-0.2, 3.2), ylim=(0, 1)
    )

    # Plot ray using its actual length
    plot_ray(ax, ray, color="blue")

    # Draw vertical line to show ray.length limit
    ax.axvline(
        x=ray.length,
        color="blue",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
        label=f"ray.length = {ray.length}",
    )

    # Plot obstacles with different colors based on whether they're within ray length
    for i, obstacle in enumerate(obstacles):
        shape = obstacle.get_current_state()
        if isinstance(shape, Circle):
            # Check if this obstacle would be hit within ray length
            individual_result = ray_obstacle_intersection(ray, obstacle)
            within_ray_length = (
                individual_result.intersects and individual_result.t <= ray.length
            )
            color = "red" if within_ray_length else "lightcoral"
            alpha = 0.6 if within_ray_length else 0.3
            plot_circle(ax, shape, color=color, alpha=alpha)

            # Add label
            ax.text(
                shape.center.x,
                shape.center.y + 0.1,
                f"Obs{i+1}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        elif isinstance(shape, Rectangle):
            individual_result = ray_obstacle_intersection(ray, obstacle)
            within_ray_length = (
                individual_result.intersects and individual_result.t <= ray.length
            )
            color = "green" if within_ray_length else "lightgreen"
            alpha = 0.6 if within_ray_length else 0.3
            plot_rectangle(ax, shape, color=color, alpha=alpha)

            # Add label
            ax.text(
                shape.center.x,
                shape.center.y + 0.15,
                f"Obs{i+1}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # Plot boundary
    boundary_obj = PolygonBoundary(boundary)
    boundary_within_ray_length = (
        boundary_result.intersects and boundary_result.t <= ray.length
    )
    boundary_color = "orange" if boundary_within_ray_length else "moccasin"

    for wall in boundary_obj.walls:
        p1, p2 = wall
        line = Line(p1=Vector2(x=p1[0], y=p1[1]), p2=Vector2(x=p2[0], y=p2[1]))
        ax.plot(
            [line.p1.x, line.p2.x],
            [line.p1.y, line.p2.y],
            color=boundary_color,
            linewidth=3,
            alpha=0.8,
        )

    # Plot the actual intersection found by ray_intersection
    if result.intersects and result.intersection:
        plot_intersection_point(ax, result.intersection, color="red", size=150)
        print(
            f"✅ Main intersection found at: ({result.intersection.x:.3f}, {result.intersection.y:.3f})"
        )
        print(f"Distance along ray: {result.t:.3f}")
        print(f"Ray length: {ray.length}")
        print(f"Within ray length: {result.t <= ray.length}")
    else:
        print("❌ No intersection found within ray length")

    # Print all individual intersections for comparison
    print(f"\nIndividual intersection analysis:")
    print(f"Ray length: {ray.length}")
    for name, individual_result in individual_results:
        within_ray_length = individual_result.t <= ray.length
        status = "✅ INCLUDED" if within_ray_length else "❌ EXCLUDED"
        print(
            f"  {name}: t={individual_result.t:.3f} - within_ray_length:{within_ray_length} - {status}"
        )

    # Add information to the plot
    ax.text(
        ray.length / 2,
        0.9,
        f"Ray length: {ray.length}\nIntersections limited by ray length",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"),
        fontsize=10,
    )

    ax.legend()
    plt.show()
    return result


def test_ray_length_vs_intersections():
    """Test to show how ray length affects intersections"""
    print("Testing ray length vs intersections")

    # Create obstacles at different distances
    circle1 = Circle(center=Vector2(x=0.8, y=0.5), radius=0.15)
    circle2 = Circle(center=Vector2(x=1.6, y=0.5), radius=0.15)
    circle3 = Circle(center=Vector2(x=2.4, y=0.5), radius=0.15)

    # Test with different ray lengths
    ray_lengths = [1.0, 2.0, 3.0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, length in enumerate(ray_lengths):
        ax = axes[i]

        # Create ray with specific length
        ray = Ray(
            origin=Vector2(x=0, y=0.5), direction=Vector2(x=1, y=0), length=length
        )

        # Test intersections
        result1 = ray_circle_intersection(ray, circle1)
        result2 = ray_circle_intersection(ray, circle2)
        result3 = ray_circle_intersection(ray, circle3)

        # Plot
        ax.set_xlim(-0.2, 3.0)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Ray Length = {length}")

        # Plot ray
        origin = ray.origin.to_numpy()
        direction = ray.direction.to_numpy()
        direction_norm = direction / np.linalg.norm(direction)

        ax.arrow(
            origin[0],
            origin[1],
            direction_norm[0] * length,
            direction_norm[1] * length,
            head_width=0.03,
            head_length=0.05,
            fc="blue",
            ec="blue",
            linewidth=2,
        )

        # Plot circles with different colors based on intersection
        plot_circle(
            ax, circle1, color="red" if result1.intersects else "lightcoral", alpha=0.6
        )
        plot_circle(
            ax, circle2, color="red" if result2.intersects else "lightcoral", alpha=0.6
        )
        plot_circle(
            ax, circle3, color="red" if result3.intersects else "lightcoral", alpha=0.6
        )

        # Plot intersections
        if result1.intersects and result1.intersection:
            plot_intersection_point(ax, result1.intersection, color="red", size=100)
        if result2.intersects and result2.intersection:
            plot_intersection_point(ax, result2.intersection, color="red", size=100)
        if result3.intersects and result3.intersection:
            plot_intersection_point(ax, result3.intersection, color="red", size=100)

        # Add distance labels
        ax.text(0.8, 0.3, "0.8", ha="center", fontsize=10)
        ax.text(1.6, 0.3, "1.6", ha="center", fontsize=10)
        ax.text(2.4, 0.3, "2.4", ha="center", fontsize=10)

    plt.tight_layout()
    plt.show()

    # Print results
    for length in ray_lengths:
        ray = Ray(
            origin=Vector2(x=0, y=0.5), direction=Vector2(x=1, y=0), length=length
        )
        result1 = ray_circle_intersection(ray, circle1)
        result2 = ray_circle_intersection(ray, circle2)
        result3 = ray_circle_intersection(ray, circle3)

        print(f"\nRay length {length}:")
        print(f"  Circle 1 (dist ~0.65): {'HIT' if result1.intersects else 'MISS'}")
        print(f"  Circle 2 (dist ~1.45): {'HIT' if result2.intersects else 'MISS'}")
        print(f"  Circle 3 (dist ~2.25): {'HIT' if result3.intersects else 'MISS'}")


def test_batch_ray_circle_intersection():
    """Test batch ray-circle intersection with multiple rays and circles"""
    print("Testing batch ray-circle intersection")

    # Create multiple rays in a fan pattern
    num_rays = 8
    ray_angles = np.linspace(-np.pi / 4, np.pi / 4, num_rays)
    ray_origins = np.array([[0.0, 0.5] for _ in range(num_rays)])
    ray_directions = np.array([[np.cos(angle), np.sin(angle)] for angle in ray_angles])
    ray_lengths = np.full(num_rays, 2.5)

    # Create rays array [N, 5] format
    rays = np.column_stack(
        [
            ray_origins[:, 0],  # origin_x
            ray_origins[:, 1],  # origin_y
            ray_directions[:, 0],  # dir_x
            ray_directions[:, 1],  # dir_y
            ray_lengths,  # length
        ]
    )

    # Create multiple circles
    circles = np.array(
        [
            [1.0, 0.3, 0.2],  # [center_x, center_y, radius]
            [1.5, 0.7, 0.15],
            [2.0, 0.5, 0.25],
            [0.8, 0.8, 0.1],
        ]
    )

    # Test batch intersection
    print("Running batch intersection...")
    start_time = time.time()
    batch_distances = batch_ray_circle_intersection(rays, circles)
    batch_time = time.time() - start_time

    # Compare with individual intersections
    print("Running individual intersections for comparison...")
    start_time = time.time()
    individual_distances = np.full((num_rays, len(circles)), np.inf)

    for i in range(num_rays):
        ray = Ray(
            origin=Vector2(x=rays[i, 0], y=rays[i, 1]),
            direction=Vector2(x=rays[i, 2], y=rays[i, 3]),
            length=rays[i, 4],
        )

        for j, circle_data in enumerate(circles):
            circle = Circle(
                center=Vector2(x=circle_data[0], y=circle_data[1]),
                radius=circle_data[2],
            )
            result = ray_circle_intersection(ray, circle)
            if result.intersects:
                individual_distances[i, j] = result.t

    individual_time = time.time() - start_time

    # Check if results match
    matches = np.allclose(batch_distances, individual_distances, rtol=1e-6)
    print(f"Batch vs Individual results match: {matches}")
    print(f"Batch time: {batch_time:.4f}s, Individual time: {individual_time:.4f}s")
    print(f"Speedup: {individual_time/batch_time:.2f}x")

    # Visualize results
    fig, ax = setup_plot(
        "Batch Ray-Circle Intersection", xlim=(-0.2, 2.8), ylim=(-0.2, 1.2)
    )

    # Plot rays
    for i in range(num_rays):
        ray_start = rays[i, :2]
        ray_dir = rays[i, 2:4]
        ray_length = rays[i, 4]

        ax.arrow(
            ray_start[0],
            ray_start[1],
            ray_dir[0] * ray_length,
            ray_dir[1] * ray_length,
            head_width=0.02,
            head_length=0.03,
            fc="blue",
            ec="blue",
            alpha=0.7,
        )

    # Plot circles
    for circle_data in circles:
        circle_patch = patches.Circle(
            (circle_data[0], circle_data[1]),
            circle_data[2],
            facecolor="red",
            edgecolor="darkred",
            alpha=0.3,
            linewidth=2,
        )
        ax.add_patch(circle_patch)

    # Plot intersections
    for i in range(num_rays):
        for j in range(len(circles)):
            if batch_distances[i, j] < np.inf:
                # Calculate intersection point
                ray_start = rays[i, :2]
                ray_dir = rays[i, 2:4]
                intersection_point = ray_start + ray_dir * batch_distances[i, j]

                ax.scatter(
                    intersection_point[0],
                    intersection_point[1],
                    color="red",
                    s=50,
                    marker="x",
                    linewidth=3,
                )

    ax.legend(["Rays", "Circles", "Intersections"])
    plt.show()

    return batch_distances, individual_distances


def test_batch_ray_rectangle_intersection():
    """Test batch ray-rectangle intersection"""
    print("Testing batch ray-rectangle intersection")

    # Create multiple rays
    num_rays = 12
    ray_angles = np.linspace(-np.pi / 3, np.pi / 3, num_rays)
    ray_origins = np.array([[0.0, 0.5] for _ in range(num_rays)])
    ray_directions = np.array([[np.cos(angle), np.sin(angle)] for angle in ray_angles])
    ray_lengths = np.full(num_rays, 3.0)

    rays = np.column_stack(
        [
            ray_origins[:, 0],
            ray_origins[:, 1],
            ray_directions[:, 0],
            ray_directions[:, 1],
            ray_lengths,
        ]
    )

    # Create multiple rectangles
    rectangles = np.array(
        [
            [
                1.2,
                0.8,
                0.3,
                0.4,
                30,
            ],  # [center_x, center_y, width, height, rotation_degrees]
            [1.8, 0.2, 0.4, 0.3, -20],
            [2.2, 0.7, 0.5, 0.2, 45],
            [0.8, 0.3, 0.2, 0.6, 0],
        ]
    )

    # Test batch intersection
    print("Running batch intersection...")
    start_time = time.time()
    batch_distances = batch_ray_rectangle_intersection(rays, rectangles)
    batch_time = time.time() - start_time

    # Compare with individual intersections
    print("Running individual intersections for comparison...")
    start_time = time.time()
    individual_distances = np.full((num_rays, len(rectangles)), np.inf)

    for i in range(num_rays):
        ray = Ray(
            origin=Vector2(x=rays[i, 0], y=rays[i, 1]),
            direction=Vector2(x=rays[i, 2], y=rays[i, 3]),
            length=rays[i, 4],
        )

        for j, rect_data in enumerate(rectangles):
            rectangle = Rectangle(
                center=Vector2(x=rect_data[0], y=rect_data[1]),
                width=rect_data[2],
                height=rect_data[3],
                rotation=rect_data[4],
            )
            result = ray_rectangle_intersection(ray, rectangle)
            if result.intersects:
                individual_distances[i, j] = result.t

    individual_time = time.time() - start_time

    # Check if results match
    matches = np.allclose(batch_distances, individual_distances, rtol=1e-6)
    print(f"Batch vs Individual results match: {matches}")
    print(f"Batch time: {batch_time:.4f}s, Individual time: {individual_time:.4f}s")
    print(f"Speedup: {individual_time/batch_time:.2f}x")

    # Visualize results
    fig, ax = setup_plot(
        "Batch Ray-Rectangle Intersection", xlim=(-0.2, 3.0), ylim=(-0.2, 1.2)
    )

    # Plot rays
    for i in range(num_rays):
        ray_start = rays[i, :2]
        ray_dir = rays[i, 2:4]
        ray_length = rays[i, 4]

        ax.arrow(
            ray_start[0],
            ray_start[1],
            ray_dir[0] * ray_length,
            ray_dir[1] * ray_length,
            head_width=0.02,
            head_length=0.03,
            fc="blue",
            ec="blue",
            alpha=0.7,
        )

    # Plot rectangles
    for rect_data in rectangles:
        rect = patches.Rectangle(
            (-rect_data[2] / 2, -rect_data[3] / 2),
            rect_data[2],
            rect_data[3],
            facecolor="green",
            edgecolor="darkgreen",
            alpha=0.3,
            linewidth=2,
        )

        # Apply rotation and translation
        transform = (
            plt.matplotlib.transforms.Affine2D()
            .rotate_deg(rect_data[4])
            .translate(rect_data[0], rect_data[1])
            + ax.transData
        )
        rect.set_transform(transform)
        ax.add_patch(rect)

    # Plot intersections
    for i in range(num_rays):
        for j in range(len(rectangles)):
            if batch_distances[i, j] < np.inf:
                ray_start = rays[i, :2]
                ray_dir = rays[i, 2:4]
                intersection_point = ray_start + ray_dir * batch_distances[i, j]

                ax.scatter(
                    intersection_point[0],
                    intersection_point[1],
                    color="red",
                    s=50,
                    marker="x",
                    linewidth=3,
                )

    ax.legend(["Rays", "Rectangles", "Intersections"])
    plt.show()

    return batch_distances, individual_distances


def test_batch_ray_line_intersection():
    """Test batch ray-line intersection"""
    print("Testing batch ray-line intersection")

    # Create multiple rays
    num_rays = 10
    ray_angles = np.linspace(-np.pi / 4, np.pi / 4, num_rays)
    ray_origins = np.array([[0.0, 0.5] for _ in range(num_rays)])
    ray_directions = np.array([[np.cos(angle), np.sin(angle)] for angle in ray_angles])
    ray_lengths = np.full(num_rays, 2.5)

    rays = np.column_stack(
        [
            ray_origins[:, 0],
            ray_origins[:, 1],
            ray_directions[:, 0],
            ray_directions[:, 1],
            ray_lengths,
        ]
    )

    # Create multiple lines
    lines = np.array(
        [
            [0.8, 0.2, 0.8, 0.8],  # [p1_x, p1_y, p2_x, p2_y] - vertical line
            [1.0, 0.1, 1.5, 0.9],  # diagonal line
            [1.8, 0.3, 2.2, 0.3],  # horizontal line
            [0.5, 0.6, 2.0, 0.4],  # diagonal line
        ]
    )

    # Test batch intersection
    print("Running batch intersection...")
    start_time = time.time()
    batch_distances = batch_ray_line_intersection(rays, lines)
    batch_time = time.time() - start_time

    # Compare with individual intersections
    print("Running individual intersections for comparison...")
    start_time = time.time()
    individual_distances = np.full((num_rays, len(lines)), np.inf)

    for i in range(num_rays):
        ray = Ray(
            origin=Vector2(x=rays[i, 0], y=rays[i, 1]),
            direction=Vector2(x=rays[i, 2], y=rays[i, 3]),
            length=rays[i, 4],
        )

        for j, line_data in enumerate(lines):
            line = Line(
                p1=Vector2(x=line_data[0], y=line_data[1]),
                p2=Vector2(x=line_data[2], y=line_data[3]),
            )
            result = ray_line_intersection(ray, line)
            if result.intersects:
                individual_distances[i, j] = result.t

    individual_time = time.time() - start_time

    # Check if results match
    matches = np.allclose(batch_distances, individual_distances, rtol=1e-6)
    print(f"Batch vs Individual results match: {matches}")
    print(f"Batch time: {batch_time:.4f}s, Individual time: {individual_time:.4f}s")
    print(f"Speedup: {individual_time/batch_time:.2f}x")

    # Visualize results
    fig, ax = setup_plot(
        "Batch Ray-Line Intersection", xlim=(-0.2, 2.5), ylim=(-0.2, 1.2)
    )

    # Plot rays
    for i in range(num_rays):
        ray_start = rays[i, :2]
        ray_dir = rays[i, 2:4]
        ray_length = rays[i, 4]

        ax.arrow(
            ray_start[0],
            ray_start[1],
            ray_dir[0] * ray_length,
            ray_dir[1] * ray_length,
            head_width=0.02,
            head_length=0.03,
            fc="blue",
            ec="blue",
            alpha=0.7,
        )

    # Plot lines
    for line_data in lines:
        ax.plot(
            [line_data[0], line_data[2]],
            [line_data[1], line_data[3]],
            color="purple",
            linewidth=3,
            alpha=0.7,
        )

    # Plot intersections
    for i in range(num_rays):
        for j in range(len(lines)):
            if batch_distances[i, j] < np.inf:
                ray_start = rays[i, :2]
                ray_dir = rays[i, 2:4]
                intersection_point = ray_start + ray_dir * batch_distances[i, j]

                ax.scatter(
                    intersection_point[0],
                    intersection_point[1],
                    color="red",
                    s=50,
                    marker="x",
                    linewidth=3,
                )

    ax.legend(["Rays", "Lines", "Intersections"])
    plt.show()

    return batch_distances, individual_distances


def test_batch_lidar_simulation():
    """Test batch processing with LiDAR-like scenario"""
    print("Testing batch LiDAR simulation")

    # Create LiDAR-like rays (180 degrees, 180 rays)
    num_rays = 180
    agent_pos = Vector2(x=1.0, y=1.0)
    base_direction = Vector2(x=0.5, y=0.5)  # Forward direction
    max_range = 2.0

    print(f"Creating {num_rays} LiDAR rays...")
    rays = create_lidar_rays(
        agent_pos, base_direction, num_rays, max_range, fov_degrees=180
    )

    # Create a complex environment with multiple obstacles
    circles = np.array(
        [
            [1.5, 0.5, 0.2],
            [0.3, 1.5, 0.15],
            [2.2, 1.8, 0.25],
            [0.8, 0.3, 0.1],
            [2.5, 0.8, 0.18],
        ]
    )

    rectangles = np.array(
        [[1.8, 1.2, 0.3, 0.4, 30], [0.5, 0.8, 0.2, 0.5, -15], [2.0, 0.4, 0.4, 0.2, 45]]
    )

    lines = np.array(
        [
            [0.0, 0.0, 0.0, 3.0],  # left boundary
            [0.0, 0.0, 3.0, 0.0],  # bottom boundary
            [3.0, 0.0, 3.0, 3.0],  # right boundary
            [0.0, 3.0, 3.0, 3.0],  # top boundary
        ]
    )

    # Test batch intersection
    print("Running batch intersection...")
    start_time = time.time()

    # Get distances for each shape type
    circle_distances = batch_ray_circle_intersection(rays, circles)
    rectangle_distances = batch_ray_rectangle_intersection(rays, rectangles)
    line_distances = batch_ray_line_intersection(rays, lines)

    # Find closest intersection for each ray
    closest_distances = np.full(num_rays, np.inf)

    # Check circles
    min_circle_dist = np.min(circle_distances, axis=1)
    closer_mask = min_circle_dist < closest_distances
    closest_distances[closer_mask] = min_circle_dist[closer_mask]

    # Check rectangles
    min_rect_dist = np.min(rectangle_distances, axis=1)
    closer_mask = min_rect_dist < closest_distances
    closest_distances[closer_mask] = min_rect_dist[closer_mask]

    # Check lines
    min_line_dist = np.min(line_distances, axis=1)
    closer_mask = min_line_dist < closest_distances
    closest_distances[closer_mask] = min_line_dist[closer_mask]

    batch_time = time.time() - start_time

    print(f"Batch processing time: {batch_time:.4f}s")
    print(f"Rays processed: {num_rays}")
    print(
        f"Obstacles: {len(circles)} circles, {len(rectangles)} rectangles, {len(lines)} lines"
    )
    print(f"Time per ray: {batch_time/num_rays*1000:.3f}ms")

    # Count hits
    hits = np.sum(closest_distances < np.inf)
    print(f"Intersection hits: {hits}/{num_rays} ({hits/num_rays*100:.1f}%)")

    # Visualize results
    fig, ax = setup_plot("Batch LiDAR Simulation", xlim=(-0.2, 3.2), ylim=(-0.2, 3.2))

    # Plot environment boundaries
    for line_data in lines:
        ax.plot(
            [line_data[0], line_data[2]],
            [line_data[1], line_data[3]],
            color="black",
            linewidth=3,
            alpha=0.8,
        )

    # Plot obstacles
    for circle_data in circles:
        circle_patch = patches.Circle(
            (circle_data[0], circle_data[1]),
            circle_data[2],
            facecolor="red",
            edgecolor="darkred",
            alpha=0.4,
            linewidth=2,
        )
        ax.add_patch(circle_patch)

    for rect_data in rectangles:
        rect = patches.Rectangle(
            (-rect_data[2] / 2, -rect_data[3] / 2),
            rect_data[2],
            rect_data[3],
            facecolor="green",
            edgecolor="darkgreen",
            alpha=0.4,
            linewidth=2,
        )
        transform = (
            plt.matplotlib.transforms.Affine2D()
            .rotate_deg(rect_data[4])
            .translate(rect_data[0], rect_data[1])
            + ax.transData
        )
        rect.set_transform(transform)
        ax.add_patch(rect)

    # Plot rays (every 10th ray to avoid clutter) with enhanced colors
    hit_labeled = False
    miss_labeled = False

    for i in range(0, num_rays, 3):
        ray_start = rays[i, :2]
        ray_dir = rays[i, 2:4]

        if closest_distances[i] < np.inf:
            # Ray hits something - use bright hit color
            hit_point = ray_start + ray_dir * closest_distances[i]
            ax.plot(
                [ray_start[0], hit_point[0]],
                [ray_start[1], hit_point[1]],
                color="limegreen",
                alpha=0.8,
                linewidth=2,
                label="Hit" if not hit_labeled else "",
                zorder=5,
            )
            ax.scatter(
                hit_point[0],
                hit_point[1],
                color="darkgreen",
                s=15,
                marker="o",
                edgecolors="white",
                linewidth=1,
                zorder=8,
            )
            hit_labeled = True
        else:
            # Ray doesn't hit anything within range - use miss color
            end_point = ray_start + ray_dir * rays[i, 4]
            ax.plot(
                [ray_start[0], end_point[0]],
                [ray_start[1], end_point[1]],
                color="lightsteelblue",
                alpha=0.4,
                linewidth=1,
                linestyle="--",
                label="Miss" if not miss_labeled else "",
                zorder=3,
            )
            miss_labeled = True

    # Plot agent
    ax.scatter(
        agent_pos.x,
        agent_pos.y,
        color="black",
        s=100,
        marker="o",
        edgecolors="white",
        linewidth=2,
        label="Agent",
        zorder=10,
    )

    # Plot agent facing direction
    facing_length = 0.3  # Length of the direction arrow
    ax.arrow(
        agent_pos.x,
        agent_pos.y,
        base_direction.x * facing_length,
        base_direction.y * facing_length,
        head_width=0.05,
        head_length=0.1,
        fc="yellow",
        ec="black",
        linewidth=2,
        zorder=11,
        label="Facing Direction",
    )

    ax.legend()
    plt.show()

    return closest_distances


def test_batch_performance_comparison():
    """Test performance comparison between batch and individual processing"""
    print("Testing batch vs individual performance")

    # Test with increasing numbers of rays
    ray_counts = [10, 50, 100, 200, 500]
    obstacle_counts = [5, 10, 20, 50]

    results = []

    for num_rays in ray_counts:
        for num_obstacles in obstacle_counts:
            print(f"\nTesting {num_rays} rays vs {num_obstacles} obstacles...")

            # Create rays
            ray_angles = np.linspace(-np.pi / 2, np.pi / 2, num_rays)
            ray_origins = np.array([[0.0, 0.5] for _ in range(num_rays)])
            ray_directions = np.array(
                [[np.cos(angle), np.sin(angle)] for angle in ray_angles]
            )
            ray_lengths = np.full(num_rays, 3.0)

            rays = np.column_stack(
                [
                    ray_origins[:, 0],
                    ray_origins[:, 1],
                    ray_directions[:, 0],
                    ray_directions[:, 1],
                    ray_lengths,
                ]
            )

            # Create obstacles (circles for simplicity)
            circles = np.random.rand(num_obstacles, 3)
            circles[:, 0] *= 2.5  # x position
            circles[:, 1] *= 1.0  # y position
            circles[:, 2] *= 0.2  # radius
            circles[:, 2] += 0.05  # minimum radius

            # Test batch processing
            start_time = time.time()
            batch_distances = batch_ray_circle_intersection(rays, circles)
            batch_time = time.time() - start_time

            # Test individual processing
            start_time = time.time()
            individual_distances = np.full((num_rays, num_obstacles), np.inf)

            for i in range(num_rays):
                ray = Ray(
                    origin=Vector2(x=rays[i, 0], y=rays[i, 1]),
                    direction=Vector2(x=rays[i, 2], y=rays[i, 3]),
                    length=rays[i, 4],
                )

                for j, circle_data in enumerate(circles):
                    circle = Circle(
                        center=Vector2(x=circle_data[0], y=circle_data[1]),
                        radius=circle_data[2],
                    )
                    result = ray_circle_intersection(ray, circle)
                    if result.intersects:
                        individual_distances[i, j] = result.t

            individual_time = time.time() - start_time

            speedup = individual_time / batch_time if batch_time > 0 else 0

            results.append(
                {
                    "num_rays": num_rays,
                    "num_obstacles": num_obstacles,
                    "batch_time": batch_time,
                    "individual_time": individual_time,
                    "speedup": speedup,
                }
            )

            print(f"  Batch time: {batch_time:.4f}s")
            print(f"  Individual time: {individual_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")

            # Verify results match
            matches = np.allclose(batch_distances, individual_distances, rtol=1e-6)
            print(f"  Results match: {matches}")

    # Display results summary
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)
    print(
        f"{'Rays':<6} {'Obstacles':<10} {'Batch (s)':<10} {'Individual (s)':<14} {'Speedup':<8}"
    )
    print("-" * 60)

    for result in results:
        print(
            f"{result['num_rays']:<6} {result['num_obstacles']:<10} "
            f"{result['batch_time']:<10.4f} {result['individual_time']:<14.4f} "
            f"{result['speedup']:<8.2f}x"
        )

    return results


def test_batch_ray_intersection_full():
    """Test the main batch_ray_intersection function with mixed obstacles"""
    print("Testing batch_ray_intersection with mixed obstacles")

    # Create LiDAR-like rays
    num_rays = 72  # 5-degree intervals
    agent_pos = Vector2(x=1.5, y=1.5)
    base_direction = Vector2(x=1.0, y=0.0)  # Forward direction
    max_range = 2.0

    rays = create_lidar_rays(agent_pos, base_direction, num_rays, max_range)

    # Create mixed obstacles
    obstacles = []

    # Add circle obstacles
    circle_configs = [
        ObstacleConfig(shape=Circle(center=Vector2(x=1.0, y=0.8), radius=0.2)),
        ObstacleConfig(shape=Circle(center=Vector2(x=2.5, y=2.0), radius=0.15)),
        ObstacleConfig(shape=Circle(center=Vector2(x=0.5, y=2.2), radius=0.25)),
    ]

    for config in circle_configs:
        obstacles.append(CircleObstacle(config))

    # Add rectangle obstacles
    rect_configs = [
        ObstacleConfig(
            shape=Rectangle(
                center=Vector2(x=2.0, y=1.0), width=0.3, height=0.4, rotation=30
            )
        ),
        ObstacleConfig(
            shape=Rectangle(
                center=Vector2(x=0.8, y=1.8), width=0.4, height=0.2, rotation=-15
            )
        ),
    ]

    for config in rect_configs:
        obstacles.append(RectangleObstacle(config))

    # Create boundaries
    boundaries = [
        PolygonBoundaryConfig(
            vertices=[
                Vector2(x=0.0, y=0.0),
                Vector2(x=3.0, y=0.0),
                Vector2(x=3.0, y=3.0),
                Vector2(x=0.0, y=3.0),
            ]
        )
    ]

    # Test batch intersection
    print("Running batch intersection...")
    start_time = time.time()
    batch_distances = batch_ray_intersection(rays, obstacles, boundaries)
    batch_time = time.time() - start_time

    # Compare with individual intersections
    print("Running individual intersections for comparison...")
    start_time = time.time()
    individual_distances = np.full(num_rays, np.inf)

    for i in range(num_rays):
        ray = Ray(
            origin=Vector2(x=rays[i, 0], y=rays[i, 1]),
            direction=Vector2(x=rays[i, 2], y=rays[i, 3]),
            length=rays[i, 4],
        )

        result = ray_intersection(ray, obstacles, boundaries)
        if result.intersects:
            individual_distances[i] = result.t

    individual_time = time.time() - start_time

    # Check if results match
    matches = np.allclose(batch_distances, individual_distances, rtol=1e-6)
    print(f"Batch vs Individual results match: {matches}")
    print(f"Batch time: {batch_time:.4f}s, Individual time: {individual_time:.4f}s")
    print(f"Speedup: {individual_time/batch_time:.2f}x")

    # Count hits
    batch_hits = np.sum(batch_distances < np.inf)
    individual_hits = np.sum(individual_distances < np.inf)
    print(f"Batch hits: {batch_hits}/{num_rays} ({batch_hits/num_rays*100:.1f}%)")
    print(
        f"Individual hits: {individual_hits}/{num_rays} ({individual_hits/num_rays*100:.1f}%)"
    )

    # Visualize results
    fig, ax = setup_plot(
        "Batch Ray Intersection - Mixed Obstacles", xlim=(-0.2, 3.2), ylim=(-0.2, 3.2)
    )

    # Plot boundaries
    for boundary in boundaries:
        boundary_obj = PolygonBoundary(boundary)
        for wall in boundary_obj.walls:
            p1, p2 = wall
            ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=3, alpha=0.8
            )

    # Plot obstacles
    for obstacle in obstacles:
        shape = obstacle.get_current_state()
        if isinstance(shape, Circle):
            circle_patch = patches.Circle(
                (shape.center.x, shape.center.y),
                shape.radius,
                facecolor="red",
                edgecolor="darkred",
                alpha=0.4,
                linewidth=2,
            )
            ax.add_patch(circle_patch)
        elif isinstance(shape, Rectangle):
            rect = patches.Rectangle(
                (-shape.width / 2, -shape.height / 2),
                shape.width,
                shape.height,
                facecolor="green",
                edgecolor="darkgreen",
                alpha=0.4,
                linewidth=2,
            )
            transform = (
                plt.matplotlib.transforms.Affine2D()
                .rotate_deg(shape.rotation)
                .translate(shape.center.x, shape.center.y)
                + ax.transData
            )
            rect.set_transform(transform)
            ax.add_patch(rect)

    # Plot agent
    ax.scatter(
        agent_pos.x,
        agent_pos.y,
        color="blue",
        s=100,
        marker="o",
        edgecolors="darkblue",
        linewidth=2,
        label="Agent",
    )

    # Plot rays with hits (every 6th ray to avoid clutter) with enhanced colors
    hit_labeled = False
    miss_labeled = False

    for i in range(0, num_rays, 6):
        ray_start = rays[i, :2]
        ray_dir = rays[i, 2:4]

        if batch_distances[i] < np.inf:
            # Ray hits something - use bright hit color
            hit_point = ray_start + ray_dir * batch_distances[i]
            ax.plot(
                [ray_start[0], hit_point[0]],
                [ray_start[1], hit_point[1]],
                color="mediumorchid",
                alpha=0.7,
                linewidth=2,
                label="Hit" if not hit_labeled else "",
                zorder=5,
            )
            ax.scatter(
                hit_point[0],
                hit_point[1],
                color="purple",
                s=20,
                marker="*",
                edgecolors="white",
                linewidth=1,
                zorder=8,
            )
            hit_labeled = True
        else:
            # Ray doesn't hit anything within range - use miss color
            end_point = ray_start + ray_dir * rays[i, 4]
            ax.plot(
                [ray_start[0], end_point[0]],
                [ray_start[1], end_point[1]],
                color="lightsteelblue",
                alpha=0.4,
                linewidth=1,
                linestyle="--",
                label="Miss" if not miss_labeled else "",
                zorder=3,
            )
            miss_labeled = True

    ax.legend()
    plt.show()

    return batch_distances, individual_distances


def test_batch_ray_intersection_detailed():
    """Test the detailed batch ray intersection function with full RayIntersectionOutput results"""
    print("Testing batch_ray_intersection_detailed with full results")

    # Create LiDAR-like rays
    num_rays = 16  # Smaller number for detailed output
    agent_pos = Vector2(x=1.5, y=1.5)
    base_direction = Vector2(x=1.0, y=0.0)  # Forward direction
    max_range = 2.0

    rays = create_lidar_rays(
        agent_pos, base_direction, num_rays, max_range, fov_degrees=180
    )

    # Create mixed obstacles
    obstacles = []

    # Add circle obstacles
    circle_configs = [
        ObstacleConfig(shape=Circle(center=Vector2(x=2.0, y=1.2), radius=0.2)),
        ObstacleConfig(shape=Circle(center=Vector2(x=1.0, y=2.2), radius=0.15)),
    ]

    for config in circle_configs:
        obstacles.append(CircleObstacle(config))

    # Add rectangle obstacles
    rect_configs = [
        ObstacleConfig(
            shape=Rectangle(
                center=Vector2(x=2.5, y=1.8), width=0.3, height=0.4, rotation=30
            )
        ),
    ]

    for config in rect_configs:
        obstacles.append(RectangleObstacle(config))

    # Create boundaries
    boundaries = [
        PolygonBoundaryConfig(
            vertices=[
                Vector2(x=0.5, y=0.5),
                Vector2(x=3.5, y=0.5),
                Vector2(x=3.5, y=2.5),
                Vector2(x=0.5, y=2.5),
            ]
        )
    ]

    # Test detailed batch intersection
    print("Running detailed batch intersection...")
    start_time = time.time()
    detailed_results = batch_ray_intersection_detailed(rays, obstacles, boundaries)
    batch_time = time.time() - start_time

    # Compare with individual intersections
    print("Running individual intersections for comparison...")
    start_time = time.time()
    individual_results = []

    for i in range(num_rays):
        ray = Ray(
            origin=Vector2(x=rays[i, 0], y=rays[i, 1]),
            direction=Vector2(x=rays[i, 2], y=rays[i, 3]),
            length=rays[i, 4],
        )

        result = ray_intersection(ray, obstacles, boundaries)
        individual_results.append(result)

    individual_time = time.time() - start_time

    # Verify results match
    matches = True
    for i, (batch_result, individual_result) in enumerate(
        zip(detailed_results, individual_results)
    ):
        if batch_result.intersects != individual_result.intersects:
            matches = False
            break
        if batch_result.intersects and individual_result.intersects:
            if abs(batch_result.t - individual_result.t) > 1e-6:
                matches = False
                break

    print(f"Detailed batch vs Individual results match: {matches}")
    print(f"Batch time: {batch_time:.4f}s, Individual time: {individual_time:.4f}s")
    print(f"Speedup: {individual_time/batch_time:.2f}x")

    # Analyze detailed results
    print(f"\nDetailed Results Analysis:")
    print(f"Total rays: {num_rays}")

    hits = sum(1 for result in detailed_results if result.intersects)
    obstacle_hits = sum(
        1
        for result in detailed_results
        if result.intersects and result.intersecting_with == "obstacle"
    )
    boundary_hits = sum(
        1
        for result in detailed_results
        if result.intersects and result.intersecting_with == "boundary"
    )

    print(f"Total hits: {hits}/{num_rays} ({hits/num_rays*100:.1f}%)")
    print(f"Obstacle hits: {obstacle_hits} ({obstacle_hits/num_rays*100:.1f}%)")
    print(f"Boundary hits: {boundary_hits} ({boundary_hits/num_rays*100:.1f}%)")
    print(f"Misses: {num_rays - hits} ({(num_rays - hits)/num_rays*100:.1f}%)")

    # Print detailed intersection data
    print(f"\nDetailed Intersection Data:")
    print(f"{'Ray':<4} {'Hit':<5} {'Type':<9} {'Distance':<8} {'Point':<20}")
    print("-" * 50)

    for i, result in enumerate(detailed_results):
        if result.intersects:
            hit_type = result.intersecting_with or "unknown"
            point_str = f"({result.intersection.x:.2f}, {result.intersection.y:.2f})"
            print(f"{i:<4} {'Yes':<5} {hit_type:<9} {result.t:<8.3f} {point_str:<20}")
        else:
            print(f"{i:<4} {'No':<5} {'-':<9} {'-':<8} {'-':<20}")

    # Visualize results
    fig, ax = setup_plot("Detailed Batch Ray Intersection", xlim=(0, 4), ylim=(0, 3))

    # Plot boundaries
    boundary_labeled = False
    for boundary in boundaries:
        boundary_obj = PolygonBoundary(boundary)
        for wall in boundary_obj.walls:
            p1, p2 = wall
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                color="black",
                linewidth=3,
                alpha=0.8,
                label="Boundary" if not boundary_labeled else "",
            )
            boundary_labeled = True

    # Plot obstacles
    for obstacle in obstacles:
        shape = obstacle.get_current_state()
        if isinstance(shape, Circle):
            circle_patch = patches.Circle(
                (shape.center.x, shape.center.y),
                shape.radius,
                facecolor="red",
                edgecolor="darkred",
                alpha=0.4,
                linewidth=2,
            )
            ax.add_patch(circle_patch)
        elif isinstance(shape, Rectangle):
            rect = patches.Rectangle(
                (-shape.width / 2, -shape.height / 2),
                shape.width,
                shape.height,
                facecolor="green",
                edgecolor="darkgreen",
                alpha=0.4,
                linewidth=2,
            )
            transform = (
                plt.matplotlib.transforms.Affine2D()
                .rotate_deg(shape.rotation)
                .translate(shape.center.x, shape.center.y)
                + ax.transData
            )
            rect.set_transform(transform)
            ax.add_patch(rect)

    # Plot agent
    ax.scatter(
        agent_pos.x,
        agent_pos.y,
        color="blue",
        s=100,
        marker="o",
        edgecolors="darkblue",
        linewidth=2,
        label="Agent",
        zorder=10,
    )

    # Plot rays with different colors based on what they hit
    obstacle_hit_labeled = False
    boundary_hit_labeled = False
    miss_labeled = False

    for i, result in enumerate(detailed_results):
        ray_start = rays[i, :2]
        ray_dir = rays[i, 2:4]

        if result.intersects:
            # Ray hits something - color by object type
            hit_point = np.array([result.intersection.x, result.intersection.y])

            if result.intersecting_with == "obstacle":
                ray_color = "crimson"
                hit_color = "darkred"
                marker = "X"
                marker_size = 40
                ray_width = 2.5
                ray_alpha = 0.8
                label = "Obstacle Hit" if not obstacle_hit_labeled else ""
                obstacle_hit_labeled = True
            else:  # boundary
                ray_color = "darkorange"
                hit_color = "chocolate"
                marker = "s"
                marker_size = 35
                ray_width = 2.5
                ray_alpha = 0.8
                label = "Boundary Hit" if not boundary_hit_labeled else ""
                boundary_hit_labeled = True

            # Draw ray line with distinct color
            ax.plot(
                [ray_start[0], hit_point[0]],
                [ray_start[1], hit_point[1]],
                color=ray_color,
                alpha=ray_alpha,
                linewidth=ray_width,
                label=label,
                zorder=5,
            )

            # Draw intersection marker
            ax.scatter(
                hit_point[0],
                hit_point[1],
                color=hit_color,
                s=marker_size,
                marker=marker,
                linewidth=2,
                edgecolors="white",
                zorder=8,
            )
        else:
            # Ray doesn't hit anything - use distinct miss color
            end_point = ray_start + ray_dir * rays[i, 4]
            ax.plot(
                [ray_start[0], end_point[0]],
                [ray_start[1], end_point[1]],
                color="lightsteelblue",
                alpha=0.6,
                linewidth=1.5,
                linestyle="--",
                label="Miss" if not miss_labeled else "",
                zorder=3,
            )
            miss_labeled = True

    ax.legend()
    plt.show()

    return detailed_results


def run_all_batch_tests():
    """Run all batch test functions"""
    print("=" * 60)
    print("RUNNING ALL BATCH RAY INTERSECTION TESTS")
    print("=" * 60)

    batch_test_functions = [
        test_batch_ray_circle_intersection,
        test_batch_ray_rectangle_intersection,
        test_batch_ray_line_intersection,
        test_batch_lidar_simulation,
        test_batch_ray_intersection_full,
        test_batch_ray_intersection_detailed,
        test_batch_performance_comparison,
    ]

    for test_func in batch_test_functions:
        print(f"\n{'-' * 50}")
        try:
            test_func()
        except Exception as e:
            print(f"Test {test_func.__name__} failed: {e}")
            import traceback

            traceback.print_exc()
        print(f"{'-' * 50}")

    print("\nAll batch tests completed!")


def run_all_tests():
    """Run all test functions"""
    print("=" * 50)
    print("RUNNING ALL RAY INTERSECTION TESTS")
    print("=" * 50)

    test_functions = [
        test_ray_circle_intersection_hit,
        test_ray_circle_intersection_miss,
        test_ray_rectangle_intersection_hit,
        test_ray_rectangle_intersection_miss,
        test_ray_line_intersection_hit,
        test_ray_line_intersection_miss,
        test_ray_boundary_intersection,
        test_multiple_intersections,
        test_ray_intersection_with_length,
        test_ray_length_vs_intersections,
    ]

    for test_func in test_functions:
        print(f"\n{'-' * 30}")
        test_func()
        print(f"{'-' * 30}")

    print("\nAll individual tests completed!")

    # Run batch tests
    print("\n" + "=" * 50)
    print("RUNNING BATCH TESTS")
    print("=" * 50)
    run_all_batch_tests()


if __name__ == "__main__":
    # Example usage - you can run individual tests or all tests
    print("Ray Intersection Test Module")
    print("Available test functions:")
    print("- Individual ray tests:")
    print("  - test_ray_circle_intersection_hit()")
    print("  - test_ray_circle_intersection_miss()")
    print("  - test_ray_rectangle_intersection_hit()")
    print("  - test_ray_rectangle_intersection_miss()")
    print("  - test_ray_line_intersection_hit()")
    print("  - test_ray_line_intersection_miss()")
    print("  - test_ray_boundary_intersection()")
    print("  - test_multiple_intersections()")
    print("  - test_ray_intersection_with_length()")
    print("  - test_ray_length_vs_intersections()")
    print("- Batch ray tests:")
    print("  - test_batch_ray_circle_intersection()")
    print("  - test_batch_ray_rectangle_intersection()")
    print("  - test_batch_ray_line_intersection()")
    print("  - test_batch_lidar_simulation()")
    print("  - test_batch_ray_intersection_full()")
    print("  - test_batch_ray_intersection_detailed()")
    print("  - test_batch_performance_comparison()")
    print("- Combined tests:")
    print("  - run_all_tests() - runs both individual and batch tests")
    print("  - run_all_batch_tests() - runs only batch tests")
    print("\nRun any function individually or call run_all_tests() to run everything")

    # Quick demo test
    print("\n" + "=" * 40)
    print("QUICK DEMO - Testing batch ray-circle intersection")
    print("=" * 40)

    # Simple batch test without plotting
    rays = np.array(
        [
            [0.0, 0.0, 1.0, 0.0, 2.0],  # [origin_x, origin_y, dir_x, dir_y, length]
            [0.0, 0.5, 1.0, 0.5, 2.0],
            [0.0, 1.0, 1.0, 0.0, 2.0],
        ]
    )

    circles = np.array(
        [[1.0, 0.0, 0.2], [1.5, 0.8, 0.15]]  # [center_x, center_y, radius]
    )

    result = batch_ray_circle_intersection(rays, circles)
    print(f"✅ Batch intersection result shape: {result.shape}")
    print(f"Intersection distances:\n{result}")

    print(
        "\nTo see visual plots and detailed comparisons, run individual test functions!"
    )
    print("Example: test_batch_ray_circle_intersection()")
    print("For comprehensive testing: run_all_batch_tests()")

    # Uncomment to run batch tests automatically
    # run_all_batch_tests()
    test_batch_lidar_simulation()
