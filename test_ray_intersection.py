import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
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
        test_ray_intersection_with_length,  # Updated name
        test_ray_length_vs_intersections,  # New test
    ]

    for test_func in test_functions:
        print(f"\n{'-' * 30}")
        test_func()
        print(f"{'-' * 30}")

    print("\nAll tests completed!")


if __name__ == "__main__":
    # Example usage - you can run individual tests or all tests
    print("Ray Intersection Test Module")
    print("Available test functions:")
    print("- test_ray_circle_intersection_hit()")
    print("- test_ray_circle_intersection_miss()")
    print("- test_ray_rectangle_intersection_hit()")
    print("- test_ray_rectangle_intersection_miss()")
    print("- test_ray_line_intersection_hit()")
    print("- test_ray_line_intersection_miss()")
    print("- test_ray_boundary_intersection()")
    print("- test_multiple_intersections()")
    print("- test_ray_intersection_with_length()")  # Updated name
    print("- test_ray_length_vs_intersections()")  # New test
    print("- run_all_tests()")
    print("\nRun any function individually or call run_all_tests() to run everything")

    # Quick demo test
    print("\n" + "=" * 40)
    print("QUICK DEMO - Testing ray-circle intersection")
    print("=" * 40)

    # Simple test without plotting
    ray = Ray(origin=Vector2(x=0, y=0), direction=Vector2(x=1, y=0), length=2.0)
    circle = Circle(center=Vector2(x=1, y=0), radius=0.5)
    result = ray_circle_intersection(ray, circle)

    if result.intersects:
        print(
            f"✅ SUCCESS: Ray intersects circle at ({result.intersection.x:.3f}, {result.intersection.y:.3f})"
        )
        print(f"Distance along ray: {result.t:.3f}")
        print(f"Ray length: {ray.length}")
    else:
        print("❌ No intersection found")

    print("\nTo see visual plots, run individual test functions!")
    print("Example: test_ray_circle_intersection_hit()")

    # Uncomment to run all tests automatically
    # run_all_tests()
    # test_ray_circle_intersection_hit()
    test_ray_intersection_with_length()
