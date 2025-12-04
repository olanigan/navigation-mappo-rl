# Tutorial 2: Environment Design

In this tutorial, we'll dive into the design of our custom RL environment. The environment defines the world in which our agents live, learn, and interact.

## Observation Space

The observation space defines what the agent sees. It's the information the agent uses to make decisions. In our environment, the observation space is a combination of:

*   **LIDAR Scans:** We use LIDAR to give the agent a sense of its surroundings. We shoot out rays in all directions and record what they hit. This gives the agent a "scan" of its local area.

    ```
        ^
       /|\
      / | \
     /  |  \
    <---|--->
     \  |  /
      \ | /
       \|/
        v
    ```

*   **Agent State:** We also provide the agent with information about itself, such as:
    *   Time spent in the environment
    *   Current velocity
    *   Distance to the goal
    *   Position and angle to the goal

All of these values are normalized between -1 and 1 to make them easier for the neural network to process.

## Action Space

The action space defines what the agent can do. In our case, the action space is a 2D vector representing the agent's target velocity.

```
      [-1, 1]
         ^
         |
-1 <---- O ----> 1
         |
         v
      [1, -1]
```

The agent outputs an `(x, y)` velocity, and the environment applies it in the agent's local coordinate space.

### Local vs. Global Space

A key design decision is to use a **local coordinate space**. This means that all observations and actions are relative to the agent itself. The agent is always at the center of its own world, and the y-axis is aligned with its goal.

This helps with generalization. The agent's policy doesn't depend on its absolute position in the environment, so it can apply what it has learned in one area to any other area.

## Reward Design

The reward function is crucial for teaching the agent what to do. We use a combination of sparse and dense rewards:

*   **Sparse Rewards:**
    *   `+1` for reaching the goal (success)
    *   `-1` for colliding with an obstacle (failure)

*   **Dense Rewards:**
    *   A small positive reward for making progress towards the goal.
    *   A small negative reward for moving away from the goal.

This combination of rewards provides a clear success/failure signal while also giving the agent a hint in the right direction at every step.

## Generalization

To ensure our agents learn a general policy that works in any environment, we randomize the environment at the beginning of each episode:

*   Random starting locations for the agents.
*   Random goal locations.
*   Randomly generated obstacles.

This forces the agent to learn general principles of navigation rather than just memorizing a single map.

---
[**Previous: Tutorial 1: Introduction**](01_introduction.md) | [**Next: Tutorial 3: Agent Architecture**](03_agent_architecture.md)
