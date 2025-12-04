# Tutorial 3: Agent Architecture

In this tutorial, we'll explore the neural network architecture that powers our agents. The agent's "brain" is a neural network that takes in observations and outputs actions.

## Network Architecture

Our network is composed of two main parts:

1.  **A 1D Convolutional Neural Network (CNN):** This part of the network processes the LIDAR scans. CNNs are great at extracting spatial features from data, which is perfect for interpreting the LIDAR information. It gives the agent a "vision embedding" of its local space.

2.  **A Fully Connected Neural Network (MLP):** This part of the network processes the additional agent state information (velocity, distance to goal, etc.).

The outputs of these two networks are then combined and passed through a final set of layers to produce the agent's action.

```
+-----------------+   +------------------+
|   LIDAR Scans   |   |    Agent State   |
+-----------------+   +------------------+
        |                     |
        v                     v
+-----------------+   +------------------+
|     1D CNN      |   |       MLP        |
+-----------------+   +------------------+
        |                     |
        +---------+-----------+
                  |
                  v
          +------------------+
          |  Combined Network  |
          +------------------+
                  |
                  v
          +------------------+
          |      Action      |
          +------------------+
```

## Stacking Observations

To help the agent understand the dynamic nature of the environment, we stack the last four observations together. This gives the network a history of frames, allowing it to perceive motion and make decisions based on how the environment is changing over time.

---
[**Previous: Tutorial 2: Environment Design**](02_environment_design.md) | [**Next: Tutorial 4: Training a Single Agent**](04_training_single_agent.md)
