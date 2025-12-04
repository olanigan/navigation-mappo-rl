# Tutorial 1: Introduction to Reinforcement Learning

Welcome to the Multi-Agent Navigation project! This tutorial series will guide you through the process of building and training autonomous agents to navigate complex environments.

## What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. The goal is to maximize a numerical reward signal. The agent learns from its own experiences, much like a human learns through trial and error.

### The Agent-Environment Loop

The core of RL is the agent-environment loop. Here's how it works:

1.  **Observation:** The agent observes the current state of the environment.
2.  **Action:** Based on the observation, the agent takes an action.
3.  **Reward:** The environment gives the agent a reward (or penalty) for its action.
4.  **New State:** The environment transitions to a new state.

This loop continues until a terminal state is reached.

```
      +-----------------+      +-----------+
      |                 |      |           |
      |   Environment   |----->|   Agent   |
      |                 |      |           |
      +-----------------+      +-----------+
            ^       |
            |       |
          Reward   Action
            |       |
            +-------+
```

### Key Concepts

*   **Agent:** The learner or decision-maker.
*   **Environment:** The world in which the agent operates.
*   **State:** A snapshot of the environment at a particular time.
*   **Action:** A move the agent can make.
*   **Reward:** A feedback signal that tells the agent how well it's doing.
*   **Policy:** The agent's strategy for choosing actions based on states.

## Project Overview

In this project, we'll be working with a custom RL environment where multiple agents learn to navigate obstacle courses. Here's a breakdown of the components:

*   **Environment:** Defines the task, physics, and simulation of the world.
*   **Agent:** Makes decisions in the environment based on its observations.

Our goal is to train these agents not only to reach their goals but also to cooperate and avoid collisions with each other.

---
[**Next: Tutorial 2: Environment Design**](02_environment_design.md)
