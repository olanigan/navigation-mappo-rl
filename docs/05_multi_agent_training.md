# Tutorial 5: Multi-Agent Training

Now that we have a capable single agent, let's scale up to a multi-agent setting. We'll explore two approaches: Independent PPO (IPPO) and Multi-Agent PPO (MAPPO).

## Independent PPO (IPPO)

The simplest way to handle multiple agents is to treat them as independent entities. In IPPO, each agent is a single-player agent trying to maximize its own rewards. Other agents are simply treated as moving obstacles.

To make this efficient, we use **parameter sharing**. Instead of training a separate network for each agent, we train a single "brain" that controls all of them. This is possible because all agents have the same goals and physical bodies, and they are all trained in their own local coordinate space.

```
+-----------+   +-----------+   +-----------+
|  Agent 1  |   |  Agent 2  |   |  Agent 3  |
+-----------+   +-----------+   +-----------+
      |               |               |
      +-------+-------+-------+-------+
              |               |
              v               v
      +---------------+ +----------------+
      | Single Actor  | | Single Critic  |
      +---------------+ +----------------+
```

### Issues with IPPO

1.  **No Cooperation:** Agents learn to act selfishly, which can lead to suboptimal group behavior.
2.  **Non-Stationary Environment:** The environment is constantly changing as other agents learn and adapt. This makes it difficult for the critic to converge.

## Multi-Agent PPO (MAPPO)

MAPPO addresses the issues of IPPO by using a paradigm called **Centralized Training, Decentralized Execution (CTDE)**.

*   **Centralized Training:** During training, the critic has access to global information about the environment.
*   **Decentralized Execution:** During execution, each agent acts independently based on its local observations.

### The Centralized Critic

The key difference in MAPPO is the critic. Instead of just seeing the local observations of a single agent, the critic receives the **global state** of the environment. This includes the positions, velocities, and goals of all agents.

```
+-------------+
| Global State|
+-------------+
        |
        v
+---------------------+
| Centralized Critic  |
+---------------------+
        |
        v
+---------------------+
|  Global Value       |
+---------------------+
```

By seeing the whole picture, the critic can learn to evaluate the actions of the entire group and guide the actors towards cooperative behavior.

### The Football Team Analogy

Think of a football team.

*   **Training (Centralized):** The coach has a full view of the field and can give instructions to the entire team.
*   **Game Time (Decentralized):** The players are on their own and have to make decisions based on their own skills and what they see.

MAPPO works in the same way. The centralized critic is the coach, and the decentralized actors are the players.

## Emergent Behavior

With MAPPO, we can see cooperative behaviors emerge naturally. For example, in a hallway environment, agents will learn to move to one side to let others pass, even though we haven't explicitly programmed them to do so. This is the power of centralized training!

---
[**Previous: Tutorial 4: Training a Single Agent**](04_training_single_agent.md) | [**Back to Table of Contents**](README.md)
