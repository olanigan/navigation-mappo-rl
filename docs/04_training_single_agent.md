# Tutorial 4: Training a Single Agent

Now that we have our environment and agent architecture, it's time to train the agent. We'll be using an algorithm called **Proximal Policy Optimization (PPO)**.

## Actor-Critic Model

PPO uses an **actor-critic** model, which consists of two networks:

*   **Actor:** The actor is the policy network. It takes in the agent's observations and decides which action to take.
*   **Critic:** The critic evaluates the state of the environment. It takes in the agent's observations and outputs a "value" that represents how good that state is.

```
+-------------+
| Observation |
+-------------+
      |
      +-----------------+
      |                 |
      v                 v
+-------+         +--------+
| Actor |         | Critic |
+-------+         +--------+
      |                 |
      v                 v
+--------+        +-------+
| Action |        | Value |
+--------+        +-------+
```

### The Critic as a Coach

Think of the critic as an experienced coach. It looks at the current situation and tells you how "winnable" it is.

*   If the agent is about to collide, the critic should predict a low value.
*   If the agent is about to reach the goal, the critic should predict a high value.

The critic is trained by comparing its predictions to the actual rewards the agent receives.

### The Actor Learning from the Coach

The actor learns from the critic's feedback. It takes an action and then asks the critic, "Did I do better or worse than you expected?" This difference is called the **advantage**.

*   **Positive Advantage:** If the action leads to a better-than-expected outcome, the actor is updated to make that action more likely in the future.
*   **Negative Advantage:** If the action leads to a worse-than-expected outcome, the actor is updated to make that action less likely.

## PPO Training Loop

1.  **Collect Data:** The agent plays in the environment for a while, collecting observations, actions, and rewards.
2.  **Calculate Advantages:** The advantages are calculated for each action taken.
3.  **Update Critic:** The critic is updated to make better predictions.
4.  **Update Actor:** The actor is updated based on the advantages.

This loop is repeated until the agent's performance converges.

---
[**Previous: Tutorial 3: Agent Architecture**](03_agent_architecture.md) | [**Next: Tutorial 5: Multi-Agent Training**](05_multi_agent_training.md)
