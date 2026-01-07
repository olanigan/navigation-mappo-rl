# Adapting Multi-Agent Navigation for Coding Tasks

## Research Report

This document outlines the feasibility of extending or reusing the current Multi-Agent Navigation RL setup for "coding" (software engineering, code generation, or program synthesis) tasks.

### Executive Summary

- **Direct Reuse:** **Low**. The current environment and network architecture are highly specialized for continuous 2D navigation with LiDAR sensors. They cannot be directly used for coding tasks without major modification.
- **Algorithm Reuse:** **High**. The underlying MAPPO (Multi-Agent Proximal Policy Optimization) algorithm and the training loop infrastructure are general-purpose and can be adapted to coding tasks if the environment and networks are replaced.
- **Extensibility:** The codebase is clean and modular, making it a good template for a new RL project, but it would effectively require rewriting the `nav/` module and the `networks/` module.

---

### Gap Analysis

| Feature | Current Navigation Setup | Required for Coding Tasks |
| :--- | :--- | :--- |
| **Environment** | Continuous 2D Physics (Position, Velocity, Collisions). | Discrete/Symbolic (Text editors, ASTs, Compilers). |
| **Action Space** | Continuous Vectors `Box(2,)` (Velocity X, Y). | Discrete Tokens (Vocab size) or Discrete Actions (Insert, Delete). |
| **Observation Space** | LiDAR (Distance rays) + State Vector (Speed, Goal). | Text Sequences, Abstract Syntax Trees (AST), or Execution Traces. |
| **Network Architecture** | `ObservationEncoder` (1D CNN + MLP) -> `DecentralizedActor` (DiagGaussian). | Transformers (Attention), LSTMs, or GNNs -> Categorical Distribution. |

### Feasibility of Extension

#### 1. Reusing the RL Algorithm (MAPPO)
The `rl/mappo.py` file contains a standard implementation of PPO for multi-agent settings. This logic is domain-agnostic.
- **Reusable:** `MAPPO.learn()`, `MAPPO.compute_loss()`, `RolloutBuffer` (mostly).
- **Needs Modification:**
    - `DecentralizedActorNetwork`: Currently outputs `DiagGaussianDistribution` (for continuous actions). Must be changed to output `CategoricalDistribution` (for selecting tokens/actions).
    - `ObservationEncoder`: Currently tailored for `LiDAR`. Must be replaced with an embedding layer + Transformer/LSTM to process code/text.

#### 2. Reusing the Environment
The `nav/` module is entirely specific to 2D geometry and physics.
- **Reuse:** None. You cannot "navigate" through code using velocity vectors.
- **Action:** You would need to implement a new `pettingzoo.ParallelEnv` class that:
    - Defines `step(action)`: Takes a token/edit and applies it to the code state.
    - Defines `reward`: +1 for passing tests, -1 for syntax errors, etc.

#### 3. "Coding" as Scripting Agents
If "coding" refers to **programming the behavior of agents** (rather than the agents writing code), the project is highly extensible:
- **Configuration:** You can "code" new scenarios by creating YAML files in `configs/`.
- **Logic:** You can subclass `Agent` or `Obstacle` in `nav/` to implement rule-based behaviors alongside RL agents.

### Adaptation Strategy

To adapt this codebase for Code Generation:

1.  **Create a New Environment (`coding/environment.py`)**:
    -   Implement the `gymnasium.Env` or `pettingzoo.ParallelEnv` interface.
    -   State: The current code buffer.
    -   Action: Select a token from a vocabulary.
    -   Reward: feedback from a compiler or unit tests.

2.  **Modify the Network (`networks/transformer_policy.py`)**:
    -   Create a new `TransformerActor` and `TransformerCritic`.
    -   Input: Token IDs.
    -   Output: Logits over vocabulary.

3.  **Update `MAPPO` Class**:
    -   Allow swapping the `actor_network` class to use the new Discrete Policy.
    -   Update `compute_loss` to handle categorical cross-entropy instead of Gaussian log-likelihood.

### Conclusion

While the *framework* (RL loop) is sound, the *application* (Navigation) is too specific to be "extended" to Coding. It is better to view this codebase as a **reference implementation** of MAPPO. You would essentially be building a new project using the `rl/` folder as a library, while discarding `nav/`.
