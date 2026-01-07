# PufferLib for Coding RL

## Research Report

This document outlines the role of **PufferLib** in a coding Reinforcement Learning (RL) stack and how it complements the previously discussed tools (OpenPipe and Turbopuffer).

### What is PufferLib?

**PufferLib** is a high-performance library designed to simplify and accelerate RL training, particularly for complex game environments. Its key features include:
-   **Vectorization:** Efficiently runs thousands of environment instances in parallel.
-   **Wrapper/Emulation:** Wraps existing `Gymnasium` or `PettingZoo` environments to make them compatible with standard RL libraries (CleanRL, SB3).
-   **Optimized Data Handling:** Uses shared memory and C-struct-like data formats to minimize overhead when passing observations between the environment (CPU) and the policy (GPU).

### Role in Coding Tasks

While PufferLib is famous for speeding up games (Atari, Neural MMO), its **vectorization** capabilities are highly relevant for RL-based Coding tasks (Program Synthesis).

#### 1. The Bottleneck: Execution Speed
In "Coding RL", the environment step involves:
1.  Agent writes code.
2.  **Environment executes code** (runs unit tests, checks syntax).
3.  Environment returns reward (Pass/Fail).

Step 2 is extremely slow compared to a neural network forward pass. Executing Python scripts, spinning up sandboxes, or compiling code takes milliseconds to seconds. A standard RL loop waiting for one execution at a time will be intolerably slow.

#### 2. The Solution: Massive Parallelism
PufferLib can manage a vector of **hundreds or thousands of coding environments**.
-   It handles the multiprocessing/threading complexity.
-   It ensures that while some environments are blocked on code execution, the GPU is busy processing observations from others.
-   It flattens the complex observation spaces (text/tokens) into efficient tensor formats.

### Integration Architecture

A "High-Performance Coding RL" stack would look like this:

1.  **Environment (PufferLib + Sandbox):**
    -   A `Gymnasium` environment where `step()` executes code in a secure sandbox (e.g., `gVisor` or `docker`).
    -   **PufferLib** wraps this environment, creating a vectorized interface that exposes `num_envs=1024` to the learner.
2.  **Policy (PufferL / CleanRL):**
    -   A Transformer-based Actor-Critic network.
    -   It receives a batch of observations (tokens) from 1024 environments simultaneously.
3.  **Memory (Turbopuffer):**
    -   Used for retrieval (RAG) within the environment or policy.
4.  **Optimization (OpenPipe - Optional):**
    -   You could still use OpenPipe for offline fine-tuning, but PufferLib allows for **Online RL** (PPO) directly on the code execution task because it makes data collection fast enough.

### Why use PufferLib for Coding?
-   **Throughput:** You need millions of samples to learn complex logic. PufferLib enables collecting these samples in hours instead of weeks.
-   **Simplicity:** It standardizes the interface. You write one `CodingEnv`, and PufferLib handles the asynchronous execution logic.

---

## Experiment Plan

We will implement a small `PufferEnv` wrapper around our existing `CodingProblem` logic to demonstrate how to vectorize the "Code Execution" environment.
