# Open Source RL Environments for Coding

## Executive Summary

There are several high-quality open-source Reinforcement Learning (RL) environments designed specifically for coding, program synthesis, and software engineering tasks. This report highlights the top candidates that provide standard interfaces (Gymnasium/PettingZoo) and are suitable for integrating into an RL pipeline.

### Top Candidates

1.  **InterCode** (Princeton NLP)
2.  **CodeRL** (Salesforce / Open Source implementations)
3.  **Gym-Coding** (Various community implementations)
4.  **Apps / MBPP RL Environments**

---

## 1. InterCode (Best for Interactive Coding)

**Repository:** [https://github.com/princeton-nlp/intercode](https://github.com/princeton-nlp/intercode)

**InterCode** is a lightweight, flexible framework for designing interactive coding environments. It treats coding as a multi-turn decision-making process, making it perfect for RL.

-   **Features:**
    -   **Gym Interface:** Fully compatible with OpenAI Gym/Gymnasium.
    -   **Languages:** Supports Bash, Python, SQL, and CTF (Capture The Flag).
    -   **Observation:** Returns the stdout/stderr of the executed command, allowing the agent to "see" the result of its code.
    -   **Reward:** Configurable. Can be binary (pass/fail tests) or dense (intermediate outputs).
    -   **Sandboxing:** Uses **Docker** containers for safe execution. This is critical for RL agents that might generate harmful commands (`rm -rf /`).

-   **Why it fits:** It abstracts away the complexity of managing Docker containers and provides a clean `env.step(action)` interface where `action` is a code snippet or shell command.

-   **Installation:**
    ```bash
    pip install intercode-bench
    ```

---

## 2. RLTF (RL for Tools & Feedback) / TextWorld

**Repository:** [https://github.com/microsoft/TextWorld](https://github.com/microsoft/TextWorld) (and related "CodeWorld" extensions)

While primarily for text-adventure games, TextWorld has been adapted for coding tasks where the "state" is the file system and the "actions" are shell commands.

-   **Features:**
    -   Focuses on language understanding and sequential decision making.
    -   Highly customizable "Quest" system (can be adapted to "Fix this bug").
    -   Pure Python (easier to install than Docker-based solutions for simple tests).

---

## 3. RL-for-Program-Synthesis (Community)

**Repository:** [https://github.com/jabhinav/Reinforcement-Learning-for-Program-Synthesis](https://github.com/jabhinav/Reinforcement-Learning-for-Program-Synthesis)

This is a more specific research repo focusing on generating code to match Input/Output examples (similar to the PufferLib experiment we built, but more robust).

-   **Features:**
    -   Implements REINFORCE and other policy gradient methods.
    -   Focuses on the **Karel** domain (educational programming language) and standard Python synthesis.
    -   Good for studying the algorithmic side of RL-for-Code without the overhead of full Linux environments.

---

## 4. Generic "Gym" for LLMs (NVIDIA NeMo)

**Repository:** [https://github.com/NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym)

NVIDIA NeMo provides a set of Gym environments specifically for training Large Language Models (LLMs) via RL (RLHF/RLAIF).

-   **Features:**
    -   Designed for PPO training of LLMs.
    -   Includes text-based environments.
    -   Scalable to large clusters.

---

## Recommendation

For a robust "Coding RL" experiment that goes beyond simple toy problems:

**Use InterCode.**

It is the most mature, provides real sandboxing (Docker), supports multiple languages (Bash/Python/SQL), and uses the standard API you are already familiar with.

### Proposed InterCode Experiment
1.  **Install:** `pip install intercode-bench`
2.  **Task:** Use the `PythonEnv` or `BashEnv`.
3.  **Agent:** Your existing OpenPipe/PufferLib agent can send actions (strings) to this environment.
4.  **Feedback:** The environment returns the execution result, which you feed back into the context.

This replaces our "Mock" environment with a real, safe, interactive OS environment.
