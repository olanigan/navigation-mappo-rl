# Coding RL Experiment

This directory contains a "Tiny RL" experiment for Python coding, designed to work with **OpenPipe** (for policy optimization) and **turbopuffer** (for retrieval memory).

## Structure

*   `problems.py`: Defines a set of simple coding tasks (Add, Multiply, Reverse String) with unit tests.
*   `agent.py`: A wrapper around LLM APIs. It defaults to a "Mock Mode" that solves the toy problems heuristically if no API keys are present.
*   `memory.py`: A wrapper around `turbopuffer` for RAG. Defaults to mock if no key.
*   `run_experiment.py`: The main RL loop (Retrieve -> Act -> Reward -> Log).

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install turbopuffer-client openpipe openai
    ```

2.  **Set API Keys (Optional):**
    To use real LLMs and Vector DBs, set these environment variables:
    ```bash
    export OPENAI_API_KEY="sk-..."
    # OR
    export OPENPIPE_API_KEY="op-..."

    export TURBOPUFFER_API_KEY="tp-..."
    ```
    *If you don't set these, the experiment runs in "Mock Mode" using hardcoded logic.*

3.  **Run the Experiment:**
    ```bash
    python -m coding_rl.run_experiment
    ```

## The RL Workflow

1.  **Data Collection (Rollout):** The script runs the agent against the problems. It saves the trajectories (Prompt, Code, Reward) to `experiment_data.jsonl`.
2.  **Training (OpenPipe):**
    *   Upload `experiment_data.jsonl` to OpenPipe.
    *   Filter for rows where `metadata.reward == 1.0`.
    *   Fine-tune a model on this high-reward data (Expert Iteration).
3.  **Memory (Turbopuffer):**
    *   Successful solutions are (theoretically) embedded and stored in Turbopuffer.
    *   Future runs query this DB to find similar solved problems.
