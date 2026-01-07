import json
import time
from .problems import PROBLEMS
from .memory import RAGMemory
from .agent import Agent

def run_experiment(num_episodes=5, output_file="coding_rl/experiment_data.jsonl"):
    print(f"Starting Coding RL Experiment...")

    memory = RAGMemory()
    agent = Agent()

    data_log = []

    for episode in range(num_episodes):
        # 1. Sample a problem
        problem = PROBLEMS[episode % len(PROBLEMS)]
        print(f"\n--- Episode {episode+1}: {problem.id} ---")

        # 2. Retrieve Context (Memory)
        context = memory.query_similar(problem.prompt)
        if not context and not memory.has_key:
            context = memory.get_mock_context(problem.id)

        print(f"Context Retrieved: {context}")

        # 3. Agent Action (Generate Code)
        code = agent.generate_code(context, problem.prompt)
        print(f"Generated Code:\n{code}")

        # 4. Environment Feedback (Run Tests)
        success, message = problem.check_solution(code)
        reward = 1.0 if success else 0.0
        print(f"Result: {'PASS' if success else 'FAIL'} (Reward: {reward}) - {message}")

        # 5. Log Data (Trajectory)
        # This format is compatible with OpenPipe's fine-tuning
        log_entry = {
            "messages": [
                {"role": "system", "content": "You are a python coding assistant."},
                {"role": "user", "content": problem.prompt},
                {"role": "assistant", "content": code}
            ],
            "metadata": {
                "reward": reward,
                "problem_id": problem.id,
                "context_used": bool(context)
            }
        }
        data_log.append(log_entry)

        # 6. Update Memory (if successful)
        if success:
            memory.save_solution(problem.id, problem.prompt, code)

    # Save to JSONL for OpenPipe
    with open(output_file, "w") as f:
        for entry in data_log:
            f.write(json.dumps(entry) + "\n")

    print(f"\nExperiment finished. Data saved to {output_file}")
    print("To train on this data with OpenPipe, upload this file to https://app.openpipe.ai/")

if __name__ == "__main__":
    run_experiment()
