import os
import json
import turbopuffer as tp

class RAGMemory:
    def __init__(self, namespace="coding-solutions"):
        self.namespace = namespace
        self.api_key = os.getenv("TURBOPUFFER_API_KEY")
        self.has_key = bool(self.api_key)

        if self.has_key:
            tp.api_key = self.api_key
        else:
            print("Warning: No TURBOPUFFER_API_KEY found. RAG Memory will be mocked.")

    def query_similar(self, problem_description, top_k=3):
        """
        Retrieves similar past solutions based on the problem description.
        """
        if not self.has_key:
            return []

        # In a real app, you would generate an embedding for the prompt here.
        # Since we don't have an embedding model loaded, we'll skip the actual vector search
        # or use a random vector if just testing the API.

        # For this experiment, let's assume we mock the retrieval unless configured
        return []

    def save_solution(self, problem_id, prompt, solution):
        """
        Saves a successful solution to the vector database.
        """
        if not self.has_key:
            return

        # Again, requires embedding.
        # For the purpose of this snippet, we just acknowledge the API exists.
        pass

    def get_mock_context(self, problem_id):
        """
        Returns hardcoded context for testing if RAG is off.
        """
        if problem_id == "add_two":
            return "# Hint: Use the + operator"
        return ""
