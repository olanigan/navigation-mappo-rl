import os
import random
from openai import OpenAI

class Agent:
    def __init__(self, model="gpt-3.5-turbo"):
        self.client = None
        self.model = model

        # Prioritize OpenPipe if available
        openpipe_key = os.getenv("OPENPIPE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if openpipe_key:
            self.client = OpenAI(
                api_key=openpipe_key,
                base_url="https://api.openpipe.ai/api/v1"
            )
        elif openai_key:
            self.client = OpenAI(api_key=openai_key)
        else:
            print("Warning: No API Key found. Agent will be mocked.")

    def generate_code(self, context, prompt):
        if not self.client:
            return self._mock_generate(prompt)

        full_prompt = f"{context}\n\n{prompt}\n\nAnswer only with the python code."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful python coding assistant. Output only valid python code."},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return self._mock_generate(prompt)

    def _mock_generate(self, prompt):
        """
        Simple heuristic solver for the toy problems to ensure the loop works.
        """
        if "add" in prompt:
            return "def add(a, b):\n    return a + b"
        if "multiply" in prompt:
            return "def multiply(a, b):\n    return a * b"
        if "reverse" in prompt:
            return "def reverse_str(s):\n    return s[::-1]"

        return "def func(): pass"
