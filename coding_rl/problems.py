import unittest

class CodingProblem:
    def __init__(self, id, prompt, test_cases):
        self.id = id
        self.prompt = prompt
        self.test_cases = test_cases # List of (input_args, expected_output)

    def check_solution(self, func_impl):
        """
        Executes the provided function implementation string against test cases.
        Returns (success: bool, message: str)
        """
        # Create a clean local scope
        local_scope = {}
        try:
            exec(func_impl, {}, local_scope)
        except Exception as e:
            return False, f"Syntax Error: {e}"

        # Assume the function name is the one requested or just pick the first callable
        func = None
        for name, obj in local_scope.items():
            if callable(obj):
                func = obj
                break

        if not func:
            return False, "No function defined."

        for i, (args, expected) in enumerate(self.test_cases):
            try:
                if isinstance(args, tuple):
                    result = func(*args)
                else:
                    result = func(args)

                if result != expected:
                    return False, f"Test Case {i} Failed: Expected {expected}, Got {result}"
            except Exception as e:
                return False, f"Runtime Error on Test Case {i}: {e}"

        return True, "All tests passed."

# Define a set of toy problems
PROBLEMS = [
    CodingProblem(
        id="add_two",
        prompt="Write a python function `add(a, b)` that returns the sum of a and b.",
        test_cases=[
            ((1, 2), 3),
            ((-1, 1), 0),
            ((10, 10), 20)
        ]
    ),
    CodingProblem(
        id="multiply_two",
        prompt="Write a python function `multiply(a, b)` that returns the product of a and b.",
        test_cases=[
            ((2, 3), 6),
            ((-1, 5), -5),
            ((0, 100), 0)
        ]
    ),
    CodingProblem(
        id="reverse_string",
        prompt="Write a python function `reverse_str(s)` that returns the reversed string.",
        test_cases=[
            ("hello", "olleh"),
            ("abc", "cba"),
            ("", "")
        ]
    )
]
