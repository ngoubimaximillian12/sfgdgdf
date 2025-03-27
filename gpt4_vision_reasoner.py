import openai
import numpy as np
import os

# GPT-4 Vision Reasoning Client
class GPT4VisionReasoner:
    def __init__(self, api_key):
        openai.api_key = api_key

    def grid_to_ascii(self, grid):
        return '\n'.join(' '.join(str(cell) for cell in row) for row in grid)

    def build_prompt(self, train_pairs, test_input):
        prompt = """
You are an expert AI in solving abstract reasoning tasks involving colored grids. Each grid is a 2D matrix with integers from 0 to 9, representing different colors.
Below are examples of input/output grid pairs. Your task is to understand the transformation logic and apply it to the test input to produce the correct output.
"""
        for i, pair in enumerate(train_pairs):
            prompt += f"\nTrain Pair {i+1}:\nInput:\n{self.grid_to_ascii(pair['input'])}\nOutput:\n{self.grid_to_ascii(pair['output'])}\n"
        prompt += f"\nTest Input:\n{self.grid_to_ascii(test_input)}\n"
        prompt += "\nPlease generate the Output grid and explain your reasoning step by step."
        return prompt

    def ask_gpt4(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return response['choices'][0]['message']['content']

    def extract_grid(self, response_text):
        lines = response_text.strip().split('\n')
        grid_lines = []
        for line in lines:
            if all(char in "0123456789 " for char in line.strip()) and any(c.isdigit() for c in line):
                grid_lines.append([int(x) for x in line.strip().split()])
        return grid_lines

    def solve(self, train_pairs, test_input):
        prompt = self.build_prompt(train_pairs, test_input)
        response = self.ask_gpt4(prompt)
        predicted_grid = self.extract_grid(response)
        return predicted_grid, response

# Example usage:
# api_key = os.getenv("OPENAI_API_KEY")
# reasoner = GPT4VisionReasoner(api_key)
# output_grid, explanation = reasoner.solve(train_pairs, test_input)
