"""
Baseline: Curator without GEPA optimization
Generates math problem-solution pairs
"""

import json
import random
from pathlib import Path
from typing import Dict, List

from bespokelabs import curator
from pydantic import BaseModel, Field


class MathProblem(BaseModel):
    problem: str = Field(description="A math problem")
    solution: str = Field(description="Step-by-step solution")
    final_answer: str = Field(description="Final numerical answer")


class MathProblemGenerator(curator.LLM):
    response_format = MathProblem
    system_prompt = (
        "You are a precise math problem generator. "
        "Always return one well-structured problem with a correct solution."
    )

    def prompt(self, input: Dict) -> str:
        return (
            f"Generate one {input['difficulty']} level math problem "
            f"about {input['topic']}. "
            f"Provide a clear step-by-step solution and the final numerical answer."
        )

    def parse(self, input: Dict, response: MathProblem) -> Dict:
        return {
            "difficulty": input["difficulty"],
            "topic": input["topic"],
            "problem": response.problem,
            "solution": response.solution,
            "final_answer": response.final_answer,
        }


difficulties = ["easy", "medium", "hard"]
topics = [
    "algebra",
    "geometry",
    "calculus",
    "probability",
    "trigonometry",
    "number theory",
    "linear algebra",
]

inputs: List[Dict] = []

# uniformly generate samples
for _ in range(5):
    inputs.append(
        {"difficulty": random.choice(difficulties), "topic": random.choice(topics)}
    )


generator = MathProblemGenerator(model_name="gpt-4o-mini", batch=False)

print("Generating math problems...")
dataset = generator(inputs)

output_data = dataset.dataset.to_pandas().to_dict("records")

# Save to results folder
current_dir = Path(__file__).parent.parent  # Go up to project root
results_dir = current_dir / "results"
results_dir.mkdir(exist_ok=True)  # Create results folder if it doesn't exist
output_path = results_dir / "math_dataset.json"

with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nSaved dataset to: {output_path}")
print(f"Generated {len(output_data)} problems\n")

print("Sample output:")
print(json.dumps(output_data[0], indent=2))
