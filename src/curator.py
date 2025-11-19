"""
Baseline: Curator without GEPA optimization
Generates math problem-solution pairs with enhanced sophistication
"""

import json
import random
from pathlib import Path
from typing import Dict, List

from bespokelabs import curator
from pydantic import BaseModel, Field


class MathProblem(BaseModel):
    problem: str = Field(description="A 3-hop word problem")
    solution: str = Field(description="Step-by-step solution")
    final_answer: str
    reference: str = Field(
        description="A real URL/DOI/ISBN that supports the key numeric fact"
    )


class MathProblemGenerator(curator.LLM):
    response_format = MathProblem
    system_prompt = (
        "You are a picky math professor who hates textbook fluff. "
        "Every problem you create MUST chain three distinct numeric facts and cite the exact "
        "textbook/paper URL/DOI/ISBN that contains the key number. "
        "If you cannot locate a real reference, return nothing."
    )

    def prompt(self, input: Dict) -> str:
        return (
            f"Create one {input['difficulty']} {input['problem_type']} that "
            f"combines at least THREE separate numeric ideas from {input['topic']}. "
            f"State the problem, solve it step-by-step, give the final number, "
            f"and supply the real reference that holds the critical value. "
            f"No made-up numbers, no fake URLs."
        )

    def parse(self, input: Dict, response: MathProblem) -> Dict:
        return {
            "difficulty": input["difficulty"],
            "topic": input["topic"],
            "subtopics": [input["topic"]],  # Default to main topic as subtopic
            "problem_type": input["problem_type"],
            "problem": response.problem,
            "solution": response.solution,
            "final_answer": response.final_answer,
            "reference": response.reference,
        }


# Enhanced difficulty and topic lists
difficulties = ["beginner", "intermediate", "advanced", "expert"]
topics = [
    "algebra",
    "geometry",
    "calculus",
    "probability",
    "trigonometry",
    "number theory",
    "linear algebra",
    "statistics",
    "combinatorics",
    "differential equations",
]

problem_types = [
    "word problem",
    "proof",
    "calculation",
    "optimization",
    "logic puzzle",
    "application problem",
    "theoretical problem",
]

inputs: List[Dict] = []

# Generate more diverse and sophisticated samples
for _ in range(100):  # Increase sample size
    inputs.append(
        {
            "difficulty": random.choice(difficulties),
            "topic": random.choice(topics),
            "problem_type": random.choice(problem_types),
        }
    )


generator = MathProblemGenerator(model_name="gpt-4o-mini", batch=False)

print("Generating sophisticated math problems...")
dataset = generator(inputs)

# Convert to pandas and then to JSON-serializable format
df = dataset.dataset.to_pandas()

# Convert any non-serializable objects to strings or regular Python types
output_data = []
for _, row in df.iterrows():
    record = {}
    for col in df.columns:
        val = row[col]
        # Convert numpy types and other non-serializable types to regular Python types
        if hasattr(val, "item"):  # numpy scalars
            record[col] = val.item()
        elif isinstance(val, (list, tuple)):
            # Handle lists/tuples with potential numpy elements
            record[col] = [v.item() if hasattr(v, "item") else v for v in val]
        else:
            record[col] = val
    output_data.append(record)

# Save to results folder
current_dir = Path(__file__).parent.parent  # Go up to project root
results_dir = current_dir / "results"
results_dir.mkdir(exist_ok=True)  # Create results folder if it doesn't exist
output_path = results_dir / "math_dataset.json"

with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2, default=str)  # Use default=str as backup

print(f"\nSaved dataset to: {output_path}")
print(f"Generated {len(output_data)} sophisticated problems\n")

print("Sample output:")
print(json.dumps(output_data[0], indent=2))
