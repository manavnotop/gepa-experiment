"""
Evaluation: Compare baseline vs GEPA datasets using an LLM grader
Supports sophisticated dataset format with problem types and subtopics
"""

import json
import statistics
from pathlib import Path

from bespokelabs import curator
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# 1. Define evaluation schema
# ---------------------------------------------------------
class EvaluationResult(BaseModel):
    score: float = Field(description="Overall quality score 0-1")
    reasoning_score: float
    structure_score: float
    final_answer_score: float
    sophistication_score: float
    feedback: str


# ---------------------------------------------------------
# 2. Define the evaluator LLM
# ---------------------------------------------------------
class MathEvaluator(curator.LLM):
    response_format = EvaluationResult
    system_prompt = (
        "You are a strict math dataset evaluator for sophisticated mathematical problems. "
        "Evaluate the problem based on its depth, real-world context, connection to multiple mathematical concepts, "
        "solution clarity, and final answer accuracy. Provide numerical scores for each aspect."
    )

    def prompt(self, sample):
        # Handle both old and new format datasets
        problem_type = sample.get("problem_type", "calculation")
        subtopics = sample.get("subtopics", [sample.get("topic", "unknown")])
        difficulty = sample.get("difficulty", "unknown")
        topic = sample.get("topic", "unknown")

        return f"""
Evaluate the following generated math example:

Difficulty: {difficulty}
Topic: {topic}
Problem Type: {problem_type}
Subtopics: {', '.join(subtopics)}

Problem:
{sample["problem"]}

Solution:
{sample["solution"]}

Final Answer:
{sample["final_answer"]}

Score each STRICTLY between 0 and 1:

- reasoning_score: depth, clarity, correctness of mathematical steps
- structure_score: formatting, step-by-step flow, readability
- final_answer_score: correctness & presence of final answer
- sophistication_score: complexity, real-world context, multi-concept integration

Then output:
- score = average of four scores
- feedback = one paragraph explaining strengths & weaknesses
        """

    def parse(self, sample, response: EvaluationResult):
        return response.model_dump()


# ---------------------------------------------------------
# 3. Load datasets
# ---------------------------------------------------------
root = Path(__file__).parent.parent / "results"

baseline_path = root / "math_dataset.json"
gepa_path = root / "math_dataset_gepa.json"

baseline_data = json.load(open(baseline_path))
gepa_data = json.load(open(gepa_path))


# ---------------------------------------------------------
# 4. Run evaluation
# ---------------------------------------------------------
evaluator = MathEvaluator(model_name="gpt-4o-mini", batch=False)

print("Evaluating baseline dataset...")
baseline_scores = evaluator(baseline_data).dataset.to_pandas().to_dict("records")

print("Evaluating GEPA dataset...")
gepa_scores = evaluator(gepa_data).dataset.to_pandas().to_dict("records")


# ---------------------------------------------------------
# 5. Save results
# ---------------------------------------------------------
with open(root / "baseline_eval.json", "w") as f:
    json.dump(baseline_scores, f, indent=2)

with open(root / "gepa_eval.json", "w") as f:
    json.dump(gepa_scores, f, indent=2)


# ---------------------------------------------------------
# 6. Summaries
# ---------------------------------------------------------
def stats(scores, key="score"):
    arr = [s[key] for s in scores]
    return {
        "mean": statistics.mean(arr),
        "median": statistics.median(arr),
        "std": statistics.stdev(arr) if len(arr) > 1 else 0,
    }


print("\n=== RESULTS SUMMARY ===")
print("Baseline:", stats(baseline_scores))
print("GEPA:", stats(gepa_scores))
