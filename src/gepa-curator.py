"""
Minimal: Curator + GEPA (Working Version)
Generates math problems using a GEPA-optimized system prompt
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List

import gepa
from bespokelabs import curator
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
from pydantic import BaseModel, Field


class MathProblem(BaseModel):
    problem: str = Field(description="A math problem")
    solution: str = Field(description="Step-by-step solution")
    final_answer: str = Field(description="Final numerical answer")


trainset = [
    {"input": {"difficulty": "high school", "topic": "algebra"}},
    {"input": {"difficulty": "college", "topic": "calculus"}},
    {"input": {"difficulty": "middle school", "topic": "arithmetic"}},
    {"input": {"difficulty": "high school", "topic": "trigonometry"}},
    {"input": {"difficulty": "college", "topic": "linear algebra"}},
]

valset = [
    {"input": {"difficulty": "high school", "topic": "geometry"}},
    {"input": {"difficulty": "college", "topic": "probability"}},
    {"input": {"difficulty": "high school", "topic": "statistics"}},
]

seed_prompt = {
    "system_prompt": (
        "Generate a math problem with:\n"
        "- Clear problem statement\n"
        "- Step-by-step solution\n"
        "- Final answer in the format '### <answer>'"
    )
}

print("Running GEPA optimization (custom metric)...")


class MathQualityAdapter(DefaultAdapter):
    def __init__(self, model: str):
        super().__init__(model=model)

    def evaluate(self, batch, candidate, capture_traces=False):
        """
        Evaluate the candidate program on a batch of inputs.
        Returns EvaluationBatch with outputs, scores, and optionally trajectories.
        """
        from gepa.core.adapter import EvaluationBatch

        # Extract system prompt from candidate
        system_content = candidate.get("system_prompt", "")

        # Prepare messages for each input
        litellm_requests = []
        for data in batch:
            # Convert input dict to user message
            input_dict = data.get("input", {})
            user_content = (
                f"Generate a {input_dict.get('difficulty', '')} level math problem "
                f"about {input_dict.get('topic', '')}."
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]
            litellm_requests.append(messages)

        # Call LLM
        try:
            if isinstance(self.model, str):
                # Suppress Pydantic warnings from litellm responses
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=UserWarning, module="pydantic"
                    )
                    responses = [
                        resp.choices[0].message.content.strip()
                        for resp in self.litellm.batch_completion(
                            model=self.model,
                            messages=litellm_requests,
                            max_workers=self.max_litellm_workers,
                            **self.litellm_batch_completion_kwargs,
                        )
                    ]
            else:
                responses = [self.model(messages) for messages in litellm_requests]
        except Exception:
            # On error, return failure scores
            outputs = [{"full_assistant_response": ""} for _ in batch]
            scores = [self.failure_score] * len(batch)
            trajectories = (
                None
                if not capture_traces
                else [{"data": data, "full_assistant_response": ""} for data in batch]
            )
            return EvaluationBatch(
                outputs=outputs, scores=scores, trajectories=trajectories
            )

        # Score each response
        outputs = []
        scores = []
        trajectories = [] if capture_traces else None

        for data, assistant_response in zip(batch, responses, strict=False):
            output = {"full_assistant_response": assistant_response}
            outputs.append(output)

            # Calculate quality score with more nuanced criteria
            text = assistant_response.lower()
            score = 0.0

            # 1. Problem statement quality (0.3 points)
            # Check for actual problem structure, not just keyword
            if "?" in text or "problem" in text or "solve" in text or "find" in text:
                # Check if it's a real problem (has numbers or variables)
                if re.search(r"\d+|[a-z]\s*[=+\-*/]", text):
                    score += 0.3

            # 2. Step-by-step solution quality (0.4 points)
            # More strict: need actual numbered steps or clear step indicators
            step_patterns = [
                r"step\s*\d+",
                r"\d+\.\s+[^.]{20,}",  # Numbered list with substantial content
                r"first[^.]{10,}second[^.]{10,}",  # Sequential indicators
                r"solution:",
            ]
            if any(re.search(pattern, text) for pattern in step_patterns):
                # Check if steps have mathematical content
                if re.search(r"[=+\-*/]|\d+", text):
                    score += 0.4

            # 3. Final answer quality (0.3 points)
            # Prefer the exact format requested
            if "###" in assistant_response:  # Check original case for ###
                score += 0.3
            elif "final answer" in text or "answer:" in text or "answer is" in text:
                # Partial credit if format is close
                score += 0.15

            # Bonus: Structure and clarity (0.1 points)
            # Check for good formatting/organization
            if len(assistant_response.split("\n")) >= 3:  # Has some structure
                score += 0.1

            # Penalty: If response is too short or seems incomplete
            if len(assistant_response) < 50:
                score *= 0.5  # Halve the score for very short responses

            scores.append(min(score, 1.0))  # Cap at 1.0

            if capture_traces:
                trajectories.append(
                    {"data": data, "full_assistant_response": assistant_response}
                )

        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=trajectories
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch,
        components_to_update: list[str],
    ):
        """
        Build a reflective dataset for instruction refinement.
        """
        from typing import Any

        ret_d: dict[str, list[dict[str, Any]]] = {}

        assert len(components_to_update) == 1
        comp = components_to_update[0]

        trajectories = eval_batch.trajectories
        assert trajectories is not None, (
            "Trajectories are required to build a reflective dataset."
        )

        items: list[dict[str, Any]] = []
        trace_instances = list(
            zip(trajectories, eval_batch.scores, eval_batch.outputs, strict=False)
        )

        for trace_instance in trace_instances:
            traj, score, output = trace_instance
            data = traj["data"]
            generated_outputs = traj["full_assistant_response"]
            input_dict = data.get("input", {})

            # Create feedback based on score (updated thresholds for new scoring)
            if score >= 0.8:
                feedback = (
                    "The generated response is high quality. It includes a clear problem statement with "
                    "mathematical content, well-structured step-by-step solution, and properly formatted final answer."
                )
            elif score >= 0.5:
                feedback = (
                    "The generated response is moderate quality. It includes some required elements "
                    "but could be improved. Ensure it has: (1) a clear problem statement with numbers/variables, "
                    "(2) numbered steps or clear sequential solution steps with mathematical operations, "
                    "(3) final answer in the format '### <answer>'."
                )
            else:
                feedback = (
                    "The generated response is low quality. It is missing key elements or lacks proper structure. "
                    "The response must include: (1) a problem statement with mathematical content (numbers/variables), "
                    "(2) a step-by-step solution with numbered steps or clear sequential indicators, "
                    "(3) a final answer clearly marked with '### <answer>'. The response should also be well-formatted "
                    "with multiple lines for clarity."
                )

            record: dict[str, Any] = {
                "Inputs": f"Difficulty: {input_dict.get('difficulty', '')}, Topic: {input_dict.get('topic', '')}",
                "Generated Outputs": generated_outputs,
                "Feedback": feedback,
            }

            items.append(record)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d


print("Running GEPA optimization...")

gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm=None,
    reflection_lm="openai/gpt-4o-mini",
    adapter=MathQualityAdapter(model="openai/gpt-4o-mini"),
    max_metric_calls=10,
)

optimized_prompt = gepa_result.best_candidate["system_prompt"]

print("GEPA optimization complete.\n")


class MathProblemGeneratorGEPA(curator.LLM):
    response_format = MathProblem

    def __init__(self, optimized_prompt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimized_prompt = optimized_prompt

    def prompt(self, input: Dict) -> str:
        return (
            f"{self.optimized_prompt}\n\n"
            f"Task: Generate a {input['difficulty']} level math problem "
            f"about {input['topic']}."
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

inputs: List[Dict] = [
    {"difficulty": random.choice(difficulties), "topic": random.choice(topics)}
    for _ in range(5)
]


generator = MathProblemGeneratorGEPA(
    optimized_prompt=optimized_prompt, model_name="gpt-4o-mini", batch=False
)

print("Generating GEPA-optimized math problems...")
dataset = generator(inputs)

output_data = dataset.dataset.to_pandas().to_dict("records")

# Save to results folder
current_dir = Path(__file__).parent.parent  # Go up to project root
results_dir = current_dir / "results"
results_dir.mkdir(exist_ok=True)  # Create results folder if it doesn't exist
output_path = results_dir / "math_dataset_gepa.json"

with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"\nSaved dataset to: {output_path}")
print(f"Generated {len(output_data)} problems.")
print(json.dumps(output_data[0], indent=2))
