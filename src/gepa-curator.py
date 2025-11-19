"""
Sophisticated: Curator + GEPA (Enhanced Version)
Generates math problems using a GEPA-optimized system prompt with better configuration
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


# Create more diverse training and validation sets
trainset = [
    {"input": {"difficulty": "intermediate", "topic": "algebra", "problem_type": "word problem"}},
    {"input": {"difficulty": "advanced", "topic": "calculus", "problem_type": "optimization"}},
    {"input": {"difficulty": "intermediate", "topic": "probability", "problem_type": "application problem"}},
    {"input": {"difficulty": "advanced", "topic": "linear algebra", "problem_type": "theoretical problem"}},
    {"input": {"difficulty": "beginner", "topic": "geometry", "problem_type": "calculation"}},
    {"input": {"difficulty": "expert", "topic": "combinatorics", "problem_type": "logic puzzle"}},
    {"input": {"difficulty": "intermediate", "topic": "trigonometry", "problem_type": "application problem"}},
    {"input": {"difficulty": "advanced", "topic": "differential equations", "problem_type": "calculation"}},
]

valset = [
    {"input": {"difficulty": "advanced", "topic": "geometry", "problem_type": "proof"}},
    {"input": {"difficulty": "intermediate", "topic": "statistics", "problem_type": "application problem"}},
    {"input": {"difficulty": "expert", "topic": "number theory", "problem_type": "theoretical problem"}},
    {"input": {"difficulty": "beginner", "topic": "algebra", "problem_type": "calculation"}},
]

seed_prompt = {
    "system_prompt": (
        "You are a precise and creative math problem generator. You MUST respond in the following Pydantic format:\n"
        "{\n"
        '  "problem": "Clear problem statement",\n'
        '  "solution": "Step-by-step solution",\n'
        '  "final_answer": "Final numerical or symbolic answer",\n'
        '  "problem_type": "Type of math problem",\n'
        '  "difficulty": "Difficulty level",\n'
        '  "topic": "Main math topic",\n'
        '  "subtopics": ["List", "of", "subtopics"]\n'
        "}\n\n"
        "Generate problems that require multi-step reasoning, incorporate real-world contexts where possible, "
        "and connect multiple mathematical concepts. Provide a clear step-by-step solution with explanations, "
        "a final numerical answer, and specify the problem type and subtopics involved."
    )
}

print("Running GEPA optimization with enhanced configuration...")


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
                f"Generate a sophisticated {input_dict.get('difficulty', '')} level math problem "
                f"primarily about {input_dict.get('topic', '')} but potentially incorporating other mathematical concepts. "
                f"The problem should be of type '{input_dict.get('problem_type', '')}' and involve real-world context when possible. "
                f"Provide a clear step-by-step solution with explanations and the final numerical answer. "
                f"Also specify the main problem type and any subtopics involved."
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

            # Calculate quality score with more nuanced criteria for sophisticated problems
            text = assistant_response.lower()
            score = 0.0

            # 1. Problem sophistication (0.25 points)
            # Check for multi-step reasoning, real-world context, interdisciplinary elements
            sophistication_indicators = [
                r"real.*world",
                r"application",
                r"context",
                r"model.*situation",
                r"scenario",
                r"word problem",
                r"find.*given",
            ]
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in sophistication_indicators):
                score += 0.15
            # Check for complex problem structure
            if re.search(r"[a-z]\s*[=+\-*/]\s*[a-z]|[a-z]\s*[=+\-*/]\s*\d+", text):
                score += 0.10

            # 2. Solution depth and reasoning (0.3 points)
            # Look for detailed explanations and multiple approaches
            step_patterns = [
                r"step\s*\d+",
                r"\d+\.\s+[^.]{20,}",  # Numbered list with substantial content
                r"first.*then|initially.*finally",
                r"solution:",
                r"explanation:",
            ]
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in step_patterns):
                # Check if steps have mathematical content
                if re.search(r"(?<!\d)\d+(?!\d)|[a-z]\s*[=+\-*/]", text):
                    score += 0.3

            # 3. Problem statement quality (0.2 points)
            # Check for clear problem formulation
            if "?" in text or re.search(r"(solve|find|calculate|determine|prove)\s+", text):
                if re.search(r"\d+|[a-z]\s*[=+\-*/]", text):
                    score += 0.2

            # 4. Final answer clarity (0.15 points)
            if "final answer" in text or "answer:" in text or re.search(r"###\s*\S", assistant_response):
                score += 0.15
            elif "=" in text and re.search(r"\d+|[a-z]+", text):
                score += 0.08

            # 5. Subtopic identification (0.1 points) - bonus for recognizing multiple concepts
            # This is harder to evaluate from text alone, but look for indicators of multiple math areas
            multiple_concept_indicators = [
                "and.*algebra", "and.*geometry", "and.*calculus",
                "statistics.*probability", "algebra.*geometry", "trig.*geometry",
                "applied.*math", "multi.*step", "combined.*concepts"
            ]
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in multiple_concept_indicators):
                score += 0.1

            # Penalty: If response is too short or seems incomplete
            if len(assistant_response) < 80:
                score *= 0.6  # Reduce score for very short responses
            elif len(assistant_response) < 120:
                score *= 0.8  # Reduce score for short responses

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

            # Create feedback based on score (updated thresholds for sophisticated scoring)
            if score >= 0.8:
                feedback = (
                    "The generated response is high quality. It includes a sophisticated problem statement with "
                    "real-world context, multi-step reasoning, well-structured solution with explanations, "
                    "and properly formatted final answer. The problem connects multiple mathematical concepts."
                )
            elif score >= 0.6:
                feedback = (
                    "The generated response is moderate to good quality. It includes most required elements "
                    "but could be improved. Ensure it has: (1) a sophisticated problem statement with real-world context, "
                    "(2) detailed step-by-step solution with mathematical operations and explanations, "
                    "(3) clearly formatted final answer. Consider connecting multiple mathematical concepts."
                )
            else:
                feedback = (
                    "The generated response is low quality. It is missing key elements or lacks proper sophistication. "
                    "The response must include: (1) a sophisticated problem statement with multi-step reasoning and "
                    "real-world context, (2) a detailed solution with numbered steps and mathematical operations, "
                    "(3) a clearly formatted final answer. The problem should connect multiple mathematical concepts "
                    "and demonstrate complex reasoning."
                )

            record: dict[str, Any] = {
                "Inputs": f"Difficulty: {input_dict.get('difficulty', '')}, Topic: {input_dict.get('topic', '')}, Problem Type: {input_dict.get('problem_type', '')}",
                "Generated Outputs": generated_outputs,
                "Feedback": feedback,
            }

            items.append(record)

        ret_d[comp] = items

        if len(items) == 0:
            raise Exception("No valid predictions found for any module.")

        return ret_d


# Enhanced GEPA configuration with better parameters
print("Running GEPA optimization with enhanced configuration...")

gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm=None,
    reflection_lm="openai/gpt-4o-mini",
    adapter=MathQualityAdapter(model="openai/gpt-4o-mini"),
    max_metric_calls=25,  # Increased for better optimization
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
            f"Task: Generate a sophisticated {input['difficulty']} level math problem "
            f"primarily about {input['topic']} but potentially incorporating other mathematical concepts. "
            f"The problem should be of type '{input['problem_type']}' and involve real-world context when possible. "
            f"Provide a clear step-by-step solution with explanations and the final numerical answer. "
            f"Also specify the problem type ({input['problem_type']}), difficulty ({input['difficulty']}), "
            f"main topic ({input['topic']}), and any subtopics involved as part of your response. "
            f"Format your response with clear sections for the problem statement, solution steps, and final answer."
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
        }


# Use the same enhanced lists as the baseline
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
    "differential equations"
]

problem_types = [
    "word problem",
    "proof",
    "calculation",
    "optimization",
    "logic puzzle",
    "application problem",
    "theoretical problem"
]

inputs: List[Dict] = [
    {
        "difficulty": random.choice(difficulties),
        "topic": random.choice(topics),
        "problem_type": random.choice(problem_types)
    }
    for _ in range(50)  # Match the baseline sample size
]


generator = MathProblemGeneratorGEPA(
    optimized_prompt=optimized_prompt, model_name="gpt-4o-mini", batch=False
)

print("Generating sophisticated GEPA-optimized math problems...")
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
        if hasattr(val, 'item'):  # numpy scalars
            record[col] = val.item()
        elif isinstance(val, (list, tuple)):
            # Handle lists/tuples with potential numpy elements
            record[col] = [v.item() if hasattr(v, 'item') else v for v in val]
        else:
            record[col] = val
    output_data.append(record)

# Save to results folder
current_dir = Path(__file__).parent.parent  # Go up to project root
results_dir = current_dir / "results"
results_dir.mkdir(exist_ok=True)  # Create results folder if it doesn't exist
output_path = results_dir / "math_dataset_gepa.json"

with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2, default=str)  # Use default=str as backup

print(f"\nSaved dataset to: {output_path}")
print(f"Generated {len(output_data)} sophisticated problems.")
print(json.dumps(output_data[0], indent=2))
