"""
GEPA version: only the prompts are swapped to force 3-hop + real reference.
Everything else (adapter, training, saving) stays identical.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict

import gepa
from bespokelabs import curator
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter
from pydantic import BaseModel, Field


# ── 1. Pydantic model (unchanged) ───────────────────────────────────────────
class MathProblem(BaseModel):
    problem: str = Field(description="A math problem")
    solution: str = Field(description="Step-by-step solution")
    final_answer: str = Field(description="Final numerical answer")


# ── 2. Hard task prompts (ONLY THING THAT CHANGED) ───────────────────────────
seed_prompt = {
    "system_prompt": (
        "You are a picky math professor who hates textbook fluff. "
        "Every problem you create MUST chain three distinct numeric facts and cite the exact "
        "textbook/paper URL/DOI/ISBN that contains the key number. "
        "If you cannot locate a real reference, return nothing."
    )
}

# ── 3. Dataset splits (unchanged) ────────────────────────────────────────────
pilot_file = Path(__file__).parent.parent / "results" / "math_dataset.json"
with open(pilot_file) as f:
    pilot = json.load(f)

random.shuffle(pilot)
split = int(0.8 * len(pilot))
trainset = [{"input": s} for s in pilot[:split]]
valset   = [{"input": s} for s in pilot[split:]]


# ── 4. Adapter (unchanged) ───────────────────────────────────────────────────
class MathQualityAdapter(DefaultAdapter):
    def __init__(self, model: str):
        super().__init__(model=model)

    def evaluate(self, batch, candidate, capture_traces=False):
        from gepa.core.adapter import EvaluationBatch

        system_content = candidate.get("system_prompt", "")
        litellm_requests = [
            [
                {"role": "system", "content": system_content},
                {
                    "role": "user",
                    "content": f"Create one {d['input']['difficulty']} {d['input']['problem_type']} about {d['input']['topic']}.",
                },
            ]
            for d in batch
        ]

        import warnings

        import litellm

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
            responses = [
                resp.choices[0].message.content.strip()
                for resp in litellm.batch_completion(
                    model=self.model,
                    messages=litellm_requests,
                    max_workers=self.max_litellm_workers,
                    **self.litellm_batch_completion_kwargs,
                )
            ]

        outputs, scores, trajectories = [], [], []
        for idx, ar in enumerate(responses):
            outputs.append({"full_assistant_response": ar})
            trajectories.append({"data": batch[idx], "full_assistant_response": ar})

            score = 0.0

            # Has step-by-step structure (0-0.4 points)
            steps = len(
                re.findall(
                    r"(step \d+|^\d+\.|firstly|secondly|then|next)", ar, re.I | re.M
                )
            )
            score += min(steps * 0.08, 0.4)

            # Has reasonable length (0.2 points)
            if 200 < len(ar) < 800:
                score += 0.2

            # Has clear final answer (0.2 points)
            if re.search(
                r"(final answer|therefore|thus|answer is)[:\s]+[-\d\.]+", ar, re.I
            ):
                score += 0.2

            # Shows work/reasoning (0.2 points)
            if re.search(r"(because|since|substitute|simplify|solve)", ar, re.I):
                score += 0.2

            scores.append(min(score, 1.0))

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        from typing import Any

        ret_d: dict[str, list[dict[str, Any]]] = {}
        comp = components_to_update[0]
        items = []
        for traj, score in zip(eval_batch.trajectories, eval_batch.scores):
            if score >= 0.8:
                feedback = "Excellent: clear steps, proper length, shows reasoning, has final answer."
            elif score >= 0.5:
                feedback = "Needs improvement: add more step-by-step detail or clearer reasoning."
            else:
                feedback = "Poor: missing structure, too short/long, or lacks clear solution steps."

            items.append(
                {
                    "Inputs": str(traj["data"]["input"]),
                    "Generated Outputs": traj["full_assistant_response"],
                    "Feedback": feedback,
                }
            )
        ret_d[comp] = items
        return ret_d


# ── 5. GEPA optimisation (unchanged config) ───────────────────────────────────
print("Running GEPA optimization...")
gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm=None,
    reflection_lm="openai/gpt-4o-mini",
    adapter=MathQualityAdapter(model="openai/gpt-4o-mini"),
    max_metric_calls=50,
)
optimized_prompt = gepa_result.best_candidate["system_prompt"]
print("GEPA optimisation complete.\n")


# ── 6. Curator generator with optimised prompt (only prompt changed) ───────────
class MathProblemGeneratorGEPA(curator.LLM):
    response_format = MathProblem

    def __init__(self, optimized_prompt: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimized_prompt = optimized_prompt

    def prompt(self, input: Dict) -> str:
        return f"{self.optimized_prompt}\n\nTask: Create one {input['difficulty']} {input['problem_type']} that combines THREE separate numeric ideas from {input['topic']}, solve step-by-step, give final number, and cite the real reference. No fake URLs."

    def parse(self, input: Dict, response: MathProblem) -> Dict:
        return {
            "difficulty": input["difficulty"],
            "topic": input["topic"],
            "subtopics": [input["topic"]],
            "problem_type": input["problem_type"],
            "problem": response.problem,
            "solution": response.solution,
            "final_answer": response.final_answer,
        }


# ── 7. Generate 50 samples and save (unchanged logic) ─────────────────────────
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
inputs = [
    {
        "difficulty": random.choice(difficulties),
        "topic": random.choice(topics),
        "problem_type": random.choice(problem_types),
    }
    for _ in range(100)
]

generator = MathProblemGeneratorGEPA(
    optimized_prompt=optimized_prompt, model_name="gpt-4o-mini", batch=False
)
dataset = generator(inputs)

output_data = dataset.dataset.to_pandas().to_dict(orient="records")
results_dir = Path(__file__).parent.parent / "results"
results_dir.mkdir(exist_ok=True)
output_path = results_dir / "math_dataset_gepa.json"
with open(output_path, "w") as f:
    json.dump(output_data, f, indent=2, default=str)

print(f"Saved {len(output_data)} GEPA-optimised problems to {output_path}")
