# BespokeLab: Math Problem Generation with GEPA Optimization

A research project comparing baseline math problem generation against GEPA-optimized prompts using the BespokeLabs Curator framework.

## Overview

This project generates sophisticated mathematical problems using LLMs and evaluates two approaches:
- **Baseline**: Direct problem generation with a fixed system prompt
- **GEPA**: Problem generation with a GEPA-optimized system prompt

## Results

### Evaluation Summary

| Method | Mean Score | Median Score | Std Dev |
|--------|-----------|--------------|---------|
| Baseline | 0.8675 | 0.9000 | 0.0956 |
| GEPA | 0.8545 | 0.8825 | 0.1063 |

**Evaluation Metrics**: Each problem is scored on four dimensions (0-1 scale):
- Reasoning score: Depth, clarity, and correctness of mathematical steps
- Structure score: Formatting, step-by-step flow, readability
- Final answer score: Correctness and presence of final answer
- Sophistication score: Complexity, real-world context, multi-concept integration

## Project Structure

```
bespokeslab/
├── src/
│   ├── curator.py          # Baseline problem generation
│   ├── gepa-curator.py     # GEPA-optimized problem generation
│   └── evals.py            # Evaluation framework
├── results/
│   ├── math_dataset.json           # Baseline dataset
│   ├── math_dataset_gepa.json     # GEPA dataset
│   ├── baseline_eval.json         # Baseline evaluation results
│   └── gepa_eval.json             # GEPA evaluation results
└── main.py
```

## Usage

### Generate Datasets

```bash
# Generate baseline dataset
uv run python src/curator.py

# Generate GEPA-optimized dataset
uv run python src/gepa-curator.py
```

### Run Evaluations

```bash
uv run python src/evals.py
```

## Future Improvements

While GEPA shows promise, there are several avenues to improve performance beyond the baseline:

### Option 1: Increase Search Steps
Increase `max_metric_calls` to allow GEPA more exploration:
- Set `max_metric_calls=40` (often beats baseline)
- Or `max_metric_calls=80` for more thorough optimization

### Option 2: Improve the Metric
The current evaluation metric is good but fairly rigid. Potential improvements:
- Simplify and stabilize scoring mechanisms
- Reduce variance in evaluation criteria
- Fine-tune weightings of different score components

### Option 3: Broaden Training Set
Use ~20 diverse training examples from multiple domains:
- Algebra
- Geometry
- Probability
- Calculus
- Number theory
- Word problems
- Optimization problems
- Multi-step derivations

More diverse training data helps GEPA find stronger, more generalizable prompts.

### Option 4: Expand Optimization Scope
Currently, GEPA optimizes only the `system_prompt`. Allow optimization of additional components:
- Instruction style
- Step-by-step hints
- Output structuring
- Problem formatting guidelines

Expanding the search space increases the potential for finding better configurations.

## Dependencies

- `bespokelabs-curator>=0.1.26` - Dataset generation framework
- `gepa>=0.0.22` - Prompt optimization
- `pydantic>=2.12.4` - Data validation

## License

[Add license information]

