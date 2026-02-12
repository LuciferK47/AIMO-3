# AIMO-3 Solver - Clean Architecture

## Philosophy

**LLM proposes theorems. Python verifies theorems. Only verified answers survive.**

## Structure

```
AIMC-3/
├── solver/
│   ├── parser.py      # LaTeX → normalized text
│   ├── domain.py      # Domain classification (lightweight)
│   ├── prompt.py      # Single canonical prompt
│   ├── llm.py         # LLM interface (deterministic)
│   ├── verifier.py    # Symbolic & numeric verification (AUTHORITATIVE)
│   ├── arbiter.py     # Answer selection (verification-based, no voting)
│   └── solve.py       # Main orchestrator
│
├── main.py            # Kaggle API entry point
└── requirements_clean.txt
```

## Pipeline

1. **Normalize** problem text (LaTeX → clean text)
2. **Classify** domain (algebra, number_theory, combinatorics, geometry)
3. **LLM reasoning** (constrained, deterministic)
4. **Verify** proposed answer (hard constraints + symbolic checks)
5. **Return** verified answer or (0, 0)

## Key Principles

- **Single dominant pipeline** (no parallel solvers, no voting)
- **Deterministic execution** (temperature=0.0, fixed seed)
- **Verification dominance** (Python checks, not LLM self-verification)
- **Strict output format** (retries only on format violation or verification failure)
- **Minimal dependencies** (Python stdlib + SymPy + OpenAI)

## Usage

```python
from solver import solve_problem

result = solve_problem(problem_text, timeout_seconds=30)
print(f"Answer: {result}")
```

Or use the Kaggle interface:

```python
from main import AIMO3Solver

solver = AIMO3Solver(timeout_seconds=30)
answer = solver.solve(problem_text)
```

## Configuration

Set environment variables:

```bash
export OPENAI_API_KEY="your-api-key"
export LLM_MODEL="gpt-4-turbo"  # Optional, default is gpt-4-turbo
```

## Installation

```bash
pip install -r requirements_clean.txt
```

## Removed Noise

- Multiple competing pipelines
- Ensemble logic / voting
- Confidence scoring
- Temperature sweeps
- Random retries
- Verbose chain-of-thought logging
- Unused dependencies
- Experimental artifacts
