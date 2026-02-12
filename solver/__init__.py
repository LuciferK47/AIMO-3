"""
AIMO-3 Solver - Single Dominant Pipeline

Architecture:
- LLM proposes theorems
- Python verifies theorems
- Only verified answers survive
"""

from .solve import solve_problem

__all__ = ['solve_problem']
