"""
AIMO-3 Submission Entry Point - Minimal Kaggle Interface

This is the entry point for Kaggle evaluation.
It loads a problem, calls the solver, and outputs the answer pair.

Format: (answer1, answer2) for double-run evaluation
Scoring: 1.0 (both correct), 0.5 (one correct), 0.0 (both wrong)
"""

import logging
import sys
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

logger = logging.getLogger(__name__)

try:
    from solver import solve_problem
except ImportError as e:
    logger.error(f"Failed to import solver: {e}")
    raise


def solve(problem_text: str) -> Tuple[int, int]:
    """
    Kaggle entry point: solve problem and return two answers.
    
    Args:
        problem_text: Raw problem statement
        
    Returns:
        (answer1, answer2) - integer tuple for submission
    """
    try:
        logger.info("Calling clean solver pipeline...")
        answer1, answer2 = solve_problem(problem_text)
        logger.info(f"Answer: ({answer1}, {answer2})")
        return (answer1, answer2)
    except Exception as e:
        logger.error(f"Error solving: {e}", exc_info=True)
        return (0, 0)


if __name__ == "__main__":
    # Local testing
    logger.info("AIMO-3 Submission Module - call solve(problem_text)")
