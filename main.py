"""
Kaggle API entry point for AIMO-3 competition.
"""

import logging
from typing import Tuple
from solver import solve_problem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AIMO3Solver:
    """Kaggle-compatible solver interface."""
    
    def __init__(self, timeout_seconds: int = 30):
        """
        Initialize solver.
        
        Args:
            timeout_seconds: Time limit per problem
        """
        self.timeout = timeout_seconds
        logger.info(f"AIMO3Solver initialized (timeout={timeout_seconds}s)")
    
    def solve(self, problem_text: str, timeout_seconds: int = None) -> Tuple[int, int]:
        """
        Solve a single problem.
        
        Args:
            problem_text: LaTeX problem statement
            timeout_seconds: Optional override for timeout
        
        Returns:
            (answer1, answer2) tuple
        """
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout
        
        logger.info("="*60)
        logger.info("Solving problem")
        logger.info("="*60)
        
        result = solve_problem(problem_text, timeout)
        
        logger.info(f"Final result: {result}")
        logger.info("="*60)
        
        return result


# For direct invocation
if __name__ == "__main__":
    # Example usage
    solver = AIMO3Solver(timeout_seconds=30)
    
    test_problem = """
    Find the smallest positive integer $n$ such that $n^2 + n + 1$ is divisible by 7.
    """
    
    answer = solver.solve(test_problem)
    print(f"Answer: {answer}")
