"""
Answer arbitration - Verification-based (NO VOTING).
Returns verified answer or (0, 0) if verification fails.
"""

import logging
from typing import Tuple
from .verifier import Verifier

logger = logging.getLogger(__name__)


def select_answer(proposed_answer: int, problem_text: str, constraints: dict, domain: dict) -> Tuple[int, int]:
    """
    Select final answer based on VERIFICATION ONLY (no voting, no confidence).
    
    Args:
        proposed_answer: LLM's proposed answer
        problem_text: Original problem
        constraints: Extracted constraints
        domain: Domain classification
    
    Returns:
        (answer, answer) if verified, else (0, 0)
    """
    verifier = Verifier()
    
    if verifier.verify_answer(proposed_answer, problem_text, constraints, domain):
        logger.info(f"Answer verified: {proposed_answer}")
        return (proposed_answer, proposed_answer)
    else:
        logger.warning(f"Answer failed verification: {proposed_answer}")
        return (0, 0)
