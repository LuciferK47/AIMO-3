"""
Main solver orchestrator - Single dominant pipeline.
"""

import logging
import random
from typing import Tuple

from .parser import normalize_problem, extract_constraints
from .domain import classify_domain
from .prompt import build_prompt, TEMPERATURE, MAX_TOKENS
from .llm import query_llm
from .verifier import parse_llm_output, Verifier
from .arbiter import select_answer

logger = logging.getLogger(__name__)

# Determinism
RANDOM_SEED = 42
MAX_RETRIES = 1  # Retries allowed ONLY on format violation or verification failure


def solve_problem(problem_text: str, timeout_seconds: int = 30) -> Tuple[int, int]:
    """
    Solve an AIMO-3 problem using the single dominant pipeline.
    
    Pipeline:
    1. Normalize problem text
    2. Classify domain (lightweight)
    3. LLM proposes answer (constrained)
    4. Python verifies answer
    5. Return verified answer or (0, 0)
    
    Args:
        problem_text: LaTeX problem statement
        timeout_seconds: Time limit (not strictly enforced here)
    
    Returns:
        (answer, answer) if verified, else (0, 0)
    """
    # Ensure determinism
    random.seed(RANDOM_SEED)
    
    try:
        # Step 1: Normalize
        logger.info("Step 1: Normalizing problem text")
        normalized_text = normalize_problem(problem_text)
        constraints = extract_constraints(normalized_text)
        logger.debug(f"Constraints: {constraints}")
        
        # Step 2: Classify domain
        logger.info("Step 2: Classifying domain")
        domain = classify_domain(normalized_text)
        logger.debug(f"Domain: {domain}")
        
        # Step 3: LLM reasoning (with retry on format violation)
        logger.info("Step 3: LLM constrained reasoning")
        proposed_answer = None
        retry_reason = None
        
        for attempt in range(MAX_RETRIES + 1):
            prompt = build_prompt(normalized_text, retry_reason)
            response = query_llm(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
            
            if not response:
                logger.error("LLM query failed")
                if attempt < MAX_RETRIES:
                    retry_reason = "LLM query failed. Try again."
                    continue
                return (0, 0)
            
            # Parse output
            proposed_answer = parse_llm_output(response)
            
            if proposed_answer is not None:
                logger.info(f"Proposed answer: {proposed_answer}")
                break
            
            # Format violation - retry
            if attempt < MAX_RETRIES:
                logger.warning(f"Format violation (attempt {attempt + 1})")
                retry_reason = "Output format invalid. Follow the format exactly."
            else:
                logger.error("Max retries reached, format still invalid")
                return (0, 0)
        
        if proposed_answer is None:
            return (0, 0)
        
        # Step 4: Verify answer
        logger.info("Step 4: Verifying proposed answer")
        verifier = Verifier()
        verification_passed = verifier.verify_answer(
            proposed_answer, 
            normalized_text, 
            constraints, 
            domain
        )
        
        # Step 5: Select final answer
        if verification_passed:
            logger.info(f"Verification passed: {proposed_answer}")
            return (proposed_answer, proposed_answer)
        else:
            # Retry ONCE on verification failure
            if MAX_RETRIES > 0:
                logger.warning("Verification failed, retrying")
                retry_reason = "Proposed answer failed verification checks. Review constraints."
                prompt = build_prompt(normalized_text, retry_reason)
                response = query_llm(prompt, max_tokens=MAX_TOKENS, temperature=TEMPERATURE)
                
                if response:
                    retry_answer = parse_llm_output(response)
                    if retry_answer is not None:
                        if verifier.verify_answer(retry_answer, normalized_text, constraints, domain):
                            logger.info(f"Retry succeeded: {retry_answer}")
                            return (retry_answer, retry_answer)
            
            logger.warning("Verification failed, returning (0, 0)")
            return (0, 0)
    
    except Exception as e:
        logger.error(f"Solver error: {e}", exc_info=True)
        return (0, 0)
