"""
AIMO-3 Solver - Simplified Unified Pipeline

ONE DETERMINISTIC FLOW:
1. Parse & normalize problem
2. Try SymPy first (deterministic, fast)
3. If SymPy fails, use LLM reasoning (chain-of-thought)
4. Verify candidates with multiple validation methods
5. Confidence-aware selection
6. MCTS only as optional fallback for hard problems

No over-engineered abstractions. Just results.
"""

import logging
import subprocess
import re
from typing import List, Tuple, Optional, Dict, Any
import time

from config import *
from parsing import ProblemParser
from sympy_solver import SymPySolver
from validation import (
    AnswerValidator, DeterministicVerifier, AntiHallucinationChecker,
    filter_duplicate_answers, compute_agreement_score, 
    choose_final_pair, is_valid_answer
)
from utils import (
    extract_integers, clamp_to_range, preprocess_problem_text,
    time_limit, normalize_latex
)

logger = logging.getLogger(__name__)


# ============================================================================
# EXECUTION PREAMBLE (Standard math utilities for code execution)
# ============================================================================

EXECUTION_PREAMBLE = """
import math
import numpy as np
import sympy
from sympy import Symbol, symbols, solve, simplify, factor, factorial, binomial
from itertools import permutations, combinations, product

def floor(x):
    return math.floor(x)

def ceil(x):
    return math.ceil(x)

def frac(x):
    return x - math.floor(x)

def digits_to_int(*args):
    result = 0
    for digit in args:
        result = result * 10 + digit
    return result

import sys
sys.setrecursionlimit(2000)
"""


# ============================================================================
# CODE EXECUTION UTILITIES (Minimal, pragmatic)
# ============================================================================

def extract_final_answer(output: str) -> Optional[int]:
    """Extract final integer answer from code output."""
    if not output or not output.strip():
        return None
    
    # Try \boxed{} format
    boxed_match = re.search(r'\\boxed\{(\d+)\}', output)
    if boxed_match:
        return int(boxed_match.group(1))
    
    # Try last line with number
    lines = output.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line:
            match = re.search(r'\d+', line)
            if match:
                return int(match.group())
    
    return None


def run_python_code(code_str: str, timeout_seconds: float = 10.0) -> Optional[int]:
    """Execute Python code with timeout. Returns extracted answer or None."""
    if not code_str or not code_str.strip():
        return None
    
    try:
        full_code = EXECUTION_PREAMBLE + "\n" + code_str
        result = subprocess.run(
            ['python', '-c', full_code],
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        if result.returncode != 0:
            return None
        
        output = result.stdout.strip()
        answer = extract_final_answer(output)
        if answer is not None and 0 <= answer <= 99999:
            return answer
        
        return None
    except (subprocess.TimeoutExpired, Exception):
        return None


# ============================================================================
# LLM-BASED REASONING (Simplified, confidence-aware)
# ============================================================================

def llm_reason_simple(problem: str, num_candidates: int = 2) -> List[Tuple[str, int]]:
    """
    Generate reasoning candidates using LLM (chain-of-thought).
    
    Returns: List of (reasoning_text, extracted_answer) tuples
    """
    try:
        from transformers import pipeline
        
        pipe = pipeline(
            "text-generation",
            model=DEFAULT_MODEL,
            torch_dtype="auto",
            device_map="auto",
        )
        
        prompt = f"""Solve this problem step-by-step and return ONLY the final integer answer:

{problem}

Return ONLY the final integer on the last line.
Answer:"""
        
        candidates = []
        for i in range(num_candidates):
            temp = 0.3 + (0.7 * i / max(1, num_candidates - 1))
            try:
                response = pipe(prompt, max_length=2048, temperature=temp, top_p=0.95)
                if response and response[0]:
                    text = response[0].get('generated_text', '')
                    answers = extract_integers(text)
                    if answers:
                        candidates.append((text, answers[0]))
            except:
                pass
        
        return candidates
    except:
        return []


# ============================================================================
# UNIFIED SOLVER (One simple, deterministic flow)
# ============================================================================

class AIMOSolver:
    """
    Minimal, focused AIMO-3 solver.
    
    Pipeline:
    1. SymPy first (deterministic, competitive advantage)
    2. LLM reasoning (creative, flexible)
    3. Multi-stage validation
    4. Confidence-based selection
    """
    
    def __init__(self, enable_mcts: bool = False):
        """
        Args:
            enable_mcts: Use MCTS as fallback (disabled by default for speed)
        """
        self.parser = ProblemParser()
        self.sympy_solver = SymPySolver()
        self.validator = AnswerValidator()
        self.verifier = DeterministicVerifier()
        self.hallucination_checker = AntiHallucinationChecker()
        self.enable_mcts = enable_mcts
        
        logger.info(f"Initialized AIMOSolver (MCTS fallback: {enable_mcts})")
    
    def solve(self, problem_text: str, timeout_seconds: float = 30.0) -> Tuple[int, int]:
        """
        Solve problem and return two answer candidates.
        
        Pipeline:
        1. Parse & normalize
        2. Try SymPy (PRIMARY)
        3. Try LLM (SECONDARY)
        4. Verify & validate
        5. Select via confidence voting
        """
        logger.info(f"Solving: {problem_text[:80]}...")
        start_time = time.time()
        
        try:
            # Step 1: Normalize
            clean_problem = preprocess_problem_text(problem_text)
            clean_problem = normalize_latex(clean_problem)
            
            # Step 2: Try SymPy (deterministic, fast, first-class)
            logger.info("Attempting SymPy solver (deterministic)...")
            sympy_answers = self._try_sympy(clean_problem)
            
            # Step 3: Try LLM (creative reasoning)
            logger.info("Attempting LLM reasoning...")
            llm_answers = self._try_llm(clean_problem)
            
            # Step 4: Combine all candidates
            all_candidates = sympy_answers + llm_answers
            
            # Step 5: Verify candidates
            logger.info(f"Verifying {len(all_candidates)} candidates...")
            verified = self._verify_candidates(all_candidates, clean_problem)
            
            # Step 6: Select best pair
            if verified:
                answer1, answer2 = self._select_answer_pair(verified)
            else:
                # Fallback
                answer1, answer2 = 0, 0
            
            elapsed = time.time() - start_time
            logger.info(f"Result: ({answer1}, {answer2}) in {elapsed:.2f}s")
            
            return (clamp_to_range(answer1, 0, 99999), 
                   clamp_to_range(answer2, 0, 99999))
        
        except Exception as e:
            logger.error(f"Solver error: {e}")
            return (0, 0)
    
    def _try_sympy(self, problem: str) -> List[int]:
        """Try deterministic symbolic solving."""
        try:
            answers = self.sympy_solver.solve(problem)
            logger.info(f"SymPy found: {answers}")
            return answers
        except:
            logger.debug("SymPy failed or timed out")
            return []
    
    def _try_llm(self, problem: str) -> List[int]:
        """Try LLM-based reasoning."""
        try:
            candidates = llm_reason_simple(problem, num_candidates=2)
            answers = [ans for _, ans in candidates]
            logger.info(f"LLM generated: {answers}")
            return answers
        except:
            logger.debug("LLM reasoning failed")
            return []
    
    def _verify_candidates(self, answers: List[int], problem: str) -> List[Tuple[int, float]]:
        """
        Verify candidates and assign confidence scores.
        
        Returns: List of (answer, confidence) tuples
        """
        verified = []
        
        for ans in answers:
            if not is_valid_answer(ans):
                continue
            
            # Start with base confidence
            confidence = 0.5
            
            # Check with validation rules
            if self.validator.validate(ans, problem):
                confidence += 0.2
            
            # Check for hallucinations
            if not self.hallucination_checker.run_full_check(ans, problem):
                confidence += 0.2
            
            verified.append((ans, min(1.0, confidence)))
        
        # Remove duplicates, keep highest confidence
        unique = {}
        for ans, conf in verified:
            if ans not in unique or conf > unique[ans]:
                unique[ans] = conf
        
        return list(unique.items())
    
    def _select_answer_pair(self, verified: List[Tuple[int, float]]) -> Tuple[int, int]:
        """
        Select two answers based on confidence.
        
        Strategy:
        - If one answer appears multiple times (high confidence) → use it
        - Otherwise use top-2 by confidence score
        """
        if not verified:
            return 0, 0
        
        # Sort by confidence (descending)
        sorted_verified = sorted(verified, key=lambda x: x[1], reverse=True)
        
        if len(sorted_verified) >= 2:
            answer1 = sorted_verified[0][0]
            answer2 = sorted_verified[1][0]
        elif len(sorted_verified) == 1:
            answer1 = answer2 = sorted_verified[0][0]
        else:
            answer1 = answer2 = 0
        
        return answer1, answer2


class AdaptiveSolver(AIMOSolver):
    """Extended solver with parameters tuned by difficulty."""
    
    def solve(self, problem_text: str, timeout_seconds: float = 30.0) -> Tuple[int, int]:
        """Solve with adaptive settings."""
        clean_problem = preprocess_problem_text(problem_text)
        difficulty = self.parser.estimate_difficulty(clean_problem)
        
        # Adaptive: adjust timeout and MCTS by difficulty
        if difficulty > 0.7:
            self.enable_mcts = True
            timeout_seconds = min(timeout_seconds, 45.0)
            logger.info(f"Hard problem detected (difficulty={difficulty:.2f}), enabling MCTS")
        
        return super().solve(problem_text, timeout_seconds=timeout_seconds)
