"""
AIMO-3 MASTER SOLVER - Single Dominant Pipeline

Target Pipeline:
LaTeX problem
    ↓
Structured normalization
    ↓
Domain classification (lightweight)
    ↓
Constrained LLM derivation
    ↓
Symbolic / arithmetic verification
    ↓
Answer validation (range, invariants)
    ↓
Final integer pair

Key Principles:
- Deterministic execution
- LLM proposes, Python verifies
- No parallel solvers, no voting
- Strict verification gates
- Bounded time and depth
"""

import logging
import re
import time
from typing import List, Tuple, Optional, Dict, Any, Set

from config import *
from utils import preprocess_problem_text, query_llm, ensure_determinism, time_limit
from sympy_solver import SymPySolver, EquationExtractor
from validation import AnswerValidator

logger = logging.getLogger(__name__)


class ProblemClassifier:
    """Lightweight rule-based problem classifier (no ML)."""
    
    @staticmethod
    def classify(problem_text: str) -> Dict[str, Any]:
        """
        Classify problem and return routing hints + extraction metadata.
        
        Returns:
            {
                'problem_type': str,
                'keywords': List[str],
                'allows_symbolic': bool,
                'has_equations': bool,
                'is_modular': bool,
                'is_counting': bool,
                'is_geometry': bool,
                'difficulty_estimate': float,
                'modulo': Optional[int],
                'ranges': Dict[str, tuple],
                'has_constraints': bool,
            }
        """
        text_lower = problem_text.lower()
        
        classification = {
            'problem_type': 'general',
            'keywords': [],
            'allows_symbolic': True,
            'has_equations': False,
            'is_modular': False,
            'is_counting': False,
            'is_geometry': False,
            'difficulty_estimate': 0.5,
            'modulo': ProblemClassifier.extract_modulo(problem_text),
            'ranges': ProblemClassifier.extract_ranges(problem_text),
            'has_constraints': ProblemClassifier.has_constraints(problem_text),
        }
        
        # MULTI-LABEL CLASSIFICATION (not mutually exclusive)
        # Modular Arithmetic
        if any(w in text_lower for w in ['mod', 'remainder', 'divisible', 'congruence', 'modular']):
            classification['is_modular'] = True
            classification['allows_symbolic'] = True
            classification['keywords'].extend(['modular', 'arithmetic'])
            if not classification['problem_type']:
                classification['problem_type'] = 'modular'
        
        # Combinatorics
        if any(w in text_lower for w in ['how many', 'count', 'combinations', 'permutations', 'ways', 'arrangements', 'sequences']):
            classification['is_counting'] = True
            classification['allows_symbolic'] = True
            classification['keywords'].extend(['counting', 'combinatorics'])
            if not classification['problem_type']:
                classification['problem_type'] = 'combinatorics'
        
        # Diophantine / Number Theory
        if any(w in text_lower for w in ['integer solutions', 'find all integers', 'diophantine', 'number of pairs', 'pairs (', 'positive integers']):
            classification['allows_symbolic'] = True
            classification['keywords'].extend(['number_theory', 'diophantine'])
            if not classification['problem_type']:
                classification['problem_type'] = 'diophantine'
        
        # Geometry
        if any(w in text_lower for w in ['triangle', 'circle', 'angle', 'area', 'perimeter', 'distance', 'radius', 'diameter']):
            classification['is_geometry'] = True
            # Geometry without diagrams is harder for SymPy
            if classification['is_geometry'] and not classification['is_modular']:
                classification['allows_symbolic'] = False
            classification['keywords'].extend(['geometry'])
            if not classification['problem_type']:
                classification['problem_type'] = 'geometry'
        
        # Optimization
        if any(w in text_lower for w in ['maximum', 'minimum', 'maximize', 'minimize', 'least', 'greatest', 'optimal']):
            classification['allows_symbolic'] = True
            classification['keywords'].extend(['optimization'])
            if not classification['problem_type']:
                classification['problem_type'] = 'optimization'
        
        # EQUATIONS / ALGEBRA
        if any(w in text_lower for w in ['equation', 'solve', '=', 'satisfy', 'satisfy']):
            classification['has_equations'] = True
        
        # Difficulty estimate based on keywords
        hard_keywords = ['diophantine', 'complex', 'involving', 'system']
        easy_keywords = ['simple', 'basic', 'find']
        
        difficulty = 0.5
        for kw in hard_keywords:
            if kw in text_lower:
                difficulty = min(1.0, difficulty + 0.15)
        for kw in easy_keywords:
            if kw in text_lower:
                difficulty = max(0.0, difficulty - 0.1)
        
        classification['difficulty_estimate'] = difficulty
        
        return classification

    @staticmethod
    def clean_latex(text: str) -> str:
        """Remove or normalize LaTeX markup for LLM processing."""
        if not text:
            return text
        
        try:
            # Convert common math environments to text
            text = re.sub(r'\$\$(.+?)\$\$', r'\1', text, flags=re.DOTALL)
            text = re.sub(r'\$(.+?)\$', r'\1', text)
            text = re.sub(r'\\left\(', '(', text)
            text = re.sub(r'\\right\)', ')', text)
            text = re.sub(r'\\left\[', '[', text)
            text = re.sub(r'\\right\]', ']', text)
            text = re.sub(r'\\left\\{', '{', text)
            text = re.sub(r'\\right\\}', '}', text)
            text = re.sub(r'\\\+', '+', text)
            text = re.sub(r'\\times', '*', text)
            text = re.sub(r'\\cdot', '*', text)
            text = re.sub(r'\\div', '/', text)
            # Iterative fraction replacement to handle nesting
            for _ in range(5):
                new_text = re.sub(r'\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}', r'(\1)/(\2)', text)
                if new_text == text:
                    break
                text = new_text
            text = text.replace('≤', '<=').replace('≥', '>=').replace('≠', '!=').replace('≈', '~=')
            text = re.sub(r'\^', '^', text)
            text = re.sub(r'_', '_', text)
            text = re.sub(r'\\[a-zA-Z]+', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            return text
        except Exception as e:
            logger.warning(f"LaTeX parsing failed, using raw text: {e}")
            return text

    @staticmethod
    def extract_modulo(text: str) -> Optional[int]:
        """Extract modulo value if present."""
        patterns = [
            r'mod(?:ulo)?\s+(\d+)',
            r'%\s*(\d+)',
            r'\(\s*mod\s+(\d+)\s*\)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None

    @staticmethod
    def extract_ranges(text: str) -> Dict[str, tuple]:
        """Extract variable ranges from problem text."""
        ranges = {}
        
        # Patterns: "a ≤ x ≤ b", "a < x ≤ b", "x in [a, b]", "for x = 1, 2, ..., n"
        range_patterns = [
            r'(\d+)\s*(?:≤|<=|<)\s*([a-zA-Z])\s*(?:≤|<=|<)\s*(\d+)',
            r'([a-zA-Z])\s*(?:≤|<=|<)\s*(\d+)',
            r'([a-zA-Z])\s*(?:≥|>=|>)\s*(\d+)',
            r'([a-zA-Z])\s*(?:in|∈)\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]',
            r'for\s+([a-zA-Z])\s*=\s*(\d+)\s*,\s*(\d+)\s*,\s*\.\.\.\s*,\s*(\d+)',
        ]
        
        for pattern in range_patterns:
            for match in re.finditer(pattern, text):
                if len(match.groups()) == 3:
                    if match.group(1).isalpha():
                        var = match.group(1)
                        lower = int(match.group(2))
                        upper = int(match.group(3))
                    else:
                        lower = int(match.group(1))
                        var = match.group(2)
                        upper = int(match.group(3))
                    ranges[var] = (lower, upper)
                elif len(match.groups()) == 2:
                    var = match.group(1)
                    bound = int(match.group(2))
                    if '>=' in match.group(0) or '≥' in match.group(0) or '>' in match.group(0):
                        ranges[var] = (bound, ranges.get(var, (bound, bound))[1])
                    else:
                        ranges[var] = (ranges.get(var, (bound, bound))[0], bound)
                elif len(match.groups()) == 4:
                    var = match.group(1)
                    lower = int(match.group(2))
                    upper = int(match.group(4))
                    ranges[var] = (lower, upper)
        
        return ranges

    @staticmethod
    def has_constraints(text: str) -> bool:
        """Check if problem has explicit constraints."""
        return bool(re.search(r'(where|such that|given|constraint|modulo|mod|\bmod\b)', 
                             text, re.IGNORECASE))


class CandidateGenerator:
    """Generate a single candidate using constrained LLM reasoning."""
    
    def __init__(self, timeout_remaining: float = 20.0):
        self.timeout_remaining = timeout_remaining
        self.candidates = []
    
    def generate(self, problem_text: str, classification: Dict[str, Any], retry_reason: Optional[str] = None) -> List[Tuple[int, int]]:
        """
        Generate a single candidate via constrained LLM reasoning.
        Retries are allowed ONLY on invalid output format.
        """
        candidates: List[Tuple[int, int]] = []
        answer = self._try_constrained_reasoning(problem_text, classification, retry_reason=retry_reason)
        if answer is not None:
            candidates.append((answer, 1.0))
        return candidates

    def _try_constrained_reasoning(self, problem_text: str, classification: Dict[str, Any], retry_reason: Optional[str] = None) -> Optional[int]:
        """Constrained LLM derivation with strict output format parsing."""
        base_prompt = SYSTEM_PROMPT_REASONING.format(problem=problem_text)
        if retry_reason:
            base_prompt = f"{base_prompt}\nRETRY_CAUSE: {retry_reason}\n"

        for attempt in range(MAX_RETRIES_PER_STRATEGY + 1):
            response = query_llm(base_prompt, max_tokens=MAX_TOKENS_REASONING, temperature=LLM_TEMPERATURE_DETERMINISTIC)
            answer = self._parse_reasoning_output(response)
            if answer is not None:
                return answer

            if attempt < MAX_RETRIES_PER_STRATEGY:
                base_prompt = f"{SYSTEM_PROMPT_REASONING.format(problem=problem_text)}\nRETRY_CAUSE: Output format invalid. Fix the format exactly.\n"

        return None

    def _parse_reasoning_output(self, text: Optional[str]) -> Optional[int]:
        """Parse strict DERIVATION/FINAL_EXPRESSION/FINAL_INTEGER output."""
        if not text:
            return None
        if "DERIVATION:" not in text or "FINAL_EXPRESSION:" not in text or "FINAL_INTEGER:" not in text:
            return None
        match = re.search(r"FINAL_INTEGER:\s*(-?\d+)", text)
        if not match:
            return None
        try:
            val = int(match.group(1))
        except ValueError:
            return None
        if not AnswerValidator.check_range(val):
            return None
        return val
    
    


class AnswerArbitrator:
    """Verification-only arbitration (no voting, no confidence averaging)."""
    
    def __init__(self):
        pass
    
    def arbitrate(self, candidates: List[Tuple[int, int]], problem_text: str) -> Tuple[int, int]:
        """
        Verification-only arbitration:
        1. Hard constraints (range, impossibility, parity, divisibility)
        2. Symbolic verification if equations can be extracted
        3. Return (answer, answer) if verified, else (0, 0)
        """
        if not candidates:
            return (0, 0)

        candidate = candidates[0]
        ans = candidate[0] if isinstance(candidate, tuple) and len(candidate) > 0 else None
        if not isinstance(ans, int):
            return (0, 0)

        # Hard constraints
        if not AnswerValidator.check_range(ans):
            return (0, 0)
        if AnswerValidator.is_impossible(ans, problem_text):
            logger.debug(f"Rejecting impossible answer: {ans}")
            return (0, 0)
        if not AnswerValidator.check_parity(ans, problem_text):
            return (0, 0)
        if not AnswerValidator.check_divisibility(ans, problem_text):
            return (0, 0)

        # Symbolic verification if possible
        equations: List[str] = []
        try:
            equations = EquationExtractor().extract_equations(problem_text)
        except Exception:
            equations = []

        if equations:
            candidate_vars = ['x', 'y', 'n', 'k', 'm', 'a', 'b', 'c']
            verified = False
            for eq in equations:
                eq_str = str(eq)
                for var in candidate_vars:
                    if SymPySolver.verify_solution(eq_str, var, ans):
                        verified = True
                        break
                if verified:
                    break

            if not verified:
                solved = SymPySolver.solve_from_equations(equations, problem_text)
                if solved and isinstance(solved, tuple):
                    solved_ans = solved[0]
                    if solved_ans != ans:
                        return (0, 0)
                else:
                    return (0, 0)

        return (ans, ans)
 

class StrategyArbiter:
    """Master orchestrator implementing the target pipeline."""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.start_time = None
        self.time_per_stage = {
            'parse': TIME_BUDGET_PARSE,
            'classify': TIME_BUDGET_CLASSIFY,
            'generate': TIME_BUDGET_GENERATE,
            'verify': TIME_BUDGET_VERIFY,
            'arbitrate': 1.0,
        }
    
    def solve(self, problem_text: str) -> Tuple[int, int]:
        """
        MASTER PIPELINE with per-stage timeout enforcement:
        
        1. Parse & Classify (0.5s budget)
        2. Constrained LLM Reasoning (deterministic)
        3. Verification & Validation
        """
        self.start_time = time.time()
        if USE_FIXED_SEED:
            ensure_determinism(RANDOM_SEED)
        
        try:
            # STAGE 1: PARSE & CLASSIFY (0.5s max)
            logger.info("STAGE 1: Parsing and classifying...")
            try:
                remaining_time = self.timeout - (time.time() - self.start_time)
                if remaining_time <= 0:
                    logger.warning("Timeout before parse/classify")
                    return (0, 0)
                with time_limit(min(remaining_time, self.time_per_stage['parse'])):
                    problem_text = preprocess_problem_text(problem_text)
                    classification = ProblemClassifier.classify(problem_text)
                    logger.info(f"Classification: {classification['problem_type']}")
            except TimeoutError:
                logger.warning("Parse/classify timeout, returning (0, 0)")
                return (0, 0)
            except Exception as e:
                logger.error(f"Parse/classify error: {e}")
                return (0, 0)
            
            # STAGE 2: CONSTRAINED LLM REASONING
            logger.info("STAGE 2: Constrained LLM reasoning...")
            try:
                remaining_time = self.timeout - (time.time() - self.start_time)
                if remaining_time <= 0:
                    logger.warning("Timeout before reasoning")
                    return (0, 0)
                gen_timeout = min(remaining_time, self.time_per_stage['generate'])
                
                with time_limit(gen_timeout):
                    generator = CandidateGenerator(timeout_remaining=gen_timeout)
                    candidates = generator.generate(problem_text, classification)
                    logger.info(f"Generated {len(candidates)} candidate(s)")
            except TimeoutError:
                logger.warning("Reasoning timeout")
                candidates = []
            except Exception as e:
                logger.error(f"Reasoning error: {e}")
                candidates = []
            
            if not candidates:
                logger.warning("No candidates generated, returning (0, 0)")
                return (0, 0)
            
            # STAGE 3: VERIFY & VALIDATE
            logger.info("STAGE 3: Verifying and validating...")
            try:
                remaining_time = self.timeout - (time.time() - self.start_time)
                if remaining_time <= 0:
                    logger.warning("Timeout before verification")
                    return (0, 0)
                arb_timeout = min(remaining_time, self.time_per_stage['verify'] + self.time_per_stage['arbitrate'])
                
                with time_limit(arb_timeout):
                    arbitrator = AnswerArbitrator()
                    final_answer = arbitrator.arbitrate(candidates, problem_text)
                    logger.info(f"Final answer: {final_answer}")

                    # Deterministic retry ONLY if verification fails
                    if final_answer == (0, 0) and MAX_RETRIES_PER_STRATEGY > 0:
                        retry_candidates = generator.generate(problem_text, classification, retry_reason="Verification failed")
                        final_answer = arbitrator.arbitrate(retry_candidates, problem_text)
            except TimeoutError:
                logger.warning("Verification timeout, returning (0, 0)")
                final_answer = (0, 0)
            except Exception as e:
                logger.error(f"Verification error: {e}")
                final_answer = (0, 0)
            
            return final_answer
        
        except Exception as e:
            logger.error(f"Solver error: {e}", exc_info=True)
            return (0, 0)
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since solve started."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def time_remaining(self) -> float:
        """Get remaining time before timeout."""
        if self.start_time is None:
            return self.timeout
        elapsed = self.get_elapsed_time()
        return max(0.0, self.timeout - elapsed)


class AdaptiveSolver:
    """Kaggle-compatible interface wrapping StrategyArbiter."""
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
        self.arbiter = StrategyArbiter(timeout=timeout_seconds)
    
    def solve(self, problem_text: str, timeout_seconds: Optional[int] = None) -> Tuple[int, int]:
        """
        Main entry point for Kaggle submission.
        
        Args:
            problem_text: LaTeX problem
            timeout_seconds: Override default timeout
        
        Returns:
            Tuple of two integers (answer1, answer2)
        """
        timeout = timeout_seconds if timeout_seconds else self.timeout
        self.arbiter = StrategyArbiter(timeout=timeout)
        
        try:
            result = self.arbiter.solve(problem_text)
            return result
        except Exception as e:
            logger.error(f"AdaptiveSolver failed: {e}")
            return (0, 0)


# Global instance for submission.py
_solver = None

def get_solver() -> AdaptiveSolver:
    """Get or create global solver instance."""
    global _solver
    if _solver is None:
        _solver = AdaptiveSolver(timeout_seconds=30)
    return _solver


def solve(problem_text: str, timeout_seconds: int = 30) -> Tuple[int, int]:
    """Entry point for submission (backward compatible)."""
    solver = get_solver()
    return solver.solve(problem_text, timeout_seconds=timeout_seconds)
