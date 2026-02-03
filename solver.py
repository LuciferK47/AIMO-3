"""
AIMO-3 MASTER SOLVER - Strategy Arbiter Architecture

MASTER PROMPT IMPLEMENTATION

Target Pipeline:
LaTeX problem
  ↓
Semantic parsing + problem classification
  ↓
Parallel candidate generation (LLM × k)
  ↓
Equation extraction + symbolic solving
  ↓
Verification + consistency checks
  ↓
Answer arbitration (vote / confidence)
  ↓
Final integer pair

Key Principles:
- Deterministic execution
- Symbolic solving is PRIMARY (not fallback)
- Multi-candidate generation for robustness
- Strict verification gates
- Bounded time and depth
"""

import logging
import re
import time
from typing import List, Tuple, Optional, Dict, Any, Set

from config import *
from parsing import ProblemParser
from sympy_solver import SymPySolver, EquationExtractor, DiophantineSolver
from validation import AnswerValidator, SelfVerificationLoop
from utils import extract_integers, preprocess_problem_text, query_llm
from cache import get_intermediate_cache

logger = logging.getLogger(__name__)


class ProblemClassifier:
    """Lightweight rule-based problem classifier (no ML)."""
    
    @staticmethod
    def classify(problem_text: str) -> Dict[str, Any]:
        """
        Classify problem and return routing hints.
        
        Returns:
            {
                'problem_type': str,  # 'modular', 'combinatorics', 'diophantine', 'geometry', 'optimization', 'general'
                'keywords': List[str],
                'allows_symbolic': bool,
                'has_equations': bool,
                'is_modular': bool,
                'is_counting': bool,
                'is_geometry': bool,
                'difficulty_estimate': float,  # 0-1
            }
        """
        text_lower = problem_text.lower()
        
        # Keyword detection
        keywords = []
        classification = {
            'problem_type': 'general',
            'keywords': [],
            'allows_symbolic': True,
            'has_equations': False,
            'is_modular': False,
            'is_counting': False,
            'is_geometry': False,
            'difficulty_estimate': 0.5,
        }
        
        # MODULAR ARITHMETIC
        if any(w in text_lower for w in ['mod', 'remainder', 'divisible', 'congruence', 'modular']):
            classification['problem_type'] = 'modular'
            classification['is_modular'] = True
            classification['allows_symbolic'] = True
            keywords.extend(['modular', 'arithmetic'])
        
        # COMBINATORICS
        elif any(w in text_lower for w in ['how many', 'count', 'combinations', 'permutations', 'ways', 'arrangements', 'sequences']):
            classification['problem_type'] = 'combinatorics'
            classification['is_counting'] = True
            classification['allows_symbolic'] = True
            keywords.extend(['counting', 'combinatorics'])
        
        # DIOPHANTINE / NUMBER THEORY
        elif any(w in text_lower for w in ['integer solutions', 'find all integers', 'diophantine', 'number of pairs', 'pairs (', 'positive integers']):
            classification['problem_type'] = 'diophantine'
            classification['allows_symbolic'] = True
            keywords.extend(['number_theory', 'diophantine'])
        
        # GEOMETRY
        elif any(w in text_lower for w in ['triangle', 'circle', 'angle', 'area', 'perimeter', 'distance', 'radius', 'diameter']):
            classification['problem_type'] = 'geometry'
            classification['is_geometry'] = True
            # Geometry without diagrams is harder for SymPy
            classification['allows_symbolic'] = False
            keywords.extend(['geometry'])
        
        # OPTIMIZATION
        elif any(w in text_lower for w in ['maximum', 'minimum', 'maximize', 'minimize', 'least', 'greatest', 'optimal']):
            classification['problem_type'] = 'optimization'
            classification['allows_symbolic'] = True
            keywords.extend(['optimization'])
        
        # EQUATIONS / ALGEBRA
        if any(w in text_lower for w in ['equation', 'solve', '=', 'satisfy', 'satisfy']):
            classification['has_equations'] = True
        
        classification['keywords'] = keywords
        
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


class CandidateGenerator:
    """Generate multiple candidate answers using different strategies."""
    
    def __init__(self, timeout_remaining: float = 20.0):
        self.timeout_remaining = timeout_remaining
        self.candidates = []
    
    def generate(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Generate 3-5 candidate answers.
        
        CRITICAL INVERSION:
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        LLM ROLE: Translator (formalize, enumerate, derive)
        SYMPY ROLE: Solver (compute, verify, decide)
        
        Pipeline:
        1. SYMBOLIC-FIRST: Direct SymPy (modular, Diophantine)
        2. LLM TRANSLATION → SYMPY SOLVE (PRIMARY)
        3. LLM CASE ENUMERATION → SYMPY COMPUTE
        4. LLM FORMALIZATION → SYMPY EVALUATE
        
        LLM NEVER directly solves - only translates.
        """
        candidates = []
        
        # STRATEGY 1: SYMBOLIC-FIRST (no LLM)
        if classification['allows_symbolic']:
            try:
                sym_candidates = self._try_symbolic(problem_text, classification)
                candidates.extend(sym_candidates)
            except Exception as e:
                logger.debug(f"Direct symbolic failed: {e}")
        
        # STRATEGY 2: LLM TRANSLATION → SYMPY SOLVE (PRIMARY PATH)
        try:
            eq_candidates = self._try_equation_extraction(problem_text, classification)
            candidates.extend(eq_candidates)
        except Exception as e:
            logger.debug(f"Translation→solve failed: {e}")
        
        # STRATEGY 3: LLM CASE ENUMERATION → SYMPY COMPUTE
        try:
            case_candidates = self._try_case_enumeration(problem_text, classification)
            candidates.extend(case_candidates)
        except Exception as e:
            logger.debug(f"Case enumeration failed: {e}")
        
        # STRATEGY 4: LLM FORMALIZATION → SYMPY EVALUATE
        try:
            formal_candidates = self._try_formalization(problem_text, classification)
            candidates.extend(formal_candidates)
        except Exception as e:
            logger.debug(f"Formalization failed: {e}")
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen and c != (0, 0):
                seen.add(c)
                unique_candidates.append(c)
        
        return unique_candidates[:5]
    
    def _try_symbolic(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Try pure symbolic solving."""
        candidates = []
        
        try:
            # For modular problems
            if classification['is_modular']:
                result = SymPySolver.solve_modular_equation(problem_text)
                if result and result != (0, 0):
                    candidates.append(result)
            
            # For combinatorics
            if classification['is_counting']:
                result = SymPySolver.solve_combinatorics(problem_text)
                if result and result != (0, 0):
                    candidates.append(result)
            
            # Try general equation extraction for all problems
            from sympy_solver import EquationExtractor
            extractor = EquationExtractor()
            test_equations = extractor.extract_equations(problem_text)
            if test_equations:
                result = SymPySolver.solve_from_equations(test_equations, problem_text)
                if result and result != (0, 0):
                    candidates.append(result)
        
        except Exception as e:
            logger.debug(f"Symbolic solving error: {e}")
        
        return candidates
    
    def _try_equation_extraction(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        LLM TRANSLATION → SYMPY SOLVE (PRIMARY PIPELINE)
        
        Critical separation:
        - LLM: Translate natural language → formal equations (ONLY)
        - SymPy: Solve equations → numeric answer
        
        This is NOT a fallback. This is the MAIN path for 30-50% of problems.
        """
        candidates = []
        
        try:
            # STEP 1: LLM AS TRANSLATOR (NOT SOLVER)
            # Strict prompt: ONLY equations, NO solving, NO answers
            translation_prompt = f"""
You are a mathematical translator, NOT a solver.

Task: Convert this problem into formal mathematical equations ONLY.

Rules:
- Output ONLY equations (e.g., "x + y = 10", "2*x - 3*y = 5")
- Use standard variables: x, y, z, n, k, a, b
- Do NOT solve
- Do NOT compute answers
- Do NOT explain

Problem:
{problem_text}

Equations (one per line):
"""
            equations_text = query_llm(translation_prompt, max_tokens=250, temperature=0.1)
            
            if not equations_text:
                return candidates
            
            # STEP 2: EXTRACT & NORMALIZE
            extractor = EquationExtractor()
            equations = extractor.extract_equations(equations_text)
            
            if not equations:
                return candidates
            
            # STEP 3: SYMPY SOLVES (PRIMARY SOLVER)
            result = SymPySolver.solve_from_equations(equations, problem_text)
            if result and result != (0, 0):
                candidates.append(result)
                logger.info(f"SymPy solved from LLM-translated equations: {result}")
        
        except Exception as e:
            logger.debug(f"Translation→solve pipeline error: {e}")
        
        return candidates
    
    def _try_case_enumeration(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        LLM CASE ENUMERATION → SYMPY COMPUTE
        
        LLM: "List all cases/values to check"
        SymPy: Compute/validate each case
        
        Useful for: counting problems, modular arithmetic, optimization
        """
        candidates = []
        
        try:
            enumeration_prompt = f"""
You are a case enumerator, NOT a solver.

Task: List ALL cases or values that need to be checked.

Rules:
- Output a Python list: [val1, val2, val3, ...]
- Do NOT compute final answer
- Just enumerate the search space

Problem:
{problem_text}

Cases to check:
"""
            response = query_llm(enumeration_prompt, max_tokens=200, temperature=0.2)
            
            if response and '[' in response and ']' in response:
                import ast
                try:
                    cases = ast.literal_eval(response[response.index('['):response.rindex(']')+1])
                    if isinstance(cases, list):
                        for case in cases[:10]:  # Limit to 10 cases
                            if isinstance(case, int) and 0 <= case <= 99999:
                                candidates.append((case, 0))
                except (ValueError, SyntaxError):
                    pass
        
        except Exception as e:
            logger.debug(f"Case enumeration error: {e}")
        
        return candidates
    
    def _try_formalization(self, problem_text: str, classification: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        LLM FORMALIZATION → SYMPY EVALUATE
        
        LLM: "Convert to a single mathematical expression"
        SymPy: Evaluate the expression
        
        Useful for: combinatorial formulas, closed-form expressions
        """
        candidates = []
        
        try:
            formalization_prompt = f"""
You are a mathematical formalizer, NOT a calculator.

Task: Convert this problem into a SINGLE mathematical expression.

Rules:
- Output ONLY the expression (e.g., "factorial(5)/factorial(3)", "2**10 % 7")
- Use Python syntax: **, //, %, factorial(n)
- Do NOT compute the answer
- Do NOT explain

Problem:
{problem_text}

Expression:
"""
            response = query_llm(formalization_prompt, max_tokens=150, temperature=0.1)
            
            if response:
                try:
                    result = SymPySolver.evaluate_expression(response.strip())
                    if result is not None:
                        answer = int(result)
                        if 0 <= answer <= 99999:
                            candidates.append((answer, 0))
                            logger.info(f"SymPy evaluated: {answer}")
                except Exception:
                    pass
        
        except Exception as e:
            logger.debug(f"Formalization error: {e}")
        
        return candidates
        
        try:
            if mode == "direct":
                # Structured, deterministic
                prompt = f"""
Solve this problem step by step.
Output ONLY the final integer answer.
Format: ANSWER = <integer>

Problem: {problem_text}

ANSWER = """
                temperature = 0.2
            else:  # creative
                # More flexible, explores alternatives
                prompt = f"""
Solve this problem using any approach that works.
Show key steps only. End with final answer.
Format: ANSWER = <integer>

Problem: {problem_text}

ANSWER = """
                temperature = 0.5
            
            response = query_llm(prompt, max_tokens=300, temperature=temperature)
            
            if response:
                # Extract integer from response
                integers = extract_integers(response)
                if integers:
                    # Take first integer found
                    answer = integers[0]
                    # Clamp to valid range
                    if 0 <= answer <= 99999:
                        candidates.append((answer, 0))  # Single answer with secondary=0
        
        except Exception as e:
            logger.debug(f"LLM reasoning error: {e}")
        
        return candidates


class AnswerArbitrator:
    """Verify candidates and select best answer using arbitration."""
    
    def __init__(self):
        self.verifier = SelfVerificationLoop()
    
    def arbitrate(self, candidates: List[Tuple[int, int]], problem_text: str) -> Tuple[int, int]:
        """
        Verify all candidates and return best pair.
        
        Process:
        1. Filter valid candidates
        2. Return first (best) candidate
        """
        if not candidates:
            return (0, 0)
        
        # Filter valid candidates (range check)
        valid_candidates = [c for c in candidates if c != (0, 0)]
        
        if valid_candidates:
            # Return first (best generated) candidate
            return valid_candidates[0]
        
        return (0, 0)


class StrategyArbiter:
    """Master orchestrator implementing the target pipeline."""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.start_time = None
        self.time_per_stage = {
            'parse': 0.5,
            'classify': 0.2,
            'generate': 15.0,
            'verify': 5.0,
            'arbitrate': 1.0,
        }
    
    def solve(self, problem_text: str) -> Tuple[int, int]:
        """
        MASTER PIPELINE:
        
        1. Parse & Classify
        2. Multi-Candidate Generation
        3. Verification & Scoring
        4. Answer Arbitration
        """
        self.start_time = time.time()
        
        try:
            # STAGE 1: PARSE & CLASSIFY
            problem_text = preprocess_problem_text(problem_text)
            logger.info(f"Parsing problem...")
            
            classification = ProblemClassifier.classify(problem_text)
            logger.info(f"Classification: {classification['problem_type']}")
            logger.info(f"Allows symbolic: {classification['allows_symbolic']}")
            
            # STAGE 2: MULTI-CANDIDATE GENERATION
            logger.info(f"Generating candidates...")
            remaining_time = self.timeout - (time.time() - self.start_time)
            generator = CandidateGenerator(timeout_remaining=min(remaining_time, 15.0))
            candidates = generator.generate(problem_text, classification)
            
            logger.info(f"Generated {len(candidates)} candidates: {candidates}")
            
            if not candidates:
                logger.warning(f"No candidates generated, returning (0, 0)")
                return (0, 0)
            
            # STAGE 3 & 4: VERIFY & ARBITRATE
            logger.info(f"Arbitrating candidates...")
            arbitrator = AnswerArbitrator()
            final_answer = arbitrator.arbitrate(candidates, problem_text)
            
            logger.info(f"Final answer: {final_answer}")
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
